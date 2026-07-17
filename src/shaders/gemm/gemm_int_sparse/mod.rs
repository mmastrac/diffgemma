//! int8-accumulated block-sparse MoE expert GEMM (E18 redirect prototype).
//!
//! Standalone bench kernel — NOT wired to production. Settles whether skipping
//! the q4->half dequant + half-simdgroup-MMA path (`gemm_tunable_sparse`) and
//! accumulating `int32 += int8*int8` instead wins on M3, where there is no
//! dedicated tensor hardware and the synthetic microbench is unreliable in
//! absolute terms (PLAN.md E18 redirect).
//!
//! Dispatch contract mirrors `gemm_tunable_sparse` (BM=32 / BN=128 / BK=32)
//! for direct comparability; tile geometry is sweepable since int8 has no MMA
//! register-pressure trap. The CPU twin in `cpu.rs` is the oracle forever
//! (AGENTS.md §5 Tier 1).

pub mod cpu;

use crate::Error;
use crate::metal::{BlockGroupedJob, RouteScratch};
use crate::shaders::gpu_common;

pub const ENTRY: &str = "gemm_int_sparse";

pub const SHADER: &str = include_str!("gemm_int_sparse.metal");

/// Production-equivalent tile config: matches `gemm_tunable_sparse`
/// (SPARSE_BM=32 / SPARSE_BN=128 / BK=32) for apples-to-apples comparison.
pub const INT_BM: usize = 32;
pub const INT_BN: usize = 128;
pub const INT_BK: usize = 32;

pub fn tuned_source(bm: usize, bn: usize, bk: usize) -> String {
    format!("#define INT_BM {bm}\n#define INT_BN {bn}\n#define INT_BK {bk}\n{SHADER}")
}

/// Bench fixture: f32 reference weights + activations (so the oracle can
/// re-quantize to int8 with the same scale envelope), plus the int8 side-channel
/// blobs the kernel consumes. Mirrors `gemm_linear_grouped::Fixture` shape
/// contract (counts > 128 exercise ragged last-block padding).
#[derive(Debug, Clone)]
pub struct IntFixture {
    /// f32 reference activations `[slots, k]` row-major (the oracle re-quants).
    pub a_f32: Vec<f32>,
    /// f32 reference expert weights, one `[n*k]` row-major vec per expert.
    pub w_f32_experts: Vec<Vec<f32>>,
    pub row_starts: Vec<u32>,
    pub k: usize,
    pub n: usize,
}

impl IntFixture {
    pub fn total_m(&self) -> usize {
        *self.row_starts.last().unwrap_or(&0) as usize
    }
    pub fn num_jobs(&self) -> usize {
        self.row_starts.len().saturating_sub(1)
    }
    pub fn out_len(&self) -> usize {
        self.total_m() * self.n
    }
    pub fn jobs(&self) -> Vec<BlockGroupedJob> {
        let mut off = 0u64;
        self.w_f32_experts
            .iter()
            .map(|_| {
                let job = BlockGroupedJob {
                    w_byte_off: off,
                    groups_per_row: 0, // unused by the int8 kernel
                    _pad: 0,
                };
                off += (self.n * self.k) as u64;
                job
            })
            .collect()
    }

    /// Build the int8 side-channel: packed int8 A blob `[slots, K]`, packed
    /// int8 W blob (each expert `[N, K]` row-major at byte offset `jobs[e]`),
    /// per-slot A scales, and per-column W scales (shared across experts).
    /// The CPU oracle and the GPU kernel consume the same bytes + scales.
    pub fn int8_side_channel(&self) -> (Vec<i8>, Vec<u8>, Vec<f32>, Vec<f32>) {
        // A: per-row absmax int8.
        let (a_int8, scale_a) =
            cpu::quantize_matrix_int8_row_major(&self.a_f32, self.total_m(), self.k);
        // W: per-expert per-N-row absmax int8; collect scales per output column
        // (shared envelope across experts — see the kernel header comment).
        let mut w_int8 = Vec::<u8>::new();
        let mut scale_w = vec![0.0f32; self.n];
        for w in &self.w_f32_experts {
            let (q, s) = cpu::quantize_expert_int8_row_major(w, self.n, self.k);
            for c in 0..self.n {
                scale_w[c] = scale_w[c].max(s[c]);
            }
            w_int8.extend_from_slice(unsafe {
                std::slice::from_raw_parts(q.as_ptr() as *const u8, q.len())
            });
        }
        // Re-quantize each expert with the shared per-column scale_w envelope
        // so the oracle and the kernel see the same int8 weights.
        let mut w_int8 = Vec::<u8>::new();
        for w in &self.w_f32_experts {
            for c in 0..self.n {
                let row = &w[c * self.k..(c + 1) * self.k];
                let s = scale_w[c];
                for &v in row {
                    let r = (v / s).round().clamp(-128.0, 127.0) as i8;
                    w_int8.push(r as u8);
                }
            }
        }
        (a_int8, w_int8, scale_a, scale_w)
    }
}

/// Build a synthetic IntFixture with the same shape contract as
/// `gemm_linear_grouped::grouped_fixture` (patterned values, distinct experts).
pub fn grouped_int_fixture(k: usize, n: usize, rows_per_expert: &[usize]) -> IntFixture {
    let mut row_starts = vec![0u32];
    let mut a = Vec::new();
    let mut w_f32_experts = Vec::new();
    for (ei, &rows) in rows_per_expert.iter().enumerate() {
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| (((ei + 1) as f32) * 0.11 + (i as f32) * 0.007).cos() * 0.03)
            .collect();
        w_f32_experts.push(w_f32);
        for r in 0..rows {
            let base = (row_starts.last().copied().unwrap_or(0) as usize + r) * k;
            for p in 0..k {
                a.push(((base + p) as f32 * 0.013).sin() * 0.25 + ei as f32 * 0.01);
            }
        }
        row_starts.push(row_starts.last().copied().unwrap_or(0) + rows as u32);
    }
    IntFixture {
        a_f32: a,
        w_f32_experts,
        row_starts,
        k,
        n,
    }
}

#[cfg(target_os = "macos")]
pub fn pipeline_for_tile(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    bm: usize,
    bn: usize,
    bk: usize,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let src = tuned_source(bm, bn, bk);
    // The int8 kernel uses FC5/FC6 (GEMM_N/GEMM_K) from gemm_fc.metal but no
    // FC3 quant-format dispatch — int8 is a side-channel format, not a .dgq
    // quant. Compile as a Raw (qf=4) GEMM subkernel with the source hash folded
    // into the cache label (closes the tile-define collision class per the
    // DGQ_MOE_PREFILL_BM root-cause fix in device.rs).
    ctx.compile_gemm_subkernel(
        &src,
        ENTRY,
        n,
        k,
        false,
        crate::shaders::QuantFormat::Raw as u32,
        false,
    )
}

/// Build the production-equivalent block-sparse route (bucket_fill phase-1
/// mirror) at the requested block height `bm`. Uses the actual `row_starts` so
/// experts with different row counts get correctly-bounded blocks (mirrors
/// `gemm_tunable::gpu_sparse_tunable`'s route builder).
pub fn build_route(row_starts: &[u32], num_experts: usize, bm: usize) -> (Box<RouteScratch>, u32) {
    let mut route: Box<RouteScratch> = unsafe { Box::new(std::mem::zeroed()) };
    for (i, &rs) in row_starts.iter().enumerate().take(num_experts + 1) {
        route.row_start[i] = rs;
    }
    route.num_slots = row_starts.last().copied().unwrap_or(0);
    route.num_active_experts = num_experts as u32;
    let mut blk = 0usize;
    for e in 0..num_experts {
        route.active_expert[e] = e as u32;
        let base = row_starts[e];
        let count = row_starts[e + 1] - row_starts[e];
        for b in 0..(count as usize).div_ceil(bm) {
            route.block_expert[blk] = e as u32;
            route.block_row0[blk] = base + (b * bm) as u32;
            blk += 1;
        }
    }
    route.num_blocks = blk as u32;
    (route, blk as u32)
}

/// Run `gemm_int_sparse` on an IntFixture at tile `(bm, bn, bk)` and return the
/// f32 slot output. Mirrors `gemm_tunable::gpu_sparse_tunable_bm` wiring.
#[cfg(target_os = "macos")]
pub fn gpu_int_sparse(f: &IntFixture, bm: usize, bn: usize, bk: usize) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let jobs = f.jobs();
    let num_jobs = f.num_jobs();
    let (a_int8, w_int8, scale_a, scale_w) = f.int8_side_channel();
    let out_len = f.out_len();

    let buf_a = pool
        .allocate(&ctx.device, a_int8.len())
        .ok_or(Error::Gpu("alloc a_int8"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_int8.len())
        .ok_or(Error::Gpu("alloc w_int8"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc c"))?;
    let buf_sa = pool
        .allocate(&ctx.device, scale_a.len() * 4)
        .ok_or(Error::Gpu("alloc scale_a"))?;
    let buf_sw = pool
        .allocate(&ctx.device, scale_w.len() * 4)
        .ok_or(Error::Gpu("alloc scale_w"))?;
    let buf_jobs = pool
        .allocate(
            &ctx.device,
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
        .ok_or(Error::Gpu("alloc jobs"))?;
    let buf_rs = pool
        .allocate(&ctx.device, f.row_starts.len() * 4)
        .ok_or(Error::Gpu("alloc rs"))?;
    let (route, num_blocks) = build_route(&f.row_starts, num_jobs, bm);
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Gpu("alloc route"))?;

    BufferPool::write_bytes(&buf_a, unsafe {
        std::slice::from_raw_parts(a_int8.as_ptr() as *const u8, a_int8.len())
    });
    BufferPool::write_bytes(&buf_w, &w_int8);
    BufferPool::write_f32(&buf_sa, &scale_a);
    BufferPool::write_f32(&buf_sw, &scale_w);
    BufferPool::write_bytes(&buf_jobs, unsafe {
        std::slice::from_raw_parts(
            jobs.as_ptr() as *const u8,
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
    });
    BufferPool::write_bytes(&buf_rs, unsafe {
        std::slice::from_raw_parts(f.row_starts.as_ptr() as *const u8, f.row_starts.len() * 4)
    });
    BufferPool::write_bytes(&buf_route, unsafe {
        std::slice::from_raw_parts(
            route.as_ref() as *const RouteScratch as *const u8,
            std::mem::size_of::<RouteScratch>(),
        )
    });
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);

    let pipe = pipeline_for_tile(&ctx, f.n as u32, f.k as u32, bm, bn, bk)?;
    let grid = MTLSize {
        width: f.n.div_ceil(bn),
        height: num_blocks as usize,
        depth: 1,
    };
    let tg = MTLSize {
        width: crate::shaders::gemm_common::THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipe.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_jobs), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&buf_rs), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&buf_route), 0, 6);
        enc.setBuffer_offset_atIndex(Some(&buf_sa), 0, 7);
        enc.setBuffer_offset_atIndex(Some(&buf_sw), 0, 8);
    }
    gpu_common::set_bytes(&enc, &(num_jobs as u32), 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            d += (*x as f64) * (*y as f64);
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
        }
        (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Tier-1 oracle: GPU int8 sparse GEMM tracks the CPU int8 oracle within
    /// quant tolerance. Counts > 128 exercise the ragged last-block padding
    /// path (same shape contract as gemm_tunable::sparse_block_m_invariant).
    #[test]
    fn sparse_int8_matches_cpu_oracle() {
        let counts: &[usize] = &[200, 77, 150, 5, 300, 33, 128];
        let f = grouped_int_fixture(256, 128, counts);
        let (a_int8, w_int8, scale_a, scale_w) = f.int8_side_channel();
        let jobs = f.jobs();
        let (route, _num_blocks) = build_route(&f.row_starts, f.num_jobs(), INT_BM);
        let oracle = cpu::gemm_int_sparse_cpu(
            &a_int8,
            &w_int8,
            &scale_a,
            &scale_w,
            &jobs,
            &f.row_starts,
            &route,
            f.n,
            f.k,
            INT_BM,
        );
        let gpu = gpu_int_sparse(&f, INT_BM, INT_BN, INT_BK).expect("int8 sparse gpu");
        assert_eq!(gpu.len(), oracle.len());
        let c = cos(&gpu, &oracle);
        println!("  sparse_int8_matches_cpu_oracle: cos={c:.6}");
        assert!(c > 0.999, "int8 sparse cos={c:.6} < 0.999");
    }

    /// Tile invariance: BM=32/64/128 must all match the CPU oracle. The int8
    /// path has no MMA register pressure (no simdgroup_float8x8 array), so tile
    /// height is mathematically irrelevant. Also guards the pipeline-cache-label
    /// collision class (the DGQ_MOE_PREFILL_BM root cause) — loops tiles in one
    /// process to force the collision if the source-hash fix regresses.
    #[test]
    fn sparse_int8_block_m_invariant() {
        let counts: &[usize] = &[200, 77, 150, 5, 300, 33, 128];
        let f = grouped_int_fixture(256, 128, counts);
        let (a_int8, w_int8, scale_a, scale_w) = f.int8_side_channel();
        let jobs = f.jobs();
        for (bm, bn) in [(32usize, 128usize), (64, 64), (128, 64)] {
            let (route, _num_blocks) = build_route(&f.row_starts, f.num_jobs(), bm);
            let oracle = cpu::gemm_int_sparse_cpu(
                &a_int8,
                &w_int8,
                &scale_a,
                &scale_w,
                &jobs,
                &f.row_starts,
                &route,
                f.n,
                f.k,
                bm,
            );
            let gpu = gpu_int_sparse(&f, bm, bn, INT_BK).expect("int8 sparse gpu bm");
            assert_eq!(gpu.len(), oracle.len());
            let c = cos(&gpu, &oracle);
            println!("  bm={bm:>3} bn={bn:>3}: cos={c:.6} vs CPU oracle");
            assert!(
                c > 0.999,
                "int8 sparse tile bm={bm} bn={bn} diverged: cos={c:.6}"
            );
        }
    }
}
