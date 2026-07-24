//! Sparse / db-variant GEMM benches (bench-gemm oracle modes). Split from
//! bench_gemm.rs; shares its GemmShape/GemmBenchRow API and gflops timer.

use super::bench_gemm::{GemmBenchRow, GemmShape, gflops};
use crate::Error;
use crate::metal::buffer::BufferPool;
use crate::shaders::bf16;
use std::time::Instant;

/// Block-sparse tunable MoE GEMM at the PRODUCTION route distributions
/// (`bench-gemm --shapes sparse`): 128 experts with uniform per-expert row
/// counts (64 rows = batched-prefill M=1024 top-8; 16 rows = denoise M=256),
/// 128 DISTINCT expert weight regions (real cache behavior), block lists
/// built exactly like moe_bucket_fill phase 1. Times the production
/// `gemm_tunable_sparse` pipelines (BM 32 and 64) against the legacy
/// `gemm_block_sparse` reference (bitwise-compared) and reports USEFUL
/// TF/s (routed rows only — padding excluded, matching the
/// methodology).
#[cfg(target_os = "macos")]
pub fn bench_gemm_tunable_sparse(iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    use crate::metal::device::MetalContext;
    use crate::metal::{BlockGroupedJob, RouteScratch};
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let warmup = 3usize;
    let mut rows = Vec::new();
    const N_EXPERTS: usize = 128;

    // (n, k, tag): MoE gate_up (fused gate+up columns) and down shapes.
    const SHAPES: &[(usize, usize, &str)] = &[(1408, 2816, "gate_up"), (2816, 704, "down")];
    // Uniform per-expert rows: 64 = prefill super-chunk, 16 = denoise-ish.
    const ROWS_PER_EXPERT: &[usize] = &[64, 16];

    for &(n, k, tag) in SHAPES {
        // Synthetic q4 blob: 128 distinct expert regions, patterned nibbles,
        // constant sane scales (perf fidelity needs distinct addresses, not
        // realistic values).
        let row_b = (k / 32) * 20;
        let expert_b = n * row_b;
        let mut blob = vec![0u8; N_EXPERTS * expert_b];
        let s_bits = bf16::f32_to_bf16_bits(0.02).to_le_bytes();
        let m_bits = bf16::f32_to_bf16_bits(-0.16).to_le_bytes();
        for g in 0..(blob.len() / 20) {
            let base = g * 20;
            blob[base] = s_bits[0];
            blob[base + 1] = s_bits[1];
            blob[base + 2] = m_bits[0];
            blob[base + 3] = m_bits[1];
            for j in 0..16 {
                let idx = (base + 4 + j) as u32;
                blob[base + 4 + j] = (idx.wrapping_mul(2654435761) >> 24) as u8;
            }
        }
        let buf_w = pool
            .allocate(&ctx.device, blob.len())
            .ok_or(Error::Runtime("sparse bench w"))?;
        BufferPool::write_bytes(&buf_w, &blob);
        drop(blob);

        let jobs: Vec<BlockGroupedJob> = (0..N_EXPERTS)
            .map(|e| BlockGroupedJob {
                w_byte_off: (e * expert_b) as u64,
                groups_per_row: (k / 32) as u32,
                _pad: 0,
            })
            .collect();
        let buf_jobs = pool
            .allocate(
                &ctx.device,
                jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            )
            .ok_or(Error::Runtime("sparse bench jobs"))?;
        BufferPool::write_bytes(&buf_jobs, unsafe {
            std::slice::from_raw_parts(
                jobs.as_ptr().cast::<u8>(),
                jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            )
        });

        for &rpe in ROWS_PER_EXPERT {
            let slots = N_EXPERTS * rpe;
            let shape = GemmShape { m: slots, k, n };

            let a: Vec<f32> = (0..slots * k)
                .map(|i| ((i as f32) * 0.013).sin() * 0.2)
                .collect();
            let buf_a = pool
                .allocate(&ctx.device, a.len() * 4)
                .ok_or(Error::Runtime("sparse bench a"))?;
            BufferPool::write_f32(&buf_a, &a);
            drop(a);

            let row_starts: Vec<u32> = (0..=N_EXPERTS).map(|e| (e * rpe) as u32).collect();
            let buf_rs = pool
                .allocate(&ctx.device, row_starts.len() * 4)
                .ok_or(Error::Runtime("sparse bench rs"))?;
            BufferPool::write_bytes(&buf_rs, unsafe {
                std::slice::from_raw_parts(row_starts.as_ptr().cast::<u8>(), row_starts.len() * 4)
            });

            let buf_c = pool
                .allocate(&ctx.device, slots * n * 4)
                .ok_or(Error::Runtime("sparse bench c"))?;
            let buf_route = pool
                .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
                .ok_or(Error::Runtime("sparse bench route"))?;

            // Route with block list at height `bm` (bucket_fill phase 1 mirror).
            let write_route = |bm: usize| -> u32 {
                let mut route: Box<RouteScratch> = unsafe { Box::new(std::mem::zeroed()) };
                for (i, &rs) in row_starts.iter().enumerate() {
                    route.row_start[i] = rs;
                }
                route.num_slots = slots as u32;
                route.num_active_experts = N_EXPERTS as u32;
                let mut blk = 0usize;
                for e in 0..N_EXPERTS {
                    route.active_expert[e] = e as u32;
                    for b in 0..rpe.div_ceil(bm) {
                        route.block_expert[blk] = e as u32;
                        route.block_row0[blk] = (e * rpe + b * bm) as u32;
                        blk += 1;
                    }
                }
                route.num_blocks = blk as u32;
                BufferPool::write_bytes(&buf_route, unsafe {
                    std::slice::from_raw_parts(
                        route.as_ref() as *const RouteScratch as *const u8,
                        std::mem::size_of::<RouteScratch>(),
                    )
                });
                blk as u32
            };

            for (bm, bn) in [(32usize, 64usize), (32, 128), (64, 64)] {
                // Wider-than-32 blocks only pay off when an expert spans
                // multiple 32-blocks (the prefill case); 32 always runs.
                if bm != 32 && bm > rpe {
                    continue;
                }
                let blocks = write_route(bm);
                let pipe = crate::shaders::gemm_tunable::pipeline_for_sparse_tile(
                    &ctx,
                    n as u32,
                    k as u32,
                    false,
                    crate::shaders::QuantFormat::Q4Affine,
                    bm,
                    bn,
                )?;
                let grid = MTLSize {
                    width: n.div_ceil(bn),
                    height: blocks as usize,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: crate::shaders::gemm_common::THREADS_PER_TG,
                    height: 1,
                    depth: 1,
                };
                let dispatch = |count: usize| -> Result<(), Error> {
                    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
                    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
                    for _ in 0..count {
                        enc.setComputePipelineState(&pipe.pipeline);
                        crate::shaders::gemm_block_grouped::bind_gpu_buffers(
                            &enc,
                            &buf_a,
                            &buf_w,
                            &buf_c,
                            &buf_jobs,
                            &buf_rs,
                            &buf_route,
                            N_EXPERTS as u32,
                        );
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    }
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    Ok(())
                };
                dispatch(1)?;
                dispatch(warmup)?;
                let started = Instant::now();
                dispatch(iters)?;
                let rate = gflops(slots, k, n, iters, started.elapsed().as_secs_f64());
                rows.push(GemmBenchRow {
                    shape,
                    label: format!("sparse_{tag}_r{rpe}_{bm}x{bn}/x{iters}"),
                    gflops: rate,
                });
            }
            pool.release(slots * k * 4, buf_a);
            pool.release(slots * n * 4, buf_c);
        }
        pool.release(N_EXPERTS * expert_b, buf_w);
    }
    Ok(rows)
}

/// int8-accumulated block-sparse MoE expert GEMM bench.
/// Mirrors `bench_gemm_tunable_sparse` shapes (128 experts,
/// prefill rpe=64 + denoise rpe=16, gate_up 1408×K and down 2816×K) but swaps
/// the production q4→half-MMA path for the int8 dot-product path
/// (`gemm_int_sparse`): activations and weights are per-row-absmax int8,
/// `int32 += int8*int8` per output cell, rescaled once at the end. Head-to-head
/// vs `bench_gemm_tunable_sparse` rows in the same `bench-gemm --shapes sparse`
/// output. Reports USEFUL TF/s (routed rows only — padding excluded, matching
/// the production methodology).
#[cfg(target_os = "macos")]
pub fn bench_gemm_int_sparse(iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    use crate::metal::device::MetalContext;
    use crate::metal::{BlockGroupedJob, RouteScratch};
    use crate::shaders::gemm_int_sparse;
    use crate::shaders::gpu_common;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let warmup = 3usize;
    let mut rows = Vec::new();
    const N_EXPERTS: usize = 128;
    const SHAPES: &[(usize, usize, &str)] = &[(1408, 2816, "gate_up"), (2816, 704, "down")];
    const ROWS_PER_EXPERT: &[usize] = &[64, 16];

    for &(n, k, tag) in SHAPES {
        // Synthetic f32 expert weights + activations (distinct addresses, not
        // realistic values — perf fidelity needs the cache footprint, not the
        // numeric content). The int8 side-channel re-quants these per-row.
        let w_f32_experts: Vec<Vec<f32>> = (0..N_EXPERTS)
            .map(|e| {
                (0..n * k)
                    .map(|i| (((e + 1) as f32) * 0.11 + (i as f32) * 0.007).cos() * 0.03)
                    .collect()
            })
            .collect();
        let jobs: Vec<BlockGroupedJob> = (0..N_EXPERTS)
            .map(|e| BlockGroupedJob {
                w_byte_off: (e * n * k) as u64,
                groups_per_row: 0,
                _pad: 0,
            })
            .collect();

        // Pack the int8 W blob + per-column scale envelope (shared across
        // experts — see the kernel header comment).
        let mut w_int8 = Vec::<u8>::with_capacity(N_EXPERTS * n * k);
        let mut scale_w = vec![0.0f32; n];
        for w in &w_f32_experts {
            for c in 0..n {
                let row = &w[c * k..(c + 1) * k];
                let max_abs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                scale_w[c] = scale_w[c].max(max_abs / 127.0);
            }
        }
        for w in &w_f32_experts {
            for c in 0..n {
                let row = &w[c * k..(c + 1) * k];
                let s = scale_w[c];
                for &v in row {
                    let r = (v / s).round().clamp(-128.0, 127.0) as i8;
                    w_int8.push(r as u8);
                }
            }
        }
        drop(w_f32_experts);

        let buf_w = pool
            .allocate(&ctx.device, w_int8.len())
            .ok_or(Error::Runtime("int8 sparse bench w"))?;
        BufferPool::write_bytes(&buf_w, &w_int8);
        drop(w_int8);

        let buf_sw = pool
            .allocate(&ctx.device, scale_w.len() * 4)
            .ok_or(Error::Runtime("int8 sparse bench sw"))?;
        BufferPool::write_f32(&buf_sw, &scale_w);

        let buf_jobs = pool
            .allocate(
                &ctx.device,
                jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            )
            .ok_or(Error::Runtime("int8 sparse bench jobs"))?;
        BufferPool::write_bytes(&buf_jobs, unsafe {
            std::slice::from_raw_parts(
                jobs.as_ptr() as *const u8,
                jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            )
        });

        for &rpe in ROWS_PER_EXPERT {
            let slots = N_EXPERTS * rpe;
            let shape = GemmShape { m: slots, k, n };

            // Same f32 A reference as bench_gemm_tunable_sparse.
            let a_f32: Vec<f32> = (0..slots * k)
                .map(|i| ((i as f32) * 0.013).sin() * 0.2)
                .collect();
            // Per-row absmax int8 A.
            let mut a_int8 = vec![0u8; slots * k];
            let mut scale_a = vec![0.0f32; slots];
            for s in 0..slots {
                let row = &a_f32[s * k..(s + 1) * k];
                let max_abs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let sc = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                scale_a[s] = sc;
                for (j, &v) in row.iter().enumerate() {
                    let r = (v / sc).round().clamp(-128.0, 127.0) as i8;
                    a_int8[s * k + j] = r as u8;
                }
            }
            drop(a_f32);

            let buf_a = pool
                .allocate(&ctx.device, a_int8.len())
                .ok_or(Error::Runtime("int8 sparse bench a"))?;
            BufferPool::write_bytes(&buf_a, &a_int8);
            drop(a_int8);

            let buf_sa = pool
                .allocate(&ctx.device, scale_a.len() * 4)
                .ok_or(Error::Runtime("int8 sparse bench sa"))?;
            BufferPool::write_f32(&buf_sa, &scale_a);

            let row_starts: Vec<u32> = (0..=N_EXPERTS).map(|e| (e * rpe) as u32).collect();
            let buf_rs = pool
                .allocate(&ctx.device, row_starts.len() * 4)
                .ok_or(Error::Runtime("int8 sparse bench rs"))?;
            BufferPool::write_bytes(&buf_rs, unsafe {
                std::slice::from_raw_parts(row_starts.as_ptr() as *const u8, row_starts.len() * 4)
            });

            let buf_c = pool
                .allocate(&ctx.device, slots * n * 4)
                .ok_or(Error::Runtime("int8 sparse bench c"))?;
            let buf_route = pool
                .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
                .ok_or(Error::Runtime("int8 sparse bench route"))?;

            for (bm, bn) in [(32usize, 128usize), (64, 64)] {
                // Wider-than-32 blocks only pay off when an expert spans
                // multiple 32-blocks (the prefill case rpe=64); 32 always runs.
                if bm != 32 && bm > rpe {
                    continue;
                }
                let (route, num_blocks) = gemm_int_sparse::build_route(&row_starts, N_EXPERTS, bm);
                BufferPool::write_bytes(&buf_route, unsafe {
                    std::slice::from_raw_parts(
                        route.as_ref() as *const RouteScratch as *const u8,
                        std::mem::size_of::<RouteScratch>(),
                    )
                });
                let pipe = gemm_int_sparse::pipeline_for_tile(
                    &ctx,
                    n as u32,
                    k as u32,
                    bm,
                    bn,
                    gemm_int_sparse::INT_BK,
                )?;
                let grid = MTLSize {
                    width: n.div_ceil(bn),
                    height: num_blocks as usize,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: crate::shaders::gemm_common::THREADS_PER_TG,
                    height: 1,
                    depth: 1,
                };
                let dispatch = |count: usize| -> Result<(), Error> {
                    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
                    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
                    for _ in 0..count {
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
                        gpu_common::set_bytes(&enc, &(N_EXPERTS as u32), 5);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                    }
                    enc.endEncoding();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    Ok(())
                };
                dispatch(1)?;
                dispatch(warmup)?;
                let started = Instant::now();
                dispatch(iters)?;
                let rate = gflops(slots, k, n, iters, started.elapsed().as_secs_f64());
                rows.push(GemmBenchRow {
                    shape,
                    label: format!("int8_{tag}_r{rpe}_{bm}x{bn}/x{iters}"),
                    gflops: rate,
                });
            }
            pool.release(slots * k, buf_a);
            pool.release(slots * 4, buf_sa);
            pool.release(slots * n * 4, buf_c);
            pool.release(row_starts.len() * 4, buf_rs);
            pool.release(std::mem::size_of::<RouteScratch>(), buf_route);
        }
        pool.release(N_EXPERTS * n * k, buf_w);
        pool.release(n * 4, buf_sw);
        pool.release(
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            buf_jobs,
        );
    }
    Ok(rows)
}

/// Double-buffered dense tunable GEMM bench: overlaps the K-tile device->tgmem
/// load of tile N+1 with the MMA of tile N (one barrier per K-tile vs two in
/// the single-buffered kernel). Same K-accumulation chain -> BIT-EXACT vs the
/// single-buffered `gemm_tunable`; the bench row carries the BITEXACT/MISMATCH
/// tag so the perf number is gated on correctness in-place (the
/// DGQ_MOE_PREFILL_BM fake-win lesson). Wired to `bench-gemm` via the
/// dense-shape path so rows are directly comparable to `tunable_*`.
#[cfg(target_os = "macos")]
pub fn bench_gemm_tunable_db(
    shapes: &[GemmShape],
    iters: usize,
) -> Result<Vec<GemmBenchRow>, Error> {
    use crate::metal::device::MetalContext;
    use crate::shaders::gemm_q4;
    use crate::shaders::gemm_tunable;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLSize,
    };

    const SHADER_TUNE: &str = crate::shaders::gemm_tunable::SHADER;
    let ctx = MetalContext::new()?;
    let warmup = 3usize;
    let mut rows = Vec::new();
    let mut pool = BufferPool::new();

    for &shape in shapes {
        let GemmShape { m, k, n } = shape;
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.0007).cos() * 0.02)
            .collect();
        let fixture = gemm_q4::Fixture {
            x: (0..m * k)
                .map(|i| ((i as f32) * 0.013).sin() * 0.2)
                .collect(),
            w_f32,
            m,
            n,
            k,
            global_scale: 1.0,
        };
        let w_q4 = gemm_q4::w_q4(&fixture);
        let buf_x = pool
            .allocate(&ctx.device, m * k * 2)
            .ok_or(Error::Runtime("bench db x"))?;
        let buf_y = pool
            .allocate(&ctx.device, m * n * 2)
            .ok_or(Error::Runtime("bench db y"))?;
        let buf_ref = pool
            .allocate(&ctx.device, m * n * 2)
            .ok_or(Error::Runtime("bench db ref"))?;
        let buf_w = pool
            .allocate(&ctx.device, w_q4.len())
            .ok_or(Error::Runtime("bench db w"))?;
        BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&fixture.x));
        BufferPool::write_bytes(&buf_w, &w_q4);

        // Reference: single-buffered gemm_tunable at the production tile.
        let (bm, bn) = (gemm_tunable::TUNE_BM, gemm_tunable::TUNE_BN);
        {
            let src = format!("#define TUNE_BM {bm}\n#define TUNE_BN {bn}\n{SHADER_TUNE}");
            let ref_pipe = ctx.compile_gemm_subkernel(
                &src,
                gemm_tunable::ENTRY,
                n as u32,
                k as u32,
                false,
                crate::shaders::QuantFormat::Q4Affine as u32,
                false,
            )?;
            let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
            enc.setComputePipelineState(&ref_pipe.pipeline);
            gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_ref, &buf_w, 0, m as u32);
            let grid = MTLSize {
                width: n.div_ceil(bn),
                height: m.div_ceil(bm),
                depth: 1,
            };
            let tg = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        // Double-buffered variant at the same tile.
        let pipe = gemm_tunable::pipeline_for_db(
            &ctx,
            n as u32,
            k as u32,
            crate::shaders::QuantFormat::Q4Affine,
        )?;
        let grid = MTLSize {
            width: n.div_ceil(bn),
            height: m.div_ceil(bm),
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        let dispatch = |count: usize| -> Result<(), Error> {
            let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
            for _ in 0..count {
                enc.setComputePipelineState(&pipe.pipeline);
                gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, m as u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            Ok(())
        };
        dispatch(1)?;
        // Correctness: bitwise compare vs single-buffered reference.
        let mut mismatches = 0usize;
        let mut maxd = 0f32;
        {
            let ry = buf_ref.contents().as_ptr() as *const u16;
            let sy = buf_y.contents().as_ptr() as *const u16;
            for i in 0..(m * n) {
                let rb = unsafe { *ry.add(i) };
                let sb = unsafe { *sy.add(i) };
                if rb != sb {
                    mismatches += 1;
                    let a = bf16::bf16_bits_to_f32(rb);
                    let b = bf16::bf16_bits_to_f32(sb);
                    maxd = maxd.max((a - b).abs());
                }
            }
        }
        dispatch(warmup)?;
        let started = Instant::now();
        dispatch(iters)?;
        let rate = gflops(m, k, n, iters, started.elapsed().as_secs_f64());
        let bits = if mismatches == 0 {
            "BITEXACT".to_string()
        } else {
            format!("MISMATCH x{mismatches} max|d|={maxd:.5}")
        };
        rows.push(GemmBenchRow {
            shape,
            label: format!("tunable_db_{bm}x{bn}/x{iters} [{bits}]"),
            gflops: rate,
        });

        pool.release(w_q4.len(), buf_w);
        pool.release(m * k * 2, buf_x);
        pool.release(m * n * 2, buf_y);
        pool.release(m * n * 2, buf_ref);
    }
    Ok(rows)
}
