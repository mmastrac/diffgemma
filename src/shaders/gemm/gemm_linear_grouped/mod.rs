//! Grouped MoE block GEMM: flattened expert rows in one dispatch.

use crate::Error;
use crate::dgq::block::quantize_row_q4;
use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes, q4_row_bytes};
use crate::dgq::nvfp4::quantize_f32_matrix_nvfp4;
use crate::metal::BlockGroupedJob;
use crate::shaders::QuantFormat;
use crate::shaders::cpu::gemm_linear_grouped::gemm_linear_grouped_cpu;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "gemm_linear_grouped";

pub const SHADER: &str = include_str!("gemm_linear_grouped.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub a: Vec<f32>,
    pub w_f32_experts: Vec<Vec<f32>>,
    pub row_starts: Vec<u32>,
    pub k: usize,
    pub n: usize,
    pub format: QuantFormat,
}

impl Fixture {
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
        let gpr = match self.format {
            QuantFormat::NvFp4 => (self.k as u32).div_ceil(16),
            _ => (self.k as u32).div_ceil(32),
        };
        let mut off = 0u64;
        self.w_f32_experts
            .iter()
            .map(|_| {
                let job = BlockGroupedJob {
                    w_byte_off: off,
                    groups_per_row: gpr,
                    _pad: 0,
                };
                off += expert_matrix_bytes(self.format, self.n, self.k) as u64;
                job
            })
            .collect()
    }

    pub fn w_blob(&self) -> Vec<u8> {
        let mut blob = Vec::new();
        for w in &self.w_f32_experts {
            blob.extend_from_slice(&quantize_expert_matrix(self.format, w, self.n, self.k));
        }
        blob
    }
}

pub(crate) fn expert_matrix_bytes(format: QuantFormat, n: usize, k: usize) -> usize {
    match format {
        QuantFormat::NvFp4 => nvfp4_matrix_bytes(n, k),
        _ => q4_matrix_bytes(n, k),
    }
}

fn quantize_expert_matrix(format: QuantFormat, rows: &[f32], n: usize, k: usize) -> Vec<u8> {
    match format {
        QuantFormat::NvFp4 => {
            let mut dst = vec![0u8; nvfp4_matrix_bytes(n, k)];
            quantize_f32_matrix_nvfp4(rows, n, k, &mut dst);
            dst
        }
        _ => {
            let mut dst = vec![0u8; q4_matrix_bytes(n, k)];
            let row_bytes = q4_row_bytes(k);
            for row in 0..n {
                let off = row * k;
                quantize_row_q4(&rows[off..off + k], k, &mut dst[row * row_bytes..]);
            }
            dst
        }
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

/// One expert with M=100 (>32 M-tile stress) plus a smaller second expert.
pub fn tiny_fixture_q4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::Q4Affine, 64, 32, &[100, 4])
}

pub fn tiny_fixture_nvfp4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::NvFp4, 64, 32, &[100, 4])
}

/// M=100 with N/K=128. Scalar grouped path tiles M via `grid.height = total_m` (no in-kernel M cap).
pub fn tile_fixture_q4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::Q4Affine, 128, 128, &[100, 48, 4])
}

pub fn tile_fixture_nvfp4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::NvFp4, 128, 128, &[100, 48, 4])
}

/// Encoder prefill MoE row volume: 25 prompt tokens × 8 experts = 200 grouped rows.
pub fn prefill_moe_fixture_nvfp4(_: ElemFormat) -> Fixture {
    grouped_fixture(
        QuantFormat::NvFp4,
        128,
        128,
        &[25, 25, 25, 25, 25, 25, 25, 25],
    )
}

pub fn prefill_moe_fixture_q4(_: ElemFormat) -> Fixture {
    grouped_fixture(
        QuantFormat::Q4Affine,
        128,
        128,
        &[25, 25, 25, 25, 25, 25, 25, 25],
    )
}

pub(crate) fn grouped_fixture(
    format: QuantFormat,
    k: usize,
    n: usize,
    rows_per_expert: &[usize],
) -> Fixture {
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
    Fixture {
        a,
        w_f32_experts,
        row_starts,
        k,
        n,
        format,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    gemm_linear_grouped_cpu(
        &f.a,
        f.total_m(),
        f.k,
        f.n,
        &f.w_blob(),
        &f.jobs(),
        &f.row_starts,
        f.format,
    )
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    format: QuantFormat,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let v = KernelVariant {
        quant_format: format,
        ..variant
    };
    ctx.compile_subkernel(SHADER, ENTRY, v)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &ProtocolObject<dyn MTLBuffer>,
    w_blob: &ProtocolObject<dyn MTLBuffer>,
    c: &ProtocolObject<dyn MTLBuffer>,
    jobs: &ProtocolObject<dyn MTLBuffer>,
    row_starts: &ProtocolObject<dyn MTLBuffer>,
    k: u32,
    n: u32,
    num_jobs: u32,
) {
    let dims = [k, n, num_jobs];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(w_blob), 0, 1);
        enc.setBuffer_offset_atIndex(Some(c), 0, 2);
        enc.setBuffer_offset_atIndex(Some(jobs), 0, 3);
        enc.setBuffer_offset_atIndex(Some(row_starts), 0, 4);
    }
    gpu_common::set_bytes(enc, &dims, 5);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.format, variant)?;
    let mut pool = BufferPool::new();
    let jobs = f.jobs();
    let w_blob = f.w_blob();
    let total_m = f.total_m();
    let out_len = f.out_len();

    let buf_a = pool
        .allocate(&ctx.device, f.a.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_blob.len())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_jobs = pool
        .allocate(
            &ctx.device,
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
        .ok_or(Error::Gpu("alloc"))?;
    let buf_rs = pool
        .allocate(&ctx.device, f.row_starts.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_f32(&buf_a, &f.a);
    BufferPool::write_bytes(&buf_w, &w_blob);
    BufferPool::write_bytes(&buf_jobs, unsafe {
        std::slice::from_raw_parts(
            jobs.as_ptr().cast::<u8>(),
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
    });
    BufferPool::write_bytes(&buf_rs, unsafe {
        std::slice::from_raw_parts(f.row_starts.as_ptr().cast::<u8>(), f.row_starts.len() * 4)
    });
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);

    let grid = MTLSize {
        width: f.n,
        height: total_m,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_a,
        &buf_w,
        &buf_c,
        &buf_jobs,
        &buf_rs,
        f.k as u32,
        f.n as u32,
        f.num_jobs() as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(target_os = "macos")]
pub fn gpu_on_blob(
    p: &crate::shaders::gemm_block_grouped::BlobGroupedParams<'_>,
    _variant: KernelVariant,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, p.format, KernelVariant::PRODUCTION)?;
    let mut pool = BufferPool::new();
    let out_len = p.out_len();
    let total_m = p.total_m();
    let num_jobs = p.num_jobs();

    let buf_a = pool
        .allocate(&ctx.device, p.a.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_jobs = pool
        .allocate(&ctx.device, std::mem::size_of_val(p.jobs))
        .ok_or(Error::Gpu("alloc"))?;
    let buf_rs = pool
        .allocate(&ctx.device, p.row_starts.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_f32(&buf_a, p.a);
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);
    BufferPool::write_bytes(&buf_jobs, unsafe {
        std::slice::from_raw_parts(p.jobs.as_ptr().cast::<u8>(), std::mem::size_of_val(p.jobs))
    });
    BufferPool::write_bytes(&buf_rs, unsafe {
        std::slice::from_raw_parts(p.row_starts.as_ptr().cast::<u8>(), p.row_starts.len() * 4)
    });

    let grid = MTLSize {
        width: p.n,
        height: total_m,
        depth: 1,
    };
    let tg = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_a,
        p.blob,
        &buf_c,
        &buf_jobs,
        &buf_rs,
        p.k as u32,
        p.n as u32,
        num_jobs as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(target_os = "macos")]
pub fn gpu_on_blob_block(
    p: &crate::shaders::gemm_block_grouped::BlobGroupedParams<'_>,
    variant: KernelVariant,
) -> Result<Vec<f32>, Error> {
    crate::shaders::gemm_block_grouped::gpu_on_blob(p, variant)
}

#[cfg(target_os = "macos")]
pub fn gpu_q4(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        KernelVariant {
            quant_format: QuantFormat::Q4Affine,
            ..variant
        },
    )
}

#[cfg(target_os = "macos")]
pub fn gpu_nvfp4(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        KernelVariant {
            quant_format: QuantFormat::NvFp4,
            ..variant
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny_q4,
        cpu = crate::shaders::gemm_linear_grouped::cpu,
        cpu_oracle = crate::shaders::gemm_linear_grouped::cpu_oracle,
        gpu = crate::shaders::gemm_linear_grouped::gpu_q4,
        fixture = crate::shaders::gemm_linear_grouped::tiny_fixture_q4,
        out_len = crate::shaders::gemm_linear_grouped::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tiny_nvfp4,
        cpu = crate::shaders::gemm_linear_grouped::cpu,
        cpu_oracle = crate::shaders::gemm_linear_grouped::cpu_oracle,
        gpu = crate::shaders::gemm_linear_grouped::gpu_nvfp4,
        fixture = crate::shaders::gemm_linear_grouped::tiny_fixture_nvfp4,
        out_len = crate::shaders::gemm_linear_grouped::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tile_q4,
        cpu = crate::shaders::gemm_linear_grouped::cpu,
        cpu_oracle = crate::shaders::gemm_linear_grouped::cpu_oracle,
        gpu = crate::shaders::gemm_linear_grouped::gpu_q4,
        fixture = crate::shaders::gemm_linear_grouped::tile_fixture_q4,
        out_len = crate::shaders::gemm_linear_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile_nvfp4,
        cpu = crate::shaders::gemm_linear_grouped::cpu,
        cpu_oracle = crate::shaders::gemm_linear_grouped::cpu_oracle,
        gpu = crate::shaders::gemm_linear_grouped::gpu_nvfp4,
        fixture = crate::shaders::gemm_linear_grouped::tile_fixture_nvfp4,
        out_len = crate::shaders::gemm_linear_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod prefill_moe_nvfp4,
        cpu = crate::shaders::gemm_linear_grouped::cpu,
        cpu_oracle = crate::shaders::gemm_linear_grouped::cpu_oracle,
        gpu = crate::shaders::gemm_linear_grouped::gpu_nvfp4,
        fixture = crate::shaders::gemm_linear_grouped::prefill_moe_fixture_nvfp4,
        out_len = crate::shaders::gemm_linear_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod prefill_moe_q4,
        cpu = crate::shaders::gemm_linear_grouped::cpu,
        cpu_oracle = crate::shaders::gemm_linear_grouped::cpu_oracle,
        gpu = crate::shaders::gemm_linear_grouped::gpu_q4,
        fixture = crate::shaders::gemm_linear_grouped::prefill_moe_fixture_q4,
        out_len = crate::shaders::gemm_linear_grouped::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_tiled_grouped_matches_linear_grouped_nvfp4() {
        use crate::shaders::gemm_block_grouped;
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        use crate::shaders::variant::KernelVariant;

        for fixture_fn in [
            gemm_block_grouped::tiny_fixture_nvfp4 as fn(_) -> _,
            gemm_block_grouped::tile_fixture_nvfp4,
            tiny_fixture_nvfp4,
            tile_fixture_nvfp4,
        ] {
            let fix = fixture_fn(ElemFormat::F32);
            let tiled = gemm_block_grouped::gpu_nvfp4(&fix, KernelVariant::PRODUCTION)
                .expect("gemm_block_grouped gpu");
            let linear =
                gpu_nvfp4(&fix, KernelVariant::PRODUCTION).expect("gemm_linear_grouped gpu");
            assert_oracle(&tiled, &linear, 0.05, 0.999);
        }
    }
}

pub mod cpu;

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "gemm_linear_grouped",
    entry: "gemm_linear_grouped",
    quant_formats: &[
        crate::shaders::variant::QuantFormat::Q4Affine,
        crate::shaders::variant::QuantFormat::NvFp4,
    ],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};
