//! ORACLE (not production): validation twin of `gemm_tunable` dense q4. Shared
//! fixture/runner in [`super::fixture`]; this module supplies the q4 CPU
//! reference + weight quant. Kernel source: `gemm_block/gemm_block.metal`.

use crate::Error;
use crate::dgq::block::{q4_gemm_cpu, quantize_row_q4};
use crate::dgq::layout::{q4_matrix_bytes, q4_row_bytes};
use crate::shaders::QuantFormat;
use crate::shaders::bf16;
use crate::shaders::test_util::ElemFormat;

#[allow(unused_imports)]
pub use super::fixture::{Fixture, bind_gpu_buffers, fixture_len};

pub fn w_q4(f: &Fixture) -> Vec<u8> {
    let mut dst = vec![0u8; q4_matrix_bytes(f.n, f.k)];
    let row_bytes = q4_row_bytes(f.k);
    for row in 0..f.n {
        let off = row * f.k;
        quantize_row_q4(&f.w_f32[off..off + f.k], f.k, &mut dst[row * row_bytes..]);
    }
    dst
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    super::fixture::make_fixture(4, 64, 64, (0.013, 0.25, 0.0), (0.007, 0.02, 0.0), 1.0)
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    super::fixture::make_fixture(8, 128, 128, (0.011, 0.1, 0.0), (0.005, 0.03, 0.0), 1.0)
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w_q4 = w_q4(f);
    let mut out = vec![0.0f32; f.out_len()];
    q4_gemm_cpu(&f.x, f.m, f.k, &w_q4, f.n, &mut out);
    out.iter()
        .map(|&v| bf16::store_bf16_round_half(v))
        .collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    super::fixture::pipeline_for(ctx, n, k, QuantFormat::Q4Affine)
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, _variant: crate::shaders::KernelVariant) -> Result<Vec<f32>, Error> {
    super::fixture::run_gpu(f, QuantFormat::Q4Affine, &w_q4(f))
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::gemm_q4::cpu,
        cpu_oracle = crate::shaders::gemm_q4::cpu_oracle,
        gpu = crate::shaders::gemm_q4::gpu,
        fixture = crate::shaders::gemm_q4::tiny_fixture,
        out_len = crate::shaders::gemm_q4::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::shaders::gemm_q4::cpu,
        cpu_oracle = crate::shaders::gemm_q4::cpu_oracle,
        gpu = crate::shaders::gemm_q4::gpu,
        fixture = crate::shaders::gemm_q4::tile_fixture,
        out_len = crate::shaders::gemm_q4::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }
}
