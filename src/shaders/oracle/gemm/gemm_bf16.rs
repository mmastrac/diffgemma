//! ORACLE (not production): validation twin of `gemm_tunable` dense raw (bf16).
//! Shared fixture/runner in [`super::fixture`]; this module supplies the bf16
//! (no-quant) CPU reference + weight bytes. The `Raw` weight format loads bf16
//! weights directly (no dequant). Kernel source: `gemm_block/gemm_block.metal`.

use crate::Error;
use crate::shaders::QuantFormat;
use crate::shaders::bf16;
use crate::shaders::test_util::ElemFormat;

#[allow(unused_imports)]
pub use super::fixture::{ENTRY, Fixture, SHADER, fixture_len};

/// bf16 weights as raw bf16-bit bytes (`Raw` = no quantization).
fn w_bytes(f: &Fixture) -> Vec<u8> {
    bf16::f32_slice_to_bf16_bits(&f.w_f32)
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect()
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    // Non-uniform, multi-tile shape (N=160 > 128 N-tile, K=160 > 32 K-tile,
    // M=40 > 32 M-tile) so the test exercises grid tiling + tail handling.
    super::fixture::make_fixture(40, 160, 160, (0.017, 0.3, -0.05), (0.011, 0.04, 0.01), 1.0)
}

/// `y[m,n] = sum_k bf16(x) * bf16(w)`, f32 accumulate, bf16-rounded result —
/// mirrors the kernel (bf16 inputs into half tiles, float8x8 accumulate).
pub fn cpu(f: &Fixture) -> Vec<f32> {
    let xb: Vec<f32> = f.x.iter().map(|&v| bf16::round_bf16_f32(v)).collect();
    let wb: Vec<f32> = f.w_f32.iter().map(|&v| bf16::round_bf16_f32(v)).collect();
    let mut out = vec![0.0f32; f.out_len()];
    for mi in 0..f.m {
        for ni in 0..f.n {
            let mut acc = 0.0f32;
            for ki in 0..f.k {
                acc += xb[mi * f.k + ki] * wb[ni * f.k + ki];
            }
            out[mi * f.n + ni] = bf16::store_bf16_round_half(acc);
        }
    }
    out
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
    super::fixture::pipeline_for(ctx, n, k, QuantFormat::Raw)
}

/// lm_head logits pipeline: forces bf16 output (FC29) so logits keep bf16's
/// range even when K_ACT_F16 (f16 activations) is on for the input.
#[cfg(target_os = "macos")]
pub fn pipeline_for_logits(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_out_bf16(SHADER, ENTRY, n, k, QuantFormat::Raw as u32)
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, _variant: crate::shaders::KernelVariant) -> Result<Vec<f32>, Error> {
    super::fixture::run_gpu(f, QuantFormat::Raw, &w_bytes(f))
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::shaders::gemm_bf16::cpu,
        cpu_oracle = crate::shaders::gemm_bf16::cpu_oracle,
        gpu = crate::shaders::gemm_bf16::gpu,
        fixture = crate::shaders::gemm_bf16::tile_fixture,
        out_len = crate::shaders::gemm_bf16::fixture_len,
        formats: [F32],
        max_tol = 0.02,
        min_cos = 0.999,
    }
}
