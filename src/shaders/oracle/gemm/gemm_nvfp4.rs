//! ORACLE (not production): validation twin of `gemm_tunable` dense nvfp4.
//! Shared fixture/runner in [`super::fixture`]; this module supplies the nvfp4
//! CPU reference + weight quant (with per-fixture `global_scale`). Kernel
//! source: `gemm_block/gemm_block.metal`.

use crate::Error;
use crate::dgq::layout::{NVFP4_HEADER_BYTES, nvfp4_matrix_bytes};
use crate::dgq::nvfp4::{nvfp4_gemm_cpu, quantize_f32_matrix_nvfp4_with_scale};
use crate::shaders::QuantFormat;
use crate::shaders::bf16;
use crate::shaders::test_util::ElemFormat;

#[allow(unused_imports)]
pub use super::fixture::{Fixture, bind_gpu_buffers, fixture_len};

pub fn w_nvfp4(f: &Fixture) -> Vec<u8> {
    let mut dst = vec![0u8; nvfp4_matrix_bytes(f.n, f.k)];
    quantize_f32_matrix_nvfp4_with_scale(&f.w_f32, f.n, f.k, &mut dst, f.global_scale);
    dst
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    super::fixture::make_fixture(4, 64, 64, (0.013, 0.25, 0.0), (0.007, 0.02, 0.0), 1.0)
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    super::fixture::make_fixture(8, 128, 128, (0.011, 0.1, 0.0), (0.005, 0.03, 0.0), 1.0)
}

pub fn gscale_fixture(_: ElemFormat) -> Fixture {
    let mut fix = tiny_fixture(ElemFormat::F32);
    fix.global_scale = 1.234_567;
    fix
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w = w_nvfp4(f);
    let gscale = f32::from_le_bytes(w[0..4].try_into().unwrap());
    let body = &w[NVFP4_HEADER_BYTES..];
    let mut out = vec![0.0f32; f.out_len()];
    nvfp4_gemm_cpu(&f.x, f.m, f.k, body, f.n, gscale, &mut out);
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
    super::fixture::pipeline_for(ctx, n, k, QuantFormat::NvFp4)
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, _variant: crate::shaders::KernelVariant) -> Result<Vec<f32>, Error> {
    super::fixture::run_gpu(f, QuantFormat::NvFp4, &w_nvfp4(f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;
    use crate::shaders::test_util::assert_oracle;
    use crate::shaders::variant::KernelVariant;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::gemm_nvfp4::cpu,
        cpu_oracle = crate::shaders::gemm_nvfp4::cpu_oracle,
        gpu = crate::shaders::gemm_nvfp4::gpu,
        fixture = crate::shaders::gemm_nvfp4::tiny_fixture,
        out_len = crate::shaders::gemm_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 0.08,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::shaders::gemm_nvfp4::cpu,
        cpu_oracle = crate::shaders::gemm_nvfp4::cpu_oracle,
        gpu = crate::shaders::gemm_nvfp4::gpu,
        fixture = crate::shaders::gemm_nvfp4::tile_fixture,
        out_len = crate::shaders::gemm_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 0.08,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod gscale,
        cpu = crate::shaders::gemm_nvfp4::cpu,
        cpu_oracle = crate::shaders::gemm_nvfp4::cpu_oracle,
        gpu = crate::shaders::gemm_nvfp4::gpu,
        fixture = crate::shaders::gemm_nvfp4::gscale_fixture,
        out_len = crate::shaders::gemm_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 0.08,
        min_cos = 0.999,
    }

    fn linear_f32_fixture(gf: &Fixture) -> crate::shaders::gemm_linear_f32::Fixture {
        crate::shaders::gemm_linear_f32::Fixture {
            x: gf.x.clone(),
            w_f32: gf.w_f32.clone(),
            m: gf.m,
            n: gf.n,
            k: gf.k,
            format: QuantFormat::NvFp4,
            global_scale: gf.global_scale,
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_tiled_matches_linear_f32() {
        for fixture_fn in [tiny_fixture as fn(_) -> _, tile_fixture, gscale_fixture] {
            let gf = fixture_fn(ElemFormat::F32);
            let lf = linear_f32_fixture(&gf);
            let tiled = gpu(&gf, KernelVariant::PRODUCTION).expect("gemm_block gpu");
            let linear = crate::shaders::gemm_linear_f32::gpu_nvfp4(&lf, KernelVariant::PRODUCTION)
                .expect("gemm_linear_f32 gpu");
            assert_oracle(&tiled, &linear, 0.08, 0.999);
        }
    }
}
