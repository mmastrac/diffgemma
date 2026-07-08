//! NVFP4 fused 32-wide tile decode parity (`dequant_nvfp4_tile_half_fused` vs `nvfp4_at_col`).

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::layout::nvfp4_matrix_bytes;
use crate::dgq::nvfp4::{nvfp4_weight_at, quantize_f32_matrix_nvfp4_with_scale};
use crate::kernels::cpu::dequant::nvfp4_tile_k_order_matrix;
use crate::safetensors::Error;

pub const ENTRY: &str = "nvfp4_tile_k_order";

const SHADER: &str = shader_include::include_metal!("kernels/nvfp4_tile_k_order.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub w_f32: Vec<f32>,
    pub n: usize,
    pub k: usize,
    pub row: usize,
    pub k0: usize,
    pub global_scale: f32,
}

impl Fixture {
    pub fn matrix(&self) -> Vec<u8> {
        let mut dst = vec![0u8; nvfp4_matrix_bytes(self.n, self.k)];
        quantize_f32_matrix_nvfp4_with_scale(
            &self.w_f32,
            self.n,
            self.k,
            &mut dst,
            self.global_scale,
        );
        dst
    }
}

pub fn fixture_len(_: &Fixture) -> usize {
    64
}

pub fn tiny_k0_fixture(_: ElemFormat) -> Fixture {
    row_matrix_fixture(1, 64, 0, 0, 1.0)
}

pub fn tile_k32_fixture(_: ElemFormat) -> Fixture {
    row_matrix_fixture(4, 128, 2, 32, 1.0)
}

pub fn gscale_fixture(_: ElemFormat) -> Fixture {
    row_matrix_fixture(1, 64, 0, 0, 1.234_567)
}

pub fn wide_k48_fixture(_: ElemFormat) -> Fixture {
    row_matrix_fixture(2, 80, 1, 48, 0.875)
}

fn row_matrix_fixture(n: usize, k: usize, row: usize, k0: usize, global_scale: f32) -> Fixture {
    assert!(row < n);
    assert!(k0 + 32 <= k);
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.017).sin() * 0.22 + ((i as f32) * 0.003).cos() * 0.04)
        .collect();
    Fixture {
        w_f32,
        n,
        k,
        row,
        k0,
        global_scale,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    nvfp4_tile_k_order_matrix(&f.matrix(), f.row, f.k0, f.k)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    let matrix = f.matrix();
    let gscale = f32::from_le_bytes(matrix[..4].try_into().expect("nvfp4 header"));
    let body = &matrix[4..];
    let mut out = Vec::with_capacity(64);
    for m in 0..32 {
        out.push(nvfp4_weight_at(body, f.row, f.k0 + m, f.k, gscale));
    }
    out.extend(out.clone());
    out
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    matrix: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    k0: u32,
    k_dim: u32,
    row: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(matrix), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 4);
    }
    gpu_common::set_bytes(enc, &k0, 1);
    gpu_common::set_bytes(enc, &k_dim, 2);
    gpu_common::set_bytes(enc, &row, 3);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let mut run_variant = variant;
    if run_variant.dump_stage == 0 {
        run_variant.dump_stage = 1;
    }
    let pipeline = pipeline_for(&ctx, run_variant)?;
    let matrix = f.matrix();
    let mut pool = BufferPool::new();
    let buf_matrix = pool
        .allocate(&ctx.device, matrix.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_dump = pool
        .allocate(&ctx.device, 64 * 4)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bytes(&buf_matrix, &matrix);

    let k0 = f.k0 as u32;
    let k_dim = f.k as u32;
    let row = f.row as u32;
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, 1, |enc| {
        bind_gpu_buffers(enc, &buf_matrix, &buf_dump, k0, k_dim, row);
    })?;

    let mut out = vec![0.0f32; 64];
    BufferPool::read_f32(&buf_dump, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;
    use crate::kernels::sub::test_util::assert_oracle;

    kernel_oracle_matrix! {
        mod tiny_k0,
        cpu = crate::kernels::sub::nvfp4_tile_k_order::cpu,
        cpu_oracle = crate::kernels::sub::nvfp4_tile_k_order::cpu_oracle,
        gpu = crate::kernels::sub::nvfp4_tile_k_order::gpu,
        fixture = crate::kernels::sub::nvfp4_tile_k_order::tiny_k0_fixture,
        out_len = crate::kernels::sub::nvfp4_tile_k_order::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.99999,
    }

    kernel_oracle_matrix! {
        mod tile_k32,
        cpu = crate::kernels::sub::nvfp4_tile_k_order::cpu,
        cpu_oracle = crate::kernels::sub::nvfp4_tile_k_order::cpu_oracle,
        gpu = crate::kernels::sub::nvfp4_tile_k_order::gpu,
        fixture = crate::kernels::sub::nvfp4_tile_k_order::tile_k32_fixture,
        out_len = crate::kernels::sub::nvfp4_tile_k_order::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.99999,
    }

    kernel_oracle_matrix! {
        mod gscale,
        cpu = crate::kernels::sub::nvfp4_tile_k_order::cpu,
        cpu_oracle = crate::kernels::sub::nvfp4_tile_k_order::cpu_oracle,
        gpu = crate::kernels::sub::nvfp4_tile_k_order::gpu,
        fixture = crate::kernels::sub::nvfp4_tile_k_order::gscale_fixture,
        out_len = crate::kernels::sub::nvfp4_tile_k_order::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.99999,
    }

    kernel_oracle_matrix! {
        mod wide_k48,
        cpu = crate::kernels::sub::nvfp4_tile_k_order::cpu,
        cpu_oracle = crate::kernels::sub::nvfp4_tile_k_order::cpu_oracle,
        gpu = crate::kernels::sub::nvfp4_tile_k_order::gpu,
        fixture = crate::kernels::sub::nvfp4_tile_k_order::wide_k48_fixture,
        out_len = crate::kernels::sub::nvfp4_tile_k_order::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.99999,
    }

    #[test]
    fn cpu_fused_matches_column() {
        for fixture_fn in [
            tiny_k0_fixture as fn(ElemFormat) -> Fixture,
            tile_k32_fixture,
            gscale_fixture,
            wide_k48_fixture,
        ] {
            let fix = fixture_fn(ElemFormat::F32);
            let got = cpu(&fix);
            let expected = cpu_oracle(&fix);
            assert_oracle(&got, &expected, 1e-5, 0.999999);
        }
    }
}
