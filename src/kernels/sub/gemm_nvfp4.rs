//! Tiled NVFP4 GEMM: `y[M,N] = x[M,K] @ Wnvfp4[N,K]^T` (monolith `k_gemm_nvfp4` body).

use super::bf16;
use super::f16;
use super::gemm_common;
use super::test_util::ElemFormat;
use crate::dgq::layout::{nvfp4_matrix_bytes, NVFP4_HEADER_BYTES};
use crate::dgq::nvfp4::{nvfp4_gemm_cpu, quantize_f32_matrix_nvfp4_with_scale};
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_block";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_block.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub w_f32: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub global_scale: f32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }

    pub fn w_nvfp4(&self) -> Vec<u8> {
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

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let m = 4usize;
    let n = 64usize;
    let k = 64usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.02)
        .collect();
    Fixture { x, w_f32, m, n, k, global_scale: 1.0 }
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    let m = 8usize;
    let n = 128usize;
    let k = 128usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.011).sin() * 0.1)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.005).cos() * 0.03)
        .collect();
    Fixture {
        x,
        w_f32,
        m,
        n,
        k,
        global_scale: 1.0,
    }
}

pub fn gscale_fixture(_: ElemFormat) -> Fixture {
    let mut fix = tiny_fixture(ElemFormat::F32);
    fix.global_scale = 1.234_567;
    fix
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w = f.w_nvfp4();
    let gscale = f32::from_le_bytes(w[0..4].try_into().unwrap());
    let body = &w[NVFP4_HEADER_BYTES..];
    let mut out = vec![0.0f32; f.out_len()];
    nvfp4_gemm_cpu(&f.x, f.m, f.k, body, f.n, gscale, &mut out);
    out.iter().map(|&v| bf16::store_bf16_round_half(v)).collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(
        SHADER,
        ENTRY,
        n,
        k,
        false,
        super::QuantFormat::NvFp4 as u32,
        false,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    blob: &ProtocolObject<dyn MTLBuffer>,
    w_off: u64,
    m: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(y), 0, 1);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 2);
    }
    super::gpu_common::set_bytes(enc, &w_off, 3);
    super::gpu_common::set_bytes(enc, &m, 4);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, _variant: super::KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.n as u32, f.k as u32)?;
    let mut pool = BufferPool::new();
    let w_nvfp4 = f.w_nvfp4();
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.m * f.n * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_nvfp4.len())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bytes(&buf_w, &w_nvfp4);
    let (grid, tg) = gemm_common::dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, f.m as u32);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: super::KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;
    use crate::kernels::sub::test_util::{assert_oracle, ElemFormat};
    use crate::kernels::sub::variant::KernelVariant;
    use crate::kernels::sub::QuantFormat;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::gemm_nvfp4::cpu,
        cpu_oracle = crate::kernels::sub::gemm_nvfp4::cpu_oracle,
        gpu = crate::kernels::sub::gemm_nvfp4::gpu,
        fixture = crate::kernels::sub::gemm_nvfp4::tiny_fixture,
        out_len = crate::kernels::sub::gemm_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 0.08,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::kernels::sub::gemm_nvfp4::cpu,
        cpu_oracle = crate::kernels::sub::gemm_nvfp4::cpu_oracle,
        gpu = crate::kernels::sub::gemm_nvfp4::gpu,
        fixture = crate::kernels::sub::gemm_nvfp4::tile_fixture,
        out_len = crate::kernels::sub::gemm_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 0.08,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod gscale,
        cpu = crate::kernels::sub::gemm_nvfp4::cpu,
        cpu_oracle = crate::kernels::sub::gemm_nvfp4::cpu_oracle,
        gpu = crate::kernels::sub::gemm_nvfp4::gpu,
        fixture = crate::kernels::sub::gemm_nvfp4::gscale_fixture,
        out_len = crate::kernels::sub::gemm_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 0.08,
        min_cos = 0.999,
    }

    fn linear_f32_fixture(gf: &Fixture) -> crate::kernels::sub::gemm_linear_f32::Fixture {
        crate::kernels::sub::gemm_linear_f32::Fixture {
            x: gf.x.clone(),
            w_f32: gf.w_f32.clone(),
            m: gf.m,
            n: gf.n,
            k: gf.k,
            format: QuantFormat::NvFp4,
            global_scale: gf.global_scale,
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_tiled_matches_linear_f32() {
        for fixture_fn in [tiny_fixture as fn(_) -> _, tile_fixture, gscale_fixture] {
            let gf = fixture_fn(ElemFormat::F32);
            let lf = linear_f32_fixture(&gf);
            let tiled = gpu(&gf, KernelVariant::PRODUCTION).expect("gemm_block gpu");
            let linear =
                crate::kernels::sub::gemm_linear_f32::gpu_nvfp4(&lf, KernelVariant::PRODUCTION)
                    .expect("gemm_linear_f32 gpu");
            assert_oracle(&tiled, &linear, 0.08, 0.999);
        }
    }
}
