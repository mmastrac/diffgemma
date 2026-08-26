//! Threadgroup RMSNorm (monolith): half or f32 in, half out, optional bf16 weight blob.

use crate::Error;
use crate::shaders::bf16;
use crate::shaders::cpu;
use crate::shaders::gpu_common;
use crate::shaders::manifest::{self, RmsNormRowsTiledVariant};
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::{ElemDtype, KernelVariant};

pub const ENTRY: &str = "rms_norm_rows_tiled";
pub const RMS_EPS: f32 = 1e-6;
pub const THREADS_PER_TG: usize = 256;

pub const SHADER: &str = include_str!("rms_norm_rows_tiled.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TiledVariant {
    pub in_dtype: ElemDtype,
}

impl TiledVariant {
    pub const HALF_IN: Self = Self {
        in_dtype: ElemDtype::Half,
    };
    pub const F32_IN: Self = Self {
        in_dtype: ElemDtype::F32,
    };

    fn manifest(self) -> RmsNormRowsTiledVariant {
        RmsNormRowsTiledVariant {
            in_dtype: self.in_dtype,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub weight: Option<Vec<f32>>,
    pub rows: usize,
    pub dim: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows * self.dim
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: Some(vec![1.0, 0.5, 2.0, 1.5]),
        rows: 2,
        dim: 4,
    }
}

pub fn no_scale_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, -1.0, 2.0, 0.0, 0.5, -0.5, 1.5, 2.0],
        weight: None,
        rows: 2,
        dim: 4,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    if let Some(ref w) = f.weight {
        for r in 0..f.rows {
            let off = r * f.dim;
            let x = &f.x[off..off + f.dim];
            let mut row = vec![0.0f32; f.dim];
            cpu::rms_norm(&mut row, x, w, RMS_EPS);
            for i in 0..f.dim {
                out[off + i] = bf16::store_bf16_round_half(row[i]);
            }
        }
    } else {
        for r in 0..f.rows {
            let off = r * f.dim;
            let x = &f.x[off..off + f.dim];
            let mut row = vec![0.0f32; f.dim];
            cpu::rms_norm_no_scale(&mut row, x, RMS_EPS);
            for i in 0..f.dim {
                out[off + i] = bf16::store_bf16_round_half(row[i]);
            }
        }
    }
    out
}

pub fn cpu_f32_in(f: &Fixture) -> Vec<f32> {
    let mut rounded = f.clone();
    rounded.x = f.x.iter().map(|&v| bf16::round_bf16_f32(v)).collect();
    cpu(&rounded)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(rows: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        },
    )
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    tiled: TiledVariant,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    manifest::validate_shared(ENTRY, variant)?;
    let local = manifest::rms_norm_rows_tiled_variant(tiled.in_dtype)?;
    manifest::assert_no_fc_collisions(ENTRY, &[4])?;
    ctx.compile_subkernel_ex(
        SHADER,
        ENTRY,
        variant,
        local.cache_suffix(),
        &[],
        &local.local_fcs(),
    )
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    blob: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    w_off: u64,
    dim: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(y), 0, 1);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    gpu_common::set_bytes(enc, &w_off, 3);
    gpu_common::set_bytes(enc, &dim, 4);
}

#[cfg(target_os = "macos")]
pub fn gpu_half(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu_tiled(f, variant, TiledVariant::HALF_IN)
}

#[cfg(target_os = "macos")]
pub fn gpu_f32_in(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu_tiled(f, variant, TiledVariant::F32_IN)
}

#[cfg(target_os = "macos")]
fn gpu_tiled(f: &Fixture, variant: KernelVariant, tiled: TiledVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, tiled, variant)?;
    let mut pool = BufferPool::new();
    let len = f.out_len();
    let x_bytes = match tiled.in_dtype {
        ElemDtype::F32 => len * 4,
        ElemDtype::Half => len * 2,
    };
    let buf_x = pool
        .allocate(&ctx.device, x_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, len * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let (blob, w_off) = match &f.weight {
        Some(w) => {
            let mut b = vec![0u8; 2];
            b.extend(bf16::pack_bf16_slice(w));
            (b, 2u64)
        }
        None => (vec![0u8; 2], 0u64),
    };
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        f.rows * 4
    } else {
        4
    };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    if tiled.in_dtype == ElemDtype::F32 {
        BufferPool::write_f32(&buf_x, &f.x);
    } else {
        BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    }
    BufferPool::write_bytes(&buf_blob, &blob);
    let (grid, tg) = dispatch_shape(f.rows);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_blob, &buf_d, w_off, f.dim as u32);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..len)
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

crate::kernel_spec! {
    pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
        name: "rms_norm_rows_tiled",
        entry: "rms_norm_rows_tiled",
        quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
        fc: &[(4, "K_IN_DTYPE")],
        variants: crate::shaders::manifest::KernelVariants::RmsNormRowsTiled {
            rows: &[
                crate::shaders::manifest::RmsNormRowsTiledVariant {
                    in_dtype: crate::shaders::variant::ElemDtype::F32,
                },
                crate::shaders::manifest::RmsNormRowsTiledVariant {
                    in_dtype: crate::shaders::variant::ElemDtype::Half,
                },
            ],
        },
    };
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod half_tiny,
        cpu = crate::shaders::rms_norm_rows_tiled::cpu,
        cpu_oracle = crate::shaders::rms_norm_rows_tiled::cpu_oracle,
        gpu = crate::shaders::rms_norm_rows_tiled::gpu_half,
        fixture = crate::shaders::rms_norm_rows_tiled::tiny_fixture,
        out_len = crate::shaders::rms_norm_rows_tiled::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod half_no_scale,
        cpu = crate::shaders::rms_norm_rows_tiled::cpu,
        cpu_oracle = crate::shaders::rms_norm_rows_tiled::cpu_oracle,
        gpu = crate::shaders::rms_norm_rows_tiled::gpu_half,
        fixture = crate::shaders::rms_norm_rows_tiled::no_scale_fixture,
        out_len = crate::shaders::rms_norm_rows_tiled::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod f32_in_tiny,
        cpu = crate::shaders::rms_norm_rows_tiled::cpu_f32_in,
        cpu_oracle = crate::shaders::rms_norm_rows_tiled::cpu_f32_in,
        gpu = crate::shaders::rms_norm_rows_tiled::gpu_f32_in,
        fixture = crate::shaders::rms_norm_rows_tiled::tiny_fixture,
        out_len = crate::shaders::rms_norm_rows_tiled::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }
}
