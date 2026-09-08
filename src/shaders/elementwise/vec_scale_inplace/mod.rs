//! `x *= scale` elementwise.

use crate::Error;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

crate::shader_kernel! {
    name = "vec_scale_inplace",
    metal = "vec_scale_inplace.metal",
    pipeline = |ctx, variant| ctx.compile_subkernel(SHADER, ENTRY, variant),
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = [
        tiny => tiny_fixture => (1e-6, 0.9999),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub scale: f32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.x.len()
    }
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, -2.0, 3.0, 0.0],
        scale: 0.5,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = f.x.clone();
    for v in &mut out {
        *v *= f.scale;
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    scale: f32,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    gpu_common::set_bytes(enc, &scale, 1);
    gpu_common::set_bytes(enc, &len, 2);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.out_len();
    let buf = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf, &f.x);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf, &buf_d, f.scale, len as u32);
    })?;
    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf, &mut out);
    Ok(out)
}
