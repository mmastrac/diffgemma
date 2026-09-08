//! `out += addend` elementwise.

use crate::Error;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

crate::shader_kernel! {
    name = "vec_add_inplace",
    metal = "vec_add_inplace.metal",
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
    pub out: Vec<f32>,
    pub addend: Vec<f32>,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.out.len()
    }
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        out: vec![1.0, 2.0, 3.0, 4.0],
        addend: vec![0.5, -1.0, 2.0, 0.0],
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = f.out.clone();
    for (o, a) in out.iter_mut().zip(f.addend.iter()) {
        *o += *a;
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
    out: &ProtocolObject<dyn MTLBuffer>,
    addend: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(addend), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
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
    let buf_o = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_a = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_o, &f.out);
    BufferPool::write_f32(&buf_a, &f.addend);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf_o, &buf_a, &buf_d, len as u32);
    })?;
    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_o, &mut out);
    Ok(out)
}
