//! Fused GELU×up from `[batch, 2*moe_inter]` gate_up layout.

use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::{gelu_pytorch_tanh_f32, gelu_pytorch_tanh};
use crate::safetensors::Error;

pub const ENTRY: &str = "gelu_swiglu_gate_up";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/activations.metal"),
    include_str!("../../../shaders/kernels/gelu_swiglu_gate_up.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub gate_up: Vec<f32>,
    pub batch_size: usize,
    pub moe_inter: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.batch_size * self.moe_inter
    }
}

pub fn fixture_len(fix: &Fixture) -> usize {
    fix.out_len()
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    Fixture {
        gate_up: vec![1.0, -1.0, 2.0, 0.5, 0.0, 1.0, -0.5, 2.0],
        batch_size: 2,
        moe_inter: 2,
    }
}

pub fn moe_fixture(_fmt: ElemFormat) -> Fixture {
    let batch_size = 8;
    let moe_inter = 704;
    let len = batch_size * moe_inter * 2;
    Fixture {
        gate_up: (0..len).map(|i| ((i as f32) * 0.002).sin()).collect(),
        batch_size,
        moe_inter,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.out_len()];
    let mi = fix.moe_inter;
    for b in 0..fix.batch_size {
        for j in 0..mi {
            let off = b * (2 * mi) + j;
            let g = gelu_pytorch_tanh_f32(fix.gate_up[off]);
            let u = fix.gate_up[off + mi];
            out[b * mi + j] = g * u;
        }
    }
    out
}

pub fn cpu_oracle(fix: &Fixture) -> Vec<f32> {
    let mut gate: Vec<f32> = fix
        .gate_up
        .chunks(fix.moe_inter * 2)
        .flat_map(|row| {
            let mi = fix.moe_inter;
            let mut g: Vec<f32> = row[..mi].to_vec();
            gelu_pytorch_tanh(&mut g);
            g.into_iter()
                .zip(row[mi..].iter())
                .map(|(gv, &u)| gv * u)
                .collect::<Vec<_>>()
        })
        .collect();
    debug_assert_eq!(gate.len(), fix.out_len());
    gate
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(fix: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let in_len = fix.gate_up.len();
    let out_len = fix.out_len();
    let buf_in = pool.allocate(&ctx.device, in_len * 4).ok_or(Error::Format("alloc"))?;
    let buf_out = pool.allocate(&ctx.device, out_len * 4).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { out_len * 4 } else { 4 };
    let buf_dump = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;

    BufferPool::write_f32(&buf_in, &fix.gate_up);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    let dims = [fix.batch_size as u32, fix.moe_inter as u32];
    bind_gpu_buffers(&enc, &buf_in, &buf_out, &buf_dump, &dims);
    let tg = 256usize.min(out_len);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: div_up(out_len, tg), height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_out, &mut out);
    pool.release(in_len * 4, buf_in);
    pool.release(out_len * 4, buf_out);
    pool.release(dump_bytes, buf_dump);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_fix: &Fixture, _variant: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_up), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    set_bytes(enc, dims, 2);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::gelu_swiglu_gate_up::cpu,
        cpu_oracle = crate::kernels::sub::gelu_swiglu_gate_up::cpu_oracle,
        gpu = crate::kernels::sub::gelu_swiglu_gate_up::gpu,
        fixture = crate::kernels::sub::gelu_swiglu_gate_up::tiny_fixture,
        out_len = crate::kernels::sub::gelu_swiglu_gate_up::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe,
        cpu = crate::kernels::sub::gelu_swiglu_gate_up::cpu,
        cpu_oracle = crate::kernels::sub::gelu_swiglu_gate_up::cpu_oracle,
        gpu = crate::kernels::sub::gelu_swiglu_gate_up::gpu,
        fixture = crate::kernels::sub::gelu_swiglu_gate_up::moe_fixture,
        out_len = crate::kernels::sub::gelu_swiglu_gate_up::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}
