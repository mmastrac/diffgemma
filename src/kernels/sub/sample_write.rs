//! Sampler pass 4: apply accept mask and renoise rejected positions.

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::sampler::sample_write_cpu;
use crate::metal::{CANVAS, CanvasState};
use crate::safetensors::Error;

pub const ENTRY: &str = "sample_write";
pub const THREADGROUP_WIDTH: usize = 256;

const SHADER: &str = shader_include::include_metal!("kernels/sample_write.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub ids: Vec<u32>,
    pub accept: Vec<u32>,
    pub new_sample: Vec<u32>,
    pub canvas_size: usize,
    pub vocab_size: u32,
    pub rng_state: u64,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.canvas_size + 1
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        ids: vec![1, 2, 3, 4],
        accept: vec![1, 0, 1, 0],
        new_sample: vec![10, 20, 30, 40],
        canvas_size: 4,
        vocab_size: 256,
        rng_state: 99,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        ids: (0..8).collect(),
        accept: vec![1, 1, 0, 0, 1, 0, 1, 0],
        new_sample: (100..108).collect(),
        canvas_size: 8,
        vocab_size: 512,
        rng_state: 42,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut ids = f.ids.clone();
    let rng = sample_write_cpu(
        &mut ids,
        &f.accept,
        &f.new_sample,
        f.canvas_size,
        f.vocab_size,
        f.rng_state,
    );
    let mut out: Vec<f32> = ids.iter().map(|&v| v as f32).collect();
    out.push(rng as f32);
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

fn canvas_state_for_gpu(f: &Fixture) -> CanvasState {
    let mut ids = [0u32; crate::metal::PREFILL_M];
    let mut accept = [0u32; CANVAS];
    let mut new_sample = [0u32; CANVAS];
    for i in 0..f.canvas_size {
        ids[i] = f.ids[i];
        accept[i] = f.accept[i];
        new_sample[i] = f.new_sample[i];
    }
    CanvasState {
        ids,
        prev_argmax: [0; CANVAS],
        new_sample,
        entropy: [0.0; CANVAS],
        sorted_idx: [0; CANVAS],
        accept,
        u_cat: [0.0; CANVAS],
        rng_state: f.rng_state,
        step: 0,
        stop_flag: 0,
        argmax_hist_len: 0,
        argmax_hist_base: 0,
        argmax_hist: [0; CANVAS * crate::sample::ARGMAX_HIST_MAX],
        canvas_stable: 0,
        mean_entropy: 0.0,
        accept_plateau: 0,
        prev_accept_sig: 0,
        frozen: [0; crate::metal::FROZEN_WORDS],
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_state = pool
        .allocate(&ctx.device, std::mem::size_of::<CanvasState>())
        .ok_or(Error::Format("alloc"))?;
    let state = canvas_state_for_gpu(f);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &state as *const CanvasState as *const u8,
            std::mem::size_of::<CanvasState>(),
        )
    };
    BufferPool::write_bytes(&buf_state, bytes);

    let canvas = f.canvas_size as u32;
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 0);
    }
    gpu_common::set_bytes(&enc, &canvas, 1);
    gpu_common::set_bytes(&enc, &f.vocab_size, 2);
    let freeze_enable: u32 = 1; // historic default; CPU mirror has no freeze concept
    let use_argmax: u32 = 0;
    gpu_common::set_bytes(&enc, &freeze_enable, 4);
    gpu_common::set_bytes(&enc, &use_argmax, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut ids = vec![0u32; f.canvas_size];
    BufferPool::read_u32(&buf_state, &mut ids);
    let mut rng_words = [0u32; 2];
    BufferPool::read_u32_at_offset(
        &buf_state,
        std::mem::offset_of!(CanvasState, rng_state),
        &mut rng_words,
    );
    let rng_state = (rng_words[0] as u64) | ((rng_words[1] as u64) << 32);
    let mut out: Vec<f32> = ids.iter().map(|&v| v as f32).collect();
    out.push(rng_state as f32);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::sample_write::cpu,
        cpu_oracle = crate::kernels::sub::sample_write::cpu_oracle,
        gpu = crate::kernels::sub::sample_write::gpu,
        fixture = crate::kernels::sub::sample_write::tiny_fixture,
        out_len = crate::kernels::sub::sample_write::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::sample_write::cpu,
        cpu_oracle = crate::kernels::sub::sample_write::cpu_oracle,
        gpu = crate::kernels::sub::sample_write::gpu,
        fixture = crate::kernels::sub::sample_write::wide_fixture,
        out_len = crate::kernels::sub::sample_write::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
