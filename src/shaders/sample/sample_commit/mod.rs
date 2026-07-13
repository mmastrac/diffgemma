//! Sampler pass 2: LCG u_cat, entropy sort, accept mask, early stop.

use crate::metal::{CANVAS, CanvasState, StepParams};
use crate::safetensors::Error;
use crate::sample::ARGMAX_HIST_MAX;
use crate::sample::{FILLER_TOKEN_ID, PAD_TOKEN_ID};
use crate::shaders::cpu::sampler::{CommitParams, sample_commit_cpu};
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "sample_commit";
pub const THREADGROUP_WIDTH: usize = 256;

pub const SHADER: &str = include_str!("sample_commit.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub entropy: Vec<f32>,
    pub prev_argmax: Vec<u32>,
    pub canvas_size: usize,
    pub step: u32,
    pub argmax_hist_len: u32,
    pub argmax_hist_base: u32,
    pub rng_state: u64,
    pub params: StepParams,
    pub pad_token: u32,
    pub filler_token: u32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.canvas_size * 3 + 6
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        entropy: vec![0.05, 0.15, 0.25, 0.35],
        prev_argmax: vec![10, 20, 30, 40],
        canvas_size: 4,
        step: 1,
        argmax_hist_len: 1,
        argmax_hist_base: 0,
        rng_state: 42,
        params: StepParams {
            kv_len: 0,
            max_steps: 8,
            entropy_bound: 0.4,
            t_min: 0.3,
            t_max: 1.0,
            conf_threshold: f32::MAX,
            stability_threshold: 99,
            min_early_stop_steps: 12,
            accept_plateau_threshold: 2,
            plateau_prefix_mean_max: 0.05,
            eos_token_id: 1,
            kv_write_end: u32::MAX,
        },
        pad_token: PAD_TOKEN_ID,
        filler_token: FILLER_TOKEN_ID,
    }
}

pub fn final_step_fixture(_: ElemFormat) -> Fixture {
    let mut f = tiny_fixture(ElemFormat::F32);
    f.step = 7;
    f.params.max_steps = 8;
    f
}

fn pack_out(state: &CanvasState, canvas: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(canvas * 3 + 6);
    for i in 0..canvas {
        out.push(state.u_cat[i]);
        out.push(state.accept[i] as f32);
        out.push(state.sorted_idx[i] as f32);
    }
    out.push(state.mean_entropy);
    out.push(state.canvas_stable as f32);
    out.push(state.argmax_hist_len as f32);
    out.push(state.step as f32);
    out.push(state.stop_flag as f32);
    out.push(state.rng_state as f32);
    out
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut state = blank_state(f);
    let mut hist = state.argmax_hist;
    let out = sample_commit_cpu(
        f.step,
        f.rng_state,
        CommitParams {
            max_steps: f.params.max_steps,
            entropy_bound: f.params.entropy_bound,
            conf_threshold: f.params.conf_threshold,
            stability_threshold: f.params.stability_threshold,
            min_early_stop_steps: f.params.min_early_stop_steps,
            accept_plateau_threshold: f.params.accept_plateau_threshold,
            plateau_prefix_mean_max: f.params.plateau_prefix_mean_max,
            canvas_size: f.canvas_size,
            pad_token: f.pad_token,
            filler_token: f.filler_token,
            eos_token_id: f.params.eos_token_id,
            entropy: &f.entropy,
            prev_argmax: &f.prev_argmax,
            accept_plateau: state.accept_plateau,
            prev_accept_sig: state.prev_accept_sig,
            argmax_hist_len: f.argmax_hist_len,
            argmax_hist_base: f.argmax_hist_base,
            argmax_hist: &mut hist,
        },
    );
    for i in 0..f.canvas_size {
        state.u_cat[i] = out.u_cat[i];
        state.accept[i] = out.accept[i];
        state.sorted_idx[i] = out.sorted_idx[i];
    }
    state.mean_entropy = out.mean_entropy;
    state.canvas_stable = out.canvas_stable;
    state.argmax_hist_len = out.argmax_hist_len;
    state.argmax_hist_base = out.argmax_hist_base;
    state.step = out.step;
    state.stop_flag = out.stop_flag;
    state.rng_state = out.rng_state;
    state.accept_plateau = out.accept_plateau;
    state.prev_accept_sig = out.prev_accept_sig;
    state.argmax_hist = hist;
    pack_out(&state, f.canvas_size)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

fn blank_state(f: &Fixture) -> CanvasState {
    let mut entropy = [0.0f32; CANVAS];
    let mut prev = [0u32; CANVAS];
    let mut ids = [0u32; crate::metal::PREFILL_M];
    for i in 0..f.canvas_size {
        entropy[i] = f.entropy[i];
        prev[i] = f.prev_argmax[i];
        ids[i] = 100 + i as u32;
    }
    CanvasState {
        ids,
        prev_argmax: prev,
        new_sample: [0; CANVAS],
        entropy,
        sorted_idx: [0; CANVAS],
        accept: [0; CANVAS],
        u_cat: [0.0; CANVAS],
        rng_state: f.rng_state,
        step: f.step,
        stop_flag: 0,
        argmax_hist_len: f.argmax_hist_len,
        argmax_hist_base: f.argmax_hist_base,
        argmax_hist: [0; CANVAS * ARGMAX_HIST_MAX],
        canvas_stable: 0,
        mean_entropy: 0.0,
        accept_plateau: 0,
        prev_accept_sig: 0,
        frozen: [0; crate::metal::FROZEN_WORDS],
    }
}

fn state_to_bytes(state: &CanvasState) -> Vec<u8> {
    let len = std::mem::size_of::<CanvasState>();
    let mut bytes = vec![0u8; len];
    unsafe {
        std::ptr::copy_nonoverlapping(
            state as *const CanvasState as *const u8,
            bytes.as_mut_ptr(),
            len,
        );
    }
    bytes
}

fn bytes_to_state(bytes: &[u8]) -> CanvasState {
    assert!(bytes.len() >= std::mem::size_of::<CanvasState>());
    unsafe { std::ptr::read(bytes.as_ptr() as *const CanvasState) }
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_state = pool
        .allocate(&ctx.device, std::mem::size_of::<CanvasState>())
        .ok_or(Error::Gpu("alloc"))?;
    let state = blank_state(f);
    BufferPool::write_bytes(&buf_state, &state_to_bytes(&state));

    let canvas = f.canvas_size as u32;
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 0);
    }
    gpu_common::set_bytes(&enc, &f.params, 1);
    gpu_common::set_bytes(&enc, &canvas, 2);
    gpu_common::set_bytes(&enc, &f.pad_token, 3);
    gpu_common::set_bytes(&enc, &f.filler_token, 4);
    let eos = f.params.eos_token_id;
    gpu_common::set_bytes(&enc, &eos, 5);
    let early_stop_mean_ent: f32 = 0.0; // entropy-only stop disabled in unit tests
    gpu_common::set_bytes(&enc, &early_stop_mean_ent, 7);
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

    let mut out_buf = vec![0u8; std::mem::size_of::<CanvasState>()];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf_state.contents().as_ptr() as *const u8,
            out_buf.as_mut_ptr(),
            out_buf.len(),
        );
    }
    let state_out = bytes_to_state(&out_buf);
    Ok(pack_out(&state_out, f.canvas_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::sample_commit::cpu,
        cpu_oracle = crate::shaders::sample_commit::cpu_oracle,
        gpu = crate::shaders::sample_commit::gpu,
        fixture = crate::shaders::sample_commit::tiny_fixture,
        out_len = crate::shaders::sample_commit::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod final_step,
        cpu = crate::shaders::sample_commit::cpu,
        cpu_oracle = crate::shaders::sample_commit::cpu_oracle,
        gpu = crate::shaders::sample_commit::gpu,
        fixture = crate::shaders::sample_commit::final_step_fixture,
        out_len = crate::shaders::sample_commit::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "sample_commit",
    entry: "sample_commit",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};
