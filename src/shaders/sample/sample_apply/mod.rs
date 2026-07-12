//! Sampler pass 3: categorical inverse-CDF per row from tempered logits.

use crate::metal::{CANVAS, CanvasState, StepParams};
use crate::safetensors::Error;
use crate::shaders::bf16;
use crate::shaders::cpu::sampler::{
    TemperedRowStats, temp_at, tempered_row_stats, tempered_sample_row,
};
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "sample_apply";
pub const THREADGROUP_WIDTH: usize = 256;

pub const SHADER: &str = include_str!("sample_apply.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rowstat: Vec<f32>,
    pub u_cat: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub step: u32,
    pub params: StepParams,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows
    }

    pub fn temperature(&self) -> f32 {
        temp_at(
            self.step.saturating_sub(1),
            self.params.max_steps,
            self.params.t_min,
            self.params.t_max,
        )
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let rows = 4usize;
    let cols = 256usize;
    let logits: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.05).cos() * 2.0)
        .collect();
    let logits = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&logits));
    let step = 2u32;
    let params = StepParams {
        kv_len: 0,
        max_steps: 8,
        entropy_bound: 2.0,
        t_min: 0.3,
        t_max: 1.0,
        conf_threshold: 0.5,
        stability_threshold: 2,
        min_early_stop_steps: 12,
        accept_plateau_threshold: 2,
        plateau_prefix_mean_max: 0.05,
        eos_token_id: 1,
        kv_write_end: u32::MAX,
    };
    let t = temp_at(
        step.saturating_sub(1),
        params.max_steps,
        params.t_min,
        params.t_max,
    );
    let mut rowstat = vec![0.0f32; rows * 2];
    for row in 0..rows {
        let lr = &logits[row * cols..(row + 1) * cols];
        let st: TemperedRowStats = tempered_row_stats(lr, t);
        rowstat[row * 2] = st.mx;
        rowstat[row * 2 + 1] = st.sum;
    }
    Fixture {
        logits,
        rowstat,
        u_cat: vec![0.12, 0.34, 0.56, 0.78],
        rows,
        cols,
        step,
        params,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let rows = 8usize;
    let cols = 512usize;
    let logits: Vec<f32> = (0..rows * cols).map(|i| (i as f32 % 19.0) - 9.0).collect();
    let logits = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&logits));
    let t = temp_at(1, 8, 0.25, 0.95);
    let mut rowstat = vec![0.0f32; rows * 2];
    for row in 0..rows {
        let lr = &logits[row * cols..(row + 1) * cols];
        let st = tempered_row_stats(lr, t);
        rowstat[row * 2] = st.mx;
        rowstat[row * 2 + 1] = st.sum;
    }
    Fixture {
        logits,
        rowstat,
        u_cat: (0..rows).map(|i| (i as f32 + 1.0) * 0.07).collect(),
        rows,
        cols,
        step: 2,
        params: StepParams {
            kv_len: 64,
            max_steps: 8,
            entropy_bound: 1.5,
            t_min: 0.25,
            t_max: 0.95,
            conf_threshold: 0.4,
            stability_threshold: 3,
            min_early_stop_steps: 12,
            accept_plateau_threshold: 2,
            plateau_prefix_mean_max: 0.05,
            eos_token_id: 1,
            kv_write_end: u32::MAX,
        },
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let logits = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.logits));
    let t = f.temperature();
    let mut out = vec![0.0f32; f.rows];
    for row in 0..f.rows {
        let lr = &logits[row * f.cols..(row + 1) * f.cols];
        let mx = f.rowstat[row * 2];
        let z = f.rowstat[row * 2 + 1];
        let pick = tempered_sample_row(lr, mx, z, f.u_cat[row], t);
        out[row] = pick as f32;
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

fn canvas_state_for_gpu(f: &Fixture) -> CanvasState {
    let mut u_cat = [0.0f32; CANVAS];
    for (i, &u) in f.u_cat.iter().enumerate() {
        u_cat[i] = u;
    }
    CanvasState {
        ids: [0; crate::metal::PREFILL_M],
        prev_argmax: [0; CANVAS],
        new_sample: [0; CANVAS],
        entropy: [0.0; CANVAS],
        sorted_idx: [0; CANVAS],
        accept: [0; CANVAS],
        u_cat,
        rng_state: 0,
        step: f.step,
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

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_logits = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_rowstat = pool
        .allocate(&ctx.device, f.rowstat.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_state = pool
        .allocate(&ctx.device, std::mem::size_of::<CanvasState>())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_logits, &bf16::f32_slice_to_bf16_bits(&f.logits));
    BufferPool::write_f32(&buf_rowstat, &f.rowstat);
    let state = canvas_state_for_gpu(f);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &state as *const CanvasState as *const u8,
            std::mem::size_of::<CanvasState>(),
        )
    };
    BufferPool::write_bytes(&buf_state, bytes);

    let cols = f.cols as u32;
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_rowstat), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 2);
    }
    gpu_common::set_bytes(&enc, &f.params, 3);
    gpu_common::set_bytes(&enc, &cols, 4);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.rows,
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

    let mut picks = vec![0u32; f.rows];
    BufferPool::read_u32_at_offset(
        &buf_state,
        std::mem::offset_of!(CanvasState, new_sample),
        &mut picks,
    );
    Ok(picks.iter().map(|&v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::sample_apply::cpu,
        cpu_oracle = crate::shaders::sample_apply::cpu_oracle,
        gpu = crate::shaders::sample_apply::gpu,
        fixture = crate::shaders::sample_apply::tiny_fixture,
        out_len = crate::shaders::sample_apply::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::shaders::sample_apply::cpu,
        cpu_oracle = crate::shaders::sample_apply::cpu_oracle,
        gpu = crate::shaders::sample_apply::gpu,
        fixture = crate::shaders::sample_apply::wide_fixture,
        out_len = crate::shaders::sample_apply::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "sample_apply",
    entry: "sample_apply",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};
