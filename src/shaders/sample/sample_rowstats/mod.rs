//! Tempered row stats + entropy + argmax for monolithic sampler pass 1.

use crate::Error;
use crate::metal::{CANVAS, CanvasState, StepParams};
use crate::sample::PAD_TOKEN_ID;
use crate::shaders::bf16;
use crate::shaders::cpu::sampler::{temp_at, tempered_row_stats};
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "sample_rowstats";
pub const THREADGROUP_WIDTH: usize = 256;

pub const SHADER: &str = include_str!("sample_rowstats.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub step: u32,
    pub params: StepParams,
    pub prev_argmax: Vec<u32>,
    pub ids: Vec<u32>,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows * 4
    }

    pub fn temperature(&self) -> f32 {
        temp_at(
            self.step,
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
    Fixture {
        logits: (0..rows * cols)
            .map(|i| (i as f32 * 0.07).sin() * 3.0)
            .collect(),
        rows,
        cols,
        step: 0,
        params: StepParams {
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
        },
        prev_argmax: vec![u32::MAX; rows],
        ids: (0..rows).map(|i| 100 + i as u32).collect(),
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let rows = 8usize;
    let cols = 512usize;
    Fixture {
        logits: (0..rows * cols).map(|i| (i as f32 % 23.0) - 11.0).collect(),
        rows,
        cols,
        step: 2,
        params: StepParams {
            kv_len: 128,
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
        prev_argmax: (0..rows).map(|i| i as u32 * 3).collect(),
        ids: (0..rows).map(|i| 200 + i as u32).collect(),
    }
}

pub fn pad_tail_fixture(_: ElemFormat) -> Fixture {
    let mut f = tiny_fixture(ElemFormat::F32);
    f.ids = vec![PAD_TOKEN_ID, PAD_TOKEN_ID, 100, 101];
    f.prev_argmax = vec![999, 888, u32::MAX, u32::MAX];
    f
}

fn pack_out(rowstat: &[f32], entropy: &[f32], prev_argmax: &[u32]) -> Vec<f32> {
    let rows = entropy.len();
    let mut out = Vec::with_capacity(rows * 4);
    for row in 0..rows {
        out.push(rowstat[row * 2]);
        out.push(rowstat[row * 2 + 1]);
        out.push(entropy[row]);
        out.push(prev_argmax[row] as f32);
    }
    out
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let logits = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.logits));
    let t = f.temperature();
    let mut rowstat = vec![0.0f32; f.rows * 2];
    let mut entropy = vec![0.0f32; f.rows];
    let mut prev = f.prev_argmax.clone();
    for row in 0..f.rows {
        let lr = &logits[row * f.cols..(row + 1) * f.cols];
        let st = tempered_row_stats(lr, t);
        rowstat[row * 2] = st.mx;
        rowstat[row * 2 + 1] = st.sum;
        entropy[row] = st.entropy;
        prev[row] = st.argmax;
    }
    pack_out(&rowstat, &entropy, &prev)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

fn canvas_state_for_gpu(f: &Fixture) -> CanvasState {
    let mut prev = [u32::MAX; CANVAS];
    let mut ids = [0u32; crate::metal::PREFILL_M];
    for (i, &v) in f.prev_argmax.iter().enumerate() {
        prev[i] = v;
    }
    for (i, &v) in f.ids.iter().enumerate() {
        ids[i] = v;
    }
    CanvasState {
        ids,
        prev_argmax: prev,
        new_sample: [0; CANVAS],
        entropy: [0.0; CANVAS],
        sorted_idx: [0; CANVAS],
        accept: [0; CANVAS],
        u_cat: [0.0; CANVAS],
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
pub fn dispatch_shape(rows: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    )
}

#[cfg(target_os = "macos")]
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_logits = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.rows * 2 * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_state = pool
        .allocate(&ctx.device, std::mem::size_of::<CanvasState>())
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_logits, &bf16::f32_slice_to_bf16_bits(&f.logits));
    let state = canvas_state_for_gpu(f);
    let state_bytes = unsafe {
        std::slice::from_raw_parts(
            &state as *const CanvasState as *const u8,
            std::mem::size_of::<CanvasState>(),
        )
    };
    BufferPool::write_bytes(&buf_state, state_bytes);

    let cols = f.cols as u32;
    let (grid, tg) = dispatch_shape(f.rows);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 2);
    }
    gpu_common::set_bytes(&enc, &f.params, 3);
    gpu_common::set_bytes(&enc, &cols, 4);
    let pad = crate::sample::PAD_TOKEN_ID;
    let filler = crate::sample::FILLER_TOKEN_ID;
    gpu_common::set_bytes(&enc, &pad, 5);
    gpu_common::set_bytes(&enc, &filler, 6);
    let eos = f.params.eos_token_id;
    gpu_common::set_bytes(&enc, &eos, 7);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut rowstat = vec![0.0f32; f.rows * 2];
    BufferPool::read_f32(&buf_out, &mut rowstat);
    let mut entropy = vec![0.0f32; f.rows];
    BufferPool::read_f32_at_offset(
        &buf_state,
        std::mem::offset_of!(CanvasState, entropy),
        &mut entropy,
    );
    let mut prev = vec![0u32; f.rows];
    BufferPool::read_u32_at_offset(
        &buf_state,
        std::mem::offset_of!(CanvasState, prev_argmax),
        &mut prev,
    );
    Ok(pack_out(&rowstat, &entropy, &prev))
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "sample_rowstats",
    entry: "sample_rowstats",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::sample_rowstats::cpu,
        cpu_oracle = crate::shaders::sample_rowstats::cpu_oracle,
        gpu = crate::shaders::sample_rowstats::gpu,
        fixture = crate::shaders::sample_rowstats::tiny_fixture,
        out_len = crate::shaders::sample_rowstats::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::shaders::sample_rowstats::cpu,
        cpu_oracle = crate::shaders::sample_rowstats::cpu_oracle,
        gpu = crate::shaders::sample_rowstats::gpu,
        fixture = crate::shaders::sample_rowstats::wide_fixture,
        out_len = crate::shaders::sample_rowstats::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod pad_tail,
        cpu = crate::shaders::sample_rowstats::cpu,
        cpu_oracle = crate::shaders::sample_rowstats::cpu_oracle,
        gpu = crate::shaders::sample_rowstats::gpu,
        fixture = crate::shaders::sample_rowstats::pad_tail_fixture,
        out_len = crate::shaders::sample_rowstats::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}

crate::register_kernel_specs!(SPEC);
