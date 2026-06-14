//! Tempered row stats + entropy + argmax for monolithic sampler pass 1.

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::sampler::{temp_at, tempered_row_stats};
use crate::metal::{CanvasState, StepParams, CANVAS};
use crate::safetensors::Error;

pub const ENTRY: &str = "sample_rowstats";
pub const THREADGROUP_WIDTH: usize = 256;

const SHADER: &str = shader_include::include_metal!("kernels/sample_rowstats.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub step: u32,
    pub params: StepParams,
    pub prev_argmax: Vec<u32>,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows * 4 + 1
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
            .map(|i| ((i as f32 * 0.07).sin() * 3.0))
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
        },
        prev_argmax: vec![u32::MAX; rows],
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let rows = 8usize;
    let cols = 512usize;
    Fixture {
        logits: (0..rows * cols)
            .map(|i| (i as f32 % 23.0) - 11.0)
            .collect(),
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
        },
        prev_argmax: (0..rows).map(|i| i as u32 * 3).collect(),
    }
}

fn pack_out(
    rowstat: &[f32],
    entropy: &[f32],
    prev_argmax: &[u32],
    argmax_changed: u32,
) -> Vec<f32> {
    let rows = entropy.len();
    let mut out = Vec::with_capacity(rows * 4 + 1);
    for row in 0..rows {
        out.push(rowstat[row * 2]);
        out.push(rowstat[row * 2 + 1]);
        out.push(entropy[row]);
        out.push(prev_argmax[row] as f32);
    }
    out.push(argmax_changed as f32);
    out
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let logits = f16::f16_slice_to_f32(&f16::f32_slice_to_f16(&f.logits));
    let t = f.temperature();
    let mut rowstat = vec![0.0f32; f.rows * 2];
    let mut entropy = vec![0.0f32; f.rows];
    let mut prev = f.prev_argmax.clone();
    let mut changed = 0u32;
    for row in 0..f.rows {
        let lr = &logits[row * f.cols..(row + 1) * f.cols];
        let st = tempered_row_stats(lr, t);
        rowstat[row * 2] = st.mx;
        rowstat[row * 2 + 1] = st.sum;
        entropy[row] = st.entropy;
        if prev[row] != st.argmax {
            changed = 1;
        }
        prev[row] = st.argmax;
    }
    pack_out(&rowstat, &entropy, &prev, changed)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

fn canvas_state_for_gpu(f: &Fixture) -> CanvasState {
    let mut prev = [u32::MAX; CANVAS];
    for (i, &v) in f.prev_argmax.iter().enumerate() {
        prev[i] = v;
    }
    CanvasState {
        ids: [0; CANVAS],
        prev_argmax: prev,
        new_sample: [0; CANVAS],
        entropy: [0.0; CANVAS],
        sorted_idx: [0; CANVAS],
        accept: [0; CANVAS],
        u_cat: [0.0; CANVAS],
        rng_state: 0,
        step: f.step,
        stop_flag: 0,
        argmax_stable: 0,
        argmax_changed: 0,
        mean_entropy: 0.0,
        _pad2: 0,
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

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_logits = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.rows * 2 * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_state = pool
        .allocate(&ctx.device, std::mem::size_of::<CanvasState>())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_logits, &f16::f32_slice_to_f16(&f.logits));
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
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 2);
    }
    gpu_common::set_bytes(&enc, &f.params, 3);
    gpu_common::set_bytes(&enc, &cols, 4);
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
    let mut changed = [0u32; 1];
    BufferPool::read_u32_at_offset(
        &buf_state,
        std::mem::offset_of!(CanvasState, argmax_changed),
        &mut changed,
    );
    Ok(pack_out(
        &rowstat,
        &entropy,
        &prev,
        changed[0],
    ))
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
        cpu = crate::kernels::sub::sample_rowstats::cpu,
        cpu_oracle = crate::kernels::sub::sample_rowstats::cpu_oracle,
        gpu = crate::kernels::sub::sample_rowstats::gpu,
        fixture = crate::kernels::sub::sample_rowstats::tiny_fixture,
        out_len = crate::kernels::sub::sample_rowstats::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::sample_rowstats::cpu,
        cpu_oracle = crate::kernels::sub::sample_rowstats::cpu_oracle,
        gpu = crate::kernels::sub::sample_rowstats::gpu,
        fixture = crate::kernels::sub::sample_rowstats::wide_fixture,
        out_len = crate::kernels::sub::sample_rowstats::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }
}
