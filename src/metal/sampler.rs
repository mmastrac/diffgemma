//! GPU entropy-bound sampler: lm_head logits stay on device through softcap + temperature.
//! Categorical sample / accept / early-stop still run on CPU after one logits readback (parity).

use crate::metal::batch::{begin_engine_batch, set_bytes, GpuBatch};
use crate::metal::buffer::BufferPool;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::sampler_kernels::GpuSamplerKernels;
use crate::sample::{
    accept_canvas, argmax_canvas, renoise_canvas, sample_canvas, Rng, SamplerConfig,
    StableConfidentStopper,
};
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

/// Persistent GPU logits buffer for one canvas forward (avoids per-chunk lm_head readback).
pub struct GpuLogitsBuf {
    buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    cap: usize,
}

impl GpuLogitsBuf {
    pub fn new() -> Self {
        Self {
            buf: None,
            cap: 0,
        }
    }

    pub fn ensure<'a>(
        &mut self,
        batch: &mut GpuBatch<'a>,
        len: usize,
    ) -> Result<&ProtocolObject<dyn MTLBuffer>, Error> {
        if self.cap < len || self.buf.is_none() {
            self.buf = Some(batch.alloc_f32_out(len)?);
            self.cap = len;
        }
        Ok(self.buf.as_ref().expect("logits buf"))
    }

    pub fn as_buf(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.buf.as_ref().map(|b| b.as_ref())
    }
}

pub struct GpuSamplerScratch {
    pub logits: GpuLogitsBuf,
    pub processed: Vec<f32>,
    pub sample_probs: Vec<f32>,
}

impl GpuSamplerScratch {
    pub fn new(canvas_len: usize, vocab: usize) -> Self {
        Self {
            logits: GpuLogitsBuf::new(),
            processed: vec![0.0; canvas_len * vocab.max(1)],
            sample_probs: vec![0.0; canvas_len * vocab.max(1)],
        }
    }

    pub fn ensure_len(&mut self, canvas_len: usize, vocab: usize) {
        let n = canvas_len * vocab;
        if self.processed.len() != n {
            self.processed.resize(n, 0.0);
            self.sample_probs.resize(n, 0.0);
        }
    }
}

pub struct GpuSamplerStepOut {
    pub argmax: Vec<u32>,
    pub finished: bool,
    pub readback_bytes: u64,
}

fn dispatch_logit_softcapping(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    cap: f32,
) {
    let len_u = len as u32;
    batch.dispatch_1d(&sk.logit_softcapping.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        set_bytes(enc, &cap, 1);
        set_bytes(enc, &len_u, 2);
    });
}

fn dispatch_scale_logits(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    temperature: f32,
) {
    let inv_t = 1.0 / temperature.max(1e-6);
    let len_u = len as u32;
    batch.dispatch_1d(&sk.scale_logits.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        set_bytes(enc, &inv_t, 1);
        set_bytes(enc, &len_u, 2);
    });
}

/// GPU softcap + temperature, then CPU sampler (bit-exact vs oracle). One logits readback/step.
pub fn sampler_step_gpu(
    engine: &mut GpuDecoderEngine,
    scratch: &mut GpuSamplerScratch,
    current_canvas: &mut [u32],
    cur_step: usize,
    canvas_len: usize,
    vocab: usize,
    logit_cap: f32,
    sampler: &SamplerConfig,
    stopper: &mut StableConfidentStopper,
    rng: &mut Rng,
) -> Result<GpuSamplerStepOut, Error> {
    scratch.ensure_len(canvas_len, vocab);
    let logits_buf = scratch
        .logits
        .as_buf()
        .ok_or(Error::Format("gpu sampler: logits buffer not set"))?;
    let total = canvas_len * vocab;
    let temperature = sampler.temperature_at_step(cur_step);

    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;

    dispatch_logit_softcapping(
        &mut batch,
        &engine.sampler_kernels,
        logits_buf,
        total,
        logit_cap,
    );
    dispatch_scale_logits(
        &mut batch,
        &engine.sampler_kernels,
        logits_buf,
        total,
        temperature,
    );
    batch.register_read(
        scratch.logits.buf.as_ref().expect("logits").clone(),
        &mut scratch.processed,
    );
    batch.end()?;

    let readback_bytes = total as u64 * 4;

    scratch.sample_probs.copy_from_slice(&scratch.processed);
    let denoiser = sample_canvas(&mut scratch.sample_probs, canvas_len, vocab, rng);
    let argmax = argmax_canvas(&scratch.processed, canvas_len, vocab);
    let (accepted, accepted_mask) = accept_canvas(
        current_canvas,
        &denoiser,
        &scratch.processed,
        canvas_len,
        vocab,
        sampler.entropy_bound,
    );
    let renoised = renoise_canvas(&accepted, &accepted_mask, vocab, rng);
    current_canvas.copy_from_slice(&renoised);

    let finished = stopper.should_stop(&argmax, &scratch.processed, canvas_len, vocab);

    Ok(GpuSamplerStepOut {
        argmax,
        finished,
        readback_bytes,
    })
}

/// Upload processed logits back to the persistent GPU buffer (self-conditioning path).
pub fn upload_logits_gpu(
    scratch: &mut GpuSamplerScratch,
    _engine: &mut GpuDecoderEngine,
    logits: &[f32],
) -> Result<(), Error> {
    let Some(buf) = scratch.logits.as_buf() else {
        return Err(Error::Format("gpu logits buffer missing"));
    };
    BufferPool::write_f32(buf, logits);
    Ok(())
}
