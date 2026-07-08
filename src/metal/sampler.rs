//! GPU entropy-bound sampler: logits stay on device; read back scalars + canvas only.

use crate::metal::batch::{GpuBatch, begin_engine_batch, set_bytes};
use crate::metal::buffer::BufferPool;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::sampler_kernels::GpuSamplerKernels;
use crate::safetensors::Error;
use crate::sample::{
    Rng, SamplerConfig, StableConfidentStopper, accept_canvas_from_entropies,
    denoise_steps_completed, renoise_canvas,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice};

/// Persistent GPU logits buffer for one canvas forward (survives batch `end()`).
pub struct GpuLogitsBuf {
    buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    cap: usize,
}

impl GpuLogitsBuf {
    pub fn new() -> Self {
        Self { buf: None, cap: 0 }
    }

    pub fn ensure(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        len: usize,
    ) -> Result<&ProtocolObject<dyn MTLBuffer>, Error> {
        let bytes = len * 4;
        if self.cap < len || self.buf.is_none() {
            self.buf = Some(
                pool.allocate(device, bytes)
                    .ok_or(Error::Format("Metal logits buffer alloc failed"))?,
            );
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
    pub entropies: Vec<f32>,
    pub argmax: Vec<u32>,
    pub denoiser: Vec<u32>,
    pub sample_rand: Vec<f32>,
}

impl GpuSamplerScratch {
    pub fn new(canvas_len: usize, _vocab: usize) -> Self {
        Self {
            logits: GpuLogitsBuf::new(),
            entropies: vec![0.0; canvas_len],
            argmax: vec![0; canvas_len],
            denoiser: vec![0; canvas_len],
            sample_rand: vec![0.0; canvas_len],
        }
    }

    pub fn ensure_len(&mut self, canvas_len: usize, _vocab: usize) {
        if self.entropies.len() != canvas_len {
            self.entropies.resize(canvas_len, 0.0);
            self.argmax.resize(canvas_len, 0);
            self.denoiser.resize(canvas_len, 0);
            self.sample_rand.resize(canvas_len, 0.0);
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
    batch.dispatch_1d_ranged(&sk.logit_softcapping.pipeline, len, |enc, base, chunk| {
        let range = [base, chunk];
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        set_bytes(enc, &cap, 1);
        set_bytes(enc, &range, 2);
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
    batch.dispatch_1d_ranged(&sk.scale_logits.pipeline, len, |enc, base, chunk| {
        let range = [base, chunk];
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        set_bytes(enc, &inv_t, 1);
        set_bytes(enc, &range, 2);
    });
}

fn dispatch_copy_f32(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    batch.dispatch_1d_ranged(&sk.copy_f32.pipeline, len, |enc, base, chunk| {
        let range = [base, chunk];
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
        }
        set_bytes(enc, &range, 2);
    });
}

fn row_grid(canvas_len: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    let tg = objc2_metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = objc2_metal::MTLSize {
        width: 1,
        height: canvas_len,
        depth: 1,
    };
    (grid, tg)
}

fn dispatch_row_entropy(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    logits: &ProtocolObject<dyn MTLBuffer>,
    entropies_buf: &ProtocolObject<dyn MTLBuffer>,
    canvas_len: usize,
    vocab: usize,
) {
    let dims = [canvas_len as u32, vocab as u32];
    let (grid, tg) = row_grid(canvas_len);
    batch.dispatch_with_grid(&sk.row_entropy.pipeline, grid, tg, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
            enc.setBuffer_offset_atIndex(Some(entropies_buf), 0, 1);
        }
        set_bytes(enc, &dims, 2);
    });
}

fn dispatch_argmax_rows(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    logits: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    canvas_len: usize,
    vocab: usize,
) {
    let dims = [canvas_len as u32, vocab as u32];
    let (grid, tg) = row_grid(canvas_len);
    batch.dispatch_with_grid(&sk.argmax_rows.pipeline, grid, tg, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
        }
        set_bytes(enc, &dims, 2);
    });
}

fn dispatch_softmax_rows(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    probs: &ProtocolObject<dyn MTLBuffer>,
    canvas_len: usize,
    vocab: usize,
) -> Result<(), Error> {
    let dims = [canvas_len as u32, vocab as u32];
    let (grid, tg) = row_grid(canvas_len);
    let buf_dump = batch.alloc_f32_out(4)?;
    batch.dispatch_with_grid(&sk.softmax_rows.pipeline, grid, tg, |enc| {
        crate::kernels::sub::softmax_rows::bind_gpu_in_place(enc, probs, &buf_dump, &dims);
    });
    Ok(())
}

fn dispatch_sample_from_probs(
    batch: &mut GpuBatch<'_>,
    sk: &GpuSamplerKernels,
    probs: &ProtocolObject<dyn MTLBuffer>,
    rand: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    canvas_len: usize,
    vocab: usize,
) {
    let dims = [canvas_len as u32, vocab as u32];
    let grid = objc2_metal::MTLSize {
        width: 1,
        height: canvas_len,
        depth: 1,
    };
    let tg = objc2_metal::MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    batch.dispatch_with_grid(&sk.sample_from_probs_rows.pipeline, grid, tg, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(probs), 0, 0);
            enc.setBuffer_offset_atIndex(Some(rand), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
        }
        set_bytes(enc, &dims, 3);
    });
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::metal::batch::GpuBatch;
    use crate::metal::device::MetalContext;
    use crate::metal::sampler_kernels::GpuSamplerKernels;
    use crate::sample::{Rng, argmax_canvas, sample_canvas, token_entropy};

    #[test]
    fn gpu_postprocess_matches_cpu() {
        let canvas_len = 8usize;
        let vocab = 512usize;
        let total = canvas_len * vocab;
        let mut logits = vec![0.0f32; total];
        for pos in 0..canvas_len {
            logits[pos * vocab + (pos * 17 % vocab)] = 2.0;
            logits[pos * vocab + (pos * 31 % vocab)] = 1.0;
        }

        let mut cpu = logits.clone();
        crate::model::embed::logit_softcapping(&mut cpu, 30.0);
        let temperature = 0.65f32;
        for v in cpu.iter_mut() {
            *v /= temperature;
        }
        let cpu_ent = token_entropy(&cpu, canvas_len, vocab);
        let cpu_argmax = argmax_canvas(&cpu, canvas_len, vocab);
        let rand_vals: Vec<f32> = {
            let mut rng = Rng::new(99);
            (0..canvas_len).map(|_| rng.next_f32()).collect()
        };
        let cpu_denoiser = {
            let mut probs = cpu.clone();
            sample_canvas(&mut probs, canvas_len, vocab, &mut Rng::new(99))
        };

        let ctx = MetalContext::new().expect("metal");
        let sk = GpuSamplerKernels::new(&ctx).expect("kernels");
        let mut pool = BufferPool::new();
        let mut logits_buf = GpuLogitsBuf::new();
        logits_buf
            .ensure(&ctx.device, &mut pool, total)
            .expect("ensure");
        BufferPool::write_f32(logits_buf.as_buf().expect("buf"), &logits);

        let mut scratch = GpuSamplerScratch::new(canvas_len, vocab);
        scratch.sample_rand.copy_from_slice(&rand_vals);

        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let logits_ref = logits_buf.as_buf().expect("buf");
        dispatch_logit_softcapping(&mut batch, &sk, logits_ref, total, 30.0);
        dispatch_scale_logits(&mut batch, &sk, logits_ref, total, temperature);
        let buf_ent = batch.alloc_f32_out(canvas_len).expect("ent");
        dispatch_row_entropy(&mut batch, &sk, logits_ref, &buf_ent, canvas_len, vocab);
        let buf_argmax = batch.alloc_u32_out(canvas_len).expect("argmax");
        dispatch_argmax_rows(&mut batch, &sk, logits_ref, &buf_argmax, canvas_len, vocab);
        let buf_probs = batch.alloc_f32_out(total).expect("probs");
        dispatch_copy_f32(&mut batch, &sk, logits_ref, &buf_probs, total);
        dispatch_softmax_rows(&mut batch, &sk, &buf_probs, canvas_len, vocab).expect("softmax");
        let buf_rand = batch.alloc_f32(&scratch.sample_rand).expect("rand");
        let buf_denoiser = batch.alloc_u32_out(canvas_len).expect("denoiser");
        dispatch_sample_from_probs(
            &mut batch,
            &sk,
            &buf_probs,
            &buf_rand,
            &buf_denoiser,
            canvas_len,
            vocab,
        );
        batch.register_read(buf_ent, &mut scratch.entropies);
        batch.register_read_u32(buf_argmax, &mut scratch.argmax);
        batch.register_read_u32(buf_denoiser, &mut scratch.denoiser);
        batch.end().expect("end");

        for i in 0..canvas_len {
            assert!(
                (scratch.entropies[i] - cpu_ent[i]).abs() < 1e-4,
                "entropy[{i}] gpu={} cpu={}",
                scratch.entropies[i],
                cpu_ent[i]
            );
            assert_eq!(scratch.argmax[i], cpu_argmax[i], "argmax[{i}]");
            assert_eq!(scratch.denoiser[i], cpu_denoiser[i], "denoiser[{i}]");
        }
    }
}
