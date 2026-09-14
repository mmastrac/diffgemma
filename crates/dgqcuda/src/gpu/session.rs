//! A resident denoise session: the model, its buffers, and the self-conditioning
//! weights stay on the device across the whole denoise loop, so one step is
//! only the 30 decoder layers plus the LM head it needs.
//!
//! The sequence layout of a step is [prompt tokens][canvas tokens]: prompt rows
//! attend causally, canvas rows attend the whole sequence (the diffusion canvas
//! is bidirectional). Positions are absolute over that layout, so the canvas
//! occupies [prompt_len, prompt_len + canvas).

use crate::config::{Error, ModelConfig};
use crate::gpu::cuda::{
    Bufs, GpuModel, Runner, Stage, emit_layer_checkpoint, layer_forward, up_f32,
};
use crate::weights::Weights;
use gpukit::cuda::{Context, DeviceBuffer, KernelArgs, cached_source_kernel};

fn rows(n: usize) -> (u32, u32, u32) {
    (n as u32, 1, 1)
}

fn flat(n: usize, block: u32) -> (u32, u32, u32) {
    (n.div_ceil(block as usize) as u32, 1, 1)
}

fn launch(
    ctx: &Context,
    entry: &'static str,
    grid: (u32, u32, u32),
    block: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    let kernel = cached_source_kernel(crate::gpu::cuda::KERNELS, entry)?;
    Ok(ctx.launch(&kernel, grid, (block, 1, 1), 0, args)?)
}

/// The self-conditioning MLP weights.
struct ScWeights {
    pre_norm: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    down: DeviceBuffer,
}

/// Resident model + canvas buffers for one denoise loop.
pub struct Session {
    model: GpuModel,
    /// Layers the step runs (all of them unless a bisect overrides it).
    layers: usize,
    cfg: ModelConfig,
    /// `prompt_len + canvas` the buffers are sized for.
    seq: usize,
    prompt_len: usize,
    canvas: usize,
    bufs: Bufs,
    sc: ScWeights,
    /// Canvas logits of the previous step (pre-softcap), the self-conditioning
    /// signal for the next one.
    prev_logits: Option<DeviceBuffer>,
    /// Reusable host staging for the logits readback.
    host_logits: Vec<f32>,
    /// The first step has no previous prediction: it seeds self-conditioning
    /// with the canvas embeddings instead of the soft embedding.
    first_step: bool,
}

impl Session {
    pub fn open(
        w: &Weights,
        cfg: &ModelConfig,
        prompt_len: usize,
        canvas: usize,
    ) -> Result<Self, Error> {
        let seq = prompt_len + canvas;
        let model = GpuModel::load(w, cfg, None)?;
        let ctx = model.ctx.clone();
        let t = &cfg.text_config;
        let bufs = Bufs::new(seq, cfg, &ctx)?;
        let sc = ScWeights {
            pre_norm: up_f32(
                &ctx,
                &w.tensor_f32("model.decoder.self_conditioning.pre_norm.weight")?,
            )?,
            gate: up_f32(
                &ctx,
                &w.tensor_f32("model.decoder.self_conditioning.gate_proj.weight")?,
            )?,
            up: up_f32(
                &ctx,
                &w.tensor_f32("model.decoder.self_conditioning.up_proj.weight")?,
            )?,
            down: up_f32(
                &ctx,
                &w.tensor_f32("model.decoder.self_conditioning.down_proj.weight")?,
            )?,
        };
        let prev_logits = DeviceBuffer::alloc(&ctx, canvas * t.vocab_size * 4)?;
        Ok(Self {
            model,
            layers: cfg.text_config.num_hidden_layers,
            cfg: cfg.clone(),
            seq,
            prompt_len,
            canvas,
            bufs,
            sc,
            prev_logits: Some(prev_logits),
            host_logits: vec![0.0f32; canvas * t.vocab_size],
            first_step: true,
        })
    }

    pub fn canvas(&self) -> usize {
        self.canvas
    }

    /// Bisect helper: run only the first `n` layers of each step.
    pub fn set_layers(&mut self, n: usize) {
        self.layers = n;
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    fn runner(&self, causal_split: usize) -> Runner<'_> {
        Runner {
            m: &self.model,
            cfg: &self.cfg,
            seq: self.seq,
            pos0: 0,
            causal_split,
            stage: std::cell::RefCell::new(Stage::new()),
        }
    }

    /// Gather `ids` into `hidden_a` starting at row `row0`.
    fn embed_rows(&self, ids: &[u32], row0: usize) -> Result<(), Error> {
        let ctx = &self.model.ctx;
        let hidden = self.cfg.text_config.hidden_size;
        let scale = (hidden as f32).sqrt();
        self.bufs.ids.write_bytes(unsafe {
            std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4)
        })?;
        let k = cached_source_kernel(dgops::ops::embed_gather::CUDA, "embed_gather")?;
        for s in 0..ids.len() {
            let mut args = KernelArgs::new();
            args.device_ptr(self.model.embed.device_ptr())
                .device_ptr(unsafe { self.bufs.ids.device_ptr() + (s as u64) * 4 })
                .device_ptr(unsafe {
                    self.bufs.hidden_a.device_ptr() + ((row0 + s) as u64) * (hidden as u64) * 4
                })
                .u32(hidden as u32)
                .u32(1)
                .u32(0)
                .u32(0)
                .f32(scale)
                .u32(self.cfg.text_config.vocab_size as u32)
                .u32(1);
            ctx.launch(&k, flat(hidden, 256), (256, 1, 1), 0, &mut args)?;
        }
        Ok(())
    }

    /// The self-conditioning MLP over the canvas rows, in place on `hidden_a`:
    /// `x = rms_norm_rows(signal, pre_norm)`, `x = down(up * gelu(gate(x)))`,
    /// `hidden += x`, then a scale-free RMS norm of the sum.
    ///
    /// The MLP runs on every step; only the signal changes. From step 2 on it
    /// is the soft embedding of the previous step's logits, in `norm_scratch`.
    /// Step 1 has no previous prediction, so the engine seeds it with the
    /// canvas embeddings themselves -- the initial canvas read as the step-0
    /// prediction (`interpret_step`'s `first_step == 1` branch, which norms
    /// `hidden_off` rather than `soft_off`). Skipping the MLP there is a zero
    /// signal, and that branch's comment records the model as degenerate under
    /// it: a cold-start empty reply.
    fn self_condition(&self, signal_first_step: bool) -> Result<(), Error> {
        let ctx = &self.model.ctx;
        let t = &self.cfg.text_config;
        let hidden = t.hidden_size;
        let inter = t.intermediate_size;
        let eps = t.rms_norm_eps as f32;
        let canvas = self.canvas;
        let r = self.runner(self.prompt_len);
        let base = self.prompt_len * hidden;

        // Step 1 norms the canvas embeddings already sitting in `hidden_a`;
        // every later step norms the soft embedding in `norm_scratch`.
        let src = if signal_first_step {
            unsafe { self.bufs.hidden_a.device_ptr() + (base as u64) * 4 }
        } else {
            self.bufs.norm_scratch.device_ptr()
        };
        let mut args = KernelArgs::new();
        args.device_ptr(src)
            .device_ptr(self.sc.pre_norm.device_ptr())
            .device_ptr(self.bufs.normed.device_ptr())
            .u32(canvas as u32)
            .u32(hidden as u32)
            .f32(eps);
        launch(ctx, "dgq_rms_norm", rows(canvas), 256, &mut args)?;
        if std::env::var("DGQCUDA_TIME").is_ok() {
            ctx.synchronize()?;
            eprintln!("  [sc] rms ok");
        }

        r.gemm(
            canvas,
            inter,
            hidden,
            self.bufs.normed.device_ptr(),
            self.sc.gate.device_ptr(),
            self.bufs.mlp_gate.device_ptr(),
        )?;
        r.gemm(
            canvas,
            inter,
            hidden,
            self.bufs.normed.device_ptr(),
            self.sc.up.device_ptr(),
            self.bufs.mlp_up.device_ptr(),
        )?;
        // gate/up -> gelu(gate) * up, in place in mlp_gate (kernel args are
        // gate, up, weight, out, len).
        let mut args = KernelArgs::new();
        args.device_ptr(self.bufs.mlp_gate.device_ptr())
            .device_ptr(self.bufs.mlp_up.device_ptr())
            .f32(1.0)
            .device_ptr(self.bufs.mlp_gate.device_ptr())
            .u32((canvas * inter) as u32);
        launch(
            ctx,
            "dgq_swiglu_weighted",
            flat(canvas * inter, 256),
            256,
            &mut args,
        )?;
        r.gemm(
            canvas,
            hidden,
            inter,
            self.bufs.mlp_gate.device_ptr(),
            self.sc.down.device_ptr(),
            self.bufs.mlp_down.device_ptr(),
        )?;
        if std::env::var("DGQCUDA_TIME").is_ok() {
            ctx.synchronize()?;
            eprintln!("  [sc] mlp ok");
        }

        // hidden[canvas] += signal, then rms_norm_no_scale in place.
        let dst = unsafe { self.bufs.hidden_a.device_ptr() + (base as u64) * 4 };
        let mut args = KernelArgs::new();
        args.device_ptr(dst)
            .device_ptr(self.bufs.mlp_down.device_ptr())
            .u32((canvas * hidden) as u32);
        launch(
            ctx,
            "dgq_vec_add",
            flat(canvas * hidden, 256),
            256,
            &mut args,
        )?;
        if std::env::var("DGQCUDA_TIME").is_ok() {
            ctx.synchronize()?;
            eprintln!("  [sc] add ok");
        }
        // Scale-free norm: a dedicated entry, because the weighted kernel with
        // a null weight pointer faults and with `pre_norm` would apply the
        // wrong weight.
        let mut args = KernelArgs::new();
        args.device_ptr(dst)
            .device_ptr(dst)
            .u32(canvas as u32)
            .u32(hidden as u32)
            .f32(eps);
        launch(ctx, "dgq_rms_norm_ns", rows(canvas), 256, &mut args)?;
        if std::env::var("DGQCUDA_TIME").is_ok() {
            ctx.synchronize()?;
            eprintln!("  [sc] residual rms ok");
        }
        Ok(())
    }

    /// Sparse soft embedding of `logits` (canvas rows) into `norm_scratch`.
    fn soft_embed(&self, logits: &DeviceBuffer) -> Result<(), Error> {
        let t = &self.cfg.text_config;
        let hidden = t.hidden_size;
        let scale = (hidden as f32).sqrt();
        let mut args = KernelArgs::new();
        args.device_ptr(logits.device_ptr())
            .device_ptr(self.model.embed.device_ptr())
            .device_ptr(self.bufs.norm_scratch.device_ptr())
            .u32(self.canvas as u32)
            .u32(t.vocab_size as u32)
            .u32(hidden as u32)
            .f32(-10.0)
            .f32(scale);
        launch(
            &self.model.ctx,
            "dgq_soft_embed",
            rows(self.canvas),
            256,
            &mut args,
        )
    }

    /// Warm the device: embed the prompt, then run the prompt rows through the
    /// layers once. This is not an oracle -- the step embeds both row groups
    /// into the same sequence, so it does not consume this result. What it buys
    /// is that module load, first-touch allocations and the prompt-width
    /// dispatch shapes are all paid before the first timed step.
    pub fn warm(&mut self, prompt: &[u32]) -> Result<(), Error> {
        assert_eq!(prompt.len(), self.prompt_len, "prompt length");
        let ctx = self.model.ctx.clone();
        self.embed_rows(prompt, 0)?;
        let n_layers = self.layers.min(self.model.layers.len());
        {
            let Self {
                model,
                cfg,
                prompt_len,
                bufs,
                ..
            } = self;
            let r = Runner {
                m: model,
                cfg,
                seq: *prompt_len,
                pos0: 0,
                causal_split: *prompt_len,
                stage: std::cell::RefCell::new(Stage::new()),
            };
            for (i, lw) in model.layers.iter().take(n_layers).enumerate() {
                layer_forward(&r, bufs, lw, i, 0)?;
                std::mem::swap(&mut bufs.hidden_a, &mut bufs.hidden_b);
            }
        }
        ctx.synchronize()?;
        Ok(())
    }

    /// The prompt's hidden state after the first `n` causal layers, for
    /// bisecting a layer against the CPU oracle. Runs the same graph as
    /// `prompt_hidden`; the caller pays one prompt pass per layer.
    pub fn prompt_hidden_after(&mut self, prompt: &[u32], n: usize) -> Result<Vec<f32>, Error> {
        assert_eq!(prompt.len(), self.prompt_len, "prompt length");
        let ctx = self.model.ctx.clone();
        let hidden = self.cfg.text_config.hidden_size;
        self.embed_rows(prompt, 0)?;
        {
            let Self {
                model,
                cfg,
                prompt_len,
                bufs,
                ..
            } = self;
            let r = Runner {
                m: model,
                cfg,
                seq: *prompt_len,
                pos0: 0,
                causal_split: *prompt_len,
                stage: std::cell::RefCell::new(Stage::new()),
            };
            for (i, lw) in model.layers.iter().take(n).enumerate() {
                layer_forward(&r, bufs, lw, i, 0)?;
                std::mem::swap(&mut bufs.hidden_a, &mut bufs.hidden_b);
            }
        }
        ctx.synchronize()?;
        let mut out = vec![0.0f32; self.prompt_len * hidden];
        self.bufs.hidden_a.read_f32(&mut out)?;
        Ok(out)
    }

    /// One denoise forward pass. Returns the canvas's pre-softcap logits,
    /// `[canvas, vocab]` row-major.
    ///
    /// The prompt rows enter from their EMBEDDINGS, not from the post-layer
    /// hidden `prompt_hidden` computes. The engine's denoise step runs the
    /// canvas alone (its attention grids are `active_canvas` wide) and reads
    /// the prompt from the KV cache the prefill wrote; the port has no resident
    /// KV cache, so it rebuilds the prompt's K/V by running the prompt rows
    /// through the layers as part of the same sequence. Handing the layers a
    /// post-layer state re-applies the whole stack to the prompt rows, which
    /// leaves the prompt residual stream 30 layers "ahead" and hands the canvas
    /// the wrong K/V to attend.
    ///
    /// Embedding the prompt instead is exact, not an approximation: a prompt row
    /// is causal (`causal_split` is the prompt length), so it attends only
    /// prompt keys up to itself and cannot see the canvas. Running
    /// `[prompt][canvas]` therefore reproduces the standalone causal pass's
    /// prompt hidden and K/V bit for bit.
    pub fn step(&mut self, prompt: &[u32], canvas_ids: &[u32]) -> Result<Vec<f32>, Error> {
        assert_eq!(prompt.len(), self.prompt_len, "prompt length");
        assert_eq!(canvas_ids.len(), self.canvas, "canvas length");
        let ctx = self.model.ctx.clone();
        let t = self.cfg.text_config.clone();
        let hidden = t.hidden_size;
        let timing = std::env::var("DGQCUDA_TIME").is_ok_and(|v| v != "0");

        // Both row groups enter the layer stack as embeddings, exactly once.
        self.embed_rows(prompt, 0)?;
        self.embed_rows(canvas_ids, self.prompt_len)?;
        // The buffers are sized for [prompt][canvas]; the canvas is written
        // into the tail, leaving no spare row today, but the zeroing keeps a
        // future short canvas from feeding stale rows into attention, which
        // attends every position up to seq and cannot tell a filler row from
        // a live one.
        {
            let base = (self.prompt_len + self.canvas) * hidden;
            let spare = self.seq - (self.prompt_len + self.canvas);
            if spare > 0 {
                self.model.ctx.set_current()?;
                let dst = unsafe { self.bufs.hidden_a.device_ptr() + (base as u64) * 4 };
                self.model.ctx.driver().check(
                    unsafe { (self.model.ctx.driver().cu_memset_d8)(dst, 0, spare * hidden * 4) },
                    "cuMemsetD8",
                )?;
            }
        }
        if timing {
            ctx.synchronize()?;
            eprintln!("  [d] embed ok");
        }

        // Self-conditioning: soft embedding of the previous step's logits, or
        // the canvas embeddings themselves on the first step.
        let first = self.first_step;
        self.first_step = false;
        if !first {
            let lg = self.prev_logits.as_ref().expect("prev logits");
            self.soft_embed(lg)?;
        }
        self.self_condition(first)?;
        if let Some(pos) = std::env::var("DGQCUDA_LAYER_DUMP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&p| p < self.canvas)
        {
            let hidden = self.cfg.text_config.hidden_size;
            self.model.ctx.synchronize()?;
            let mut all = vec![0.0f32; self.seq * hidden];
            self.bufs.hidden_a.read_f32(&mut all)?;
            let off = (self.prompt_len + pos) * hidden;
            emit_layer_checkpoint("after_preamble", &all[off..off + hidden]);
        }
        if timing {
            ctx.synchronize()?;
            eprintln!("  [d] self-condition ok");
        }

        // Split the borrow so Runner can hold the model while bufs is mutated
        // by the layer body.
        let timing = std::env::var("DGQCUDA_TIME").is_ok_and(|v| v != "0");
        let Self {
            model,
            cfg,
            seq,
            prompt_len,
            canvas,
            bufs,
            ..
        } = self;
        let r = Runner {
            m: model,
            cfg,
            seq: *seq,
            pos0: 0,
            causal_split: *prompt_len,
            stage: std::cell::RefCell::new(Stage::new()),
        };
        // `DGQCUDA_LAYER_DUMP=<canvas row>` traces one canvas row's residual
        // down the stack, in the shape of the engine's `step-layer-probe`, so
        // the two can be read side by side to find the first layer that
        // disagrees. Off by default and never allocated when off.
        let inject_after: Option<usize> = std::env::var("DGQCUDA_INJECT_AFTER")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| *seq > *prompt_len);
        let layer_dump: Option<usize> = std::env::var("DGQCUDA_LAYER_DUMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&p| p < *canvas);
        let n_layers = self.layers.min(model.layers.len());
        for (i, lw) in model.layers.iter().take(n_layers).enumerate() {
            layer_forward(&r, bufs, lw, i, 0)?;
            std::mem::swap(&mut bufs.hidden_a, &mut bufs.hidden_b);
            // `DGQCUDA_INJECT_AFTER=<layer>` + `DGQCUDA_INJECT_BIN=<path>`
            // overwrite the canvas rows with the engine's state at that same
            // checkpoint, then let the port run the REST of the stack. If the
            // logits come out right, every layer below the injection point is
            // correct and the divergence is above it; walking the layer down
            // finds the first one that is not. Only the canvas rows are
            // replaced -- the prompt rows stay the port's own, which is what
            // supplies K/V, and they already reproduce the causal prefill.
            if inject_after == Some(i) {
                let path = std::env::var("DGQCUDA_INJECT_BIN").map_err(|_| {
                    Error::Msg("DGQCUDA_INJECT_AFTER needs DGQCUDA_INJECT_BIN".into())
                })?;
                let bytes =
                    std::fs::read(&path).map_err(|e| Error::Msg(format!("inject {path}: {e}")))?;
                let want = *canvas * hidden * 4;
                if bytes.len() != want {
                    return Err(Error::Msg(format!(
                        "inject {path}: {} bytes, expected {want} ({} canvas rows x {hidden})",
                        bytes.len(),
                        *canvas
                    )));
                }
                let vals: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                ctx.synchronize()?;
                let mut all = vec![0.0f32; *seq * hidden];
                bufs.hidden_a.read_f32(&mut all)?;
                let base = *prompt_len * hidden;
                all[base..base + *canvas * hidden].copy_from_slice(&vals);
                bufs.hidden_a.write_f32(&all)?;
                ctx.synchronize()?;
                eprintln!("  [inject] canvas rows replaced after layer {i} from {path}");
            }
            if let Some(pos) = layer_dump {
                // The swap above leaves the layer's output in hidden_a.
                ctx.synchronize()?;
                let mut all = vec![0.0f32; *seq * hidden];
                bufs.hidden_a.read_f32(&mut all)?;
                let off = (*prompt_len + pos) * hidden;
                emit_layer_checkpoint(&format!("after_layer_{i}"), &all[off..off + hidden]);
                // The last PROMPT row through the same layers, as a control.
                // The prompt path is verified against the causal prefill, so a
                // layer that mishandles a good input shows up here too; one
                // that only mishandles the canvas does not.
                let poff = (*prompt_len - 1) * hidden;
                emit_layer_checkpoint(
                    &format!("after_layer_{i}_prompt"),
                    &all[poff..poff + hidden],
                );
            }
            if timing {
                ctx.synchronize()?;
                eprintln!("  [d] layer {i} ok");
            }
        }
        r.rms(&bufs.hidden_a, &model.final_norm, &bufs.hidden_b, *seq)?;
        if let Some(pos) = layer_dump {
            ctx.synchronize()?;
            let mut all = vec![0.0f32; *seq * hidden];
            bufs.hidden_b.read_f32(&mut all)?;
            let off = (*prompt_len + pos) * hidden;
            emit_layer_checkpoint("after_final_norm", &all[off..off + hidden]);
            // The same buffer's last PROMPT row, as an absolute control: the
            // prompt path is verified, so if the canvas row is an outlier here
            // and the prompt row is not, the canvas state is pathological in
            // its own right and not merely different from the engine's.
            let poff = (*prompt_len - 1) * hidden;
            emit_layer_checkpoint("after_final_norm_prompt", &all[poff..poff + hidden]);
        }
        if timing {
            r.mark("d:final_norm");
        }
        r.gemm(
            *canvas,
            t.vocab_size,
            hidden,
            unsafe { bufs.hidden_b.device_ptr() + (*prompt_len * hidden) as u64 * 4 },
            model.lm_head.device_ptr(),
            bufs.logits.device_ptr(),
        )?;
        if timing {
            r.mark("d:lm_head");
        }
        ctx.synchronize()?;
        self.bufs.logits.read_f32(&mut self.host_logits)?;
        if timing {
            r.stage.borrow().report();
        }
        Ok(self.host_logits.clone())
    }

    /// Diagnostic: the device hidden state of the last `step` (pre-final-norm
    /// is `hidden_b`; the post-layer value is what the LM head reads).
    pub fn read_hidden_b(&self, elems: usize) -> Result<Vec<f32>, Error> {
        self.model.ctx.synchronize()?;
        let mut out = vec![0.0f32; elems];
        self.bufs.hidden_b.read_f32(&mut out)?;
        Ok(out)
    }

    /// Keep `logits` as the self-conditioning signal for the next step.
    pub fn set_prev_logits(&mut self, logits: &[f32]) -> Result<(), Error> {
        let buf = self.prev_logits.as_ref().expect("prev_logits");
        buf.write_f32(logits)?;
        Ok(())
    }
}
