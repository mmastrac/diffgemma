//! The CUDA forward pass: the same computation as `crate::forward`, with every
//! weight and intermediate buffer resident on the device.
//!
//! GEMMs go through cuBLAS (resolved with `dlopen` like the driver, so the
//! crate still builds without a CUDA toolkit); the elementwise and attention
//! kernels live in `kernels.cu` and are compiled by NVRTC on first use.

use crate::config::{Error, ModelConfig};
use crate::forward::{LogitRows, Scratch};
use crate::weights::{LayerKeys, Weights};
use gpukit::cuda::{
    BufferPool, Context, DeviceBuffer, KernelArgs, cached_context, cached_source_kernel,
};

pub const KERNELS: &str = include_str!("../kernels.cu");

fn pool() -> Result<(Context, BufferPool), Error> {
    let ctx = cached_context()?.clone();
    Ok((ctx, BufferPool::new()))
}

/// One kernel launch with a fresh argument list.
fn launch(
    ctx: &Context,
    entry: &'static str,
    grid: (u32, u32, u32),
    block: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    let kernel = cached_source_kernel(KERNELS, entry)?;
    Ok(ctx.launch(&kernel, grid, (block, 1, 1), 0, args)?)
}

fn rows(rows: usize, block: u32) -> (u32, u32, u32) {
    (rows as u32, 1, 1)
}

fn flat(n: usize, block: u32) -> (u32, u32, u32) {
    (n.div_ceil(block as usize) as u32, 1, 1)
}

/// Per-stage wall-clock attribution, enabled by `DGQCUDA_TIME=1`. Each mark
/// synchronizes, so this is a diagnostic mode, never a timing baseline.
pub(crate) struct Stage {
    on: bool,
    last: std::time::Instant,
    rows: Vec<(&'static str, f32)>,
}

impl Stage {
    pub(crate) fn new() -> Self {
        Self {
            on: std::env::var("DGQCUDA_TIME").is_ok_and(|v| v != "0"),
            last: std::time::Instant::now(),
            rows: Vec::new(),
        }
    }

    fn mark(&mut self, ctx: &Context, name: &'static str) {
        if !self.on {
            return;
        }
        let _ = ctx.synchronize();
        let now = std::time::Instant::now();
        self.rows.push((name, (now - self.last).as_secs_f32()));
        self.last = now;
    }

    pub(crate) fn report(&self) {
        if !self.on {
            return;
        }
        let total: f32 = self.rows.iter().map(|r| r.1).sum();
        let mut agg: Vec<(&str, f32)> = Vec::new();
        for (n, s) in &self.rows {
            match agg.iter_mut().find(|a| a.0 == *n) {
                Some(a) => a.1 += s,
                None => agg.push((n, *s)),
            }
        }
        agg.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("  [time] total {total:.1}s");
        for (n, s) in agg {
            eprintln!(
                "  [time]   {n:<12} {s:7.1}s  {:4.1}%",
                100.0 * s / total.max(1e-6)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// cuBLAS: row-major C = A @ B (no transposes) via the column-major identity
// C^T = B^T A^T, i.e. sgemm('N','N', n, m, k, B, n, A, k, C, n).
// ---------------------------------------------------------------------------

mod gemm {
    use crate::config::Error;
    use dgemm::Problem;
    use gpukit::cuda::{Context, KernelArgs, cached_source_kernel, div_up, launch_grid};

    /// C[m,n] = A[m,k] @ B[n,k]^T (row-major, PyTorch linear weight layout),
    /// using the tier-1-validated tiled kernel from crates/dgemm.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        ctx: &Context,
        m: usize,
        n: usize,
        k: usize,
        a: gpukit::cuda::driver::CUdeviceptr,
        b: gpukit::cuda::driver::CUdeviceptr,
        c: gpukit::cuda::driver::CUdeviceptr,
    ) -> Result<(), Error> {
        let problem = Problem::linear(m, n, k);
        let kernel = cached_source_kernel(dgemm::CUDA, dgemm::ENTRY)?;
        let p = dgemm::abi_params(&problem);
        let mut args = KernelArgs::new();
        args.device_ptr(a).device_ptr(b).device_ptr(c).bytes(&p);
        if std::env::var("DGQCUDA_GEMM_LOG").is_ok() {
            eprintln!(
                "  [gemm] m={m} n={n} k={k} a=0x{a:x} b=0x{b:x} c=0x{c:x} grid=({},{})",
                div_up(n, dgemm::BN),
                div_up(m, dgemm::BM)
            );
        }
        launch_grid(
            ctx,
            &kernel,
            div_up(n, dgemm::BN),
            div_up(m, dgemm::BM),
            dgemm::THREADS,
            &mut args,
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------

pub(crate) struct GpuLayer {
    pub(crate) input_layernorm: DeviceBuffer,
    pub(crate) q_norm: DeviceBuffer,
    pub(crate) k_norm: DeviceBuffer,
    pub(crate) q_proj: DeviceBuffer,
    pub(crate) k_proj: DeviceBuffer,
    pub(crate) v_proj: Option<DeviceBuffer>,
    pub(crate) o_proj: DeviceBuffer,
    pub(crate) post_attention_layernorm: DeviceBuffer,
    pub(crate) pre_feedforward_layernorm: DeviceBuffer,
    pub(crate) post_feedforward_layernorm: DeviceBuffer,
    pub(crate) post_feedforward_layernorm_1: DeviceBuffer,
    pub(crate) post_feedforward_layernorm_2: DeviceBuffer,
    pub(crate) pre_feedforward_layernorm_2: DeviceBuffer,
    pub(crate) mlp_gate: DeviceBuffer,
    pub(crate) mlp_up: DeviceBuffer,
    pub(crate) mlp_down: DeviceBuffer,
    pub(crate) router_proj: DeviceBuffer,
    pub(crate) router_scale: DeviceBuffer,
    pub(crate) router_per_expert_scale: DeviceBuffer,
    /// The expert stacks as raw q4 bytes ([n_experts, n, k] rows), which
    /// the grouped kernel decodes on the fly. Keeping them quantized is both
    /// 4x less resident memory and 4x fewer bytes read per step.
    pub(crate) experts_gate_up_q4: DeviceBuffer,
    pub(crate) experts_down_q4: DeviceBuffer,
    pub(crate) layer_scalar: f32,
}

pub(crate) struct GpuModel {
    pub(crate) ctx: Context,

    pub(crate) layers: Vec<GpuLayer>,
    /// bf16 embed table, decoded by the gather kernel.
    pub(crate) embed: DeviceBuffer,
    /// The same table widened to f32 for the tied LM-head GEMM (the tiled f32
    /// body has no bf16 input). 2.75 GiB; the f32 GEMM would otherwise read
    /// past the 1.375 GiB bf16 allocation.
    pub(crate) lm_head: DeviceBuffer,
    pub(crate) final_norm: DeviceBuffer,
}

/// Upload raw bytes (the bf16 embed table the gather kernel decodes).
fn up_bf16(ctx: &Context, data: Vec<u8>) -> Result<DeviceBuffer, Error> {
    let b = DeviceBuffer::alloc(ctx, data.len().max(4))?;
    b.write_bytes(&data)?;
    Ok(b)
}

/// bf16 bytes -> f32 (bit pattern shifted, exact).
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([c[0], c[1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

pub(crate) fn up_bytes(ctx: &Context, data: &[u8]) -> Result<DeviceBuffer, Error> {
    let b = DeviceBuffer::alloc(ctx, data.len().max(1))?;
    b.write_bytes(data)?;
    Ok(b)
}

pub(crate) fn up_f32(ctx: &Context, data: &[f32]) -> Result<DeviceBuffer, Error> {
    let b = DeviceBuffer::alloc(ctx, data.len().max(1) * 4)?;
    b.write_f32(data)?;
    Ok(b)
}

impl GpuModel {
    /// Load `n_layers` decoder layers (or all of them when `None`). Loading is
    /// the dominant cost — every tensor is dequantized to f32 and uploaded —
    /// so a bisect run only pays for the layers it uses.
    pub(crate) fn load(
        w: &Weights,
        cfg: &ModelConfig,
        n_layers: Option<usize>,
    ) -> Result<Self, Error> {
        let ctx = cached_context()?.clone();
        let n = n_layers
            .unwrap_or(cfg.text_config.num_hidden_layers)
            .min(cfg.text_config.num_hidden_layers);

        let embed_bytes = w.raw_bf16_bytes("model.decoder.embed_tokens.weight")?;
        let lm_head = up_f32(&ctx, &bf16_to_f32(&embed_bytes))?;
        let embed = up_bf16(&ctx, embed_bytes)?;
        let final_norm = up_f32(&ctx, &w.tensor_f32("model.decoder.norm.weight")?)?;
        let mut layers = Vec::with_capacity(n);
        for layer in 0..n {
            let k = LayerKeys::new(layer);
            let v = if w.has(&k.v_proj) {
                Some(up_f32(&ctx, &w.tensor_f32(&k.v_proj)?)?)
            } else {
                None
            };
            layers.push(GpuLayer {
                input_layernorm: up_f32(&ctx, &w.tensor_f32(&k.input_layernorm)?)?,
                q_norm: up_f32(&ctx, &w.tensor_f32(&k.q_norm)?)?,
                k_norm: up_f32(&ctx, &w.tensor_f32(&k.k_norm)?)?,
                q_proj: up_f32(&ctx, &w.tensor_f32(&k.q_proj)?)?,
                k_proj: up_f32(&ctx, &w.tensor_f32(&k.k_proj)?)?,
                v_proj: v,
                o_proj: up_f32(&ctx, &w.tensor_f32(&k.o_proj)?)?,
                post_attention_layernorm: up_f32(
                    &ctx,
                    &w.tensor_f32(&k.post_attention_layernorm)?,
                )?,
                pre_feedforward_layernorm: up_f32(
                    &ctx,
                    &w.tensor_f32(&k.pre_feedforward_layernorm)?,
                )?,
                post_feedforward_layernorm: up_f32(
                    &ctx,
                    &w.tensor_f32(&k.post_feedforward_layernorm)?,
                )?,
                post_feedforward_layernorm_1: up_f32(
                    &ctx,
                    &w.tensor_f32(&k.post_feedforward_layernorm_1)?,
                )?,
                post_feedforward_layernorm_2: up_f32(
                    &ctx,
                    &w.tensor_f32(&k.post_feedforward_layernorm_2)?,
                )?,
                pre_feedforward_layernorm_2: up_f32(
                    &ctx,
                    &w.tensor_f32(&k.pre_feedforward_layernorm_2)?,
                )?,
                mlp_gate: up_f32(&ctx, &w.tensor_f32(&k.mlp_gate)?)?,
                mlp_up: up_f32(&ctx, &w.tensor_f32(&k.mlp_up)?)?,
                mlp_down: up_f32(&ctx, &w.tensor_f32(&k.mlp_down)?)?,
                router_proj: up_f32(&ctx, &w.tensor_f32(&k.router_proj)?)?,
                router_scale: up_f32(&ctx, &w.tensor_f32(&k.router_scale)?)?,
                router_per_expert_scale: up_f32(&ctx, &w.tensor_f32(&k.router_per_expert_scale)?)?,
                experts_gate_up_q4: up_bytes(&ctx, &w.raw_bf16_bytes(&k.experts_gate_up)?)?,
                experts_down_q4: up_bytes(&ctx, &w.raw_bf16_bytes(&k.experts_down)?)?,
                layer_scalar: w.scalar(&k.layer_scalar)?,
            });
        }
        Ok(Self {
            ctx,
            layers,
            embed,
            lm_head,
            final_norm,
        })
    }
}

pub(crate) struct Bufs {
    pub(crate) ids: DeviceBuffer,
    pub(crate) hidden_a: DeviceBuffer,
    pub(crate) hidden_b: DeviceBuffer,
    pub(crate) normed: DeviceBuffer,
    pub(crate) residual: DeviceBuffer,
    pub(crate) q: DeviceBuffer,
    pub(crate) k: DeviceBuffer,
    pub(crate) v: DeviceBuffer,
    pub(crate) kv: DeviceBuffer,
    pub(crate) attn_out: DeviceBuffer,
    pub(crate) proj_out: DeviceBuffer,
    pub(crate) mlp_gate: DeviceBuffer,
    pub(crate) mlp_up: DeviceBuffer,
    pub(crate) mlp_down: DeviceBuffer,
    pub(crate) moe_in: DeviceBuffer,
    pub(crate) moe_out: DeviceBuffer,
    pub(crate) norm_scratch: DeviceBuffer,
    pub(crate) router_in: DeviceBuffer,
    pub(crate) router_logits: DeviceBuffer,
    pub(crate) top_idx: DeviceBuffer,
    pub(crate) top_w: DeviceBuffer,
    pub(crate) freqs: DeviceBuffer,
    pub(crate) expert_gu: DeviceBuffer,
    pub(crate) expert_act: DeviceBuffer,
    pub(crate) expert_out: DeviceBuffer,
    /// Grouped MoE: expert-major rows of the flat [rows, hidden] input, the
    /// [rows, 2*inter] gate/up result, the [rows, inter] activation, the
    /// [rows, hidden] down result, and the per-row bucket tables.
    pub(crate) moe_rows_a: DeviceBuffer,
    pub(crate) moe_rows_gu: DeviceBuffer,
    pub(crate) moe_rows_act: DeviceBuffer,
    pub(crate) moe_rows_down: DeviceBuffer,
    pub(crate) moe_starts: DeviceBuffer,
    pub(crate) moe_tok_idx: DeviceBuffer,
    pub(crate) moe_row_w: DeviceBuffer,
    /// [seq, vocab] — the full canvas's logits on the `LogitRows::All` path,
    /// one row on the others.
    pub(crate) logits: DeviceBuffer,
}

impl Bufs {
    pub(crate) fn new(seq: usize, cfg: &ModelConfig, ctx: &Context) -> Result<Self, Error> {
        let t = &cfg.text_config;
        let hidden = t.hidden_size;
        let max_q = t.num_attention_heads * t.global_head_dim;
        let max_kv = t.num_key_value_heads.max(t.num_global_key_value_heads) * t.global_head_dim;
        let a = |n: usize| DeviceBuffer::alloc(ctx, n.max(1) * 4);
        Ok(Self {
            ids: a(seq)?,
            hidden_a: a(seq * hidden)?,
            hidden_b: a(seq * hidden)?,
            normed: a(seq * hidden)?,
            residual: a(seq * hidden)?,
            q: a(seq * max_q)?,
            k: a(seq * max_kv)?,
            v: a(seq * max_kv)?,
            kv: a(seq * max_kv * 2)?,
            attn_out: a(seq * max_q)?,
            proj_out: a(seq * hidden)?,
            mlp_gate: a(seq * t.intermediate_size)?,
            mlp_up: a(seq * t.intermediate_size)?,
            mlp_down: a(seq * hidden)?,
            moe_in: a(seq * hidden)?,
            moe_out: a(seq * hidden)?,
            norm_scratch: a(seq * hidden)?,
            router_in: a(seq * hidden)?,
            router_logits: a(seq * t.num_experts)?,
            top_idx: a(seq * t.top_k_experts)?,
            top_w: a(seq * t.top_k_experts)?,
            freqs: a(seq * t.global_head_dim)?,
            expert_gu: a(t.moe_intermediate_size * 2)?,
            expert_act: a(t.moe_intermediate_size)?,
            expert_out: a(hidden)?,
            moe_rows_a: a(seq * t.top_k_experts * hidden)?,
            moe_rows_gu: a(seq * t.top_k_experts * t.moe_intermediate_size * 2)?,
            moe_rows_act: a(seq * t.top_k_experts * t.moe_intermediate_size)?,
            moe_rows_down: a(seq * t.top_k_experts * hidden)?,
            moe_starts: a(t.num_experts + 1)?,
            moe_tok_idx: a(seq * t.top_k_experts)?,
            moe_row_w: a(seq * t.top_k_experts)?,
            logits: a(seq * t.vocab_size)?,
        })
    }
}

pub(crate) struct Runner<'a> {
    pub(crate) m: &'a GpuModel,
    pub(crate) cfg: &'a ModelConfig,
    pub(crate) seq: usize,
    /// Absolute position of row 0 (0 for a from-scratch forward pass; the
    /// prompt length when a denoise pass runs the canvas after the prompt).
    pub(crate) pos0: usize,
    /// Leading rows that attend causally; `seq` is a plain causal pass.
    pub(crate) causal_split: usize,
    pub(crate) stage: std::cell::RefCell<Stage>,
}

impl<'a> Runner<'a> {
    fn t(&self) -> &crate::config::TextConfig {
        &self.cfg.text_config
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: gpukit::cuda::driver::CUdeviceptr,
        b: gpukit::cuda::driver::CUdeviceptr,
        c: gpukit::cuda::driver::CUdeviceptr,
    ) -> Result<(), Error> {
        gemm::gemm(&self.m.ctx, m, n, k, a, b, c)
    }

    pub(crate) fn rms(
        &self,
        x: &DeviceBuffer,
        w: &DeviceBuffer,
        out: &DeviceBuffer,
        n_rows: usize,
    ) -> Result<(), Error> {
        let hidden = self.t().hidden_size;
        let eps = self.t().rms_norm_eps as f32;
        let mut args = KernelArgs::new();
        args.device_ptr(x.device_ptr())
            .device_ptr(w.device_ptr())
            .device_ptr(out.device_ptr())
            .u32(n_rows as u32)
            .u32(hidden as u32)
            .f32(eps);
        launch(
            &self.m.ctx,
            "dgq_rms_norm",
            rows(n_rows, 256),
            256,
            &mut args,
        )
    }

    pub(crate) fn mark(&self, name: &'static str) {
        self.stage.borrow_mut().mark(&self.m.ctx, name);
    }

    fn add_in_place(&self, dst: &DeviceBuffer, src: &DeviceBuffer, n: usize) -> Result<(), Error> {
        let mut args = KernelArgs::new();
        args.device_ptr(dst.device_ptr())
            .device_ptr(src.device_ptr())
            .u32(n as u32);
        Ok(launch(
            &self.m.ctx,
            "dgq_vec_add",
            flat(n, 256),
            256,
            &mut args,
        )?)
    }
}

/// The CUDA forward pass. Mirrors `crate::forward::forward`; the parity test
/// compares the two logit buffers.
pub fn forward(
    w: &Weights,
    cfg: &ModelConfig,
    ids: &[u32],
    layers: Option<usize>,
    rows: LogitRows,
    sc: &mut Scratch,
) -> Result<crate::forward::ForwardOutput, Error> {
    forward_stop(w, cfg, ids, layers, rows, sc, 0)
}

/// `forward` with a diagnostic stop point (1 embed, 2 final norm, 3 lm_head
/// GEMM) for bisecting a device fault.
pub fn forward_stop(
    w: &Weights,
    cfg: &ModelConfig,
    ids: &[u32],
    layers: Option<usize>,
    rows: LogitRows,
    sc: &mut Scratch,
    stop_after: u8,
) -> Result<crate::forward::ForwardOutput, Error> {
    forward_full(w, cfg, ids, layers, rows, sc, stop_after, None, 0, 0, true)
}

/// The device forward pass with every knob the denoise loop needs: an optional
/// pre-computed hidden state to start from (the prompt's, so a denoise step only
/// runs the layers above it), the absolute position of row 0, the number of
/// leading rows that attend causally, and whether to apply the final logit
/// softcap.
#[allow(clippy::too_many_arguments)]
pub fn forward_full(
    w: &Weights,
    cfg: &ModelConfig,
    ids: &[u32],
    layers: Option<usize>,
    rows: LogitRows,
    sc: &mut Scratch,
    stop_after: u8,
    init_hidden: Option<&[f32]>,
    pos0: usize,
    causal_split: usize,
    softcap: bool,
) -> Result<crate::forward::ForwardOutput, Error> {
    let _ = sc;
    let mut stage = Stage::new();
    let model = GpuModel::load(w, cfg, layers)?;
    let ctx = model.ctx.clone();
    let t = cfg.text_config.clone();
    let seq = ids.len();
    let hidden = t.hidden_size;
    stage.mark(&ctx, "load");
    let mut b = Bufs::new(seq, cfg, &ctx)?;
    let r = Runner {
        m: &model,
        cfg,
        seq,
        pos0,
        causal_split,
        stage: std::cell::RefCell::new(Stage::new()),
    };

    let skip_layers = match init_hidden {
        Some(h) => {
            assert_eq!(h.len(), seq * hidden, "init_hidden shape");
            b.hidden_a.write_f32(h)?;
            true
        }
        None => false,
    };

    if !skip_layers {
        // ---- embed gather ------------------------------------------------
        b.ids.write_bytes(unsafe {
            std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4)
        })?;
        // out[t, d] = embed[id[t], d] * scale, one row at a time (the op's
        // gather is row-wise; the table is huge so this stays O(seq * hidden)
        // of reads).
        let embed_scale = (hidden as f32).sqrt();
        for s in 0..seq {
            let mut args = KernelArgs::new();
            args.device_ptr(model.embed.device_ptr())
                .device_ptr(unsafe { b.ids.device_ptr() + (s as u64) * 4 })
                .device_ptr(unsafe { b.hidden_a.device_ptr() + (s as u64) * (hidden as u64) * 4 })
                .u32(hidden as u32)
                .u32(1)
                .u32(0)
                .u32(0)
                .f32(embed_scale)
                .u32(t.vocab_size as u32)
                .u32(1);
            let k = cached_source_kernel(dgops::ops::embed_gather::CUDA, "embed_gather")?;
            ctx.launch(&k, flat(hidden, 256), (256, 1, 1), 0, &mut args)?;
        }
    }
    stage.mark(&ctx, "embed");
    if stop_after == 1 {
        ctx.synchronize()?;
        eprintln!("  [stop] after embed");
        return Ok(crate::forward::ForwardOutput {
            logits: vec![0.0; t.vocab_size],
        });
    }

    let n_layers = layers
        .unwrap_or(t.num_hidden_layers)
        .min(t.num_hidden_layers);
    for layer in 0..n_layers {
        layer_forward(&r, &mut b, &model.layers[layer], layer, 0)?;
        std::mem::swap(&mut b.hidden_a, &mut b.hidden_b);
        stage.mark(&ctx, "layer");
    }

    r.rms(&b.hidden_a, &model.final_norm, &b.hidden_b, seq)?;
    stage.mark(&ctx, "final_norm");
    if stop_after == 2 {
        ctx.synchronize()?;
        eprintln!("  [stop] after final_norm");
        return Ok(crate::forward::ForwardOutput {
            logits: vec![0.0; t.vocab_size],
        });
    }

    // tied LM head. `All` runs the whole canvas in one GEMM (the denoise
    // step's shape); the single-row modes still do m=1.
    let (m, src) = match rows {
        LogitRows::Last => (1, seq - 1),
        LogitRows::Only(r) => (1, r),
        LogitRows::All => (seq, 0),
    };
    r.gemm(
        m,
        t.vocab_size,
        hidden,
        unsafe { offset_view(&b.hidden_b, src * hidden).device_ptr() },
        model.lm_head.device_ptr(),
        b.logits.device_ptr(),
    )?;
    if stop_after == 3 {
        ctx.synchronize()?;
        eprintln!("  [stop] after lm_head gemm");
        return Ok(crate::forward::ForwardOutput {
            logits: vec![0.0; t.vocab_size],
        });
    }
    let cap = t.final_logit_softcapping as f32;
    if softcap && cap > 0.0 {
        let mut args = KernelArgs::new();
        args.device_ptr(b.logits.device_ptr())
            .f32(cap)
            .u32((m * t.vocab_size) as u32);
        launch(
            &ctx,
            "dgq_softcap",
            flat(m * t.vocab_size, 256),
            256,
            &mut args,
        )?;
    }
    ctx.synchronize()?;
    stage.mark(&ctx, "lm_head");
    let mut logits = vec![0.0f32; m * t.vocab_size];
    b.logits.read_f32(&mut logits)?;
    stage.mark(&ctx, "readback");
    r.stage.borrow().report();
    stage.report();
    Ok(crate::forward::ForwardOutput { logits })
}

/// A device view offset by `elems` f32 elements, sharing the parent allocation.
struct View<'a>(&'a DeviceBuffer, u64);

impl<'a> View<'a> {
    fn device_ptr(&self) -> gpukit::cuda::driver::CUdeviceptr {
        unsafe { self.0.device_ptr() + self.1 * 4 }
    }
}

fn offset_view<'a>(buf: &'a DeviceBuffer, elems: usize) -> View<'a> {
    View(buf, elems as u64)
}

pub(crate) fn layer_forward(
    r: &Runner<'_>,
    b: &mut Bufs,
    lw: &GpuLayer,
    layer: usize,
    stop_at: u8,
) -> Result<(), Error> {
    let t = r.t();
    let hidden = t.hidden_size;
    let seq = r.seq;
    let eps = t.rms_norm_eps as f32;
    let (n_kv, head_dim, rotary_dim, theta, window) = t.attn_geometry(layer);
    let pos0 = r.pos0;
    let causal_split = r.causal_split;
    let n_heads = t.num_attention_heads;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let ctx = &r.m.ctx;

    // residual = hidden_a
    copy_device(ctx, &b.residual, &b.hidden_a, seq * hidden)?;
    r.mark("l:residual");

    // ---- attention -------------------------------------------------------
    r.rms(&b.hidden_a, &lw.input_layernorm, &b.normed, seq)?;
    r.gemm(
        seq,
        q_dim,
        hidden,
        b.normed.device_ptr(),
        lw.q_proj.device_ptr(),
        b.q.device_ptr(),
    )?;
    r.gemm(
        seq,
        kv_dim,
        hidden,
        b.normed.device_ptr(),
        lw.k_proj.device_ptr(),
        b.k.device_ptr(),
    )?;
    match &lw.v_proj {
        Some(vw) => r.gemm(
            seq,
            kv_dim,
            hidden,
            b.normed.device_ptr(),
            vw.device_ptr(),
            b.v.device_ptr(),
        )?,
        None => copy_device(ctx, &b.v, &b.k, seq * kv_dim)?,
    }

    // per-head QK-norm (V unweighted)
    let mut args = KernelArgs::new();
    args.device_ptr(b.q.device_ptr())
        .device_ptr(lw.q_norm.device_ptr())
        .device_ptr(b.q.device_ptr())
        .u32(seq as u32)
        .u32(n_heads as u32)
        .u32(head_dim as u32)
        .f32(eps);
    launch(
        ctx,
        "dgq_rms_norm_heads",
        (n_heads as u32, seq as u32, 1),
        128,
        &mut args,
    )?;
    let mut args = KernelArgs::new();
    args.device_ptr(b.k.device_ptr())
        .device_ptr(lw.k_norm.device_ptr())
        .device_ptr(b.k.device_ptr())
        .u32(seq as u32)
        .u32(n_kv as u32)
        .u32(head_dim as u32)
        .f32(eps);
    launch(
        ctx,
        "dgq_rms_norm_heads",
        (n_kv as u32, seq as u32, 1),
        128,
        &mut args,
    )?;
    let mut args = KernelArgs::new();
    args.device_ptr(b.v.device_ptr())
        .device_ptr(b.v.device_ptr()) // weight ignored when null_ptr = 0
        .device_ptr(b.v.device_ptr())
        .u32(seq as u32)
        .u32(n_kv as u32)
        .u32(head_dim as u32)
        .f32(eps);
    // pass a null weight pointer for the V norm
    let mut args = KernelArgs::new();
    args.device_ptr(b.v.device_ptr())
        .u64(0)
        .device_ptr(b.v.device_ptr())
        .u32(seq as u32)
        .u32(n_kv as u32)
        .u32(head_dim as u32)
        .f32(eps);
    launch(
        ctx,
        "dgq_rms_norm_heads",
        (n_kv as u32, seq as u32, 1),
        128,
        &mut args,
    )?;

    // RoPE
    let freqs = crate::forward::rope_freqs(seq, rotary_dim, head_dim, theta);
    b.freqs.write_f32(&freqs)?;
    let mut args = KernelArgs::new();
    args.device_ptr(b.q.device_ptr())
        .device_ptr(b.freqs.device_ptr())
        .u32(seq as u32)
        .u32(n_heads as u32)
        .u32(head_dim as u32)
        .u32(rotary_dim as u32);
    launch(ctx, "dgq_rope", flat(seq * n_heads, 128), 128, &mut args)?;
    let mut args = KernelArgs::new();
    args.device_ptr(b.k.device_ptr())
        .device_ptr(b.freqs.device_ptr())
        .u32(seq as u32)
        .u32(n_kv as u32)
        .u32(head_dim as u32)
        .u32(rotary_dim as u32);
    launch(ctx, "dgq_rope", flat(seq * n_kv, 128), 128, &mut args)?;

    // KV region: [t, n_kv, 2*head_dim] = K then V
    interleave_kv(ctx, &b.k, &b.v, &b.kv, seq, n_kv, head_dim)?;

    let mut args = KernelArgs::new();
    args.device_ptr(b.q.device_ptr())
        .device_ptr(b.kv.device_ptr())
        .device_ptr(b.attn_out.device_ptr())
        .u32(seq as u32)
        .u32(n_heads as u32)
        .u32(n_kv as u32)
        .u32(head_dim as u32)
        .u32(seq as u32)
        .u32(window.unwrap_or(0) as u32)
        .u32(pos0 as u32)
        .u32(causal_split as u32);
    launch(
        ctx,
        "dgq_attention_v2",
        rows(seq * n_heads, 128),
        128,
        &mut args,
    )?;

    r.gemm(
        seq,
        hidden,
        q_dim,
        b.attn_out.device_ptr(),
        lw.o_proj.device_ptr(),
        b.proj_out.device_ptr(),
    )?;
    r.rms(&b.proj_out, &lw.post_attention_layernorm, &b.normed, seq)?;
    r.add_in_place(&b.normed, &b.residual, seq * hidden)?;
    r.mark("l:attn");
    if stop_at == 1 {
        return Ok(());
    }

    // ---- dense MLP -------------------------------------------------------
    copy_device(ctx, &b.residual, &b.normed, seq * hidden)?;
    r.rms(&b.residual, &lw.pre_feedforward_layernorm, &b.normed, seq)?;
    let inter = t.intermediate_size;
    r.gemm(
        seq,
        inter,
        hidden,
        b.normed.device_ptr(),
        lw.mlp_gate.device_ptr(),
        b.mlp_gate.device_ptr(),
    )?;
    r.gemm(
        seq,
        inter,
        hidden,
        b.normed.device_ptr(),
        lw.mlp_up.device_ptr(),
        b.mlp_up.device_ptr(),
    )?;
    // out = gelu_tanh(gate) * up, written into mlp_gate
    // (kernel args: gate, up, weight, out, len)
    let mut args = KernelArgs::new();
    args.device_ptr(b.mlp_gate.device_ptr())
        .device_ptr(b.mlp_up.device_ptr())
        .f32(1.0)
        .device_ptr(b.mlp_gate.device_ptr())
        .u32((seq * inter) as u32);
    let _ = &args;
    launch(
        ctx,
        "dgq_swiglu_weighted",
        flat(seq * inter, 256),
        256,
        &mut args,
    )?;
    r.gemm(
        seq,
        hidden,
        inter,
        b.mlp_gate.device_ptr(),
        lw.mlp_down.device_ptr(),
        b.mlp_down.device_ptr(),
    )?;
    // CPU oracle: scratch = mlp_down; mlp_down = rms(scratch) * w1; then the
    // dense branch's output is ADDED to normed (which still holds the residual
    // from before the pre-feedforward norm).
    r.rms(
        &b.mlp_down,
        &lw.post_feedforward_layernorm_1,
        &b.norm_scratch,
        seq,
    )?;
    copy_device(ctx, &b.mlp_down, &b.norm_scratch, seq * hidden)?;
    r.add_in_place(&b.normed, &b.mlp_down, seq * hidden)?;
    r.mark("l:dense_mlp");

    if stop_at == 2 {
        return Ok(());
    }
    // ---- MoE -------------------------------------------------------------
    let root = (hidden as f32).powf(-0.5);
    let mut args = KernelArgs::new();
    args.device_ptr(b.residual.device_ptr())
        .device_ptr(lw.router_scale.device_ptr())
        .device_ptr(b.router_in.device_ptr())
        .u32(seq as u32)
        .u32(hidden as u32)
        .f32(eps)
        .f32(root);
    launch(ctx, "dgq_router_input", rows(seq, 256), 256, &mut args)?;
    r.gemm(
        seq,
        t.num_experts,
        hidden,
        b.router_in.device_ptr(),
        lw.router_proj.device_ptr(),
        b.router_logits.device_ptr(),
    )?;

    r.rms(&b.residual, &lw.pre_feedforward_layernorm_2, &b.moe_in, seq)?;

    let mut args = KernelArgs::new();
    args.device_ptr(b.router_logits.device_ptr())
        .device_ptr(lw.router_per_expert_scale.device_ptr())
        .device_ptr(b.top_idx.device_ptr())
        .device_ptr(b.top_w.device_ptr())
        .u32(seq as u32)
        .u32(t.num_experts as u32)
        .u32(t.top_k_experts as u32);
    launch(ctx, "dgq_router_topk", rows(seq, 32), 32, &mut args)?;
    ctx.synchronize()?;
    r.mark("l:router");
    let mut idx = vec![0u32; seq * t.top_k_experts];
    let mut wts = vec![0.0f32; seq * t.top_k_experts];
    b.top_idx.read_bytes(unsafe {
        std::slice::from_raw_parts_mut(idx.as_mut_ptr().cast::<u8>(), idx.len() * 4)
    })?;
    b.top_w.read_f32(&mut wts)?;
    r.mark("l:route_readback");

    // Grouped MoE: bucket the tokens by expert, gather their normalized rows
    // into one expert-major matrix, and run one tiled GEMM per expert bucket —
    // the expert weight tile is then reused across every token in the bucket.
    b.moe_out.zero()?;
    let moe_inter = t.moe_intermediate_size;
    let plan =
        crate::moe_grouped::GroupedPlan::new(&idx, &wts, seq, t.top_k_experts, t.num_experts);
    let n_rows = plan.rows();
    // The plan tables live in the session buffers: uploading fresh ones per
    // layer would add 60 host-to-device copies (and their syncs) per step.
    b.moe_starts.write_bytes(unsafe {
        std::slice::from_raw_parts(plan.starts.as_ptr().cast::<u8>(), plan.starts.len() * 4)
    })?;
    b.moe_tok_idx.write_bytes(unsafe {
        std::slice::from_raw_parts(plan.tok_idx.as_ptr().cast::<u8>(), plan.tok_idx.len() * 4)
    })?;
    b.moe_row_w.write_f32(&plan.row_w)?;
    // Gather the expert-major input rows.
    {
        let mut args = KernelArgs::new();
        args.device_ptr(b.moe_in.device_ptr())
            .device_ptr(b.moe_rows_a.device_ptr())
            .device_ptr(b.moe_tok_idx.device_ptr())
            .u32(n_rows as u32)
            .u32(hidden as u32);
        launch(ctx, "dgq_moe_gather", rows(n_rows, 256), 256, &mut args)?;
    }
    let gemm_gu = crate::moe_grouped::GroupedGemm::new(hidden, moe_inter * 2, "dgq_moe_gate_up");
    gemm_gu.run(
        ctx,
        &b.moe_rows_a,
        &lw.experts_gate_up_q4,
        &b.moe_starts,
        &b.moe_tok_idx,
        &b.moe_rows_gu,
        n_rows,
        plan.num_jobs(),
    )?;
    {
        let mut args = KernelArgs::new();
        args.device_ptr(b.moe_rows_gu.device_ptr())
            .device_ptr(b.moe_rows_act.device_ptr())
            .device_ptr(b.moe_row_w.device_ptr())
            .u32(n_rows as u32)
            .u32(moe_inter as u32);
        launch(
            ctx,
            "dgq_moe_swiglu_weighted",
            flat(n_rows * moe_inter, 256),
            256,
            &mut args,
        )?;
    }
    let gemm_down = crate::moe_grouped::GroupedGemm::new(moe_inter, hidden, "dgq_moe_down");
    gemm_down.run(
        ctx,
        &b.moe_rows_act,
        &lw.experts_down_q4,
        &b.moe_starts,
        &b.moe_tok_idx,
        &b.moe_rows_down,
        n_rows,
        plan.num_jobs(),
    )?;
    {
        let mut args = KernelArgs::new();
        args.device_ptr(b.moe_out.device_ptr())
            .device_ptr(b.moe_rows_down.device_ptr())
            .device_ptr(b.moe_tok_idx.device_ptr())
            .u32(n_rows as u32)
            .u32(hidden as u32);
        launch(
            ctx,
            "dgq_moe_scatter",
            flat(n_rows * hidden, 256),
            256,
            &mut args,
        )?;
    }
    // CPU oracle: scratch = moe_out; moe_out = rms(scratch) * w2; normed += moe_out.
    // rms writes into b.normed (which holds this branch's residual), so the
    // normalized value is moved into moe_out before the accumulate, leaving
    // b.normed = residual + rms(moe_out).
    // CPU oracle: scratch = moe_out; moe_out = rms(scratch) * w2; normed += moe_out.
    // rms leaves the normalized value in b.normed; b.moe_out still holds the raw
    // value, so add first and then move the normalized result into moe_out.
    // CPU oracle: scratch = moe_out; moe_out = rms(scratch) * w2; normed += moe_out.
    // b.normed holds the layer residual and must not be overwritten, so the
    // normalized value goes to the scratch, the residual accumulates into it,
    // and the result moves back to moe_out for the next stage.
    r.mark("l:experts");
    r.rms(
        &b.moe_out,
        &lw.post_feedforward_layernorm_2,
        &b.norm_scratch,
        seq,
    )?;
    r.add_in_place(&b.norm_scratch, &b.normed, seq * hidden)?;
    copy_device(ctx, &b.normed, &b.norm_scratch, seq * hidden)?;
    copy_device(ctx, &b.moe_out, &b.norm_scratch, seq * hidden)?;

    if stop_at == 3 {
        return Ok(());
    }
    // ---- output ----------------------------------------------------------
    r.rms(&b.normed, &lw.post_feedforward_layernorm, &b.hidden_b, seq)?;
    r.add_in_place(&b.hidden_b, &b.residual, seq * hidden)?;
    let mut args = KernelArgs::new();
    args.device_ptr(b.hidden_b.device_ptr())
        .f32(lw.layer_scalar)
        .u32((seq * hidden) as u32);
    launch(ctx, "dgq_scale", flat(seq * hidden, 256), 256, &mut args)?;
    Ok(())
}

fn copy_device(
    ctx: &Context,
    dst: &DeviceBuffer,
    src: &DeviceBuffer,
    elems: usize,
) -> Result<(), Error> {
    ctx.set_current()?;
    Ok(ctx.driver().check(
        unsafe { (ctx.driver().cu_memcpy_dtod)(dst.device_ptr(), src.device_ptr(), elems * 4) },
        "cuMemcpyDtoD",
    )?)
}

/// [seq, n_kv, head_dim] K and V -> [seq, 2*n_kv*head_dim]: for each position,
/// all K heads followed by all V heads. dgq_attention reads
/// \`k = kv + ((t*nkv + h)*hd)\` and \`v = k + nkv*hd\`, and the kernel test
/// (tests/kernels.rs::attention_layouts) pins this layout: the per-head
/// alternative scores cos 0.098.
fn interleave_kv(
    ctx: &Context,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    kv: &DeviceBuffer,
    seq: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<(), Error> {
    ctx.set_current()?;
    let row = n_kv * head_dim;
    for t in 0..seq {
        let base = t * 2 * row;
        ctx.driver().check(
            unsafe {
                (ctx.driver().cu_memcpy_dtod)(
                    kv.device_ptr() + (base as u64) * 4,
                    k.device_ptr() + (t * row) as u64 * 4,
                    row * 4,
                )
            },
            "cuMemcpyDtoD",
        )?;
        ctx.driver().check(
            unsafe {
                (ctx.driver().cu_memcpy_dtod)(
                    kv.device_ptr() + ((base + row) as u64) * 4,
                    v.device_ptr() + (t * row) as u64 * 4,
                    row * 4,
                )
            },
            "cuMemcpyDtoD",
        )?;
    }
    Ok(())
}

/// Hidden state after \`layers\` decoder layers (device-resident path).
pub fn hidden_after(
    w: &Weights,
    cfg: &ModelConfig,
    ids: &[u32],
    layers: usize,
    stop_at: u8,
    _sc: &mut Scratch,
) -> Result<Vec<f32>, Error> {
    let model = GpuModel::load(w, cfg, Some(layers))?;
    let ctx = model.ctx.clone();
    let t = cfg.text_config.clone();
    let seq = ids.len();
    let hidden = t.hidden_size;
    let mut b = Bufs::new(seq, cfg, &ctx)?;
    let r = Runner {
        m: &model,
        cfg,
        seq,
        pos0: 0,
        causal_split: seq,
        stage: std::cell::RefCell::new(Stage::new()),
    };
    b.ids.write_bytes(unsafe {
        std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4)
    })?;
    let embed_scale = (hidden as f32).sqrt();
    for s in 0..seq {
        let mut args = KernelArgs::new();
        args.device_ptr(model.embed.device_ptr())
            .device_ptr(unsafe { b.ids.device_ptr() + (s as u64) * 4 })
            .device_ptr(unsafe { b.hidden_a.device_ptr() + (s as u64) * (hidden as u64) * 4 })
            .u32(hidden as u32)
            .u32(1)
            .u32(0)
            .u32(0)
            .f32(embed_scale)
            .u32(t.vocab_size as u32)
            .u32(1);
        let k = cached_source_kernel(dgops::ops::embed_gather::CUDA, "embed_gather")?;
        ctx.launch(&k, flat(hidden, 256), (256, 1, 1), 0, &mut args)?;
    }
    for layer in 0..layers {
        let stop = if layer + 1 == layers { stop_at } else { 0 };
        layer_forward(&r, &mut b, &model.layers[layer], layer, stop)?;
        std::mem::swap(&mut b.hidden_a, &mut b.hidden_b);
    }
    ctx.synchronize()?;
    let mut out = vec![0.0f32; seq * hidden];
    if stop_at == 1 || stop_at == 2 || stop_at == 3 {
        b.normed.read_f32(&mut out)?;
    } else {
        b.hidden_a.read_f32(&mut out)?;
    }
    Ok(out)
}

/// Attention sub-layer output on the device path (bisect helper).
pub fn attn_stage(
    w: &Weights,
    cfg: &ModelConfig,
    ids: &[u32],
    _sc: &mut Scratch,
) -> Result<Vec<f32>, Error> {
    let model = GpuModel::load(w, cfg, Some(1))?;
    let ctx = model.ctx.clone();
    let t = cfg.text_config.clone();
    let seq = ids.len();
    let hidden = t.hidden_size;
    let mut b = Bufs::new(seq, cfg, &ctx)?;
    let r = Runner {
        m: &model,
        cfg,
        seq,
        pos0: 0,
        causal_split: seq,
        stage: std::cell::RefCell::new(Stage::new()),
    };
    b.ids.write_bytes(unsafe {
        std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4)
    })?;
    let embed_scale = (hidden as f32).sqrt();
    for s in 0..seq {
        let mut args = KernelArgs::new();
        args.device_ptr(model.embed.device_ptr())
            .device_ptr(unsafe { b.ids.device_ptr() + (s as u64) * 4 })
            .device_ptr(unsafe { b.hidden_a.device_ptr() + (s as u64) * (hidden as u64) * 4 })
            .u32(hidden as u32)
            .u32(1)
            .u32(0)
            .u32(0)
            .f32(embed_scale)
            .u32(t.vocab_size as u32)
            .u32(1);
        let k = cached_source_kernel(dgops::ops::embed_gather::CUDA, "embed_gather")?;
        ctx.launch(&k, flat(hidden, 256), (256, 1, 1), 0, &mut args)?;
    }
    layer_forward(&r, &mut b, &model.layers[0], 0, 1)?;
    ctx.synchronize()?;
    let mut out = vec![0.0f32; seq * hidden];
    b.normed.read_f32(&mut out)?;
    Ok(out)
}

/// Test hook: C[m,n] = A[m,k] @ B[k,n] through the same cuBLAS path the
/// forward pass uses.
/// Diagnostic: run the dgemm body on a zeroed device-only B of the given
/// shape, with A and C small — the LM head's shape without the 737 MiB embed
/// table. Returns C's first element (the value itself is meaningless).
pub fn gemm_probe(m: usize, k: usize, n: usize) -> Result<f32, Error> {
    let ctx = cached_context()?.clone();
    let ba = up_f32(&ctx, &vec![1.0f32; m * k])?;
    let bb = DeviceBuffer::alloc(&ctx, n * k * 4)?;
    bb.zero()?;
    let bc = DeviceBuffer::alloc(&ctx, m * n * 4)?;
    eprintln!(
        "  [probe] a={:?} b={:?} c={:?}",
        ba.size(),
        bb.size(),
        bc.size()
    );
    gemm::gemm(
        &ctx,
        m,
        n,
        k,
        ba.device_ptr(),
        bb.device_ptr(),
        bc.device_ptr(),
    )?;
    eprintln!("  [probe] launched, synchronizing");
    ctx.synchronize()?;
    let mut out = vec![0.0f32; m * n];
    bc.read_f32(&mut out)?;
    Ok(out[0])
}

pub fn cublas_probe(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, Error> {
    let ctx = cached_context()?.clone();

    let ba = up_f32(&ctx, a)?;
    let bb = up_f32(&ctx, b)?;
    let bc = DeviceBuffer::alloc(&ctx, m * n * 4)?;
    gemm::gemm(
        &ctx,
        m,
        n,
        k,
        ba.device_ptr(),
        bb.device_ptr(),
        bc.device_ptr(),
    )?;
    ctx.synchronize()?;
    let mut out = vec![0.0f32; m * n];
    bc.read_f32(&mut out)?;
    Ok(out)
}
