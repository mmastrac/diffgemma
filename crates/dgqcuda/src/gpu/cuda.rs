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

struct GpuLayer {
    input_layernorm: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: Option<DeviceBuffer>,
    o_proj: DeviceBuffer,
    post_attention_layernorm: DeviceBuffer,
    pre_feedforward_layernorm: DeviceBuffer,
    post_feedforward_layernorm: DeviceBuffer,
    post_feedforward_layernorm_1: DeviceBuffer,
    post_feedforward_layernorm_2: DeviceBuffer,
    pre_feedforward_layernorm_2: DeviceBuffer,
    mlp_gate: DeviceBuffer,
    mlp_up: DeviceBuffer,
    mlp_down: DeviceBuffer,
    router_proj: DeviceBuffer,
    router_scale: DeviceBuffer,
    router_per_expert_scale: DeviceBuffer,
    experts_gate_up: DeviceBuffer,
    experts_down: DeviceBuffer,
    layer_scalar: f32,
}

struct GpuModel {
    ctx: Context,

    layers: Vec<GpuLayer>,
    embed: DeviceBuffer,
    final_norm: DeviceBuffer,
}

/// Upload raw bytes (the bf16 embed table the gather kernel decodes).
fn up_bf16(ctx: &Context, data: Vec<u8>) -> Result<DeviceBuffer, Error> {
    let b = DeviceBuffer::alloc(ctx, data.len().max(4))?;
    b.write_bytes(&data)?;
    Ok(b)
}

fn up_f32(ctx: &Context, data: &[f32]) -> Result<DeviceBuffer, Error> {
    let b = DeviceBuffer::alloc(ctx, data.len().max(1) * 4)?;
    b.write_f32(data)?;
    Ok(b)
}

impl GpuModel {
    fn load(w: &Weights, cfg: &ModelConfig) -> Result<Self, Error> {
        let ctx = cached_context()?.clone();

        let embed = up_bf16(&ctx, w.raw_bf16_bytes("model.decoder.embed_tokens.weight")?)?;
        let final_norm = up_f32(&ctx, &w.tensor_f32("model.decoder.norm.weight")?)?;
        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        for layer in 0..cfg.text_config.num_hidden_layers {
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
                experts_gate_up: up_f32(&ctx, &w.tensor_f32(&k.experts_gate_up)?)?,
                experts_down: up_f32(&ctx, &w.tensor_f32(&k.experts_down)?)?,
                layer_scalar: w.scalar(&k.layer_scalar)?,
            });
        }
        Ok(Self {
            ctx,
            layers,
            embed,
            final_norm,
        })
    }
}

struct Bufs {
    ids: DeviceBuffer,
    hidden_a: DeviceBuffer,
    hidden_b: DeviceBuffer,
    normed: DeviceBuffer,
    residual: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    kv: DeviceBuffer,
    attn_out: DeviceBuffer,
    proj_out: DeviceBuffer,
    mlp_gate: DeviceBuffer,
    mlp_up: DeviceBuffer,
    mlp_down: DeviceBuffer,
    moe_in: DeviceBuffer,
    moe_out: DeviceBuffer,
    router_in: DeviceBuffer,
    router_logits: DeviceBuffer,
    top_idx: DeviceBuffer,
    top_w: DeviceBuffer,
    freqs: DeviceBuffer,
    expert_gu: DeviceBuffer,
    expert_act: DeviceBuffer,
    expert_out: DeviceBuffer,
    logits: DeviceBuffer,
}

impl Bufs {
    fn new(seq: usize, cfg: &ModelConfig, ctx: &Context) -> Result<Self, Error> {
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
            router_in: a(seq * hidden)?,
            router_logits: a(seq * t.num_experts)?,
            top_idx: a(seq * t.top_k_experts)?,
            top_w: a(seq * t.top_k_experts)?,
            freqs: a(seq * t.global_head_dim)?,
            expert_gu: a(t.moe_intermediate_size * 2)?,
            expert_act: a(t.moe_intermediate_size)?,
            expert_out: a(hidden)?,
            logits: a(t.vocab_size)?,
        })
    }
}

struct Runner<'a> {
    m: &'a GpuModel,
    cfg: &'a ModelConfig,
    seq: usize,
}

impl<'a> Runner<'a> {
    fn t(&self) -> &crate::config::TextConfig {
        &self.cfg.text_config
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
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

    fn rms(
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
    let _ = sc;
    let model = GpuModel::load(w, cfg)?;
    let ctx = model.ctx.clone();
    let t = cfg.text_config.clone();
    let seq = ids.len();
    let hidden = t.hidden_size;
    let mut b = Bufs::new(seq, cfg, &ctx)?;
    let r = Runner {
        m: &model,
        cfg,
        seq,
    };

    // ---- embed gather ----------------------------------------------------
    b.ids.write_bytes(unsafe {
        std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4)
    })?;
    // out[t, d] = embed[id[t], d] * scale, one row at a time (the op's gather is
    // row-wise; the table is huge so this stays O(seq * hidden) of reads).
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
    ctx.synchronize()?;

    let n_layers = layers
        .unwrap_or(t.num_hidden_layers)
        .min(t.num_hidden_layers);
    for layer in 0..n_layers {
        layer_forward(&r, &mut b, &model.layers[layer], layer, 0)?;
        std::mem::swap(&mut b.hidden_a, &mut b.hidden_b);
    }

    r.rms(&b.hidden_a, &model.final_norm, &b.hidden_b, seq)?;

    // tied LM head for the requested position(s)
    let s = match rows {
        LogitRows::Last => seq - 1,
        LogitRows::Only(r) => r,
        LogitRows::All => seq - 1,
    };
    r.gemm(
        1,
        t.vocab_size,
        hidden,
        unsafe { offset_view(&b.hidden_b, s * hidden).device_ptr() },
        model.embed.device_ptr(),
        b.logits.device_ptr(),
    )?;
    let cap = t.final_logit_softcapping as f32;
    if cap > 0.0 {
        let mut args = KernelArgs::new();
        args.device_ptr(b.logits.device_ptr())
            .f32(cap)
            .u32(t.vocab_size as u32);
        launch(&ctx, "dgq_softcap", flat(t.vocab_size, 256), 256, &mut args)?;
    }
    ctx.synchronize()?;
    let mut logits = vec![0.0f32; t.vocab_size];
    b.logits.read_f32(&mut logits)?;
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

fn layer_forward(
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
    let n_heads = t.num_attention_heads;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let ctx = &r.m.ctx;

    // residual = hidden_a
    copy_device(ctx, &b.residual, &b.hidden_a, seq * hidden)?;

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
        .u32(window.unwrap_or(0) as u32);
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
    let mut args = KernelArgs::new();
    args.device_ptr(b.mlp_gate.device_ptr())
        .device_ptr(b.mlp_up.device_ptr())
        .f32(1.0)
        .device_ptr(b.mlp_gate.device_ptr())
        .u32((seq * inter) as u32);
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
    r.rms(
        &b.mlp_down,
        &lw.post_feedforward_layernorm_1,
        &b.mlp_down,
        seq,
    )?;

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
    let mut idx = vec![0u32; seq * t.top_k_experts];
    let mut wts = vec![0.0f32; seq * t.top_k_experts];
    b.top_idx.read_bytes(unsafe {
        std::slice::from_raw_parts_mut(idx.as_mut_ptr().cast::<u8>(), idx.len() * 4)
    })?;
    b.top_w.read_f32(&mut wts)?;

    // zero moe_out, then accumulate each token's experts
    b.moe_out.zero()?;
    let moe_inter = t.moe_intermediate_size;
    let gu_stride = moe_inter * 2 * hidden;
    let down_stride = hidden * moe_inter;
    for s in 0..seq {
        for kk in 0..t.top_k_experts {
            let e = idx[s * t.top_k_experts + kk] as usize;
            let w = wts[s * t.top_k_experts + kk];
            let x = offset_view(&b.moe_in, s * hidden);
            let gu = offset_view(&lw.experts_gate_up, e * gu_stride);
            r.gemm(
                1,
                moe_inter * 2,
                hidden,
                x.device_ptr(),
                gu.device_ptr(),
                b.expert_gu.device_ptr(),
            )?;
            let mut args = KernelArgs::new();
            args.device_ptr(b.expert_gu.device_ptr())
                .device_ptr(unsafe { b.expert_gu.device_ptr() + (moe_inter as u64) * 4 })
                .f32(1.0)
                .device_ptr(b.expert_act.device_ptr())
                .u32(moe_inter as u32);
            launch(
                ctx,
                "dgq_swiglu_weighted",
                flat(moe_inter, 256),
                256,
                &mut args,
            )?;
            let dn = offset_view(&lw.experts_down, e * down_stride);
            r.gemm(
                1,
                hidden,
                moe_inter,
                b.expert_act.device_ptr(),
                dn.device_ptr(),
                b.expert_out.device_ptr(),
            )?;
            let dst = offset_view(&b.moe_out, s * hidden);
            let mut args = KernelArgs::new();
            args.device_ptr(dst.device_ptr())
                .device_ptr(b.expert_out.device_ptr())
                .f32(w)
                .u32(hidden as u32);
            launch(ctx, "dgq_accum", flat(hidden, 256), 256, &mut args)?;
        }
    }
    r.rms(&b.moe_out, &lw.post_feedforward_layernorm_2, &b.normed, seq)?;
    copy_device(ctx, &b.moe_out, &b.normed, seq * hidden)?;
    r.add_in_place(&b.normed, &b.moe_out, seq * hidden)?;

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
    let model = GpuModel::load(w, cfg)?;
    let ctx = model.ctx.clone();
    let t = cfg.text_config.clone();
    let seq = ids.len();
    let hidden = t.hidden_size;
    let mut b = Bufs::new(seq, cfg, &ctx)?;
    let r = Runner {
        m: &model,
        cfg,
        seq,
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
    let model = GpuModel::load(w, cfg)?;
    let ctx = model.ctx.clone();
    let t = cfg.text_config.clone();
    let seq = ids.len();
    let hidden = t.hidden_size;
    let mut b = Bufs::new(seq, cfg, &ctx)?;
    let r = Runner {
        m: &model,
        cfg,
        seq,
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
    let mut rb = vec![0.0f32; b.len()];
    bb.read_f32(&mut rb)?;
    let mut ra = vec![0.0f32; a.len()];
    ba.read_f32(&mut ra)?;
    eprintln!("[probe] a={a:?} read_a={ra:?} b={b:?} read_b={rb:?} m={m} n={n} k={k}");
    let mut out = vec![0.0f32; m * n];
    bc.read_f32(&mut out)?;
    Ok(out)
}
