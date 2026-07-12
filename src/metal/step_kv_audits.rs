//! step_kv audits: KV-cache parity / probe harnesses backing the CLI
//! `step-kv-*` / `step-attn-probe` subcommands. Extracted verbatim from
//! step_kv.rs (backlog item 4). Non-production paths only; the monolithic-KV
//! prefill/extend plumbing stays in the parent module.

use crate::config::ModelConfig;
use crate::dgq::DgqStore;
use crate::metal::device::MetalContext;
use crate::metal::step_kernel::{
    CANVAS, ModelLayout, N_LAYERS, StepFinishMode, StepSmokeConfig, VOCAB, build_layout,
    build_offsets_from_store, build_step_runtime, run_step_forward, step_params_from_sampler,
};
use crate::model::Model;
use crate::safetensors::Error;
use crate::sample::{Rng, SamplerConfig, step_entropy_stats};
use crate::shaders::f16::f16_bits_to_f32;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::path::Path;

use super::*;

// ---- bf16-cross parity ----
#[derive(Debug)]
pub struct StepKvBf16CrossResult {
    pub kv_len: usize,
    pub layers: usize,
    pub max_kv_diff: f32,
    pub max_kv_diff_layer: usize,
    pub max_kv_diff_pos: usize,
    pub gpu_prefix_max_l0: f32,
    pub cpu_prefix_max_l0: f32,
}

/// Compare GPU monolithic encoder KV (.dgq) vs CPU bf16 encoder KV (safetensors).
pub fn run_step_kv_bf16_cross_parity(
    dgq_dir: &Path,
    bf16_dir: &Path,
    token_ids: &[u32],
    layers: usize,
    max_seq: usize,
) -> Result<StepKvBf16CrossResult, Error> {
    if token_ids.is_empty() {
        return Err(Error::Format("kv cross parity requires at least one token"));
    }
    let layers = layers.max(1).min(N_LAYERS);
    let dgq_store = DgqStore::open(dgq_dir)?;
    let layout = build_layout(&build_offsets_from_store(&dgq_store), max_seq);
    let bf16_model = Model::open(bf16_dir)?;
    if bf16_model.weights.is_quantized() {
        return Err(Error::Format(
            "bf16 cross parity requires bf16 safetensors dir",
        ));
    }
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
    let alloc_kv = || -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        ctx.device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Format("kv cross buffer alloc failed"))
    };
    let gpu_buf = alloc_kv()?;
    let cpu_buf = alloc_kv()?;
    let mut gpu_cache = MonolithicEncoderCache::open_opt(dgq_dir, CANVAS, max_seq, None)?;
    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut gpu_cache,
        token_ids,
        &gpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    let cpu_kv_len = prefill_monolithic_kv_cpu(
        &bf16_model.weights,
        &bf16_model.config,
        token_ids,
        &cpu_buf,
        &layout,
        max_seq,
    )?;
    if cpu_kv_len != kv_len {
        return Err(Error::Format("gpu/cpu bf16 cross kv_len mismatch"));
    }
    let (max_kv_diff, max_kv_diff_layer, max_kv_diff_pos) =
        monolithic_kv_prefix_max_diff(&gpu_buf, &cpu_buf, &layout, kv_len, layers);
    Ok(StepKvBf16CrossResult {
        kv_len,
        layers,
        max_kv_diff,
        max_kv_diff_layer,
        max_kv_diff_pos,
        gpu_prefix_max_l0: kvcache_prefix_max_abs(&gpu_buf, &layout, 0, kv_len),
        cpu_prefix_max_l0: kvcache_prefix_max_abs(&cpu_buf, &layout, 0, kv_len),
    })
}

// ---- kv-audit / attn-probe / kv-parity / encoder-moe ----
/// Max abs half value in layer `L` prefix `[0, kv_len)`.
pub fn kvcache_prefix_max_abs(
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    layer: usize,
    kv_len: usize,
) -> f32 {
    if kv_len == 0 || layer >= N_LAYERS {
        return 0.0;
    }
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let token_stride = nkv * hd * 2;
    let byte_base = l.kv_region as usize;
    let slot_base = byte_base / 2;
    let mut max_abs = 0.0f32;
    for pos in 0..kv_len {
        let start = (slot_base + pos * token_stride) * 2;
        let end = start + token_stride * 2;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                kv_buf.contents().as_ptr().add(start) as *const u8,
                end - start,
            )
        };
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            max_abs = max_abs.max(f16_bits_to_f32(bits).abs());
        }
    }
    max_abs
}

#[derive(Debug)]
pub struct StepKvAuditResult {
    pub kv_len: usize,
    pub prefix_max_abs_l0: f32,
    pub hidden_max_abs_vs_zero: f32,
    pub logits_max_abs_vs_zero: f32,
    pub extend_kv_len: Option<usize>,
    pub extend_hidden_diff: Option<f32>,
    pub pass: bool,
}

/// M1.4: verify b4 prefix is populated and forward changes vs kv_len=0.
pub fn run_step_kv_audit(
    model_dir: &Path,
    kv_len: usize,
    layers: usize,
    seed: u64,
    max_seq: usize,
) -> Result<StepKvAuditResult, Error> {
    let vocab = crate::metal::step_kernel::VOCAB;
    let mut prompt = vec![0u32; kv_len];
    for (i, id) in prompt.iter_mut().enumerate() {
        *id = ((i * 131 + 7) % vocab.max(1)) as u32;
    }
    let base_cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    let zero = run_step_forward(model_dir, &base_cfg)?;
    let mut kv_cfg = base_cfg.clone();
    kv_cfg.prefill_token_ids = Some(prompt.clone());
    let with_kv = run_step_forward(model_dir, &kv_cfg)?;

    let store = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
    let kv_buf = ctx
        .device
        .newBufferWithLength_options(kv_bytes, objc2_metal::MTLResourceOptions::StorageModeShared)
        .ok_or(Error::Format("kv audit buffer alloc failed"))?;
    let actual_kv = prefill_monolithic_kv(model_dir, &prompt, &kv_buf, &layout, max_seq, layers)?;
    let prefix_max = kvcache_prefix_max_abs(&kv_buf, &layout, 0, actual_kv);

    let hidden_diff = zero
        .norm_hidden
        .iter()
        .zip(with_kv.norm_hidden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let logits_diff = zero
        .logits
        .iter()
        .zip(with_kv.logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mut extend_kv_len = None;
    let mut extend_hidden_diff = None;
    let mut pass = prefix_max > 1e-4 && hidden_diff > 1e-3;

    if pass && kv_len >= 32 {
        let prefill_len = kv_len / 2;
        let extend_tokens = prompt[prefill_len..].to_vec();
        let prefill_only = &prompt[..prefill_len];
        let extend_buf = ctx
            .device
            .newBufferWithLength_options(
                kv_bytes,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .ok_or(Error::Format("kv extend audit buffer alloc failed"))?;
        let prefill_actual = prefill_monolithic_kv(
            model_dir,
            prefill_only,
            &extend_buf,
            &layout,
            max_seq,
            layers,
        )?;
        let extended = extend_monolithic_kv(
            model_dir,
            &extend_buf,
            &layout,
            prefill_actual,
            &extend_tokens,
            max_seq,
            layers,
        )?;
        extend_kv_len = Some(extended);
        if extended != kv_len {
            pass = false;
        } else {
            let mut prefill_cfg = base_cfg.clone();
            prefill_cfg.prefill_token_ids = Some(prefill_only.to_vec());
            let half_kv = run_step_forward(model_dir, &prefill_cfg)?;
            let ext_diff = half_kv
                .norm_hidden
                .iter()
                .zip(with_kv.norm_hidden.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            extend_hidden_diff = Some(ext_diff);
            if ext_diff <= 1e-3 {
                pass = false;
            }
        }
    }

    Ok(StepKvAuditResult {
        kv_len,
        prefix_max_abs_l0: prefix_max,
        hidden_max_abs_vs_zero: hidden_diff,
        logits_max_abs_vs_zero: logits_diff,
        extend_kv_len,
        extend_hidden_diff,
        pass,
    })
}

fn copy_metal_buffer(
    dst: &ProtocolObject<dyn MTLBuffer>,
    src: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.contents().as_ptr() as *const u8,
            dst.contents().as_ptr() as *mut u8,
            len,
        );
    }
}

/// Max abs f32 diff between two monolithic b4 prefixes `[0, kv_len)` over `layers`.
pub fn monolithic_kv_prefix_max_diff(
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    kv_len: usize,
    layers: usize,
) -> (f32, usize, usize) {
    let mut max_diff = 0.0f32;
    let mut max_layer = 0usize;
    let mut max_pos = 0usize;
    for layer in 0..layers.min(N_LAYERS) {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let token_stride = nkv * hd * 2;
        let slot_base = l.kv_region as usize / 2;
        for pos in 0..kv_len {
            for hidx in 0..token_stride {
                let byte = (slot_base + pos * token_stride + hidx) * 2;
                let va = f16_bits_to_f32(read_half_at(a, byte));
                let vb = f16_bits_to_f32(read_half_at(b, byte));
                let d = (va - vb).abs();
                if d > max_diff {
                    max_diff = d;
                    max_layer = layer;
                    max_pos = pos;
                }
            }
        }
    }
    (max_diff, max_layer, max_pos)
}

/// One denoise step with an external b4 KV prefix; returns min position entropy (nats).
pub fn step_min_entropy_with_kv(
    model_dir: &Path,
    kv_src: &ProtocolObject<dyn MTLBuffer>,
    kv_len: usize,
    layers: usize,
    max_seq: usize,
    seed: u64,
) -> Result<f32, Error> {
    let cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: kv_len as u32,
        seed,
        max_seq,
        finish: StepFinishMode::Full,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    let (mut rt, _) = build_step_runtime(model_dir, &cfg)?;
    let kv_bytes = kv_cache_total_bytes(rt.layout(), max_seq) as usize;
    copy_metal_buffer(rt.kvcache(), kv_src, kv_bytes);
    rt.set_kv_len(kv_len as u32);
    let params = step_params_from_sampler(&SamplerConfig::default(), kv_len as u32, false, 1);
    let mut rng = Rng::new(seed);
    rt.reset_block(VOCAB, &mut rng, params);
    rt.run_denoise_step()?;
    let st = rt.read_canvas_state();
    Ok(step_entropy_stats(&st.entropy, &st.accept).min_entropy)
}

/// Max abs half in K or V plane only for layer `L` prefix `[0, kv_len)`.
pub fn kvcache_plane_max_abs(
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    layer: usize,
    kv_len: usize,
    plane: u8,
) -> f32 {
    if kv_len == 0 || layer >= N_LAYERS || plane > 1 {
        return 0.0;
    }
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let plane_slots = nkv * hd;
    let token_stride = plane_slots * 2;
    let slot_base = l.kv_region as usize / 2;
    let plane_off = if plane == 0 { 0 } else { plane_slots };
    let mut max_abs = 0.0f32;
    for pos in 0..kv_len {
        let start = (slot_base + pos * token_stride + plane_off) * 2;
        let end = start + plane_slots * 2;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                kv_buf.contents().as_ptr().add(start) as *const u8,
                end - start,
            )
        };
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            max_abs = max_abs.max(f16_bits_to_f32(bits).abs());
        }
    }
    max_abs
}

#[derive(Debug)]
pub struct StepAttnProbeResult {
    pub kv_len: usize,
    pub layer: usize,
    pub k_plane_max_l0: f32,
    pub v_plane_max_l0: f32,
    pub q_norm_weight_mean_abs: f32,
    pub q_norm_weight_rms: f32,
    pub k_norm_weight_mean_abs: f32,
    pub k_norm_weight_rms: f32,
    pub cpu_raw_dot_max: f32,
    pub cpu_raw_dot_min: f32,
    pub cpu_mean_softmax_entropy: f32,
    pub cpu_mean_max_prob: f32,
    pub cpu_q_head_rms: f32,
    pub cpu_k_head_rms: f32,
    pub canvas_len: usize,
    pub attn_keys_t: usize,
    pub cpu_keys_per_row: usize,
    pub cpu_mean_weight_sum: f32,
}

/// P1.0: attention magnitude + KV plane probe (monolithic b4 + CPU score model with real QK-norm weights).
pub fn run_step_attn_probe(
    model_dir: &Path,
    token_ids: &[u32],
    layer: usize,
    seed: u64,
    max_seq: usize,
) -> Result<StepAttnProbeResult, Error> {
    use crate::model::attention::{
        AttentionParams, attn_score_stats_decoder, qk_norm_weight_stats,
    };
    use crate::model::mask::DecoderAttnMask;
    use crate::sample::Rng;
    use crate::shaders::cpu::rms_norm;

    if token_ids.is_empty() {
        return Err(Error::Format("step-attn-probe requires prompt tokens"));
    }
    let layer = layer.min(N_LAYERS.saturating_sub(1));
    let canvas_len = CANVAS;
    let cfg = ModelConfig::load(model_dir)?;
    let text = &cfg.text_config;
    let dgq = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&dgq), max_seq);
    let params = AttentionParams::for_layer(text, layer)?;

    let native_buf = MetalContext::new()?
        .device
        .newBufferWithLength_options(
            kv_cache_total_bytes(&layout, max_seq) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or(Error::Format("kv buffer alloc failed"))?;
    let mut native_cache = MonolithicEncoderCache::open_opt(model_dir, canvas_len, max_seq, None)?;
    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut native_cache,
        token_ids,
        &native_buf,
        &layout,
        max_seq,
        layer + 1,
    )?;

    let p = format!("model.decoder.layers.{layer}.self_attn");
    let q_norm_w = dgq.tensor_f32(&format!("{p}.q_norm.weight"))?;
    let k_norm_w = dgq.tensor_f32(&format!("{p}.k_norm.weight"))?;
    let (q_na, q_nr) = qk_norm_weight_stats(&q_norm_w);
    let (k_na, k_nr) = qk_norm_weight_stats(&k_norm_w);

    // Build synthetic post-QK-norm Q/K from monolithic K-cache prefix + canvas-sized query grid.
    let hd = params.head_dim;
    let nkv = params.n_kv_heads;
    let nheads = params.n_heads;
    let total_kv = kv_len + canvas_len;
    let kv_dim = nkv * hd;
    let q_dim = nheads * hd;
    let l = &layout.layers[layer];
    let slot_base = l.kv_region as usize / 2;
    let token_stride = nkv * hd * 2;

    let mut k_full = vec![0.0f32; total_kv * kv_dim];
    let mut v_full = vec![0.0f32; total_kv * kv_dim];
    for pos in 0..kv_len {
        let k_off = pos * kv_dim;
        let byte_k = (slot_base + pos * token_stride) * 2;
        let byte_v = byte_k + nkv * hd * 2;
        for i in 0..nkv * hd {
            k_full[k_off + i] = f16_bits_to_f32(read_half_at(&native_buf, byte_k + i * 2));
            v_full[k_off + i] = f16_bits_to_f32(read_half_at(&native_buf, byte_v + i * 2));
        }
    }

    let mut rng = Rng::new(seed);
    let mut q = vec![0.0f32; canvas_len * q_dim];
    let mut k_canvas = vec![0.0f32; canvas_len * kv_dim];
    let eps = text.rms_norm_eps as f32;
    let mut head = vec![0.0f32; hd];
    for tok in 0..canvas_len {
        for h in 0..nheads {
            let off = (tok * nheads + h) * hd;
            for d in 0..hd {
                head[d] = (rng.next_f32() - 0.5) * 2.0;
            }
            rms_norm(&mut q[off..off + hd], &head, &q_norm_w, eps);
        }
        for h in 0..nkv {
            let off = (tok * nkv + h) * hd;
            for d in 0..hd {
                head[d] = (rng.next_f32() - 0.5) * 2.0;
            }
            rms_norm(&mut k_canvas[off..off + hd], &head, &k_norm_w, eps);
        }
    }
    k_full[kv_len * kv_dim..].copy_from_slice(&k_canvas);

    let mask = DecoderAttnMask::all_valid(canvas_len, kv_len);
    let stats = attn_score_stats_decoder(&q, &k_full, canvas_len, total_kv, &params, &mask);
    let q_rms = head_rms_probe(&q, canvas_len, nheads, hd);
    let k_rms = head_rms_probe(&k_full, total_kv, nkv, hd);

    Ok(StepAttnProbeResult {
        kv_len,
        layer,
        k_plane_max_l0: kvcache_plane_max_abs(&native_buf, &layout, 0, kv_len, 0),
        v_plane_max_l0: kvcache_plane_max_abs(&native_buf, &layout, 0, kv_len, 1),
        q_norm_weight_mean_abs: q_na,
        q_norm_weight_rms: q_nr,
        k_norm_weight_mean_abs: k_na,
        k_norm_weight_rms: k_nr,
        cpu_raw_dot_max: stats.raw_dot_max,
        cpu_raw_dot_min: stats.raw_dot_min,
        cpu_mean_softmax_entropy: stats.mean_row_softmax_entropy,
        cpu_mean_max_prob: stats.mean_row_max_prob,
        cpu_q_head_rms: q_rms,
        cpu_k_head_rms: k_rms,
        canvas_len,
        attn_keys_t: total_kv,
        cpu_keys_per_row: stats.keys_per_row,
        cpu_mean_weight_sum: stats.mean_weight_sum,
    })
}

fn head_rms_probe(buf: &[f32], seq_len: usize, n_heads: usize, head_dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let n = (seq_len * n_heads).max(1) as f32;
    for s in 0..seq_len {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                let v = buf[off + d];
                ss += v * v;
            }
            sum += (ss / head_dim as f32).sqrt();
        }
    }
    sum / n
}

#[derive(Debug)]
pub struct StepKvParityResult {
    pub kv_len: usize,
    pub layers: usize,
    pub prefix_max_l0: f32,
    pub prefix_max_l0_b: f32,
    pub max_kv_diff: f32,
    pub max_kv_diff_layer: usize,
    pub max_kv_diff_pos: usize,
    pub min_ent: f32,
    pub min_ent_b: f32,
    pub min_ent_diff: f32,
    pub entropy_pass: bool,
    pub ln_vocab: f32,
    pub pass: bool,
}

/// Compare monolithic b4 KV from two independent encoder prefills, then one denoise step each.
pub fn run_step_kv_parity(
    model_dir: &Path,
    token_ids: &[u32],
    layers: usize,
    max_seq: usize,
    seed: u64,
) -> Result<StepKvParityResult, Error> {
    if token_ids.is_empty() {
        return Err(Error::Format("step-kv-parity requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Format("token_ids exceed max_seq"));
    }
    let layers = layers.max(1).min(N_LAYERS);
    let store = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;

    let alloc_kv = || -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        ctx.device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Format("kv parity buffer alloc failed"))
    };

    let mut cache_a = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;
    let mut cache_b = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;

    let buf_a = alloc_kv()?;
    let buf_b = alloc_kv()?;

    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut cache_a,
        token_ids,
        &buf_a,
        &layout,
        max_seq,
        layers,
    )?;
    let (kv_len_b, _) = prefill_monolithic_kv_with_cache(
        &mut cache_b,
        token_ids,
        &buf_b,
        &layout,
        max_seq,
        layers,
    )?;
    if kv_len_b != kv_len {
        return Err(Error::Format("encoder prefill kv_len mismatch"));
    }

    let prefix_max_l0 = kvcache_prefix_max_abs(&buf_a, &layout, 0, kv_len);
    let prefix_max_l0_b = kvcache_prefix_max_abs(&buf_b, &layout, 0, kv_len);
    let (max_kv_diff, max_kv_diff_layer, max_kv_diff_pos) =
        monolithic_kv_prefix_max_diff(&buf_a, &buf_b, &layout, kv_len, layers);

    let min_ent = step_min_entropy_with_kv(model_dir, &buf_a, kv_len, layers, max_seq, seed)?;
    let min_ent_b = step_min_entropy_with_kv(model_dir, &buf_b, kv_len, layers, max_seq, seed)?;

    let ln_vocab = (VOCAB as f32).ln();
    let min_ent_diff = (min_ent_b - min_ent).abs();
    let kv_ok = max_kv_diff < 0.5 && prefix_max_l0_b > 1e-4;
    const MAX_MIN_ENT_DIFF: f32 = 0.25;
    let entropy_pass = min_ent_diff < MAX_MIN_ENT_DIFF;
    let pass = kv_ok && entropy_pass;

    Ok(StepKvParityResult {
        kv_len,
        layers,
        prefix_max_l0,
        prefix_max_l0_b,
        max_kv_diff,
        max_kv_diff_layer,
        max_kv_diff_pos,
        min_ent,
        min_ent_b,
        min_ent_diff,
        entropy_pass,
        ln_vocab,
        pass,
    })
}

#[cfg(test)]
/// Compare monolithic b4 KV from encoder prefill with CPU MoE vs grouped GPU MoE (same dense path).
pub fn run_encoder_moe_kv_parity(
    model_dir: &Path,
    token_ids: &[u32],
    layers: usize,
    max_seq: usize,
) -> Result<(f32, usize, usize), Error> {
    if token_ids.is_empty() {
        return Err(Error::Format(
            "encoder moe kv parity requires at least one token",
        ));
    }
    let layers = layers.max(1).min(N_LAYERS);
    let store = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;

    let alloc_kv = || -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        ctx.device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Format("encoder moe kv buffer alloc failed"))
    };

    let mut cpu_cache = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;
    cpu_cache.engine.set_encoder_gpu_moe(false);
    let mut gpu_cache = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;
    gpu_cache.engine.set_encoder_gpu_moe(true);

    let cpu_buf = alloc_kv()?;
    let gpu_buf = alloc_kv()?;

    let (cpu_kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut cpu_cache,
        token_ids,
        &cpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    let (gpu_kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut gpu_cache,
        token_ids,
        &gpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    if gpu_kv_len != cpu_kv_len {
        return Err(Error::Format("cpu/gpu encoder moe prefill kv_len mismatch"));
    }
    let (max_diff, layer, pos) =
        monolithic_kv_prefix_max_diff(&cpu_buf, &gpu_buf, &layout, cpu_kv_len, layers);
    Ok((max_diff, layer, pos))
}
