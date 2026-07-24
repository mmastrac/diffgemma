//! step_kernel diagnostics: probe/capture harnesses (hidden-state, preamble, embed, attention) that back the
//! CLI `step-*` debug subcommands and the `step-smoke` gate. Extracted verbatim
//! from step_kernel.rs (backlog item 4). Non-production paths only; the hot
//! per-step engine stays in the parent module.

use crate::Error;
use crate::dgq::DgqStore;
use crate::sample::Rng;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::path::Path;
use std::time::Instant;

use super::*;

pub fn run_step_probe(model_dir: &Path, cfg: StepSmokeConfig) -> Result<StepProbeResult, Error> {
    let started = Instant::now();
    let (mut rt, _) = build_step_runtime(model_dir, &cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;
    let mut checkpoints = Vec::new();

    let mut push = |label: &str, finite: bool, max_abs: f32| {
        checkpoints.push(StepProbeCheckpoint {
            label: label.to_string(),
            finite,
            max_abs,
        });
    };

    rt.dispatch_and_wait(|enc| {
        let first_step = 1u32;
        enc.encode_step_preamble(&layout, first_step)?;
        Ok(())
    })?;
    let (f, m, _n) = arena_hidden_stats(&rt.bufs.arena, &rt.bufs.arena_map);
    push("after_preamble", f, m);

    for layer in 0..layers {
        rt.encode_full_layer(layer)?;
        let (f, m, _n) = arena_hidden_stats(&rt.bufs.arena, &rt.bufs.arena_map);
        push(&format!("after_layer_{layer}"), f, m);
    }

    rt.dispatch_and_wait(|enc| {
        enc.rmsnorm(
            enc.arena().hidden_off(),
            enc.arena().tmp_off(),
            layout.final_norm,
            HID as u32,
            CANVAS,
        );
        enc.gemm_q8_logits(
            enc.arena().tmp_off(),
            layout.embed,
            CANVAS as u32,
            VOCAB as u32,
            HID as u32,
            0,
        )?;
        enc.dispatch_softcap();
        Ok(())
    })?;
    let (bad, m) = count_non_finite_half(&rt.bufs.logits, CANVAS * VOCAB);
    push("after_lm_head_softcap", bad == 0, m);

    Ok(StepProbeResult {
        checkpoints,
        elapsed: started.elapsed(),
    })
}

#[derive(Debug, Clone)]
pub struct LayerHiddenProbeCheckpoint {
    pub label: String,
    pub layer: Option<usize>,
    pub hidden: Vec<f32>,
    pub hidden_l2: f32,
    pub hidden_max_abs: f32,
}

#[derive(Debug)]
pub struct LayerHiddenProbeResult {
    pub position: usize,
    /// Denoise steps run before the instrumented forward (0 = seeded canvas).
    pub warm_steps: usize,
    pub canvas_token: u32,
    pub token_ids: Vec<u32>,
    pub checkpoints: Vec<LayerHiddenProbeCheckpoint>,
}

pub(super) fn read_arena_hidden_row(
    arena: &ProtocolObject<dyn MTLBuffer>,
    base: u64,
    row: usize,
) -> Vec<f32> {
    read_arena_row(arena, base, row, HID)
}

pub(super) fn read_arena_row(
    arena: &ProtocolObject<dyn MTLBuffer>,
    base: u64,
    row: usize,
    width: usize,
) -> Vec<f32> {
    let byte_off = base as usize + row * width * 2;
    read_arena_buffer_f32(arena, byte_off, width)
}

fn hidden_vec_stats(v: &[f32]) -> (f32, f32) {
    let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    (l2, max_abs)
}

/// Per-layer hidden readback at one canvas row (for MLX parity).
///
/// `warm_steps` denoise steps run BEFORE the instrumented forward, so the
/// canvas can be probed once it has actually resolved rather than while it is
/// still seeded noise. 0 (the default everywhere except an explicit
/// `--warm-steps`) reproduces the original step-1 behaviour exactly.
pub fn run_step_layer_hidden_probe(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    position: usize,
    warm_steps: usize,
) -> Result<LayerHiddenProbeResult, Error> {
    if position >= CANVAS {
        return Err(Error::Runtime("layer probe position out of range"));
    }
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    // build_step_runtime seeds the canvas BEFORE the fast prefill runs, and the
    // prefill chunks stream the prompt through the same ids plane — at >256
    // prompt tokens the probe would otherwise measure a canvas of prefill
    // leftovers (last chunk's tokens + zero padding) instead of the seeded
    // canvas. Re-seed in PRODUCTION order (prefill → reset_block → denoise);
    // a fresh Rng reproduces the exact build-time draw, so short/engine-prefill
    // probes are unchanged (idempotent).
    let params = rt.read_params();
    let mut rng = Rng::new(cfg.seed);
    rt.reset_block(VOCAB, &mut rng, params);
    let layout = rt.layout;
    let layers = rt.layers;
    let mut checkpoints = Vec::new();

    // Advance the canvas to the state we actually want to inspect. Each call
    // is a full denoise step (forward + sampler + commit), so after N the
    // canvas holds what the model has settled on, not the seeded noise.
    for _ in 0..warm_steps {
        rt.run_denoise_step()?;
    }
    let step_index = warm_steps as u32 + 1;

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, step_index)?;
        Ok(())
    })?;
    {
        let hidden =
            read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);
        let (hidden_l2, hidden_max_abs) = hidden_vec_stats(&hidden);
        checkpoints.push(LayerHiddenProbeCheckpoint {
            label: "after_preamble".into(),
            layer: None,
            hidden,
            hidden_l2,
            hidden_max_abs,
        });
    }

    for layer in 0..layers {
        rt.encode_full_layer(layer)?;
        let hidden =
            read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);
        let (hidden_l2, hidden_max_abs) = hidden_vec_stats(&hidden);
        checkpoints.push(LayerHiddenProbeCheckpoint {
            label: format!("after_layer_{layer}"),
            layer: Some(layer),
            hidden,
            hidden_l2,
            hidden_max_abs,
        });
    }

    rt.dispatch_and_wait(|enc| {
        enc.rmsnorm(
            enc.arena().hidden_off(),
            enc.arena().tmp_off(),
            layout.final_norm,
            HID as u32,
            CANVAS,
        );
        Ok(())
    })?;
    {
        let hidden = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.tmp_off(), position);
        let (hidden_l2, hidden_max_abs) = hidden_vec_stats(&hidden);
        checkpoints.push(LayerHiddenProbeCheckpoint {
            label: "after_final_norm".into(),
            layer: Some(layers),
            hidden,
            hidden_l2,
            hidden_max_abs,
        });
    }

    let state: CanvasState = read_struct(&rt.bufs.state);
    Ok(LayerHiddenProbeResult {
        position,
        warm_steps,
        canvas_token: state.ids[position],
        token_ids: state.ids.to_vec(),
        checkpoints,
    })
}

#[derive(Debug, Clone)]
pub struct PreambleCapture {
    pub position: usize,
    pub canvas_token: u32,
    pub token_ids: Vec<u32>,
    pub kv_len: u32,
    pub embed_scaled: Vec<f32>,
    pub after_preamble: Vec<f32>,
}

/// Step-1 preamble hidden at one canvas row (embed gather + no-scale RMSNorm).
pub fn run_step_preamble_capture(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    position: usize,
) -> Result<PreambleCapture, Error> {
    if position >= CANVAS {
        return Err(Error::Runtime("preamble capture position out of range"));
    }
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;

    rt.dispatch_and_wait(|enc| enc.encode_preamble_embed_only(&layout))?;
    let embed_scaled =
        read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);

    rt.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, 1))?;
    let after_preamble =
        read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);

    let state = rt.read_canvas_state();
    Ok(PreambleCapture {
        position,
        canvas_token: state.ids[position],
        token_ids: state.ids.to_vec(),
        kv_len: rt.read_params().kv_len,
        embed_scaled,
        after_preamble,
    })
}

/// GPU `k_embed_gather` for a canvas filled uniformly with `token` (row 0 readback).
pub fn run_embed_row_gpu(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    token: u32,
) -> Result<Vec<f32>, Error> {
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let mut state = rt.read_canvas_state();
    state.ids.fill(token);
    rt.write_canvas_state(&state);
    let layout = rt.layout;
    rt.dispatch_and_wait(|enc| enc.encode_preamble_embed_only(&layout))?;
    Ok(read_arena_hidden_row(
        &rt.bufs.arena,
        rt.bufs.arena_map.hidden_off(),
        0,
    ))
}

/// Query heads in the monolithic step-kernel shader (`NQ_HEADS`).
pub const STEP_NQ_HEADS: usize = 16;

#[derive(Debug, Clone)]
pub struct LayerAttnCapture {
    pub layer: usize,
    pub position: usize,
    pub canvas_token: u32,
    pub token_ids: Vec<u32>,
    pub kv_len: u32,
    pub total_kv: usize,
    pub head_dim: u32,
    pub n_heads: usize,
    pub n_kv_heads: u32,
    pub is_full: bool,
    pub hidden_in: Vec<f32>,
    pub hidden_ln: Vec<f32>,
    pub q_raw_proj: Vec<f32>,
    pub q_pre_rope: Vec<f32>,
    pub q_post_rope: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub raw_scores: Vec<f32>,
    pub attn_probs: Vec<f32>,
    pub k_samples: Vec<(usize, Vec<f32>)>,
}

fn row_raw_scores(
    q: &[f32],
    k: &[f32],
    qi: usize,
    total_kv: usize,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; n_heads * total_kv];
    let groups = n_heads / n_kv.max(1);
    for h in 0..n_heads {
        let kvh = h / groups.max(1);
        let q_off = qi * n_heads * hd + h * hd;
        for t in 0..total_kv {
            let k_off = t * n_kv * hd + kvh * hd;
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[q_off + d] * k[k_off + d];
            }
            scores[h * total_kv + t] = dot;
        }
    }
    scores
}

fn softmax_attn_rows(scores: &[f32], n_heads: usize, total_kv: usize) -> Vec<f32> {
    let mut probs = vec![0.0f32; scores.len()];
    for h in 0..n_heads {
        let base = h * total_kv;
        let row = &scores[base..base + total_kv];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = row.iter().map(|&s| (s - max).exp()).collect::<Vec<_>>();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            for v in &mut exps {
                *v /= sum;
            }
        }
        probs[base..base + total_kv].copy_from_slice(&exps);
    }
    probs
}

fn top_key_positions(probs: &[f32], n_heads: usize, total_kv: usize, k: usize) -> Vec<usize> {
    let mut mass = vec![0.0f32; total_kv];
    for h in 0..n_heads {
        let base = h * total_kv;
        for t in 0..total_kv {
            mass[t] += probs[base + t];
        }
    }
    let mut order: Vec<usize> = (0..total_kv).collect();
    order.sort_by(|&a, &b| {
        mass[b]
            .partial_cmp(&mass[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    order.into_iter().take(k).collect()
}

fn k_vec_at(k: &[f32], t: usize, n_kv: usize, hd: usize) -> Vec<f32> {
    let off = t * n_kv * hd;
    k[off..off + n_kv * hd].to_vec()
}

/// Step-1 forward through `layer` attention; read Q/K/scores/attn_out at one canvas row.
pub fn run_step_attn_layer_capture(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    layer: usize,
    position: usize,
) -> Result<LayerAttnCapture, Error> {
    use crate::metal::step_kv::read_layer_k_cache_f32;

    if position >= CANVAS {
        return Err(Error::Runtime("attn capture position out of range"));
    }
    let layer = layer.min(cfg.layers.saturating_sub(1));
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
        Ok(())
    })?;
    for l in 0..layer {
        rt.encode_full_layer(l)?;
    }

    let hidden_in = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_qkv_gemm(layer, &layout)?;
        Ok(())
    })?;

    let l = &layout.layers[layer];
    let hd = l.head_dim as usize;
    let nkv = l.n_kv_heads as usize;
    let n_heads = STEP_NQ_HEADS;
    let q_width = n_heads * hd;
    let kv_len = rt.read_params().kv_len;
    let total_kv = kv_len as usize + CANVAS;

    let hidden_ln = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.tmp_off(), position);
    let q_raw_proj = read_arena_row(
        &rt.bufs.arena,
        rt.bufs.arena_map.attnq_off(),
        position,
        q_width,
    );
    let q_norm_w = DgqStore::open(model_dir)?.tensor_f32(&format!(
        "model.decoder.layers.{layer}.self_attn.q_norm.weight"
    ))?;
    let mut q_pre_rope = q_raw_proj.clone();
    for h in 0..n_heads {
        let off = h * hd;
        crate::shaders::cpu::attention::rms_norm_head(
            &mut q_pre_rope[off..off + hd],
            Some(&q_norm_w),
            crate::shaders::cpu::attention::RMS_EPS,
        );
    }

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_qk_rope_and_attention(layer, &layout)?;
        Ok(())
    })?;

    let q_all = read_arena_buffer_f32(
        &rt.bufs.arena,
        rt.bufs.arena_map.attnq_off() as usize,
        CANVAS * q_width,
    );
    let q_post_rope = read_arena_row(
        &rt.bufs.arena,
        rt.bufs.arena_map.attnq_off(),
        position,
        q_width,
    );
    let attn_out = read_arena_row(
        &rt.bufs.arena,
        rt.bufs.arena_map.attno_off(),
        position,
        q_width,
    );
    let k_cache = read_layer_k_cache_f32(rt.kvcache(), &layout, layer, total_kv);

    let raw_scores = row_raw_scores(&q_all, &k_cache, position, total_kv, n_heads, nkv, hd);
    let attn_probs = softmax_attn_rows(&raw_scores, n_heads, total_kv);

    let canvas_abs = kv_len as usize + position;
    let mut sample_pos = vec![
        0usize,
        kv_len.saturating_sub(1) as usize,
        kv_len as usize,
        canvas_abs,
    ];
    for t in top_key_positions(&attn_probs, n_heads, total_kv, 8) {
        sample_pos.push(t);
    }
    sample_pos.sort_unstable();
    sample_pos.dedup();
    let k_samples = sample_pos
        .into_iter()
        .map(|t| (t, k_vec_at(&k_cache, t, nkv, hd)))
        .collect();

    let state = rt.read_canvas_state();
    Ok(LayerAttnCapture {
        layer,
        position,
        canvas_token: state.ids[position],
        token_ids: state.ids.to_vec(),
        kv_len,
        total_kv,
        head_dim: l.head_dim,
        n_heads,
        n_kv_heads: l.n_kv_heads,
        is_full: l.is_full != 0,
        hidden_in,
        hidden_ln,
        q_raw_proj,
        q_pre_rope,
        q_post_rope,
        attn_out,
        raw_scores,
        attn_probs,
        k_samples,
    })
}

/// Dump the FULL step-1 canvas Q plane (post-RoPE, all 256 rows × 16 heads
/// × hd) and the layer's complete K cache, as raw f32 binaries + a meta json,
/// for offline block-mass analysis. Same capture flow as
/// `run_step_attn_layer_capture` but skips the per-row CPU score math — the
/// planes are the product. Returns (kv_len, total_kv).
pub fn run_step_attn_qk_plane_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    layer: usize,
    out_dir: &Path,
) -> Result<(u32, usize), Error> {
    use crate::metal::step_kv::read_layer_k_cache_f32;

    let layer = layer.min(cfg.layers.saturating_sub(1));
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
        Ok(())
    })?;
    for l in 0..layer {
        rt.encode_full_layer(l)?;
    }
    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_qkv_gemm(layer, &layout)?;
        Ok(())
    })?;
    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_qk_rope_and_attention(layer, &layout)?;
        Ok(())
    })?;

    let l = &layout.layers[layer];
    let hd = l.head_dim as usize;
    let nkv = l.n_kv_heads as usize;
    let q_width = STEP_NQ_HEADS * hd;
    let kv_len = rt.read_params().kv_len;
    let total_kv = kv_len as usize + CANVAS;

    let q_all = read_arena_buffer_f32(
        &rt.bufs.arena,
        rt.bufs.arena_map.attnq_off() as usize,
        CANVAS * q_width,
    );
    let k_cache = read_layer_k_cache_f32(rt.kvcache(), &layout, layer, total_kv);

    std::fs::create_dir_all(out_dir).map_err(Error::Io)?;
    let write_f32 = |name: String, v: &[f32]| -> Result<(), Error> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        std::fs::write(out_dir.join(name), bytes).map_err(Error::Io)
    };
    write_f32(format!("q_L{layer}_kv{kv_len}.f32"), &q_all)?;
    write_f32(format!("k_L{layer}_kv{kv_len}.f32"), &k_cache)?;
    let is_full = l.is_full != 0;
    let meta = format!(
        "{{\"layer\":{layer},\"kv_len\":{kv_len},\"total_kv\":{total_kv},\
         \"canvas\":{CANVAS},\"n_heads\":{STEP_NQ_HEADS},\"n_kv_heads\":{nkv},\
         \"head_dim\":{hd},\"is_full\":{is_full}}}"
    );
    std::fs::write(out_dir.join(format!("meta_L{layer}_kv{kv_len}.json")), meta)
        .map_err(Error::Io)?;
    Ok((kv_len, total_kv))
}
