//! step_kernel diagnostics: probe / capture / bench harnesses that back the
//! CLI `step-*` debug subcommands and the `step-smoke` gate. Extracted verbatim
//! from step_kernel.rs (backlog item 4). Non-production paths only; the hot
//! per-step engine stays in the parent module.

use crate::Error;
use crate::dgq::DgqStore;
use crate::metal::step_quant::MoeExecutionStyle;
use crate::model::moe::RouteResult;
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
    pub canvas_token: u32,
    pub token_ids: Vec<u32>,
    pub checkpoints: Vec<LayerHiddenProbeCheckpoint>,
}

fn read_arena_hidden_row(arena: &ProtocolObject<dyn MTLBuffer>, base: u64, row: usize) -> Vec<f32> {
    read_arena_row(arena, base, row, HID)
}

fn read_arena_row(
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

/// Step-1 forward with per-layer hidden readback at one canvas row (for MLX parity).
pub fn run_step_layer_hidden_probe(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    position: usize,
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

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
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

#[derive(Debug, Clone)]
pub struct LayerMoeCapture {
    pub layer: usize,
    pub position: usize,
    pub canvas_token: u32,
    pub token_ids: Vec<u32>,
    pub kv_len: u32,
    pub post_attn: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub experts: Vec<u32>,
    pub expert_weights: Vec<u16>,
    pub moe_out: Vec<f32>,
    /// MoE output after `post_ff_ln_2` (matches MLX `moe_out_ln`).
    pub moe_out_ln: Vec<f32>,
    pub layer_out: Vec<f32>,
}

pub(crate) fn routes_from_route_scratch(route: &RouteScratch) -> Vec<RouteResult> {
    let mut routes = Vec::with_capacity(CANVAS);
    for tok in 0..CANVAS {
        let indices = (0..TOP_K).map(|k| route.expert[tok][k] as usize).collect();
        let weights = (0..TOP_K)
            .map(|k| crate::shaders::bf16::bf16_bits_to_f32(route.weight[tok][k]))
            .collect();
        routes.push(RouteResult { indices, weights });
    }
    routes
}

pub(crate) fn write_f32_arena(arena: &ProtocolObject<dyn MTLBuffer>, base: u64, data: &[f32]) {
    let byte_off = base as usize;
    unsafe {
        let ptr = arena.contents().as_ptr().add(byte_off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *ptr.add(i) = v;
        }
    }
}

fn read_f32_arena(arena: &ProtocolObject<dyn MTLBuffer>, base: u64, elems: usize) -> Vec<f32> {
    let byte_off = base as usize;
    unsafe {
        let ptr = arena.contents().as_ptr().add(byte_off) as *const f32;
        (0..elems).map(|i| *ptr.add(i)).collect()
    }
}

fn read_f32_arena_row(
    arena: &ProtocolObject<dyn MTLBuffer>,
    base: u64,
    row: usize,
    width: usize,
) -> Vec<f32> {
    let byte_off = base as usize + row * width * 4;
    let ptr = unsafe { arena.contents().as_ptr().add(byte_off) as *const f32 };
    (0..width).map(|i| unsafe { *ptr.add(i) }).collect()
}

fn rebucket_route_scratch(route: &mut RouteScratch) {
    let experts: Vec<Vec<u32>> = route.expert.iter().map(|row| row.to_vec()).collect();
    let state = crate::shaders::cpu::moe_router::moe_bucket_phases(
        &experts,
        N_EXPERTS as u32,
        TOP_K as u32,
    );
    route.count = [0; N_EXPERTS];
    for e in 0..N_EXPERTS {
        route.row_start[e] = state.offset[e];
    }
    route.row_start[N_EXPERTS] = state.num_slots;
    route.num_slots = state.num_slots;
    let mut has = [false; N_EXPERTS];
    for row in &experts {
        for &e in row {
            has[e as usize] = true;
        }
    }
    let mut active = 0u32;
    for e in 0..N_EXPERTS {
        if has[e] {
            route.active_expert[active as usize] = e as u32;
            active += 1;
        }
    }
    route.num_active_experts = active;
    for (i, &tok) in state.token_list.iter().enumerate() {
        route.token_list[i] = tok;
        route.slot_list[i] = state.slot_list[i];
    }
    fill_token_slot(route);
}

fn patch_route_position(
    route: &mut RouteScratch,
    position: usize,
    experts: &[u32],
    weights: &[u16],
) {
    assert!(position < CANVAS);
    for k in 0..TOP_K {
        route.expert[position][k] = experts[k];
        route.weight[position][k] = weights[k];
    }
    rebucket_route_scratch(route);
}

fn f32_to_f16_bits(v: f32) -> u16 {
    crate::shaders::f16::f32_to_f16_bits(v)
}

fn route_override_from_ref_json(
    path: &Path,
    position: usize,
) -> Result<Option<(Vec<u32>, Vec<u16>)>, Error> {
    let text = std::fs::read_to_string(path).map_err(Error::Io)?;
    let doc: serde_json::Value = serde_json::from_str(&text).map_err(Error::Json)?;
    if doc.get("position").and_then(|v| v.as_u64()) != Some(position as u64) {
        return Ok(None);
    }
    let experts: Vec<u32> = doc
        .get("experts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Runtime("route ref missing experts"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let weights: Vec<u16> = doc
        .get("expert_weights")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Runtime("route ref missing expert_weights"))?
        .iter()
        .map(|v| {
            if let Some(n) = v.as_u64() {
                n as u16
            } else {
                f32_to_f16_bits(v.as_f64().unwrap_or(0.0) as f32)
            }
        })
        .collect();
    if experts.len() != TOP_K || weights.len() != TOP_K {
        return Err(Error::Runtime("route ref experts/weights must be top_k"));
    }
    Ok(Some((experts, weights)))
}

fn read_route_at_position(
    route: &ProtocolObject<dyn MTLBuffer>,
    position: usize,
) -> (Vec<u32>, Vec<u16>) {
    unsafe {
        let ptr = route.contents().as_ptr();
        let weight = ptr as *const u16;
        let expert = ptr.add(CANVAS * TOP_K * 2) as *const u32;
        let experts = (0..TOP_K)
            .map(|k| *expert.add(position * TOP_K + k))
            .collect();
        let expert_weights = (0..TOP_K)
            .map(|k| *weight.add(position * TOP_K + k))
            .collect();
        (experts, expert_weights)
    }
}

/// Router bucket state after `encode_layer_router_buckets` (denoise path).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteScratchStats {
    pub num_slots: u32,
    pub expected_num_slots: u32,
    pub row_start: Vec<u32>,
    pub count: Vec<u32>,
    pub per_expert_slots: Vec<u32>,
    pub experts_used: u32,
    pub slots_ok: bool,
}

pub fn route_scratch_stats(route: &RouteScratch) -> RouteScratchStats {
    let expected = (CANVAS * TOP_K) as u32;
    let row_start: Vec<u32> = route.row_start.to_vec();
    let count: Vec<u32> = route.count.to_vec();
    let mut per_expert = Vec::with_capacity(N_EXPERTS);
    let mut experts_used = 0u32;
    for e in 0..N_EXPERTS {
        let n = row_start[e + 1].saturating_sub(row_start[e]);
        per_expert.push(n);
        if n > 0 {
            experts_used += 1;
        }
    }
    let slots_ok = route.num_slots == expected && row_start[N_EXPERTS] == expected;
    RouteScratchStats {
        num_slots: route.num_slots,
        expected_num_slots: expected,
        row_start,
        count,
        per_expert_slots: per_expert,
        experts_used,
        slots_ok,
    }
}

fn vector_l2(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let xf = a[i] as f64;
        let yf = b[i] as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    }
}

fn rel_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut diff2 = 0.0f64;
    let mut na = 0.0f64;
    for i in 0..n {
        let d = b[i] as f64 - a[i] as f64;
        diff2 += d * d;
        na += a[i] as f64 * a[i] as f64;
    }
    if na > 0.0 {
        (diff2.sqrt() / na.sqrt()) as f32
    } else {
        0.0
    }
}

fn count_nonzero_f32(data: &[f32], eps: f32) -> usize {
    data.iter().filter(|x| x.abs() > eps).count()
}

/// Capture MoE router bucketing on denoise step 1 (`encode_layer` through router buckets).
#[derive(Debug, Clone, serde::Serialize)]
pub struct MoeRouteCapture {
    pub layer: usize,
    pub step: u32,
    pub kv_len: u32,
    pub moe_style: String,
    pub route: RouteScratchStats,
    pub grouped_dispatched: bool,
    pub moe_out_l2: Option<f32>,
    pub moe_out_nonzero: Option<usize>,
    /// Full-canvas cosine: batched grouped GPU `moe_out` vs `fill_moe_out_dgq_cpu` oracle.
    pub moe_out_gpu_cpu_cos: Option<f32>,
    pub moe_out_gpu_cpu_rel_l2: Option<f32>,
}

fn read_scratch_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
) -> Vec<f32> {
    unsafe {
        let ptr = buf.contents().as_ptr().add(byte_off) as *const f32;
        (0..elems).map(|i| *ptr.add(i)).collect()
    }
}

/// Per-stage GPU vs CPU oracle cosines inside the batched MoE pipeline.
pub fn run_step_moe_batched_pin_capture(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    layer: usize,
) -> Result<crate::shaders::moe_batched_pin::MoeBatchedPinDump, Error> {
    use crate::shaders::moe_batched_pin::{
        MoeBatchedPinDump, MoeBatchedPinLayout, MoeBatchedPinRoute,
        verify_batched_stages_cpu_with_verdict,
    };

    const SCHEMA: u32 = 3;
    let layer = layer.min(cfg.layers.saturating_sub(1));
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;
    let format = rt.block_profile.format;

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
        for l in 0..layer {
            enc.encode_full_layer(l, &layout)?;
        }
        enc.encode_layer(layer, &layout)?;
        Ok(())
    })?;

    let route: RouteScratch = read_struct(&rt.bufs.route);
    let moe_in = read_arena_buffer_f32(
        &rt.bufs.arena,
        rt.bufs.arena_map.moein_off() as usize,
        CANVAS * HID,
    );
    let slots = route.num_slots as usize;
    let gu_elems = slots * (MOE_FF as usize) * 2;
    let act_elems = slots * MOE_FF as usize;
    let slot_elems = slots * HID;

    rt.dispatch_and_wait(|enc| {
        enc.encode_moe_batched_gather()?;
        Ok(())
    })?;
    let gpu_gather = read_scratch_f32(&rt.bufs.gemm_b, moe_w_byte_off_a(), slot_elems);

    rt.dispatch_and_wait(|enc| {
        enc.encode_moe_batched_gate_up(layer, &layout)?;
        Ok(())
    })?;
    let gpu_gate_up = read_scratch_f32(&rt.bufs.gemm_b, moe_w_byte_off_gu(), gu_elems);

    rt.dispatch_and_wait(|enc| {
        enc.encode_moe_batched_swiglu()?;
        Ok(())
    })?;
    let gpu_swiglu = read_scratch_f32(&rt.bufs.gemm_a, 0, act_elems);

    rt.dispatch_and_wait(|enc| {
        enc.encode_moe_batched_down(layer, &layout)?;
        Ok(())
    })?;
    let gpu_down = read_scratch_f32(&rt.bufs.gemm_b, moe_w_byte_off_a(), slot_elems);

    rt.dispatch_and_wait(|enc| {
        enc.encode_moe_batched_scatter()?;
        Ok(())
    })?;
    let gpu_scatter = read_f32_arena(&rt.bufs.arena, rt.bufs.arena_map.moeout_off(), CANVAS * HID);

    let blob = unsafe {
        std::slice::from_raw_parts(
            rt.gpu_blob.buffer.contents().as_ptr().cast::<u8>(),
            rt.gpu_blob.len,
        )
    };
    let layer_off = &layout.layers[layer];
    let (stages, rel_l2, gate_up_diff, first_divergent_stage) =
        verify_batched_stages_cpu_with_verdict(
            &moe_in,
            &route,
            blob,
            layer_off,
            format,
            &gpu_gather,
            &gpu_gate_up,
            &gpu_swiglu,
            &gpu_down,
            &gpu_scatter,
        );

    Ok(MoeBatchedPinDump {
        schema_version: SCHEMA,
        prompt: cfg
            .prefill_token_ids
            .as_ref()
            .map(|_| "prefill".to_string())
            .unwrap_or_else(|| "Hello".to_string()),
        seed: cfg.seed,
        layer,
        kv_len: rt.read_params().kv_len,
        format: format!("{format:?}"),
        layout: MoeBatchedPinLayout {
            moe_w_off_gu_bytes: moe_w_byte_off_gu(),
            gate_up_elems: gu_elems,
            swiglu_elems: act_elems,
            hidden: HID,
            moe_ff: MOE_FF as usize,
        },
        route: MoeBatchedPinRoute::from_route(&route),
        stages,
        rel_l2,
        gate_up_diff,
        first_divergent_stage,
    })
}

pub fn run_step_moe_route_capture(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    layer: usize,
    run_grouped: bool,
) -> Result<MoeRouteCapture, Error> {
    let layer = layer.min(cfg.layers.saturating_sub(1));
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;
    let moe_style = match rt.block_profile.moe_style() {
        MoeExecutionStyle::BatchedGrouped => "batched_grouped",
        MoeExecutionStyle::ScalarPerExpert => "scalar_per_expert",
    }
    .to_string();

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
        for l in 0..layer {
            enc.encode_full_layer(l, &layout)?;
        }
        enc.encode_layer(layer, &layout)?;
        Ok(())
    })?;

    let route: RouteScratch = read_struct(&rt.bufs.route);
    let route_stats = route_scratch_stats(&route);

    rt.dispatch_and_wait(|enc| {
        if run_grouped {
            enc.encode_layer_moe_grouped(layer, &layout)?;
        } else {
            enc.encode_layer_moe_scalar(layer, &layout)?;
        }
        Ok(())
    })?;
    let grouped_dispatched = true;
    let moe_out_gpu = read_f32_arena(&rt.bufs.arena, rt.bufs.arena_map.moeout_off(), CANVAS * HID);
    let moe_out_l2 = Some(vector_l2(&moe_out_gpu));
    let moe_out_nonzero = Some(count_nonzero_f32(&moe_out_gpu, 1e-9));
    rt.fill_moe_out_dgq_cpu(layer)?;
    let moe_out_cpu = read_f32_arena(&rt.bufs.arena, rt.bufs.arena_map.moeout_off(), CANVAS * HID);
    let moe_out_gpu_cpu_cos = Some(cosine_f32(&moe_out_gpu, &moe_out_cpu));
    let moe_out_gpu_cpu_rel_l2 = Some(rel_l2_f32(&moe_out_cpu, &moe_out_gpu));

    Ok(MoeRouteCapture {
        layer,
        step: 1,
        kv_len: rt.read_params().kv_len,
        moe_style,
        route: route_stats,
        grouped_dispatched,
        moe_out_l2,
        moe_out_nonzero,
        moe_out_gpu_cpu_cos,
        moe_out_gpu_cpu_rel_l2,
    })
}

/// Step-1 forward through `layer` FFN/MoE; read checkpoints at one canvas row.
pub fn run_step_moe_layer_capture(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    layer: usize,
    position: usize,
) -> Result<LayerMoeCapture, Error> {
    if position >= CANVAS {
        return Err(Error::Runtime("moe capture position out of range"));
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

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_through_attention(layer, &layout)?;
        enc.encode_layer_o_proj_post_attn(layer, &layout)?;
        Ok(())
    })?;
    let post_attn = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.stream_off(), position);

    let store = DgqStore::open(model_dir)?;
    let p = format!("model.decoder.layers.{layer}.router");
    let router_scale = store.tensor_f32(&format!("{p}.scale"))?;
    let router_proj = store.tensor_f32(&format!("{p}.proj.weight"))?;
    let router_logits = crate::shaders::cpu::moe_router::router_logits_row(
        &post_attn,
        &router_scale,
        &router_proj,
        HID,
        N_EXPERTS,
    );

    rt.dispatch_and_wait(|enc| enc.encode_layer_dense_ffn(layer, &layout))?;
    let dense_out = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.dense_off(), position);

    rt.dispatch_and_wait(|enc| enc.encode_layer_router_buckets(layer, &layout))?;

    if let Some(path) = crate::flags::moe_route_ref_path() {
        if let Some((experts, weights)) = route_override_from_ref_json(Path::new(&path), position)?
        {
            let mut route: RouteScratch = read_struct(&rt.bufs.route);
            patch_route_position(&mut route, position, &experts, &weights);
            write_struct(&rt.bufs.route, &route);
            eprintln!(
                "moe-capture: route override pos={position} experts={experts:?} (from {path})"
            );
        }
    }

    rt.dispatch_and_wait(|enc| enc.encode_layer_moe_grouped(layer, &layout))?;
    let (experts, expert_weights) = read_route_at_position(&rt.bufs.route, position);

    let moe_out = read_f32_arena_row(
        &rt.bufs.arena,
        rt.bufs.arena_map.moeout_off(),
        position,
        HID,
    );

    rt.dispatch_and_wait(|enc| enc.encode_layer_moe_post_norm(layer, &layout))?;
    let moe_out_ln = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.moein_off(), position);

    rt.dispatch_and_wait(|enc| enc.encode_layer_moe_post_combine(layer, &layout))?;
    let layer_out = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);

    let state = rt.read_canvas_state();
    Ok(LayerMoeCapture {
        layer,
        position,
        canvas_token: state.ids[position],
        token_ids: state.ids.to_vec(),
        kv_len: rt.read_params().kv_len,
        post_attn,
        dense_out,
        router_logits,
        experts,
        expert_weights,
        moe_out,
        moe_out_ln,
        layer_out,
    })
}

#[derive(Debug, Clone)]
pub struct MoeKernelInputProbe {
    pub tok: u32,
    pub slot: u32,
    pub expert: u32,
    pub weight: f32,
    pub x_head: [f32; 8],
    pub moe_in_row0_head: [f32; 8],
    pub down_o_head: [f32; 8],
    pub moe_out_tok_row_head: [f32; 8],
}

#[derive(Debug, Clone)]
pub struct SingleExpertMoeCapture {
    pub layer: usize,
    pub position: usize,
    pub expert_id: u32,
    pub canvas_token: u32,
    pub token_ids: Vec<u32>,
    pub kv_len: u32,
    pub moe_in: Vec<f32>,
    pub gpu_out: Vec<f32>,
    pub gpu_act_after_barrier: Vec<f32>,
    pub gpu_act_at_down_read: Vec<f32>,
    pub gpu_input: MoeKernelInputProbe,
}

/// Step-1 forward through `layer`; run grouped MoE for one expert at one row (no router).
pub fn run_step_moe_single_expert_capture(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    layer: usize,
    position: usize,
    expert_id: u32,
) -> Result<SingleExpertMoeCapture, Error> {
    if position >= CANVAS {
        return Err(Error::Format(
            "single-expert moe capture position out of range",
        ));
    }
    if expert_id as usize >= N_EXPERTS {
        return Err(Error::Format(
            "single-expert moe capture expert_id out of range",
        ));
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

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_through_attention(layer, &layout)?;
        enc.encode_layer_o_proj_post_attn(layer, &layout)?;
        enc.encode_layer_moe_single_expert_setup(layer, &layout, position, expert_id);
        Ok(())
    })?;

    let moe_in = read_arena_row(&rt.bufs.arena, rt.bufs.arena_map.moein_off(), position, HID);

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_moe_grouped_act_probe(layer, &layout)?;
        Ok(())
    })?;
    let gpu_out = read_f32_arena_row(
        &rt.bufs.arena,
        rt.bufs.arena_map.moeout_off(),
        position,
        HID,
    );
    let act_probe = read_f32_arena(
        &rt.bufs.arena,
        rt.bufs.arena_map.soft_off(),
        MOE_ACT_PROBE_FLOATS,
    );
    let moe_ff = MOE_FF as usize;
    let gpu_act_after_barrier = act_probe[..moe_ff].to_vec();
    let gpu_act_at_down_read = act_probe[moe_ff..moe_ff * 2].to_vec();
    let meta = MOE_ACT_PROBE_ACT_FLOATS;
    let gpu_input = MoeKernelInputProbe {
        tok: act_probe[meta] as u32,
        slot: act_probe[meta + 1] as u32,
        expert: act_probe[meta + 2] as u32,
        weight: act_probe[meta + 3],
        x_head: act_probe[meta + 4..meta + 12].try_into().expect("x_head"),
        moe_in_row0_head: act_probe[meta + 12..meta + 20]
            .try_into()
            .expect("row0_head"),
        down_o_head: act_probe[meta + 20..meta + 28].try_into().expect("down_o"),
        moe_out_tok_row_head: act_probe[meta + 28..meta + 36]
            .try_into()
            .expect("moe_out_tok"),
    };
    let state = rt.read_canvas_state();
    Ok(SingleExpertMoeCapture {
        layer,
        position,
        expert_id,
        canvas_token: state.ids[position],
        token_ids: state.ids.to_vec(),
        kv_len: rt.read_params().kv_len,
        moe_in,
        gpu_out,
        gpu_act_after_barrier,
        gpu_act_at_down_read,
        gpu_input,
    })
}

pub fn bench_step_kernel(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    iters: usize,
) -> Result<StepBenchResult, Error> {
    let iters = iters.max(1);
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    let finish = cfg.finish;

    let warmup_started = Instant::now();
    rt.run_forward_once(finish)?;
    let warmup = warmup_started.elapsed();

    let started = Instant::now();
    for _ in 0..iters {
        rt.run_forward_once(finish)?;
    }
    let elapsed = started.elapsed();
    let per_step = elapsed / iters as u32;

    Ok(StepBenchResult {
        compile: build.compile,
        warmup,
        per_step,
        iters,
        finish,
    })
}

pub fn bench_step_kernel_profile(
    model_dir: &Path,
    cfg: StepSmokeConfig,
) -> Result<StepProfileResult, Error> {
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    let finish = cfg.finish;
    rt.run_forward_once(finish)?;
    let mut prof = rt.profile_forward_once(finish)?;
    prof.compile = build.compile;
    Ok(prof)
}

pub fn bench_step_kernel_encode_subprofile(
    model_dir: &Path,
    cfg: StepSmokeConfig,
) -> Result<EncodeSubProfileResult, Error> {
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    rt.run_forward_once(cfg.finish)?;
    if crate::flags::trace_ranges_enabled() {
        // Warm to a steady-state (denoised) step, then trace one step's ranges.
        // Under DGQ_PREFILL_F16 the traced step runs on the fp16 pipeline set
        // (per-stage localization for the E11 arena dtype flip).
        rt.arena_f16_mode = rt.pipelines_prefill_f16.is_some();
        rt.run_forward_once(cfg.finish)?;
        rt.trace_step_ranges()?;
        rt.arena_f16_mode = false;
    }
    let mut prof = rt.profile_encode_subprofile()?;
    prof.compile = build.compile;
    Ok(prof)
}

/// Holistic prefill proxy (task #87): build the runtime (compiling pipelines
/// per the current tile flags) and time one M=1024 super-chunk at `kv_len`.
/// Returns mean ms/super-chunk.
pub fn bench_step_kernel_prefill_super(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    kv_len: u32,
    iters: usize,
) -> Result<std::time::Duration, Error> {
    let (mut rt, _build) = build_step_runtime(model_dir, &cfg)?;
    rt.bench_prefill_super(kv_len, iters)
}

/// Floor decomposition: full super-chunk time + per-stage-group cost (ablation).
pub fn bench_step_kernel_prefill_super_stages(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    kv_len: u32,
    iters: usize,
) -> Result<(f64, Vec<(&'static str, f64)>), Error> {
    let (mut rt, _build) = build_step_runtime(model_dir, &cfg)?;
    rt.bench_prefill_super_stages(kv_len, iters)
}

/// Profile the first `n_steps` denoise forwards (canvas `st.step` 0..n_steps-1).
/// Step 0 has no SC preamble; step >= 1 includes self-conditioning.
pub fn bench_step_kernel_profile_steps(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    n_steps: usize,
) -> Result<Vec<(u32, StepProfileResult)>, Error> {
    if cfg.finish != StepFinishMode::Full {
        return Err(Error::Format(
            "bench_step_kernel_profile_steps requires StepFinishMode::Full",
        ));
    }
    let n_steps = n_steps.max(1);
    let watch = crate::flags::mem_watch_enabled();
    let mem_before = watch.then(crate::metal::memwatch::snapshot);
    let (mut rt, build) = build_step_runtime(model_dir, cfg)?;
    if let Some(before) = mem_before {
        crate::metal::memwatch::report_section("session-open", before, &rt.ctx.device);
    }
    let finish = StepFinishMode::Full;
    let mut out = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let st: CanvasState = read_struct(&rt.bufs.state);
        let mem_before = watch.then(crate::metal::memwatch::snapshot);
        let mut prof = rt.profile_forward_once(finish)?;
        if i == 0 {
            prof.compile = build.compile;
        }
        if let Some(before) = mem_before {
            crate::metal::memwatch::report_section(
                &format!("profile-step {i}"),
                before,
                &rt.ctx.device,
            );
        }
        out.push((st.step, prof));
    }
    Ok(out)
}

/// Wall-clock timing for fused vs split QKV / gate_up GEMM dispatches (all layers).
#[derive(Debug, Clone)]
pub struct FusedGemmDispatchBenchResult {
    pub compile: std::time::Duration,
    pub layers: usize,
    pub iters: usize,
    /// One command buffer per layer: untimed rmsnorm submit, timed GEMM submit.
    pub qkv_gemm_stacked: std::time::Duration,
    pub qkv_gemm_split: std::time::Duration,
    pub gate_up_gemm_stacked: std::time::Duration,
    pub gate_up_gemm_split: std::time::Duration,
    /// One command buffer: interleaved per-layer rmsnorm + GEMM (production ordering).
    pub qkv_batched_stacked: std::time::Duration,
    pub qkv_batched_split: std::time::Duration,
    pub gate_up_batched_stacked: std::time::Duration,
    pub gate_up_batched_split: std::time::Duration,
    pub qkv_stacked_dispatches_per_pass: usize,
    pub qkv_split_dispatches_per_pass: usize,
    #[allow(dead_code)]
    pub gate_up_dispatches_per_pass: usize,
}

fn qkv_split_dispatch_count(layout: &ModelLayout, layers: usize) -> usize {
    (0..layers)
        .map(|layer| {
            let l = &layout.layers[layer];
            if l.v_proj != 0 { 3 } else { 2 }
        })
        .sum()
}

fn time_qkv_gemm_only(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let mut total = std::time::Duration::ZERO;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        rt.dispatch_and_wait(|enc| {
            enc.rmsnorm(
                enc.arena().hidden_off(),
                enc.arena().tmp_off(),
                l.input_ln,
                HID as u32,
                CANVAS,
            );
            Ok(())
        })?;
        let t0 = Instant::now();
        rt.dispatch_and_wait(|enc| enc.dispatch_qkv_gemms(layer, &layout, stacked))?;
        total += t0.elapsed();
    }
    Ok(total)
}

fn time_gate_up_gemm_only(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let mut total = std::time::Duration::ZERO;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        rt.dispatch_and_wait(|enc| {
            enc.rmsnorm(
                enc.arena().stream_off(),
                enc.arena().tmp_off(),
                l.pre_ff_ln,
                HID as u32,
                CANVAS,
            );
            Ok(())
        })?;
        let t0 = Instant::now();
        rt.dispatch_and_wait(|enc| enc.dispatch_gate_up_gemms(layer, &layout, stacked))?;
        total += t0.elapsed();
    }
    Ok(total)
}

fn time_qkv_batched(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let t0 = Instant::now();
    rt.dispatch_and_wait(|enc| {
        for layer in 0..layers {
            let l = &layout.layers[layer];
            enc.rmsnorm(
                enc.arena().hidden_off(),
                enc.arena().tmp_off(),
                l.input_ln,
                HID as u32,
                CANVAS,
            );
            enc.dispatch_qkv_gemms(layer, &layout, stacked)?;
        }
        Ok(())
    })?;
    Ok(t0.elapsed())
}

fn time_gate_up_batched(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let t0 = Instant::now();
    rt.dispatch_and_wait(|enc| {
        for layer in 0..layers {
            let l = &layout.layers[layer];
            enc.rmsnorm(
                enc.arena().stream_off(),
                enc.arena().tmp_off(),
                l.pre_ff_ln,
                HID as u32,
                CANVAS,
            );
            enc.dispatch_gate_up_gemms(layer, &layout, stacked)?;
        }
        Ok(())
    })?;
    Ok(t0.elapsed())
}

/// Profile QKV and dense gate/up GEMM dispatches in isolation (real weights + canvas activations).
pub fn bench_fused_gemm_dispatches(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    iters: usize,
) -> Result<FusedGemmDispatchBenchResult, Error> {
    let iters = iters.max(1);
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;

    // Realistic activations: one forward preamble + embed (no MoE/finish).
    rt.dispatch_and_wait(|enc| {
        let st: CanvasState = read_struct(&enc.bufs.state);
        let first_step = if st.step == 0 { 1u32 } else { 0u32 };
        enc.encode_step_preamble(&layout, first_step)?;
        for layer in 0..layers {
            enc.encode_layer(layer, &layout)?;
        }
        Ok(())
    })?;

    let qkv_split_dispatches = qkv_split_dispatch_count(&layout, layers);
    let gate_up_dispatches = layers * 2;

    let mut qkv_gemm_stacked = std::time::Duration::ZERO;
    let mut qkv_gemm_split = std::time::Duration::ZERO;
    let mut gate_up_gemm_stacked = std::time::Duration::ZERO;
    let mut gate_up_gemm_split = std::time::Duration::ZERO;
    let mut qkv_batched_stacked = std::time::Duration::ZERO;
    let mut qkv_batched_split = std::time::Duration::ZERO;
    let mut gate_up_batched_stacked = std::time::Duration::ZERO;
    let mut gate_up_batched_split = std::time::Duration::ZERO;

    // Warmup each mode once.
    let _ = time_qkv_gemm_only(&mut rt, layers, true)?;
    let _ = time_qkv_gemm_only(&mut rt, layers, false)?;
    let _ = time_gate_up_gemm_only(&mut rt, layers, true)?;
    let _ = time_gate_up_gemm_only(&mut rt, layers, false)?;

    for _ in 0..iters {
        qkv_gemm_stacked += time_qkv_gemm_only(&mut rt, layers, true)?;
        qkv_gemm_split += time_qkv_gemm_only(&mut rt, layers, false)?;
        gate_up_gemm_stacked += time_gate_up_gemm_only(&mut rt, layers, true)?;
        gate_up_gemm_split += time_gate_up_gemm_only(&mut rt, layers, false)?;
        qkv_batched_stacked += time_qkv_batched(&mut rt, layers, true)?;
        qkv_batched_split += time_qkv_batched(&mut rt, layers, false)?;
        gate_up_batched_stacked += time_gate_up_batched(&mut rt, layers, true)?;
        gate_up_batched_split += time_gate_up_batched(&mut rt, layers, false)?;
    }

    let div = |d: std::time::Duration| d / iters as u32;
    Ok(FusedGemmDispatchBenchResult {
        compile: build.compile,
        layers,
        iters,
        qkv_gemm_stacked: div(qkv_gemm_stacked),
        qkv_gemm_split: div(qkv_gemm_split),
        gate_up_gemm_stacked: div(gate_up_gemm_stacked),
        gate_up_gemm_split: div(gate_up_gemm_split),
        qkv_batched_stacked: div(qkv_batched_stacked),
        qkv_batched_split: div(qkv_batched_split),
        gate_up_batched_stacked: div(gate_up_batched_stacked),
        gate_up_batched_split: div(gate_up_batched_split),
        qkv_stacked_dispatches_per_pass: layers,
        qkv_split_dispatches_per_pass: qkv_split_dispatches,
        gate_up_dispatches_per_pass: gate_up_dispatches,
    })
}

/// Read `elems` bf16 arena values from a shared Metal buffer as f32.
pub fn read_arena_buffer_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
) -> Vec<f32> {
    use crate::shaders::bf16;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    (0..elems)
        .map(|i| unsafe { bf16::bf16_bits_to_f32(*ptr.add(i)) })
        .collect()
}

/// Read `elems` bf16 values from a shared Metal buffer as f32 (logits / KV cache).
pub fn read_half_buffer_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
) -> Vec<f32> {
    use crate::shaders::bf16;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    (0..elems)
        .map(|i| unsafe { bf16::bf16_bits_to_f32(*ptr.add(i)) })
        .collect()
}

#[derive(Debug)]
pub struct StepForwardOutput {
    pub norm_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// Forward-only monolithic pass: final norm hidden + softcapped logits.
pub fn run_step_forward(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<StepForwardOutput, Error> {
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;
    let st_before: CanvasState = read_struct(&rt.bufs.state);
    let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };

    rt.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;
    for layer in 0..layers {
        rt.dispatch_and_wait(|enc| enc.encode_full_layer(layer, &layout))?;
    }
    // Snapshot final norm before lm_head; gemm_q8_logits clobbers self.arena().tmp_off() on GPU.
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
    let norm_hidden = read_arena_buffer_f32(
        &rt.bufs.arena,
        rt.bufs.arena_map.tmp_off() as usize,
        CANVAS * HID,
    );
    rt.dispatch_and_wait(|enc| {
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
    let state: CanvasState = read_struct(&rt.bufs.state);
    Ok(StepForwardOutput {
        norm_hidden,
        logits: read_half_buffer_f32(&rt.bufs.logits, 0, CANVAS * VOCAB),
        token_ids: state.ids.to_vec(),
    })
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct DenoiseStepStats {
    pub accept_count: u32,
    pub mean_entropy: f32,
    pub min_entropy: f32,
    pub low_entropy_positions: u32,
}

/// Chat-templated `-p Hello` prefill token ids (matches `generate-monolithic` default).
#[cfg(test)]
pub fn hello_chat_prefill_token_ids(model_dir: &Path) -> Result<Vec<u32>, Error> {
    use crate::chat_template::{ChatFormatOptions, ChatTurn, format_chat_token_ids};
    use crate::tokenizer::Tokenizer;
    let tok = Tokenizer::load(&model_dir.join("tokenizer.json"))?;
    format_chat_token_ids(
        &tok,
        &[ChatTurn::user("Hello")],
        &ChatFormatOptions::default(),
    )
}

/// Run `cfg.steps` monolithic denoise iterations; one stats record per iteration.
#[cfg(test)]
pub fn run_denoise_steps(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<Vec<DenoiseStepStats>, Error> {
    if cfg.finish != StepFinishMode::Full {
        return Err(Error::Format(
            "run_denoise_steps requires StepFinishMode::Full",
        ));
    }
    use crate::sample::StableConfidentStopper;
    let steps = cfg.steps.max(1);
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let sampler = crate::sample::sampler_for_steps(steps, cfg.no_early_stop);
    let params = step_params_from_sampler(
        &sampler,
        rt.read_params().kv_len,
        cfg.no_early_stop,
        rt.read_params().eos_token_id,
    );
    let mut rng = Rng::new(cfg.seed);
    rt.reset_block(VOCAB, &mut rng, params);
    let mut stopper = StableConfidentStopper::new(
        sampler.stability_threshold,
        if cfg.no_early_stop {
            f32::MAX
        } else {
            sampler.confidence_threshold
        },
    );
    stopper.reset();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        rt.run_denoise_step()?;
        let st = rt.read_canvas_state();
        let ent = crate::sample::step_entropy_stats(&st.entropy, &st.accept);
        out.push(DenoiseStepStats {
            accept_count: ent.accept_count,
            mean_entropy: st.mean_entropy,
            min_entropy: ent.min_entropy,
            low_entropy_positions: ent.low_entropy_positions,
        });
        let max_steps_reached = st.step >= params.max_steps;
        let confident_stop = !cfg.no_early_stop && st.stop_flag != 0;
        if confident_stop || max_steps_reached {
            break;
        }
    }
    Ok(out)
}

pub fn run_step_smoke(model_dir: &Path, cfg: StepSmokeConfig) -> Result<StepSmokeResult, Error> {
    let finish = cfg.finish;
    let steps = cfg.steps;
    let (mut rt, _) = build_step_runtime(model_dir, &cfg)?;
    let started = Instant::now();
    for step_i in 0..steps {
        rt.run_forward_once(finish)?;
        eprintln!(
            "step-smoke: completed denoise step {}/{}",
            step_i + 1,
            steps
        );
        if finish == StepFinishMode::Full {
            let st: CanvasState = read_struct(&rt.bufs.state);
            if st.stop_flag != 0 && !cfg.no_early_stop {
                eprintln!("step-smoke: early stop at step {}", st.step);
                break;
            }
        }
    }
    let elapsed = started.elapsed();

    let final_state: CanvasState = read_struct(&rt.bufs.state);
    let (logits_finite, max_abs_logit) = check_logits_finite(&rt.bufs.logits);
    let ent_stats = crate::sample::step_entropy_stats(&final_state.entropy, &final_state.accept);

    Ok(StepSmokeResult {
        step: final_state.step,
        stop_flag: final_state.stop_flag,
        mean_entropy: final_state.mean_entropy,
        min_entropy: ent_stats.min_entropy,
        low_entropy_positions: ent_stats.low_entropy_positions,
        ids: final_state.ids[..CANVAS].try_into().expect("canvas prefix"),
        logits_finite,
        max_abs_logit,
        elapsed,
    })
}
