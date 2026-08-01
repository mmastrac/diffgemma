//! step_kernel MoE diagnostics: route/expert captures and scratch stats.
//! Extracted from step_kernel_diagnostics.rs (see diag_probe for the family).

use crate::Error;
use crate::dgq::DgqStore;
use crate::metal::step_quant::MoeExecutionStyle;
use crate::model::moe::RouteResult;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::path::Path;

use super::*;

use super::diag_probe::{read_arena_hidden_row, read_arena_row};

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
    for (e, &present) in has.iter().enumerate() {
        if present {
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
    route.expert[position][..TOP_K].copy_from_slice(&experts[..TOP_K]);
    route.weight[position][..TOP_K].copy_from_slice(&weights[..TOP_K]);
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
        .ok_or(Error::Runtime("route ref missing experts"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let weights: Vec<u16> = doc
        .get("expert_weights")
        .and_then(|v| v.as_array())
        .ok_or(Error::Runtime("route ref missing expert_weights"))?
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

    if let Some(path) = crate::flags::moe_route_ref_path()
        && let Some((experts, weights)) = route_override_from_ref_json(Path::new(&path), position)?
    {
        let mut route: RouteScratch = read_struct(&rt.bufs.route);
        patch_route_position(&mut route, position, &experts, &weights);
        write_struct(&rt.bufs.route, &route);
        eprintln!("moe-capture: route override pos={position} experts={experts:?} (from {path})");
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
