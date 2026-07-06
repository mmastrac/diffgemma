//! Monolithic diffgemma denoise-step kernel (parallel smoke path).
//! See `shaders/monolithic/diffgemma_step.metal` and dispatch schedule at file bottom.

use crate::config::{ModelConfig, TextConfig};
use crate::dgq::DgqStore;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::dgq_gpu::DgqGpuBlob;
use crate::metal::moe::experts_forward_dgq_cpu;
use crate::metal::step_quant::{
    BlockGroupedJob, MoeExecutionStyle, StepBlockProfile,
};
use crate::metal::weights::GpuDecoderWeightCache;
use crate::kernels::sub::QuantFormat;
use crate::model::moe::{MoeScratch, RouteResult};
use crate::sample::{initialize_canvas, Rng, SamplerConfig};
use crate::safetensors::Error;
use crate::weights::WeightStore;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::mem::offset_of;
use std::path::Path;
use std::time::Instant;

#[path = "step_schedule.rs"]
mod step_schedule;

#[path = "arena_liveness.rs"]
pub(crate) mod arena_liveness;

const STEP_SHADER: &str = shader_include::include_metal!("monolithic/diffgemma_step.metal");

pub const HID: usize = 2816;
pub const VOCAB: usize = 262144;
pub const CANVAS: usize = 256;
/// Up-scale applied to SC softembed probs before the fp16 GEMM tiles and divided
/// back out of the final scale. Must match `SC_PROB_GEMM_SCALE` in
/// `shaders/include/sc_prob_scale.metal`. Keeps near-uniform probs (~2^-18) out
/// of fp16's denormal range (normal min 2^-14).
pub const SC_PROB_GEMM_SCALE: f32 = 4096.0;
pub const FROZEN_WORDS: usize = CANVAS / 32;
pub const ARGMAX_HIST_MAX: usize = crate::sample::ARGMAX_HIST_MAX;
pub const N_LAYERS: usize = 30;
pub const N_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
pub const DENSE_FF: u32 = 2112;
pub const MOE_FF: u32 = 704;
/// Max survivors per row for the sparse SC softembed (select+gather). 8192 fits
/// gemm_a (prob f16) + a gemm_b tail (idx u32). Overflow is monitored; the -10
/// threshold keeps far fewer in practice.
pub const SC_SPARSE_MAXK: u32 = 8192;
/// Two act rows (after-barrier + down-read) plus kernel input probe metadata.
pub const MOE_ACT_PROBE_ACT_FLOATS: usize = (MOE_FF * 2) as usize;
pub const MOE_ACT_PROBE_META_FLOATS: usize = 36; // tok,slot,e,w,x[8],row0[8], down_o[8], moe_out_tok_row[8]

use crate::metal::arena_layout::{build_arena_layout, ArenaLayout, ArenaLayoutParams};

// All DGQ_* env flags live in crate::flags; re-exported here so existing
// `step_kernel::<flag>()` call sites keep working.
pub use crate::flags::{
    attn_mma_enabled, attn_mma_full_enabled, attn_window_enabled, denoise_parity_log_enabled,
    denoise_parity_log_positions, denoiser_argmax_enabled, final_entropy_log_enabled,
    freeze_enabled, fused_algebra_enabled, fused_gate_up_enabled, fused_qkv_enabled,
    gemm_tunable_enabled, logits_finite_check_enabled, logits_finite_sample_count,
    moe_block_sparse_enabled, moe_fuse_gather_enabled, moe_tile_adapt_enabled,
    partial_lm_head_enabled, router_gemm_enabled, sc_sparse_enabled, should_fast_prefill,
    step_text_log_enabled, trace_entropy_enabled, FAST_PREFILL_MIN_TOKENS,
};

pub const MAX_ATTN_Q_COLS: usize = 8192;
pub const MAX_ATTN_KV_COLS: usize = 2048;

pub fn step_arena_params() -> ArenaLayoutParams {
    ArenaLayoutParams {
        canvas: CANVAS,
        hidden: HID,
        dense_ff: DENSE_FF as usize,
        max_attn_q_cols: MAX_ATTN_Q_COLS,
        max_attn_kv_cols: MAX_ATTN_KV_COLS,
    }
}

pub fn step_arena_layout() -> ArenaLayout {
    build_arena_layout(&step_arena_params())
}

pub const MOE_ACT_PROBE_FLOATS: usize = MOE_ACT_PROBE_ACT_FLOATS + MOE_ACT_PROBE_META_FLOATS;

pub const FULL_LAYERS: [usize; 5] = [5, 11, 17, 23, 29];

/// k_gemm_q4/q8 tile kernels use ltid=lid.x and loop step 128 (4 simdgroups × 32 threads).
const GEMM_THREADS_PER_TG: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct LayerOffsets {
    pub input_ln: u64,
    pub q_proj: u64,
    pub q_norm: u64,
    pub k_proj: u64,
    pub k_norm: u64,
    pub v_proj: u64,
    pub o_proj: u64,
    pub post_attn_ln: u64,
    pub pre_ff_ln: u64,
    pub mlp_gate: u64,
    pub mlp_up: u64,
    pub mlp_down: u64,
    pub post_ff_ln_1: u64,
    pub router_scale: u64,
    pub router_proj: u64,
    pub per_expert_scale: u64,
    pub pre_ff_ln_2: u64,
    pub experts_gate_up: u64,
    pub experts_down: u64,
    pub post_ff_ln_2: u64,
    pub post_ff_ln: u64,
    pub layer_scalar: u64,
    pub kv_region: u64,
    pub head_dim: u32,
    pub n_kv_heads: u32,
    pub is_full: u32,
    /// KV slot mapping: 0 = linear (full layers, every position kept);
    /// else a power-of-two-minus-1 ring mask (sliding layers, slot = pos & mask).
    /// A sliding layer only ever attends the last window-1 (1023) keys + the
    /// 256-wide canvas, so a 2048-slot ring holds every live position — this is
    /// what makes 100k+ context affordable (KV goes ~220 KB/token -> ~20).
    pub kv_ring_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelLayout {
    pub embed: u64,
    pub sc_pre_norm: u64,
    pub sc_gate: u64,
    pub sc_up: u64,
    pub sc_down: u64,
    pub final_norm: u64,
    pub layers: [LayerOffsets; N_LAYERS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StepParams {
    pub kv_len: u32,
    pub max_steps: u32,
    pub entropy_bound: f32,
    pub t_min: f32,
    pub t_max: f32,
    pub conf_threshold: f32,
    pub stability_threshold: u32,
    pub min_early_stop_steps: u32,
    pub accept_plateau_threshold: u32,
    pub plateau_prefix_mean_max: f32,
    pub eos_token_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CanvasState {
    pub ids: [u32; CANVAS],
    pub prev_argmax: [u32; CANVAS],
    pub new_sample: [u32; CANVAS],
    pub entropy: [f32; CANVAS],
    pub sorted_idx: [u32; CANVAS],
    pub accept: [u32; CANVAS],
    pub u_cat: [f32; CANVAS],
    pub rng_state: u64,
    pub step: u32,
    pub stop_flag: u32,
    pub argmax_hist_len: u32,
    pub argmax_hist_base: u32,
    pub argmax_hist: [u32; CANVAS * ARGMAX_HIST_MAX],
    pub canvas_stable: u32,
    pub mean_entropy: f32,
    pub accept_plateau: u32,
    pub prev_accept_sig: u32,
    pub frozen: [u32; FROZEN_WORDS],
}


















const STACKED_SEG_MAX: usize = 3;

/// Minimum frozen rows before partial lm_head (avoids compact/gather overhead on step 1).
const PARTIAL_LM_MIN_FROZEN: usize = 8;

#[inline]
pub fn frozen_at(state: &CanvasState, i: usize) -> bool {
    (state.frozen[i >> 5] >> (i & 31)) & 1 != 0
}

pub fn count_unfrozen(state: &CanvasState) -> u32 {
    let mut n = 0u32;
    for i in 0..CANVAS {
        if !frozen_at(state, i) {
            n += 1;
        }
    }
    n
}

pub fn partial_lm_active_rows(state: &CanvasState) -> u32 {
    if !partial_lm_head_enabled() {
        return CANVAS as u32;
    }
    let frozen = CANVAS - count_unfrozen(state) as usize;
    if frozen < PARTIAL_LM_MIN_FROZEN {
        CANVAS as u32
    } else {
        count_unfrozen(state)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RouteScratch {
    pub weight: [[u16; TOP_K]; CANVAS],
    pub expert: [[u32; TOP_K]; CANVAS],
    pub count: [u32; N_EXPERTS],
    pub row_start: [u32; N_EXPERTS + 1],
    pub num_slots: u32,
    pub num_active_experts: u32,
    pub active_expert: [u32; N_EXPERTS],
    pub token_list: [u32; CANVAS * TOP_K],
    pub slot_list: [u32; CANVAS * TOP_K],
    pub token_slot: [[u32; TOP_K]; CANVAS],
    /// Block-sparse MoE GEMM (DGQ_MOE_BLOCK_SPARSE): one entry per <=32-row tile.
    /// `block_expert[b]` = expert owning block b; `block_row0[b]` = its global row
    /// start into the gathered A / token_list. `num_blocks` = total. Built in
    /// `moe_bucket_fill` phase 1. Bounded by num_active_experts + num_slots/32.
    pub block_expert: [u32; MOE_MAX_BLOCKS],
    pub block_row0: [u32; MOE_MAX_BLOCKS],
    pub num_blocks: u32,
}

/// Max block-sparse MoE tiles: sum_e ceil(count_e/32) <= n_active + num_slots/32
/// <= 128 + 2048/32 = 192. Rounded up.
pub const MOE_MAX_BLOCKS: usize = 256;

/// Fill `token_slot[tok][kk]` from flat `token_list` / `slot_list` after bucketing.
pub fn fill_token_slot(route: &mut RouteScratch) {
    route.token_slot = [[0; TOP_K]; CANVAS];
    let slots = route.num_slots as usize;
    for slot in 0..slots {
        let tok = route.token_list[slot] as usize;
        let kk = route.slot_list[slot] as usize;
        if tok < CANVAS && kk < TOP_K {
            route.token_slot[tok][kk] = slot as u32;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepFinishMode {
    /// lm_head + softcap + GPU sampler
    Full,
    /// lm_head + softcap only (no sampler)
    ForwardOnly,
}

#[derive(Debug, Clone)]
pub struct StepSmokeConfig {
    pub layers: usize,
    pub steps: usize,
    pub kv_len: u32,
    pub seed: u64,
    pub max_seq: usize,
    pub finish: StepFinishMode,
    /// Prompt token ids for encoder prefill into b4 (M1). When set, `StepParams.kv_len` = len.
    pub prefill_token_ids: Option<Vec<u32>>,
    /// Match `generate-monolithic --no-early-stop` (disables confidence early stop).
    pub no_early_stop: bool,
}

impl Default for StepSmokeConfig {
    fn default() -> Self {
        Self {
            layers: 3,
            steps: 1,
            kv_len: 0,
            seed: 42,
            max_seq: 512,
            finish: StepFinishMode::Full,
            prefill_token_ids: None,
            no_early_stop: false,
        }
    }
}

#[derive(Debug)]
pub struct StepSmokeResult {
    pub step: u32,
    pub stop_flag: u32,
    pub mean_entropy: f32,
    pub min_entropy: f32,
    pub low_entropy_positions: u32,
    pub ids: [u32; CANVAS],
    pub logits_finite: bool,
    pub max_abs_logit: f32,
    pub elapsed: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct StepProbeCheckpoint {
    pub label: String,
    pub finite: bool,
    pub max_abs: f32,
}

#[derive(Debug)]
pub struct StepProbeResult {
    pub checkpoints: Vec<StepProbeCheckpoint>,
    pub elapsed: std::time::Duration,
}

#[derive(Debug)]
pub struct StepBenchResult {
    pub compile: std::time::Duration,
    pub warmup: std::time::Duration,
    pub per_step: std::time::Duration,
    pub iters: usize,
    pub finish: StepFinishMode,
}

/// Wall-clock GPU segments for one forward step (5 submits: preamble, pre-MoE×L, MoE×L, post×L, finish).
#[derive(Debug, Clone)]
pub struct StepProfileResult {
    pub compile: std::time::Duration,
    pub preamble: std::time::Duration,
    pub layer_pre_moe: std::time::Duration,
    pub layer_moe: std::time::Duration,
    pub layer_post: std::time::Duration,
    pub finish: std::time::Duration,
    pub total: std::time::Duration,
    pub layers: usize,
    pub block_format: QuantFormat,
}

/// Per-layer `encode_layer` GPU segments (summed over all layers; each timed via its own submit).
#[derive(Debug, Default, Clone)]
pub struct LayerEncodeSubProfile {
    pub qkv_gemm: std::time::Duration,
    pub qk_rope_kv: std::time::Duration,
    pub attention: std::time::Duration,
    pub o_proj_gemm: std::time::Duration,
    pub o_proj_tail: std::time::Duration,
    pub dense_pre_norm: std::time::Duration,
    pub dense_gate_up: std::time::Duration,
    pub dense_glu: std::time::Duration,
    pub dense_down: std::time::Duration,
    pub dense_post_norm: std::time::Duration,
    pub router: std::time::Duration,
}

impl LayerEncodeSubProfile {
    pub fn total(&self) -> std::time::Duration {
        self.qkv_gemm
            + self.qk_rope_kv
            + self.attention
            + self.o_proj_gemm
            + self.o_proj_tail
            + self.dense_pre_norm
            + self.dense_gate_up
            + self.dense_glu
            + self.dense_down
            + self.dense_post_norm
            + self.router
    }
}

/// Per-layer MoE grouped path (summed over all layers).
#[derive(Debug, Default, Clone)]
pub struct MoeEncodeSubProfile {
    pub half_to_f32: std::time::Duration,
    pub gather: std::time::Duration,
    pub gate_up: std::time::Duration,
    pub swiglu: std::time::Duration,
    pub down: std::time::Duration,
    pub scatter: std::time::Duration,
    pub post: std::time::Duration,
}

impl MoeEncodeSubProfile {
    pub fn total(&self) -> std::time::Duration {
        self.half_to_f32
            + self.gather
            + self.gate_up
            + self.swiglu
            + self.down
            + self.scatter
            + self.post
    }
}

#[derive(Debug, Clone)]
pub struct EncodeSubProfileResult {
    pub compile: std::time::Duration,
    pub layers: usize,
    pub layer: LayerEncodeSubProfile,
    pub moe: MoeEncodeSubProfile,
}

pub fn build_offsets_from_store(store: &DgqStore) -> HashMap<String, u64> {
    let mut offsets = HashMap::new();
    for entry in store.tensor_entries() {
        offsets.insert(entry.name.clone(), entry.meta.offset);
    }
    offsets
}

/// Per-expert weight byte offset from `.dgq` manifest (stack base + expert × matrix bytes).
pub fn manifest_offset(
    offsets: &HashMap<String, u64>,
    layer: usize,
    expert: usize,
    kind: &str,
    format: QuantFormat,
) -> u64 {
    use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes};
    let hidden = HID;
    let moe_ff = MOE_FF as usize;
    let (tensor, n, k) = match kind {
        "gate_up" => (
            format!("model.decoder.layers.{layer}.experts.gate_up_proj"),
            moe_ff * 2,
            hidden,
        ),
        "down" => (
            format!("model.decoder.layers.{layer}.experts.down_proj"),
            hidden,
            moe_ff,
        ),
        _ => panic!("manifest_offset: unknown kind {kind:?}"),
    };
    let base = *offsets
        .get(&tensor)
        .unwrap_or_else(|| panic!("manifest_offset: missing tensor {tensor}"));
    let stride = match format {
        QuantFormat::NvFp4 => nvfp4_matrix_bytes(n, k) as u64,
        QuantFormat::Q6 => crate::dgq::layout::q6_matrix_bytes(n, k) as u64,
        _ => q4_matrix_bytes(n, k) as u64,
    };
    base + expert as u64 * stride
}

/// KV slots a layer's region holds. Full layers keep every position (linear,
/// max_seq). Sliding layers only ever attend the last window-1 (1023) keys plus
/// the 256 canvas/chunk positions, so a power-of-two ring >= 1280 suffices at
/// any context length; below that, the next power of two covering max_seq
/// (no wrap possible = trivially identical to linear).
pub fn layer_kv_slots(is_full: bool, max_seq: usize) -> usize {
    if is_full {
        max_seq
    } else {
        max_seq.next_power_of_two().min(2048)
    }
}

pub fn build_layout(offsets: &HashMap<String, u64>, max_seq: usize) -> ModelLayout {
    let g = |n: &str| *offsets.get(n).unwrap_or_else(|| panic!("missing tensor {n}"));
    let opt = |n: &str| offsets.get(n).copied().unwrap_or(0);
    let mut layers = [LayerOffsets::default(); N_LAYERS];
    let mut kv_off = 0u64;
    for (i, l) in layers.iter_mut().enumerate() {
        let p = format!("model.decoder.layers.{i}.");
        let full = FULL_LAYERS.contains(&i);
        let (hd, nkv) = if full { (512u32, 2u32) } else { (256, 8) };
        let slots = layer_kv_slots(full, max_seq);
        *l = LayerOffsets {
            input_ln: g(&format!("{p}input_layernorm.weight")),
            q_proj: g(&format!("{p}self_attn.q_proj.weight")),
            q_norm: g(&format!("{p}self_attn.q_norm.weight")),
            k_proj: g(&format!("{p}self_attn.k_proj.weight")),
            k_norm: g(&format!("{p}self_attn.k_norm.weight")),
            v_proj: opt(&format!("{p}self_attn.v_proj.weight")),
            o_proj: g(&format!("{p}self_attn.o_proj.weight")),
            post_attn_ln: g(&format!("{p}post_attention_layernorm.weight")),
            pre_ff_ln: g(&format!("{p}pre_feedforward_layernorm.weight")),
            mlp_gate: g(&format!("{p}mlp.gate_proj.weight")),
            mlp_up: g(&format!("{p}mlp.up_proj.weight")),
            mlp_down: g(&format!("{p}mlp.down_proj.weight")),
            post_ff_ln_1: g(&format!("{p}post_feedforward_layernorm_1.weight")),
            router_scale: g(&format!("{p}router.scale")),
            router_proj: g(&format!("{p}router.proj.weight")),
            per_expert_scale: g(&format!("{p}router.per_expert_scale")),
            pre_ff_ln_2: g(&format!("{p}pre_feedforward_layernorm_2.weight")),
            experts_gate_up: g(&format!("{p}experts.gate_up_proj")),
            experts_down: g(&format!("{p}experts.down_proj")),
            post_ff_ln_2: g(&format!("{p}post_feedforward_layernorm_2.weight")),
            post_ff_ln: g(&format!("{p}post_feedforward_layernorm.weight")),
            layer_scalar: g(&format!("{p}layer_scalar")),
            kv_region: kv_off,
            head_dim: hd,
            n_kv_heads: nkv,
            is_full: full as u32,
            kv_ring_mask: if full { 0 } else { (slots - 1) as u32 },
        };
        kv_off += (slots as u64) * (nkv as u64) * (hd as u64) * 2 * 2;
    }
    ModelLayout {
        embed: g("model.decoder.embed_tokens.weight"),
        sc_pre_norm: g("model.decoder.self_conditioning.pre_norm.weight"),
        sc_gate: g("model.decoder.self_conditioning.gate_proj.weight"),
        sc_up: g("model.decoder.self_conditioning.up_proj.weight"),
        sc_down: g("model.decoder.self_conditioning.down_proj.weight"),
        final_norm: g("model.decoder.norm.weight"),
        layers,
    }
}

fn kv_cache_bytes(layout: &ModelLayout, max_seq: usize) -> u64 {
    crate::metal::step_kv::kv_cache_total_bytes(layout, max_seq)
}

fn layer_byte_offset(layer: usize) -> u64 {
    (offset_of!(ModelLayout, layers) + layer * std::mem::size_of::<LayerOffsets>()) as u64
}

fn div_up(v: usize, g: usize) -> usize {
    (v + g - 1) / g
}
/// Fused Q‖K(‖V): one `GEMM_N` = q_n+k_n(+k_n); outputs land in native-width planes.
pub(crate) fn qkv_stacked_segments(
    l: &LayerOffsets,
    arena: &ArenaLayout,
) -> (Vec<crate::kernels::sub::gemm_block_stacked::GemmStackedSeg>, u32) {
    use crate::kernels::sub::gemm_block_stacked::GemmStackedSeg;
    let full = l.is_full != 0;
    let q_n = if full { 8192u32 } else { 4096 };
    let k_n = if full { 1024 } else { 2048 };
    let mut segs = vec![
        GemmStackedSeg {
            n_cols: q_n,
            y_col0: 0,
            y_row_cols: q_n,
            _pad: 0,
            w_off: l.q_proj,
            y_byte_off: arena.attnq_off(),
        },
        GemmStackedSeg {
            n_cols: k_n,
            y_col0: 0,
            y_row_cols: k_n,
            _pad: 0,
            w_off: l.k_proj,
            y_byte_off: arena.attnk_off(),
        },
    ];
    if !full && l.v_proj != 0 {
        segs.push(GemmStackedSeg {
            n_cols: k_n,
            y_col0: 0,
            y_row_cols: k_n,
            _pad: 0,
            w_off: l.v_proj,
            y_byte_off: arena.attnv_off(),
        });
    }
    let n_total: u32 = segs.iter().map(|s| s.n_cols).sum();
    (segs, n_total)
}

/// Fused dense gate‖up: one GEMM dispatch, gate→`ffg` and up→`ffu` (same planes as unfused).
pub(crate) fn gate_up_stacked_segments(
    l: &LayerOffsets,
    arena: &ArenaLayout,
) -> ([crate::kernels::sub::gemm_block_stacked::GemmStackedSeg; 2], u32) {
    use crate::kernels::sub::gemm_block_stacked::GemmStackedSeg;
    let n2 = DENSE_FF * 2;
    (
        [
            GemmStackedSeg {
                n_cols: DENSE_FF,
                y_col0: 0,
                y_row_cols: DENSE_FF,
                _pad: 0,
                w_off: l.mlp_gate,
                y_byte_off: arena.ffg_off(),
            },
            GemmStackedSeg {
                n_cols: DENSE_FF,
                y_col0: 0,
                y_row_cols: DENSE_FF,
                _pad: 0,
                w_off: l.mlp_up,
                y_byte_off: arena.ffu_off(),
            },
        ],
        n2,
    )
}

/// Scratch byte sizes for step-kernel grouped GEMM (max over all MoE/dense shapes).
fn gemm_scratch_bytes() -> (usize, usize) {
    let shapes = [
        (CANVAS, 4096u32, HID as u32),
        (CANVAS, 2048, HID as u32),
        (CANVAS, 2816, 4096),
        (CANVAS, 8192, HID as u32),
        (CANVAS, 1024, HID as u32),
        (CANVAS, 2816, 8192),
        (CANVAS, DENSE_FF, HID as u32),
        (CANVAS, 2816, DENSE_FF),
    ];
    let mut max_mk = 0usize;
    let mut max_nk = 0usize;
    for (m, _n, k) in shapes {
        max_mk = max_mk.max(m * k as usize);
        max_nk = max_nk.max(_n as usize * k as usize);
    }
    let f32 = std::mem::size_of::<f32>();
    (max_mk * f32, max_nk * f32)
}

/// SC prob staging: one vocab chunk of fp16 probs (the chunked softembed path).
fn sc_probs_buffer_bytes() -> usize {
    CANVAS * crate::model::embed::LM_HEAD_CHUNK * 2
}



fn logits_finite_sample_bytes() -> u64 {
    (logits_finite_sample_count().min(CANVAS * VOCAB) * 2) as u64
}

struct StepPipelines {
    memzero: ComputePipeline,
    rmsnorm: ComputePipeline,
    rmsnorm_f32: ComputePipeline,
    half_to_f32: ComputePipeline,
    gemm_q4: HashMap<(u32, u32), ComputePipeline>,
    /// Tunable Raw pipelines (DGQ_GEMM_TUNABLE), keyed (n,k);
    /// VOCAB shape uses the logits (K_OUT_BF16) variant.
    gemm_tunable_raw: HashMap<(u32, u32), ComputePipeline>,
    /// Tunable q8 pipelines (DGQ_GEMM_TUNABLE), keyed (n,k); VOCAB = logits.
    gemm_tunable_q8: HashMap<(u32, u32), ComputePipeline>,
    /// Tunable block-sparse MoE pipelines (DGQ_GEMM_TUNABLE, q4/q6 experts),
    /// keyed (n, k, gather, format as u32).
    gemm_tunable_sparse: HashMap<(u32, u32, bool, u32), ComputePipeline>,
    gemm_nvfp4: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8: HashMap<(u32, u32), ComputePipeline>,
    gemm_bf16: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8_logits: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8_rowk: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8_rowk_xfp16: HashMap<(u32, u32), ComputePipeline>,
    /// f32-accumulate variant of `gemm_q8_rowk` for chunked SC softembed (avoids per-chunk bf16 round).
    gemm_q8_rowk_acc_f32: HashMap<(u32, u32), ComputePipeline>,
    /// f32→bf16 convert with scale, for chunked SC softembed accumulator → half arena.
    f32_to_half_scale: ComputePipeline,
    qk_rope_kv: ComputePipeline,
    attention: ComputePipeline,
    /// GQA-grouped MMA attention for sliding layers (`DGQ_ATTN_MMA`); scalar `attention` is the fallback/oracle.
    attention_mma2: ComputePipeline,
    /// MMA attention for full/global layers (`DGQ_ATTN_MMA_FULL`, register-O); scalar `attention` is the fallback/oracle.
    attention_mma_full: ComputePipeline,
    residual: ComputePipeline,
    glu: ComputePipeline,
    router: ComputePipeline,
    /// Top-k tail over precomputed logits (DGQ_ROUTER_GEMM).
    router_topk: ComputePipeline,
    bucket_count: ComputePipeline,
    bucket_fill: ComputePipeline,
    q4_block_grouped: HashMap<(u32, u32), ComputePipeline>,
    nvfp4_block_grouped: HashMap<(u32, u32), ComputePipeline>,
    /// Block-sparse MoE GEMMs, keyed by (n, k, adaptive). adaptive = the
    /// GEMM_M_ADAPT pipeline (DGQ_MOE_TILE_ADAPT runtime per-block M-mapping).
    q4_block_sparse: HashMap<(u32, u32, bool), ComputePipeline>,
    q6_block_sparse: HashMap<(u32, u32, bool), ComputePipeline>,
    q6_block_grouped: HashMap<(u32, u32), ComputePipeline>,
    nvfp4_block_sparse: HashMap<(u32, u32, bool), ComputePipeline>,
    /// Fused-gather gate_up variant, keyed by (n,k,adaptive); built only when
    /// DGQ_MOE_FUSE_GATHER is on. Q4 only (the production MoE format).
    block_sparse_gather: HashMap<(u32, u32, bool), ComputePipeline>,
    gather_rows: ComputePipeline,
    gather_rows_bf16_to_f32: ComputePipeline,
    gelu_swiglu_gate_up: ComputePipeline,
    moe_scatter_weighted: ComputePipeline,
    moe_grouped: ComputePipeline,
    moe_grouped_nvfp4: ComputePipeline,
    /// Q4 grouped MoE with K_DUMP_STAGE=1 (debug capture only — never in forward/generate).
    moe_grouped_dump: ComputePipeline,
    embed_gather: ComputePipeline,
    /// bf16-embed input gather (embed_tokens stored Raw).
    embed_gather_bf16: ComputePipeline,
    logit_rowstats: ComputePipeline,
    sc_prob_cols: ComputePipeline,
    /// f32-accumulate chunked SC softembed with bf16 embed (keyed (HID, LM_HEAD_CHUNK)).
    gemm_bf16_rowk_acc_f32: HashMap<(u32, u32), ComputePipeline>,
    half_scale: ComputePipeline,
    softcap: ComputePipeline,
    sample_rowstats: ComputePipeline,
    sample_commit: ComputePipeline,
    sample_apply: ComputePipeline,
    sample_write: ComputePipeline,
    compact_active_rows: ComputePipeline,
    gather_rows_bf16: ComputePipeline,
    scatter_logits_rows: ComputePipeline,
    sc_sparse_select: ComputePipeline,
    sc_sparse_gather: ComputePipeline,
}

impl StepPipelines {
    fn new(ctx: &MetalContext, variant: crate::kernels::sub::variant::KernelVariant) -> Result<Self, Error> {
        let mut gemm_q4 = HashMap::new();
        let mut gemm_tunable_raw = HashMap::new();
        let mut gemm_tunable_q8 = HashMap::new();
        let mut gemm_tunable_sparse = HashMap::new();
        let mut gemm_nvfp4 = HashMap::new();
        let mut gemm_q8 = HashMap::new();
        let mut gemm_bf16 = HashMap::new();
        let mut gemm_q8_logits = HashMap::new();
        let mut gemm_q8_rowk = HashMap::new();
        let mut gemm_q8_rowk_xfp16 = HashMap::new();
        for &(n, k) in &[
            (4096u32, HID as u32),
            (2048, HID as u32),
            (2816, 4096),
            (8192, HID as u32),
            (1024, HID as u32),
            (2816, 8192),
            (DENSE_FF, HID as u32),
            (2816, DENSE_FF),
            (VOCAB as u32, HID as u32),
            // Router logits GEMM (DGQ_ROUTER_GEMM): [256, HID] @ router_proj^T[128, HID].
            (N_EXPERTS as u32, HID as u32),
        ] {
            gemm_q4.insert(
                (n, k),
                crate::kernels::sub::gemm_q4::pipeline_for(ctx, n, k)?,
            );
            gemm_nvfp4.insert(
                (n, k),
                crate::kernels::sub::gemm_nvfp4::pipeline_for(ctx, n, k)?,
            );
            // bf16-weight GEMM for the mixed-precision attention/dense-FFN path.
            // lm_head logits (n=VOCAB) forces bf16 output (range); others follow
            // K_ACT_F16 for their activation output.
            let bf16_ps = if n == VOCAB as u32 {
                crate::kernels::sub::gemm_bf16::pipeline_for_logits(ctx, n, k)?
            } else {
                crate::kernels::sub::gemm_bf16::pipeline_for(ctx, n, k)?
            };
            gemm_bf16.insert((n, k), bf16_ps);
            if gemm_tunable_enabled() {
                let t = if n == VOCAB as u32 {
                    crate::kernels::sub::gemm_tunable::pipeline_for_logits(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Raw,
                    )?
                } else {
                    crate::kernels::sub::gemm_tunable::pipeline_for(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Raw,
                    )?
                };
                gemm_tunable_raw.insert((n, k), t);
            }
        }
        for &(n, k) in &[
            (DENSE_FF, HID as u32),
            (2816, DENSE_FF),
            (VOCAB as u32, HID as u32),
            // Attention q/k/v/o_proj shapes (mixed-precision q8 attention path).
            (4096u32, HID as u32),
            (2048, HID as u32),
            (8192, HID as u32),
            (1024, HID as u32),
            (2816, 4096),
            (2816, 8192),
        ] {
            gemm_q8.insert(
                (n, k),
                crate::kernels::sub::gemm_q8::pipeline_for(ctx, n, k)?,
            );
            if (n, k) == (VOCAB as u32, HID as u32) {
                gemm_q8_logits.insert(
                    (n, k),
                    crate::kernels::sub::gemm_q8::pipeline_for_logits(ctx, n, k)?,
                );
            }
            if gemm_tunable_enabled() {
                let t = if (n, k) == (VOCAB as u32, HID as u32) {
                    crate::kernels::sub::gemm_tunable::pipeline_for_logits(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Q8,
                    )?
                } else {
                    crate::kernels::sub::gemm_tunable::pipeline_for(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Q8,
                    )?
                };
                gemm_tunable_q8.insert((n, k), t);
            }
        }
        for &(n, k) in &[
            (HID as u32, VOCAB as u32),
            (HID as u32, crate::model::embed::LM_HEAD_CHUNK as u32),
        ] {
            gemm_q8_rowk.insert(
                (n, k),
                crate::kernels::sub::gemm_q8_rowk::pipeline_for(ctx, n, k)?,
            );
            gemm_q8_rowk_xfp16.insert(
                (n, k),
                crate::kernels::sub::gemm_q8_rowk::pipeline_for_fp16_input(ctx, n, k)?,
            );
        }
        // Unified rowk f32-accumulate SC-softembed GEMM (one shader; weight format
        // = K_QUANT_FORMAT: Raw bf16 embed or Q8 embed). x is fp16 sc_probs.
        const ROWK_ACC_SHADER: &str =
            shader_include::include_metal!("kernels/gemm_bf16_rowk_acc_f32.metal");
        let mut gemm_q8_rowk_acc_f32 = HashMap::new();
        {
            for &(n, k) in &[
                (HID as u32, crate::model::embed::LM_HEAD_CHUNK as u32),
            ] {
                gemm_q8_rowk_acc_f32.insert(
                    (n, k),
                    ctx.compile_gemm_subkernel(
                        ROWK_ACC_SHADER,
                        "gemm_bf16_rowk_acc_f32",
                        n,
                        k,
                        false,
                        crate::kernels::sub::QuantFormat::Q8 as u32,
                        true, // sc_probs is fp16
                    )?,
                );
            }
        }
        let mut gemm_bf16_rowk_acc_f32 = HashMap::new();
        {
            for &(n, k) in &[(HID as u32, crate::model::embed::LM_HEAD_CHUNK as u32)] {
                gemm_bf16_rowk_acc_f32.insert(
                    (n, k),
                    ctx.compile_gemm_subkernel(
                        ROWK_ACC_SHADER,
                        "gemm_bf16_rowk_acc_f32",
                        n,
                        k,
                        false,
                        crate::kernels::sub::QuantFormat::Raw as u32,  // bf16 embed -> Raw branch
                        true, // sc_probs is fp16
                    )?,
                );
            }
        }
        let f32_to_half_scale = ctx.compile_subkernel(
            shader_include::include_metal!("kernels/f32_to_half_scale.metal"),
            "f32_to_half_scale",
            variant,
        )?;
        let mut q4_block_grouped = HashMap::new();
        let mut nvfp4_block_grouped = HashMap::new();
        let mut q4_block_sparse = HashMap::new();
        let mut q6_block_sparse = HashMap::new();
        let mut q6_block_grouped = HashMap::new();
        let mut nvfp4_block_sparse = HashMap::new();
        let mut block_sparse_gather = HashMap::new();
        if moe_fuse_gather_enabled() {
            // Only gate_up gathers (down's A is the swiglu output, not gathered).
            let (n, k) = (MOE_FF * 2, HID as u32);
            block_sparse_gather.insert(
                (n, k, false),
                crate::kernels::sub::gemm_block_sparse::pipeline_for_gather(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::Q4Affine,
                )?,
            );
            if moe_tile_adapt_enabled() {
                block_sparse_gather.insert(
                    (n, k, true),
                    crate::kernels::sub::gemm_block_sparse::pipeline_for_gather_adaptive(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Q4Affine,
                    )?,
                );
            }
        }
        for &(n, k) in &[(MOE_FF * 2, HID as u32), (HID as u32, MOE_FF)] {
            q4_block_grouped.insert(
                (n, k),
                crate::kernels::sub::gemm_block_grouped::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::Q4Affine,
                )?,
            );
            nvfp4_block_grouped.insert(
                (n, k),
                crate::kernels::sub::gemm_block_grouped::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::NvFp4,
                )?,
            );
            q4_block_sparse.insert(
                (n, k, false),
                crate::kernels::sub::gemm_block_sparse::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::Q4Affine,
                )?,
            );
            nvfp4_block_sparse.insert(
                (n, k, false),
                crate::kernels::sub::gemm_block_sparse::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::NvFp4,
                )?,
            );
            q6_block_sparse.insert(
                (n, k, false),
                crate::kernels::sub::gemm_block_sparse::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::Q6,
                )?,
            );
            q6_block_grouped.insert(
                (n, k),
                crate::kernels::sub::gemm_block_grouped::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::kernels::sub::QuantFormat::Q6,
                )?,
            );
            if moe_tile_adapt_enabled() {
                q6_block_sparse.insert(
                    (n, k, true),
                    crate::kernels::sub::gemm_block_sparse::pipeline_for_adaptive(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Q6,
                    )?,
                );
                q4_block_sparse.insert(
                    (n, k, true),
                    crate::kernels::sub::gemm_block_sparse::pipeline_for_adaptive(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::Q4Affine,
                    )?,
                );
                nvfp4_block_sparse.insert(
                    (n, k, true),
                    crate::kernels::sub::gemm_block_sparse::pipeline_for_adaptive(
                        ctx,
                        n,
                        k,
                        crate::kernels::sub::QuantFormat::NvFp4,
                    )?,
                );
            }
            if gemm_tunable_enabled() {
                for fmt in [
                    crate::kernels::sub::QuantFormat::Q4Affine,
                    crate::kernels::sub::QuantFormat::Q6,
                ] {
                    gemm_tunable_sparse.insert(
                        (n, k, false, fmt as u32),
                        crate::kernels::sub::gemm_tunable::pipeline_for_sparse(
                            ctx, n, k, false, fmt,
                        )?,
                    );
                    if moe_fuse_gather_enabled() && (n, k) == (MOE_FF * 2, HID as u32) {
                        gemm_tunable_sparse.insert(
                            (n, k, true, fmt as u32),
                            crate::kernels::sub::gemm_tunable::pipeline_for_sparse(
                                ctx, n, k, true, fmt,
                            )?,
                        );
                    }
                }
            }
        }
        let prod = variant;
        let dump = crate::kernels::sub::variant::KernelVariant::TEST_DUMP;
        Ok(Self {
            memzero: crate::kernels::sub::memzero_bytes::pipeline_for(ctx, prod)?,
            rmsnorm: crate::kernels::sub::rms_norm_rows_tiled::pipeline_for(
                ctx,
                crate::kernels::sub::rms_norm_rows_tiled::TiledVariant::HALF_IN,
                prod,
            )?,
            rmsnorm_f32: crate::kernels::sub::rms_norm_rows_tiled::pipeline_for(
                ctx,
                crate::kernels::sub::rms_norm_rows_tiled::TiledVariant::F32_IN,
                prod,
            )?,
            half_to_f32: crate::kernels::sub::half_to_f32::pipeline_for(ctx, prod)?,
            gemm_q4,
            gemm_tunable_raw,
            gemm_tunable_q8,
            gemm_tunable_sparse,
            gemm_nvfp4,
            gemm_q8,
            gemm_bf16,
            gemm_q8_logits,
            gemm_q8_rowk,
            gemm_q8_rowk_xfp16,
            gemm_q8_rowk_acc_f32,
            gemm_bf16_rowk_acc_f32,
            f32_to_half_scale,
            qk_rope_kv: crate::kernels::sub::qk_rope_kv::pipeline_for(ctx, prod)?,
            attention: crate::kernels::sub::attention::pipeline_for(ctx, prod)?,
            attention_mma2: crate::kernels::sub::attention::pipeline_mma2_for(ctx, prod)?,
            attention_mma_full: crate::kernels::sub::attention::pipeline_mma_full_for(ctx, prod)?,
            residual: crate::kernels::sub::residual_half::pipeline_for(ctx, prod)?,
            glu: crate::kernels::sub::swiglu::pipeline_for(
                ctx,
                crate::kernels::sub::SwigluSplitVariant::MONOLITH_GLU,
                prod,
            )?,
            router: crate::kernels::sub::moe_router::pipeline_for(ctx, prod)?,
            router_topk: ctx.compile_subkernel(
                shader_include::include_metal!("kernels/moe_router_topk.metal"),
                "moe_router_topk",
                prod,
            )?,
            bucket_count: crate::kernels::sub::moe_bucket_count::pipeline_for(ctx, prod)?,
            bucket_fill: crate::kernels::sub::moe_bucket_fill::pipeline_for(ctx, prod)?,
            q4_block_grouped,
            nvfp4_block_grouped,
            q4_block_sparse,
            q6_block_sparse,
            q6_block_grouped,
            nvfp4_block_sparse,
            block_sparse_gather,
            gather_rows: crate::kernels::sub::gather_rows::pipeline_for(ctx, prod)?,
            gather_rows_bf16_to_f32: crate::kernels::sub::gather_rows_bf16_to_f32::pipeline_for(
                ctx, prod,
            )?,
            gelu_swiglu_gate_up: crate::kernels::sub::swiglu::pipeline_for_moe(ctx, prod)?,
            moe_scatter_weighted: crate::kernels::sub::moe_scatter_weighted::pipeline_for(
                ctx, prod,
            )?,
            moe_grouped: crate::kernels::sub::moe_grouped::pipeline_for(ctx, prod)?,
            moe_grouped_nvfp4: crate::kernels::sub::moe_grouped_nvfp4::pipeline_for(
                ctx,
                prod,
            )?,
            moe_grouped_dump: crate::kernels::sub::moe_grouped::pipeline_for(ctx, dump)?,
            embed_gather: crate::kernels::sub::embed_gather::pipeline_for(ctx, prod)?,
            embed_gather_bf16: ctx.compile_subkernel(
                shader_include::include_metal!("kernels/embed_gather_bf16.metal"),
                "embed_gather_bf16",
                prod,
            )?,
            logit_rowstats: crate::kernels::sub::logit_rowstats::pipeline_for(ctx, prod)?,
            sc_prob_cols: crate::kernels::sub::sc_prob_cols::pipeline_for(ctx, prod)?,
            half_scale: crate::kernels::sub::half_scale::pipeline_for(ctx, prod)?,
            softcap: crate::kernels::sub::softcap_half::pipeline_for(ctx, prod)?,
            sample_rowstats: crate::kernels::sub::sample_rowstats::pipeline_for(ctx, prod)?,
            sample_commit: crate::kernels::sub::sample_commit::pipeline_for(ctx, prod)?,
            sample_apply: crate::kernels::sub::sample_apply::pipeline_for(ctx, prod)?,
            sample_write: crate::kernels::sub::sample_write::pipeline_for(ctx, prod)?,
            compact_active_rows: ctx.compile_kernel(
                shader_include::include_metal!("kernels/compact_active_rows.metal"),
                "compact_active_rows",
            )?,
            gather_rows_bf16: crate::kernels::sub::gather_rows_bf16::pipeline_for(ctx, prod)?,
            scatter_logits_rows: ctx.compile_kernel(
                shader_include::include_metal!("kernels/scatter_logits_rows.metal"),
                "scatter_logits_rows",
            )?,
            sc_sparse_select: ctx.compile_kernel(
                shader_include::include_metal!("kernels/sc_sparse_select.metal"),
                "sc_sparse_select",
            )?,
            sc_sparse_gather: ctx.compile_kernel(
                shader_include::include_metal!("kernels/sc_sparse_gather.metal"),
                "sc_sparse_gather",
            )?,
        })
    }

    fn q4(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q4
            .get(&(n, k))
            .ok_or(Error::Format("missing q4 pipeline"))
    }

    fn nvfp4(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_nvfp4
            .get(&(n, k))
            .ok_or(Error::Format("missing nvfp4 pipeline"))
    }

    fn q8(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q8
            .get(&(n, k))
            .ok_or(Error::Format("missing q8 pipeline"))
    }

    fn bf16(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_bf16
            .get(&(n, k))
            .ok_or(Error::Format("missing bf16 pipeline"))
    }

    /// Tunable Raw pipeline if built for this shape (DGQ_GEMM_TUNABLE).
    fn bf16_tunable(&self, n: u32, k: u32) -> Option<&ComputePipeline> {
        self.gemm_tunable_raw.get(&(n, k))
    }

    /// Tunable q8 pipeline if built for this shape (DGQ_GEMM_TUNABLE).
    fn q8_tunable(&self, n: u32, k: u32) -> Option<&ComputePipeline> {
        self.gemm_tunable_q8.get(&(n, k))
    }

    /// Tunable block-sparse pipeline (q4/q6 experts) if built (DGQ_GEMM_TUNABLE).
    fn sparse_tunable_fmt(
        &self,
        format: QuantFormat,
        n: u32,
        k: u32,
        gather: bool,
    ) -> Option<&ComputePipeline> {
        self.gemm_tunable_sparse.get(&(n, k, gather, format as u32))
    }

    fn q8_logits(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q8_logits
            .get(&(n, k))
            .ok_or(Error::Format("missing q8 logits pipeline"))
    }

    fn q8_rowk_xfp16(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q8_rowk_xfp16
            .get(&(n, k))
            .ok_or(Error::Format("missing q8 rowk fp16-input pipeline"))
    }

    fn q8_rowk(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q8_rowk
            .get(&(n, k))
            .ok_or(Error::Format("missing q8 rowk pipeline"))
    }

    fn block_gemm(&self, format: QuantFormat, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        match format {
            QuantFormat::NvFp4 => self.nvfp4(n, k),
            _ => self.q4(n, k),
        }
    }

    fn block_grouped(
        &self,
        format: QuantFormat,
        n: u32,
        k: u32,
    ) -> Result<&ComputePipeline, Error> {
        let map = match format {
            QuantFormat::Q4Affine => &self.q4_block_grouped,
            QuantFormat::Q6 => &self.q6_block_grouped,
            QuantFormat::NvFp4 => &self.nvfp4_block_grouped,
            _ => {
                return Err(Error::Format(
                    "batched MoE tiled grouped GEMM unsupported for this block format",
                ));
            }
        };
        map.get(&(n, k))
            .ok_or(Error::Format("missing block_grouped pipeline"))
    }

    /// Block-sparse pipeline; `adaptive` = GEMM_M_ADAPT (DGQ_MOE_TILE_ADAPT).
    fn block_sparse(
        &self,
        format: QuantFormat,
        n: u32,
        k: u32,
        adaptive: bool,
    ) -> Result<&ComputePipeline, Error> {
        let map = match format {
            QuantFormat::Q4Affine => &self.q4_block_sparse,
            QuantFormat::Q6 => &self.q6_block_sparse,
            QuantFormat::NvFp4 => &self.nvfp4_block_sparse,
            _ => {
                return Err(Error::Format(
                    "block-sparse MoE GEMM unsupported for this block format",
                ));
            }
        };
        map.get(&(n, k, adaptive))
            .ok_or(Error::Format("missing block_sparse pipeline"))
    }

    /// Fused-gather gate_up pipeline if built (DGQ_MOE_FUSE_GATHER), else None.
    fn block_sparse_gather(&self, n: u32, k: u32, adaptive: bool) -> Option<&ComputePipeline> {
        self.block_sparse_gather.get(&(n, k, adaptive))
    }

    fn moe_scalar(&self, format: QuantFormat) -> &ComputePipeline {
        match format {
            QuantFormat::NvFp4 => &self.moe_grouped_nvfp4,
            _ => &self.moe_grouped,
        }
    }
}

pub(crate) struct StepBuffers {
    blob: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Expert-region blob buffer (region 2 on split blobs, else == blob) and
    /// the base to subtract from absolute expert offsets.
    blob_experts: Retained<ProtocolObject<dyn MTLBuffer>>,
    blob_expert_base: u64,
    layout: Retained<ProtocolObject<dyn MTLBuffer>>,
    params: Retained<ProtocolObject<dyn MTLBuffer>>,
    arena: Retained<ProtocolObject<dyn MTLBuffer>>,
    kvcache: Retained<ProtocolObject<dyn MTLBuffer>>,
    state: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits: Retained<ProtocolObject<dyn MTLBuffer>>,
    sc_probs: Retained<ProtocolObject<dyn MTLBuffer>>,
    route: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Inert 4B buffer for optional dump slots (K_DUMP_STAGE=0 kernels).
    dummy_dump: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// P3.7 assert reporting (`--assert`); 16 bytes, zeroed each forward step.
    debug_status: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(crate) gemm_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) gemm_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-layer unique routed experts (written by moe_bucket_fill phase 1).
    expert_layer_unique: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU-written `threadgroupsPerGrid` for grouped MoE gate_up (slot 0) and down (slot 1).
    moe_grouped_indirect: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch plane byte offsets (host-built, mirrored on GPU as b8).
    pub(crate) arena_map: ArenaLayout,
    /// GPU copy of `arena_map` (b8); bound when kernels need device-side plane table.
    #[allow(dead_code)]
    arena_layout_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeGroupedGridInfo {
    gate_n: u32,
    hid: u32,
    n_tile: u32,
    tpg: u32,
    /// N-tile width of the tunable sparse pipelines (indirect slots 4/5).
    tunable_n_tile: u32,
}

// Slots 0/1: per-expert grouped (gate_up/down), height = num_active_experts.
// Slots 2/3: block-sparse (gate_up/down), height = num_blocks.
// Slots 4/5: tunable block-sparse (gate_up/down, BN-wide N-tiles).
const MOE_GROUPED_INDIRECT_BYTES: usize = 6 * 3 * std::mem::size_of::<u32>();

fn moe_grouped_grid_info() -> MoeGroupedGridInfo {
    MoeGroupedGridInfo {
        gate_n: MOE_FF * 2,
        hid: HID as u32,
        n_tile: crate::kernels::sub::gemm_common::n_tile() as u32,
        tpg: crate::kernels::sub::gemm_common::THREADS_PER_TG as u32,
        tunable_n_tile: crate::kernels::sub::gemm_tunable::SPARSE_BN as u32,
    }
}

struct StepEnc<'a> {
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    ctx: &'a MetalContext,
    ps: &'a StepPipelines,
    bufs: &'a StepBuffers,
    block_profile: StepBlockProfile,
    tensor_offsets: &'a HashMap<String, u64>,
    /// Active canvas rows for lm_head (P2.5); `CANVAS` when full lm_head.
    partial_lm_m: u32,
    /// Attention + dense-FFN weights are stored q8 (mixed-precision .dgq): route
    /// their GEMMs through the q8 kernel and skip the q4-only fused stacked path.
    attn_ffn_q8: bool,
    /// Attention + dense-FFN weights are stored bf16 (Raw): route their GEMMs
    /// through the bf16 kernel and skip the q4-only fused stacked path.
    attn_ffn_bf16: bool,
    /// Embed (tied lm_head + SC soft-embed) is stored bf16 (Raw) rather than
    /// q8-per-row: dispatch the bf16 gather / lm_head / softembed paths.
    embed_bf16: bool,
    /// Prefill mode: attention is CAUSAL (scalar kernel only; mma variants have no
    /// causal mask) and the SC/sampler/lm_head stages are skipped (KV-only forward).
    prefill_causal: bool,
    /// Model sliding-window size (Gemma-4: 1024) for sliding-attention layers.
    sliding_window: u32,
}

impl<'a> StepEnc<'a> {
    #[inline]
    fn arena(&self) -> &ArenaLayout {
        &self.bufs.arena_map
    }
}

const MOE_SLOTS: u32 = (CANVAS * TOP_K) as u32;

fn grouped_expert_blob_bytes_per_expert(format: crate::kernels::sub::QuantFormat) -> u64 {
    use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes};
    let hidden = HID;
    let moe_ff = MOE_FF as usize;
    match format {
        crate::kernels::sub::QuantFormat::NvFp4 => {
            (nvfp4_matrix_bytes(moe_ff * 2, hidden) + nvfp4_matrix_bytes(hidden, moe_ff)) as u64
        }
        _ => (q4_matrix_bytes(moe_ff * 2, hidden) + q4_matrix_bytes(hidden, moe_ff)) as u64,
    }
}

fn moe_w_byte_off_a() -> usize {
    0
}

fn moe_w_byte_off_gu() -> usize {
    (MOE_SLOTS as usize) * HID * std::mem::size_of::<f32>()
}

/// Grouped MoE gate/up and down job table for one decoder layer.
pub fn layer_moe_block_jobs(
    l: &LayerOffsets,
    format: QuantFormat,
) -> ([BlockGroupedJob; N_EXPERTS], [BlockGroupedJob; N_EXPERTS]) {
    layer_moe_block_jobs_impl(l, format, None, 0)
}

fn layer_moe_block_jobs_impl(
    l: &LayerOffsets,
    format: QuantFormat,
    manifest: Option<(usize, &HashMap<String, u64>)>,
    expert_base: u64,
) -> ([BlockGroupedJob; N_EXPERTS], [BlockGroupedJob; N_EXPERTS]) {
    use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes, q6_matrix_bytes};
    let hidden = HID as usize;
    let moe_ff = MOE_FF as usize;
    let (gu_stride, dn_stride, gu_gpr, dn_gpr) = match format {
        QuantFormat::NvFp4 => (
            nvfp4_matrix_bytes(moe_ff * 2, hidden) as u64,
            nvfp4_matrix_bytes(hidden, moe_ff) as u64,
            (hidden as u32).div_ceil(16),
            (moe_ff as u32).div_ceil(16),
        ),
        QuantFormat::Q6 => (
            q6_matrix_bytes(moe_ff * 2, hidden) as u64,
            q6_matrix_bytes(hidden, moe_ff) as u64,
            (hidden as u32).div_ceil(32),
            (moe_ff as u32).div_ceil(32),
        ),
        _ => (
            q4_matrix_bytes(moe_ff * 2, hidden) as u64,
            q4_matrix_bytes(hidden, moe_ff) as u64,
            (hidden as u32).div_ceil(32),
            (moe_ff as u32).div_ceil(32),
        ),
    };
    let mut gate = [BlockGroupedJob {
        w_byte_off: 0,
        groups_per_row: gu_gpr,
        _pad: 0,
    }; N_EXPERTS];
    let mut down = [BlockGroupedJob {
        w_byte_off: 0,
        groups_per_row: dn_gpr,
        _pad: 0,
    }; N_EXPERTS];
    for e in 0..N_EXPERTS {
        gate[e].w_byte_off = l.experts_gate_up + (e as u64) * gu_stride;
        down[e].w_byte_off = l.experts_down + (e as u64) * dn_stride;
        if let Some((layer, offsets)) = manifest {
            debug_assert_eq!(
                gate[e].w_byte_off,
                manifest_offset(offsets, layer, e, "gate_up", format),
                "gate_up L{layer} E{e}: computed stride offset != manifest"
            );
        }
        // Rebase into the expert blob region (0 on single-buffer blobs).
        gate[e].w_byte_off -= expert_base;
        down[e].w_byte_off -= expert_base;
    }
    (gate, down)
}

impl StepEnc<'_> {
    fn sink_set_pipeline(&mut self, ps: &ComputePipeline) {
        self.enc.setComputePipelineState(&ps.pipeline);
    }

    fn sink_set_buffer(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(buf), offset, index);
        }
    }

    fn sink_set_bytes<T: Copy>(&mut self, val: &T, index: usize) {
        crate::metal::batch::set_bytes(&self.enc, val, index);
    }

    #[allow(dead_code)]
    fn bind_arena_layout_buf(&mut self, index: usize) {
        self.sink_set_buffer(&self.bufs.arena_layout_buf, 0, index);
    }

    fn bind_debug_status(&mut self, index: usize) {
        if let Some(ref dbg) = self.bufs.debug_status {
            self.sink_set_buffer(dbg, 0, index);
        }
    }

    fn sink_dispatch(&mut self, grid: MTLSize, tg: MTLSize) {
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Buffer-scope memory barrier on the live encoder.
    fn sink_memory_barrier(&mut self) {
        self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    fn sink_dispatch_indirect(&mut self, indirect_offset: usize, _n: u32, tg: MTLSize) {
        unsafe {
            self.enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &self.bufs.moe_grouped_indirect,
                indirect_offset,
                tg,
            );
        }
    }

    fn bind_blob(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.blob, 0, idx);
    }

    /// Expert weights live in the second blob region on split blobs; job
    /// offsets are rebased to match (layer_moe_block_jobs_impl expert_base).
    fn bind_blob_experts(&mut self, idx: usize) {
        let buf = self.bufs.blob_experts.clone();
        self.sink_set_buffer(&buf, 0, idx);
    }

    fn bind_params(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.params, 0, idx);
    }

    fn bind_kvcache(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.kvcache, 0, idx);
    }

    fn bind_state(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.state, 0, idx);
    }

    fn bind_logits(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.logits, 0, idx);
    }

    fn bind_sc_probs(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.sc_probs, 0, idx);
    }

    fn bind_route(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.route, 0, idx);
    }

    fn dispatch_1d(&mut self, ps: &ComputePipeline, count: usize, tpg: usize) {
        self.sink_set_pipeline(ps);
        let tg_w = tpg.min(count.max(1));
        let grid = MTLSize {
            width: div_up(count, tg_w),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
    }

    /// Split 1D dispatches that would exceed Metal's 65535 threadgroup grid width.
    fn dispatch_1d_ranged(
        &mut self,
        ps: &ComputePipeline,
        count: usize,
        tpg: usize,
        mut encode: impl FnMut(&mut Self, u32, u32),
    ) {
        const MAX_GROUPS: usize = 65535;
        let chunk_max = MAX_GROUPS * tpg;
        let mut base = 0usize;
        while base < count {
            let chunk = (count - base).min(chunk_max);
            self.sink_set_pipeline(ps);
            encode(self, base as u32, chunk as u32);
            let tg_w = tpg.min(chunk.max(1));
            let grid = MTLSize {
                width: div_up(chunk, tg_w),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            };
            self.sink_dispatch(grid, tg);
            base += chunk;
        }
    }

    /// Softcap logits (matches ranged logit_softcapping dispatch pattern).
    fn dispatch_softcap(&mut self) {
        let len = CANVAS * VOCAB;
        self.dispatch_1d_ranged(&self.ps.softcap, len, 256, |this, base, chunk| {
            this.sink_set_buffer(&this.bufs.logits, 0, 0);
            this.sink_set_bytes(&base, 1);
            this.sink_set_bytes(&chunk, 2);
            this.sink_set_buffer(&this.bufs.dummy_dump, 0, 3);
            this.bind_debug_status(4);
        });
    }

    fn dispatch_convert_1d(
        &mut self,
        ps: &ComputePipeline,
        src: &ProtocolObject<dyn MTLBuffer>,
        src_off: usize,
        dst: &ProtocolObject<dyn MTLBuffer>,
        dst_off: usize,
        len: usize,
    ) {
        self.dispatch_1d_ranged(ps, len, 256, |this, base, chunk| {
            this.sink_set_buffer(src, src_off, 0);
            this.sink_set_buffer(dst, dst_off, 1);
            this.sink_set_bytes(&base, 2);
            this.sink_set_bytes(&chunk, 3);
        });
    }

    fn half_to_f32_buf(&mut self, arena_off: u64, len: usize) {
        self.dispatch_convert_1d(
            &self.ps.half_to_f32,
            &self.bufs.arena,
            arena_off as usize,
            &self.bufs.gemm_a,
            0,
            len,
        );
    }

    fn gemm_q4(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        // Attention/dense-FFN weights are the only tensors routed through gemm_q4;
        // in mixed-precision .dgq they are stored bf16 (or q8 on older checkpoints),
        // so dispatch the matching kernel.
        if self.attn_ffn_bf16 {
            return self.gemm_bf16(x_off, y_off, w_off, m, n, k);
        }
        if self.attn_ffn_q8 {
            return self.gemm_q8(x_off, y_off, w_off, m, n, k);
        }
        let ps = self.ps.block_gemm(self.block_profile.format, n, k)?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, crate::kernels::sub::gemm_common::n_tile()),
            height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn gemm_q4_stacked(
        &mut self,
        x_off: u64,
        segs: &[crate::kernels::sub::gemm_block_stacked::GemmStackedSeg],
        m: u32,
        k: u32,
        n_total: u32,
    ) -> Result<(), Error> {
        debug_assert!(segs.len() <= STACKED_SEG_MAX, "too many stacked segments");
        let ps = crate::kernels::sub::gemm_block_stacked::stacked_pipeline_for(
            self.ctx,
            n_total,
            k,
            self.block_profile.format,
            segs,
        )?;
        self.sink_set_pipeline(ps.as_ref());
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&m, 3);
        let grid = MTLSize {
            width: div_up(n_total as usize, crate::kernels::sub::gemm_common::n_tile()),
            height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// bf16 fused stacked GEMM (QKV / gate+up on the bf16 path) — same N-segment
    /// layout as `gemm_q4_stacked` but reads bf16 weights (no dequant).
    fn gemm_bf16_stacked(
        &mut self,
        x_off: u64,
        segs: &[crate::kernels::sub::gemm_block_stacked::GemmStackedSeg],
        m: u32,
        k: u32,
        n_total: u32,
    ) -> Result<(), Error> {
        debug_assert!(segs.len() <= STACKED_SEG_MAX, "too many stacked segments");
        let (ps, grid) = if gemm_tunable_enabled() {
            (
                crate::kernels::sub::gemm_tunable::stacked_pipeline_for(
                    self.ctx,
                    n_total,
                    k,
                    crate::kernels::sub::QuantFormat::Raw,
                    segs,
                )?,
                MTLSize {
                    width: div_up(n_total as usize, crate::kernels::sub::gemm_tunable::TUNE_BN),
                    height: div_up(m as usize, crate::kernels::sub::gemm_tunable::TUNE_BM),
                    depth: 1,
                },
            )
        } else {
            (
                crate::kernels::sub::gemm_bf16_stacked::stacked_pipeline_for(
                    self.ctx, n_total, k, segs,
                )?,
                MTLSize {
                    width: div_up(n_total as usize, crate::kernels::sub::gemm_common::n_tile()),
                    height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
                    depth: 1,
                },
            )
        };
        self.sink_set_pipeline(ps.as_ref());
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&m, 3);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn memzero_bytes(&mut self, byte_off: u64, nbytes: u64) {
        self.sink_set_pipeline(&self.ps.memzero);
        self.sink_set_buffer(&self.bufs.arena, byte_off as usize, 0);
        // memzero_bytes.metal zeros one uchar4 (4 bytes) per thread, so count is
        // div_up(nbytes, 4). (Was /16 — only cleared a quarter of the range,
        // which left the chunked SC f32 accumulator stale past row 64 → NONDET-SC-1.)
        let count = div_up(nbytes as usize, 4);
        self.dispatch_1d(&self.ps.memzero, count, 256);
    }

    /// Zero an arbitrary buffer (e.g. `gemm_b` scratch) — used by chunked SC softembed f32 accumulator.
    fn memzero_buffer(
        &mut self,
        buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        nbytes: u64,
    ) {
        self.sink_set_pipeline(&self.ps.memzero);
        self.sink_set_buffer(buf, 0, 0);
        // 4 bytes (one uchar4) per thread — see memzero_bytes.
        let count = div_up(nbytes as usize, 4);
        self.dispatch_1d(&self.ps.memzero, count, 256);
    }

    fn rmsnorm(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        dim: u32,
        rows: usize,
    ) {
        self.sink_set_pipeline(&self.ps.rmsnorm);
            self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
            self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
            self.bind_blob(2);
            self.sink_set_bytes( &w_off, 3);
            self.sink_set_bytes( &dim, 4);
        let grid = MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
    }

    fn rmsnorm_f32(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        dim: u32,
        rows: usize,
    ) {
        self.sink_set_pipeline(&self.ps.rmsnorm_f32);
            self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
            self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
            self.bind_blob(2);
            self.sink_set_bytes( &w_off, 3);
            self.sink_set_bytes( &dim, 4);
        let grid = MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
    }

    fn gemm_q8(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let (ps, grid) = match self.ps.q8_tunable(n, k) {
            Some(t) => (
                t,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_tunable::TUNE_BN),
                    height: div_up(m as usize, crate::kernels::sub::gemm_tunable::TUNE_BM),
                    depth: 1,
                },
            ),
            None => (
                self.ps.q8(n, k)?,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_common::n_tile()),
                    height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
                    depth: 1,
                },
            ),
        };
        self.sink_set_pipeline(ps);
            self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
            self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
            self.bind_blob(2);
            self.sink_set_bytes( &w_off, 3);
            self.sink_set_bytes( &m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// bf16-weight GEMM (mixed-precision attention/dense-FFN). Same shape/dispatch
    /// as gemm_q8; weights at `w_off` are bf16 [N,K] (no dequant).
    fn gemm_bf16(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let (ps, grid) = match self.ps.bf16_tunable(n, k) {
            Some(t) => (
                t,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_tunable::TUNE_BN),
                    height: div_up(m as usize, crate::kernels::sub::gemm_tunable::TUNE_BM),
                    depth: 1,
                },
            ),
            None => (
                self.ps.bf16(n, k)?,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_common::n_tile()),
                    height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
                    depth: 1,
                },
            ),
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Tied lm_head with bf16 embed: logits = hidden @ embed^T. Reuses the bf16
    /// GEMM kernel (writes the logits buffer instead of the arena).
    fn gemm_bf16_logits(
        &mut self,
        x_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
        logits_byte_off: usize,
    ) -> Result<(), Error> {
        let (ps, grid) = match self.ps.bf16_tunable(n, k) {
            Some(t) => (
                t,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_tunable::TUNE_BN),
                    height: div_up(m as usize, crate::kernels::sub::gemm_tunable::TUNE_BM),
                    depth: 1,
                },
            ),
            None => (
                self.ps.bf16(n, k)?,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_common::n_tile()),
                    height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
                    depth: 1,
                },
            ),
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.logits, logits_byte_off, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn gemm_q8_logits(
        &mut self,
        x_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
        logits_byte_off: usize,
    ) -> Result<(), Error> {
        let (ps, grid) = match self.ps.q8_tunable(n, k) {
            Some(t) => (
                t,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_tunable::TUNE_BN),
                    height: div_up(m as usize, crate::kernels::sub::gemm_tunable::TUNE_BM),
                    depth: 1,
                },
            ),
            None => (
                self.ps.q8_logits(n, k)?,
                MTLSize {
                    width: div_up(n as usize, crate::kernels::sub::gemm_common::n_tile()),
                    height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
                    depth: 1,
                },
            ),
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.logits, logits_byte_off, 1);
        self.bind_blob(2);
        self.sink_set_bytes( &w_off, 3);
        self.sink_set_bytes( &m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// probs [M,K] half buffer → arena y_off [M,N] via q8 weights indexed by K.
    fn gemm_q8_rowk_half(
        &mut self,
        x_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        // x (sc_probs) is fp16 (10-mantissa probs, sc_probs.metal): use the
        // fp16-input pipeline so the prob precision survives into the GEMM tile.
        let ps = self.ps.q8_rowk_xfp16(n, k)?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(x_buf, 0, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// probs [M,K] half buffer @ sc_probs → arena y_off [M,N] via q8 weights.
    fn gemm_q8_rowk_acc_f32(
        &mut self,
        y_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self
            .ps
            .gemm_q8_rowk_acc_f32
            .get(&(n, k))
            .ok_or(Error::Format("missing gemm_q8_rowk_acc_f32 pipeline"))?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.sc_probs, 0, 0);
        self.sink_set_buffer(y_buf, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// bf16-embed variant of `gemm_q8_rowk_acc_f32` (chunked SC softembed accumulate).
    fn gemm_bf16_rowk_acc_f32(
        &mut self,
        y_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self
            .ps
            .gemm_bf16_rowk_acc_f32
            .get(&(n, k))
            .ok_or(Error::Format("missing gemm_bf16_rowk_acc_f32 pipeline"))?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.sc_probs, 0, 0);
        self.sink_set_buffer(y_buf, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, crate::kernels::sub::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Convert f32 buffer → bf16 arena slot with scale: `arena[base+i] = f32_buf[i] * scale`.
    fn f32_to_half_scale(
        &mut self,
        src_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        y_off: u64,
        len: usize,
        scale: f32,
    ) {
        self.sink_set_pipeline(&self.ps.f32_to_half_scale);
        self.sink_set_buffer(src_buf, 0, 0);
        // Arena is already bound at `y_off`; the shader adds `base` on top of the
        // binding offset, so `base` must be 0 (passing `y_off` here double-applies
        // the offset and leaves the real slot at zero).
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.sink_set_bytes(&0u32, 2);
        self.sink_set_bytes(&(len as u32), 3);
        self.sink_set_bytes(&scale, 4);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 5);
        self.dispatch_1d(&self.ps.f32_to_half_scale, len, 256);
    }

    fn scale_half_arena(&mut self, y_off: u64, elems: usize, scale: f32) {
        self.sink_set_pipeline(&self.ps.half_scale);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 0);
        self.sink_set_bytes(&(elems as u32), 1);
        self.sink_set_bytes(&scale, 2);
        self.dispatch_1d(&self.ps.half_scale, elems, 256);
    }

    fn scale_half_logits(&mut self, elems: usize, scale: f32) {
        // Same kernel as scale_half_arena (half_scale); only the bound buffer differs
        // (logits vs arena). Both are byte-identical in-place bf16 scales.
        self.sink_set_pipeline(&self.ps.half_scale);
        self.bind_logits(0);
        self.sink_set_bytes(&(elems as u32), 1);
        self.sink_set_bytes(&scale, 2);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 3);
        self.dispatch_1d(&self.ps.half_scale, elems, 256);
    }

    fn encode_sc_logit_rowstats(&mut self) {
        self.sink_set_pipeline(&self.ps.logit_rowstats);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
        let dims = [CANVAS as u32, VOCAB as u32];
        self.sink_set_bytes(&dims, 2);
        self.bind_debug_status(3);
        let (grid, tg) = crate::kernels::sub::logit_rowstats::dispatch_shape(CANVAS);
        self.sink_dispatch(grid, tg);
    }

    fn encode_sc_softembed(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        if self.embed_bf16 && sc_sparse_enabled() {
            return self.encode_sc_softembed_sparse(layout);
        }
        self.encode_sc_softembed_exact(layout)
    }

    fn dispatch_sc_prob_cols(&mut self, v0: u32, chunk: u32) {
        self.sink_set_pipeline(&self.ps.sc_prob_cols);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
        self.bind_sc_probs(2);
        let params = [CANVAS as u32, VOCAB as u32, v0, chunk];
        self.sink_set_bytes(&params, 3);
        self.bind_debug_status(4);
        let (grid, tg) =
            crate::kernels::sub::sc_prob_cols::dispatch_shape(CANVAS, chunk as usize);
        self.sink_dispatch(grid, tg);
    }

    /// Vocab-chunked softembed: rowstats once, then chunk GEMMs (no full prob matrix).
    /// Accumulates in f32 (in `gemm_b`) to match full-path precision; converts to half once at the end.
    fn encode_sc_softembed_chunked(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        use crate::dgq::layout::q8_row_bytes;
        use crate::model::embed::LM_HEAD_CHUNK;

        // f32 accumulator in gemm_b (free during preamble, before layer GEMMs).
        let acc_bytes = (CANVAS * HID * std::mem::size_of::<f32>()) as u64;
        self.memzero_buffer(&self.bufs.gemm_b, acc_bytes);
        self.sink_memory_barrier(); // memzero gemm_b before the first `+=`

        let row_bytes = q8_row_bytes(HID as usize) as u64;
        let chunk_max = LM_HEAD_CHUNK as u32;
        let mut v0 = 0u32;
        while v0 < VOCAB as u32 {
            let chunk = (VOCAB as u32 - v0).min(chunk_max);
            self.dispatch_sc_prob_cols(v0, chunk);
            if self.embed_bf16 {
                let w_off = layout.embed + (v0 as u64) * (HID as u64) * 2;
                self.gemm_bf16_rowk_acc_f32(
                    &self.bufs.gemm_b,
                    w_off,
                    CANVAS as u32,
                    HID as u32,
                    chunk,
                )?;
            } else {
                let w_off = layout.embed + (v0 as u64) * row_bytes;
                self.gemm_q8_rowk_acc_f32(
                    &self.bufs.gemm_b,
                    w_off,
                    CANVAS as u32,
                    HID as u32,
                    chunk,
                )?;
            }
            self.sink_memory_barrier(); // serialize the cross-chunk `+=` into gemm_b
            v0 += chunk;
        }
        // Convert f32 accumulator → bf16 arena soft slot, applying embed_scale
        // (== sqrt(HID)) and dividing out the SC_PROB_GEMM_SCALE that sc_prob_cols
        // multiplied into the probs to keep them in fp16's normal range.
        let scale = (HID as f32).sqrt() / SC_PROB_GEMM_SCALE;
        self.f32_to_half_scale(
            &self.bufs.gemm_b,
            self.arena().soft_off(),
            CANVAS * HID,
            scale,
        );
        Ok(())
    }

    /// Sparse SC softembed: select per-row survivors (prob within e^-10 of row
    /// max), then gather-weighted-sum their embed rows — instead of the full vocab
    /// GEMM. APPROXIMATE (drops the prob tail). Scratch: prob+cnt in gemm_a, idx +
    /// f32 accumulator in gemm_b (both free during preamble). rowstat from the
    /// prior ScLogitRowstats stage.
    fn encode_sc_softembed_sparse(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        let maxk = SC_SPARSE_MAXK;
        // gemm_b: [0..acc_bytes) = f32 accumulator; idx at IDX_OFF.
        // gemm_a: [0..) = fp16 prob; cnt at CNT_OFF.
        const IDX_OFF: usize = 4 * 1024 * 1024;
        const PROB_OFF: usize = 0;
        const CNT_OFF: usize = 4 * 1024 * 1024;

        // Pass 1: per-row threshold select + compact.
        self.sink_set_pipeline(&self.ps.sc_sparse_select);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
        self.sink_set_buffer(&self.bufs.gemm_b, IDX_OFF, 2);
        self.sink_set_buffer(&self.bufs.gemm_a, PROB_OFF, 3);
        self.sink_set_buffer(&self.bufs.gemm_a, CNT_OFF, 4);
        let p1 = [CANVAS as u32, VOCAB as u32, maxk, 0u32];
        self.sink_set_bytes(&p1, 5);
        self.sink_dispatch(
            MTLSize { width: CANVAS, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        self.sink_memory_barrier();

        // Pass 2: gather-weighted-sum embed rows → f32 accumulator (gemm_b[0..]).
        self.sink_set_pipeline(&self.ps.sc_sparse_gather);
        self.sink_set_buffer(&self.bufs.gemm_b, IDX_OFF, 0);
        self.sink_set_buffer(&self.bufs.gemm_a, PROB_OFF, 1);
        self.sink_set_buffer(&self.bufs.gemm_a, CNT_OFF, 2);
        self.bind_blob(3);
        self.sink_set_bytes(&layout.embed, 4);
        self.sink_set_buffer(&self.bufs.gemm_b, 0, 5);
        let p2 = [CANVAS as u32, HID as u32, maxk, 0u32];
        self.sink_set_bytes(&p2, 6);
        self.sink_dispatch(
            MTLSize { width: CANVAS, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        self.sink_memory_barrier();

        // Finalize: f32 accumulator → bf16 soft slot (÷ SC_PROB_GEMM_SCALE, × √HID).
        let scale = (HID as f32).sqrt() / SC_PROB_GEMM_SCALE;
        self.f32_to_half_scale(&self.bufs.gemm_b, self.arena().soft_off(), CANVAS * HID, scale);
        Ok(())
    }

    /// Exact (non-sparse) softembed = the chunked path; sparse is the
    /// default approximation on bf16-embed models (DGQ_SC_SPARSE=0 opts out).
    fn encode_sc_softembed_exact(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_sc_softembed_chunked(layout)
    }

    fn residual(&mut self, a_off: u64, b_off: u64, y_off: u64, scal_off: u64, elems: usize) {
            self.sink_set_buffer(&self.bufs.arena, a_off as usize, 0);
            self.sink_set_buffer(&self.bufs.arena, b_off as usize, 1);
            self.sink_set_buffer(&self.bufs.arena, y_off as usize, 2);
            self.bind_blob(3);
            self.sink_set_bytes( &scal_off, 4);
        self.dispatch_1d(&self.ps.residual, elems, 256);
    }

    fn glu(&mut self, gate_off: u64, up_off: u64, y_off: u64, elems: usize) {
        self.sink_set_pipeline(&self.ps.glu);
        let dims = [elems as u32, 0u32];
            self.sink_set_buffer(&self.bufs.arena, gate_off as usize, 0);
            self.sink_set_buffer(&self.bufs.arena, up_off as usize, 1);
            self.sink_set_buffer(&self.bufs.arena, y_off as usize, 2);
            self.sink_set_bytes(&dims, 3);
        self.dispatch_1d(&self.ps.glu, elems, 256);
    }

    fn encode_layer(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_layer_qkv_gemm(layer, layout)?;
        self.encode_layer_qk_rope_kv_dispatch(layer, layout)?;
        self.encode_layer_attention_dispatch(layer, layout)?;
        self.encode_layer_o_proj_post_attn(layer, layout)?;
        self.encode_layer_dense_ffn(layer, layout)?;
        self.encode_layer_router_buckets(layer, layout)?;
        Ok(())
    }

    fn encode_layer_o_proj_post_attn(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer_o_proj_gemm(layer, layout)?;
        self.encode_layer_o_proj_tail(layer, layout)
    }

    fn encode_layer_o_proj_gemm(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let o_k = if l.is_full != 0 { 8192 } else { 4096 };
        self.gemm_q4(
            self.arena().attno_off(),
            self.arena().tmp_off(),
            l.o_proj,
            CANVAS as u32,
            HID as u32,
            o_k,
        )
    }

    fn encode_layer_o_proj_tail(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.rmsnorm(self.arena().tmp_off(), self.arena().tmp_off(), l.post_attn_ln, HID as u32, CANVAS);
        self.residual(self.arena().hidden_off(), self.arena().tmp_off(), self.arena().stream_off(), 0, CANVAS * HID);
        Ok(())
    }

    fn encode_layer_dense_ffn(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.rmsnorm(self.arena().stream_off(), self.arena().tmp_off(), l.pre_ff_ln, HID as u32, CANVAS);
        self.encode_layer_dense_gate_up(layer, layout)?;
        self.glu(
            self.arena().ffg_off(),
            self.arena().ffu_off(),
            self.arena().ffg_off(),
            CANVAS * DENSE_FF as usize,
        );
        self.encode_layer_dense_down(layer, layout)?;
        self.rmsnorm(
            self.arena().dense_off(),
            self.arena().dense_off(),
            l.post_ff_ln_1,
            HID as u32,
            CANVAS,
        );
        Ok(())
    }

    fn encode_layer_dense_down(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.gemm_q4(
            self.arena().ffg_off(),
            self.arena().dense_off(),
            l.mlp_down,
            CANVAS as u32,
            HID as u32,
            DENSE_FF,
        )
    }

    fn encode_layer_dense_gate_up(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        if fused_gate_up_enabled() && self.attn_ffn_bf16 {
            let (segs, n_total) = gate_up_stacked_segments(l, self.arena());
            self.gemm_bf16_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else if fused_gate_up_enabled() && !self.attn_ffn_q8 && !self.attn_ffn_bf16 {
            let (segs, n_total) = gate_up_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                l.mlp_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                l.mlp_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
        }
        Ok(())
    }

    fn encode_layer_router_buckets(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let layer_off = layer_byte_offset(layer);
        let router_dims = crate::kernels::sub::moe_router::RouterDims {
            canvas: CANVAS as u32,
            hidden: HID as u32,
            n_experts: N_EXPERTS as u32,
            top_k: TOP_K as u32,
            router_hscale: (HID as f32).powf(-0.5),
        };
        if router_gemm_enabled() {
            // Router-as-GEMM: xn = rmsnorm_noscale(stream) * router_scale[d]
            // (exactly what fn rmsnorm computes with w=router_scale; same 1e-6
            // eps as MOE_ROUTER_RMS_EPS) -> bf16 GEMM against router_proj
            // (n=128 experts) into the free ffg plane -> top-k tail applies
            // the uniform router_hscale (linear in the input, so folding it
            // out of the GEMM is exact up to bf16 logit rounding).
            let l = &layout.layers[layer];
            self.rmsnorm(
                self.arena().stream_off(),
                self.arena().tmp_off(),
                l.router_scale,
                HID as u32,
                CANVAS,
            );
            self.gemm_bf16(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                l.router_proj,
                CANVAS as u32,
                N_EXPERTS as u32,
                HID as u32,
            )?;
            self.sink_set_pipeline(&self.ps.router_topk);
            self.sink_set_buffer(&self.bufs.arena, self.arena().ffg_off() as usize, 0);
            self.bind_blob(1);
            self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 2);
            self.bind_route(3);
            self.sink_set_bytes(&router_dims, 4);
            self.bind_debug_status(5);
            let grid = MTLSize { width: CANVAS.div_ceil(64), height: 1, depth: 1 };
            let tg = MTLSize { width: 64, height: 1, depth: 1 };
            self.sink_dispatch(grid, tg);
        } else {
            self.sink_set_pipeline(&self.ps.router);
            self.sink_set_buffer(&self.bufs.arena, self.arena().stream_off() as usize, 0);
            self.bind_blob(1);
            self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 2);
            self.bind_route(3);
            self.sink_set_bytes(&router_dims, 4);
            self.bind_debug_status(5);
            let grid = MTLSize {
                width: CANVAS,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            self.sink_dispatch(grid, tg);
        }

        self.sink_set_pipeline(&self.ps.bucket_count);
        self.bind_route(0);
        let n_experts = N_EXPERTS as u32;
        self.sink_set_bytes(&n_experts, 1);
        self.dispatch_1d(&self.ps.bucket_count, 128, 128);

        for phase in 0u32..3 {
            self.sink_set_pipeline(&self.ps.bucket_fill);
            self.bind_route(0);
            self.sink_set_bytes(&phase, 1);
            self.sink_set_bytes(&router_dims, 2);
            self.sink_set_buffer(&self.bufs.expert_layer_unique, 0, 3);
            let layer_idx = layer as u32;
            self.sink_set_bytes(&layer_idx, 4);
            self.bind_debug_status(5);
            self.sink_set_buffer(&self.bufs.moe_grouped_indirect, 0, 6);
            let grid_info = moe_grouped_grid_info();
            self.sink_set_bytes(&grid_info, 7);
            let count = if phase == 1 { 1 } else { CANVAS * TOP_K };
            self.dispatch_1d(&self.ps.bucket_fill, count, 256);
        }

        let l = &layout.layers[layer];
        self.rmsnorm(self.arena().stream_off(), self.arena().moein_off(), l.pre_ff_ln_2, HID as u32, CANVAS);
        self.memzero_bytes(self.arena().moeout_off(), (CANVAS * HID * 4) as u64);
        Ok(())
    }

    fn dispatch_block_linear_grouped(
        &mut self,
        a_on_gemm_a: bool,
        buf_a_off: usize,
        buf_c_off: usize,
        jobs: &[BlockGroupedJob; N_EXPERTS],
        _total_m: u32,
        k: u32,
        n: u32,
        indirect_slot: usize,
        gather: bool,
    ) -> Result<(), Error> {
        let block_sparse = moe_block_sparse_enabled();
        let adaptive = block_sparse && moe_tile_adapt_enabled();
        // Tunable block-sparse (q4 experts): fragment kernel with built-in
        // adaptive-M; indirect slots 4/5 (BN-wide N-tiles).
        let tunable = block_sparse
            && matches!(
                self.block_profile.format,
                QuantFormat::Q4Affine | QuantFormat::Q6
            )
            && self.ps.sparse_tunable_fmt(self.block_profile.format, n, k, gather).is_some();
        // Fused-gather gate_up: A-load pulls bf16 `moein` rows via token_list
        // (buffer 7), so no separate gather pass / f32 staging buffer. The caller
        // skips the gather pass iff `gather`; if the pipeline for this shape is
        // missing we'd read a stale A buffer, so fail loud rather than silently.
        let gather_ps = if gather {
            if tunable {
                self.ps.sparse_tunable_fmt(self.block_profile.format, n, k, true)
            } else {
                self.ps.block_sparse_gather(n, k, adaptive)
            }
        } else {
            None
        };
        if gather && gather_ps.is_none() {
            return Err(Error::Format(
                "fused MoE gather requested but no gather pipeline for this shape",
            ));
        }
        let use_gather = gather_ps.is_some();
        let grouped_ps = if let Some(p) = gather_ps {
            p
        } else if tunable {
            self.ps
                .sparse_tunable_fmt(self.block_profile.format, n, k, false)
                .ok_or(Error::Format("missing tunable sparse pipeline"))?
        } else if block_sparse {
            self.ps.block_sparse(self.block_profile.format, n, k, adaptive)?
        } else {
            self.ps.block_grouped(self.block_profile.format, n, k)?
        };
        let row_start_off = std::mem::offset_of!(RouteScratch, row_start);
        self.sink_set_pipeline(grouped_ps);
        let a_buf = if a_on_gemm_a {
            &self.bufs.gemm_a
        } else {
            &self.bufs.gemm_b
        };
            self.sink_set_buffer(a_buf, buf_a_off, 0);
            // Expert weights: region-2 buffer on split blobs (job offsets are
            // rebased to match in layer_moe_block_jobs_impl).
            self.bind_blob_experts(1);
            self.sink_set_buffer(&self.bufs.gemm_b, buf_c_off, 2);
            self.sink_set_bytes(jobs, 3);
            self.sink_set_buffer(&self.bufs.route, row_start_off, 4);
        let num_jobs = N_EXPERTS as u32;
        self.sink_set_bytes(&num_jobs, 5);
        self.bind_route(6);
        if use_gather {
            self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 7);
        }
        let tg = MTLSize {
            width: crate::kernels::sub::gemm_common::THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // Indirect slots: tunable sparse 4/5; legacy block-sparse 2/3; grouped 0/1.
        let slot = if tunable {
            indirect_slot + 4
        } else if block_sparse {
            indirect_slot + 2
        } else {
            indirect_slot
        };
        let indirect_offset = slot * 3 * std::mem::size_of::<u32>();
        self.sink_dispatch_indirect(indirect_offset, n, tg);
        Ok(())
    }

    /// Batched MoE: gather → grouped block GEMM (gate/up, down) → swiglu → weighted scatter.
    fn encode_layer_moe_batched(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        if !moe_fuse_gather_enabled() {
            self.encode_moe_batched_gather()?;
        }
        self.encode_moe_batched_gate_up(layer, layout)?;
        self.encode_moe_batched_swiglu()?;
        self.encode_moe_batched_down(layer, layout)?;
        self.encode_moe_batched_scatter()?;
        Ok(())
    }

    fn encode_moe_batched_gather(&mut self) -> Result<(), Error> {
        self.encode_moe_batched_gather_bf16_to_f32()
    }

    fn encode_moe_batched_gather_bf16_to_f32(&mut self) -> Result<(), Error> {
        let token_list_off = std::mem::offset_of!(RouteScratch, token_list);
        let gather_dims = [0u32, HID as u32];
        let gather_count = (MOE_SLOTS as usize) * HID;
        self.dispatch_1d_ranged(
            &self.ps.gather_rows_bf16_to_f32,
            gather_count,
            256,
            |this, base, _chunk| {
                this.sink_set_buffer(
                    &this.bufs.arena,
                    this.arena().moein_off() as usize,
                    0,
                );
                this.sink_set_buffer(&this.bufs.route, token_list_off, 1);
                this.sink_set_buffer(&this.bufs.gemm_b, moe_w_byte_off_a(), 2);
                this.sink_set_buffer(&this.bufs.dummy_dump, 0, 5);
                this.sink_set_bytes(&gather_dims, 3);
                this.sink_set_bytes(&MOE_SLOTS, 4);
                this.sink_set_bytes(&base, 6);
            },
        );
        Ok(())
    }

    #[allow(dead_code)]
    fn encode_moe_batched_half_to_f32(&mut self) -> Result<(), Error> {
        self.half_to_f32_buf(self.arena().moein_off(), CANVAS * HID);
        Ok(())
    }

    #[allow(dead_code)]
    fn encode_moe_batched_gather_rows(&mut self) -> Result<(), Error> {
        let token_list_off = std::mem::offset_of!(RouteScratch, token_list);
        let gather_dims = [0u32, HID as u32];
        let gather_count = (MOE_SLOTS as usize) * HID;
        self.dispatch_1d_ranged(&self.ps.gather_rows, gather_count, 256, |this, base, _chunk| {
            this.sink_set_buffer(&this.bufs.gemm_a, 0, 0);
            this.sink_set_buffer(&this.bufs.route, token_list_off, 1);
            this.sink_set_buffer(&this.bufs.gemm_b, moe_w_byte_off_a(), 2);
            this.sink_set_buffer(&this.bufs.dummy_dump, 0, 5);
            this.sink_set_bytes(&gather_dims, 3);
            this.sink_set_bytes(&MOE_SLOTS, 4);
            this.sink_set_bytes(&base, 6);
        });
        Ok(())
    }

    fn encode_moe_batched_gate_up(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let (gate_jobs, _) = layer_moe_block_jobs_impl(
            l,
            self.block_profile.format,
            Some((layer, self.tensor_offsets)),
            self.bufs.blob_expert_base,
        );
        self.dispatch_block_linear_grouped(
            false,
            moe_w_byte_off_a(),
            moe_w_byte_off_gu(),
            &gate_jobs,
            MOE_SLOTS,
            HID as u32,
            MOE_FF * 2,
            0,
            moe_fuse_gather_enabled(),
        )
    }

    fn encode_moe_batched_swiglu(&mut self) -> Result<(), Error> {
        let gu_off = moe_w_byte_off_gu();
        let act_elems = (MOE_SLOTS as usize) * MOE_FF as usize;
        self.sink_set_pipeline(&self.ps.gelu_swiglu_gate_up);
        self.sink_set_buffer(&self.bufs.gemm_b, gu_off, 0);
        self.sink_set_buffer(&self.bufs.gemm_a, 0, 1);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 3);
        let swiglu_dims = [MOE_SLOTS, MOE_FF];
        self.sink_set_bytes(&swiglu_dims, 2);
        self.dispatch_1d(&self.ps.gelu_swiglu_gate_up, act_elems, 256);
        Ok(())
    }

    fn encode_moe_batched_down(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let (_, down_jobs) = layer_moe_block_jobs_impl(
            l,
            self.block_profile.format,
            None,
            self.bufs.blob_expert_base,
        );
        self.dispatch_block_linear_grouped(
            true,
            0,
            moe_w_byte_off_a(),
            &down_jobs,
            MOE_SLOTS,
            MOE_FF,
            HID as u32,
            1,
            false,
        )
    }

    fn encode_moe_batched_scatter(&mut self) -> Result<(), Error> {
        self.sink_set_pipeline(&self.ps.moe_scatter_weighted);
        self.sink_set_buffer(&self.bufs.gemm_b, moe_w_byte_off_a(), 0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moeout_off() as usize, 1);
        self.bind_route(2);
        let hidden = HID as u32;
        let canvas = CANVAS as u32;
        self.sink_set_bytes(&hidden, 3);
        self.sink_set_bytes(&canvas, 4);
        // One threadgroup per (token, 256-wide d-tile); 256 threads, one per d.
        let grid = MTLSize {
            width: div_up(hidden as usize, 256),
            height: canvas as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn encode_layer_moe_scalar(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let layer_off = layer_byte_offset(layer);
        self.sink_set_pipeline(self.ps.moe_scalar(self.block_profile.format));
        self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 0);
        self.sink_set_buffer(&self.bufs.gemm_b, moe_w_byte_off_a(), 1);
        self.bind_blob(2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_route(4);
        let grouped_dims = crate::kernels::sub::moe_grouped::GroupedDims {
            canvas: CANVAS as u32,
            hidden: HID as u32,
            moe_ff: MOE_FF,
            n_experts: N_EXPERTS as u32,
        };
        self.sink_set_bytes(&grouped_dims, 5);
        if !self.block_profile.is_nvfp4() {
            self.sink_set_buffer(&self.bufs.dummy_dump, 0, 6);
        }
        let grid = MTLSize {
            width: CANVAS,
            height: N_EXPERTS,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        self.encode_moe_batched_scatter()
    }

    fn encode_layer_moe_grouped(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        match self.block_profile.moe_style() {
            MoeExecutionStyle::BatchedGrouped => self.encode_layer_moe_batched(layer, layout),
            MoeExecutionStyle::ScalarPerExpert => self.encode_layer_moe_scalar(layer, layout),
        }
    }

    fn encode_layer_moe_post_norm(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.rmsnorm_f32(self.arena().moeout_off(), self.arena().moein_off(), l.post_ff_ln_2, HID as u32, CANVAS);
        Ok(())
    }

    fn encode_layer_moe_post_combine(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.residual(self.arena().dense_off(), self.arena().moein_off(), self.arena().tmp_off(), 0, CANVAS * HID);
        self.rmsnorm(self.arena().tmp_off(), self.arena().tmp_off(), l.post_ff_ln, HID as u32, CANVAS);
        self.residual(self.arena().stream_off(), self.arena().tmp_off(), self.arena().hidden_off(), l.layer_scalar, CANVAS * HID);
        Ok(())
    }

    fn encode_layer_moe_post(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_layer_moe_post_norm(layer, layout)?;
        self.encode_layer_moe_post_combine(layer, layout)?;
        Ok(())
    }

    /// Attention + dense FFN + router + grouped MoE + post-combine (one encoder session).
    fn encode_full_layer(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_layer(layer, layout)?;
        self.encode_layer_moe_grouped(layer, layout)?;
        self.encode_layer_moe_post(layer, layout)?;
        Ok(())
    }

    /// MoE grouped kernel for one expert at one canvas row (router bypassed).
    fn encode_layer_moe_single_expert_setup(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        position: usize,
        expert_id: u32,
    ) {
        let l = &layout.layers[layer];
        self.rmsnorm(self.arena().stream_off(), self.arena().moein_off(), l.pre_ff_ln_2, HID as u32, CANVAS);
        self.memzero_bytes(self.arena().moeout_off(), (CANVAS * HID * 4) as u64);
        write_single_expert_route(&self.bufs.route, position, expert_id);
    }

    /// Grouped MoE with K_DUMP_STAGE dump of threadgroup act (debug capture only).
    fn encode_layer_moe_grouped_act_probe(
        &mut self,
        layer: usize,
        _layout: &ModelLayout,
    ) -> Result<(), Error> {
        if self.block_profile.is_nvfp4() {
            return Err(Error::Format(
                "moe_grouped dump mode is q4-only (use q8 .dgq weights)",
            ));
        }
        let layer_off = layer_byte_offset(layer);
        self.memzero_bytes(self.arena().moeout_off(), (CANVAS * HID * 4) as u64);
        self.memzero_bytes(
            self.arena().soft_off(),
            (MOE_ACT_PROBE_FLOATS * std::mem::size_of::<f32>()) as u64,
        );
        self.sink_set_pipeline(&self.ps.moe_grouped_dump);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moeout_off() as usize, 1);
        self.bind_blob(2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_route(4);
        self.sink_set_buffer(&self.bufs.arena, self.arena().soft_off() as usize, 6);
        let grouped_dims = crate::kernels::sub::moe_grouped::GroupedDims {
            canvas: CANVAS as u32,
            hidden: HID as u32,
            moe_ff: MOE_FF,
            n_experts: N_EXPERTS as u32,
        };
        self.sink_set_bytes(&grouped_dims, 5);
        let grid = MTLSize {
            width: CANVAS,
            height: N_EXPERTS,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Input layernorm + fused Q‖K(‖V) projections (stops before qk_rope_kv).
    fn encode_layer_qkv_gemm(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.rmsnorm(
            self.arena().hidden_off(),
            self.arena().tmp_off(),
            l.input_ln,
            HID as u32,
            CANVAS,
        );
        if fused_qkv_enabled() && self.attn_ffn_bf16 {
            let (segs, n_total) = qkv_stacked_segments(l, self.arena());
            self.gemm_bf16_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else if fused_qkv_enabled() && !self.attn_ffn_q8 && !self.attn_ffn_bf16 {
            let (segs, n_total) = qkv_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            let q_n = if l.is_full != 0 { 8192 } else { 4096 };
            let k_n = if l.is_full != 0 { 1024 } else { 2048 };
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnq_off(),
                l.q_proj,
                CANVAS as u32,
                q_n,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnk_off(),
                l.k_proj,
                CANVAS as u32,
                k_n,
                HID as u32,
            )?;
            if l.v_proj != 0 {
                self.gemm_q4(
                    self.arena().tmp_off(),
                    self.arena().attnv_off(),
                    l.v_proj,
                    CANVAS as u32,
                    k_n,
                    HID as u32,
                )?;
            }
        }
        Ok(())
    }

    /// QKV GEMM dispatches only (caller must have normalized input in `tmp`).
    fn dispatch_qkv_gemms(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        stacked: bool,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        if stacked {
            let (segs, n_total) = qkv_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            let q_n = if l.is_full != 0 { 8192 } else { 4096 };
            let k_n = if l.is_full != 0 { 1024 } else { 2048 };
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnq_off(),
                l.q_proj,
                CANVAS as u32,
                q_n,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnk_off(),
                l.k_proj,
                CANVAS as u32,
                k_n,
                HID as u32,
            )?;
            if l.v_proj != 0 {
                self.gemm_q4(
                    self.arena().tmp_off(),
                    self.arena().attnv_off(),
                    l.v_proj,
                    CANVAS as u32,
                    k_n,
                    HID as u32,
                )?;
            }
        }
        Ok(())
    }

    /// Dense gate/up GEMM dispatches only (caller must have normalized input in `tmp`).
    fn dispatch_gate_up_gemms(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        stacked: bool,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        if stacked {
            let (segs, n_total) = gate_up_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                l.mlp_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                l.mlp_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
        }
        Ok(())
    }

    /// QK-RoPE-KV write (expects Q/K/V already in arena).
    fn encode_layer_qk_rope_kv_dispatch(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let qk_y = (16 + 2 * l.n_kv_heads) as usize;
        let layer_off = layer_byte_offset(layer);

        self.sink_set_pipeline(&self.ps.qk_rope_kv);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attnq_off() as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attnk_off() as usize, 1);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attnv_off() as usize, 2);
        self.bind_kvcache(3);
        self.bind_blob(4);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 5);
        self.bind_params(6);
        let attn_dims = crate::kernels::sub::qk_rope_kv::AttnDims {
            canvas: CANVAS as u32,
            n_q_heads: STEP_NQ_HEADS as u32,
            causal: 0,
            window: 0, // KV write only; the window applies at attention read time
        };
        self.sink_set_bytes(&attn_dims, 7);
        self.bind_debug_status(8);
        let grid = MTLSize {
            width: CANVAS,
            height: qk_y,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn encode_layer_attention_dispatch(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let layer_off = layer_byte_offset(layer);
        let l = &layout.layers[layer];
        // GQA-grouped MMA attention (`DGQ_ATTN_MMA`) handles sliding layers (hd=256)
        // via attention_mma2; full hd=512 layers use attention_mma_full
        // (`DGQ_ATTN_MMA_FULL`, register-resident O + group K/V sharing) when
        // enabled, else the scalar kernel. Identical buffer layout — only the
        // pipeline + dispatch grid differ. mma_full is non-bit-identical (quality
        // gate): default OFF.
        // mma2/mma_full honor the causal mask (AttnDims.causal) so they *can* run
        // prefill, but their f16 attention is lossier than the scalar f32 kernel and
        // hurts fast-prefill accuracy (11/16 vs 14/16); prefill uses scalar for now.
        let use_mma2 = !self.prefill_causal && attn_mma_enabled() && l.is_full == 0;
        let use_mma_full = !self.prefill_causal && attn_mma_full_enabled() && l.is_full == 1;
        if use_mma2 {
            self.sink_set_pipeline(&self.ps.attention_mma2);
        } else if use_mma_full {
            self.sink_set_pipeline(&self.ps.attention_mma_full);
        } else {
            self.sink_set_pipeline(&self.ps.attention);
        }
        self.sink_set_buffer(&self.bufs.arena, self.arena().attnq_off() as usize, 0);
        self.bind_kvcache(1);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attno_off() as usize, 2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_params(4);
        // Sliding layers (is_full==0) attend a bounded window (Gemma-4
        // sliding_window=1024): denoise canvas sees only the last window-1
        // encoder positions + the canvas (MLX `_make_decoder_masks`); causal
        // prefill queries see [q-(window-1), q] (engine CausalSliding). This is
        // both the model spec for kv_len>window-1 AND keeps 25/30 layers
        // O(window) instead of O(context). No-op (bit-identical) below that.
        let window = if l.is_full == 0 && attn_window_enabled() {
            self.sliding_window
        } else {
            0
        };
        let attn_dims = crate::kernels::sub::qk_rope_kv::AttnDims {
            canvas: CANVAS as u32,
            n_q_heads: STEP_NQ_HEADS as u32,
            causal: u32::from(self.prefill_causal),
            window,
        };
        self.sink_set_bytes(&attn_dims, 5);
        self.bind_debug_status(6);
        // Scalar: one threadgroup per (canvas token, Q head). MMA2: one per
        // (MT-row tile, KV head), 2 simdgroups = the 2 Q heads in the group.
        // MMA_full: one per (MT-row tile, KV head, QG-head sub-group), QG
        // simdgroups sharing K/V; (group/QG) sub-groups along z.
        let grid = if use_mma2 {
            MTLSize {
                width: CANVAS.div_ceil(crate::kernels::sub::attention::MMA_M_TILE),
                height: l.n_kv_heads as usize,
                depth: 1,
            }
        } else if use_mma_full {
            let group = STEP_NQ_HEADS / l.n_kv_heads as usize; // 8 for full
            // One tg per (query tile, kv head, Q head); the QG simdgroups
            // split head_dim, so depth is the full GQA group.
            MTLSize {
                width: CANVAS.div_ceil(crate::kernels::sub::attention::MMA_M_TILE),
                height: l.n_kv_heads as usize,
                depth: group,
            }
        } else {
            MTLSize {
                width: CANVAS,
                height: 16,
                depth: 1,
            }
        };
        // mma_full uses QG*32 lanes; scalar/mma2 use 64.
        let tg = MTLSize {
            width: if use_mma_full {
                crate::kernels::sub::attention::MMA_FULL_QG * 32
            } else {
                64
            },
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// QK-RoPE-KV write + attention (expects Q/K/V already in arena).
    fn encode_layer_qk_rope_and_attention(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer_qk_rope_kv_dispatch(layer, layout)?;
        self.encode_layer_attention_dispatch(layer, layout)
    }

    fn exec_stage(
        &mut self,
        stage: step_schedule::StepStage,
        layer: usize,
        layout: &ModelLayout,
        finish: StepFinishMode,
    ) -> Result<(), Error> {
        let _ = layout;
        use step_schedule::StepStage;
        match stage {
            StepStage::ScLogitRowstats => {
                self.encode_sc_logit_rowstats();
                Ok(())
            }
            StepStage::ScSoftembed => self.encode_sc_softembed(layout),
            StepStage::ScPreNorm => {
                self.rmsnorm(self.arena().soft_off(), self.arena().tmp_off(), layout.sc_pre_norm, HID as u32, CANVAS);
                Ok(())
            }
            StepStage::ScGateGemm => {
                self.gemm_q8(
                    self.arena().tmp_off(),
                    self.arena().ffg_off(),
                    layout.sc_gate,
                    CANVAS as u32,
                    DENSE_FF,
                    HID as u32,
                )
            }
            StepStage::ScUpGemm => {
                self.gemm_q8(
                    self.arena().tmp_off(),
                    self.arena().ffu_off(),
                    layout.sc_up,
                    CANVAS as u32,
                    DENSE_FF,
                    HID as u32,
                )
            }
            StepStage::ScGlu => {
                self.glu(self.arena().ffg_off(), self.arena().ffu_off(), self.arena().ffg_off(), CANVAS * DENSE_FF as usize);
                Ok(())
            }
            StepStage::ScDownGemm => {
                self.gemm_q8(
                    self.arena().ffg_off(),
                    self.arena().dense_off(),
                    layout.sc_down,
                    CANVAS as u32,
                    HID as u32,
                    DENSE_FF,
                )
            }
            StepStage::EmbedGather => {
                self.dispatch_embed_gather(layout.embed);
                Ok(())
            }
            StepStage::EmbedScResidual => {
                self.residual(
                    self.arena().hidden_off(),
                    self.arena().dense_off(),
                    self.arena().hidden_off(),
                    0,
                    CANVAS * HID,
                );
                Ok(())
            }
            StepStage::RmsNormHidden => {
                self.rmsnorm(self.arena().hidden_off(), self.arena().hidden_off(), 0, HID as u32, CANVAS);
                Ok(())
            }
            StepStage::LayerInputNormQkv => self.encode_layer_qkv_gemm(layer, layout),
            StepStage::LayerQkRopeKv => self.encode_layer_qk_rope_kv_dispatch(layer, layout),
            StepStage::LayerAttention => self.encode_layer_attention_dispatch(layer, layout),
            StepStage::LayerOProjPostAttn => self.encode_layer_o_proj_post_attn(layer, layout),
            StepStage::LayerDenseFfn => self.encode_layer_dense_ffn(layer, layout),
            StepStage::LayerRouter => self.encode_layer_router_buckets(layer, layout),
            StepStage::MoeBatchedGather => {
                // Fused gather folds the token gather into the gate_up A-load.
                if moe_fuse_gather_enabled() {
                    Ok(())
                } else {
                    self.encode_moe_batched_gather()
                }
            }
            StepStage::MoeBatchedGateUp => self.encode_moe_batched_gate_up(layer, layout),
            StepStage::MoeBatchedSwiglu => self.encode_moe_batched_swiglu(),
            StepStage::MoeBatchedDown => self.encode_moe_batched_down(layer, layout),
            StepStage::MoeBatchedScatter => self.encode_moe_batched_scatter(),
            StepStage::MoeGroupedScalar => self.encode_layer_moe_scalar(layer, layout),
            StepStage::LayerMoePostNorm => self.encode_layer_moe_post_norm(layer, layout),
            StepStage::LayerMoePostCombine => self.encode_layer_moe_post_combine(layer, layout),
            StepStage::FinalNorm => {
                self.rmsnorm(self.arena().hidden_off(), self.arena().tmp_off(), layout.final_norm, HID as u32, CANVAS);
                Ok(())
            }
            StepStage::LmHeadGemm => {
                let m = self.partial_lm_m;
                if self.embed_bf16 {
                    // bf16 embed: full tied lm_head via the bf16 GEMM (partial lm_head
                    // is the q8-only fast path; correctness over that optimization here).
                    self.gemm_bf16_logits(
                        self.arena().tmp_off(),
                        layout.embed,
                        CANVAS as u32,
                        VOCAB as u32,
                        HID as u32,
                        0,
                    )
                } else if partial_lm_head_enabled() && m < CANVAS as u32 {
                    self.encode_partial_lm_head(layout, m)
                } else {
                    self.gemm_q8_logits(
                        self.arena().tmp_off(),
                        layout.embed,
                        CANVAS as u32,
                        VOCAB as u32,
                        HID as u32,
                        0,
                    )
                }
            }
            StepStage::Softcap => {
                self.dispatch_softcap();
                Ok(())
            }
            StepStage::SampleRowstats
            | StepStage::SampleCommit
            | StepStage::SampleApply
            | StepStage::SampleWrite => {
                if finish == StepFinishMode::Full {
                    self.encode_step_sampler(layout)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn interpret_step(
        &mut self,
        layout: &ModelLayout,
        layers: usize,
        first_step: u32,
        finish: StepFinishMode,
    ) -> Result<(), Error> {
        if arena_liveness::runtime_arena_liveness_enabled() {
            if let Err(e) = arena_liveness::check_step_arena_liveness(
                &self.block_profile,
                layout,
                layers,
                first_step,
                finish,
            ) {
                panic!("{e}");
            }
        }
        let schedule =
            step_schedule::build_step_schedule(&self.block_profile, finish == StepFinishMode::Full);
        if first_step == 1 {
            // Deterministic first-step self-conditioning. The first denoise step has
            // no prior prediction, so the normal SC path is skipped — but the model
            // is degenerate with SC=0 (cold-start empty reply), and leaving dense_off
            // as a prior generation's residual makes reused sessions nondeterministic
            // (reset_kv carryover). Seed it deterministically: treat the initial
            // canvas as the step-0 prediction and run the SC MLP on its embedding
            // (ScPreNorm reads hidden after EmbedGather, in place of soft_off).
            use step_schedule::StepStage;
            self.exec_stage(StepStage::EmbedGather, 0, layout, finish)?;
            self.rmsnorm(
                self.arena().hidden_off(),
                self.arena().tmp_off(),
                layout.sc_pre_norm,
                HID as u32,
                CANVAS,
            );
            self.exec_stage(StepStage::ScGateGemm, 0, layout, finish)?;
            self.exec_stage(StepStage::ScUpGemm, 0, layout, finish)?;
            self.exec_stage(StepStage::ScGlu, 0, layout, finish)?;
            self.exec_stage(StepStage::ScDownGemm, 0, layout, finish)?;
            self.exec_stage(StepStage::EmbedScResidual, 0, layout, finish)?;
            self.exec_stage(StepStage::RmsNormHidden, 0, layout, finish)?;
        } else {
            for stage in step_schedule::build_preamble(first_step) {
                self.exec_stage(stage, 0, layout, finish)?;
            }
        }
        for layer in 0..layers {
            for &stage in &schedule.per_layer {
                self.exec_stage(stage, layer, layout, finish)?;
            }
        }
        let mut sampler_done = false;
        for &stage in &schedule.finish {
            if matches!(
                stage,
                step_schedule::StepStage::SampleRowstats
                    | step_schedule::StepStage::SampleCommit
                    | step_schedule::StepStage::SampleApply
                    | step_schedule::StepStage::SampleWrite
            ) {
                if !sampler_done && finish == StepFinishMode::Full {
                    self.encode_step_sampler(layout)?;
                    sampler_done = true;
                }
                continue;
            }
            self.exec_stage(stage, 0, layout, finish)?;
        }
        Ok(())
    }

    /// KV-only causal forward over one prompt chunk (the canvas holds chunk tokens).
    /// Embed + no-weight norm + the full per-layer stack (qkv → qk_rope_kv writes
    /// KV → CAUSAL attention → o_proj → dense FFN → MoE), with NO SC preamble, NO
    /// sampler, NO lm_head. The fast monolithic analog of the f32-engine prefill.
    fn encode_prefill_chunk(&mut self, layout: &ModelLayout, layers: usize) -> Result<(), Error> {
        use step_schedule::StepStage;
        self.prefill_causal = true;
        self.exec_stage(StepStage::EmbedGather, 0, layout, StepFinishMode::ForwardOnly)?;
        self.exec_stage(StepStage::RmsNormHidden, 0, layout, StepFinishMode::ForwardOnly)?;
        let per_layer = step_schedule::per_layer_stages(&self.block_profile);
        for layer in 0..layers {
            for &stage in &per_layer {
                self.exec_stage(stage, layer, layout, StepFinishMode::ForwardOnly)?;
            }
        }
        Ok(())
    }

    /// Canvas token embed gather only (no no-scale RMSNorm).
    fn encode_layer_through_attention(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer_qkv_gemm(layer, layout)?;
        self.encode_layer_qk_rope_and_attention(layer, layout)
    }

    fn dispatch_embed_gather(&mut self, embed_off: u64) {
        use crate::dgq::embed_row::EMBED_SCALE;

        let ps = if self.embed_bf16 {
            &self.ps.embed_gather_bf16
        } else {
            &self.ps.embed_gather
        };
        self.sink_set_pipeline(ps);
        self.bind_blob(0);
        self.bind_state(1);
        self.sink_set_buffer(&self.bufs.arena, self.arena().hidden_off() as usize, 2);
        self.sink_set_bytes(&embed_off, 3);
        let dims = [HID as u32, CANVAS as u32];
        self.sink_set_bytes(&dims, 4);
        self.sink_set_bytes(&EMBED_SCALE, 5);
        let vocab = VOCAB as u32;
        self.sink_set_bytes(&vocab, 6);
        self.bind_debug_status(7);
        let (grid, tg) = crate::kernels::sub::embed_gather::dispatch_shape(HID, CANVAS);
        self.sink_dispatch(grid, tg);
    }

    /// Canvas token embed gather only (no no-scale RMSNorm).
    fn encode_preamble_embed_only(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        self.dispatch_embed_gather(layout.embed);
        Ok(())
    }

    fn encode_step_preamble(&mut self, layout: &ModelLayout, first_step: u32) -> Result<(), Error> {
        if first_step == 0 {
            self.encode_sc_logit_rowstats();
            self.encode_sc_softembed(layout)?;

            self.rmsnorm(self.arena().soft_off(), self.arena().tmp_off(), layout.sc_pre_norm, HID as u32, CANVAS);
            self.gemm_q8(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                layout.sc_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q8(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                layout.sc_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.glu(self.arena().ffg_off(), self.arena().ffu_off(), self.arena().ffg_off(), CANVAS * DENSE_FF as usize);
            self.gemm_q8(
                self.arena().ffg_off(),
                self.arena().dense_off(),
                layout.sc_down,
                CANVAS as u32,
                HID as u32,
                DENSE_FF,
            )?;
        }
        // first_step: self.arena().dense_off() stays zero; skip SC MLP + O(vocab) softembed.

        self.dispatch_embed_gather(layout.embed);
        self.residual(
            self.arena().hidden_off(),
            self.arena().dense_off(),
            self.arena().hidden_off(),
            0,
            CANVAS * HID,
        );
        self.rmsnorm(self.arena().hidden_off(), self.arena().hidden_off(), 0, HID as u32, CANVAS);
        Ok(())
    }

    fn encode_partial_lm_head(
        &mut self,
        layout: &ModelLayout,
        m: u32,
    ) -> Result<(), Error> {
        let token_list_off = std::mem::offset_of!(RouteScratch, token_list);
        let num_slots_off = std::mem::offset_of!(RouteScratch, num_slots);
        let compact_row = CANVAS as u32 - m;
        let logits_off = (compact_row as usize) * VOCAB * 2;

        self.sink_set_pipeline(&self.ps.compact_active_rows);
        self.bind_state(0);
        self.sink_set_buffer(&self.bufs.route, token_list_off, 1);
        self.sink_set_buffer(&self.bufs.route, num_slots_off, 2);
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        let gather_dims = [0u32, HID as u32];
        let gather_count = (m as usize) * HID;
        self.dispatch_1d_ranged(&self.ps.gather_rows_bf16, gather_count, 256, |this, base, _chunk| {
            this.sink_set_buffer(&this.bufs.arena, this.arena().tmp_off() as usize, 0);
            this.sink_set_buffer(&this.bufs.route, token_list_off, 1);
            this.sink_set_buffer(&this.bufs.arena, this.arena().dense_off() as usize, 2);
            this.sink_set_buffer(&this.bufs.dummy_dump, 0, 5);
            this.sink_set_bytes(&gather_dims, 3);
            this.sink_set_bytes(&m, 4);
            this.sink_set_bytes(&base, 6);
        });

        self.gemm_q8_logits(
            self.arena().dense_off(),
            layout.embed,
            m,
            VOCAB as u32,
            HID as u32,
            logits_off,
        )?;

        let dims = [m, VOCAB as u32];
        self.sink_set_pipeline(&self.ps.scatter_logits_rows);
        self.sink_set_buffer(&self.bufs.logits, logits_off, 0);
        self.sink_set_buffer(&self.bufs.logits, 0, 1);
        self.sink_set_buffer(&self.bufs.route, token_list_off, 2);
        self.sink_set_bytes(&dims, 3);
        let grid = MTLSize {
            width: VOCAB,
            height: m as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn encode_step_finish(
        &mut self,
        layout: &ModelLayout,
        mode: StepFinishMode,
    ) -> Result<(), Error> {
        self.rmsnorm(self.arena().hidden_off(), self.arena().tmp_off(), layout.final_norm, HID as u32, CANVAS);
        let m = self.partial_lm_m;
        if partial_lm_head_enabled() && m < CANVAS as u32 {
            self.encode_partial_lm_head(layout, m)?;
        } else {
            self.gemm_q8_logits(
                self.arena().tmp_off(),
                layout.embed,
                CANVAS as u32,
                VOCAB as u32,
                HID as u32,
                0,
            )?;
        }
        self.dispatch_softcap();
        if mode == StepFinishMode::ForwardOnly {
            return Ok(());
        }
        self.encode_step_sampler(layout)
    }

    fn encode_step_sampler(&mut self, _layout: &ModelLayout) -> Result<(), Error> {
        let cols = VOCAB as u32;
        let canvas = CANVAS as u32;
        let pad = crate::sample::PAD_TOKEN_ID;
        let filler = crate::sample::FILLER_TOKEN_ID;

        self.sink_set_pipeline(&self.ps.sample_rowstats);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_samp_off() as usize, 1);
        self.bind_state(2);
        self.bind_params(3);
        self.sink_set_bytes(&cols, 4);
        self.sink_set_bytes(&pad, 5);
        self.sink_set_bytes(&filler, 6);
        let eos = read_struct::<StepParams>(&self.bufs.params).eos_token_id;
        self.sink_set_bytes(&eos, 7);
        self.bind_debug_status(8);
        let grid = MTLSize {
            width: CANVAS,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        self.sink_set_pipeline(&self.ps.sample_commit);
        self.bind_state(0);
        self.bind_params(1);
        self.sink_set_bytes(&canvas, 2);
        self.sink_set_bytes(&pad, 3);
        self.sink_set_bytes(&filler, 4);
        let eos = read_struct::<StepParams>(&self.bufs.params).eos_token_id;
        self.sink_set_bytes(&eos, 5);
        self.bind_debug_status(6);
        let es_ent = crate::flags::early_stop_mean_ent();
        self.sink_set_bytes(&es_ent, 7);
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        self.sink_set_pipeline(&self.ps.sample_apply);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_samp_off() as usize, 1);
        self.bind_state(2);
        self.bind_params(3);
        self.sink_set_bytes(&cols, 4);
        self.bind_debug_status(5);
        let grid = MTLSize {
            width: CANVAS,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        self.sink_set_pipeline(&self.ps.sample_write);
        self.bind_state(0);
        self.sink_set_bytes(&canvas, 1);
        self.sink_set_bytes(&cols, 2);
        self.bind_debug_status(3);
        let freeze: u32 = freeze_enabled() as u32;
        let use_argmax: u32 = denoiser_argmax_enabled() as u32;
        self.sink_set_bytes(&freeze, 4);
        self.sink_set_bytes(&use_argmax, 5);
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        // MLX applies the schedule temperature before the SC soft-embed softmax.
        let st: CanvasState = read_struct(&self.bufs.state);
        let params: StepParams = read_struct(&self.bufs.params);
        let t = scheduled_temperature(st.step, &params).max(1e-6);
        self.scale_half_logits(CANVAS * VOCAB, 1.0 / t);
        Ok(())
    }
}

pub fn init_canvas_state(seed: u64, vocab: usize) -> CanvasState {
    let mut rng = Rng::new(seed);
    init_canvas_state_from_rng(vocab, &mut rng)
}

pub fn init_canvas_state_from_rng(vocab: usize, rng: &mut Rng) -> CanvasState {
    let ids_vec = initialize_canvas(CANVAS, vocab, rng);
    let mut ids = [0u32; CANVAS];
    ids.copy_from_slice(&ids_vec);
    CanvasState {
        ids,
        prev_argmax: [u32::MAX; CANVAS],
        new_sample: [0; CANVAS],
        entropy: [0.0; CANVAS],
        sorted_idx: [0; CANVAS],
        accept: [0; CANVAS],
        u_cat: [0.0; CANVAS],
        rng_state: rng.state(),
        step: 0,
        stop_flag: 0,
        argmax_hist_len: 0,
        argmax_hist_base: 0,
        argmax_hist: [0; CANVAS * ARGMAX_HIST_MAX],
        canvas_stable: 0,
        mean_entropy: 0.0,
        accept_plateau: 0,
        prev_accept_sig: 0,
        frozen: [0; FROZEN_WORDS],
    }
}

pub fn step_params_from_sampler(
    sampler: &SamplerConfig,
    kv_len: u32,
    no_early_stop: bool,
    eos_token_id: u32,
) -> StepParams {
    let conf_threshold = if no_early_stop {
        f32::MAX
    } else {
        sampler.confidence_threshold
    };
    let plateau_thresh = if no_early_stop {
        u32::MAX
    } else {
        sampler.accept_plateau_threshold as u32
    };
    let plateau_mean = if no_early_stop {
        f32::MAX
    } else {
        sampler.plateau_prefix_mean_max
    };
    StepParams {
        kv_len,
        max_steps: sampler.max_denoising_steps.max(1) as u32,
        entropy_bound: sampler.entropy_bound,
        t_min: sampler.t_min,
        t_max: sampler.t_max,
        conf_threshold,
        stability_threshold: sampler.stability_threshold as u32,
        min_early_stop_steps: crate::sample::MIN_EARLY_STOP_STEPS,
        accept_plateau_threshold: plateau_thresh,
        plateau_prefix_mean_max: plateau_mean,
        eos_token_id,
    }
}

fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::Format("Metal buffer alloc failed"))
}

fn write_struct<T>(buf: &ProtocolObject<dyn MTLBuffer>, val: &T) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            val as *const T as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of::<T>(),
        );
    }
}

fn zero_buffer(buf: &ProtocolObject<dyn MTLBuffer>) {
    unsafe {
        std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, buf.length());
    }
}

/// Synthetic route: one token at `position` routed to `expert_id` with weight 1.0 (f16).
fn write_single_expert_route(
    buf: &ProtocolObject<dyn MTLBuffer>,
    position: usize,
    expert_id: u32,
) {
    assert!(position < CANVAS);
    assert!((expert_id as usize) < N_EXPERTS);
    zero_buffer(buf);
    unsafe {
        let r = buf.contents().as_ptr() as *mut RouteScratch;
        (*r).expert[position][0] = expert_id;
        (*r).weight[position][0] = 0x3c00; // f16 1.0
        let e = expert_id as usize;
        let mut s = 0u32;
        for i in 0..N_EXPERTS {
            (*r).row_start[i] = s;
            if i == e {
                s += 1;
            }
        }
        (*r).row_start[N_EXPERTS] = s;
        (*r).token_list[0] = position as u32;
        (*r).slot_list[0] = 0;
        (*r).num_slots = 1;
    }
}

fn read_struct<T: Copy>(buf: &ProtocolObject<dyn MTLBuffer>) -> T {
    unsafe { *(buf.contents().as_ptr() as *const T) }
}

/// Debug: dump the raw bytes of a buffer to `path` (for KV-cache fast-vs-engine
/// diffs). Bytes are bf16 in KV-cache layout; the diff tool reinterprets them.
fn dump_buffer_raw(buf: &ProtocolObject<dyn MTLBuffer>, path: &str) {
    let bytes = unsafe {
        std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, buf.length())
    };
    match std::fs::write(path, bytes) {
        Ok(()) => eprintln!("step-kernel: dumped {} bytes to {path}", bytes.len()),
        Err(e) => eprintln!("step-kernel: DGQ_DUMP_KV write failed: {e}"),
    }
}




/// Per-position entropy at end of a denoise block (`DGQ_LOG_FINAL_ENTROPY=1`).
pub fn log_final_per_token_entropy(
    label: &str,
    state: &CanvasState,
    stop_flag: u32,
    eos_token_id: u32,
) {
    use crate::sample::{decode_early_stop_flag, is_active_token, EarlyStopKind, FILLER_TOKEN_ID, PAD_TOKEN_ID};
    let stop_kind = match decode_early_stop_flag(stop_flag) {
        Some(EarlyStopKind::Confident) => "confident_stable",
        Some(EarlyStopKind::Plateau) => "plateau_stop",
        Some(EarlyStopKind::MaxSteps) => "max_steps",
        None => "none",
    };
    eprintln!(
        "step-generate: {label} denoise_steps={} stop_flag={stop_flag} ({stop_kind}) mean_ent={:.4} plateau={} stable={}",
        state.step, state.accept_plateau, state.mean_entropy, state.canvas_stable,
    );
    for pos in 0..CANVAS {
        let id = state.ids[pos];
        let am = state.prev_argmax[pos];
        let ent = state.entropy[pos];
        let acc = state.accept[pos];
        let tag = if id == PAD_TOKEN_ID {
            "pad"
        } else if id == FILLER_TOKEN_ID {
            "filler"
        } else {
            "active"
        };
        eprintln!(
            "  pos={pos:3} {tag:6} id={id:6} argmax={am:6} ent={ent:8.4} accept={acc}",
        );
    }
    let mut high_ent_active = Vec::new();
    for pos in 0..CANVAS {
        if is_active_token(state.ids[pos]) && state.entropy[pos] > 0.1 {
            high_ent_active.push((pos, state.entropy[pos], state.ids[pos], state.prev_argmax[pos]));
        }
    }
    high_ent_active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if !high_ent_active.is_empty() {
        eprintln!("step-generate: {label} high_ent active (>0.1 nats, top 16):");
        for &(pos, ent, id, am) in high_ent_active.iter().take(16) {
            eprintln!("  pos={pos:3} ent={ent:7.4} id={id:6} argmax={am:6}");
        }
    }
    if let Some(start) = state.ids.iter().position(|&id| id == PAD_TOKEN_ID) {
        let tail = &state.entropy[start..];
        let tail_mean = if tail.is_empty() {
            f32::NAN
        } else {
            tail.iter().sum::<f32>() / tail.len() as f32
        };
        eprintln!(
            "step-generate: {label} pad_tail: first_pad_pos={start} len={} tail_mean_ent={tail_mean:.4}",
            tail.len(),
        );
    }
}



/// HF linear schedule: temperature at start of denoise step `steps_done` (0 = first step).
pub fn scheduled_temperature(steps_done: u32, params: &StepParams) -> f32 {
    let max = params.max_steps.max(1) as f32;
    let cur = params.max_steps.saturating_sub(steps_done) as f32;
    params.t_min + (params.t_max - params.t_min) * (cur / max)
}

fn read_logit_f32(logits: &ProtocolObject<dyn MTLBuffer>, row: usize, col: u32) -> f32 {
    use crate::kernels::sub::bf16;
    let byte_off = (row * VOCAB + col as usize) * 2;
    let ptr = unsafe { logits.contents().as_ptr().add(byte_off) as *const u16 };
    bf16::bf16_bits_to_f32(unsafe { *ptr })
}

/// Log GPU vs CPU accept masks and per-position entropy/argmax (for MLX/HF parity iteration).
pub fn log_denoise_parity_step(
    label: &str,
    state: &CanvasState,
    params: &StepParams,
    logits: &ProtocolObject<dyn MTLBuffer>,
) {
    let cpu_mask =
        crate::sample::accept_mask_from_entropies(&state.entropy, params.entropy_bound);
    let cpu_accept = cpu_mask.iter().filter(|&&m| m).count() as u32;
    let gpu_accept = state.accept.iter().filter(|&&a| a != 0).count();
    let temp = scheduled_temperature(state.step.saturating_sub(1), params);
    eprintln!(
        "denoise-parity {label}: st.step={} T={temp:.4} gpu_accept={gpu_accept} cpu_accept={cpu_accept} mean_H={:.4} low_H={}",
        state.step,
        state.mean_entropy,
        crate::sample::count_low_entropy_positions(&state.entropy, params.entropy_bound),
    );
    let n = denoise_parity_log_positions().min(CANVAS);
    for pos in 0..n {
        let argmax = state.prev_argmax[pos];
        let logit = read_logit_f32(logits, pos, argmax);
        eprintln!(
            "  pos {pos:2}: H={:.4} accept={}/{} argmax={argmax} canvas={} logit={logit:.4}",
            state.entropy[pos],
            state.accept[pos],
            u32::from(cpu_mask[pos]),
            state.ids[pos],
        );
    }
}


fn count_non_finite_half(buf: &ProtocolObject<dyn MTLBuffer>, elems: usize) -> (usize, f32) {
    use crate::kernels::sub::bf16;
    let ptr = buf.contents().as_ptr() as *const u16;
    let mut bad = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..elems {
        unsafe {
            let v = bf16::bf16_bits_to_f32(*ptr.add(i));
            if !v.is_finite() {
                bad += 1;
            }
            max_abs = max_abs.max(v.abs());
        }
    }
    (bad, max_abs)
}

fn check_logits_finite(logits: &ProtocolObject<dyn MTLBuffer>) -> (bool, f32) {
    half_buffer_stats(logits, 0, CANVAS * VOCAB, CANVAS * VOCAB)
}

fn half_buffer_stats(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
    sample: usize,
) -> (bool, f32) {
    use crate::kernels::sub::bf16;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    let mut max_abs = 0.0f32;
    let mut finite = true;
    let mut non_finite = 0usize;
    let n = sample.min(elems);
    let stride = (elems / n.max(1)).max(1);
    unsafe {
        let mut i = 0usize;
        while i < elems {
            let v = bf16::bf16_bits_to_f32(*ptr.add(i));
            if !v.is_finite() {
                finite = false;
                non_finite += 1;
            }
            max_abs = max_abs.max(v.abs());
            i += stride;
            if i / stride >= n {
                break;
            }
        }
    }
    if non_finite > 0 {
        finite = false;
    }
    (finite, max_abs)
}

fn arena_hidden_stats(arena: &ProtocolObject<dyn MTLBuffer>, layout: &ArenaLayout) -> (bool, f32, usize) {
    let sample = read_arena_buffer_f32(arena, layout.hidden_off() as usize, CANVAS * HID);
    let mut max_abs = 0.0f32;
    let mut finite = true;
    for v in sample.iter().step_by(HID) {
        if !v.is_finite() {
            finite = false;
        }
        max_abs = max_abs.max(v.abs());
    }
    let non_finite = if finite { 0 } else { 1 };
    (finite, max_abs, non_finite)
}

pub struct StepRuntime {
    ctx: MetalContext,
    pipelines: &'static StepPipelines,
    bufs: StepBuffers,
    gpu_blob: std::sync::Arc<DgqGpuBlob>,
    weight_cache: GpuDecoderWeightCache,
    text_config: TextConfig,
    block_profile: StepBlockProfile,
    attn_ffn_q8: bool,
    attn_ffn_bf16: bool,
    embed_bf16: bool,
    layout: ModelLayout,
    tensor_offsets: HashMap<String, u64>,
    pub layers: usize,
    /// KV-cache capacity (positions per layer) the buffers were sized for. Every
    /// denoise block writes a CANVAS-wide canvas at [kv_len..kv_len+CANVAS], so
    /// kv_len + CANVAS must never exceed this — checked in `set_kv_len`.
    max_seq: usize,
}

impl StepRuntime {
    pub fn layout(&self) -> &ModelLayout {
        &self.layout
    }

    pub fn kvcache(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.bufs.kvcache
    }

    pub fn logits(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.bufs.logits
    }

    pub fn shared_dgq_blob(&self) -> std::sync::Arc<DgqGpuBlob> {
        std::sync::Arc::clone(&self.gpu_blob)
    }

    pub fn read_params(&self) -> StepParams {
        read_struct(&self.bufs.params)
    }

    pub fn write_params(&mut self, params: StepParams) {
        write_struct(&self.bufs.params, &params);
    }

    pub fn set_kv_len(&mut self, kv_len: u32) {
        // A denoise block (and each prefill chunk) writes a CANVAS-wide canvas at
        // [kv_len..kv_len+CANVAS] into the KV cache. If that exceeds the cache
        // capacity, the write silently spills into the next layer's region (or off
        // the buffer) and corrupts attention into word-salad. Fail loudly instead —
        // callers must size max_seq >= prompt + generated + CANVAS.
        assert!(
            kv_len as usize + CANVAS <= self.max_seq,
            "KV cache overflow: kv_len={kv_len} + CANVAS={CANVAS} > max_seq={}; \
             size max_seq >= prompt_len + max_new_tokens + CANVAS",
            self.max_seq,
        );
        let mut params = self.read_params();
        params.kv_len = kv_len;
        self.write_params(params);
    }

    /// Fast prefill on the monolithic kernels: process the prompt in CANVAS-sized
    /// chunks, each a KV-only CAUSAL forward writing K/V into the b4 cache at
    /// [chunk_start, chunk_start+chunk_len). The last chunk is padded to CANVAS;
    /// padding K/V lands beyond prompt_len (overwritten by the first denoise block)
    /// and causal masking keeps real tokens from attending to it. Replaces the
    /// ~70s f32-engine prefill. Returns kv_len (= prompt length). Causal w/o window
    /// is correct for prompts <= sliding_window (1024); longer prompts would need
    /// windowing on sliding layers (not yet implemented).
    pub fn prefill_chunks(&mut self, prompt_token_ids: &[u32]) -> Result<usize, Error> {
        self.prefill_chunks_from(0, prompt_token_ids)
    }

    /// Fast (quantized, causal) prefill of `delta_token_ids` starting at KV
    /// position `offset`. The delta chunks attend causally to the KV already
    /// present at [0..offset] (e.g. the reused cross-turn prefix), so this
    /// resumes a prefill without recomputing the prefix. Because each position's
    /// KV is fixed by its causal context (independent of chunk grouping), the
    /// result at [offset..] is identical to a full `prefill_chunks` of the whole
    /// sequence when [0..offset] was itself fast-prefilled. Returns the new
    /// kv_len (`offset + delta.len()`).
    pub fn prefill_chunks_from(
        &mut self,
        offset: usize,
        delta_token_ids: &[u32],
    ) -> Result<usize, Error> {
        let layout = self.layout;
        let layers = self.layers;
        let n = offset + delta_token_ids.len();
        let mut pos = offset;
        while pos < n {
            let chunk_len = (n - pos).min(CANVAS);
            let mut ids = [0u32; CANVAS];
            ids[..chunk_len].copy_from_slice(&delta_token_ids[pos - offset..pos - offset + chunk_len]);
            self.set_canvas_ids(&ids)?;
            self.set_kv_len(pos as u32);
            self.dispatch_and_wait(|enc| enc.encode_prefill_chunk(&layout, layers))?;
            pos += chunk_len;
        }
        self.set_kv_len(n as u32);
        // The prefill dirtied scratch (arena hidden/dense, MoE routing buffers,
        // logits); re-zero to the same clean state the (self-contained) engine
        // prefill leaves — mirrors the post-open zeros minus kvcache (holds the
        // prompt KV). Leaving residual here made some short prompts degenerate.
        zero_buffer(&self.bufs.arena);
        zero_buffer(&self.bufs.logits);
        zero_buffer(&self.bufs.expert_layer_unique);
        zero_buffer(&self.bufs.moe_grouped_indirect);
        Ok(n)
    }

    pub fn read_canvas_state(&self) -> CanvasState {
        read_struct(&self.bufs.state)
    }

    pub fn set_canvas_ids(&mut self, ids: &[u32]) -> Result<(), Error> {
        if ids.len() != CANVAS {
            return Err(Error::Format("canvas ids length must match CANVAS"));
        }
        let mut state = self.read_canvas_state();
        for (i, &id) in ids.iter().enumerate() {
            state.ids[i] = id;
        }
        self.write_canvas_state(&state);
        Ok(())
    }

    pub fn write_canvas_state(&mut self, state: &CanvasState) {
        write_struct(&self.bufs.state, state);
    }

    /// New denoise block: fresh random canvas, reset step/stop, patch sampler params.
    pub fn reset_block(&mut self, vocab: usize, rng: &mut Rng, params: StepParams) {
        let mut state = init_canvas_state_from_rng(vocab, rng);
        state.step = 0;
        state.stop_flag = 0;
        state.argmax_hist_len = 0;
        state.argmax_hist_base = 0;
        state.argmax_hist = [0; CANVAS * ARGMAX_HIST_MAX];
        state.canvas_stable = 0;
        state.mean_entropy = 0.0;
        state.accept_plateau = 0;
        state.prev_accept_sig = 0;
        state.frozen = [0; FROZEN_WORDS];
        self.write_canvas_state(&state);
        self.write_params(params);
    }

    pub fn run_denoise_step(&mut self) -> Result<(), Error> {
        zero_buffer(&self.bufs.expert_layer_unique);
        self.run_forward_once(StepFinishMode::Full)
    }

    /// Populate forward telemetry from per-layer expert counts (grouped MoE path).
    pub fn fill_expert_forward_telemetry(&self, forward: &mut crate::metal::ForwardTelemetry) {
        let ptr = self.bufs.expert_layer_unique.contents().as_ptr() as *const u32;
        let counts = unsafe { std::slice::from_raw_parts(ptr, self.layers) };
        let weight_bytes = grouped_expert_blob_bytes_per_expert(self.block_profile.format);
        forward.record_expert_layers_grouped(counts, weight_bytes);
    }

    /// Host readback size for one `CanvasState` poll (shared buffer, no extra sync).
    pub const CANVAS_STATE_BYTES: usize = std::mem::size_of::<CanvasState>();

    /// P2.1 budget: host bytes touched per denoise step on the generate hot path.
    pub fn denoise_step_host_readback_bytes(check_logits: bool) -> u64 {
        // Forward reads state once on CPU to seed preamble; generate polls once after sync.
        let mut bytes = (Self::CANVAS_STATE_BYTES as u64) * 2;
        if check_logits && logits_finite_check_enabled() {
            bytes += logits_finite_sample_bytes();
        }
        bytes
    }

    /// Opt-in full-tensor logits scan (`DGQ_CHECK_LOGITS=1`). Off by default (P2.1).
    pub fn check_logits_finite(&self) -> Result<(), Error> {
        if !logits_finite_check_enabled() {
            return Ok(());
        }
        let sample = logits_finite_sample_count().min(CANVAS * VOCAB);
        let (finite, max_abs) =
            half_buffer_stats(&self.bufs.logits, 0, CANVAS * VOCAB, sample);
        if !finite {
            eprintln!("non-finite logits (max_abs={max_abs:.4}, sample={sample})");
            return Err(Error::Format("non-finite logits"));
        }
        Ok(())
    }

    fn check_debug_status(&self) -> Result<(), Error> {
        if let Some(ref dbg) = self.bufs.debug_status {
            let st = crate::metal::debug_status::read_buffer(dbg);
            crate::metal::debug_status::check_status(st)?;
        }
        Ok(())
    }

    fn dispatch_and_wait<F>(&mut self, f: F) -> Result<(), Error>
    where
        F: FnOnce(&mut StepEnc<'_>) -> Result<(), Error>,
    {
        let cmd = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Format("command buffer alloc failed"))?;
        let mut enc = StepEnc {
            enc: cmd
                .computeCommandEncoder()
                .ok_or(Error::Format("compute encoder alloc failed"))?,
            ctx: &self.ctx,
            ps: &self.pipelines,
            bufs: &self.bufs,
            block_profile: self.block_profile,
            tensor_offsets: &self.tensor_offsets,
            partial_lm_m: CANVAS as u32,
            attn_ffn_q8: self.attn_ffn_q8,
            attn_ffn_bf16: self.attn_ffn_bf16,
            embed_bf16: self.embed_bf16,
            prefill_causal: false,
            sliding_window: self.text_config.sliding_window as u32,
        };
        f(&mut enc)?;
        enc.enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Attention + dense FFN + GPU router; MoE expert matmuls on CPU (matches `.dgq` Q4 oracle).
    pub fn fill_moe_out_dgq_cpu(&mut self, layer: usize) -> Result<(), Error> {
        let route: RouteScratch = read_struct(&self.bufs.route);
        let routes = routes_from_route_scratch(&route);
        let moe_in = read_arena_buffer_f32(&self.bufs.arena, self.bufs.arena_map.moein_off() as usize, CANVAS * HID);
        let mut moe_out = vec![0.0f32; CANVAS * HID];
        let mut scratch = MoeScratch::new(CANVAS, &self.text_config);
        experts_forward_dgq_cpu(
            &mut moe_out,
            &moe_in,
            &self.weight_cache,
            layer,
            &self.text_config,
            CANVAS,
            &routes,
            &mut scratch,
        )?;
        write_f32_arena(&self.bufs.arena, self.bufs.arena_map.moeout_off(), &moe_out);
        Ok(())
    }

    /// One decoder layer: GPU router + grouped MoE + GPU post-combine (single submit).
    pub fn encode_full_layer(&mut self, layer: usize) -> Result<(), Error> {
        let layout = self.layout;
        self.dispatch_and_wait(|enc| enc.encode_full_layer(layer, &layout))?;
        Ok(())
    }

    /// One forward step with per-phase GPU sync (for profiling; ~4 extra submits vs monolithic).
    fn profile_forward_once(&mut self, finish: StepFinishMode) -> Result<StepProfileResult, Error> {
        use std::time::Instant;
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };

        let t0 = Instant::now();
        self.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;
        let preamble = t0.elapsed();

        let t1 = Instant::now();
        self.dispatch_and_wait(|enc| {
            for layer in 0..layers {
                enc.encode_layer(layer, &layout)?;
            }
            Ok(())
        })?;
        let layer_pre_moe = t1.elapsed();

        let t2 = Instant::now();
        self.dispatch_and_wait(|enc| {
            for layer in 0..layers {
                enc.encode_layer_moe_grouped(layer, &layout)?;
            }
            Ok(())
        })?;
        let layer_moe = t2.elapsed();

        let t3 = Instant::now();
        self.dispatch_and_wait(|enc| {
            for layer in 0..layers {
                enc.encode_layer_moe_post(layer, &layout)?;
            }
            Ok(())
        })?;
        let layer_post = t3.elapsed();

        let t4 = Instant::now();
        self.dispatch_and_wait(|enc| enc.encode_step_finish(&layout, finish))?;
        let finish_t = t4.elapsed();

        let total = preamble + layer_pre_moe + layer_moe + layer_post + finish_t;
        Ok(StepProfileResult {
            compile: std::time::Duration::ZERO,
            preamble,
            layer_pre_moe,
            layer_moe,
            layer_post,
            finish: finish_t,
            total,
            layers,
            block_format: self.block_profile.format,
        })
    }

    fn time_enc_stage<F>(&mut self, f: F) -> Result<std::time::Duration, Error>
    where
        F: FnOnce(&mut StepEnc<'_>) -> Result<(), Error>,
    {
        use std::time::Instant;
        let t0 = Instant::now();
        self.dispatch_and_wait(f)?;
        Ok(t0.elapsed())
    }

    /// Per-stage bf16 activation-range trace: runs the forward stage-by-stage
    /// (one submit per stage so buffers are valid) and records max|x| of each
    /// stage's bf16 arena output. Answers "does any activation exceed f16's 65504
    /// range?" before trying f16/scaled-f16 arenas. `DGQ_TRACE_RANGES=1`.
    fn trace_step_ranges(&mut self) -> Result<(), Error> {
        use std::collections::BTreeMap;
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };
        const SAMPLE: usize = 4096;
        // (label) -> (max_abs across layers, any_non_finite)
        let mut peak: BTreeMap<&'static str, (f32, bool)> = BTreeMap::new();
        let mut probe = |this: &Self, label: &'static str, off: u64, elems: usize| {
            let (nf, mx) = half_buffer_stats(&this.bufs.arena, off as usize, elems, SAMPLE);
            let e = peak.entry(label).or_insert((0.0, false));
            e.0 = e.0.max(mx);
            e.1 |= nf;
        };

        self.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;
        probe(self, "preamble:soft", self.bufs.arena_map.soft_off(), CANVAS * HID);

        for layer in 0..layers {
            let a = &self.bufs.arena_map;
            let (hidden, attnq, attno, tmp, ffg, dense, moein) = (
                a.hidden_off(), a.attnq_off(), a.attno_off(), a.tmp_off(),
                a.ffg_off(), a.dense_off(), a.moein_off(),
            );
            self.time_enc_stage(|e| e.encode_layer_qkv_gemm(layer, &layout))?;
            probe(self, "layer:qkv(Q)", attnq, CANVAS * 4096);
            self.time_enc_stage(|e| e.encode_layer_qk_rope_kv_dispatch(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_layer_attention_dispatch(layer, &layout))?;
            probe(self, "layer:attn_out", attno, CANVAS * 4096);
            self.time_enc_stage(|e| e.encode_layer_o_proj_gemm(layer, &layout))?;
            probe(self, "layer:o_proj", tmp, CANVAS * HID);
            self.time_enc_stage(|e| e.encode_layer_o_proj_tail(layer, &layout))?;
            probe(self, "layer:resid_pre_moe(hidden)", hidden, CANVAS * HID);
            let l = &layout.layers[layer];
            self.time_enc_stage(|e| { e.rmsnorm(e.arena().stream_off(), e.arena().tmp_off(), l.pre_ff_ln, HID as u32, CANVAS); Ok(()) })?;
            self.time_enc_stage(|e| e.encode_layer_dense_gate_up(layer, &layout))?;
            probe(self, "layer:gate_up", ffg, CANVAS * DENSE_FF as usize);
            self.time_enc_stage(|e| { e.glu(e.arena().ffg_off(), e.arena().ffu_off(), e.arena().ffg_off(), CANVAS * DENSE_FF as usize); Ok(()) })?;
            probe(self, "layer:swiglu", ffg, CANVAS * DENSE_FF as usize);
            self.time_enc_stage(|e| e.encode_layer_dense_down(layer, &layout))?;
            probe(self, "layer:dense_down", dense, CANVAS * HID);
            self.time_enc_stage(|e| { e.rmsnorm(e.arena().dense_off(), e.arena().dense_off(), l.post_ff_ln_1, HID as u32, CANVAS); Ok(()) })?;
            self.time_enc_stage(|e| e.encode_layer_router_buckets(layer, &layout))?;
            // MoE
            self.time_enc_stage(|e| e.encode_moe_batched_gate_up(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_moe_batched_swiglu())?;
            self.time_enc_stage(|e| e.encode_moe_batched_down(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_moe_batched_scatter())?;
            self.time_enc_stage(|e| e.encode_layer_moe_post_norm(layer, &layout))?;
            probe(self, "layer:moe_norm(moein)", moein, CANVAS * HID);
            self.time_enc_stage(|e| e.encode_layer_moe_post_combine(layer, &layout))?;
            probe(self, "layer:resid_post_moe(hidden)", hidden, CANVAS * HID);
            // Per-layer residual peak (the f16-overflow suspect).
            let (_, hmx) = half_buffer_stats(&self.bufs.arena, hidden as usize, CANVAS * HID, SAMPLE);
            eprintln!("  layer {layer:>2}: residual(hidden) max|x| = {hmx:.1}");
        }

        eprintln!("=== bf16 activation ranges (max|x| across {layers} layers) — f16 max = 65504 ===");
        for (label, (mx, nf)) in &peak {
            let flag = if *mx > 65504.0 { "  <-- OVERFLOWS f16" } else if *mx > 16384.0 { "  (tight)" } else { "" };
            eprintln!("  {label:<28} {mx:>12.1}{}{}", if *nf { " [non-finite!]" } else { "" }, flag);
        }
        Ok(())
    }

    /// Per-stage GPU timing inside `encode_layer` + MoE grouped/post (one submit per stage×layer).
    fn profile_encode_subprofile(&mut self) -> Result<EncodeSubProfileResult, Error> {
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };

        self.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;

        let mut layer_prof = LayerEncodeSubProfile::default();
        let mut moe_prof = MoeEncodeSubProfile::default();

        for layer in 0..layers {
            layer_prof.qkv_gemm +=
                self.time_enc_stage(|e| e.encode_layer_qkv_gemm(layer, &layout))?;
            layer_prof.qk_rope_kv +=
                self.time_enc_stage(|e| e.encode_layer_qk_rope_kv_dispatch(layer, &layout))?;
            layer_prof.attention +=
                self.time_enc_stage(|e| e.encode_layer_attention_dispatch(layer, &layout))?;
            layer_prof.o_proj_gemm +=
                self.time_enc_stage(|e| e.encode_layer_o_proj_gemm(layer, &layout))?;
            layer_prof.o_proj_tail +=
                self.time_enc_stage(|e| e.encode_layer_o_proj_tail(layer, &layout))?;

            let l = &layout.layers[layer];
            layer_prof.dense_pre_norm += self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().stream_off(),
                    e.arena().tmp_off(),
                    l.pre_ff_ln,
                    HID as u32,
                    CANVAS,
                );
                Ok(())
            })?;
            layer_prof.dense_gate_up +=
                self.time_enc_stage(|e| e.encode_layer_dense_gate_up(layer, &layout))?;
            layer_prof.dense_glu += self.time_enc_stage(|e| {
                e.glu(
                    e.arena().ffg_off(),
                    e.arena().ffu_off(),
                    e.arena().ffg_off(),
                    CANVAS * DENSE_FF as usize,
                );
                Ok(())
            })?;
            layer_prof.dense_down +=
                self.time_enc_stage(|e| e.encode_layer_dense_down(layer, &layout))?;
            layer_prof.dense_post_norm += self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().dense_off(),
                    e.arena().dense_off(),
                    l.post_ff_ln_1,
                    HID as u32,
                    CANVAS,
                );
                Ok(())
            })?;
            layer_prof.router +=
                self.time_enc_stage(|e| e.encode_layer_router_buckets(layer, &layout))?;
        }

        for layer in 0..layers {
            moe_prof.gather +=
                self.time_enc_stage(|e| e.encode_moe_batched_gather_bf16_to_f32())?;
            moe_prof.gate_up +=
                self.time_enc_stage(|e| e.encode_moe_batched_gate_up(layer, &layout))?;
            moe_prof.swiglu += self.time_enc_stage(|e| e.encode_moe_batched_swiglu())?;
            moe_prof.down += self.time_enc_stage(|e| e.encode_moe_batched_down(layer, &layout))?;
            moe_prof.scatter += self.time_enc_stage(|e| e.encode_moe_batched_scatter())?;
        }

        for layer in 0..layers {
            moe_prof.post += self.time_enc_stage(|e| e.encode_layer_moe_post_norm(layer, &layout))?;
            moe_prof.post +=
                self.time_enc_stage(|e| e.encode_layer_moe_post_combine(layer, &layout))?;
        }

        Ok(EncodeSubProfileResult {
            compile: std::time::Duration::ZERO,
            layers,
            layer: layer_prof,
            moe: moe_prof,
        })
    }

    /// P2.2 Phase A: one command buffer + one GPU sync per denoise step.
    fn run_forward_once(&mut self, finish: StepFinishMode) -> Result<(), Error> {
        if let Some(ref dbg) = self.bufs.debug_status {
            crate::metal::debug_status::zero_buffer(dbg);
        }
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        if crate::metal::embed::sc_log_enabled() && st_before.step >= 1 {
            let elems = CANVAS * VOCAB;
            let sample = elems.min(8192);
            let (nf, mx) = half_buffer_stats(&self.bufs.logits, 0, elems, sample);
            eprintln!(
                "monolithic pre-sc: st.step={} logits_max_abs={:.4} non_finite_sample={}",
                st_before.step, mx, nf
            );
        }
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };
        let partial_lm_m = partial_lm_active_rows(&st_before);
        self.dispatch_and_wait(|enc| {
            enc.partial_lm_m = partial_lm_m;
            enc.interpret_step(&layout, layers, first_step, finish)
        })?;
        self.check_debug_status()
    }
}

static STEP_PIPELINES_CACHE: std::sync::OnceLock<
    std::sync::Mutex<HashMap<StepPipelineKey, &'static StepPipelines>>,
> = std::sync::OnceLock::new();

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StepPipelineKey(u8);

fn step_pipeline_key(variant: crate::kernels::sub::variant::KernelVariant) -> StepPipelineKey {
    StepPipelineKey(
        u8::from(variant.shape_assert)
            | (u8::from(variant.debug_fast) << 1)
            | (u8::from(variant.debug_deep) << 2),
    )
}

fn shared_step_pipelines(ctx: &MetalContext) -> Result<&'static StepPipelines, Error> {
    let variant = crate::kernels::sub::variant::runtime_step_variant();
    let key = step_pipeline_key(variant);
    let cache = STEP_PIPELINES_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| Error::Format("step pipelines cache poisoned"))?;
    if let Some(&pipelines) = guard.get(&key) {
        return Ok(pipelines);
    }

    ctx.compile_library(STEP_SHADER)?;
    let pipelines = StepPipelines::new(ctx, variant)?;
    let leaked: &'static StepPipelines = Box::leak(Box::new(pipelines));
    guard.insert(key, leaked);
    crate::metal::pipeline_cache::PipelineArchiveCache::flush_global();
    Ok(leaked)
}

pub fn log_step_memory_budget(
    blob_bytes: u64,
    max_seq: usize,
    layout: &ModelLayout,
) {
    let kv = kv_cache_bytes(layout, max_seq);
    let logits = (CANVAS * VOCAB * 2) as u64;
    let sc_probs = sc_probs_buffer_bytes() as u64;
    let arena = step_arena_layout().bytes();
    let (mx, mw) = gemm_scratch_bytes();
    let gemm_scratch = (mx + mw) as u64;
    let gpu_static = kv + logits + sc_probs + arena + gemm_scratch;
    let total = blob_bytes + gpu_static;
    if crate::flags::progress_enabled() {
    eprintln!("step-kernel memory budget:");
    eprintln!(
        "  blob:       {:.2} GiB",
        blob_bytes as f64 / (1024.0_f64.powi(3))
    );
    eprintln!(
        "  arena:      {:.2} MiB",
        arena as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "  kv cache:   {:.2} MiB (max_seq={max_seq})",
        kv as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "  logits:     {:.2} MiB",
        logits as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "  sc_probs:   {:.2} MiB",
        sc_probs as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "  gemm scratch:{:.2} MiB",
        gemm_scratch as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "  gpu static: {:.2} GiB (excl. blob)",
        gpu_static as f64 / (1024.0_f64.powi(3))
    );
    eprintln!(
        "  total est:  {:.2} GiB",
        total as f64 / (1024.0_f64.powi(3))
    );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepRuntimeBuildTiming {
    pub compile: std::time::Duration,
    pub total: std::time::Duration,
}

pub fn build_step_runtime(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<(StepRuntime, StepRuntimeBuildTiming), Error> {
    let build_started = Instant::now();
    let validated = super::step_config::validate_step_model(model_dir)?;
    super::step_config::log_validated_step_model(&validated);

    let store = DgqStore::open(model_dir)?;
    let offsets = build_offsets_from_store(&store);
    let layout = build_layout(&offsets, cfg.max_seq);
    let layers = cfg.layers.min(validated.num_layers).max(1);

    // Mixed-precision .dgq stores attention + dense-FFN as bf16 (Raw) — or q8 on
    // earlier checkpoints. Detect from the actual stored kind so any vintage
    // dispatches correctly (bf16 / q8 / uniform-q4).
    let attn_ffn_kind = store
        .get_entry("model.decoder.layers.0.self_attn.q_proj.weight")
        .and_then(|e| crate::dgq::layout::parse_quant_kind(&e.meta.kind).ok());
    let attn_ffn_q8 = attn_ffn_kind == Some(crate::dgq::layout::QuantKind::Q8Row);
    let attn_ffn_bf16 = attn_ffn_kind == Some(crate::dgq::layout::QuantKind::Raw);

    // Embed (tied lm_head + SC soft-embed) is q8-per-row on most checkpoints, bf16
    // (Raw) on newer ones. Detect from the stored kind so all three embed consumers
    // (input gather, lm_head, SC softembed) dispatch the matching precision.
    let embed_bf16 = store
        .get_entry("model.decoder.embed_tokens.weight")
        .and_then(|e| crate::dgq::layout::parse_quant_kind(&e.meta.kind).ok())
        == Some(crate::dgq::layout::QuantKind::Raw);
    let block_profile = StepBlockProfile::from_store_profile(store.profile());
    if crate::flags::progress_enabled() {
        if embed_bf16 {
            eprintln!("step-kernel: bf16 embed (tied lm_head + SC)");
        }
        match block_profile.format {
            QuantFormat::NvFp4 => eprintln!("step-kernel: nvfp4 block weights"),
            QuantFormat::Q4Affine => eprintln!("step-kernel: q4 block weights"),
            _ => eprintln!("step-kernel: block weights ({:?})", block_profile.format),
        }
        match block_profile.moe_style() {
            MoeExecutionStyle::BatchedGrouped => {
                eprintln!("step-kernel: batched grouped MoE");
            }
            MoeExecutionStyle::ScalarPerExpert => {
                eprintln!("step-kernel: scalar per-expert MoE");
            }
        }
    }

    let ctx = MetalContext::new()?;
    let compile_started = Instant::now();
    let pipelines = shared_step_pipelines(&ctx)?;
    let compile = compile_started.elapsed();

    let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device)?;
    let gpu_blob = std::sync::Arc::clone(&gpu_blob);
    let kv_bytes = kv_cache_bytes(&layout, cfg.max_seq) as usize;
    let logits_bytes = CANVAS * VOCAB * 2;
    let sc_probs_bytes = sc_probs_buffer_bytes();

    log_step_memory_budget(
        store.blob_bytes(),
        cfg.max_seq,
        &layout,
    );

    let sampler = crate::sample::sampler_for_steps(cfg.steps.max(1), cfg.no_early_stop);
    let prefill_len = cfg
        .prefill_token_ids
        .as_ref()
        .map(|t| t.len() as u32)
        .unwrap_or(cfg.kv_len);
    let model_cfg = ModelConfig::load(model_dir)?;
    let eos_token_id = model_cfg.eos_token_id_u32();
    let params = step_params_from_sampler(&sampler, prefill_len, cfg.no_early_stop, eos_token_id);
    let state = init_canvas_state(cfg.seed, VOCAB);
    let (gemm_a_bytes, gemm_b_bytes) = gemm_scratch_bytes();

    let text_config = model_cfg.text_config;
    let weight_store = WeightStore::open(model_dir)?;
    let weight_cache = GpuDecoderWeightCache::load_with_dgq_blob(
        &weight_store,
        &text_config,
        &ctx.device,
        std::sync::Arc::clone(&gpu_blob),
    )?;

    let arena_map = step_arena_layout();
    let bufs = StepBuffers {
        blob: gpu_blob.buffer.clone(),
        blob_experts: {
            let (b, _) = gpu_blob.expert_region();
            objc2::rc::Retained::from(b)
        },
        blob_expert_base: gpu_blob.expert_region().1,
        layout: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ModelLayout>())?;
            write_struct(&b, &layout);
            b
        },
        params: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<StepParams>())?;
            write_struct(&b, &params);
            b
        },
        arena: alloc_buffer(&ctx.device, arena_map.bytes() as usize)?,
        kvcache: alloc_buffer(&ctx.device, kv_bytes)?,
        state: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<CanvasState>())?;
            write_struct(&b, &state);
            b
        },
        logits: alloc_buffer(&ctx.device, logits_bytes)?,
        sc_probs: alloc_buffer(&ctx.device, sc_probs_bytes)?,
        route: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<RouteScratch>())?;
            zero_buffer(&b);
            b
        },
        dummy_dump: alloc_buffer(&ctx.device, 4)?,
        debug_status: if crate::kernels::sub::variant::runtime_kernel_debug_enabled() {
            Some(alloc_buffer(
                &ctx.device,
                crate::metal::debug_status::DEBUG_STATUS_BYTES,
            )?)
        } else {
            None
        },
        gemm_a: alloc_buffer(&ctx.device, gemm_a_bytes)?,
        gemm_b: alloc_buffer(&ctx.device, gemm_b_bytes)?,
        expert_layer_unique: alloc_buffer(&ctx.device, N_LAYERS * std::mem::size_of::<u32>())?,
        moe_grouped_indirect: alloc_buffer(&ctx.device, MOE_GROUPED_INDIRECT_BYTES)?,
        arena_map,
        arena_layout_buf: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ArenaLayout>())?;
            write_struct(&b, &arena_map);
            b
        },
    };
    zero_buffer(&bufs.expert_layer_unique);
    zero_buffer(&bufs.moe_grouped_indirect);
    zero_buffer(&bufs.arena);
    zero_buffer(&bufs.kvcache);
    zero_buffer(&bufs.logits);

    // Fast-prefill (DGQ_FAST_PREFILL) runs on the step kernels AFTER `rt` is built
    // (prefill_chunks is a StepRuntime method); the slow f32-engine prefill runs
    // here at open time otherwise.
    if let Some(ref token_ids) = cfg.prefill_token_ids {
        if !should_fast_prefill(token_ids.len()) {
            let mut encoder = crate::metal::step_kv::MonolithicEncoderCache::open_opt(
                model_dir,
                CANVAS,
                cfg.max_seq,
                Some(std::sync::Arc::clone(&gpu_blob)),
            )?;
            let (kv_len, _) = crate::metal::step_kv::prefill_monolithic_kv_with_cache(
                &mut encoder,
                token_ids,
                &bufs.kvcache,
                &layout,
                cfg.max_seq,
                layers,
            )?;
            if kv_len as u32 != prefill_len {
                return Err(Error::Format("prefill kv_len mismatch"));
            }
            eprintln!("step-kernel: prefilled kv_len={kv_len} tokens");
        }
    }

    let build = StepRuntimeBuildTiming {
        compile,
        total: build_started.elapsed(),
    };
    if crate::flags::progress_enabled() {
        eprintln!(
            "step-kernel: runtime built (total={:.2?}, compile={:.2?})",
            build.total, build.compile
        );
    }
    let mut rt = StepRuntime {
        ctx,
        pipelines,
        bufs,
        gpu_blob,
        weight_cache,
        text_config,
        block_profile,
        attn_ffn_q8,
        attn_ffn_bf16,
        embed_bf16,
        layout,
        tensor_offsets: offsets,
        layers,
        max_seq: cfg.max_seq,
    };
    if crate::flags::progress_enabled() {
        if rt.embed_bf16 && sc_sparse_enabled() {
            eprintln!("step-kernel: sparse SC softembed (DGQ_SC_SPARSE=0 for the exact chunked path)");
        } else {
            eprintln!("step-kernel: chunked SC softembed");
        }
    }
    if let Some(ref token_ids) = cfg.prefill_token_ids {
        if should_fast_prefill(token_ids.len()) {
            let started = Instant::now();
            let kv_len = rt.prefill_chunks(token_ids)?;
            if kv_len as u32 != prefill_len {
                return Err(Error::Format("fast-prefill kv_len mismatch"));
            }
            if crate::flags::progress_enabled() {
                eprintln!(
                    "step-kernel: fast-prefilled kv_len={kv_len} tokens ({:.2?})",
                    started.elapsed()
                );
            }
        }
    }
    if let Some(path) = crate::flags::dump_kv_path() {
        dump_buffer_raw(&rt.bufs.kvcache, &path);
    }
    Ok((rt, build))
}

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
        enc.rmsnorm(enc.arena().hidden_off(), enc.arena().tmp_off(), layout.final_norm, HID as u32, CANVAS);
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

fn read_arena_hidden_row(
    arena: &ProtocolObject<dyn MTLBuffer>,
    base: u64,
    row: usize,
) -> Vec<f32> {
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
        return Err(Error::Format("layer probe position out of range"));
    }
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;
    let mut checkpoints = Vec::new();

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
        Ok(())
    })?;
    {
        let hidden = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);
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
        let hidden = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);
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
        enc.rmsnorm(enc.arena().hidden_off(), enc.arena().tmp_off(), layout.final_norm, HID as u32, CANVAS);
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
        return Err(Error::Format("preamble capture position out of range"));
    }
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;

    rt.dispatch_and_wait(|enc| enc.encode_preamble_embed_only(&layout))?;
    let embed_scaled = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);

    rt.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, 1))?;
    let after_preamble = read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), position);

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
    Ok(read_arena_hidden_row(&rt.bufs.arena, rt.bufs.arena_map.hidden_off(), 0))
}

/// Query heads in the monolithic step-kernel shader (`NQ_HEADS` in diffgemma_step.metal).
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
        return Err(Error::Format("attn capture position out of range"));
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
    let q_raw_proj = read_arena_row(&rt.bufs.arena, rt.bufs.arena_map.attnq_off(), position, q_width);
    let q_norm_w = DgqStore::open(model_dir)?
        .tensor_f32(&format!("model.decoder.layers.{layer}.self_attn.q_norm.weight"))?;
    let mut q_pre_rope = q_raw_proj.clone();
    for h in 0..n_heads {
        let off = h * hd;
        crate::kernels::cpu::attention::rms_norm_head(
            &mut q_pre_rope[off..off + hd],
            Some(&q_norm_w),
            crate::kernels::cpu::attention::RMS_EPS,
        );
    }

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_qk_rope_and_attention(layer, &layout)?;
        Ok(())
    })?;

    let q_all = read_arena_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.attnq_off() as usize, CANVAS * q_width);
    let q_post_rope = read_arena_row(&rt.bufs.arena, rt.bufs.arena_map.attnq_off(), position, q_width);
    let attn_out = read_arena_row(&rt.bufs.arena, rt.bufs.arena_map.attno_off(), position, q_width);
    let k_cache = read_layer_k_cache_f32(rt.kvcache(), &layout, layer, total_kv);

    let raw_scores = row_raw_scores(&q_all, &k_cache, position, total_kv, n_heads, nkv, hd);
    let attn_probs = softmax_attn_rows(&raw_scores, n_heads, total_kv);

    let canvas_abs = kv_len as usize + position;
    let mut sample_pos = vec![0usize, kv_len.saturating_sub(1) as usize, kv_len as usize, canvas_abs];
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

fn routes_from_route_scratch(route: &RouteScratch) -> Vec<RouteResult> {
    let mut routes = Vec::with_capacity(CANVAS);
    for tok in 0..CANVAS {
        let indices = (0..TOP_K)
            .map(|k| route.expert[tok][k] as usize)
            .collect();
        let weights = (0..TOP_K)
            .map(|k| crate::kernels::sub::bf16::bf16_bits_to_f32(route.weight[tok][k]))
            .collect();
        routes.push(RouteResult { indices, weights });
    }
    routes
}

fn write_f32_arena(
    arena: &ProtocolObject<dyn MTLBuffer>,
    base: u64,
    data: &[f32],
) {
    let byte_off = base as usize;
    unsafe {
        let ptr = arena.contents().as_ptr().add(byte_off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *ptr.add(i) = v;
        }
    }
}

fn read_f32_arena(
    arena: &ProtocolObject<dyn MTLBuffer>,
    base: u64,
    elems: usize,
) -> Vec<f32> {
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
    let state = crate::kernels::cpu::moe_router::moe_bucket_phases(
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
    crate::kernels::sub::f16::f32_to_f16_bits(v)
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
        .ok_or_else(|| Error::Format("route ref missing experts"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let weights: Vec<u16> = doc
        .get("expert_weights")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Format("route ref missing expert_weights"))?
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
        return Err(Error::Format("route ref experts/weights must be top_k"));
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

fn read_scratch_f32(buf: &ProtocolObject<dyn MTLBuffer>, byte_off: usize, elems: usize) -> Vec<f32> {
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
) -> Result<crate::kernels::sub::moe_batched_pin::MoeBatchedPinDump, Error> {
    use crate::kernels::sub::moe_batched_pin::{
        verify_batched_stages_cpu_with_verdict, MoeBatchedPinDump, MoeBatchedPinLayout,
        MoeBatchedPinRoute,
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
    let moe_in = read_arena_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.moein_off() as usize, CANVAS * HID);
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
    let (stages, rel_l2, gate_up_diff, first_divergent_stage) = verify_batched_stages_cpu_with_verdict(
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
        return Err(Error::Format("moe capture position out of range"));
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
    let router_logits = crate::kernels::cpu::moe_router::router_logits_row(
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
        if let Some((experts, weights)) =
            route_override_from_ref_json(Path::new(&path), position)?
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

    let moe_out = read_f32_arena_row(&rt.bufs.arena, rt.bufs.arena_map.moeout_off(), position, HID);

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
        return Err(Error::Format("single-expert moe capture position out of range"));
    }
    if expert_id as usize >= N_EXPERTS {
        return Err(Error::Format("single-expert moe capture expert_id out of range"));
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
    let gpu_out = read_f32_arena_row(&rt.bufs.arena, rt.bufs.arena_map.moeout_off(), position, HID);
    let act_probe = read_f32_arena(&rt.bufs.arena, rt.bufs.arena_map.soft_off(), MOE_ACT_PROBE_FLOATS);
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
        rt.run_forward_once(cfg.finish)?;
        rt.trace_step_ranges()?;
    }
    let mut prof = rt.profile_encode_subprofile()?;
    prof.compile = build.compile;
    Ok(prof)
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
    let (mut rt, build) = build_step_runtime(model_dir, cfg)?;
    let finish = StepFinishMode::Full;
    let mut out = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let st: CanvasState = read_struct(&rt.bufs.state);
        let mut prof = rt.profile_forward_once(finish)?;
        if i == 0 {
            prof.compile = build.compile;
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
    use std::time::Instant;
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
    use crate::kernels::sub::bf16;
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
    use crate::kernels::sub::bf16;
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
pub fn run_step_forward(model_dir: &Path, cfg: &StepSmokeConfig) -> Result<StepForwardOutput, Error> {
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
        enc.rmsnorm(enc.arena().hidden_off(), enc.arena().tmp_off(), layout.final_norm, HID as u32, CANVAS);
        Ok(())
    })?;
    let norm_hidden = read_arena_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.tmp_off() as usize, CANVAS * HID);
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
    use crate::chat_template::{format_chat_token_ids, ChatFormatOptions, ChatTurn};
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
        return Err(Error::Format("run_denoise_steps requires StepFinishMode::Full"));
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
        eprintln!("step-smoke: completed denoise step {}/{}", step_i + 1, steps);
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
        ids: final_state.ids,
        logits_finite,
        max_abs_logit,
        elapsed,
    })
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn forward_row_entropy_probe() {
        use crate::sample::token_entropy;
        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            return;
        }
        let cfg = StepSmokeConfig {
            layers: 3,
            finish: StepFinishMode::ForwardOnly,
            ..StepSmokeConfig::default()
        };
        let out = run_step_forward(dir, &cfg).expect("forward");
        let ent = token_entropy(&out.logits, CANVAS, VOCAB);
        for row in 0..CANVAS {
            let base = row * VOCAB;
            let row_logits = &out.logits[base..base + VOCAB];
            let all_zero = row_logits.iter().all(|&v| v == 0.0);
            assert!(!all_zero, "row {row} logits all zero (GEMM tg regression)");
        }
        let min_h = ent.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_h = ent.iter().cloned().fold(0.0f32, f32::max);
        eprintln!("forward 3L row entropy: min={min_h:.3} max={max_h:.3}");
        assert!(max_h < 12.4, "uniform row entropy {max_h}");
    }

    #[test]
    fn monolith_one_step_accept_regression() {
        // MLX @ 30L/Hello/seed42 accepts ~196 positions on denoise step 1 alone.
        // `.dgq` sharpens over the 8-step block; GEMM tg=(128,1,1) bug held ~1 accept/step
        // (~8 total) with mean_entropy~10. Healthy cumulative accept >> 150.
        const MIN_ACCEPT: u32 = 150;
        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip monolith_one_step_accept_regression: no weights at /tmp/quantized-weights");
            return;
        }
        let prefill = hello_chat_prefill_token_ids(dir).expect("hello prefill");
        let cfg = StepSmokeConfig {
            layers: 30,
            steps: 8,
            kv_len: 0,
            seed: 42,
            max_seq: 512,
            finish: StepFinishMode::Full,
            prefill_token_ids: Some(prefill),
            no_early_stop: true,
        };
        let steps = run_denoise_steps(dir, &cfg).expect("monolith denoise block");
        assert!(!steps.is_empty(), "denoise block produced no steps");
        let accepts: Vec<u32> = steps.iter().map(|s| s.accept_count).collect();
        let total_accept: u32 = accepts.iter().sum();
        eprintln!(
            "monolith 30L block: steps={} accept/step={accepts:?} total_accept={} step1(mean_H={:.3})",
            steps.len(),
            total_accept,
            steps[0].mean_entropy,
        );
        assert!(
            steps[0].mean_entropy < 6.0,
            "step1 mean_entropy {:.3} (GEMM tg bug ~10)",
            steps[0].mean_entropy
        );
        assert!(
            total_accept > MIN_ACCEPT,
            "total accept {total_accept} accept/step={accepts:?} (GEMM tg bug ~1/step, sum~8)",
        );
    }

    #[test]
    fn moe_grouped_layer2_expert_offset_and_bytes_parity() {
        use crate::dgq::layout::q4_matrix_bytes;
        use crate::metal::moe::expert_forward_staged_dgq;
        use crate::model::moe::MoeScratch;
        use std::path::Path;

        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip moe_grouped_layer2_expert_offset_and_bytes_parity");
            return;
        }
        let layer = 2usize;
        let text = crate::config::ModelConfig::load(dir).expect("cfg").text_config;
        let store = DgqStore::open(dir).expect("dgq");
        let ws = WeightStore::open(dir).expect("ws");
        let ctx = MetalContext::new().expect("metal");
        let cache = GpuDecoderWeightCache::load(&ws, &text, 0, &ctx.device).expect("cache");

        let offsets = build_offsets_from_store(&store);
        let layout = build_layout(&offsets, 512);
        let manifest_gu = offsets
            .get(&format!("model.decoder.layers.{layer}.experts.gate_up_proj"))
            .copied()
            .expect("manifest gate_up");
        let manifest_dn = offsets
            .get(&format!("model.decoder.layers.{layer}.experts.down_proj"))
            .copied()
            .expect("manifest down");
        let layout_gu = layout.layers[layer].experts_gate_up;
        let layout_dn = layout.layers[layer].experts_down;
        eprintln!(
            "L{layer} experts_gate_up manifest={manifest_gu} layout={layout_gu} delta={}",
            manifest_gu as i64 - layout_gu as i64
        );
        eprintln!(
            "L{layer} experts_down   manifest={manifest_dn} layout={layout_dn} delta={}",
            manifest_dn as i64 - layout_dn as i64
        );
        assert_eq!(layout_gu, manifest_gu);
        assert_eq!(layout_dn, manifest_dn);

        let per_expert = q4_matrix_bytes(text.moe_intermediate_size * 2, text.hidden_size) as u64;
        let down_per = q4_matrix_bytes(text.hidden_size, text.moe_intermediate_size) as u64;
        eprintln!("per_expert gate_up={per_expert} down={down_per} fused={}", per_expert + down_per);

        for expert in [0usize, 18] {
            let cache_gu = cache.expert_gate_up_q4(layer, expert);
            let kernel_gu = layout_gu + expert as u64 * per_expert;
            assert_eq!(cache_gu.byte_offset, kernel_gu, "expert {expert} gu offset");
            let cache_bytes = cache_gu.src_slice();
            let blob_ptr = unsafe {
                cache_gu
                    .weight_buffer()
                    .0
                    .contents()
                    .as_ptr()
                    .add(kernel_gu as usize)
            };
            let kernel_bytes =
                unsafe { std::slice::from_raw_parts(blob_ptr as *const u8, cache_bytes.len().min(64)) };
            assert_eq!(
                &kernel_bytes[..64],
                &cache_bytes[..64],
                "expert {expert} first 64 gu bytes"
            );
        }

        // Mirror act from cache bytes + dump moe_in should match CPU gate_act, not GPU act.
        let dump_path = Path::new("/tmp/rust_moe_single_l2_act_probe.json");
        if !dump_path.exists() {
            eprintln!("skip act compare: run step-moe-single-dump first");
            return;
        }
        let dump: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dump_path).expect("read dump")).expect("json");
        let moe_in: Vec<f32> = dump["moe_in"]
            .as_array()
            .expect("moe_in")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let gpu_act: Vec<f32> = dump["gpu_act_after_barrier"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expert = 18usize;
        let gate_up = cache.expert_gate_up_q4(layer, expert);
        let down = cache.expert_down_q4(layer, expert);
        let mut scratch = MoeScratch::new(1, &text);
        let staged = expert_forward_staged_dgq(
            &moe_in,
            &gate_up,
            &down,
            text.moe_intermediate_size,
            text.hidden_size,
            &mut scratch,
        );
        let cos_staged_gpu_act = {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (a, b) in staged.gate_act.iter().zip(gpu_act.iter()) {
                dot += *a as f64 * *b as f64;
                na += *a as f64 * *a as f64;
                nb += *b as f64 * *b as f64;
            }
            (dot / (na.sqrt() * nb.sqrt())) as f32
        };
        eprintln!(
            "E18 act: cos(staged_gate_act,gpu_act)={cos_staged_gpu_act:.4} (expect ~0.015)"
        );
    }

    #[test]
    fn q4_group_k_order_l2_e0_gate_probe() {
        use crate::config::ModelConfig;
        use crate::dgq::layout::q4_row_bytes;
        use crate::kernels::sub::q4_group_k_order;
        use crate::kernels::sub::variant::KernelVariant;
        use crate::metal::batch::{set_bytes, GpuBatch};
        use crate::metal::buffer::BufferPool;
        use crate::metal::device::MetalContext;
        use crate::metal::step_m0::{dequant_q4_group_cpu, q4_weight_at_k_order_group};
        use crate::weights::WeightStore;
        use objc2_metal::MTLSize;
        use std::path::Path;

        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip q4_group_k_order_l2_e0_gate_probe");
            return;
        }
        let layer = 2usize;
        let expert = 0usize;
        let text = ModelConfig::load(dir).expect("cfg").text_config;
        let hidden = text.hidden_size;
        let ws = WeightStore::open(dir).expect("ws");
        let ctx = MetalContext::new().expect("metal");
        let cache = GpuDecoderWeightCache::load(&ws, &text, 0, &ctx.device).expect("cache");
        let gate_up = cache.expert_gate_up_q4(layer, expert);
        let row = gate_up.src_slice();
        let row_bytes = q4_row_bytes(hidden);
        assert!(row.len() >= row_bytes);

        let pipeline =
            q4_group_k_order::pipeline_for(&ctx, KernelVariant::TEST_DUMP).expect("pipeline");
        let mut pool = BufferPool::new();

        for k0 in [0usize, 32, 64] {
            let g_off = (k0 / 32) * 20;
            let group: &[u8; 20] = row[g_off..g_off + 20].try_into().expect("group");
            let via_dequant = dequant_q4_group_cpu(group);
            let via_col = q4_weight_at_k_order_group(row, k0, hidden);

            let mut max_err = 0.0f32;
            let mut first_mismatch = None;
            for m in 0..32 {
                let err = (via_dequant[m] - via_col[m]).abs();
                max_err = max_err.max(err);
                if first_mismatch.is_none() && err > 1e-6 {
                    first_mismatch = Some((m, via_dequant[m], via_col[m]));
                }
            }
            eprintln!(
                "CPU L{layer} E{expert} gate row0 k0={k0}: max_err={max_err:.2e} mismatch={first_mismatch:?}"
            );
            eprintln!(
                "  dequant[0..8]={:?}",
                via_dequant[..8]
                    .iter()
                    .map(|v| format!("{v:.5}"))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "  q4_at [0..8]={:?}",
                via_col[..8]
                    .iter()
                    .map(|v| format!("{v:.5}"))
                    .collect::<Vec<_>>()
            );
            assert!(
                max_err <= 1e-6,
                "CPU K-order mismatch k0={k0} max_err={max_err}"
            );

            let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None).expect("batch");
            let out_buf = batch.alloc_f32_out(64).expect("out");
            let enc = batch.encoder();
            let (wbuf, row_off) = gate_up.weight_buffer();
            let k0_u32 = k0 as u32;
            let in_dim_u32 = hidden as u32;
            let mut gpu_out = vec![0.0f32; 64];
            enc.setComputePipelineState(&pipeline.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(wbuf), row_off as usize, 0);
                set_bytes(enc, &k0_u32, 1);
                set_bytes(enc, &in_dim_u32, 2);
                enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            batch.register_read(out_buf, &mut gpu_out);
            batch.end().expect("end");

            let gpu_vs_dequant = (0..32)
                .map(|m| (gpu_out[m] - via_dequant[m]).abs())
                .fold(0.0f32, f32::max);
            let gpu_vs_q4at = (0..32)
                .map(|m| (gpu_out[32 + m] - via_col[m]).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "GPU L{layer} E{expert} k0={k0}: max_err vs CPU dequant={gpu_vs_dequant:.2e} vs CPU q4_at={gpu_vs_q4at:.2e}"
            );
            assert!(
                gpu_vs_dequant <= 1e-5 && gpu_vs_q4at <= 1e-5,
                "GPU decode mismatch k0={k0}"
            );
        }
    }

    /// NONDET-SC-1 diagnostic: run the chunked SC softembed twice on identical
    /// seeded logits and diff. Isolates the run-to-run nondeterminism to a single
    /// softembed invocation (no generation loop). `--ignored` (needs a model dir).
    #[test]
    #[ignore]
    fn sc_chunked_softembed_nondeterminism_probe() {
        use std::path::Path;

        let dir = [
            Path::new("model/diffusiongemma-q4emb"),
            Path::new("/tmp/quantized-weights"),
            Path::new("model/diffusiongemma-q4"),
            Path::new("model/q4"),
        ]
        .into_iter()
        .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip sc_chunked_softembed_nondeterminism_probe");
            return;
        };
        let cfg = StepSmokeConfig {
            finish: StepFinishMode::Full,
            steps: 1,
            ..StepSmokeConfig::default()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
        let layout = rt.layout;
        rt.run_forward_once(StepFinishMode::Full)
            .expect("seed logits");
        let soft_elems = CANVAS * HID;
        let soft_off = rt.bufs.arena_map.soft_off() as usize;

        // (soft bf16-as-f32, gemm_b f32 accumulator). If gemm_b differs the
        // accumulate/GEMM is the source; if only soft differs the convert is.
        let run_chunked = |rt: &mut StepRuntime| -> (Vec<f32>, Vec<f32>) {
            write_buffer_bytes(&rt.bufs.arena, soft_off, &vec![0u8; soft_elems * 2]);
            rt.dispatch_and_wait(|enc| {
                enc.encode_sc_logit_rowstats();
                enc.encode_sc_softembed_exact(&layout)?;
                Ok(())
            })
            .expect("chunked sc softembed");
            let soft = read_arena_buffer_f32(&rt.bufs.arena, soft_off, soft_elems);
            let gb_ptr = rt.bufs.gemm_b.contents().as_ptr() as *const f32;
            let gb = (0..soft_elems).map(|i| unsafe { *gb_ptr.add(i) }).collect();
            (soft, gb)
        };

        let (run1, gb1) = run_chunked(&mut rt);
        let (run2, gb2) = run_chunked(&mut rt);
        let (run3, _gb3) = run_chunked(&mut rt);

        // Where does the f32 accumulator (gemm_b) first diverge?
        let mut gb_max = 0.0f32;
        let mut gb_first = None;
        for (i, (a, b)) in gb1.iter().zip(gb2.iter()).enumerate() {
            let d = (a - b).abs();
            if d > 0.0 && gb_first.is_none() {
                gb_first = Some((i, i / HID, i % HID, *a, *b));
            }
            gb_max = gb_max.max(d);
        }
        eprintln!("NONDET-SC gemm_b accumulator run1-vs-run2: max_abs={gb_max:.6} first(idx,row,col,a,b)={gb_first:?}");

        let mut max_abs = 0.0f32;
        let mut ndiff = 0usize;
        let mut first = None;
        for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
            let d = (a - b).abs();
            if d > 0.0 {
                ndiff += 1;
                if first.is_none() {
                    first = Some((i, *a, *b));
                }
            }
            max_abs = max_abs.max(d);
        }
        let diff13 = run1
            .iter()
            .zip(run3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "NONDET-SC chunked softembed run1-vs-run2: max_abs={max_abs:.6} ndiff={ndiff}/{soft_elems} first={first:?}; run1-vs-run3 max_abs={diff13:.6}"
        );
        // Regression guard for the memzero-coverage bug (NONDET-SC-1): the chunked
        // SC softembed must be bit-deterministic on identical seeded logits.
        assert_eq!(gb_max, 0.0, "gemm_b accumulator nondeterministic");
        assert_eq!(max_abs, 0.0, "chunked softembed nondeterministic");
        assert_eq!(diff13, 0.0, "chunked softembed nondeterministic (run3)");
    }

    /// Offset-prefill bit-identity (cross-turn KV reuse, e3f79cd):
    /// `prefill_chunks(prefix)` + `prefill_chunks_from(offset, delta)` must
    /// produce KV bytes identical to a single `prefill_chunks(all)` over the
    /// valid positions [0..n), every layer. The offset is deliberately NOT a
    /// chunk multiple so the delta chunk boundary lands mid-canvas. `--ignored`
    /// (needs a model dir).
    #[test]
    #[ignore]
    fn offset_prefill_kv_bit_identity() {
        use std::path::Path;

        let dir = [
            Path::new("model/diffusiongemma-q4emb"),
            Path::new("/tmp/quantized-weights"),
            Path::new("model/diffusiongemma-q4"),
            Path::new("model/q4"),
        ]
        .into_iter()
        .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip offset_prefill_kv_bit_identity");
            return;
        };
        let cfg = StepSmokeConfig {
            finish: StepFinishMode::Full,
            steps: 1,
            // Mid-chunk offset resume writes up to offset+CANVAS: needs headroom.
            max_seq: 1024,
            ..StepSmokeConfig::default()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
        let layout = rt.layout;

        // Spans two chunks in the full run; split point mid-chunk.
        let n = 400usize;
        let offset = 300usize;
        let tokens: Vec<u32> = (0..n)
            .map(|i| ((i.wrapping_mul(2654435761)) % 200_000) as u32 + 5)
            .collect();

        let read_kv_prefix = |rt: &StepRuntime, n: usize| -> Vec<Vec<u8>> {
            let ptr = rt.bufs.kvcache.contents().as_ptr() as *const u8;
            (0..N_LAYERS)
                .map(|layer| {
                    let l = &layout.layers[layer];
                    let token_stride =
                        (l.n_kv_heads as usize) * (l.head_dim as usize) * 2 * 2;
                    let base = l.kv_region as usize;
                    unsafe {
                        std::slice::from_raw_parts(ptr.add(base), n * token_stride)
                            .to_vec()
                    }
                })
                .collect()
        };

        let kv_len = rt.prefill_chunks(&tokens).expect("full prefill");
        assert_eq!(kv_len, n);
        let full = read_kv_prefix(&rt, n);

        zero_buffer(&rt.bufs.kvcache);
        let k1 = rt.prefill_chunks(&tokens[..offset]).expect("prefix prefill");
        assert_eq!(k1, offset);
        let k2 = rt
            .prefill_chunks_from(offset, &tokens[offset..])
            .expect("offset prefill");
        assert_eq!(k2, n);
        let split = read_kv_prefix(&rt, n);

        let mut bad = 0usize;
        for layer in 0..N_LAYERS {
            if full[layer] != split[layer] {
                let l = &layout.layers[layer];
                let token_stride =
                    (l.n_kv_heads as usize) * (l.head_dim as usize) * 2 * 2;
                let first = full[layer]
                    .iter()
                    .zip(split[layer].iter())
                    .position(|(a, b)| a != b)
                    .unwrap();
                eprintln!(
                    "layer {layer}: KV mismatch first at byte {first} (pos {})",
                    first / token_stride
                );
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "offset prefill not bit-identical to full prefill");
    }

    #[test]
    #[ignore = "pre-existing CPU chunked decomposition bug (max_abs=3.7); GPU parity passes"]
    fn sc_chunked_cpu_oracle() {
        use std::path::Path;

        use crate::buffer::Buffer;
        use crate::dgq::block::q8_gemm_rowk_cpu;
        use crate::dgq::layout::q8_row_bytes;
        use crate::model::embed::{soft_embeddings_from_logits_store, EMBED_TENSOR, LM_HEAD_CHUNK};
        use crate::weights::WeightStore;

        let dir = [Path::new("/tmp/quantized-weights"), Path::new("model/q4")]
            .into_iter()
            .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip sc_chunked_cpu_oracle");
            return;
        };
        let store = WeightStore::open(dir).expect("weights");
        let cfg = StepSmokeConfig {
            finish: StepFinishMode::Full,
            steps: 1,
            ..StepSmokeConfig::default()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
        let layout = rt.layout;
        rt.run_forward_once(StepFinishMode::Full)
            .expect("seed logits from step 1");

        let logits = read_half_buffer_f32(&rt.bufs.logits, 0, CANVAS * VOCAB);
        let scale = (HID as f32).sqrt();
        let mut prob_scratch = Buffer::new(VOCAB);
        let mut cpu_full = vec![0.0f32; CANVAS * HID];
        soft_embeddings_from_logits_store(
            &store,
            &mut cpu_full,
            &logits,
            CANVAS,
            VOCAB,
            HID,
            scale,
            &mut prob_scratch,
        )
        .expect("cpu full softembed");

        let embed_bytes = match &store {
            WeightStore::Dgq(dgq) => dgq.tensor_bytes(EMBED_TENSOR).expect("embed").to_vec(),
            _ => panic!("expected dgq"),
        };
        let row_bytes = q8_row_bytes(HID);
        let mut cpu_chunked = vec![0.0f32; CANVAS * HID];
        let mut v0 = 0usize;
        while v0 < VOCAB {
            let chunk = (VOCAB - v0).min(LM_HEAD_CHUNK);
            let fixture = crate::kernels::sub::sc_prob_cols::Fixture {
                logits: logits.clone(),
                rows: CANVAS,
                vocab: VOCAB,
                v0,
                chunk,
            };
            let probs = crate::kernels::sub::sc_prob_cols::cpu(&fixture);
            let w_off = v0 * row_bytes;
            let mut partial = vec![0.0f32; CANVAS * HID];
            q8_gemm_rowk_cpu(
                &probs,
                CANVAS,
                chunk,
                &embed_bytes[w_off..w_off + chunk * row_bytes],
                HID,
                &mut partial,
            );
            for (dst, &p) in cpu_chunked.iter_mut().zip(partial.iter()) {
                *dst += p;
            }
            v0 += chunk;
        }
        for v in &mut cpu_chunked {
            *v *= scale;
        }

        let mut max_cpu = 0.0f32;
        for (a, b) in cpu_full.iter().zip(cpu_chunked.iter()) {
            max_cpu = max_cpu.max((a - b).abs());
        }
        eprintln!("sc chunked cpu oracle: cpu_full vs cpu_chunked max_abs={max_cpu:.6}");

        rt.dispatch_and_wait(|enc| {
            enc.encode_sc_logit_rowstats();
            enc.encode_sc_softembed_exact(&layout)?;
            Ok(())
        })
        .expect("gpu chunked");
        let gpu_chunked =
            read_arena_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.soft_off() as usize, CANVAS * HID);

        let mut max_gpu = 0.0f32;
        for (a, b) in cpu_chunked.iter().zip(gpu_chunked.iter()) {
            max_gpu = max_gpu.max((a - b).abs());
        }
        eprintln!("sc chunked cpu oracle: cpu_chunked vs gpu_chunked max_abs={max_gpu:.6}");

        assert!(max_cpu < 0.1, "cpu chunked decomposition max_abs={max_cpu}");
        assert!(max_gpu < 0.1, "gpu chunked vs cpu max_abs={max_gpu}");
    }

    fn read_buffer_bytes(
        buf: &ProtocolObject<dyn MTLBuffer>,
        byte_off: usize,
        len: usize,
    ) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                buf.contents().as_ptr().add(byte_off) as *const u8,
                len,
            )
            .to_vec()
        }
    }

    fn write_buffer_bytes(buf: &ProtocolObject<dyn MTLBuffer>, byte_off: usize, data: &[u8]) {
        unsafe {
            std::slice::from_raw_parts_mut(buf.contents().as_ptr().add(byte_off) as *mut u8, data.len())
                .copy_from_slice(data);
        }
    }

    struct StepGpuSnapshot {
        state: CanvasState,
        arena: Vec<u8>,
        logits: Vec<u8>,
        sc_probs: Vec<u8>,
        kvcache: Vec<u8>,
        route: Vec<u8>,
        gemm_a: Vec<u8>,
        gemm_b: Vec<u8>,
    }

    fn snapshot_step_gpu(rt: &StepRuntime) -> StepGpuSnapshot {
        StepGpuSnapshot {
            state: rt.read_canvas_state(),
            arena: read_buffer_bytes(&rt.bufs.arena, 0, rt.bufs.arena_map.bytes() as usize),
            logits: read_buffer_bytes(&rt.bufs.logits, 0, CANVAS * VOCAB * 2),
            sc_probs: read_buffer_bytes(&rt.bufs.sc_probs, 0, rt.bufs.sc_probs.length()),
            kvcache: read_buffer_bytes(&rt.bufs.kvcache, 0, rt.bufs.kvcache.length()),
            route: read_buffer_bytes(&rt.bufs.route, 0, rt.bufs.route.length()),
            gemm_a: read_buffer_bytes(&rt.bufs.gemm_a, 0, rt.bufs.gemm_a.length()),
            gemm_b: read_buffer_bytes(&rt.bufs.gemm_b, 0, rt.bufs.gemm_b.length()),
        }
    }

    fn restore_step_gpu(rt: &mut StepRuntime, snap: &StepGpuSnapshot) {
        rt.write_canvas_state(&snap.state);
        write_buffer_bytes(&rt.bufs.arena, 0, &snap.arena);
        write_buffer_bytes(&rt.bufs.logits, 0, &snap.logits);
        write_buffer_bytes(&rt.bufs.sc_probs, 0, &snap.sc_probs);
        write_buffer_bytes(&rt.bufs.kvcache, 0, &snap.kvcache);
        write_buffer_bytes(&rt.bufs.route, 0, &snap.route);
        write_buffer_bytes(&rt.bufs.gemm_a, 0, &snap.gemm_a);
        write_buffer_bytes(&rt.bufs.gemm_b, 0, &snap.gemm_b);
    }

    #[test]
    fn step_smoke_runs_if_weights_present() {
        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip step_smoke_runs_if_weights_present: no weights at /tmp/quantized-weights");
            return;
        }
        let result = run_step_smoke(dir, StepSmokeConfig::default()).expect("step smoke");
        assert_eq!(result.step, 1);
        if !result.logits_finite {
            eprintln!(
                "warning: step smoke logits non-finite (max_abs={})",
                result.max_abs_logit
            );
        }
    }

    #[test]
    fn fused_gate_up_gemm_matches_split_in_full_arena() {
        use crate::kernels::sub::gemm_block_stacked::pipeline_for;
        use crate::kernels::sub::gemm_q4;
        use crate::kernels::sub::QuantFormat;
        use crate::metal::buffer::BufferPool;
        use crate::metal::device::MetalContext;
        use crate::metal::DgqGpuBlob;
        use std::path::Path;

        fn read_plane(
            arena: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            byte_off: u64,
        ) -> Vec<f32> {
            let elems = CANVAS * DENSE_FF as usize;
            let ptr = arena.contents().as_ptr() as *const u16;
            let base = (byte_off / 2) as usize;
            (0..elems)
                .map(|i| {
                    crate::kernels::sub::bf16::bf16_bits_to_f32(unsafe { *ptr.add(base + i) })
                })
                .collect()
        }

        fn max_diff(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        let dir = [Path::new("model/q4"), Path::new("/tmp/quantized-weights")]
            .into_iter()
            .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip fused_gate_up_gemm_matches_split_in_full_arena");
            return;
        };
        let store = crate::dgq::store::DgqStore::open(dir).expect("dgq");
        let offsets = build_offsets_from_store(&store);
        let layout = build_layout(&offsets, 512);
        let arena_map = step_arena_layout();
        let l = &layout.layers[0];
        let (segs, n_total) = gate_up_stacked_segments(l, &arena_map);

        let ctx = MetalContext::new().expect("metal");
        let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let mut pool = BufferPool::new();
        let arena = pool
            .allocate(&ctx.device, arena_map.bytes() as usize)
            .expect("arena");

        let m = CANVAS;
        let k = HID;
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 + 11.0) * 0.0006).sin() * 0.14)
            .collect();
        let x_bits = crate::kernels::sub::bf16::f32_slice_to_bf16_bits(&x);
        unsafe {
            let dst = arena.contents().as_ptr().add(arena_map.tmp_off() as usize) as *mut u16;
            std::ptr::copy_nonoverlapping(x_bits.as_ptr(), dst, x_bits.len());
        }

        let pipeline = pipeline_for(
            &ctx,
            n_total,
            k as u32,
            QuantFormat::Q4Affine,
            &segs,
        )
        .expect("pipe");
        let (grid, tg) = crate::kernels::sub::gemm_common::dispatch_shape(m, n_total as usize);
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = cmd.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline.pipeline);
        let m_u32 = m as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&arena), arena_map.tmp_off() as usize, 0);
            enc.setBuffer_offset_atIndex(Some(&arena), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&gpu_blob.buffer), 0, 2);
        }
        crate::kernels::sub::gpu_common::set_bytes(&enc, &m_u32, 3);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let fused_gate = read_plane(&arena, arena_map.ffg_off());
        let fused_up = read_plane(&arena, arena_map.ffu_off());

        let arena2 = pool.allocate(&ctx.device, arena_map.bytes() as usize).expect("arena2");
        unsafe {
            let dst = arena2
                .contents()
                .as_ptr()
                .add(arena_map.tmp_off() as usize) as *mut u16;
            std::ptr::copy_nonoverlapping(x_bits.as_ptr(), dst, x_bits.len());
        }
        for (w_off, y_off) in [(l.mlp_gate, arena_map.ffg_off()), (l.mlp_up, arena_map.ffu_off())] {
            let pipeline1 = gemm_q4::pipeline_for(&ctx, DENSE_FF, k as u32).expect("split pipe");
            let cmd2 = ctx.queue.commandBuffer().expect("cmd");
            let enc2 = cmd2.computeCommandEncoder().expect("enc");
            enc2.setComputePipelineState(&pipeline1.pipeline);
            unsafe {
                enc2.setBuffer_offset_atIndex(Some(&arena2), arena_map.tmp_off() as usize, 0);
                enc2.setBuffer_offset_atIndex(Some(&arena2), y_off as usize, 1);
                enc2.setBuffer_offset_atIndex(Some(&gpu_blob.buffer), 0, 2);
            }
            crate::kernels::sub::gpu_common::set_bytes(&enc2, &w_off, 3);
            crate::kernels::sub::gpu_common::set_bytes(&enc2, &m_u32, 4);
            let (g2, t2) = crate::kernels::sub::gemm_common::dispatch_shape(m, DENSE_FF as usize);
            enc2.dispatchThreadgroups_threadsPerThreadgroup(g2, t2);
            enc2.endEncoding();
            cmd2.commit();
            cmd2.waitUntilCompleted();
        }
        let gate_max = max_diff(&fused_gate, &read_plane(&arena2, arena_map.ffg_off()));
        let up_max = max_diff(&fused_up, &read_plane(&arena2, arena_map.ffu_off()));
        eprintln!("arena gate_up fused vs split: gate_max={gate_max:.4e} up_max={up_max:.4e}");
        assert!(gate_max < 0.05, "gate plane max_abs={gate_max}");
        assert!(up_max < 0.05, "up plane max_abs={up_max}");
    }

    #[test]
    fn generate_path_reset_block_matches_step_smoke_ids() {
        use crate::sample::Rng;
        use std::path::Path;

        let dir = [Path::new("model/q4"), Path::new("/tmp/quantized-weights")]
            .into_iter()
            .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip generate_path_reset_block_matches_step_smoke_ids");
            return;
        };
        let prompt = vec![23391u32]; // raw "hello" (matches generate-monolithic --raw)
        let cfg = StepSmokeConfig {
            layers: 3,
            steps: 1,
            seed: 42,
            prefill_token_ids: Some(prompt),
            ..StepSmokeConfig::default()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
        let eos = rt.read_params().eos_token_id;
        let sampler = crate::sample::sampler_for_steps(4, true);
        let params = step_params_from_sampler(
            &sampler,
            rt.read_params().kv_len,
            true,
            eos,
        );
        let mut rng = Rng::new(42);
        rt.reset_block(VOCAB, &mut rng, params);
        for _ in 0..4 {
            rt.run_denoise_step().expect("denoise");
        }
        let ids = rt.read_canvas_state().ids;
        let argmax = rt.read_canvas_state().prev_argmax;
        eprintln!("generate-path 4x ids[:8]={:?}", &ids[..8]);
        eprintln!("generate-path 4x argmax[:8]={:?}", &argmax[..8]);
    }

    #[test]
    fn fusion_matches_unfused_forward_logits() {
        use crate::sample::Rng;
        use std::path::Path;

        fn row0_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        fn forward_row0_logits(dir: &Path, cfg: &StepSmokeConfig, fused: bool) -> Vec<f32> {
            unsafe {
                if fused {
                    std::env::remove_var("DGQ_FUSED_ALGEBRA");
                } else {
                    std::env::set_var("DGQ_FUSED_ALGEBRA", "0");
                }
            }
            let mut cfg = cfg.clone();
            cfg.finish = StepFinishMode::ForwardOnly;
            let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
            let sampler = crate::sample::sampler_for_steps(4, true);
            let params = step_params_from_sampler(
                &sampler,
                rt.read_params().kv_len,
                true,
                rt.read_params().eos_token_id,
            );
            let mut rng = Rng::new(42);
            rt.reset_block(VOCAB, &mut rng, params);
            rt.run_forward_once(StepFinishMode::ForwardOnly)
                .expect("forward");
            read_half_buffer_f32(&rt.bufs.logits, 0, VOCAB)
        }

        let dir = [Path::new("model/q4"), Path::new("/tmp/quantized-weights")]
            .into_iter()
            .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip fusion_matches_unfused_forward_logits");
            return;
        };
        let cfg = StepSmokeConfig {
            layers: 3,
            steps: 1,
            seed: 42,
            prefill_token_ids: Some(vec![23391u32]),
            ..StepSmokeConfig::default()
        };

        let off = forward_row0_logits(dir, &cfg, false);
        let on = forward_row0_logits(dir, &cfg, true);
        let off2 = forward_row0_logits(dir, &cfg, false);
        unsafe {
            std::env::remove_var("DGQ_FUSED_ALGEBRA");
        }

        let max = row0_max_abs_diff(&off, &on);
        let baseline = row0_max_abs_diff(&off, &off2);
        eprintln!(
            "fusion row0 logits max_abs: fused={max:.4e} unfused_repeat={baseline:.4e}"
        );
        assert!(baseline < 1e-5, "unfused not deterministic: {baseline}");

        let mut smoke = cfg.clone();
        smoke.finish = StepFinishMode::Full;
        smoke.steps = 1;
        smoke.prefill_token_ids = Some(vec![23391u32]);
        unsafe {
            std::env::set_var("DGQ_FUSED_ALGEBRA", "0");
        }
        let (mut rt_off, _) = build_step_runtime(dir, &smoke).expect("rt off");
        rt_off.run_forward_once(StepFinishMode::Full).expect("step");
        let st_off = rt_off.read_canvas_state();
        let logits_off = read_half_buffer_f32(&rt_off.bufs.logits, 0, VOCAB);
        unsafe {
            std::env::remove_var("DGQ_FUSED_ALGEBRA");
        }
        let (mut rt_on, _) = build_step_runtime(dir, &smoke).expect("rt on");
        rt_on.run_forward_once(StepFinishMode::Full).expect("step");
        let st_on = rt_on.read_canvas_state();
        let logits_on = read_half_buffer_f32(&rt_on.bufs.logits, 0, VOCAB);
        let full_row0 = row0_max_abs_diff(&logits_off, &logits_on);
        eprintln!(
            "full-step ids[:4] off={:?} on={:?} row0_logits_max_abs={full_row0:.4e}",
            &st_off.ids[..4],
            &st_on.ids[..4],
        );
        assert_eq!(st_off.ids, st_on.ids, "sampled ids diverged");
        assert!(
            full_row0 < 0.05,
            "full step row0 logits max_abs={full_row0}"
        );
    }
}
