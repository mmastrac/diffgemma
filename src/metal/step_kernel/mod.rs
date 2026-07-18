//! Monolithic diffgemma denoise-step kernel (the production per-step engine).
//!
//! Per-step dispatch outline (pipelines specialized by `IS_FULL_LAYER` +
//! `GEMM_N`/`GEMM_K` function constants; shared device math lives in
//! `shaders/include/`, entry kernels in `shaders/kernels/`):
//!   SC preamble: logit_rowstats → sc softembed (sparse/chunked) → embed_gather
//!     → residual → rmsnorm.
//!   per layer: rmsnorm → QKV gemm → qk_rope_kv → attention (mma2/mma_full)
//!     → o_proj gemm → residual → rmsnorm → MLP gate/up/glu/down gemm
//!     → moe_router + bucket_count/fill → moe_grouped/block_sparse experts
//!     → moe_scatter_weighted → residual.
//!   finish: final rmsnorm → lm_head gemm → softcap → sample_rowstats
//!     → sample_commit → sample_apply → sample_write.
//! The buffer/struct ABI is authoritative in Rust (`abi.rs`, `arena_layout.rs`).

use crate::Error;
use crate::config::{ModelConfig, TextConfig};
use crate::dgq::DgqStore;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::dgq_gpu::DgqGpuBlob;
use crate::metal::moe::experts_forward_dgq_cpu;
use crate::metal::step_quant::{BlockGroupedJob, MoeExecutionStyle, StepBlockProfile};
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::moe::MoeScratch;
use crate::sample::{Rng, SamplerConfig, initialize_canvas};
use crate::shaders::QuantFormat;
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

mod step_schedule;

pub(crate) mod arena_liveness;

// Probe / capture / bench harnesses (CLI step-debug subcommands + step-smoke
// gate). Split out of this file for size (backlog item 4). A child module, so
// it sees this module's private items via ancestry; re-exported flat so the
// existing `step_kernel::<fn>` paths keep resolving.
mod diag_bench;
mod diag_moe;
mod diag_probe;
mod exec;
pub use diag_bench::*;
pub use diag_moe::*;
pub use diag_probe::*;
pub use exec::*;

pub const HID: usize = 2816;
pub const VOCAB: usize = 262144;
pub const CANVAS: usize = 256;
/// Batched prefill super-chunk: PREFILL_SUBS causal 256-token sub-chunks run
/// as ONE forward (attention/rope per sub-chunk, everything else — QKV,
/// o_proj, dense FFN, router, MoE — at M = PREFILL_M). Bit-identical to
/// sequential chunks (all batched stages are row-independent); the win is MoE
/// expert-weight streaming amortized over 4x the tokens + GEMM M-efficiency.
pub const PREFILL_SUBS: usize = 4;
pub const PREFILL_M: usize = CANVAS * PREFILL_SUBS;
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

use crate::metal::arena_layout::{ArenaLayout, ArenaLayoutParams, build_arena_layout};

// All DGQ_* env flags live in crate::flags; re-exported here so existing
// `step_kernel::<flag>()` call sites keep working.
pub use crate::flags::{
    attn_mma_enabled, attn_mma_full_enabled, attn_window_enabled, denoise_parity_log_enabled,
    denoise_parity_log_positions, denoiser_argmax_enabled, final_entropy_log_enabled,
    freeze_enabled, fused_gate_up_enabled, fused_qkv_enabled, logits_finite_check_enabled,
    logits_finite_sample_count, moe_fuse_gather_enabled, partial_lm_head_enabled,
    prefill_batch_enabled, router_gemm_enabled, sc_sparse_enabled, should_fast_prefill,
    step_text_log_enabled, trace_entropy_enabled,
};

pub const MAX_ATTN_Q_COLS: usize = 8192;
pub const MAX_ATTN_KV_COLS: usize = 2048;

pub fn step_arena_params() -> ArenaLayoutParams {
    ArenaLayoutParams {
        // Planes hold PREFILL_M rows so the batched prefill super-chunk (4x256
        // causal sub-chunks in ONE forward) fits; denoise dispatches only ever
        // touch rows [0..CANVAS) of each plane. Arena ~25.6 -> ~102 MiB.
        canvas: PREFILL_M,
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

/// Which stage-2/3 pair rides on the shared E17 QK decomposition
/// (`encode_attn_decomp`): E17's dense softmax+PV or E20's top-k pair.
#[derive(Clone, Copy)]
enum AttnDecompKind {
    Gemm,
    TopK,
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
    /// First position qk_rope_kv must NOT write to the KV cache. Prefill sets
    /// this to the prompt end so the zero-PADDED tail-chunk rows don't store
    /// pad K/V: on linear (full-attention) regions pad writes land past
    /// kv_len and are causally masked — harmless — but on the sliding RING
    /// they wrap onto (pad_pos & ring_mask) and CLOBBER the oldest live
    /// window positions. That clobber (prompts > ring size, i.e. >2k tokens)
    /// destroyed the window start on all 25 sliding layers and broke
    /// long-prompt comprehension entirely (task #64). u32::MAX = no
    /// suppression (denoise canvas writes must always land).
    pub kv_write_end: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CanvasState {
    /// 1024 = batched-prefill super-chunk M (denoise + sampler use [0..CANVAS)).
    pub ids: [u32; PREFILL_M],
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
    pub weight: [[u16; TOP_K]; PREFILL_M],
    pub expert: [[u32; TOP_K]; PREFILL_M],
    pub count: [u32; N_EXPERTS],
    pub row_start: [u32; N_EXPERTS + 1],
    pub num_slots: u32,
    pub num_active_experts: u32,
    pub active_expert: [u32; N_EXPERTS],
    pub token_list: [u32; PREFILL_M * TOP_K],
    pub slot_list: [u32; PREFILL_M * TOP_K],
    pub token_slot: [[u32; TOP_K]; PREFILL_M],
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
// Bounded by n_active_experts + max_slots/32 = 128 + PREFILL_M*8/32.
pub(super) const MOE_SLOTS: u32 = (PREFILL_M * TOP_K) as u32;

pub const MOE_MAX_BLOCKS: usize = N_EXPERTS + PREFILL_M * TOP_K / 32;

/// Fill `token_slot[tok][kk]` from flat `token_list` / `slot_list` after bucketing.
pub fn fill_token_slot(route: &mut RouteScratch) {
    route.token_slot = [[0; TOP_K]; PREFILL_M];
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
        // +8/round-to-8: the direct-load MMA kernels read whole 8-key tiles;
        // the softmax masks the tail keys but the reads must stay in-buffer
        // (layer 29 — the last region — is a full layer).
        (max_seq + 8).next_multiple_of(8)
    } else if crate::flags::kv_ring_uncapped_enabled() {
        // DIAGNOSTIC (task #64 follow-up): linear sliding storage — no ring
        // wrap ever. Isolates ring-READ defects from everything else at the
        // cost of full-length sliding KV (fine to ~8k).
        max_seq.next_power_of_two()
    } else {
        max_seq
            .next_power_of_two()
            .min(crate::flags::kv_ring_slots())
    }
}

pub fn build_layout(offsets: &HashMap<String, u64>, max_seq: usize) -> ModelLayout {
    // KV storage format is a per-session decision keyed off max_seq (plus the
    // DGQ_KV_Q8 override) — every sizing/pack/kernel site derives it the same
    // way so the layout stays coherent.
    let kv_fmt = crate::flags::kv_format(max_seq);
    let g = |n: &str| {
        *offsets
            .get(n)
            .unwrap_or_else(|| panic!("missing tensor {n}"))
    };
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
        kv_off += crate::metal::step_kv::kv_region_bytes(nkv, hd, slots, kv_fmt);
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
    v.div_ceil(g)
}
/// Fused Q‖K(‖V): one `GEMM_N` = q_n+k_n(+k_n); outputs land in native-width planes.
pub(crate) fn qkv_stacked_segments(
    l: &LayerOffsets,
    arena: &ArenaLayout,
) -> (Vec<crate::shaders::gemm_block_stacked::GemmStackedSeg>, u32) {
    use crate::shaders::gemm_block_stacked::GemmStackedSeg;
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
) -> ([crate::shaders::gemm_block_stacked::GemmStackedSeg; 2], u32) {
    use crate::shaders::gemm_block_stacked::GemmStackedSeg;
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
    // Batched prefill (M = PREFILL_M): gemm_a holds the moein f32 conversion
    // (PREFILL_M x HID) and the swiglu activations (MOE_SLOTS x MOE_FF);
    // gemm_b holds the gathered expert A (MOE_SLOTS x HID at offset 0) plus
    // the gate_up activations (MOE_SLOTS x 2*MOE_FF at moe_w_byte_off_gu).
    let moe_slots = PREFILL_M * TOP_K;
    max_mk = max_mk.max(PREFILL_M * HID).max(moe_slots * MOE_FF as usize);
    max_nk = max_nk.max(moe_slots * (HID + 2 * MOE_FF as usize));
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
    /// Tunable Raw (bf16-weight) dense pipelines, keyed (n,k); VOCAB shape uses
    /// the logits (K_OUT_BF16) variant. Sole dense path for the bf16 profile.
    gemm_tunable_raw: HashMap<(u32, u32), ComputePipeline>,
    /// Tunable q8 dense pipelines, keyed (n,k); VOCAB = logits variant.
    gemm_tunable_q8: HashMap<(u32, u32), ComputePipeline>,
    /// Tunable q4 / nvfp4 dense pipelines (attention/dense-FFN weights on a
    /// block-quantized checkpoint), keyed (n,k).
    gemm_tunable_q4: HashMap<(u32, u32), ComputePipeline>,
    gemm_tunable_nvfp4: HashMap<(u32, u32), ComputePipeline>,
    /// Tunable block-sparse MoE pipelines (DGQ_GEMM_TUNABLE, q4/q6 experts),
    /// keyed (n, k, gather, format as u32).
    gemm_tunable_sparse: HashMap<(u32, u32, bool, u32), ComputePipeline>,
    /// Wide-block (weight-stationary) tunable sparse pipelines for batched
    /// prefill (DGQ_MOE_PREFILL_BM != 32); same keying. Empty when disabled.
    gemm_tunable_sparse_wide: HashMap<(u32, u32, bool, u32), ComputePipeline>,
    /// TUNE_BM the wide sparse pipelines were compiled with (the block
    /// height moe_bucket_fill phase 1 must build during batched prefill).
    sparse_wide_bm: u32,
    #[allow(dead_code)]
    gemm_q8_rowk: HashMap<(u32, u32), ComputePipeline>,
    #[allow(dead_code)]
    gemm_q8_rowk_xfp16: HashMap<(u32, u32), ComputePipeline>,
    /// f32-accumulate variant of `gemm_q8_rowk` for chunked SC softembed (avoids per-chunk bf16 round).
    gemm_q8_rowk_acc_f32: HashMap<(u32, u32), ComputePipeline>,
    /// f32→bf16 convert with scale, for chunked SC softembed accumulator → half arena.
    f32_to_half_scale: ComputePipeline,
    qk_rope_kv: ComputePipeline,
    /// E14 prefill variants (FC30, DGQ_PREFILL_KV_F32): rope also writes the
    /// f32 side ring; mma2 reads K/V from it (all-float MMA). None = flag off.
    qk_rope_kv_side: Option<ComputePipeline>,
    attention_mma2_side: Option<ComputePipeline>,
    attention_mma_full_side: Option<ComputePipeline>,
    kv_f32_side_hydrate: Option<ComputePipeline>,
    attention: ComputePipeline,
    /// GQA-grouped MMA attention for sliding layers (`DGQ_ATTN_MMA`); scalar `attention` is the fallback/oracle.
    attention_mma2: ComputePipeline,
    /// MMA attention for full/global layers (`DGQ_ATTN_MMA_FULL`, register-O); scalar `attention` is the fallback/oracle.
    attention_mma_full: ComputePipeline,
    /// E17 GEMM-attention for full-layer PREFILL (`DGQ_GEMM_ATTN`, default on):
    /// [qk, softmax, pv]. Prefill runs the full decomp; denoise reuses the qk
    /// stage through E20 top-k (`DGQ_ATTN_TOPK_DECODE`).
    attn_gemm: Option<[ComputePipeline; 3]>,
    /// E17b f32-side-KV variant (FC30): reads the f32 side ring, all-float MMA.
    /// None unless DGQ_GEMM_ATTN && DGQ_PREFILL_KV_F32. Preferred over `attn_gemm`
    /// when present (matches attention_mma_full_side precision).
    attn_gemm_side: Option<[ComputePipeline; 3]>,
    /// E20 top-k sparse attention for full layers, BOTH phases (`DGQ_ATTN_TOPK`
    /// prefill, `DGQ_ATTN_TOPK_DECODE` denoise — both default on):
    /// [qk (reused from E17), topk_softmax, topk_pv]. Quality-gated
    /// (non-bit-identical).
    attn_topk: Option<[ComputePipeline; 3]>,
    /// E20 f32-side-KV variant (FC30). None unless DGQ_ATTN_TOPK &&
    /// DGQ_PREFILL_KV_F32.
    attn_topk_side: Option<[ComputePipeline; 3]>,
    /// E18 flash for sliding-layer PREFILL (`DGQ_FLASH_PREFILL`): hd=256,
    /// window-aware + ring-aware. None unless the flag is set.
    attn_flash_sliding: Option<ComputePipeline>,
    residual: ComputePipeline,
    glu: ComputePipeline,
    router: ComputePipeline,
    /// Top-k tail over precomputed logits (DGQ_ROUTER_GEMM).
    router_topk: ComputePipeline,
    bucket_count: ComputePipeline,
    bucket_fill: ComputePipeline,
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
    fn new(
        ctx: &MetalContext,
        variant: crate::shaders::variant::KernelVariant,
        fmt: crate::shaders::kv_quant::KvFormat,
    ) -> Result<Self, Error> {
        let mut gemm_tunable_raw = HashMap::new();
        let mut gemm_tunable_q8 = HashMap::new();
        let mut gemm_tunable_q4 = HashMap::new();
        let mut gemm_tunable_nvfp4 = HashMap::new();
        let mut gemm_tunable_sparse = HashMap::new();
        let mut gemm_tunable_sparse_wide = HashMap::new();
        let sparse_wide_bm = crate::flags::moe_prefill_block_m();
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
            // Raw (bf16-weight) dense: lm_head logits (n=VOCAB) forces bf16
            // output (range); others follow K_ACT_F16 for their activation out.
            let raw = if n == VOCAB as u32 {
                crate::shaders::gemm_tunable::pipeline_for_logits(
                    ctx,
                    n,
                    k,
                    crate::shaders::QuantFormat::Raw,
                )?
            } else {
                crate::shaders::gemm_tunable::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::shaders::QuantFormat::Raw,
                )?
            };
            gemm_tunable_raw.insert((n, k), raw);
            // q4 / nvfp4 dense (block-quant attention/dense-FFN checkpoints).
            // Never lm_head (that path is bf16 or q8), so no logits variant.
            gemm_tunable_q4.insert(
                (n, k),
                crate::shaders::gemm_tunable::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::shaders::QuantFormat::Q4Affine,
                )?,
            );
            gemm_tunable_nvfp4.insert(
                (n, k),
                crate::shaders::gemm_tunable::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::shaders::QuantFormat::NvFp4,
                )?,
            );
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
            // q8 dense: lm_head logits (VOCAB) forces bf16 output; others follow
            // the arena activation dtype.
            let q8 = if (n, k) == (VOCAB as u32, HID as u32) {
                crate::shaders::gemm_tunable::pipeline_for_logits(
                    ctx,
                    n,
                    k,
                    crate::shaders::QuantFormat::Q8,
                )?
            } else {
                crate::shaders::gemm_tunable::pipeline_for(
                    ctx,
                    n,
                    k,
                    crate::shaders::QuantFormat::Q8,
                )?
            };
            gemm_tunable_q8.insert((n, k), q8);
        }
        for &(n, k) in &[
            (HID as u32, VOCAB as u32),
            (HID as u32, crate::model::embed::LM_HEAD_CHUNK as u32),
        ] {
            gemm_q8_rowk.insert((n, k), crate::shaders::gemm_rowk::pipeline_for(ctx, n, k)?);
            gemm_q8_rowk_xfp16.insert(
                (n, k),
                crate::shaders::gemm_rowk::pipeline_for_fp16_input(ctx, n, k)?,
            );
        }
        // Unified rowk f32-accumulate SC-softembed GEMM (one shader; weight format
        // = K_QUANT_FORMAT: Raw bf16 embed or Q8 embed). x is fp16 sc_probs.
        const ROWK_ACC_SHADER: &str = crate::shaders::gemm_rowk::SHADER;
        let mut gemm_q8_rowk_acc_f32 = HashMap::new();
        {
            for &(n, k) in &[(HID as u32, crate::model::embed::LM_HEAD_CHUNK as u32)] {
                gemm_q8_rowk_acc_f32.insert(
                    (n, k),
                    ctx.compile_gemm_subkernel(
                        ROWK_ACC_SHADER,
                        "gemm_rowk",
                        n,
                        k,
                        false,
                        crate::shaders::QuantFormat::Q8 as u32,
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
                        "gemm_rowk",
                        n,
                        k,
                        false,
                        crate::shaders::QuantFormat::Raw as u32, // bf16 embed -> Raw branch
                        true,                                    // sc_probs is fp16
                    )?,
                );
            }
        }
        // f32 -> arena bf16 with scale (convert_scale src_f32=true, dst_f32=false).
        let f32_to_half_scale =
            crate::shaders::convert_scale::pipeline_for_fmt(ctx, variant, true, false)?;
        // Block-sparse MoE experts: tunable is the sole path (q4/q6/nvfp4).
        // gate_up gathers bf16 moein rows (fused gather); down's A is the
        // swiglu output. Wide variants = weight-stationary prefill height
        // (TUNE_BM=sparse_wide_bm; the batched-prefill block list is built at
        // this height).
        for &(n, k) in &[(MOE_FF * 2, HID as u32), (HID as u32, MOE_FF)] {
            for fmt in [
                crate::shaders::QuantFormat::Q4Affine,
                crate::shaders::QuantFormat::Q6,
                crate::shaders::QuantFormat::NvFp4,
            ] {
                gemm_tunable_sparse.insert(
                    (n, k, false, fmt as u32),
                    crate::shaders::gemm_tunable::pipeline_for_sparse(ctx, n, k, false, fmt)?,
                );
                if moe_fuse_gather_enabled() && (n, k) == (MOE_FF * 2, HID as u32) {
                    gemm_tunable_sparse.insert(
                        (n, k, true, fmt as u32),
                        crate::shaders::gemm_tunable::pipeline_for_sparse(ctx, n, k, true, fmt)?,
                    );
                }
                if sparse_wide_bm != 32 && prefill_batch_enabled() {
                    gemm_tunable_sparse_wide.insert(
                        (n, k, false, fmt as u32),
                        crate::shaders::gemm_tunable::pipeline_for_sparse_bm(
                            ctx,
                            n,
                            k,
                            false,
                            fmt,
                            sparse_wide_bm as usize,
                        )?,
                    );
                    if moe_fuse_gather_enabled() && (n, k) == (MOE_FF * 2, HID as u32) {
                        gemm_tunable_sparse_wide.insert(
                            (n, k, true, fmt as u32),
                            crate::shaders::gemm_tunable::pipeline_for_sparse_bm(
                                ctx,
                                n,
                                k,
                                true,
                                fmt,
                                sparse_wide_bm as usize,
                            )?,
                        );
                    }
                }
            }
        }
        let prod = variant;
        let dump = crate::shaders::variant::KernelVariant::TEST_DUMP;
        Ok(Self {
            memzero: crate::shaders::memzero_bytes::pipeline_for(ctx, prod)?,
            rmsnorm: crate::shaders::rms_norm_rows_tiled::pipeline_for(
                ctx,
                crate::shaders::rms_norm_rows_tiled::TiledVariant::HALF_IN,
                prod,
            )?,
            rmsnorm_f32: crate::shaders::rms_norm_rows_tiled::pipeline_for(
                ctx,
                crate::shaders::rms_norm_rows_tiled::TiledVariant::F32_IN,
                prod,
            )?,
            gemm_tunable_raw,
            gemm_tunable_q8,
            gemm_tunable_q4,
            gemm_tunable_nvfp4,
            gemm_tunable_sparse,
            gemm_tunable_sparse_wide,
            sparse_wide_bm,
            gemm_q8_rowk,
            gemm_q8_rowk_xfp16,
            gemm_q8_rowk_acc_f32,
            gemm_bf16_rowk_acc_f32,
            f32_to_half_scale,
            qk_rope_kv: crate::shaders::qk_rope_kv::pipeline_for_kv(ctx, prod, fmt)?,
            qk_rope_kv_side: if crate::flags::prefill_kv_f32_enabled() {
                Some(crate::shaders::qk_rope_kv::pipeline_for_kv_side(
                    ctx, prod, fmt,
                )?)
            } else {
                None
            },
            attention_mma2_side: if crate::flags::prefill_kv_f32_enabled() {
                Some(crate::shaders::attention::pipeline_mma2_for_kv_side(
                    ctx, prod, fmt,
                )?)
            } else {
                None
            },
            attention_mma_full_side: if crate::flags::prefill_kv_f32_enabled() {
                Some(crate::shaders::attention::pipeline_mma_full_for_kv_side(
                    ctx, prod, fmt,
                )?)
            } else {
                None
            },
            kv_f32_side_hydrate: if crate::flags::prefill_kv_f32_enabled() {
                Some(crate::shaders::kv_f32_side_hydrate::pipeline_for_kv(
                    ctx, prod, fmt,
                )?)
            } else {
                None
            },
            attention: crate::shaders::attention::pipeline_for_kv(ctx, prod, fmt)?,
            attention_mma2: crate::shaders::attention::pipeline_mma2_for_kv(ctx, prod, fmt)?,
            attention_mma_full: crate::shaders::attention::pipeline_mma_full_for_kv(
                ctx, prod, fmt,
            )?,
            attn_gemm: if crate::flags::gemm_attn_enabled() {
                let (qk_bm, qk_bn, pv_bm, pv_bn, sm_tpg) = crate::flags::gemm_attn_tile();
                let cfg = crate::shaders::attention_gemm::TuneCfg {
                    hc: crate::flags::gemm_attn_head_chunk(),
                    qk_bm,
                    qk_bn,
                    pv_bm,
                    pv_bn,
                    sm_tpg,
                };
                Some(crate::shaders::attention_gemm::pipelines_cfg(
                    ctx, prod, cfg, false,
                )?)
            } else {
                None
            },
            attn_gemm_side: if crate::flags::gemm_attn_enabled()
                && crate::flags::prefill_kv_f32_enabled()
            {
                let (qk_bm, qk_bn, pv_bm, pv_bn, sm_tpg) = crate::flags::gemm_attn_tile();
                let cfg = crate::shaders::attention_gemm::TuneCfg {
                    hc: crate::flags::gemm_attn_head_chunk(),
                    qk_bm,
                    qk_bn,
                    pv_bm,
                    pv_bn,
                    sm_tpg,
                };
                Some(crate::shaders::attention_gemm::pipelines_cfg(
                    ctx, prod, cfg, true,
                )?)
            } else {
                None
            },
            attn_topk: if crate::flags::attn_topk_enabled()
                || crate::flags::attn_topk_decode_enabled()
            {
                Some(crate::shaders::attention_topk::pipelines(
                    ctx,
                    prod,
                    false,
                    crate::flags::attn_topk_k_pad(),
                )?)
            } else {
                None
            },
            attn_topk_side: if crate::flags::attn_topk_enabled()
                && crate::flags::prefill_kv_f32_enabled()
            {
                Some(crate::shaders::attention_topk::pipelines(
                    ctx,
                    prod,
                    true,
                    crate::flags::attn_topk_k_pad(),
                )?)
            } else {
                None
            },
            attn_flash_sliding: {
                let (on, bq, bk) = crate::flags::flash_prefill();
                if on {
                    // Sliding layers only (hd=256). Full hd=512 stays on E17.
                    Some(crate::shaders::attention_flash::pipeline_flash(
                        ctx, prod, bq, bk, 256,
                    )?)
                } else {
                    None
                }
            },
            residual: crate::shaders::residual_half::pipeline_for(ctx, prod)?,
            glu: crate::shaders::swiglu::pipeline_for(
                ctx,
                crate::shaders::SwigluSplitVariant::MONOLITH_GLU,
                prod,
            )?,
            router: crate::shaders::moe_router::pipeline_for(ctx, prod)?,
            router_topk: ctx.compile_subkernel(
                crate::shaders::moe_router::TOPK_SHADER,
                crate::shaders::moe_router::TOPK_ENTRY,
                prod,
            )?,
            bucket_count: crate::shaders::moe_bucket_count::pipeline_for(ctx, prod)?,
            bucket_fill: crate::shaders::moe_bucket_fill::pipeline_for(ctx, prod)?,
            gather_rows_bf16_to_f32: crate::shaders::gather_rows::pipeline_for_fmt(
                ctx, prod, false, true,
            )?,
            gelu_swiglu_gate_up: crate::shaders::swiglu::pipeline_for_moe(ctx, prod)?,
            moe_scatter_weighted: crate::shaders::moe_scatter_weighted::pipeline_for(ctx, prod)?,
            moe_grouped: crate::shaders::moe_grouped::pipeline_for(ctx, prod)?,
            moe_grouped_nvfp4: crate::shaders::moe_grouped_nvfp4::pipeline_for(ctx, prod)?,
            moe_grouped_dump: crate::shaders::moe_grouped::pipeline_for(ctx, dump)?,
            embed_gather: crate::shaders::embed_gather::pipeline_for(ctx, prod)?,
            embed_gather_bf16: crate::shaders::embed_gather::pipeline_for_fmt(ctx, prod, true)?,
            logit_rowstats: crate::shaders::logit_rowstats::pipeline_for(ctx, prod)?,
            sc_prob_cols: crate::shaders::sc_prob_cols::pipeline_for(ctx, prod)?,
            // arena bf16 in-place scale (convert_scale src_f32=false, dst_f32=false).
            half_scale: crate::shaders::convert_scale::pipeline_for_fmt(ctx, prod, false, false)?,
            softcap: crate::shaders::softcap_half::pipeline_for(ctx, prod)?,
            sample_rowstats: crate::shaders::sample_rowstats::pipeline_for(ctx, prod)?,
            sample_commit: crate::shaders::sample_commit::pipeline_for(ctx, prod)?,
            sample_apply: crate::shaders::sample_apply::pipeline_for(ctx, prod)?,
            sample_write: crate::shaders::sample_write::pipeline_for(ctx, prod)?,
            compact_active_rows: ctx.compile_kernel(
                crate::shaders::compact_active_rows::SHADER,
                crate::shaders::compact_active_rows::ENTRY,
            )?,
            gather_rows_bf16: crate::shaders::gather_rows::pipeline_for_fmt(
                ctx, prod, false, false,
            )?,
            scatter_logits_rows: ctx.compile_kernel(
                crate::shaders::scatter_logits_rows::SHADER,
                crate::shaders::scatter_logits_rows::ENTRY,
            )?,
            sc_sparse_select: ctx.compile_kernel(
                crate::shaders::sc_sparse_select::SHADER,
                crate::shaders::sc_sparse_select::ENTRY,
            )?,
            sc_sparse_gather: ctx.compile_kernel(
                crate::shaders::sc_sparse_gather::SHADER,
                crate::shaders::sc_sparse_gather::ENTRY,
            )?,
        })
    }

    /// Tunable Raw (bf16-weight) dense pipeline; VOCAB shape = logits variant.
    fn dense_raw(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_tunable_raw
            .get(&(n, k))
            .ok_or(Error::Format("missing tunable raw pipeline"))
    }

    /// Tunable q8 dense pipeline; VOCAB shape = logits variant.
    fn dense_q8(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_tunable_q8
            .get(&(n, k))
            .ok_or(Error::Format("missing tunable q8 pipeline"))
    }

    /// Tunable q4 dense pipeline (block-quant attention/dense-FFN checkpoint).
    fn dense_q4(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_tunable_q4
            .get(&(n, k))
            .ok_or(Error::Format("missing tunable q4 dense pipeline"))
    }

    /// Tunable nvfp4 dense pipeline.
    fn dense_nvfp4(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_tunable_nvfp4
            .get(&(n, k))
            .ok_or(Error::Gpu("missing tunable nvfp4 dense pipeline"))
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

    /// Wide-block (weight-stationary prefill) tunable sparse pipeline if
    /// built (DGQ_MOE_PREFILL_BM != 32). Only valid against a block list
    /// built at `sparse_wide_bm`.
    fn sparse_tunable_wide_fmt(
        &self,
        format: QuantFormat,
        n: u32,
        k: u32,
        gather: bool,
    ) -> Option<&ComputePipeline> {
        self.gemm_tunable_sparse_wide
            .get(&(n, k, gather, format as u32))
    }

    #[allow(dead_code)]
    fn q8_rowk_xfp16(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q8_rowk_xfp16
            .get(&(n, k))
            .ok_or(Error::Gpu("missing q8 rowk fp16-input pipeline"))
    }

    #[allow(dead_code)]
    fn q8_rowk(&self, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        self.gemm_q8_rowk
            .get(&(n, k))
            .ok_or(Error::Gpu("missing q8 rowk pipeline"))
    }

    fn block_gemm(&self, format: QuantFormat, n: u32, k: u32) -> Result<&ComputePipeline, Error> {
        match format {
            QuantFormat::NvFp4 => self.dense_nvfp4(n, k),
            _ => self.dense_q4(n, k),
        }
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
    /// Keepalive for `DGQ_KV_MMAP`: the file-backed mapping `kvcache` wraps
    /// no-copy. Declared right after `kvcache` so it drops *after* it (the Metal
    /// buffer must be released before the mapping is torn down). `None` = the
    /// default anonymous `StorageModeShared` allocation.
    _kv_mmap: Option<KvMmapBacking>,
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
    /// Online-softmax state (f32 m/l + unnormalized O per head x canvas row)
    /// persisted between sequential kv-block dispatches of attention_mma_full.
    attn_state: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// E17 GEMM-attention prefill scratch (`DGQ_GEMM_ATTN`): score matrix S
    /// (f32), probs P (f16), row denoms lrow (f32) — sized
    /// [n_q_heads][CANVAS][n_pad(max_seq)]. None unless the flag is set.
    attn_gemm_s: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_gemm_p: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_gemm_lrow: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// E20 top-k sparse attention scratch (`DGQ_ATTN_TOPK`): compressed
    /// probs P [HC][CANVAS][K_PAD] (f32), top-k indices Idx [HC][CANVAS][K_PAD]
    /// (u32), row denoms lrow [HC][CANVAS] (f32). The S plane is SHARED with
    /// `attn_gemm_s` (identical layout; only one path is active at a time).
    /// None unless the flag is set.
    attn_topk_p: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_topk_idx: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_topk_lrow: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    attn_topk_pat: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// E14 (DGQ_PREFILL_KV_F32): f32 side K/V ring for the sliding layers —
    /// written by the prefill rope, read by the prefill attention, so chunk
    /// boundaries never round K/V to f16. Same slot = pos & ring_mask
    /// addressing as the f16 ring; per-layer byte offsets in
    /// `kv_f32_side_offs` (u64::MAX for full layers). None when the flag is
    /// off.
    kv_f32_side: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    kv_f32_side_offs: [u64; N_LAYERS],
    /// PREFILL_SUBS StepParams slots for the per-sub-chunk rope/attention
    /// dispatches of a batched prefill super-chunk (kv_len differs per sub;
    /// bufs.params is read at execution time so it can't vary within one
    /// encoder).
    params_sub: Retained<ProtocolObject<dyn MTLBuffer>>,
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
    /// N-tile width of the wide-block (E1, block_m != 32) tunable sparse
    /// pipelines — they pin BN=64. bucket_fill picks this for slots 4/5
    /// when dims.block_m != 32.
    tunable_wide_n_tile: u32,
}

// Slots 0/1: per-expert grouped (gate_up/down), height = num_active_experts.
// Slots 2/3: block-sparse (gate_up/down), height = num_blocks.
// Slots 4/5: tunable block-sparse (gate_up/down, BN-wide N-tiles).
const MOE_GROUPED_INDIRECT_BYTES: usize = 6 * 3 * std::mem::size_of::<u32>();

fn moe_grouped_grid_info() -> MoeGroupedGridInfo {
    MoeGroupedGridInfo {
        gate_n: MOE_FF * 2,
        hid: HID as u32,
        n_tile: crate::shaders::gemm_common::n_tile() as u32,
        tpg: crate::shaders::gemm_common::THREADS_PER_TG as u32,
        tunable_n_tile: crate::flags::moe_sparse_bn() as u32,
        tunable_wide_n_tile: 64,
    }
}

fn grouped_expert_blob_bytes_per_expert(format: crate::shaders::QuantFormat) -> u64 {
    use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes};
    let hidden = HID;
    let moe_ff = MOE_FF as usize;
    match format {
        crate::shaders::QuantFormat::NvFp4 => {
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
    let hidden = HID;
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

pub fn init_canvas_state(seed: u64, vocab: usize) -> CanvasState {
    let mut rng = Rng::new(seed);
    init_canvas_state_from_rng(vocab, &mut rng)
}

pub fn init_canvas_state_from_rng(vocab: usize, rng: &mut Rng) -> CanvasState {
    let ids_vec = initialize_canvas(CANVAS, vocab, rng);
    let mut ids = [0u32; PREFILL_M];
    ids[..CANVAS].copy_from_slice(&ids_vec);
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
        kv_write_end: u32::MAX,
    }
}

fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::Gpu("Metal buffer alloc failed"))
}

/// Keepalive + auto-cleanup for a `DGQ_KV_MMAP` file-backed KV buffer. Holds the
/// mapping the Metal buffer wraps no-copy; removes the backing file on drop.
struct KvMmapBacking {
    #[allow(dead_code)]
    mmap: memmap2::MmapMut,
    path: std::path::PathBuf,
}

impl Drop for KvMmapBacking {
    fn drop(&mut self) {
        // Best-effort: unlink the backing file. The mapping (`mmap`, dropped
        // after this) stays valid through the unlink on macOS.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Allocate the session KV buffer. Default: anonymous `StorageModeShared`. With
/// `DGQ_KV_MMAP`: a `MAP_SHARED` temp-file mapping wrapped no-copy as the Metal
/// buffer (same mechanism as the read-only `.dgq` weight blob), so under memory
/// pressure dirty KV pages evict to that file rather than to anonymous swap.
fn alloc_kv_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<
    (
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Option<KvMmapBacking>,
    ),
    Error,
> {
    if !crate::flags::kv_mmap() {
        return Ok((alloc_buffer(device, bytes)?, None));
    }
    // `newBufferWithBytesNoCopy` requires a page-aligned pointer (mmap gives
    // this) and length. Round up to the Apple-Silicon 16 KiB page.
    const PAGE: usize = 16 * 1024;
    let len = bytes.div_ceil(PAGE) * PAGE;
    let path = crate::flags::kv_mmap_dir().join(format!("dgq-kv-{}.bin", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;
    file.set_len(len as u64)?;
    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };
    let ptr = std::ptr::NonNull::new(mmap.as_mut_ptr() as *mut std::ffi::c_void)
        .ok_or(Error::Runtime("kv mmap null pointer"))?;
    let buffer = unsafe {
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .ok_or(Error::Gpu("kv mmap buffer alloc failed"))?
    };
    eprintln!(
        "kv cache: mmap-backed ({:.2} GiB) at {}",
        len as f64 / 1_073_741_824.0,
        path.display()
    );
    Ok((buffer, Some(KvMmapBacking { mmap, path })))
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
fn write_single_expert_route(buf: &ProtocolObject<dyn MTLBuffer>, position: usize, expert_id: u32) {
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
    let bytes =
        unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, buf.length()) };
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
    _eos_token_id: u32,
) {
    use crate::sample::{
        EarlyStopKind, FILLER_TOKEN_ID, PAD_TOKEN_ID, decode_early_stop_flag, is_active_token,
    };
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
        eprintln!("  pos={pos:3} {tag:6} id={id:6} argmax={am:6} ent={ent:8.4} accept={acc}",);
    }
    let mut high_ent_active = Vec::new();
    for pos in 0..CANVAS {
        if is_active_token(state.ids[pos]) && state.entropy[pos] > 0.1 {
            high_ent_active.push((
                pos,
                state.entropy[pos],
                state.ids[pos],
                state.prev_argmax[pos],
            ));
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
    use crate::shaders::bf16;
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
    let cpu_mask = crate::sample::accept_mask_from_entropies(&state.entropy, params.entropy_bound);
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
    use crate::shaders::bf16;
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

/// f16 twin of `half_buffer_stats` (planes written by the K_ARENA_F16 set).
fn f16_buffer_stats(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
    sample: usize,
) -> (bool, f32) {
    use crate::shaders::f16::f16_bits_to_f32;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    let mut max_abs = 0.0f32;
    let mut finite = true;
    let n = sample.min(elems);
    let stride = (elems / n.max(1)).max(1);
    let mut i = 0usize;
    while i < elems {
        let v = f16_bits_to_f32(unsafe { *ptr.add(i) });
        if !v.is_finite() {
            finite = false;
        }
        max_abs = max_abs.max(v.abs());
        i += stride;
        if i / stride >= n {
            break;
        }
    }
    (finite, max_abs)
}

fn half_buffer_stats(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
    sample: usize,
) -> (bool, f32) {
    use crate::shaders::bf16;
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

fn arena_hidden_stats(
    arena: &ProtocolObject<dyn MTLBuffer>,
    layout: &ArenaLayout,
) -> (bool, f32, usize) {
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

#[cfg(all(test, target_os = "macos"))]
mod tests;
