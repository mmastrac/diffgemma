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
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
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
pub const N_LAYERS: usize = 30;
pub const N_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
pub const DENSE_FF: u32 = 2112;
pub const MOE_FF: u32 = 704;
/// Two act rows (after-barrier + down-read) plus kernel input probe metadata.
pub const MOE_ACT_PROBE_ACT_FLOATS: usize = (MOE_FF * 2) as usize;
pub const MOE_ACT_PROBE_META_FLOATS: usize = 36; // tok,slot,e,w,x[8],row0[8], down_o[8], moe_out_tok_row[8]

use crate::metal::arena_layout::{build_arena_layout, ArenaLayout, ArenaLayoutParams};

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
    pub _pad: u32,
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
    pub argmax_stable: u32,
    pub argmax_changed: u32,
    pub mean_entropy: f32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RouteScratch {
    pub weight: [[u16; TOP_K]; CANVAS],
    pub expert: [[u32; TOP_K]; CANVAS],
    pub count: [u32; N_EXPERTS],
    pub row_start: [u32; N_EXPERTS + 1],
    pub num_slots: u32,
    pub pad_route: u32,
    pub token_list: [u32; CANVAS * TOP_K],
    pub slot_list: [u32; CANVAS * TOP_K],
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
        _ => q4_matrix_bytes(n, k) as u64,
    };
    base + expert as u64 * stride
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
            _pad: 0,
        };
        kv_off += (max_seq as u64) * (nkv as u64) * (hd as u64) * 2 * 2;
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
    (
        max_mk * std::mem::size_of::<f32>(),
        max_nk * std::mem::size_of::<f32>(),
    )
}

/// SC softembed backend: materialized probs + `k_gemm_q8_rowk` (default). Opt out with `DGQ_SC_GEMM=0`.
pub fn step_use_sc_gemm_default() -> bool {
    true
}

fn step_use_sc_gemm_from_env() -> bool {
    match std::env::var("DGQ_SC_GEMM") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => step_use_sc_gemm_default(),
    }
}

/// Opt-in logits NaN guard on generate hot path (`DGQ_CHECK_LOGITS=1`).
pub fn logits_finite_check_enabled() -> bool {
    match std::env::var("DGQ_CHECK_LOGITS") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

fn logits_finite_sample_count() -> usize {
    match std::env::var("DGQ_CHECK_LOGITS_SAMPLES") {
        Ok(v) => v.parse().unwrap_or(4096),
        Err(_) => 4096,
    }
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
    gemm_nvfp4: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8_rowk: HashMap<(u32, u32), ComputePipeline>,
    qk_rope_kv: ComputePipeline,
    attention: ComputePipeline,
    residual: ComputePipeline,
    glu: ComputePipeline,
    router: ComputePipeline,
    bucket_count: ComputePipeline,
    bucket_fill: ComputePipeline,
    q4_block_grouped: HashMap<(u32, u32), ComputePipeline>,
    nvfp4_block_grouped: HashMap<(u32, u32), ComputePipeline>,
    gather_rows: ComputePipeline,
    gelu_swiglu_gate_up: ComputePipeline,
    moe_scatter_weighted: ComputePipeline,
    moe_grouped: ComputePipeline,
    moe_grouped_nvfp4: ComputePipeline,
    /// Q4 grouped MoE with K_DUMP_STAGE=1 (debug capture only — never in forward/generate).
    moe_grouped_dump: ComputePipeline,
    embed_gather: ComputePipeline,
    logit_rowstats: ComputePipeline,
    sc_probs: ComputePipeline,
    sc_softembed: ComputePipeline,
    half_scale: ComputePipeline,
    softcap: ComputePipeline,
    sample_rowstats: ComputePipeline,
    sample_commit: ComputePipeline,
    sample_apply: ComputePipeline,
    sample_write: ComputePipeline,
}

impl StepPipelines {
    fn new(ctx: &MetalContext, variant: crate::kernels::sub::variant::KernelVariant) -> Result<Self, Error> {
        let mut gemm_q4 = HashMap::new();
        let mut gemm_nvfp4 = HashMap::new();
        let mut gemm_q8 = HashMap::new();
        let mut gemm_q8_rowk = HashMap::new();
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
        ] {
            gemm_q4.insert(
                (n, k),
                crate::kernels::sub::gemm_q4::pipeline_for(ctx, n, k)?,
            );
            gemm_nvfp4.insert(
                (n, k),
                crate::kernels::sub::gemm_nvfp4::pipeline_for(ctx, n, k)?,
            );
        }
        for &(n, k) in &[
            (DENSE_FF, HID as u32),
            (2816, DENSE_FF),
            (VOCAB as u32, HID as u32),
        ] {
            gemm_q8.insert(
                (n, k),
                crate::kernels::sub::gemm_q8::pipeline_for(ctx, n, k)?,
            );
        }
        for &(n, k) in &[(HID as u32, VOCAB as u32)] {
            gemm_q8_rowk.insert(
                (n, k),
                crate::kernels::sub::gemm_q8_rowk::pipeline_for(ctx, n, k)?,
            );
        }
        let mut q4_block_grouped = HashMap::new();
        let mut nvfp4_block_grouped = HashMap::new();
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
            gemm_nvfp4,
            gemm_q8,
            gemm_q8_rowk,
            qk_rope_kv: crate::kernels::sub::qk_rope_kv::pipeline_for(ctx, prod)?,
            attention: crate::kernels::sub::attention::pipeline_for(ctx, prod)?,
            residual: crate::kernels::sub::residual_half::pipeline_for(ctx, prod)?,
            glu: crate::kernels::sub::swiglu::pipeline_for(
                ctx,
                crate::kernels::sub::SwigluSplitVariant::MONOLITH_GLU,
                prod,
            )?,
            router: crate::kernels::sub::moe_router::pipeline_for(ctx, prod)?,
            bucket_count: crate::kernels::sub::moe_bucket_count::pipeline_for(ctx, prod)?,
            bucket_fill: crate::kernels::sub::moe_bucket_fill::pipeline_for(ctx, prod)?,
            q4_block_grouped,
            nvfp4_block_grouped,
            gather_rows: crate::kernels::sub::gather_rows::pipeline_for(ctx, prod)?,
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
            logit_rowstats: crate::kernels::sub::logit_rowstats::pipeline_for(ctx, prod)?,
            sc_probs: crate::kernels::sub::sc_probs::pipeline_for(ctx, prod)?,
            sc_softembed: crate::kernels::sub::sc_softembed::pipeline_for(ctx, prod)?,
            half_scale: crate::kernels::sub::half_scale::pipeline_for(ctx, prod)?,
            softcap: crate::kernels::sub::softcap_half::pipeline_for(ctx, prod)?,
            sample_rowstats: crate::kernels::sub::sample_rowstats::pipeline_for(ctx, prod)?,
            sample_commit: crate::kernels::sub::sample_commit::pipeline_for(ctx, prod)?,
            sample_apply: crate::kernels::sub::sample_apply::pipeline_for(ctx, prod)?,
            sample_write: crate::kernels::sub::sample_write::pipeline_for(ctx, prod)?,
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

    fn moe_scalar(&self, format: QuantFormat) -> &ComputePipeline {
        match format {
            QuantFormat::NvFp4 => &self.moe_grouped_nvfp4,
            _ => &self.moe_grouped,
        }
    }
}

pub(crate) struct StepBuffers {
    blob: Retained<ProtocolObject<dyn MTLBuffer>>,
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
    /// Scratch plane byte offsets (host-built, mirrored on GPU as b8).
    pub(crate) arena_map: ArenaLayout,
    /// GPU copy of `arena_map` (b8); bound when kernels need device-side plane table.
    #[allow(dead_code)]
    arena_layout_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

struct StepEnc<'a> {
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    ps: &'a StepPipelines,
    bufs: &'a StepBuffers,
    block_profile: StepBlockProfile,
    use_sc_gemm: bool,
    tensor_offsets: &'a HashMap<String, u64>,
    recorder: Option<&'a mut crate::metal::step_icb::IcbRecorder>,
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
    layer_moe_block_jobs_impl(l, format, None)
}

fn layer_moe_block_jobs_impl(
    l: &LayerOffsets,
    format: QuantFormat,
    manifest: Option<(usize, &HashMap<String, u64>)>,
) -> ([BlockGroupedJob; N_EXPERTS], [BlockGroupedJob; N_EXPERTS]) {
    use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes};
    let hidden = HID as usize;
    let moe_ff = MOE_FF as usize;
    let (gu_stride, dn_stride, gu_gpr, dn_gpr) = match format {
        QuantFormat::NvFp4 => (
            nvfp4_matrix_bytes(moe_ff * 2, hidden) as u64,
            nvfp4_matrix_bytes(hidden, moe_ff) as u64,
            (hidden as u32).div_ceil(16),
            (moe_ff as u32).div_ceil(16),
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
    }
    (gate, down)
}

impl StepEnc<'_> {
    fn sink_set_pipeline(&mut self, ps: &ComputePipeline) {
        if let Some(r) = self.recorder.as_deref_mut() {
            r.set_pipeline(ps);
        } else {
            self.enc.setComputePipelineState(&ps.pipeline);
        }
    }

    fn sink_set_buffer(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        if let Some(r) = self.recorder.as_deref_mut() {
            r.set_buffer(buf, offset, index);
        } else {
            unsafe {
                self.enc.setBuffer_offset_atIndex(Some(buf), offset, index);
            }
        }
    }

    fn sink_set_bytes<T: Copy>(&mut self, val: &T, index: usize) {
        if let Some(r) = self.recorder.as_deref_mut() {
            r.set_bytes(val, index);
        } else {
            crate::metal::batch::set_bytes(&self.enc, val, index);
        }
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
        if let Some(r) = self.recorder.as_deref_mut() {
            r.dispatch_threadgroups(grid, tg);
        } else {
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
    }

    fn bind_blob(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.blob, 0, idx);
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

    /// Softcap logits (matches sampler.metal ranged dispatch pattern).
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

    fn gemm_q4_fused(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.block_gemm(self.block_profile.format, n, k)?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, 32),
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

    fn gemm_q4(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        self.gemm_q4_fused(x_off, y_off, w_off, m, n, k)
    }

    fn memzero_bytes(&mut self, byte_off: u64, nbytes: u64) {
        self.sink_set_pipeline(&self.ps.memzero);
        self.sink_set_buffer(&self.bufs.arena, byte_off as usize, 0);
        let count = div_up(nbytes as usize, 16);
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
        let ps = self.ps.q8(n, k)?;
        self.sink_set_pipeline(ps);
            self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
            self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
            self.bind_blob(2);
            self.sink_set_bytes( &w_off, 3);
            self.sink_set_bytes( &m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, 32),
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

    fn gemm_q8_logits(
        &mut self,
        x_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.q8(n, k)?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.bind_logits(1);
        self.bind_blob(2);
        self.sink_set_bytes( &w_off, 3);
        self.sink_set_bytes( &m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, 32),
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
    fn gemm_q8_probs(
        &mut self,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.q8_rowk(n, k)?;
        self.sink_set_pipeline(ps);
        self.bind_sc_probs(0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes( &w_off, 3);
        self.sink_set_bytes( &m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, 32),
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

    fn scale_half_arena(&mut self, y_off: u64, elems: usize, scale: f32) {
        self.sink_set_pipeline(&self.ps.half_scale);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 0);
        self.sink_set_bytes(&(elems as u32), 1);
        self.sink_set_bytes(&scale, 2);
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
        self.encode_sc_softembed_path(layout, self.use_sc_gemm)
    }

    fn encode_sc_softembed_path(
        &mut self,
        layout: &ModelLayout,
        use_gemm: bool,
    ) -> Result<(), Error> {
        if use_gemm {
            self.sink_set_pipeline(&self.ps.sc_probs);
            self.bind_logits(0);
            self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
            self.bind_sc_probs(2);
            let dims = [CANVAS as u32, VOCAB as u32];
            self.sink_set_bytes(&dims, 3);
            self.bind_debug_status(4);
            let (grid, tg) = crate::kernels::sub::sc_probs::dispatch_shape(CANVAS);
            self.sink_dispatch(grid, tg);

            self.gemm_q8_probs(self.arena().soft_off(), layout.embed, CANVAS as u32, HID as u32, VOCAB as u32)?;
            self.scale_half_arena(self.arena().soft_off(), CANVAS * HID as usize, (HID as f32).sqrt());
        } else {
            use crate::dgq::embed_row::EMBED_SCALE;

            self.sink_set_pipeline(&self.ps.sc_softembed);
            self.bind_logits(0);
            self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
            self.bind_blob(2);
            self.sink_set_buffer(&self.bufs.arena, self.arena().soft_off() as usize, 3);
            let first_step: u32 = 0;
            self.sink_set_bytes(&layout.embed, 4);
            self.sink_set_bytes(&first_step, 5);
            let dims = [HID as u32, CANVAS as u32, VOCAB as u32];
            self.sink_set_bytes(&dims, 6);
            self.sink_set_bytes(&EMBED_SCALE, 7);
            self.bind_debug_status(8);
            let (grid, tg) =
                crate::kernels::sub::sc_softembed::dispatch_shape(HID, CANVAS);
            self.sink_dispatch(grid, tg);
        }
        Ok(())
    }

    fn residual(&mut self, a_off: u64, b_off: u64, y_off: u64, scal_off: u64, elems: usize) {
        self.sink_set_pipeline(&self.ps.residual);
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
        let l = &layout.layers[layer];
        let q_n = if l.is_full != 0 { 8192 } else { 4096 };
        let k_n = if l.is_full != 0 { 1024 } else { 2048 };
        let qk_y = (16 + 2 * l.n_kv_heads) as usize;
        let layer_off = layer_byte_offset(layer);

        self.rmsnorm(self.arena().hidden_off(), self.arena().tmp_off(), l.input_ln, HID as u32, CANVAS);
        self.gemm_q4(self.arena().tmp_off(), self.arena().attnq_off(), l.q_proj, CANVAS as u32, q_n, HID as u32)?;
        self.gemm_q4(self.arena().tmp_off(), self.arena().attnk_off(), l.k_proj, CANVAS as u32, k_n, HID as u32)?;
        if l.v_proj != 0 {
            self.gemm_q4(self.arena().tmp_off(), self.arena().attnv_off(), l.v_proj, CANVAS as u32, k_n, HID as u32)?;
        }

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
        };
        self.sink_set_bytes(&attn_dims, 7);
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

        self.sink_set_pipeline(&self.ps.attention);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attnq_off() as usize, 0);
        self.bind_kvcache(1);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attno_off() as usize, 2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_params(4);
        let attn_dims = crate::kernels::sub::qk_rope_kv::AttnDims {
            canvas: CANVAS as u32,
            n_q_heads: STEP_NQ_HEADS as u32,
        };
        self.sink_set_bytes(&attn_dims, 5);
        let grid = MTLSize {
            width: CANVAS,
            height: 16,
            depth: 1,
        };
        let tg = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

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
        let l = &layout.layers[layer];
        let o_k = if l.is_full != 0 { 8192 } else { 4096 };
        self.gemm_q4(self.arena().attno_off(), self.arena().tmp_off(), l.o_proj, CANVAS as u32, HID as u32, o_k)?;
        self.rmsnorm(self.arena().tmp_off(), self.arena().tmp_off(), l.post_attn_ln, HID as u32, CANVAS);
        self.residual(self.arena().hidden_off(), self.arena().tmp_off(), self.arena().stream_off(), 0, CANVAS * HID);
        Ok(())
    }

    fn encode_layer_dense_ffn(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        self.rmsnorm(self.arena().stream_off(), self.arena().tmp_off(), l.pre_ff_ln, HID as u32, CANVAS);
        self.gemm_q4(self.arena().tmp_off(), self.arena().ffg_off(), l.mlp_gate, CANVAS as u32, DENSE_FF, HID as u32)?;
        self.gemm_q4(self.arena().tmp_off(), self.arena().ffu_off(), l.mlp_up, CANVAS as u32, DENSE_FF, HID as u32)?;
        self.glu(self.arena().ffg_off(), self.arena().ffu_off(), self.arena().ffg_off(), CANVAS * DENSE_FF as usize);
        self.gemm_q4(self.arena().ffg_off(), self.arena().dense_off(), l.mlp_down, CANVAS as u32, HID as u32, DENSE_FF)?;
        self.rmsnorm(self.arena().dense_off(), self.arena().dense_off(), l.post_ff_ln_1, HID as u32, CANVAS);
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
    ) -> Result<(), Error> {
        let grouped_ps = self
            .ps
            .block_grouped(self.block_profile.format, n, k)?;
        let row_start_off = std::mem::offset_of!(RouteScratch, row_start);
        self.sink_set_pipeline(grouped_ps);
        let a_buf = if a_on_gemm_a {
            &self.bufs.gemm_a
        } else {
            &self.bufs.gemm_b
        };
            self.sink_set_buffer(a_buf, buf_a_off, 0);
            self.bind_blob(1);
            self.sink_set_buffer(&self.bufs.gemm_b, buf_c_off, 2);
            self.sink_set_bytes(jobs, 3);
            self.sink_set_buffer(&self.bufs.route, row_start_off, 4);
        let num_jobs = N_EXPERTS as u32;
        self.sink_set_bytes(&num_jobs, 5);
        let grid = MTLSize {
            width: crate::kernels::sub::gemm_common::div_up(n as usize, 32),
            height: N_EXPERTS,
            depth: 1,
        };
        let tg = MTLSize {
            width: crate::kernels::sub::gemm_common::THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Batched MoE: gather → grouped block GEMM (gate/up, down) → swiglu → weighted scatter.
    fn encode_layer_moe_batched(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_moe_batched_gather()?;
        self.encode_moe_batched_gate_up(layer, layout)?;
        self.encode_moe_batched_swiglu()?;
        self.encode_moe_batched_down(layer, layout)?;
        self.encode_moe_batched_scatter()?;
        Ok(())
    }

    fn encode_moe_batched_gather(&mut self) -> Result<(), Error> {
        let token_list_off = std::mem::offset_of!(RouteScratch, token_list);
        self.half_to_f32_buf(self.arena().moein_off(), CANVAS * HID);
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
        );
        self.dispatch_block_linear_grouped(
            false,
            moe_w_byte_off_a(),
            moe_w_byte_off_gu(),
            &gate_jobs,
            MOE_SLOTS,
            HID as u32,
            MOE_FF * 2,
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
        let (_, down_jobs) = layer_moe_block_jobs_impl(l, self.block_profile.format, None);
        self.dispatch_block_linear_grouped(
            true,
            0,
            moe_w_byte_off_a(),
            &down_jobs,
            MOE_SLOTS,
            MOE_FF,
            HID as u32,
        )
    }

    fn encode_moe_batched_scatter(&mut self) -> Result<(), Error> {
        let scatter_count = (MOE_SLOTS as usize) * HID;
        self.dispatch_1d_ranged(&self.ps.moe_scatter_weighted, scatter_count, 256, |this, base, _chunk| {
            this.sink_set_buffer(&this.bufs.gemm_b, moe_w_byte_off_a(), 0);
            this.sink_set_buffer(&this.bufs.arena, this.arena().moeout_off() as usize, 1);
            this.bind_route(2);
            this.sink_set_bytes(&(HID as u32), 3);
            this.sink_set_bytes(&base, 4);
        });
        Ok(())
    }

    fn encode_layer_moe_scalar(
        &mut self,
        layer: usize,
        _layout: &ModelLayout,
    ) -> Result<(), Error> {
        let layer_off = layer_byte_offset(layer);
        self.sink_set_pipeline(self.ps.moe_scalar(self.block_profile.format));
        self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moeout_off() as usize, 1);
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
        Ok(())
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

    /// Input layernorm + Q/K/V projections only (stops before qk_rope_kv).
    fn encode_layer_qkv_gemm(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let q_n = if l.is_full != 0 { 8192 } else { 4096 };
        let k_n = if l.is_full != 0 { 1024 } else { 2048 };

        self.rmsnorm(self.arena().hidden_off(), self.arena().tmp_off(), l.input_ln, HID as u32, CANVAS);
        self.gemm_q4(self.arena().tmp_off(), self.arena().attnq_off(), l.q_proj, CANVAS as u32, q_n, HID as u32)?;
        self.gemm_q4(self.arena().tmp_off(), self.arena().attnk_off(), l.k_proj, CANVAS as u32, k_n, HID as u32)?;
        if l.v_proj != 0 {
            self.gemm_q4(self.arena().tmp_off(), self.arena().attnv_off(), l.v_proj, CANVAS as u32, k_n, HID as u32)?;
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
        _layout: &ModelLayout,
    ) -> Result<(), Error> {
        let layer_off = layer_byte_offset(layer);
        self.sink_set_pipeline(&self.ps.attention);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attnq_off() as usize, 0);
        self.bind_kvcache(1);
        self.sink_set_buffer(&self.bufs.arena, self.arena().attno_off() as usize, 2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_params(4);
        let attn_dims = crate::kernels::sub::qk_rope_kv::AttnDims {
            canvas: CANVAS as u32,
            n_q_heads: STEP_NQ_HEADS as u32,
        };
        self.sink_set_bytes(&attn_dims, 5);
        self.bind_debug_status(6);
        let grid = MTLSize {
            width: CANVAS,
            height: 16,
            depth: 1,
        };
        let tg = MTLSize {
            width: 64,
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
            StepStage::MoeBatchedGather => self.encode_moe_batched_gather(),
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
                self.gemm_q8_logits(
                    self.arena().tmp_off(),
                    layout.embed,
                    CANVAS as u32,
                    VOCAB as u32,
                    HID as u32,
                )
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
        for stage in step_schedule::build_preamble(first_step) {
            self.exec_stage(stage, 0, layout, finish)?;
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

        self.sink_set_pipeline(&self.ps.embed_gather);
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

    fn encode_step_finish(
        &mut self,
        layout: &ModelLayout,
        mode: StepFinishMode,
    ) -> Result<(), Error> {
        self.rmsnorm(self.arena().hidden_off(), self.arena().tmp_off(), layout.final_norm, HID as u32, CANVAS);
        self.gemm_q8_logits(
            self.arena().tmp_off(),
            layout.embed,
            CANVAS as u32,
            VOCAB as u32,
            HID as u32,
        )?;
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

        self.sink_set_pipeline(&self.ps.sample_commit);
        self.bind_state(0);
        self.bind_params(1);
        self.sink_set_bytes(&canvas, 2);
        self.sink_set_bytes(&pad, 3);
        self.sink_set_bytes(&filler, 4);
        self.bind_debug_status(5);
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
        argmax_stable: 0,
        argmax_changed: 0,
        mean_entropy: 0.0,
        _pad2: 0,
    }
}

pub fn step_params_from_sampler(
    sampler: &SamplerConfig,
    kv_len: u32,
    no_early_stop: bool,
) -> StepParams {
    let conf_threshold = if no_early_stop {
        f32::MAX
    } else {
        sampler.confidence_threshold
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

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32) * 2f32.powi(-24);
        return if sign == 1 { -val } else { val };
    }
    if exp == 0x1f {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }
    let val = f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13));
    val
}

pub fn trace_entropy_enabled() -> bool {
    match std::env::var("DGQ_TRACE_ENTROPY") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full"),
        Err(_) => false,
    }
}

pub fn denoise_parity_log_enabled() -> bool {
    match std::env::var("DGQ_LOG_DENOISE") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

fn denoise_parity_log_positions() -> usize {
    match std::env::var("DGQ_LOG_DENOISE_POS") {
        Ok(v) => v.parse().unwrap_or(8),
        Err(_) => 8,
    }
}

/// HF linear schedule: temperature at start of denoise step `steps_done` (0 = first step).
pub fn scheduled_temperature(steps_done: u32, params: &StepParams) -> f32 {
    let max = params.max_steps.max(1) as f32;
    let cur = params.max_steps.saturating_sub(steps_done) as f32;
    params.t_min + (params.t_max - params.t_min) * (cur / max)
}

fn read_logit_f32(logits: &ProtocolObject<dyn MTLBuffer>, row: usize, col: u32) -> f32 {
    let byte_off = (row * VOCAB + col as usize) * 2;
    let ptr = unsafe { logits.contents().as_ptr().add(byte_off) as *const u16 };
    f16_bits_to_f32(unsafe { *ptr })
}

/// Log GPU vs CPU accept masks and per-position entropy/argmax (for MLX/HF parity iteration).
pub fn log_denoise_parity_step(
    label: &str,
    state: &CanvasState,
    params: &StepParams,
    logits: &ProtocolObject<dyn MTLBuffer>,
) {
    use crate::sample::accept_mask_from_entropies;
    let cpu_mask = accept_mask_from_entropies(&state.entropy, params.entropy_bound);
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
    let ptr = buf.contents().as_ptr() as *const u16;
    let mut bad = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..elems {
        unsafe {
            let v = f16_bits_to_f32(*ptr.add(i));
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
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    let mut max_abs = 0.0f32;
    let mut finite = true;
    let mut non_finite = 0usize;
    let n = sample.min(elems);
    let stride = (elems / n.max(1)).max(1);
    unsafe {
        let mut i = 0usize;
        while i < elems {
            let v = f16_bits_to_f32(*ptr.add(i));
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
    let (finite, max_abs) =
        half_buffer_stats(arena, layout.hidden_off() as usize, CANVAS * HID, CANVAS * HID);
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
    use_sc_gemm: bool,
    layout: ModelLayout,
    tensor_offsets: HashMap<String, u64>,
    pub layers: usize,
    icb: Option<crate::metal::step_icb::StepIcbPair>,
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
        let mut params = self.read_params();
        params.kv_len = kv_len;
        self.write_params(params);
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
        state.argmax_stable = 0;
        state.argmax_changed = 0;
        state.mean_entropy = 0.0;
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
            ps: &self.pipelines,
            bufs: &self.bufs,
            block_profile: self.block_profile,
            use_sc_gemm: self.use_sc_gemm,
            tensor_offsets: &self.tensor_offsets,
            recorder: None,
        };
        f(&mut enc)?;
        enc.enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    fn dispatch_record<F>(&mut self, f: F, recorder: &mut crate::metal::step_icb::IcbRecorder) -> Result<(), Error>
    where
        F: FnOnce(&mut StepEnc<'_>) -> Result<(), Error>,
    {
        let cmd = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Format("command buffer alloc failed"))?;
        let enc_obj = cmd
            .computeCommandEncoder()
            .ok_or(Error::Format("compute encoder alloc failed"))?;
        let mut enc = StepEnc {
            enc: enc_obj,
            ps: &self.pipelines,
            bufs: &self.bufs,
            block_profile: self.block_profile,
            use_sc_gemm: self.use_sc_gemm,
            tensor_offsets: &self.tensor_offsets,
            recorder: Some(recorder),
        };
        f(&mut enc)?;
        enc.enc.endEncoding();
        Ok(())
    }

    fn record_icb_plan(&mut self, with_sc: bool) -> Result<crate::metal::step_icb::StepIcbPlan, Error> {
        let mut state = self.read_canvas_state();
        state.step = if with_sc { 1 } else { 0 };
        self.write_canvas_state(&state);

        let mut recorder = crate::metal::step_icb::IcbRecorder::new(&self.ctx.device)?;
        let layout = self.layout;
        let layers = self.layers;
        let finish = StepFinishMode::Full;
        let first_step = if with_sc { 0u32 } else { 1u32 };
        self.dispatch_record(
            |enc| enc.interpret_step(&layout, layers, first_step, finish),
            &mut recorder,
        )?;
        recorder.finish()
    }

    fn ensure_no_sc_icb(&mut self) -> Result<(), Error> {
        let kv_len = self.read_params().kv_len;
        let started = Instant::now();
        let no_sc = self.record_icb_plan(false)?;
        eprintln!(
            "step-kernel: no_sc ICB ready (kv_len={kv_len}, {} cmds) in {:.2?}",
            no_sc.command_count,
            started.elapsed()
        );
        let pair = self.icb.get_or_insert_with(|| crate::metal::step_icb::StepIcbPair {
            no_sc: None,
            with_sc: None,
            no_sc_kv_len: u32::MAX,
        });
        pair.no_sc = Some(no_sc);
        pair.no_sc_kv_len = kv_len;
        pair.with_sc = None;
        Ok(())
    }

    /// Record with_sc against the current GPU buffers (post step 1).
    fn ensure_with_sc_icb(&mut self) -> Result<(), Error> {
        if self
            .icb
            .as_ref()
            .and_then(|p| p.with_sc.as_ref())
            .is_some()
        {
            return Ok(());
        }
        let started = Instant::now();
        let with_sc = self.record_icb_plan(true)?;
        eprintln!(
            "step-kernel: with_sc ICB ready ({} cmds/{} ops) in {:.2?}",
            with_sc.command_count,
            with_sc.command_count,
            started.elapsed()
        );
        self.icb.as_mut().expect("icb").with_sc = Some(with_sc);
        Ok(())
    }

    fn record_icb_pair(&mut self) -> Result<crate::metal::step_icb::StepIcbPair, Error> {
        Ok(crate::metal::step_icb::StepIcbPair {
            no_sc: None,
            with_sc: None,
            no_sc_kv_len: u32::MAX,
        })
    }

    /// Attention + dense FFN + GPU router; MoE expert matmuls on CPU (matches `.dgq` Q4 oracle).
    pub fn fill_moe_out_dgq_cpu(&mut self, layer: usize) -> Result<(), Error> {
        let route: RouteScratch = read_struct(&self.bufs.route);
        let routes = routes_from_route_scratch(&route);
        let moe_in = read_half_buffer_f32(&self.bufs.arena, self.bufs.arena_map.moein_off() as usize, CANVAS * HID);
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
        // ICB parity validated at kv_len=0 (step-smoke). Prefilled KV paths still use live encode.
        let icb_ok = self.read_params().kv_len == 0;
        if finish == StepFinishMode::Full && icb_ok && self.icb.is_some() {
            if first_step == 1 {
                self.ensure_no_sc_icb()?;
            } else {
                self.ensure_with_sc_icb()?;
            }
            if let Some(icb) = &self.icb {
                let plan = if first_step == 1 {
                    icb.no_sc.as_ref().ok_or(Error::Format("no_sc ICB missing"))?
                } else {
                    icb.with_sc.as_ref().ok_or(Error::Format("with_sc ICB missing"))?
                };
                let cmd = self
                    .ctx
                    .queue
                    .commandBuffer()
                    .ok_or(Error::Format("command buffer alloc failed"))?;
                let enc = crate::metal::step_icb::replay_step_icb(&cmd, plan)?;
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                return Ok(());
            }
        }
        self.dispatch_and_wait(|enc| {
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

pub fn log_step_memory_budget(blob_bytes: u64, max_seq: usize, layout: &ModelLayout, use_sc_gemm: bool) {
    let kv = kv_cache_bytes(layout, max_seq);
    let logits = (CANVAS * VOCAB * 2) as u64;
    let sc_probs = if use_sc_gemm { logits } else { 0 };
    let arena = step_arena_layout().bytes();
    let (mx, mw) = gemm_scratch_bytes();
    let gemm_scratch = (mx + mw) as u64;
    let gpu_static = kv + logits + sc_probs + arena + gemm_scratch;
    let total = blob_bytes + gpu_static;
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

    let block_profile = StepBlockProfile::from_store_profile(store.profile());
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

    let ctx = MetalContext::new()?;
    let compile_started = Instant::now();
    let pipelines = shared_step_pipelines(&ctx)?;
    let compile = compile_started.elapsed();

    let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device)?;
    let gpu_blob = std::sync::Arc::clone(&gpu_blob);
    let kv_bytes = kv_cache_bytes(&layout, cfg.max_seq) as usize;
    let use_sc_gemm = step_use_sc_gemm_from_env();
    let logits_bytes = CANVAS * VOCAB * 2;
    let sc_probs_bytes = if use_sc_gemm { logits_bytes } else { 1 };

    log_step_memory_budget(store.blob_bytes(), cfg.max_seq, &layout, use_sc_gemm);

    let sampler = crate::sample::sampler_for_steps(cfg.steps.max(1), cfg.no_early_stop);
    let prefill_len = cfg
        .prefill_token_ids
        .as_ref()
        .map(|t| t.len() as u32)
        .unwrap_or(cfg.kv_len);
    let params = step_params_from_sampler(&sampler, prefill_len, cfg.no_early_stop);
    let state = init_canvas_state(cfg.seed, VOCAB);
    let (gemm_a_bytes, gemm_b_bytes) = gemm_scratch_bytes();

    let model_cfg = ModelConfig::load(model_dir)?;
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
        arena_map,
        arena_layout_buf: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ArenaLayout>())?;
            write_struct(&b, &arena_map);
            b
        },
    };
    zero_buffer(&bufs.expert_layer_unique);
    zero_buffer(&bufs.arena);
    zero_buffer(&bufs.kvcache);
    zero_buffer(&bufs.logits);

    if let Some(ref token_ids) = cfg.prefill_token_ids {
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

    let build = StepRuntimeBuildTiming {
        compile,
        total: build_started.elapsed(),
    };
    eprintln!(
        "step-kernel: runtime built (total={:.2?}, compile={:.2?})",
        build.total, build.compile
    );
    let mut rt = StepRuntime {
        ctx,
        pipelines,
        bufs,
        gpu_blob,
        weight_cache,
        text_config,
        block_profile,
        use_sc_gemm: step_use_sc_gemm_from_env(),
        layout,
        tensor_offsets: offsets,
        layers,
        icb: None,
    };
    if crate::metal::step_icb::step_icb_enabled() {
        eprintln!("step-kernel: ICB replay enabled (lazy record per kv_len/step)");
        rt.icb = Some(rt.record_icb_pair()?);
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
    read_half_buffer_f32(arena, byte_off, width)
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

    let q_all = read_half_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.attnq_off() as usize, CANVAS * q_width);
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
            .map(|k| f16_bits_to_f32(route.weight[tok][k]))
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
    for (i, &tok) in state.token_list.iter().enumerate() {
        route.token_list[i] = tok;
        route.slot_list[i] = state.slot_list[i];
    }
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
    let moe_in = read_half_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.moein_off() as usize, CANVAS * HID);
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

    if let Ok(path) = std::env::var("DGQ_MOE_ROUTE_REF") {
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

/// Read `elems` half values from a shared Metal buffer as f32.
pub fn read_half_buffer_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
) -> Vec<f32> {
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    (0..elems)
        .map(|i| unsafe { f16_bits_to_f32(*ptr.add(i)) })
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
    let norm_hidden = read_half_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.tmp_off() as usize, CANVAS * HID);
    rt.dispatch_and_wait(|enc| {
        enc.gemm_q8_logits(
            enc.arena().tmp_off(),
            layout.embed,
            CANVAS as u32,
            VOCAB as u32,
            HID as u32,
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
    let params = step_params_from_sampler(&sampler, rt.read_params().kv_len, cfg.no_early_stop);
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
        let confident_stop = !cfg.no_early_stop
            && stopper.should_stop_with_entropies(&st.prev_argmax, &st.entropy, st.step);
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

    #[test]
    fn sc_gemm_softembed_matches_slow_kernel() {
        use std::path::Path;

        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip sc_gemm_softembed_matches_slow_kernel");
            return;
        }
        let cfg = StepSmokeConfig {
            finish: StepFinishMode::Full,
            steps: 1,
            ..StepSmokeConfig::default()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
        let layout = rt.layout;
        rt.run_forward_once(StepFinishMode::Full)
            .expect("seed logits from step 1");

        let soft_elems = CANVAS * HID;
        rt.dispatch_and_wait(|enc| {
            enc.encode_sc_logit_rowstats();
            enc.encode_sc_softembed_path(&layout, false)?;
            Ok(())
        })
        .expect("slow sc softembed");
        let slow = read_half_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.soft_off() as usize, soft_elems);

        rt.dispatch_and_wait(|enc| {
            enc.encode_sc_logit_rowstats();
            enc.encode_sc_softembed_path(&layout, true)?;
            Ok(())
        })
        .expect("fast sc gemm softembed");
        let fast = read_half_buffer_f32(&rt.bufs.arena, rt.bufs.arena_map.soft_off() as usize, soft_elems);

        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut max_abs = 0.0f32;
        for (a, b) in slow.iter().zip(fast.iter()) {
            dot += *a as f64 * *b as f64;
            na += *a as f64 * *a as f64;
            nb += *b as f64 * *b as f64;
            max_abs = max_abs.max((a - b).abs());
        }
        let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
        eprintln!("sc softembed fast vs slow: cos={cos:.6} max_abs={max_abs:.6}");
        assert!(cos > 0.999, "sc gemm cos={cos}");
        assert!(max_abs < 0.05, "sc gemm max_abs={max_abs}");
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

    fn icb_plan_parity(
        rt: &mut StepRuntime,
        layout: &ModelLayout,
        layers: usize,
        first_step: u32,
    ) -> f32 {
        if first_step == 1 {
            rt.ensure_no_sc_icb().expect("no_sc icb");
        } else {
            rt.ensure_with_sc_icb().expect("with_sc icb");
        }
        let snap = snapshot_step_gpu(rt);
        rt.dispatch_and_wait(|enc| {
            enc.encode_step_preamble(layout, first_step)?;
            for layer in 0..layers {
                enc.encode_full_layer(layer, layout)?;
            }
            enc.encode_step_finish(layout, StepFinishMode::Full)?;
            Ok(())
        })
        .expect("live");
        let live_logits = read_half_buffer_f32(&rt.bufs.logits, 0, CANVAS * VOCAB);

        // Sequential generate never rewinds arena/kv between live reference and ICB replay.
        rt.write_canvas_state(&snap.state);
        let icb = rt.icb.as_ref().expect("icb plan");
        let plan = if first_step == 1 {
            icb.no_sc.as_ref().expect("no_sc plan")
        } else {
            icb.with_sc.as_ref().expect("with_sc plan")
        };
        let cmd = rt.ctx.queue.commandBuffer().expect("cmd");
        let mut enc =
            crate::metal::step_icb::replay_step_icb(&cmd, plan)
                .expect("replay");
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let icb_logits = read_half_buffer_f32(&rt.bufs.logits, 0, CANVAS * VOCAB);
        live_logits
            .iter()
            .zip(icb_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Tier-2 ICB fixture per STRATEGY.md §5: real weights, few layers, seconds not minutes.
    fn icb_tier2_config() -> StepSmokeConfig {
        StepSmokeConfig {
            layers: 3,
            steps: 1,
            finish: StepFinishMode::Full,
            no_early_stop: true,
            ..StepSmokeConfig::default()
        }
    }

    fn icb_prefill_tier2_config() -> StepSmokeConfig {
        icb_tier2_config()
    }

    #[test]
    fn icb_replay_matches_live_tier2() {
        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip icb_replay_matches_live_tier2");
            return;
        }
        let cfg = icb_tier2_config();
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("build");
        let layout = rt.layout;
        let layers = rt.layers;

        let no_sc = icb_plan_parity(&mut rt, &layout, layers, 1);
        eprintln!("icb tier2 no_sc: logits_max_abs={no_sc:.6}");
        assert!(no_sc < 0.05, "no_sc drift max_abs={no_sc}");

        rt.run_forward_once(StepFinishMode::Full).expect("step1 icb");
        let with_sc = icb_plan_parity(&mut rt, &layout, layers, 0);
        eprintln!("icb tier2 with_sc: logits_max_abs={with_sc:.6}");
        assert!(with_sc < 0.05, "with_sc drift max_abs={with_sc}");
    }

    /// Prefilled-KV ICB: run manually while debugging generate path (`cargo test --ignored icb_prefill`).
    #[test]
    #[ignore = "tier-2 on demand: prefilled KV ICB (STRATEGY.md §5)"]
    fn icb_prefill_no_sc_tier2() {
        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip icb_prefill_no_sc_tier2");
            return;
        }
        let prefill = hello_chat_prefill_token_ids(dir).expect("hello prefill");
        let cfg = StepSmokeConfig {
            prefill_token_ids: Some(prefill),
            ..icb_prefill_tier2_config()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("build");
        let layout = rt.layout;
        let layers = rt.layers;
        let sampler = crate::sample::sampler_for_steps(1, true);
        let params = step_params_from_sampler(&sampler, rt.read_params().kv_len, true);
        let mut rng = Rng::new(cfg.seed);
        rt.reset_block(VOCAB, &mut rng, params);
        let max = icb_plan_parity(&mut rt, &layout, layers, 1);
        eprintln!("icb prefill tier2 no_sc: logits_max_abs={max:.6}");
        assert!(max < 0.05, "prefill no_sc drift max_abs={max}");
    }

    #[test]
    fn icb_nvfp4_fused_matches_live_encode() {
        let dir = Path::new("/tmp/nvfp4-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            eprintln!("skip icb_nvfp4_fused_matches_live_encode");
            return;
        }
        let cfg = StepSmokeConfig {
            ..icb_tier2_config()
        };
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("build");
        if rt.icb.is_none() {
            eprintln!("skip icb_nvfp4_fused: ICB disabled on fused nvfp4 path");
            return;
        }
        let layout = rt.layout;
        let layers = rt.layers;
        let no_sc = icb_plan_parity(&mut rt, &layout, layers, 1);
        eprintln!("icb nvfp4 no_sc parity: logits_max_abs={no_sc:.6}");
        assert!(no_sc < 0.05, "nvfp4 no_sc drift max_abs={no_sc}");

        rt.dispatch_and_wait(|enc| {
            enc.encode_step_preamble(&layout, 1)?;
            for layer in 0..layers {
                enc.encode_full_layer(layer, &layout)?;
            }
            enc.encode_step_finish(&layout, StepFinishMode::Full)?;
            Ok(())
        })
        .expect("step1 live");
        let with_sc = icb_plan_parity(&mut rt, &layout, layers, 0);
        eprintln!("icb nvfp4 with_sc parity: logits_max_abs={with_sc:.6}");
        assert!(with_sc < 0.05, "nvfp4 with_sc drift max_abs={with_sc}");
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
}
