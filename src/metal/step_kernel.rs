//! Monolithic diffgemma denoise-step kernel (parallel smoke path).
//! See `shaders/diffgemma_step.metal` and dispatch schedule at file bottom.

use crate::dgq::DgqStore;
use crate::metal::batch::set_bytes;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::dgq_gpu::{DgqGpuBlob, Q4LinearGpu};
use crate::metal::mps_gemm::{dispatch_dequant_nvfp4_matrix, dispatch_dequant_q4_matrix, MpsMatmulCache};
use crate::sample::{initialize_canvas, Rng, SamplerConfig, StableConfidentStopper};
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize, MTLLibrary,
};
use std::collections::HashMap;
use std::mem::offset_of;
use std::path::Path;
use std::time::Instant;

const STEP_SHADER: &str = include_str!("../../shaders/diffgemma_step.metal");
const QGEMM_SHADER: &str = include_str!("../../shaders/qgemm.metal");

pub const HID: usize = 2816;
pub const VOCAB: usize = 262144;
pub const CANVAS: usize = 256;
pub const N_LAYERS: usize = 30;
pub const N_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
pub const DENSE_FF: u32 = 2112;
pub const MOE_FF: u32 = 704;

pub const A_HIDDEN: u64 = 0;
pub const A_ATTNQ: u64 = 2_883_584;
pub const A_ATTNK: u64 = 7_077_888;
pub const A_ATTNV: u64 = 8_126_464;
pub const A_ATTNO: u64 = 9_175_040;
pub const A_FFG: u64 = 13_369_344;
pub const A_FFU: u64 = 14_450_688;
pub const A_MOEIN: u64 = 15_532_032;
pub const A_DENSE: u64 = 16_973_824;
pub const A_MOEOUT: u64 = 18_415_616;
pub const A_SOFT: u64 = 21_299_200;
pub const A_STREAM: u64 = 22_740_992;
pub const A_TMP: u64 = 24_182_784;
pub const A_RS_SC: u64 = 25_624_576;
pub const A_RS_SAMP: u64 = 25_626_624;
pub const ARENA_BYTES: u64 = 25_628_672;

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
#[derive(Clone, Copy)]
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
pub struct RouteScratch {
    pub weight: [[u16; TOP_K]; CANVAS],
    pub expert: [[u32; TOP_K]; CANVAS],
    pub count: [u32; N_EXPERTS],
    pub offset: [u32; N_EXPERTS],
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
    /// When set, overrides `DGQ_MPS_Q4` for this run (deterministic goldens use `false`).
    pub use_mps_q4: Option<bool>,
    /// Prompt token ids for encoder prefill into b4 (M1). When set, `StepParams.kv_len` = len.
    pub prefill_token_ids: Option<Vec<u32>>,
    /// Match `generate-monolithic --no-early-stop` (disables confidence early stop).
    pub no_early_stop: bool,
    /// Encoder prefill/extend Q4 path (`DGQ_MPS_Q4` when `None`).
    pub encoder_use_mps_q4: Option<bool>,
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
            use_mps_q4: None,
            prefill_token_ids: None,
            no_early_stop: false,
            encoder_use_mps_q4: None,
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
    pub accept_count: u32,
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
    pub non_finite: usize,
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

pub fn build_offsets_from_store(store: &DgqStore) -> HashMap<String, u64> {
    let mut offsets = HashMap::new();
    for entry in store.tensor_entries() {
        offsets.insert(entry.name.clone(), entry.meta.offset);
    }
    offsets
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

/// Scratch byte sizes for the MPS Q4 path (max over all step-kernel GEMM shapes).
fn mps_scratch_bytes() -> (usize, usize, usize) {
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
    let mut max_mn = 0usize;
    for (m, n, k) in shapes {
        max_mk = max_mk.max(m * k as usize);
        max_nk = max_nk.max(n as usize * k as usize);
        max_mn = max_mn.max(m * n as usize);
    }
    (
        max_mk * std::mem::size_of::<f32>(),
        max_nk * std::mem::size_of::<f32>(),
        max_mn * std::mem::size_of::<f32>(),
    )
}

fn step_use_sc_gemm_from_env() -> bool {
    match std::env::var("DGQ_SC_GEMM") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
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

/// Step-kernel dense GEMM backend. Default is MPS dequant→matmul (opt out with `DGQ_STEP_MPS_Q4=0`).
pub fn step_use_mps_q4_default() -> bool {
    match std::env::var("DGQ_STEP_MPS_Q4") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn step_use_mps_q4_from_env() -> bool {
    match std::env::var("DGQ_STEP_MPS_Q4") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => crate::metal::engine::default_use_mps_q4(),
    }
}

struct StepPipelines {
    memzero: ComputePipeline,
    rmsnorm: ComputePipeline,
    rmsnorm_f32: ComputePipeline,
    dequant_q4: ComputePipeline,
    dequant_nvfp4: ComputePipeline,
    half_to_f32: ComputePipeline,
    f32_to_half: ComputePipeline,
    gemm_q4: HashMap<(u32, u32), ComputePipeline>,
    gemm_nvfp4: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8: HashMap<(u32, u32), ComputePipeline>,
    gemm_q8_rowk: HashMap<(u32, u32), ComputePipeline>,
    qk_rope_kv: ComputePipeline,
    attention: ComputePipeline,
    residual: ComputePipeline,
    residual_f32b: ComputePipeline,
    glu: ComputePipeline,
    router: ComputePipeline,
    bucket_count: ComputePipeline,
    bucket_fill: ComputePipeline,
    moe_grouped: ComputePipeline,
    moe_grouped_nvfp4: ComputePipeline,
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
    fn new(ctx: &MetalContext, library: &ProtocolObject<dyn MTLLibrary>) -> Result<Self, Error> {
        let simple = |e: &str| ctx.compile_kernel_from_library(library, e);
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
                ctx.compile_gemm_kernel(library, "k_gemm_q4", n, k)?,
            );
            gemm_nvfp4.insert(
                (n, k),
                ctx.compile_gemm_kernel(library, "k_gemm_nvfp4", n, k)?,
            );
        }
        for &(n, k) in &[
            (DENSE_FF, HID as u32),
            (2816, DENSE_FF),
            (VOCAB as u32, HID as u32),
        ] {
            gemm_q8.insert(
                (n, k),
                ctx.compile_gemm_kernel(library, "k_gemm_q8", n, k)?,
            );
        }
        for &(n, k) in &[(HID as u32, VOCAB as u32)] {
            gemm_q8_rowk.insert(
                (n, k),
                ctx.compile_gemm_kernel(library, "k_gemm_q8_rowk", n, k)?,
            );
        }
        Ok(Self {
            memzero: simple("k_memzero")?,
            rmsnorm: simple("k_rmsnorm")?,
            rmsnorm_f32: simple("k_rmsnorm_f32")?,
            dequant_q4: ctx.compile_kernel(QGEMM_SHADER, "dequant_q4_matrix")?,
            dequant_nvfp4: ctx.compile_kernel(QGEMM_SHADER, "dequant_nvfp4_matrix")?,
            half_to_f32: simple("k_half_to_f32")?,
            f32_to_half: simple("k_f32_to_half")?,
            gemm_q4,
            gemm_nvfp4,
            gemm_q8,
            gemm_q8_rowk,
            qk_rope_kv: simple("k_qk_rope_kv")?,
            attention: simple("k_attention")?,
            residual: simple("k_residual")?,
            residual_f32b: simple("k_residual_f32b")?,
            glu: simple("k_glu")?,
            router: simple("k_router")?,
            bucket_count: simple("k_bucket_count")?,
            bucket_fill: simple("k_bucket_fill")?,
            moe_grouped: simple("k_moe_grouped")?,
            moe_grouped_nvfp4: simple("k_moe_grouped_nvfp4")?,
            embed_gather: simple("k_embed_gather")?,
            logit_rowstats: simple("k_logit_rowstats")?,
            sc_probs: simple("k_sc_probs")?,
            sc_softembed: simple("k_sc_softembed")?,
            half_scale: simple("k_half_scale")?,
            softcap: simple("k_softcap")?,
            sample_rowstats: simple("k_sample_rowstats")?,
            sample_commit: simple("k_sample_commit")?,
            sample_apply: simple("k_sample_apply")?,
            sample_write: simple("k_sample_write")?,
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
}

struct StepBuffers {
    blob: Retained<ProtocolObject<dyn MTLBuffer>>,
    layout: Retained<ProtocolObject<dyn MTLBuffer>>,
    params: Retained<ProtocolObject<dyn MTLBuffer>>,
    arena: Retained<ProtocolObject<dyn MTLBuffer>>,
    kvcache: Retained<ProtocolObject<dyn MTLBuffer>>,
    state: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits: Retained<ProtocolObject<dyn MTLBuffer>>,
    sc_probs: Retained<ProtocolObject<dyn MTLBuffer>>,
    route: Retained<ProtocolObject<dyn MTLBuffer>>,
    mps_x: Retained<ProtocolObject<dyn MTLBuffer>>,
    mps_w: Retained<ProtocolObject<dyn MTLBuffer>>,
    mps_c: Retained<ProtocolObject<dyn MTLBuffer>>,
}

struct StepEnc<'a> {
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ps: &'a StepPipelines,
    bufs: &'a StepBuffers,
    gpu_blob: &'a std::sync::Arc<DgqGpuBlob>,
    mps: &'a mut MpsMatmulCache,
    use_mps_q4: bool,
    use_nvfp4: bool,
    use_sc_gemm: bool,
}

impl StepEnc<'_> {
    fn bind_blob(&self, idx: usize) {
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.blob), 0, idx);
        }
    }

    fn bind_layout(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.layout), 0, idx);
        }
    }

    fn bind_params(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.params), 0, idx);
        }
    }

    fn bind_arena(&self, idx: usize, byte_off: u64) {
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.arena),
                byte_off as usize,
                idx,
            );
        }
    }

    fn bind_kvcache(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.kvcache), 0, idx);
        }
    }

    fn bind_state(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.state), 0, idx);
        }
    }

    fn bind_logits(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.logits), 0, idx);
        }
    }

    fn bind_sc_probs(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.sc_probs), 0, idx);
        }
    }

    fn bind_route(&self, idx: usize) {
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.route), 0, idx);
        }
    }

    fn dispatch_1d(&self, ps: &ComputePipeline, count: usize, tpg: usize) {
        self.enc.setComputePipelineState(&ps.pipeline);
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Split 1D dispatches that would exceed Metal's 65535 threadgroup grid width.
    fn dispatch_1d_ranged(
        &self,
        ps: &ComputePipeline,
        count: usize,
        tpg: usize,
        mut encode: impl FnMut(&ProtocolObject<dyn MTLComputeCommandEncoder>, u32, u32),
    ) {
        const MAX_GROUPS: usize = 65535;
        let chunk_max = MAX_GROUPS * tpg;
        let mut base = 0usize;
        while base < count {
            let chunk = (count - base).min(chunk_max);
            self.enc.setComputePipelineState(&ps.pipeline);
            encode(&self.enc, base as u32, chunk as u32);
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
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            base += chunk;
        }
    }

    /// Softcap logits (matches sampler.metal ranged dispatch pattern).
    fn dispatch_softcap(&self) {
        let len = CANVAS * VOCAB;
        self.dispatch_1d_ranged(&self.ps.softcap, len, 256, |enc, base, chunk| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.bufs.logits), 0, 0);
            }
            set_bytes(enc, &base, 1);
            set_bytes(enc, &chunk, 2);
        });
    }

    fn pause_for_mps(&mut self) {
        self.enc.endEncoding();
    }

    fn resume_compute_after_mps(&mut self) {
        self.enc = self
            .cmd
            .computeCommandEncoder()
            .expect("compute encoder alloc failed");
    }

    fn dispatch_convert_1d(
        &self,
        ps: &ComputePipeline,
        src: &ProtocolObject<dyn MTLBuffer>,
        src_off: usize,
        dst: &ProtocolObject<dyn MTLBuffer>,
        dst_off: usize,
        len: usize,
    ) {
        self.dispatch_1d_ranged(ps, len, 256, |enc, base, chunk| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(src), src_off, 0);
                enc.setBuffer_offset_atIndex(Some(dst), dst_off, 1);
            }
            set_bytes(enc, &base, 2);
            set_bytes(enc, &chunk, 3);
        });
    }

    fn half_to_f32_buf(&self, arena_off: u64, len: usize) {
        self.dispatch_convert_1d(
            &self.ps.half_to_f32,
            &self.bufs.arena,
            arena_off as usize,
            &self.bufs.mps_x,
            0,
            len,
        );
    }

    fn f32_to_half_arena(&self, arena_off: u64, len: usize) {
        self.dispatch_convert_1d(
            &self.ps.f32_to_half,
            &self.bufs.mps_c,
            0,
            &self.bufs.arena,
            arena_off as usize,
            len,
        );
    }

    fn gemm_q4_fused(
        &self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = if self.use_nvfp4 {
            self.ps.nvfp4(n, k)?
        } else {
            self.ps.q4(n, k)?
        };
        self.enc.setComputePipelineState(&ps.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), x_off as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 1);
            self.bind_blob(2);
            set_bytes(&self.enc, &w_off, 3);
            set_bytes(&self.enc, &m, 4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
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
        if !self.use_mps_q4 {
            return self.gemm_q4_fused(x_off, y_off, w_off, m, n, k);
        }
        let m_us = m as usize;
        let n_us = n as usize;
        let k_us = k as usize;
        let q4 = Q4LinearGpu::from_entry(
            std::sync::Arc::clone(self.gpu_blob),
            w_off,
            n_us,
            k_us,
            if self.use_nvfp4 {
                crate::dgq::layout::QuantKind::Nvfp4Block
            } else {
                crate::dgq::layout::QuantKind::Q4Block
            },
        );
        self.half_to_f32_buf(x_off, m_us * k_us);
        if self.use_nvfp4 {
            dispatch_dequant_nvfp4_matrix(&self.enc, &self.ps.dequant_nvfp4, &q4, &self.bufs.mps_w);
        } else {
            dispatch_dequant_q4_matrix(&self.enc, &self.ps.dequant_q4, &q4, &self.bufs.mps_w);
        }
        self.pause_for_mps();
        self.mps.encode_f32_linear(
            &self.cmd,
            &self.bufs.mps_x,
            &self.bufs.mps_w,
            &self.bufs.mps_c,
            m_us,
            k_us,
            n_us,
        );
        self.resume_compute_after_mps();
        self.f32_to_half_arena(y_off, m_us * n_us);
        Ok(())
    }

    fn dispatch_2d(&self, ps: &ComputePipeline, gx: usize, gy: usize, tpg_x: usize, tpg_y: usize) {
        self.enc.setComputePipelineState(&ps.pipeline);
        let grid = MTLSize {
            width: div_up(gx, tpg_x),
            height: div_up(gy, tpg_y),
            depth: 1,
        };
        let tg = MTLSize {
            width: tpg_x,
            height: tpg_y,
            depth: 1,
        };
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    fn memzero_bytes(&self, byte_off: u64, nbytes: u64) {
        self.enc.setComputePipelineState(&self.ps.memzero.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.arena),
                byte_off as usize,
                0,
            );
        }
        let count = div_up(nbytes as usize, 16);
        self.dispatch_1d(&self.ps.memzero, count, 256);
    }

    fn rmsnorm(
        &self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        dim: u32,
        rows: usize,
    ) {
        self.enc.setComputePipelineState(&self.ps.rmsnorm.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), x_off as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 1);
            self.bind_blob(2);
            set_bytes(&self.enc, &w_off, 3);
            set_bytes(&self.enc, &dim, 4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    fn rmsnorm_f32(
        &self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        dim: u32,
        rows: usize,
    ) {
        self.enc.setComputePipelineState(&self.ps.rmsnorm_f32.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), x_off as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 1);
            self.bind_blob(2);
            set_bytes(&self.enc, &w_off, 3);
            set_bytes(&self.enc, &dim, 4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    fn gemm_q8(
        &self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.q8(n, k)?;
        self.enc.setComputePipelineState(&ps.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), x_off as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 1);
            self.bind_blob(2);
            set_bytes(&self.enc, &w_off, 3);
            set_bytes(&self.enc, &m, 4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        Ok(())
    }

    fn gemm_q8_logits(
        &self,
        x_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.q8(n, k)?;
        self.enc.setComputePipelineState(&ps.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), x_off as usize, 0);
            self.bind_logits(1);
            self.bind_blob(2);
            set_bytes(&self.enc, &w_off, 3);
            set_bytes(&self.enc, &m, 4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        Ok(())
    }

    /// probs [M,K] half buffer @ sc_probs → arena y_off [M,N] via q8 weights.
    fn gemm_q8_probs(
        &self,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.q8_rowk(n, k)?;
        self.enc.setComputePipelineState(&ps.pipeline);
        unsafe {
            self.bind_sc_probs(0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 1);
            self.bind_blob(2);
            set_bytes(&self.enc, &w_off, 3);
            set_bytes(&self.enc, &m, 4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        Ok(())
    }

    fn scale_half_arena(&self, y_off: u64, elems: usize, scale: f32) {
        self.enc
            .setComputePipelineState(&self.ps.half_scale.pipeline);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 0);
            set_bytes(&self.enc, &(elems as u32), 1);
            set_bytes(&self.enc, &scale, 2);
        }
        self.dispatch_1d(&self.ps.half_scale, elems, 256);
    }

    fn encode_sc_softembed(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        if self.use_sc_gemm {
            self.enc
                .setComputePipelineState(&self.ps.sc_probs.pipeline);
            self.bind_logits(0);
            unsafe {
                self.enc.setBuffer_offset_atIndex(
                    Some(&self.bufs.arena),
                    A_RS_SC as usize,
                    1,
                );
            }
            self.bind_sc_probs(2);
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
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

            self.gemm_q8_probs(A_SOFT, layout.embed, CANVAS as u32, HID as u32, VOCAB as u32)?;
            self.scale_half_arena(A_SOFT, CANVAS * HID as usize, (HID as f32).sqrt());
        } else {
            self.enc
                .setComputePipelineState(&self.ps.sc_softembed.pipeline);
            unsafe {
                self.bind_logits(0);
                self.enc.setBuffer_offset_atIndex(
                    Some(&self.bufs.arena),
                    A_RS_SC as usize,
                    1,
                );
                self.bind_blob(2);
                self.bind_layout(3);
                self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_SOFT as usize, 4);
                let zero: u32 = 0;
                set_bytes(&self.enc, &zero, 5);
            }
            // One TG per (64-dim slice, canvas token): tgid.x = dim-block, tgid.y = tok.
            let grid = MTLSize {
                width: (HID as usize + 63) / 64,
                height: CANVAS,
                depth: 1,
            };
            let tg = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
        Ok(())
    }

    fn residual(&self, a_off: u64, b_off: u64, y_off: u64, scal_off: u64, elems: usize) {
        self.enc.setComputePipelineState(&self.ps.residual.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), a_off as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), b_off as usize, 1);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 2);
            self.bind_blob(3);
            set_bytes(&self.enc, &scal_off, 4);
        }
        self.dispatch_1d(&self.ps.residual, elems, 256);
    }

    fn glu(&self, gate_off: u64, up_off: u64, y_off: u64, elems: usize) {
        self.enc.setComputePipelineState(&self.ps.glu.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), gate_off as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), up_off as usize, 1);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), y_off as usize, 2);
        }
        self.dispatch_1d(&self.ps.glu, elems, 256);
    }

    fn encode_layer(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let q_n = if l.is_full != 0 { 8192 } else { 4096 };
        let k_n = if l.is_full != 0 { 1024 } else { 2048 };
        let o_k = q_n;
        let qk_y = (16 + 2 * l.n_kv_heads) as usize;
        let layer_off = layer_byte_offset(layer);

        self.rmsnorm(A_HIDDEN, A_TMP, l.input_ln, HID as u32, CANVAS);
        self.gemm_q4(A_TMP, A_ATTNQ, l.q_proj, CANVAS as u32, q_n, HID as u32)?;
        self.gemm_q4(A_TMP, A_ATTNK, l.k_proj, CANVAS as u32, k_n, HID as u32)?;
        if l.v_proj != 0 {
            self.gemm_q4(A_TMP, A_ATTNV, l.v_proj, CANVAS as u32, k_n, HID as u32)?;
        }

        self.enc.setComputePipelineState(&self.ps.qk_rope_kv.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNQ as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNK as usize, 1);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNV as usize, 2);
            self.bind_kvcache(3);
            self.bind_blob(4);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.layout),
                layer_off as usize,
                5,
            );
            self.bind_params(6);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.enc.setComputePipelineState(&self.ps.attention.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNQ as usize, 0);
            self.bind_kvcache(1);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNO as usize, 2);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.layout),
                layer_off as usize,
                3,
            );
            self.bind_params(4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.gemm_q4(A_ATTNO, A_TMP, l.o_proj, CANVAS as u32, HID as u32, o_k)?;
        self.rmsnorm(A_TMP, A_TMP, l.post_attn_ln, HID as u32, CANVAS);
        self.residual(A_HIDDEN, A_TMP, A_STREAM, 0, CANVAS * HID);

        self.rmsnorm(A_STREAM, A_TMP, l.pre_ff_ln, HID as u32, CANVAS);
        self.gemm_q4(A_TMP, A_FFG, l.mlp_gate, CANVAS as u32, DENSE_FF, HID as u32)?;
        self.gemm_q4(A_TMP, A_FFU, l.mlp_up, CANVAS as u32, DENSE_FF, HID as u32)?;
        self.glu(A_FFG, A_FFU, A_FFG, CANVAS * DENSE_FF as usize);
        self.gemm_q4(A_FFG, A_DENSE, l.mlp_down, CANVAS as u32, HID as u32, DENSE_FF)?;
        self.rmsnorm(A_DENSE, A_DENSE, l.post_ff_ln_1, HID as u32, CANVAS);

        self.enc.setComputePipelineState(&self.ps.router.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_STREAM as usize, 0);
            self.bind_blob(1);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.layout),
                layer_off as usize,
                2,
            );
            self.bind_route(3);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.enc.setComputePipelineState(&self.ps.bucket_count.pipeline);
        self.bind_route(0);
        self.dispatch_1d(&self.ps.bucket_count, 128, 128);

        for phase in 0u32..3 {
            self.enc.setComputePipelineState(&self.ps.bucket_fill.pipeline);
            self.bind_route(0);
            set_bytes(&self.enc, &phase, 1);
            let count = if phase == 1 { 1 } else { CANVAS * TOP_K };
            self.dispatch_1d(&self.ps.bucket_fill, count, 256);
        }

        self.rmsnorm(A_STREAM, A_MOEIN, l.pre_ff_ln_2, HID as u32, CANVAS);
        self.memzero_bytes(A_MOEOUT, (CANVAS * HID * 4) as u64);

        self.enc.setComputePipelineState(
            if self.use_nvfp4 {
                &self.ps.moe_grouped_nvfp4.pipeline
            } else {
                &self.ps.moe_grouped.pipeline
            },
        );
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_MOEIN as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_MOEOUT as usize, 1);
            self.bind_blob(2);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.layout),
                layer_off as usize,
                3,
            );
            self.bind_route(4);
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.rmsnorm_f32(A_MOEOUT, A_MOEIN, l.post_ff_ln_2, HID as u32, CANVAS);
        self.residual(A_DENSE, A_MOEIN, A_TMP, 0, CANVAS * HID);
        self.rmsnorm(A_TMP, A_TMP, l.post_ff_ln, HID as u32, CANVAS);
        self.residual(A_STREAM, A_TMP, A_HIDDEN, l.layer_scalar, CANVAS * HID);
        Ok(())
    }

    /// Run one decoder layer through QK-RoPE-KV + attention only (stops before o_proj).
    fn encode_layer_through_attention(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let q_n = if l.is_full != 0 { 8192 } else { 4096 };
        let k_n = if l.is_full != 0 { 1024 } else { 2048 };
        let qk_y = (16 + 2 * l.n_kv_heads) as usize;
        let layer_off = layer_byte_offset(layer);

        self.rmsnorm(A_HIDDEN, A_TMP, l.input_ln, HID as u32, CANVAS);
        self.gemm_q4(A_TMP, A_ATTNQ, l.q_proj, CANVAS as u32, q_n, HID as u32)?;
        self.gemm_q4(A_TMP, A_ATTNK, l.k_proj, CANVAS as u32, k_n, HID as u32)?;
        if l.v_proj != 0 {
            self.gemm_q4(A_TMP, A_ATTNV, l.v_proj, CANVAS as u32, k_n, HID as u32)?;
        }

        self.enc.setComputePipelineState(&self.ps.qk_rope_kv.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNQ as usize, 0);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNK as usize, 1);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNV as usize, 2);
            self.bind_kvcache(3);
            self.bind_blob(4);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.layout),
                layer_off as usize,
                5,
            );
            self.bind_params(6);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.enc.setComputePipelineState(&self.ps.attention.pipeline);
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNQ as usize, 0);
            self.bind_kvcache(1);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_ATTNO as usize, 2);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.layout),
                layer_off as usize,
                3,
            );
            self.bind_params(4);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        Ok(())
    }

    /// Canvas token embed gather only (no SC residual, no no-scale RMSNorm).
    fn encode_preamble_embed_only(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        let _ = layout;
        self.enc.setComputePipelineState(&self.ps.embed_gather.pipeline);
        unsafe {
            self.bind_blob(0);
            self.bind_layout(1);
            self.bind_state(2);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_HIDDEN as usize, 3);
        }
        let grid = MTLSize {
            width: HID,
            height: CANVAS,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        Ok(())
    }

    fn encode_step_preamble(&mut self, layout: &ModelLayout, first_step: u32) -> Result<(), Error> {
        if first_step == 0 {
            self.enc.setComputePipelineState(&self.ps.logit_rowstats.pipeline);
            self.bind_logits(0);
            unsafe {
                self.enc.setBuffer_offset_atIndex(
                    Some(&self.bufs.arena),
                    A_RS_SC as usize,
                    1,
                );
            }
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
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

            self.encode_sc_softembed(layout)?;

            self.rmsnorm(A_SOFT, A_TMP, layout.sc_pre_norm, HID as u32, CANVAS);
            self.gemm_q8(
                A_TMP,
                A_FFG,
                layout.sc_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q8(
                A_TMP,
                A_FFU,
                layout.sc_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.glu(A_FFG, A_FFU, A_FFG, CANVAS * DENSE_FF as usize);
            self.gemm_q8(
                A_FFG,
                A_DENSE,
                layout.sc_down,
                CANVAS as u32,
                HID as u32,
                DENSE_FF,
            )?;
        }
        // first_step: A_DENSE stays zero; skip SC MLP + O(vocab) softembed.

        self.enc.setComputePipelineState(&self.ps.embed_gather.pipeline);
        unsafe {
            self.bind_blob(0);
            self.bind_layout(1);
            self.bind_state(2);
            self.enc.setBuffer_offset_atIndex(Some(&self.bufs.arena), A_HIDDEN as usize, 3);
        }
        let grid = MTLSize {
            width: HID,
            height: CANVAS,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.residual(A_HIDDEN, A_DENSE, A_HIDDEN, 0, CANVAS * HID);
        self.rmsnorm(A_HIDDEN, A_HIDDEN, 0, HID as u32, CANVAS);
        Ok(())
    }

    fn encode_step_finish(
        &mut self,
        layout: &ModelLayout,
        mode: StepFinishMode,
    ) -> Result<(), Error> {
        self.rmsnorm(A_HIDDEN, A_TMP, layout.final_norm, HID as u32, CANVAS);
        self.gemm_q8_logits(
            A_TMP,
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

    fn encode_step_sampler(&self, _layout: &ModelLayout) -> Result<(), Error> {
        self.enc.setComputePipelineState(&self.ps.sample_rowstats.pipeline);
        unsafe {
            self.bind_logits(0);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.arena),
                A_RS_SAMP as usize,
                1,
            );
            self.bind_state(2);
            self.bind_params(3);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.enc.setComputePipelineState(&self.ps.sample_commit.pipeline);
        self.bind_state(0);
        self.bind_params(1);
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.enc.setComputePipelineState(&self.ps.sample_apply.pipeline);
        unsafe {
            self.bind_logits(0);
            self.enc.setBuffer_offset_atIndex(
                Some(&self.bufs.arena),
                A_RS_SAMP as usize,
                1,
            );
            self.bind_state(2);
            self.bind_params(3);
        }
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        self.enc.setComputePipelineState(&self.ps.sample_write.pipeline);
        self.bind_state(0);
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
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
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

fn arena_hidden_stats(arena: &ProtocolObject<dyn MTLBuffer>) -> (bool, f32, usize) {
    let (finite, max_abs) =
        half_buffer_stats(arena, A_HIDDEN as usize, CANVAS * HID, CANVAS * HID);
    let non_finite = if finite { 0 } else { 1 };
    (finite, max_abs, non_finite)
}

pub struct StepRuntime {
    ctx: MetalContext,
    pipelines: &'static StepPipelines,
    bufs: StepBuffers,
    gpu_blob: std::sync::Arc<DgqGpuBlob>,
    mps_matmul: MpsMatmulCache,
    use_mps_q4: bool,
    use_nvfp4: bool,
    use_sc_gemm: bool,
    layout: ModelLayout,
    pub layers: usize,
}

impl StepRuntime {
    pub fn layout(&self) -> &ModelLayout {
        &self.layout
    }

    pub fn use_mps_q4(&self) -> bool {
        self.use_mps_q4
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
        self.run_forward_once(StepFinishMode::Full)
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
            cmd: cmd.clone(),
            ps: &self.pipelines,
            bufs: &self.bufs,
            gpu_blob: &self.gpu_blob,
            mps: &mut self.mps_matmul,
            use_mps_q4: self.use_mps_q4,
            use_nvfp4: self.use_nvfp4,
            use_sc_gemm: self.use_sc_gemm,
        };
        f(&mut enc)?;
        enc.enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    fn run_forward_once(&mut self, finish: StepFinishMode) -> Result<(), Error> {
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
        self.dispatch_and_wait(|enc| {
            let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };
            enc.encode_step_preamble(&layout, first_step)?;
            for layer in 0..layers {
                enc.encode_layer(layer, &layout)?;
            }
            enc.encode_step_finish(&layout, finish)?;
            Ok(())
        })
    }
}

static STEP_PIPELINES: std::sync::OnceLock<Result<StepPipelines, String>> =
    std::sync::OnceLock::new();

fn shared_step_pipelines(ctx: &MetalContext) -> Result<&'static StepPipelines, Error> {
    STEP_PIPELINES
        .get_or_init(|| {
            let result = ctx
                .compile_library(STEP_SHADER)
                .and_then(|library| StepPipelines::new(ctx, &library))
                .map_err(|e| e.to_string());
            if result.is_ok() {
                crate::metal::pipeline_cache::PipelineArchiveCache::flush_global();
            }
            result
        })
        .as_ref()
        .map_err(|msg| Error::Format(msg.as_str()))
}

pub fn log_step_memory_budget(blob_bytes: u64, max_seq: usize, layout: &ModelLayout, use_sc_gemm: bool) {
    let kv = kv_cache_bytes(layout, max_seq);
    let logits = (CANVAS * VOCAB * 2) as u64;
    let sc_probs = if use_sc_gemm { logits } else { 0 };
    let arena = ARENA_BYTES;
    let (mx, mw, mc) = mps_scratch_bytes();
    let mps = (mx + mw + mc) as u64;
    let gpu_static = kv + logits + sc_probs + arena + mps;
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
        "  mps scratch:{:.2} MiB",
        mps as f64 / (1024.0 * 1024.0)
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

    let use_nvfp4 = store.profile() == crate::dgq::layout::QuantProfile::Nvfp4;
    if use_nvfp4 {
        eprintln!("step-kernel: nvfp4 block weights");
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
    let (mps_x_bytes, mps_w_bytes, mps_c_bytes) = mps_scratch_bytes();

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
        arena: alloc_buffer(&ctx.device, ARENA_BYTES as usize)?,
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
        mps_x: alloc_buffer(&ctx.device, mps_x_bytes)?,
        mps_w: alloc_buffer(&ctx.device, mps_w_bytes)?,
        mps_c: alloc_buffer(&ctx.device, mps_c_bytes)?,
    };
    zero_buffer(&bufs.arena);
    zero_buffer(&bufs.kvcache);
    zero_buffer(&bufs.logits);

    if let Some(ref token_ids) = cfg.prefill_token_ids {
        let mut encoder = crate::metal::step_kv::MonolithicEncoderCache::open_opt(
            model_dir,
            CANVAS,
            cfg.max_seq,
            Some(std::sync::Arc::clone(&gpu_blob)),
            cfg.encoder_use_mps_q4,
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

    let mps_matmul = MpsMatmulCache::new(ctx.device.clone());
    let build = StepRuntimeBuildTiming {
        compile,
        total: build_started.elapsed(),
    };
    eprintln!(
        "step-kernel: runtime built (total={:.2?}, compile={:.2?})",
        build.total, build.compile
    );
    Ok((
        StepRuntime {
            ctx,
            pipelines,
            bufs,
            gpu_blob,
            mps_matmul,
            use_mps_q4: cfg.use_mps_q4.unwrap_or_else(step_use_mps_q4_from_env),
            use_nvfp4,
            use_sc_gemm: step_use_sc_gemm_from_env(),
            layout,
            layers,
        },
        build,
    ))
}

pub fn run_step_probe(model_dir: &Path, cfg: StepSmokeConfig) -> Result<StepProbeResult, Error> {
    let started = Instant::now();
    let (mut rt, _) = build_step_runtime(model_dir, &cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;
    let mut checkpoints = Vec::new();

    let mut push = |label: &str, finite: bool, max_abs: f32, non_finite: usize| {
        checkpoints.push(StepProbeCheckpoint {
            label: label.to_string(),
            finite,
            max_abs,
            non_finite,
        });
    };

    rt.dispatch_and_wait(|enc| {
        let first_step = 1u32;
        enc.encode_step_preamble(&layout, first_step)?;
        Ok(())
    })?;
    let (f, m, n) = arena_hidden_stats(&rt.bufs.arena);
    push("after_preamble", f, m, n);

    for layer in 0..layers {
        rt.dispatch_and_wait(|enc| {
            enc.encode_layer(layer, &layout)?;
            Ok(())
        })?;
        let (f, m, n) = arena_hidden_stats(&rt.bufs.arena);
        push(&format!("after_layer_{layer}"), f, m, n);
    }

    rt.dispatch_and_wait(|enc| {
        enc.rmsnorm(A_HIDDEN, A_TMP, layout.final_norm, HID as u32, CANVAS);
        enc.gemm_q8_logits(
            A_TMP,
            layout.embed,
            CANVAS as u32,
            VOCAB as u32,
            HID as u32,
        )?;
        enc.dispatch_softcap();
        Ok(())
    })?;
    let (bad, m) = count_non_finite_half(&rt.bufs.logits, CANVAS * VOCAB);
    push("after_lm_head_softcap", bad == 0, m, bad);

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
    pub elapsed: std::time::Duration,
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
    let started = Instant::now();
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;
    let mut checkpoints = Vec::new();

    rt.dispatch_and_wait(|enc| {
        enc.encode_step_preamble(&layout, 1)?;
        Ok(())
    })?;
    {
        let hidden = read_arena_hidden_row(&rt.bufs.arena, A_HIDDEN, position);
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
        rt.dispatch_and_wait(|enc| {
            enc.encode_layer(layer, &layout)?;
            Ok(())
        })?;
        let hidden = read_arena_hidden_row(&rt.bufs.arena, A_HIDDEN, position);
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
        enc.rmsnorm(A_HIDDEN, A_TMP, layout.final_norm, HID as u32, CANVAS);
        Ok(())
    })?;
    {
        let hidden = read_arena_hidden_row(&rt.bufs.arena, A_TMP, position);
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
        elapsed: started.elapsed(),
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
    let embed_scaled = read_arena_hidden_row(&rt.bufs.arena, A_HIDDEN, position);

    rt.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, 1))?;
    let after_preamble = read_arena_hidden_row(&rt.bufs.arena, A_HIDDEN, position);

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
    Ok(read_arena_hidden_row(&rt.bufs.arena, A_HIDDEN, 0))
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
        rt.dispatch_and_wait(|enc| {
            enc.encode_layer(l, &layout)?;
            Ok(())
        })?;
    }

    let hidden_in = read_arena_hidden_row(&rt.bufs.arena, A_HIDDEN, position);

    rt.dispatch_and_wait(|enc| {
        enc.encode_layer_through_attention(layer, &layout)?;
        Ok(())
    })?;

    let l = &layout.layers[layer];
    let hd = l.head_dim as usize;
    let nkv = l.n_kv_heads as usize;
    let n_heads = STEP_NQ_HEADS;
    let q_width = n_heads * hd;
    let kv_len = rt.read_params().kv_len;
    let total_kv = kv_len as usize + CANVAS;

    let hidden_ln = read_arena_hidden_row(&rt.bufs.arena, A_TMP, position);
    let q_all = read_half_buffer_f32(&rt.bufs.arena, A_ATTNQ as usize, CANVAS * q_width);
    let q_post_rope = read_arena_row(&rt.bufs.arena, A_ATTNQ, position, q_width);
    let attn_out = read_arena_row(&rt.bufs.arena, A_ATTNO, position, q_width);
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
        q_post_rope,
        attn_out,
        raw_scores,
        attn_probs,
        k_samples,
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
    rt.run_forward_once(StepFinishMode::ForwardOnly)?;
    let state: CanvasState = read_struct(&rt.bufs.state);
    Ok(StepForwardOutput {
        norm_hidden: read_half_buffer_f32(&rt.bufs.arena, A_TMP as usize, CANVAS * HID),
        logits: read_half_buffer_f32(&rt.bufs.logits, 0, CANVAS * VOCAB),
        token_ids: state.ids.to_vec(),
    })
}

#[derive(Debug)]
pub struct StepQ4MpsParityResult {
    pub layers: usize,
    pub kv_len: u32,
    pub hidden_max_abs: f32,
    pub logits_max_abs: f32,
    pub native_min_ent: f32,
    pub mps_min_ent: f32,
    pub pass: bool,
}

/// P1.10: native `k_gemm_q4` vs MPS dequant→matmul forward pass (optional KV prefill).
pub fn run_step_q4_mps_parity(
    model_dir: &Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    prefill_token_ids: Option<Vec<u32>>,
) -> Result<StepQ4MpsParityResult, Error> {
    let base = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        use_mps_q4: Some(false),
        prefill_token_ids,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    let mut mps_cfg = base.clone();
    mps_cfg.use_mps_q4 = Some(true);

    let native = run_step_forward(model_dir, &base)?;
    let mps = run_step_forward(model_dir, &mps_cfg)?;

    let hidden_max_abs = native
        .norm_hidden
        .iter()
        .zip(mps.norm_hidden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let logits_max_abs = native
        .logits
        .iter()
        .zip(mps.logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let ln_vocab = (VOCAB as f32).ln();
    let native_ent = crate::sample::token_entropy(&native.logits, CANVAS, VOCAB);
    let mps_ent = crate::sample::token_entropy(&mps.logits, CANVAS, VOCAB);
    let native_min_ent = native_ent.iter().cloned().fold(f32::INFINITY, f32::min);
    let mps_min_ent = mps_ent.iter().cloned().fold(f32::INFINITY, f32::min);

    let pass = native_min_ent < ln_vocab - 0.5
        && mps_min_ent < ln_vocab - 0.5
        && (native_min_ent - mps_min_ent).abs() < 3.0;

    Ok(StepQ4MpsParityResult {
        layers,
        kv_len,
        hidden_max_abs,
        logits_max_abs,
        native_min_ent,
        mps_min_ent,
        pass,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct DenoiseStepStats {
    pub accept_count: u32,
    pub mean_entropy: f32,
    pub min_entropy: f32,
    pub low_entropy_positions: u32,
}

/// Chat-templated `-p Hello` prefill token ids (matches `generate-monolithic` default).
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

/// Run exactly one monolithic denoise step (forward + GPU sampler) and return its stats.
pub fn run_single_denoise_step(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<DenoiseStepStats, Error> {
    let mut one = cfg.clone();
    one.steps = 1;
    one.finish = StepFinishMode::Full;
    let steps = run_denoise_steps(model_dir, &one)?;
    steps
        .into_iter()
        .next()
        .ok_or(Error::Format("denoise step produced no stats"))
}

/// Run `cfg.steps` monolithic denoise iterations; one stats record per iteration.
pub fn run_denoise_steps(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<Vec<DenoiseStepStats>, Error> {
    if cfg.finish != StepFinishMode::Full {
        return Err(Error::Format("run_denoise_steps requires StepFinishMode::Full"));
    }
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
    let layout = rt.layout;
    let layers = rt.layers;
    let started = Instant::now();
    for step_i in 0..steps {
        rt.dispatch_and_wait(|enc| {
            let cur_state: CanvasState = read_struct(&enc.bufs.state);
            let first_step = if cur_state.step == 0 { 1u32 } else { 0u32 };
            enc.encode_step_preamble(&layout, first_step)?;
            for layer in 0..layers {
                enc.encode_layer(layer, &layout)?;
            }
            enc.encode_step_finish(&layout, finish)?;
            Ok(())
        })?;
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
        accept_count: ent_stats.accept_count,
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
            use_mps_q4: Some(false),
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
            use_mps_q4: Some(false),
            encoder_use_mps_q4: Some(false),
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
