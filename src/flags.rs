//! Central runtime configuration.
//!
//! Every `DGQ_*` env flag is parsed EXACTLY ONCE into a [`RuntimeConfig`] value; all
//! runtime accessors read it through [`config`]. This replaces the old scatter
//! of per-call `std::env::var` reads:
//!
//! - **Production** reads a shared, read-only base lazily loaded from
//!   [`RuntimeConfig::from_env`] once (`base`).
//! - **Tests** craft an explicit [`RuntimeConfig`] and install it for their scope via
//!   [`install_for_test`] — a THREAD-LOCAL override (RAII guard restores the
//!   prior value on drop). Thread-local so a stray override can never leak into
//!   another test (libtest runs each on its own thread) or another reader. No
//!   `set_var`/`remove_var` (`unsafe` and racy under edition 2024).
//!
//! The public accessor functions (`freeze_enabled()`, `kv_format()`, …) keep
//! their signatures and are thin typed views into [`config`], so call sites are
//! unchanged and the change is provably byte-identical (the golden pack).
//!
//! Conventions preserved from the env era:
//! - Production toggles default ON and are opt-OUT (`=0` disables); each names a
//!   shipped, evidence-backed default. They exist for A/B triage.
//! - Sampler-semantics toggles carry their sign-off history in the doc.
//! - Debug/probe flags default OFF and are opt-IN (`=1` or a path enables).
//!
//! Deliberate exceptions still read env directly: test-only probe-prompt knobs
//! that are passed from the command line into `#[ignore]`d diagnostics
//! (`DGQ_E15_PROMPT`, `DGQ_E8_PROMPT`, `DGQ_E16_CARRIER` in step_kv.rs — a
//! crafted RuntimeConfig can't inject those), and the system-env fallbacks
//! `XDG_CACHE_HOME`/`HOME` (pipeline_cache.rs) / `COLUMNS` (chat_ui.rs).

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// `=0`/`false` disables; anything else (or unset) leaves it ON.
fn env_on_unless_zero(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

/// `=1`/`true` enables; anything else (or unset) leaves it OFF.
fn env_on_if_one(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

// ===========================================================================
// RuntimeConfig data model
// ===========================================================================

/// Sampler-semantics toggles (quality-affecting; changes need the multi-seed
/// gate + census + sign-off — see the quality-ratchet rule).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerFlags {
    /// `DGQ_FREEZE`: hard-freeze accepted canvas rows (legacy). Default OFF.
    pub freeze: bool,
    /// `DGQ_DENOISER_ARGMAX`: commit row argmax vs tempered sample. Default ON.
    pub denoiser_argmax: bool,
    /// `DGQ_EARLY_STOP_MEAN_ENT` (nats; 0 disables). Default 0.05.
    pub early_stop_mean_ent: f32,
    /// `DGQ_EMPTY_REPLY_RETRY`: E6 degenerate-first-block re-roll count. Default 3.
    pub empty_reply_retry: u32,
    /// `DGQ_WS_BLOCK_STOP`: drop pure-whitespace committed blocks. Default OFF.
    pub ws_block_stop: bool,
    /// `DGQ_BLOCK_COMMIT_MAX_ENT` (nats; 0 disables): non-convergence commit
    /// guard, a STOPGAP for the OpenCode `}\n`-flood collapse class. A block
    /// that burns the whole step schedule (max_steps) and still shows
    /// late-window mean entropy above this floor is NOT committed as-is:
    /// re-roll the canvas up to `DGQ_BLOCK_COMMIT_RETRY` times, then end the
    /// turn rather than commit the garble (committed non-converged blocks are
    /// self-consistent — later blocks converge cleanly ONTO the flood).
    /// Healthy blocks end far below 0.05; strained blocks measure >= 0.26.
    /// Validated on the deterministic repro at 0.2 (debug/opencode_collapse/).
    /// Default OFF (0) — the underlying KV-lineage drift is treated as the
    /// real bug; enable at 0.2 if the collapse class bites in the field.
    pub block_commit_max_ent: f32,
    /// `DGQ_BLOCK_COMMIT_RETRY`: fresh-noise re-rolls before the commit guard
    /// gives up and ends the turn. Default 1.
    pub block_commit_retry: u32,
    /// `DGQ_FORCE_CANVAS`: force the active canvas width (diagnostic). None = 256.
    pub force_canvas: Option<u32>,
}

impl Default for SamplerFlags {
    fn default() -> Self {
        Self {
            freeze: false,
            denoiser_argmax: true,
            early_stop_mean_ent: 0.05,
            empty_reply_retry: 3,
            ws_block_stop: false,
            block_commit_max_ent: 0.0,
            block_commit_retry: 1,
            force_canvas: None,
        }
    }
}

/// Production perf toggles (all default ON, opt-out for A/B triage).
#[derive(Debug, Clone, PartialEq)]
pub struct PerfFlags {
    /// `DGQ_MOE_FUSE_GATHER`: fused gate_up A-gather in the tunable expert GEMM.
    pub moe_fuse_gather: bool,
    /// `DGQ_MOE_PREFILL_BM`: 64|128 opt-in, else 32 (OFF).
    pub moe_prefill_block_m: u32,
    pub attn_mma: bool,
    pub attn_mma_full: bool,
    pub router_gemm: bool,
    pub sc_sparse: bool,
    pub attn_window: bool,
    pub fused_algebra: bool,
    pub partial_lm_head: bool,
    pub encoder_gpu_moe: bool,
    pub prefill_batch: bool,
    pub prefill_mma: bool,
    pub fast_block_extend: bool,
    pub prefill_resident: bool,
    /// `DGQ_ATTN_KV_BLOCK`: block size for sequential full-attention. Default 0 (OFF).
    pub attn_kv_block: usize,
    /// `DGQ_GEMM_ATTN` (E17): route full-layer PREFILL attention through the
    /// GEMM decomposition (attn_gemm_qk/softmax/pv) instead of attention_mma_full.
    /// ~1.78x on the attention kernel, -16-19% real 30k prefill. Not bit-identical
    /// (decomposition batch-softmax vs the flash kernel's online softmax) but
    /// gate-passing (17/17 x{7,42,123}, longctx 4/4); signed off default ON
    /// 2026-07-13. Prefill-only; denoise keeps attention_mma_full. `=0` restores it.
    pub gemm_attn: bool,
    /// `DGQ_GEMM_ATTN_HC` (E17a): Q heads processed per E17 dispatch batch.
    /// Bounds the S/P scratch to [HC][CANVAS][n_pad(max_seq)]. Default 16 (all
    /// heads): the holistic prefill BO (task #88) found HC=16 the proxy optimum
    /// at every kv (−0.3/1.5/3.1/3.6% at 2k/8k/30k/100k vs HC=4), and HC is
    /// numerically invariant (per-head disjoint scratch), so this is a bit-
    /// identical perf ship. Scratch ~1.6 GiB @100k on 36 GB; drop to 4 for very
    /// long contexts if memory-pressured.
    pub gemm_attn_head_chunk: usize,
    /// E17 tunable tile geometry (task #87 sweep). QK GEMM tile qk_bm x qk_bn,
    /// PV GEMM tile pv_bm x pv_bn, softmax threads/row. Defaults reproduce the
    /// shipped 64x64/256 kernel. `DGQ_GEMM_ATTN_{QK_BM,QK_BN,PV_BM,PV_BN,SM_TPG}`.
    pub gemm_attn_qk_bm: usize,
    pub gemm_attn_qk_bn: usize,
    pub gemm_attn_pv_bm: usize,
    pub gemm_attn_pv_bn: usize,
    pub gemm_attn_sm_tpg: usize,
    /// Holistic prefill sweep (task #88): dense tunable-GEMM tile + MoE-sparse
    /// N-tile. Defaults reproduce the shipped kernels (64x64 dense, 128 sparse
    /// N-tile with fixed 32-row block height). `DGQ_GEMM_TUNE_BM/BN`,
    /// `DGQ_MOE_SPARSE_BN`.
    pub gemm_tune_bm: usize,
    pub gemm_tune_bn: usize,
    pub moe_sparse_bn: usize,
    /// `DGQ_FLASH_PREFILL` (E18 sliding revival): fused flash for sliding-layer
    /// PREFILL (hd=256, window=1024). Online softmax, register-resident O split
    /// across 8 simdgroups, no per-chunk PV tgmem round-trip. Full hd=512 path
    /// was 3× slower than E17 (disproven); at sliding shape flash is 2.4× faster
    /// than attention_mma2. Default ON (quality: smoketest 17/17 ×{7,42,123},
    /// longctx 4/4). `DGQ_FLASH_PREFILL=0` restores mma2. `bq`/`bk` = query-row
    /// tile / key-block streamed.
    pub flash_prefill: bool,
    /// `DGQ_ATTN_MMA_FULL_QK_ILP2` (E5 ILP2): split the 32-deep serial QK dot
    /// in `attention_mma_full` into two interleaved 16-deep accumulator chains
    /// (even/odd chunks), summed at the end. Halves the QK serial-dependency
    /// depth. Non-bit-identical (FP-associativity); quality-gated. Default OFF.
    pub attn_mma_full_qk_ilp2: bool,
    pub flash_prefill_bq: usize,
    pub flash_prefill_bk: usize,
    /// `DGQ_ATTN_TOPK` (E20): top-k sparse attention for full-layer PREFILL.
    /// Reuses E17's QK kernel (+FC32 u16 key plane), selects the top-k
    /// highest-scoring keys per (row, head) — exact by f32 score,
    /// deterministic — and gathers V at those indices. Non-bit-identical to
    /// dense. **Default ON (signed off 2026-07-16** after smoketest 17/17
    /// ×{7,42,123}, longctx 4/4, needle-exact 4/4 @121k with dynamic k, and
    /// 100k prefill within 2.5% of MLX; golden blessed on this default).
    /// `DGQ_ATTN_TOPK=0` restores dense E17.
    pub attn_topk: bool,
    /// `DGQ_ATTN_TOPK_K`: k per query row. The kernel's slot capacity K_PAD is
    /// compiled to `next_power_of_two(k)` (min 64, max 1024) via
    /// `attention_topk::tuned_source`, so any k in [1, 1024] is honored (task
    /// #95 unblocked the knob — it used to clamp silently to 64). Default 64.
    pub attn_topk_k: usize,
    /// `DGQ_ATTN_TOPK_DYN`: kv-adaptive k — per dispatch, k =
    /// clamp(t_total/128, 64, 512), i.e. a constant ~0.8% FRACTION of context
    /// instead of a constant count. Motivated by the E22-M0 finding that
    /// attention mass DIFFUSES with depth (row-top-64 mass: 41% @12k -> 13%
    /// @121k), while the findings-doc FLOP argument makes large k nearly free
    /// at long T (QK dominates). Overrides DGQ_ATTN_TOPK_K when set; K_PAD
    /// compiles at 512. **Default ON** (the ship policy — fixed k=64
    /// measurably drops the deepest needle at 121k; dyn matches dense 4/4).
    /// `DGQ_ATTN_TOPK_DYN=0` + `DGQ_ATTN_TOPK_K=n` restores fixed k.
    pub attn_topk_dyn: bool,
    /// `DGQ_ATTN_TOPK_DECODE`: top-k sparse attention for full-layer DENOISE
    /// (decode) dispatches — the same 3-kernel E20 pipeline as prefill top-k,
    /// causal=0, k from the same fixed/dyn policy. Denoise full-layer
    /// attention is issue-bound in `attention_mma_full` (~705 ms/layer @100k
    /// at canvas=256); the GEMM-decomp top-k runs the same work 3.1× faster
    /// (100k e2e: 4.42→1.84 s/step). Non-bit-identical; every denoise step
    /// compounds, so the gate is the multi-seed census + doc-QA ladder, not
    /// needles. **Default ON (signed off 2026-07-16** after smoketest 17/17
    /// ×{7,42,123}, longctx 4/4 kw 8/8 drift 0.0%, determinism ×8 short +
    /// ×3 @45.6k, census parity; golden blessed on this default).
    /// `DGQ_ATTN_TOPK_DECODE=0` restores dense mma_full denoise attention.
    pub attn_topk_decode: bool,
}

impl Default for PerfFlags {
    fn default() -> Self {
        Self {
            moe_fuse_gather: true,
            moe_prefill_block_m: 32,
            attn_mma: true,
            attn_mma_full: true,
            router_gemm: true,
            sc_sparse: true,
            attn_window: true,
            fused_algebra: true,
            partial_lm_head: true,
            encoder_gpu_moe: true,
            prefill_batch: true,
            prefill_mma: true,
            fast_block_extend: true,
            prefill_resident: true,
            attn_kv_block: 0,
            gemm_attn: true,
            gemm_attn_head_chunk: 16,
            gemm_attn_qk_bm: 64,
            gemm_attn_qk_bn: 64,
            gemm_attn_pv_bm: 64,
            gemm_attn_pv_bn: 64,
            gemm_attn_sm_tpg: 256,
            gemm_tune_bm: 64,
            gemm_tune_bn: 64,
            moe_sparse_bn: 128,
            flash_prefill: true,
            flash_prefill_bq: 16,
            flash_prefill_bk: 64,
            attn_mma_full_qk_ilp2: false,
            attn_topk: true,
            attn_topk_k: 64,
            attn_topk_dyn: true,
            attn_topk_decode: true,
        }
    }
}

/// Prefill-path selection.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillFlags {
    /// `DGQ_PREFILL_F16`: fp16 activation arena in fast prefill. Default OFF.
    pub f16: bool,
    /// `DGQ_PREFILL_KV_F32`: E14 f32 side ring for sliding KV. Default OFF.
    pub kv_f32: bool,
    /// `DGQ_FAST_PREFILL_MAX`: max prompt tokens the fast path handles (0 = uncapped).
    pub fast_prefill_max: usize,
    /// `DGQ_FAST_PREFILL`: force fast prefill on/off for all lengths; None = length band.
    pub fast_prefill_force: Option<bool>,
}

impl Default for PrefillFlags {
    fn default() -> Self {
        Self {
            f16: false,
            kv_f32: false,
            fast_prefill_max: 0,
            fast_prefill_force: None,
        }
    }
}

/// KV-cache configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct KvFlags {
    /// `DGQ_KV_Q8` override: `Some(true)` forces q8, `Some(false)` forces f16,
    /// `None` = the auto policy. Unified 2026-07-11: `kv_format` and
    /// `estimate_resident_bytes` previously disagreed on non-`1`/`0` values;
    /// both now honor this bool identically (nobody relied on the divergence).
    pub q8_override: Option<bool>,
    /// `DGQ_KV_REUSE`: cross-turn KV reuse. Default ON.
    pub reuse: bool,
    /// `DGQ_KV_MMAP`: file-backed KV buffer. Default OFF.
    pub mmap: bool,
    /// `DGQ_KV_MMAP_DIR`: backing-file dir (default system temp).
    pub mmap_dir: PathBuf,
}

impl Default for KvFlags {
    fn default() -> Self {
        Self {
            q8_override: None,
            reuse: true,
            mmap: false,
            mmap_dir: std::env::temp_dir(),
        }
    }
}

/// Server (serve) configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerFlags {
    /// `DGQ_CONV_CACHE_GB` → bytes for the RAM conversation-snapshot pool.
    pub conv_cache_bytes: usize,
    /// `DGQ_CONV_DISK_GB` → bytes for the SSD conversation-snapshot tier.
    pub conv_disk_bytes: usize,
    /// `DGQ_CONV_CACHE_DIR`: SSD tier dir (default per-process temp subdir).
    pub conv_cache_dir: PathBuf,
    /// `DGQ_TOOL_COMPACT`: enable the tool-output compactor. Default OFF.
    pub tool_compact: bool,
    /// `DGQ_TOOL_VALIDATE`: serve validates tool-call grammar on every reply
    /// and, on a malformed one, rewinds and regenerates at a bumped seed
    /// (the failed attempt never enters the causal context). Default OFF.
    pub tool_validate: bool,
    /// `DGQ_TOOL_COMPACT_THRESHOLD`: tokens above which a tool response compacts.
    pub tool_compact_threshold: usize,
    /// `DGQ_TOOL_COMPACT_DIR`: full-output store dir (default per-process temp subdir).
    pub tool_compact_dir: PathBuf,
}

impl Default for ServerFlags {
    fn default() -> Self {
        Self {
            conv_cache_bytes: 0,
            conv_disk_bytes: 0,
            conv_cache_dir: default_conv_cache_dir(),
            tool_compact: false,
            tool_validate: false,
            tool_compact_threshold: 384,
            tool_compact_dir: default_tool_compact_dir(),
        }
    }
}

fn default_conv_cache_dir() -> PathBuf {
    std::env::temp_dir().join(format!("dgq-conv-{}", std::process::id()))
}

fn default_tool_compact_dir() -> PathBuf {
    std::env::temp_dir().join(format!("dgq-toolout-{}", std::process::id()))
}

/// Debug / probe flags (all opt-in) plus the runtime `quiet` toggle.
#[derive(Debug, Clone, PartialEq)]
pub struct DebugFlags {
    /// `DGQ_QUIET`: suppress progress logs. Runtime-settable via [`set_quiet`].
    pub quiet: bool,
    /// Whether `DGQ_QUIET` was present in the env at load (so a runtime
    /// auto-quiet doesn't clobber an explicit user setting).
    pub quiet_from_env: bool,
    pub logits_finite_check: bool,
    pub logits_finite_samples: usize,
    pub trace_entropy: bool,
    pub trace_entropy_full: bool,
    pub final_entropy_log: bool,
    /// `DGQ_LOG_STEP_TEXT`: per-step decoded answer text. Default ON.
    pub step_text_log: bool,
    pub denoise_parity_log: bool,
    pub denoise_parity_positions: usize,
    pub log_early_stop: bool,
    pub sc_log: bool,
    pub trace_ranges: bool,
    pub mem_watch: bool,
    pub dump_kv_path: Option<String>,
    pub moe_route_ref_path: Option<String>,
    pub engine_layer_dump_path: Option<String>,
    pub engine_layer_dump_pos: usize,
    pub prefill_profile: bool,
    pub parity_debug: bool,
    /// `DGQ_KV_NOISE=<rel eps>`: perturb every live f16 KV value after prefill
    /// (task #67 sensitivity probe). None = off.
    pub kv_noise: Option<f32>,
    /// `DGQ_KV_RING_UNCAPPED`: linear (no-wrap) sliding KV storage (task #64
    /// ring-read isolation). Any set value enables.
    pub kv_ring_uncapped: bool,
    /// `DGQ_ARENA_F16_ALL`: build the MAIN activation set fp16 too (E11 bring-up
    /// bisect). Any set value enables.
    pub arena_f16_all: bool,
    /// `DGQ_METAL_PIPELINE_CACHE` raw override (unset = None): `0`/`false`
    /// disables the on-disk pipeline cache; any other non-empty value is used as
    /// the cache root dir. Consumed by pipeline_cache.rs via the accessors.
    pub metal_pipeline_cache: Option<String>,
}

impl Default for DebugFlags {
    fn default() -> Self {
        Self {
            quiet: false,
            quiet_from_env: false,
            logits_finite_check: false,
            logits_finite_samples: 4096,
            trace_entropy: false,
            trace_entropy_full: false,
            final_entropy_log: false,
            step_text_log: true,
            denoise_parity_log: false,
            denoise_parity_positions: 8,
            log_early_stop: false,
            sc_log: false,
            trace_ranges: false,
            mem_watch: false,
            dump_kv_path: None,
            moe_route_ref_path: None,
            engine_layer_dump_path: None,
            engine_layer_dump_pos: 129,
            prefill_profile: false,
            parity_debug: false,
            kv_noise: None,
            kv_ring_uncapped: false,
            arena_f16_all: false,
            metal_pipeline_cache: None,
        }
    }
}

/// The full runtime configuration. `Default` = the env-independent documented
/// defaults (what tests build on); [`from_env`](RuntimeConfig::from_env) overlays the
/// `DGQ_*` env vars onto those.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuntimeConfig {
    pub sampler: SamplerFlags,
    pub perf: PerfFlags,
    pub prefill: PrefillFlags,
    pub kv: KvFlags,
    pub server: ServerFlags,
    pub debug: DebugFlags,
}

impl RuntimeConfig {
    /// Parse every `DGQ_*` env flag once. The single source of env truth.
    pub fn from_env() -> Self {
        let parse_usize = |name: &str, default: usize| -> usize {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default)
        };
        let quiet_from_env = std::env::var_os("DGQ_QUIET").is_some();

        RuntimeConfig {
            sampler: SamplerFlags {
                freeze: env_on_if_one("DGQ_FREEZE"),
                denoiser_argmax: env_on_unless_zero("DGQ_DENOISER_ARGMAX"),
                early_stop_mean_ent: std::env::var("DGQ_EARLY_STOP_MEAN_ENT")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|&x| x >= 0.0)
                    .unwrap_or(0.05),
                empty_reply_retry: std::env::var("DGQ_EMPTY_REPLY_RETRY")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(3),
                ws_block_stop: env_on_if_one("DGQ_WS_BLOCK_STOP"),
                block_commit_max_ent: std::env::var("DGQ_BLOCK_COMMIT_MAX_ENT")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|&x| x >= 0.0)
                    .unwrap_or(0.0),
                block_commit_retry: std::env::var("DGQ_BLOCK_COMMIT_RETRY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1),
                force_canvas: std::env::var("DGQ_FORCE_CANVAS")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .filter(|&w| w >= 1),
            },
            perf: PerfFlags {
                moe_fuse_gather: env_on_unless_zero("DGQ_MOE_FUSE_GATHER"),
                moe_prefill_block_m: match std::env::var("DGQ_MOE_PREFILL_BM")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                {
                    Some(bm @ (64 | 128)) => bm,
                    _ => 32,
                },
                attn_mma: env_on_unless_zero("DGQ_ATTN_MMA"),
                attn_mma_full: env_on_unless_zero("DGQ_ATTN_MMA_FULL"),
                router_gemm: env_on_unless_zero("DGQ_ROUTER_GEMM"),
                sc_sparse: env_on_unless_zero("DGQ_SC_SPARSE"),
                attn_window: env_on_unless_zero("DGQ_ATTN_WINDOW"),
                fused_algebra: env_on_unless_zero("DGQ_FUSED_ALGEBRA"),
                partial_lm_head: env_on_unless_zero("DGQ_PARTIAL_LM_HEAD"),
                encoder_gpu_moe: env_on_unless_zero("DGQ_ENCODER_GPU_MOE"),
                prefill_batch: env_on_unless_zero("DGQ_PREFILL_BATCH"),
                prefill_mma: env_on_unless_zero("DGQ_PREFILL_MMA"),
                fast_block_extend: env_on_unless_zero("DGQ_FAST_BLOCK_EXTEND"),
                prefill_resident: env_on_unless_zero("DGQ_PREFILL_RESIDENT"),
                attn_kv_block: parse_usize("DGQ_ATTN_KV_BLOCK", 0),
                gemm_attn: env_on_unless_zero("DGQ_GEMM_ATTN"),
                gemm_attn_head_chunk: parse_usize("DGQ_GEMM_ATTN_HC", 16).max(1),
                gemm_attn_qk_bm: parse_usize("DGQ_GEMM_ATTN_QK_BM", 64).max(16),
                gemm_attn_qk_bn: parse_usize("DGQ_GEMM_ATTN_QK_BN", 64).max(16),
                gemm_attn_pv_bm: parse_usize("DGQ_GEMM_ATTN_PV_BM", 64).max(16),
                gemm_attn_pv_bn: parse_usize("DGQ_GEMM_ATTN_PV_BN", 64).max(16),
                gemm_attn_sm_tpg: parse_usize("DGQ_GEMM_ATTN_SM_TPG", 256).max(32),
                gemm_tune_bm: parse_usize("DGQ_GEMM_TUNE_BM", 64).max(16),
                gemm_tune_bn: parse_usize("DGQ_GEMM_TUNE_BN", 64).max(16),
                moe_sparse_bn: parse_usize("DGQ_MOE_SPARSE_BN", 128).max(16),
                flash_prefill: env_on_unless_zero("DGQ_FLASH_PREFILL"),
                flash_prefill_bq: parse_usize("DGQ_FLASH_PREFILL_BQ", 16).max(8),
                flash_prefill_bk: parse_usize("DGQ_FLASH_PREFILL_BK", 64).max(16),
                attn_mma_full_qk_ilp2: env_on_if_one("DGQ_ATTN_MMA_FULL_QK_ILP2"),
                attn_topk: env_on_unless_zero("DGQ_ATTN_TOPK"),
                attn_topk_k: parse_usize("DGQ_ATTN_TOPK_K", 64).max(1),
                attn_topk_dyn: env_on_unless_zero("DGQ_ATTN_TOPK_DYN"),
                attn_topk_decode: env_on_unless_zero("DGQ_ATTN_TOPK_DECODE"),
            },
            prefill: PrefillFlags {
                f16: env_on_if_one("DGQ_PREFILL_F16"),
                kv_f32: env_on_if_one("DGQ_PREFILL_KV_F32"),
                fast_prefill_max: parse_usize("DGQ_FAST_PREFILL_MAX", 0),
                fast_prefill_force: match std::env::var("DGQ_FAST_PREFILL").as_deref() {
                    Ok("1") | Ok("true") => Some(true),
                    Ok("0") | Ok("false") => Some(false),
                    _ => None,
                },
            },
            kv: KvFlags {
                q8_override: match std::env::var("DGQ_KV_Q8") {
                    Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => Some(true),
                    Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => Some(false),
                    _ => None, // unset or unrecognized → auto policy
                },
                reuse: env_on_unless_zero("DGQ_KV_REUSE"),
                mmap: env_on_if_one("DGQ_KV_MMAP"),
                mmap_dir: std::env::var_os("DGQ_KV_MMAP_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir),
            },
            server: ServerFlags {
                conv_cache_bytes: gib_bytes("DGQ_CONV_CACHE_GB"),
                conv_disk_bytes: gib_bytes("DGQ_CONV_DISK_GB"),
                conv_cache_dir: std::env::var_os("DGQ_CONV_CACHE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(default_conv_cache_dir),
                tool_compact: env_on_if_one("DGQ_TOOL_COMPACT"),
                tool_validate: env_on_if_one("DGQ_TOOL_VALIDATE"),
                tool_compact_threshold: std::env::var("DGQ_TOOL_COMPACT_THRESHOLD")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&t| t > 0)
                    .unwrap_or(384),
                tool_compact_dir: std::env::var_os("DGQ_TOOL_COMPACT_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(default_tool_compact_dir),
            },
            debug: DebugFlags {
                quiet: matches!(std::env::var("DGQ_QUIET").as_deref(), Ok("1"))
                    || std::env::var("DGQ_QUIET")
                        .map(|v| v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false),
                quiet_from_env,
                logits_finite_check: env_on_if_one("DGQ_CHECK_LOGITS"),
                logits_finite_samples: parse_usize("DGQ_CHECK_LOGITS_SAMPLES", 4096),
                trace_entropy: std::env::var("DGQ_TRACE_ENTROPY")
                    .map(|v| {
                        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full")
                    })
                    .unwrap_or(false),
                trace_entropy_full: std::env::var("DGQ_TRACE_ENTROPY")
                    .map(|v| v.eq_ignore_ascii_case("full"))
                    .unwrap_or(false),
                final_entropy_log: std::env::var("DGQ_LOG_FINAL_ENTROPY")
                    .map(|v| {
                        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full")
                    })
                    .unwrap_or(false),
                step_text_log: env_on_unless_zero("DGQ_LOG_STEP_TEXT"),
                denoise_parity_log: env_on_if_one("DGQ_LOG_DENOISE"),
                denoise_parity_positions: parse_usize("DGQ_LOG_DENOISE_POS", 8),
                log_early_stop: env_on_if_one("DGQ_LOG_EARLY_STOP"),
                sc_log: std::env::var("DGQ_LOG_SC").ok().as_deref() == Some("1"),
                trace_ranges: std::env::var("DGQ_TRACE_RANGES").is_ok(),
                mem_watch: env_on_if_one("DGQ_MEM_WATCH"),
                dump_kv_path: std::env::var("DGQ_DUMP_KV").ok(),
                moe_route_ref_path: std::env::var("DGQ_MOE_ROUTE_REF").ok(),
                engine_layer_dump_path: std::env::var("DGQ_ENGINE_LAYER_DUMP").ok(),
                engine_layer_dump_pos: parse_usize("DGQ_ENGINE_LAYER_POS", 129),
                prefill_profile: std::env::var("DGQ_PREFILL_PROFILE").is_ok(),
                parity_debug: std::env::var_os("DGQ_PARITY_DEBUG").is_some(),
                kv_noise: std::env::var("DGQ_KV_NOISE")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok()),
                kv_ring_uncapped: std::env::var("DGQ_KV_RING_UNCAPPED").is_ok(),
                arena_f16_all: std::env::var("DGQ_ARENA_F16_ALL").is_ok(),
                metal_pipeline_cache: std::env::var("DGQ_METAL_PIPELINE_CACHE").ok(),
            },
        }
    }
}

/// `DGQ_*_GB` gibibytes → bytes (0 / absent / non-positive → 0).
fn gib_bytes(name: &str) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|g| *g > 0.0)
        .map(|g| (g * 1024.0 * 1024.0 * 1024.0) as usize)
        .unwrap_or(0)
}

// ===========================================================================
// Process-global config + test override
// ===========================================================================

/// The shared, read-only base config — loaded from env exactly once. Every
/// thread sees the same base; a test override or [`set_quiet`] diverges only
/// the CURRENT THREAD (see `OVERRIDE`), so a stray override can never leak
/// across threads (libtest runs each test on its own thread) or corrupt other
/// readers. There is no process-global mutable config.
fn base() -> Arc<RuntimeConfig> {
    static BASE: OnceLock<Arc<RuntimeConfig>> = OnceLock::new();
    BASE.get_or_init(|| Arc::new(RuntimeConfig::from_env()))
        .clone()
}

thread_local! {
    /// Per-thread override of [`base`]. `None` = use the base. Installed by
    /// [`install_for_test`] (RAII) and [`set_quiet`]. Because it is
    /// thread-local, work that a `config()` reader spawns onto ANOTHER thread
    /// sees the base, not the override — every current caller reads on the same
    /// thread it configured (Metal dispatch is synchronous), so this is safe.
    static OVERRIDE: RefCell<Option<Arc<RuntimeConfig>>> = const { RefCell::new(None) };
}

/// The effective config on THIS thread: the thread-local override if one is
/// installed, else the shared env-loaded base.
pub fn config() -> Arc<RuntimeConfig> {
    OVERRIDE.with(|o| o.borrow().clone()).unwrap_or_else(base)
}

/// Runtime quiet toggle (this thread) — replaces the old `set_var("DGQ_QUIET")`.
/// Installs a thread-local override cloned from the current config with
/// `debug.quiet` flipped. The command handlers that call this run their
/// generation on the same thread, so the toggle is seen.
pub fn set_quiet(quiet: bool) {
    let mut cfg = (*config()).clone();
    cfg.debug.quiet = quiet;
    OVERRIDE.with(|o| *o.borrow_mut() = Some(Arc::new(cfg)));
}

/// Whether the user explicitly set `DGQ_QUIET` in the environment (so a caller
/// can auto-enable quiet without clobbering an explicit choice).
pub fn quiet_set_by_user() -> bool {
    config().debug.quiet_from_env
}

/// Install `cfg` as the config for the CURRENT THREAD's scope; the returned
/// guard restores the prior thread-local override on drop. TEST ONLY.
#[cfg(test)]
#[must_use = "the override is reverted when the guard drops"]
pub fn install_for_test(cfg: RuntimeConfig) -> TestGuard {
    let prev = OVERRIDE.with(|o| o.borrow_mut().replace(Arc::new(cfg)));
    TestGuard { prev }
}

/// RAII guard from [`install_for_test`]: restores the prior thread-local
/// override (possibly `None`) on drop.
#[cfg(test)]
pub struct TestGuard {
    prev: Option<Arc<RuntimeConfig>>,
}

#[cfg(test)]
impl Drop for TestGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        OVERRIDE.with(|o| *o.borrow_mut() = prev);
    }
}

// ===========================================================================
// Sampler semantics
// ===========================================================================

/// Hard-freeze of accepted canvas rows (`DGQ_FREEZE=1` re-enables the legacy
/// behavior). Default OFF since 2026-07-05 (user sign-off): the freeze was
/// PROVEN to be the flat-row wart driver (census 4/10 warty -> 0/10). OFF =
/// MLX/HF reference semantics (matches the CPU sampler in `sample.rs`): the
/// accept set is re-decided from fresh entropies every step, accepted rows
/// take that step's fresh denoiser token, dropped rows renoise, and the final
/// commit is the true full-canvas last-step argmax. ON additionally feeds the
/// partial-lm_head row skip (dormant at default).
pub fn freeze_enabled() -> bool {
    config().sampler.freeze
}

/// Commit the row argmax instead of the tempered categorical sample
/// (`DGQ_DENOISER_ARGMAX=0` restores HF categorical). Default ON since
/// 2026-07-05 (user sign-off): matches MLX's default user temperature=0
/// denoiser — the linear schedule temperature only shapes entropy and the SC
/// soft-embed, never the committed token. With no-freeze this is the
/// MLX-exact config (gate 16,16,11; census 0/10 warty).
pub fn denoiser_argmax_enabled() -> bool {
    config().sampler.denoiser_argmax
}

/// Entropy-only early stop (`DGQ_EARLY_STOP_MEAN_ENT=<nats>`; 0 disables).
/// A denoise block stops as soon as the full-canvas mean entropy falls below
/// this (after `min_early_stop_steps`), WITHOUT waiting for full argmax
/// stability. DEFAULT ON at 0.05 (user sign-off 2026-07-06; probe answers
/// byte-identical, multi-seed gate + wart census neutral).
pub fn early_stop_mean_ent() -> f32 {
    config().sampler.early_stop_mean_ent
}

/// Empty/degenerate-reply retry (E6). On the FIRST denoise block only, if the
/// committed canvas is degenerate, re-roll the initial canvas from the
/// advancing seed stream and re-run, up to N times. DEFAULT 3 (user sign-off
/// 2026-07-07; seed-123 gate answers 13→17, seeds 7/42 unchanged 17/17).
/// `DGQ_EMPTY_REPLY_RETRY=0` disables.
/// Non-convergence commit guard threshold (nats); 0.0 = guard disabled.
pub fn block_commit_max_ent() -> f32 {
    config().sampler.block_commit_max_ent
}

/// Fresh-noise re-rolls before the commit guard ends the turn.
pub fn block_commit_retry() -> u32 {
    config().sampler.block_commit_retry
}

pub fn empty_reply_retry() -> u32 {
    config().sampler.empty_reply_retry
}

/// Whitespace-collapse STOPGAP (`DGQ_WS_BLOCK_STOP=1` enables). A committed
/// block whose text is PURE whitespace (or all pad/filler) is dropped and the
/// turn ends. Default OFF while the newline-attractor is treated as an UNFIXED
/// BUG — a default-on stopper would truncate the evidence needed to root-cause
/// it. Flip to default ON (with gate + sign-off) only if intrinsic.
pub fn ws_block_stop_enabled() -> bool {
    config().sampler.ws_block_stop
}

/// TEST override for the denoise canvas width (E3/E6 shrink machinery). When
/// set (`DGQ_FORCE_CANVAS=64|128|...`), every denoise step runs at this active
/// canvas width instead of 256. `None` = normal (256). Clamped to [1, CANVAS]
/// at the use site. Diagnostic only; not a product flag.
pub fn force_canvas() -> Option<u32> {
    config().sampler.force_canvas
}

// ===========================================================================
// Production perf toggles (all default ON, opt-out for A/B triage)
// ===========================================================================

/// Fuse the MoE gather into the gate_up tunable expert GEMM (bit-identical,
/// ~28ms/step). `DGQ_MOE_FUSE_GATHER=0` opts out.
pub fn moe_fuse_gather_enabled() -> bool {
    config().perf.moe_fuse_gather
}

/// Weight-stationary expert GEMM for batched-prefill super-chunks
/// (`DGQ_MOE_PREFILL_BM=64|128` opts in; default 32 = OFF). BUILT + DISPROVEN
/// 2026-07-07 (ROADMAP E1): at M=1024 the expert GEMM is COMPUTE-bound at the
/// ~2.3 TF/s kernel wall, so cutting weight bytes can't speed it (64 = wash,
/// 128 = 3.6x slower via register spill). Machinery kept for re-tests.
pub fn moe_prefill_block_m() -> u32 {
    config().perf.moe_prefill_block_m
}

/// GQA matrix-unit attention on sliding layers (~3-4.5%/step). Non-bit-
/// identical (f16 MMA vs f32 scalar) but quality-neutral (multi-seed + MLX
/// bench). `DGQ_ATTN_MMA=0` restores the scalar kernel.
pub fn attn_mma_enabled() -> bool {
    config().perf.attn_mma
}

/// Matrix-unit attention on FULL layers (hd=512; attention -29% at kv=512).
/// Non-bit-identical but quality-neutral (sign-off 2026-06-28).
/// `DGQ_ATTN_MMA_FULL=0` restores the scalar kernel.
pub fn attn_mma_full_enabled() -> bool {
    config().perf.attn_mma_full
}

/// E17: route full-layer PREFILL attention through the GEMM decomposition
/// (attn_gemm_qk/softmax/pv) instead of attention_mma_full. ~1.78x on the
/// attention kernel; not bit-identical (full quality gate). `DGQ_GEMM_ATTN=1`.
pub fn gemm_attn_enabled() -> bool {
    config().perf.gemm_attn
}

/// E17a: Q heads processed per E17 dispatch batch (`DGQ_GEMM_ATTN_HC`, default
/// 4). Bounds the S/P prefill scratch; clamped to n_q_heads at dispatch.
pub fn gemm_attn_head_chunk() -> usize {
    config().perf.gemm_attn_head_chunk
}

/// Holistic sweep (task #88): dense tunable-GEMM tile (bm, bn); default 64x64.
pub fn gemm_tune_tile() -> (usize, usize) {
    let p = &config().perf;
    (p.gemm_tune_bm, p.gemm_tune_bn)
}

/// Holistic sweep (task #88): MoE-sparse N-tile; default 128 (block height
/// stays 32, baked into moe_bucket_fill).
pub fn moe_sparse_bn() -> usize {
    config().perf.moe_sparse_bn
}

/// `DGQ_ATTN_MMA_FULL_QK_ILP2`: split the QK dot in `attention_mma_full`
/// into two interleaved accumulator chains (FC31). Default OFF.
pub fn attn_mma_full_qk_ilp2() -> bool {
    config().perf.attn_mma_full_qk_ilp2
}

/// E20: top-k sparse attention for full-layer PREFILL. Default ON. Returns
/// (enabled, k).
pub fn attn_topk() -> (bool, usize) {
    let p = &config().perf;
    (p.attn_topk, p.attn_topk_k)
}

/// E20 enabled predicate (the only check `step_kernel` needs for routing).
pub fn attn_topk_enabled() -> bool {
    config().perf.attn_topk
}

/// E20 k per query row. Clamped to K_PAD at compile time host-side.
pub fn attn_topk_k() -> usize {
    config().perf.attn_topk_k.clamp(1, 1024)
}
/// Compile-time slot capacity for the top-k P/Idx planes: next power of two of
/// the requested k, floored at the shipped 64 (kernel default) and capped at
/// 1024. Pipeline compile (`tuned_source` AG_K_PAD) and the Rust-side plane
/// allocations must BOTH use this — same value, one source (#97's lesson).
pub fn attn_topk_k_pad() -> usize {
    if attn_topk_dyn() {
        return 512; // dyn k caps at 512 (see attn_topk_k_for)
    }
    attn_topk_k().next_power_of_two().clamp(64, 1024)
}
/// `DGQ_ATTN_TOPK_DYN` (kv-adaptive k).
pub fn attn_topk_dyn() -> bool {
    config().perf.attn_topk_dyn
}
/// `DGQ_ATTN_TOPK_DECODE`: top-k sparse attention on full-layer DENOISE
/// dispatches. Default ON (signed off 2026-07-16); =0 restores dense.
pub fn attn_topk_decode_enabled() -> bool {
    config().perf.attn_topk_decode
}
/// Effective k for a dispatch at `t_total` keys: fixed DGQ_ATTN_TOPK_K, or —
/// with DGQ_ATTN_TOPK_DYN — a constant ~0.8% fraction of context,
/// clamp(t_total/128, 64, 512).
pub fn attn_topk_k_for(t_total: usize) -> usize {
    if attn_topk_dyn() {
        (t_total / 128).clamp(64, 512)
    } else {
        attn_topk_k()
    }
}

/// E18 fused flash prefill. `.0` = enabled, `.1` = BQ (query-row tile), `.2` =
/// BK (streamed key block). Default off / 16 / 64.
pub fn flash_prefill() -> (bool, usize, usize) {
    let p = &config().perf;
    (p.flash_prefill, p.flash_prefill_bq, p.flash_prefill_bk)
}

/// E17 tunable tile config (task #87). Defaults reproduce the shipped
/// 64x64/256 kernel. Returns (qk_bm, qk_bn, pv_bm, pv_bn, sm_tpg).
pub fn gemm_attn_tile() -> (usize, usize, usize, usize, usize) {
    let p = &config().perf;
    (
        p.gemm_attn_qk_bm,
        p.gemm_attn_qk_bn,
        p.gemm_attn_pv_bm,
        p.gemm_attn_pv_bn,
        p.gemm_attn_sm_tpg,
    )
}

/// Router-as-GEMM (~30ms/step). Non-bit-identical (near-tie expert flips =
/// trajectory re-roll, not bias); accepted 2026-07-02 on multi-seed evidence
/// and the gate was re-baselined with it. `DGQ_ROUTER_GEMM=0` restores the
/// serial router kernel.
pub fn router_gemm_enabled() -> bool {
    config().perf.router_gemm
}

/// Sparse SC softembed (top-survivors gather instead of the full vocab GEMM).
/// APPROXIMATE (drops prob tail below e^-10 of row max); signed off ~16%/step,
/// output-level equivalent to MLX-4bit. Needs bf16 embed (gated at call
/// site). `DGQ_SC_SPARSE=0` uses the exact chunked path.
pub fn sc_sparse_enabled() -> bool {
    config().perf.sc_sparse
}

/// Sliding-window masking on sliding-attention layers (matches MLX
/// `_make_decoder_masks`; bit-identical within the window, more correct AND
/// O(window) beyond it). `DGQ_ATTN_WINDOW=0` restores unwindowed for A/B.
pub fn attn_window_enabled() -> bool {
    config().perf.attn_window
}

/// Algebraic fusion: QKV and dense gate+up as one stacked GEMM (bit-identical,
/// ~1.3%/step). `DGQ_FUSED_ALGEBRA=0` disables both for A/B.
pub fn fused_algebra_enabled() -> bool {
    config().perf.fused_algebra
}

pub fn fused_qkv_enabled() -> bool {
    fused_algebra_enabled()
}

pub fn fused_gate_up_enabled() -> bool {
    fused_algebra_enabled()
}

/// Partial lm_head: skip vocab-GEMM rows for frozen canvas rows. Only
/// meaningful under `DGQ_FREEZE=1` (no rows freeze at default semantics).
/// `DGQ_PARTIAL_LM_HEAD=0` opts out.
pub fn partial_lm_head_enabled() -> bool {
    config().perf.partial_lm_head
}

/// Engine (f32) prefill runs MoE expert GEMMs on GPU (`=0` = CPU mirror, the
/// exact `.dgq` oracle — the triage lever for engine-prefill drift).
pub fn encoder_gpu_moe_enabled() -> bool {
    config().perf.encoder_gpu_moe
}

/// Cross-turn KV reuse in chat (`DGQ_KV_REUSE=0` disables): keep the causal KV
/// for the longest common prefix of the prior sequence and the new prompt, and
/// prefill only the delta at that offset. NOT byte-identical to a full
/// re-prefill but quality-equivalent (same class as fast-vs-engine prefill).
pub fn kv_reuse_enabled() -> bool {
    config().kv.reuse
}

/// Byte budget for the server's multi-conversation KV snapshot pool
/// (`DGQ_CONV_CACHE_GB`). Default 0 = minimal (single hot conversation).
pub fn conv_cache_bytes() -> usize {
    config().server.conv_cache_bytes
}

/// Disk byte budget for the server's SSD conversation-snapshot tier
/// (`DGQ_CONV_DISK_GB`). Default 0 = no disk tier.
pub fn conv_disk_bytes() -> usize {
    config().server.conv_disk_bytes
}

/// Directory for the SSD conversation-snapshot tier (`DGQ_CONV_CACHE_DIR`).
pub fn conv_cache_dir() -> PathBuf {
    config().server.conv_cache_dir.clone()
}

/// `DGQ_TOOL_VALIDATE=1`: serve rewinds + regenerates (bumped seed) when a
/// reply's tool-call grammar is malformed. Opt-IN and default OFF
/// (quality-affecting: a retry replaces the reply). Equivalent to
/// `serve --tool-validate`.
pub fn tool_validate_enabled() -> bool {
    config().server.tool_validate
}

/// `DGQ_TOOL_COMPACT=1`: enable the serve tool-output compactor (KV rewinder).
/// Opt-IN and default OFF (quality-affecting, not yet gate-signed-off).
/// Equivalent to `serve --tool-compact`.
pub fn tool_compact_enabled() -> bool {
    config().server.tool_compact
}

/// Token threshold above which a tool response is compacted
/// (`DGQ_TOOL_COMPACT_THRESHOLD`). Default 384.
pub fn tool_compact_threshold() -> usize {
    config().server.tool_compact_threshold
}

/// Directory for full tool outputs stored by the compactor
/// (`DGQ_TOOL_COMPACT_DIR`).
pub fn tool_compact_dir() -> PathBuf {
    config().server.tool_compact_dir.clone()
}

/// `DGQ_KV_MMAP=1`: back the session KV cache with a `MAP_SHARED` temp-file mmap
/// (wrapped no-copy as the Metal buffer). Opt-in experiment. Default OFF.
pub fn kv_mmap() -> bool {
    config().kv.mmap
}

/// Directory for the `DGQ_KV_MMAP` backing file. Defaults to the system temp dir.
pub fn kv_mmap_dir() -> PathBuf {
    config().kv.mmap_dir.clone()
}

// ===========================================================================
// GPU working-set cap (a device fact, not env config)
// ===========================================================================

/// GPU working-set cap (Metal `recommendedMaxWorkingSetSize`), captured once
/// at MetalContext init so the pure `kv_q8` policy can scale to the device's
/// RAM. 0 (unset, e.g. CPU-only tests) → the q8 auto-policy stays off.
static GPU_WORKING_SET_BYTES: OnceLock<u64> = OnceLock::new();

/// Record the device working-set cap (idempotent; first writer wins).
pub fn set_gpu_working_set_cap(bytes: u64) {
    let _ = GPU_WORKING_SET_BYTES.set(bytes);
}

/// q8 KV cache storage. Group-32 symmetric i8 + f16 scales (0.92% rel-RMS),
/// halves KV memory. **AUTO at long context since 2026-07-07**: enabled when
/// the estimated f16 resident at `max_seq` would approach the GPU working-set
/// cap, so very long sessions stay resident instead of swapping. `DGQ_KV_Q8=1`
/// forces on, `=0` forces off. Format is fixed per session at open; every KV
/// writer/reader compiles a matching function-constant variant.
pub fn kv_format(max_seq: usize) -> crate::shaders::kv_quant::KvFormat {
    use crate::shaders::kv_quant::KvFormat;
    // Explicit override wins.
    if let Some(force_q8) = config().kv.q8_override {
        return if force_q8 {
            KvFormat::Q8
        } else {
            KvFormat::F16
        };
    }
    // Auto: q8 when f16 resident would exceed a safe fraction of the cap.
    let Some(&cap) = GPU_WORKING_SET_BYTES.get() else {
        return KvFormat::F16; // cap unknown (tests / CPU) → keep f16
    };
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const F16_BASE_BYTES: f64 = 19.73 * GIB; // non-KV resident (weights+arena)
    const F16_KV_BYTES_PER_TOKEN: f64 = 19.0 * 1024.0; // f16 KV linear growth
    const SAFE_FRACTION: f64 = 0.85; // switch before the >90% swap regime
    let f16_resident = F16_BASE_BYTES + F16_KV_BYTES_PER_TOKEN * max_seq as f64;
    if f16_resident > SAFE_FRACTION * cap as f64 {
        KvFormat::Q8
    } else {
        KvFormat::F16
    }
}

// Resident-memory model for the q4emb checkpoint (measured 2026-07-07; see
// `kv_format`): resident ≈ 19.73 GiB weights+arena + KV linear in tokens.
const CTX_BASE_BYTES: f64 = 19.73 * 1024.0 * 1024.0 * 1024.0;
const CTX_F16_KV_PER_TOKEN: f64 = 19.0 * 1024.0;
/// q8 KV (half) auto-enables once f16 resident exceeds this fraction of budget
/// — matches the `kv_format` auto policy so the guard mirrors the engine.
const CTX_Q8_AUTO_FRACTION: f64 = 0.85;
/// Safe resident ceiling fraction of the budget: past this Metal swaps / the
/// KV alloc fails.
pub const CTX_SAFE_FRACTION: f64 = 0.90;

/// Estimated GPU-resident bytes at `max_seq`, mirroring what the engine will
/// actually allocate: q8 KV (≈half) auto-enables once the f16 resident would
/// exceed `CTX_Q8_AUTO_FRACTION` of `budget_bytes`. `DGQ_KV_Q8` forces the
/// format either way — the SAME rule as `kv_format`.
pub fn estimate_resident_bytes(max_seq: usize, budget_bytes: u64) -> u64 {
    let f16 = CTX_BASE_BYTES + CTX_F16_KV_PER_TOKEN * max_seq as f64;
    let q8 = match config().kv.q8_override {
        Some(force_q8) => force_q8,
        None => budget_bytes > 0 && f16 > CTX_Q8_AUTO_FRACTION * budget_bytes as f64,
    };
    let per_tok = if q8 {
        CTX_F16_KV_PER_TOKEN * 0.5
    } else {
        CTX_F16_KV_PER_TOKEN
    };
    (CTX_BASE_BYTES + per_tok * max_seq as f64) as u64
}

/// Guard for a user-requested context length: `Some((needed, ceiling))` bytes
/// when the estimated resident at `max_seq` would exceed `CTX_SAFE_FRACTION` of
/// `budget_bytes`. `None` = fits, or `budget_bytes == 0` (unknown).
pub fn ctx_over_budget(max_seq: usize, budget_bytes: u64) -> Option<(u64, u64)> {
    if budget_bytes == 0 {
        return None;
    }
    let ceiling = (CTX_SAFE_FRACTION * budget_bytes as f64) as u64;
    let needed = estimate_resident_bytes(max_seq, budget_bytes);
    (needed > ceiling).then_some((needed, ceiling))
}

/// Largest `--ctx` (max_seq) whose estimate stays under the safe ceiling, for
/// the "reduce to <= N" hint. At the ceiling ctx is large, so q8 KV is
/// auto-selected. 0 when the budget can't even hold the base weights.
pub fn max_feasible_ctx(budget_bytes: u64) -> usize {
    let headroom = CTX_SAFE_FRACTION * budget_bytes as f64 - CTX_BASE_BYTES;
    (headroom / (CTX_F16_KV_PER_TOKEN * 0.5)).max(0.0) as usize
}

/// KV block size for sequential-block full-attention (`DGQ_ATTN_KV_BLOCK`;
/// DEFAULT 0 = OFF). BIT-IDENTICAL to the monolithic pass but DISPROVEN AS A
/// PERF LEVER 2026-07-06 (the SLC already serves the shared key stream in
/// near-lockstep). Kept as scaffolding for a future parallel-split/quantized-KV pass.
pub fn attn_kv_block() -> usize {
    config().perf.attn_kv_block
}

/// Batched prefill super-chunks (`DGQ_PREFILL_BATCH=0` restores plain 256-token
/// chunks): up to PREFILL_SUBS causal sub-chunks run as ONE forward. Bit-
/// identical to sequential chunks.
pub fn prefill_batch_enabled() -> bool {
    config().perf.prefill_batch
}

/// MMA attention in the fast prefill (`DGQ_PREFILL_MMA=0` restores the scalar
/// prefill attention). Scalar prefill is O(kv_len) serial per query = unusable
/// at long context.
pub fn prefill_mma_enabled() -> bool {
    config().perf.prefill_mma
}

/// Fast (quantized) between-block KV extend (`DGQ_FAST_BLOCK_EXTEND=0` restores
/// the f32 engine extend). After a canvas block commits, its 256 tokens are
/// causally extended via `prefill_chunks_from` (~0.85s) instead of the engine
/// (~10s/block). Same quality class as fast-vs-engine prefill.
pub fn fast_block_extend_enabled() -> bool {
    config().perf.fast_block_extend
}

/// Buffer-resident engine prefill (merged batches, no CPU round-trips).
/// `DGQ_PREFILL_RESIDENT=0` restores the legacy per-batch readback path.
pub fn prefill_resident_enabled() -> bool {
    config().perf.prefill_resident
}

// ===========================================================================
// Prefill path selection
// ===========================================================================

/// Min prompt length (tokens) for the fast quantized prefill under the
/// heuristic default. Short prompts stay on the accurate f32 engine.
pub const FAST_PREFILL_MIN_TOKENS: usize = 256;

/// E11 (DISPROVEN as the long-prompt fix; kept as a diagnostic): fast prefill
/// runs on fp16 activation-arena pipelines instead of bf16. The real cause of
/// task #64 was the spurious encoder RmsNormHidden (fixed 2b0d12b). Opt-in.
pub fn prefill_f16_enabled() -> bool {
    config().prefill.f16
}

/// E14: during fast prefill, sliding layers write/read K/V through an f32 side
/// ring (window-sized) instead of the f16 monolithic cache — kills the
/// chunk-boundary rounding that compounds through the causal chain and
/// destroyed long-prompt comprehension (task #64/#67). Requires the MMA
/// prefill attention (default).
pub fn prefill_kv_f32_enabled() -> bool {
    config().prefill.kv_f32
}

/// Max prompt length (tokens) the fast quantized prefill handles
/// (`DGQ_FAST_PREFILL_MAX`; 0 = no cap). DEFAULT 0 (uncapped) since the task
/// #64/#68 root-cause fix (2b0d12b, 2026-07-10): the length-dependent
/// comprehension collapse was a spurious encoder-pass RmsNormHidden, not
/// accumulating precision error.
pub fn fast_prefill_max_tokens() -> usize {
    config().prefill.fast_prefill_max
}

/// Fast monolithic prefill (quantized + causal) vs the f32 engine for a
/// `prompt_len`-token prompt. `DGQ_FAST_PREFILL=1|0` forces on/off for all
/// lengths; unset uses the length band (see [`fast_prefill_max_tokens`]).
pub fn should_fast_prefill(prompt_len: usize) -> bool {
    match config().prefill.fast_prefill_force {
        Some(v) => v,
        None => {
            let max = fast_prefill_max_tokens();
            prompt_len > FAST_PREFILL_MIN_TOKENS && (max == 0 || prompt_len <= max)
        }
    }
}

// ===========================================================================
// Debug / probe flags (all opt-in)
// ===========================================================================

/// Suppress step-generate/prefill progress logs (`DGQ_QUIET=1`).
pub fn progress_enabled() -> bool {
    !config().debug.quiet
}

/// Opt-in logits NaN guard on the generate hot path (`DGQ_CHECK_LOGITS=1`).
pub fn logits_finite_check_enabled() -> bool {
    config().debug.logits_finite_check
}

/// Sample count for the logits NaN guard (`DGQ_CHECK_LOGITS_SAMPLES`).
pub fn logits_finite_sample_count() -> usize {
    config().debug.logits_finite_samples
}

/// Per-step entropy traces in generate output (`DGQ_TRACE_ENTROPY=1|full`).
pub fn trace_entropy_enabled() -> bool {
    config().debug.trace_entropy
}

/// Full-canvas (vs 16-prefix) entropy trace payloads (`DGQ_TRACE_ENTROPY=full`).
pub fn trace_entropy_full() -> bool {
    config().debug.trace_entropy_full
}

/// Per-position entropy at end of a denoise block (`DGQ_LOG_FINAL_ENTROPY=1`).
pub fn final_entropy_log_enabled() -> bool {
    config().debug.final_entropy_log
}

/// Decode the answer-region argmax each denoise step and show it in the step
/// progress line. Log-only — generation is bit-identical either way. DEFAULT ON
/// since 2026-07-09. `DGQ_LOG_STEP_TEXT=0` disables.
pub fn step_text_log_enabled() -> bool {
    config().debug.step_text_log
}

/// GPU-vs-CPU accept-mask parity log per step (`DGQ_LOG_DENOISE=1`).
pub fn denoise_parity_log_enabled() -> bool {
    config().debug.denoise_parity_log
}

/// Positions printed by the denoise parity log (`DGQ_LOG_DENOISE_POS`).
pub fn denoise_parity_log_positions() -> usize {
    config().debug.denoise_parity_positions
}

/// Early-stop decision log + GPU/CPU mismatch check (`DGQ_LOG_EARLY_STOP=1`).
pub fn log_early_stop_enabled() -> bool {
    config().debug.log_early_stop
}

/// SC soft-embed stage logging (`DGQ_LOG_SC=1`).
pub fn sc_log_enabled() -> bool {
    config().debug.sc_log
}

/// Per-stage bf16 activation-range trace (`DGQ_TRACE_RANGES=1`): answers "does
/// any activation exceed f16's 65504 range?" before precision experiments.
pub fn trace_ranges_enabled() -> bool {
    config().debug.trace_ranges
}

/// UMA memory-pressure watch (`DGQ_MEM_WATCH=1`): per-section working-set /
/// swap-delta / pressure report; prints a LOUD "TIMINGS SUSPECT" when swap grew
/// or allocation exceeds 90% of the working-set cap.
pub fn mem_watch_enabled() -> bool {
    config().debug.mem_watch
}

/// Dump per-layer prefill KV to `<path>` (`DGQ_DUMP_KV=<path>`).
pub fn dump_kv_path() -> Option<String> {
    config().debug.dump_kv_path.clone()
}

/// Dump MoE routes to `<path>` (`DGQ_MOE_ROUTE_REF=<path>`).
pub fn moe_route_ref_path() -> Option<String> {
    config().debug.moe_route_ref_path.clone()
}

/// Engine per-layer hidden dump (`DGQ_ENGINE_LAYER_DUMP=<json path>`).
pub fn engine_layer_dump_path() -> Option<String> {
    config().debug.engine_layer_dump_path.clone()
}

/// Canvas row probed by the engine layer dump (`DGQ_ENGINE_LAYER_POS`).
pub fn engine_layer_dump_pos() -> usize {
    config().debug.engine_layer_dump_pos
}

/// Per-layer engine prefill phase timings (`DGQ_PREFILL_PROFILE=1`).
pub fn prefill_profile_enabled() -> bool {
    config().debug.prefill_profile
}

/// Extra step-parity diagnostics (argmax agreement) (`DGQ_PARITY_DEBUG=1`).
pub fn parity_debug_enabled() -> bool {
    config().debug.parity_debug
}

/// KV-noise sensitivity probe rel-eps (`DGQ_KV_NOISE=<f32>`; None = off).
pub fn kv_noise() -> Option<f32> {
    config().debug.kv_noise
}

/// Linear (no-wrap) sliding KV storage (`DGQ_KV_RING_UNCAPPED`).
pub fn kv_ring_uncapped_enabled() -> bool {
    config().debug.kv_ring_uncapped
}

/// Build the MAIN activation set fp16 too (`DGQ_ARENA_F16_ALL`).
pub fn arena_f16_all_enabled() -> bool {
    config().debug.arena_f16_all
}

/// On-disk Metal pipeline cache enabled (`DGQ_METAL_PIPELINE_CACHE`; `0`/`false`
/// disables, unset or any other value enables).
pub fn metal_pipeline_cache_enabled() -> bool {
    match config().debug.metal_pipeline_cache.as_deref() {
        Some(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        None => true,
    }
}

/// Explicit Metal pipeline-cache root dir from `DGQ_METAL_PIPELINE_CACHE` when
/// it names a (non-empty, non-off) path; `None` falls back to the XDG/HOME
/// default (kept in pipeline_cache.rs — those are system env, not DGQ config).
pub fn metal_pipeline_cache_dir_override() -> Option<PathBuf> {
    config()
        .debug
        .metal_pipeline_cache
        .as_deref()
        .filter(|v| !v.is_empty() && *v != "0" && !v.eq_ignore_ascii_case("false"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod ctx_budget_tests {
    use super::*;
    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn unknown_budget_never_trips() {
        assert!(ctx_over_budget(1_000_000, 0).is_none());
    }

    #[test]
    fn small_ctx_fits_large_ctx_refused_on_36gib() {
        let budget = 26 * GIB; // ~72% of a 36 GiB machine
        assert!(ctx_over_budget(8_192, budget).is_none(), "8k must fit");
        assert!(
            ctx_over_budget(900_000, budget).is_some(),
            "900k must be refused"
        );
    }

    #[test]
    fn max_feasible_is_a_real_boundary() {
        let budget = 26 * GIB;
        let n = max_feasible_ctx(budget);
        assert!(n > 0);
        // At the reported max it fits; just past it, it's refused.
        assert!(ctx_over_budget(n.saturating_sub(1024), budget).is_none());
        assert!(ctx_over_budget(n + 50_000, budget).is_some());
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn default_matches_documented_shipped_defaults() {
        let c = RuntimeConfig::default();
        // Sampler sign-offs.
        assert!(!c.sampler.freeze);
        assert!(c.sampler.denoiser_argmax);
        assert_eq!(c.sampler.early_stop_mean_ent, 0.05);
        assert_eq!(c.sampler.empty_reply_retry, 3);
        // Perf toggles ship ON.
        assert!(c.perf.attn_mma && c.perf.fused_algebra);
        assert_eq!(c.perf.moe_prefill_block_m, 32);
        // Prefill uncapped, KV reuse on, tool-compact off.
        assert_eq!(c.prefill.fast_prefill_max, 0);
        assert!(c.kv.reuse);
        assert!(!c.server.tool_compact);
        assert_eq!(c.server.tool_compact_threshold, 384);
    }

    #[test]
    fn install_for_test_overrides_and_restores() {
        let before = freeze_enabled();
        {
            let mut cfg = RuntimeConfig::default();
            cfg.sampler.freeze = !before;
            let _g = install_for_test(cfg);
            assert_eq!(freeze_enabled(), !before, "override takes effect");
        }
        assert_eq!(freeze_enabled(), before, "prior config restored on drop");
    }

    #[test]
    fn set_quiet_flips_progress() {
        let _g = install_for_test(RuntimeConfig::default());
        set_quiet(true);
        assert!(!progress_enabled());
        set_quiet(false);
        assert!(progress_enabled());
    }

    #[test]
    fn kv_q8_override_is_consistent_across_consumers() {
        use crate::shaders::kv_quant::KvFormat;
        let budget = 26 * 1024 * 1024 * 1024u64;
        let q8_bytes = {
            let mut cfg = RuntimeConfig::default();
            cfg.kv.q8_override = Some(true);
            let _g = install_for_test(cfg);
            assert!(matches!(kv_format(8192), KvFormat::Q8), "force-on → q8");
            estimate_resident_bytes(100_000, budget)
        };
        let f16_bytes = {
            let mut cfg = RuntimeConfig::default();
            cfg.kv.q8_override = Some(false);
            let _g = install_for_test(cfg);
            assert!(matches!(kv_format(8192), KvFormat::F16), "force-off → f16");
            estimate_resident_bytes(100_000, budget)
        };
        // Both consumers honor the SAME override: q8 halves per-token KV.
        assert!(q8_bytes < f16_bytes, "q8 override must shrink the estimate");
    }

    #[test]
    fn metal_pipeline_cache_dual_semantics() {
        {
            let _g = install_for_test(RuntimeConfig::default());
            assert!(metal_pipeline_cache_enabled(), "unset → enabled");
            assert!(metal_pipeline_cache_dir_override().is_none());
        }
        for off in ["0", "false", "FALSE"] {
            let mut c = RuntimeConfig::default();
            c.debug.metal_pipeline_cache = Some(off.to_string());
            let _g = install_for_test(c);
            assert!(!metal_pipeline_cache_enabled(), "{off} disables");
            assert!(metal_pipeline_cache_dir_override().is_none());
        }
        let mut c = RuntimeConfig::default();
        c.debug.metal_pipeline_cache = Some("/tmp/pc".to_string());
        let _g = install_for_test(c);
        assert!(metal_pipeline_cache_enabled(), "a path enables");
        assert_eq!(
            metal_pipeline_cache_dir_override(),
            Some(PathBuf::from("/tmp/pc"))
        );
    }

    #[test]
    fn override_is_thread_local() {
        // An override installed here must NOT be visible on another thread
        // (which sees the shared base) — the isolation the design guarantees.
        let mut cfg = RuntimeConfig::default();
        cfg.sampler.freeze = true;
        let _g = install_for_test(cfg);
        assert!(freeze_enabled(), "override visible on this thread");
        let base_freeze = std::thread::spawn(freeze_enabled).join().unwrap();
        assert!(!base_freeze, "other thread sees the base, not the override");
    }

    #[test]
    fn should_fast_prefill_respects_force_and_band() {
        let mut cfg = RuntimeConfig::default();
        cfg.prefill.fast_prefill_force = Some(false);
        {
            let _g = install_for_test(cfg.clone());
            assert!(!should_fast_prefill(10_000), "force-off wins at any length");
        }
        cfg.prefill.fast_prefill_force = None;
        cfg.prefill.fast_prefill_max = 0;
        let _g = install_for_test(cfg);
        assert!(
            !should_fast_prefill(FAST_PREFILL_MIN_TOKENS),
            "at floor = engine"
        );
        assert!(
            should_fast_prefill(FAST_PREFILL_MIN_TOKENS + 1),
            "above floor = fast"
        );
    }
}
