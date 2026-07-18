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
/// Env access for flag parsing, mirroring `std::env` signatures. `REAL_ENV`
/// reads the process env; `EMPTY_ENV` backs `Default`, so the documented
/// defaults and unset-env parsing CANNOT drift apart
/// (`default_equals_empty_env_parse` pins the equivalence).
struct EnvReader {
    real: bool,
}
const REAL_ENV: EnvReader = EnvReader { real: true };
const EMPTY_ENV: EnvReader = EnvReader { real: false };

impl EnvReader {
    fn var(&self, name: &str) -> Result<String, std::env::VarError> {
        if self.real {
            std::env::var(name)
        } else {
            Err(std::env::VarError::NotPresent)
        }
    }
    fn var_os(&self, name: &str) -> Option<std::ffi::OsString> {
        if self.real {
            std::env::var_os(name)
        } else {
            None
        }
    }
    /// Set-and-not-"0"/"false" — the default-ON pattern.
    fn on_unless_zero(&self, name: &str) -> bool {
        match self.var(name) {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    /// `=1`/`true` enables; anything else (or unset) leaves it OFF.
    fn on_if_one(&self, name: &str) -> bool {
        match self.var(name) {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => false,
        }
    }
    fn gib_bytes(&self, name: &str) -> usize {
        self.var(name)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|g| *g > 0.0)
            .map(|g| (g * 1024.0 * 1024.0 * 1024.0) as usize)
            .unwrap_or(0)
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
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).sampler
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
    /// 2026-07-13. Denoise reuses the qk stage through E20 top-k (DGQ_ATTN_TOPK_DECODE). `=0` restores mma_full.
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
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).perf
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
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).prefill
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
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).kv
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
    /// `DGQ_PREFILL_STATUS`: streamed requests emit a synthetic
    /// `reasoning_content` status line + elapsed heartbeat while a large
    /// prompt delta prefills (the dry-start silence). Default ON; `=0` off.
    pub prefill_status: bool,
    /// `DGQ_CONTINUE_PAST_STOP`: serve's old defer bet — a stop token inside
    /// an unfinished tool reply continues generation into the next block.
    /// Default OFF since the strain battery showed the premature stop is a
    /// degradation symptom and the forced continuation floods. Opt-in.
    pub continue_past_stop: bool,
    /// `DGQ_TOOL_COMPACT`: enable the tool-output compactor. Default OFF.
    pub tool_compact: bool,
    /// `DGQ_TOOL_REPAIR`: serve's model-guided tool-call repair — an invalid
    /// tool reply gets a synthetic error tool-response, the model emits a
    /// corrected call, and the corrupt exchange is rewound out of KV.
    /// Default OFF (quality-affecting; field sign-off pending).
    pub tool_repair: bool,
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
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).server
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
    /// `DGQ_KV_RING`: sliding-layer KV ring slots (power of two, min 2048 =
    /// window-1 + canvas live positions). Default 4096: the O(1)-truncate
    /// slack is ring − CANVAS − (window−1) — 769 slots at 2048, 2817 at
    /// 4096 — so the bigger ring turns typical thought-strip finalize
    /// truncates and validator/triage rewinds into O(1) instead of a ring
    /// rebuild (≈ a full re-prefill of the kept prefix), for a fixed
    /// few-hundred-MB of extra sliding KV independent of context length.
    pub kv_ring_slots: usize,
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
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).debug
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
        Self::from_reader(&REAL_ENV)
    }

    /// The one parse. `Default` goes through this with [`EMPTY_ENV`].
    fn from_reader(r: &EnvReader) -> Self {
        let parse_usize = |name: &str, default: usize| -> usize {
            r.var(name)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default)
        };
        let quiet_from_env = r.var_os("DGQ_QUIET").is_some();

        RuntimeConfig {
            sampler: SamplerFlags {
                freeze: r.on_if_one("DGQ_FREEZE"),
                denoiser_argmax: r.on_unless_zero("DGQ_DENOISER_ARGMAX"),
                early_stop_mean_ent: r
                    .var("DGQ_EARLY_STOP_MEAN_ENT")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|&x| x >= 0.0)
                    .unwrap_or(0.05),
                empty_reply_retry: r
                    .var("DGQ_EMPTY_REPLY_RETRY")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(3),
                ws_block_stop: r.on_if_one("DGQ_WS_BLOCK_STOP"),
                block_commit_max_ent: r
                    .var("DGQ_BLOCK_COMMIT_MAX_ENT")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|&x| x >= 0.0)
                    .unwrap_or(0.0),
                block_commit_retry: r
                    .var("DGQ_BLOCK_COMMIT_RETRY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1),
                force_canvas: r
                    .var("DGQ_FORCE_CANVAS")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .filter(|&w| w >= 1),
            },
            perf: PerfFlags {
                moe_fuse_gather: r.on_unless_zero("DGQ_MOE_FUSE_GATHER"),
                moe_prefill_block_m: match r
                    .var("DGQ_MOE_PREFILL_BM")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                {
                    Some(bm @ (64 | 128)) => bm,
                    _ => 32,
                },
                attn_mma: r.on_unless_zero("DGQ_ATTN_MMA"),
                attn_mma_full: r.on_unless_zero("DGQ_ATTN_MMA_FULL"),
                router_gemm: r.on_unless_zero("DGQ_ROUTER_GEMM"),
                sc_sparse: r.on_unless_zero("DGQ_SC_SPARSE"),
                attn_window: r.on_unless_zero("DGQ_ATTN_WINDOW"),
                fused_algebra: r.on_unless_zero("DGQ_FUSED_ALGEBRA"),
                partial_lm_head: r.on_unless_zero("DGQ_PARTIAL_LM_HEAD"),
                encoder_gpu_moe: r.on_unless_zero("DGQ_ENCODER_GPU_MOE"),
                prefill_batch: r.on_unless_zero("DGQ_PREFILL_BATCH"),
                prefill_mma: r.on_unless_zero("DGQ_PREFILL_MMA"),
                fast_block_extend: r.on_unless_zero("DGQ_FAST_BLOCK_EXTEND"),
                prefill_resident: r.on_unless_zero("DGQ_PREFILL_RESIDENT"),
                attn_kv_block: parse_usize("DGQ_ATTN_KV_BLOCK", 0),
                gemm_attn: r.on_unless_zero("DGQ_GEMM_ATTN"),
                gemm_attn_head_chunk: parse_usize("DGQ_GEMM_ATTN_HC", 16).max(1),
                gemm_attn_qk_bm: parse_usize("DGQ_GEMM_ATTN_QK_BM", 64).max(16),
                gemm_attn_qk_bn: parse_usize("DGQ_GEMM_ATTN_QK_BN", 64).max(16),
                gemm_attn_pv_bm: parse_usize("DGQ_GEMM_ATTN_PV_BM", 64).max(16),
                gemm_attn_pv_bn: parse_usize("DGQ_GEMM_ATTN_PV_BN", 64).max(16),
                gemm_attn_sm_tpg: parse_usize("DGQ_GEMM_ATTN_SM_TPG", 256).max(32),
                gemm_tune_bm: parse_usize("DGQ_GEMM_TUNE_BM", 64).max(16),
                gemm_tune_bn: parse_usize("DGQ_GEMM_TUNE_BN", 64).max(16),
                moe_sparse_bn: parse_usize("DGQ_MOE_SPARSE_BN", 128).max(16),
                flash_prefill: r.on_unless_zero("DGQ_FLASH_PREFILL"),
                flash_prefill_bq: parse_usize("DGQ_FLASH_PREFILL_BQ", 16).max(8),
                flash_prefill_bk: parse_usize("DGQ_FLASH_PREFILL_BK", 64).max(16),
                attn_mma_full_qk_ilp2: r.on_if_one("DGQ_ATTN_MMA_FULL_QK_ILP2"),
                attn_topk: r.on_unless_zero("DGQ_ATTN_TOPK"),
                attn_topk_k: parse_usize("DGQ_ATTN_TOPK_K", 64).max(1),
                attn_topk_dyn: r.on_unless_zero("DGQ_ATTN_TOPK_DYN"),
                attn_topk_decode: r.on_unless_zero("DGQ_ATTN_TOPK_DECODE"),
            },
            prefill: PrefillFlags {
                f16: r.on_if_one("DGQ_PREFILL_F16"),
                kv_f32: r.on_if_one("DGQ_PREFILL_KV_F32"),
                fast_prefill_max: parse_usize("DGQ_FAST_PREFILL_MAX", 0),
                fast_prefill_force: match r.var("DGQ_FAST_PREFILL").as_deref() {
                    Ok("1") | Ok("true") => Some(true),
                    Ok("0") | Ok("false") => Some(false),
                    _ => None,
                },
            },
            kv: KvFlags {
                q8_override: match r.var("DGQ_KV_Q8") {
                    Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => Some(true),
                    Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => Some(false),
                    _ => None, // unset or unrecognized → auto policy
                },
                reuse: r.on_unless_zero("DGQ_KV_REUSE"),
                mmap: r.on_if_one("DGQ_KV_MMAP"),
                mmap_dir: r
                    .var_os("DGQ_KV_MMAP_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir),
            },
            server: ServerFlags {
                conv_cache_bytes: r.gib_bytes("DGQ_CONV_CACHE_GB"),
                conv_disk_bytes: r.gib_bytes("DGQ_CONV_DISK_GB"),
                conv_cache_dir: r
                    .var_os("DGQ_CONV_CACHE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(default_conv_cache_dir),
                continue_past_stop: r.on_if_one("DGQ_CONTINUE_PAST_STOP"),
                prefill_status: r.on_unless_zero("DGQ_PREFILL_STATUS"),
                tool_compact: r.on_if_one("DGQ_TOOL_COMPACT"),
                tool_repair: r.on_if_one("DGQ_TOOL_REPAIR"),
                tool_validate: r.on_if_one("DGQ_TOOL_VALIDATE"),
                tool_compact_threshold: r
                    .var("DGQ_TOOL_COMPACT_THRESHOLD")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&t| t > 0)
                    .unwrap_or(384),
                tool_compact_dir: r
                    .var_os("DGQ_TOOL_COMPACT_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(default_tool_compact_dir),
            },
            debug: DebugFlags {
                quiet: matches!(r.var("DGQ_QUIET").as_deref(), Ok("1"))
                    || r.var("DGQ_QUIET")
                        .map(|v| v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false),
                quiet_from_env,
                logits_finite_check: r.on_if_one("DGQ_CHECK_LOGITS"),
                logits_finite_samples: parse_usize("DGQ_CHECK_LOGITS_SAMPLES", 4096),
                trace_entropy: r
                    .var("DGQ_TRACE_ENTROPY")
                    .map(|v| {
                        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full")
                    })
                    .unwrap_or(false),
                trace_entropy_full: r
                    .var("DGQ_TRACE_ENTROPY")
                    .map(|v| v.eq_ignore_ascii_case("full"))
                    .unwrap_or(false),
                final_entropy_log: r
                    .var("DGQ_LOG_FINAL_ENTROPY")
                    .map(|v| {
                        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full")
                    })
                    .unwrap_or(false),
                step_text_log: r.on_unless_zero("DGQ_LOG_STEP_TEXT"),
                denoise_parity_log: r.on_if_one("DGQ_LOG_DENOISE"),
                denoise_parity_positions: parse_usize("DGQ_LOG_DENOISE_POS", 8),
                log_early_stop: r.on_if_one("DGQ_LOG_EARLY_STOP"),
                sc_log: r.var("DGQ_LOG_SC").ok().as_deref() == Some("1"),
                trace_ranges: r.var("DGQ_TRACE_RANGES").is_ok(),
                mem_watch: r.on_if_one("DGQ_MEM_WATCH"),
                dump_kv_path: r.var("DGQ_DUMP_KV").ok(),
                moe_route_ref_path: r.var("DGQ_MOE_ROUTE_REF").ok(),
                engine_layer_dump_path: r.var("DGQ_ENGINE_LAYER_DUMP").ok(),
                engine_layer_dump_pos: parse_usize("DGQ_ENGINE_LAYER_POS", 129),
                prefill_profile: r.var("DGQ_PREFILL_PROFILE").is_ok(),
                parity_debug: r.var_os("DGQ_PARITY_DEBUG").is_some(),
                kv_noise: r
                    .var("DGQ_KV_NOISE")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok()),
                kv_ring_slots: r
                    .var("DGQ_KV_RING")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .map(|n| n.next_power_of_two().max(2048))
                    .unwrap_or(4096),
                kv_ring_uncapped: r.var("DGQ_KV_RING_UNCAPPED").is_ok(),
                arena_f16_all: r.var("DGQ_ARENA_F16_ALL").is_ok(),
                metal_pipeline_cache: r.var("DGQ_METAL_PIPELINE_CACHE").ok(),
            },
        }
    }
}

/// `DGQ_*_GB` gibibytes → bytes (0 / absent / non-positive → 0).

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

mod accessors;
pub use accessors::*;
