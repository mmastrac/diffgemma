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

/// Env access for flag parsing, mirroring `std::env` signatures. `REAL_ENV`
/// reads the process env; `EMPTY_ENV` backs `Default`, so the documented
/// defaults and unset-env parsing CANNOT drift apart
/// (`default_equals_empty_env_parse` pins the equivalence).
struct EnvReader<'a> {
    real: bool,
    /// An EXPLICIT set of pairs standing in for the process env (census
    /// arms, tests). Takes precedence over `real`; absent names fall
    /// through to "unset", so an arm specifies only what it overrides.
    pairs: Option<&'a [(String, String)]>,
}
const REAL_ENV: EnvReader<'static> = EnvReader {
    real: true,
    pairs: None,
};
const EMPTY_ENV: EnvReader<'static> = EnvReader {
    real: false,
    pairs: None,
};

impl EnvReader<'_> {
    fn var(&self, name: &str) -> Result<String, std::env::VarError> {
        if let Some(pairs) = self.pairs {
            return pairs
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
                .ok_or(std::env::VarError::NotPresent);
        }
        if self.real {
            return std::env::var(name);
        }
        Err(std::env::VarError::NotPresent)
    }
    fn var_os(&self, name: &str) -> Option<std::ffi::OsString> {
        // Mirror `var`'s precedence: an explicit pairs set (tests, census
        // arms) stands in for the env completely, checked BEFORE `real`.
        // This was missing — DGQ_HF_HOME/DGQ_QUIET/DGQ_KV_MMAP_DIR/
        // DGQ_CONV_CACHE_DIR/DGQ_TOOL_COMPACT_DIR/DGQ_PARITY_DEBUG (the only
        // `var_os` consumers) silently ignored `from_pairs` overrides and
        // fell straight through to "unset", even though `RuntimeConfig::
        // from_pairs`'s contract (used by `install_for_test` and census arms)
        // is "same helpers, same validation" as the real env.
        if let Some(pairs) = self.pairs {
            return pairs
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| std::ffi::OsString::from(v.clone()));
        }
        if self.real {
            std::env::var_os(name)
        } else {
            None
        }
    }
    /// Set-and-not-"0"/"false" — the default-ON pattern.
    fn on_unless_zero(&self, name: &str) -> bool {
        match self.var(name) {
            Ok(v) => bool_value(name, &v).unwrap_or(true),
            Err(_) => true,
        }
    }
    /// `=1`/`true` enables; `0`/`false` disables; unset leaves it OFF.
    fn on_if_one(&self, name: &str) -> bool {
        match self.var(name) {
            Ok(v) => bool_value(name, &v).unwrap_or(false),
            Err(_) => false,
        }
    }
    fn gib_bytes(&self, name: &str) -> usize {
        let Ok(raw) = self.var(name) else { return 0 };
        match raw.parse::<f64>() {
            Ok(g) if g > 0.0 => (g * 1024.0 * 1024.0 * 1024.0) as usize,
            _ => {
                reject(name, &raw, "a positive number of GiB");
                0
            }
        }
    }

    /// A float flag with an inclusive valid range. A set-but-invalid value is
    /// REJECTED rather than silently replaced by `default` — the
    /// `DGQ_PREFIX_EXIT=1` footgun (1.0 is out of range for a 0..1 threshold,
    /// so it silently DISABLED the very lever the operator was enabling).
    fn ranged_f32(&self, name: &str, default: f32, lo: f32, hi: f32) -> f32 {
        let Ok(raw) = self.var(name) else { return default };
        match raw.parse::<f32>() {
            Ok(v) if (lo..=hi).contains(&v) => v,
            _ => {
                reject(name, &raw, &format!("a number in [{lo}, {hi}]"));
                default
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flag validation: a set-but-invalid DGQ_* value is fatal, never a silent
// fallback to the default. Parse helpers record
// rejections here; `from_env` drains them and kills the process. Collected
// rather than failing fast so ONE run reports EVERY bad flag.
// ---------------------------------------------------------------------------

thread_local! {
    static FLAG_ERRORS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Shared truthiness for both boolean patterns. Anything outside the
/// accepted set is REJECTED rather than silently read as its "other" value
/// — `DGQ_X=yes` used to mean OFF under `on_if_one` and ON under
/// `on_unless_zero`, which is the opposite of what an operator typing it
/// intends in at least one of those cases.
fn bool_value(name: &str, raw: &str) -> Option<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => {
            reject(name, raw, "one of 1/0, true/false, yes/no, on/off");
            None
        }
    }
}

fn reject(name: &str, raw: &str, expected: &str) {
    FLAG_ERRORS.with(|e| {
        e.borrow_mut()
            .push(format!("{name}={raw:?} is not valid; expected {expected}"))
    });
}

fn take_flag_errors() -> Vec<String> {
    FLAG_ERRORS.with(|e| std::mem::take(&mut *e.borrow_mut()))
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
    /// `DGQ_EMPTY_REPLY_RETRY`: degenerate-first-block re-roll count. Default 3.
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
    /// `DGQ_COMMIT_CONF_TRIM` (p_max τ in (0,1]; 0 disables): trim a proposed
    /// block at the first answer-region row that is BOTH below τ and
    /// argmax-duplicating a neighbour — unresolved rows argmax-copy their
    /// neighbour (the duplication micro-stutter), so the tail re-denoises
    /// next block against committed context instead.
    ///
    /// DEFAULT ON at 0.9. Census evidence
    /// (`census --arm ... --battery smoke --seeds 7,42,123`, ~13.7k
    /// committed rows per arm), dup tier alone vs off:
    ///   contested-committed 1.31 -> 1.03 per 1k rows
    ///   (hard-class 5 -> 3, dup-class 13 -> 11)
    ///   smoke 17/17 x 3 seeds, retrieval 100%, +1 step total (362 -> 363)
    /// It fires rarely (1 trim across 51 prompt-runs) and is close to free.
    /// Note it lowers HARD-class commits too without a hard tier: truncating
    /// at a dup row keeps later contested rows out of KV and re-denoises
    /// them. The unconditional hard tier is the separate
    /// `DGQ_COMMIT_CONF_HARD`, still OFF — it kills the insertion class
    /// outright (5 -> 0) but fires 4x as often, needs 58 blocks instead of
    /// 57, and tips `transformer` over its 39-step convergence budget at
    /// seed 7 (that budget was ratcheted on a NON-trimming baseline).
    pub commit_conf_trim: f32,
    /// `DGQ_COMMIT_CONF_HARD` (p_max floor in [0,1]; 0 disables): trim a
    /// proposed block at the first answer-region row committing BELOW this
    /// confidence, regardless of whether it duplicates a neighbour — the
    /// insertion/omission class (`ownBut` 0.405, `("."`, whole clauses
    /// dropped at 0.296-0.49) that the dup-conjunctive tier is blind to.
    /// Split from `DGQ_COMMIT_CONF_TRIM` (which used to imply a hard tier at
    /// 0.5) so the two tiers can be gated independently: the census found
    /// the hard tier carries the value (contested-committed 5 -> 0) while
    /// the dup tier cost a step-budget failure. 0.5 is the measured
    /// separation point between broken insertions and benign creative soft
    /// rows. Default OFF (0).
    pub commit_conf_hard: f32,
    /// `DGQ_PREFIX_EXIT` (head mean-entropy threshold in nats, (0, 0.5];
    /// 0/out-of-range disables): per denoise step, if a head prefix (largest
    /// of active/2, /4, /8; ≥16 rows) has mean entropy below this while the
    /// tail still churns (≥2×) and the head argmax is 2-step stable, end the
    /// block and commit only the head — the tail re-denoises next block
    /// against committed context. Recommended 0.05 (the early-stop
    /// threshold). Default OFF (0) pending live A/B.
    pub prefix_exit: f32,
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
    /// `DGQ_MOE_FUSE_GATHER`: fuse gather into gate_up expert GEMM.
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
    /// `DGQ_GEMM_ATTN`: route full-layer PREFILL attention through the
    /// GEMM decomposition (attn_gemm_qk/softmax/pv) instead of attention_mma_full.
    /// Not bit-identical but quality-gated. Denoise reuses the qk stage through
    /// top-k (DGQ_ATTN_TOPK_DECODE). `=0` restores mma_full.
    pub gemm_attn: bool,
    /// `DGQ_GEMM_ATTN_HC`: Q heads processed per dispatch batch.
    /// Bounds the S/P scratch to [HC][CANVAS][n_pad(max_seq)]. Default 16 (all
    /// heads). HC is numerically invariant (per-head disjoint scratch).
    /// Scratch ~1.6 GiB @100k on 36 GB; drop to 4 for very long contexts
    /// if memory-pressured.
    pub gemm_attn_head_chunk: usize,
    /// Tunable tile geometry. QK GEMM tile qk_bm x qk_bn,
    /// PV GEMM tile pv_bm x pv_bn, softmax threads/row. Defaults reproduce the
    /// shipped 64x64/256 kernel. `DGQ_GEMM_ATTN_{QK_BM,QK_BN,PV_BM,PV_BN,SM_TPG}`.
    pub gemm_attn_qk_bm: usize,
    pub gemm_attn_qk_bn: usize,
    pub gemm_attn_pv_bm: usize,
    pub gemm_attn_pv_bn: usize,
    pub gemm_attn_sm_tpg: usize,
    /// Dense tunable-GEMM tile + MoE-sparse N-tile. Defaults reproduce the
    /// shipped kernels (64x64 dense, 128 sparse N-tile with fixed 32-row block
    /// height). `DGQ_GEMM_TUNE_BM/BN`, `DGQ_MOE_SPARSE_BN`.
    pub gemm_tune_bm: usize,
    pub gemm_tune_bn: usize,
    pub moe_sparse_bn: usize,
    /// `DGQ_GEMM_W32`: single aligned u32 loads for packed q4/nvfp4 weight
    /// bytes in the tunable GEMM (replaces per-byte assembly; hoists the nvfp4
    /// group-scale decode). Bit-identical — the kernel falls back to byte
    /// loads when a row pointer is not 4-byte aligned. Default OFF pending A/B.
    pub gemm_w32: bool,
    /// `DGQ_FLASH_PREFILL`: fused flash for sliding-layer PREFILL (hd=256,
    /// window=1024). Online softmax, register-resident O split across 8
    /// simdgroups, no per-chunk PV tgmem round-trip. Default ON.
    /// `DGQ_FLASH_PREFILL=0` restores mma2. `bq`/`bk` = query-row tile /
    /// key-block streamed.
    pub flash_prefill: bool,
    /// `DGQ_ATTN_MMA_FULL_QK_ILP2`: split the 32-deep serial QK dot
    /// in `attention_mma_full` into two interleaved 16-deep accumulator chains
    /// (even/odd chunks), summed at the end. Non-bit-identical; quality-gated.
    /// Default OFF.
    pub attn_mma_full_qk_ilp2: bool,
    pub flash_prefill_bq: usize,
    pub flash_prefill_bk: usize,
    /// `DGQ_ATTN_TOPK`: top-k sparse attention for full-layer PREFILL.
    /// Selects the top-k highest-scoring keys per (row, head) — exact by f32
    /// score, deterministic — and gathers V at those indices. Non-bit-identical
    /// to dense. Default ON. `DGQ_ATTN_TOPK=0` restores dense.
    pub attn_topk: bool,
    /// `DGQ_ATTN_TOPK_K`: k per query row. The kernel's slot capacity K_PAD is
    /// compiled to `next_power_of_two(k)` (min 64, max 1024) via
    /// `attention_topk::tuned_source`, so any k in [1, 1024] is honored.
    /// Default 64.
    pub attn_topk_k: usize,
    /// `DGQ_ATTN_TOPK_DYN`: kv-adaptive k — per dispatch, k is a constant
    /// fraction of context instead of a constant count. Overrides DGQ_ATTN_TOPK_K
    /// when set; K_PAD compiles at 512. Default ON.
    /// `DGQ_ATTN_TOPK_DYN=0` + `DGQ_ATTN_TOPK_K=n` restores fixed k.
    pub attn_topk_dyn: bool,
    /// `DGQ_ATTN_TOPK_DECODE`: top-k sparse attention for full-layer DENOISE
    /// (decode) dispatches — the same pipeline as prefill top-k, causal=0,
    /// k from the same fixed/dyn policy. Non-bit-identical; every denoise step
    /// compounds, so the gate is the multi-seed census + doc-QA ladder.
    /// Default ON. `DGQ_ATTN_TOPK_DECODE=0` restores dense attention.
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
    /// `DGQ_PREFILL_KV_F32`: f32 side ring for sliding KV. Default OFF.
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
    /// `None` = the auto policy. `kv_format` and
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

/// Layered `.dgq` pack loading (external HF-safetensors / pack-bin refs).
#[derive(Debug, Clone, PartialEq)]
pub struct PackFlags {
    /// `DGQ_HF_HOME`: override for the HuggingFace cache root used to
    /// resolve `HfSafetensors` external refs. Falls back to the `HF_HOME`
    /// env var, then `~/.cache/huggingface` (the `huggingface_hub`
    /// default). Unset = None (fall through the chain).
    pub hf_home: Option<PathBuf>,
}

impl Default for PackFlags {
    /// The unset-env parse IS the default — one source of truth,
    /// nothing to drift (see `EnvReader`).
    fn default() -> Self {
        RuntimeConfig::from_reader(&EMPTY_ENV).pack
    }
}

/// Server (serve) configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerFlags {
    /// `DGQ_PACED_STREAM`: hold a committed block's streamed text and release
    /// it progressively during the next block's denoise (flushes at turn end
    /// or first tool call). Default ON; `0` restores burst-per-block.
    pub paced_stream: bool,
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
    /// `DGQ_TOKEN_CLASS=<probe.json>`: classify each committed token's output
    /// mode with a linear probe (see `crate::token_class`). Pure
    /// instrumentation — it reads hidden state at COMMIT and never steers, so
    /// generation stays byte-identical. Unset = no readback is performed at
    /// all, so the hot path is untouched.
    pub token_class_probe: Option<String>,
    pub denoise_parity_log: bool,
    pub denoise_parity_positions: usize,
    pub log_early_stop: bool,
    pub sc_log: bool,
    pub trace_ranges: bool,
    pub mem_watch: bool,
    pub dump_kv_path: Option<String>,
    /// `DGQ_TRACE_PMAX_JSONL=<path>`: append one JSON line per denoise step
    /// (per-position p_max/entropy/accept/argmax) plus a block-commit line —
    /// the M0 instrumentation (readback-only, zero behavior change).
    pub trace_pmax_jsonl: Option<String>,
    /// `DGQ_DELIM_CHECK=1`: run the structural delimiter/parity checker on
    /// every committed block and log the verdict (see [`crate::delimiter`]).
    /// Observational only — it decodes committed text and counts delimiters,
    /// never steering, so generation stays byte-identical.
    pub delim_check: bool,
    /// `DGQ_DELIM_CHECK_JSONL=<path>`: append one JSON record per checked
    /// block (mode, margin, terminated, findings) — the measurement vehicle
    /// for the failure rate split by block mode.
    pub delim_check_jsonl: Option<String>,
    pub moe_route_ref_path: Option<String>,
    pub engine_layer_dump_path: Option<String>,
    pub engine_layer_dump_pos: usize,
    pub prefill_profile: bool,
    pub parity_debug: bool,
    /// `DGQ_KV_NOISE=<rel eps>`: perturb every live f16 KV value after prefill
    /// (sensitivity probe). None = off.
    pub kv_noise: Option<f32>,
    /// `DGQ_KV_RING`: sliding-layer KV ring slots (power of two, min 2048 =
    /// window-1 + canvas live positions). Default 4096: the O(1)-truncate
    /// slack is ring − CANVAS − (window−1) — 769 slots at 2048, 2817 at
    /// 4096 — so the bigger ring turns typical thought-strip finalize
    /// truncates and validator/triage rewinds into O(1) instead of a ring
    /// rebuild (≈ a full re-prefill of the kept prefix), for a fixed
    /// few-hundred-MB of extra sliding KV independent of context length.
    pub kv_ring_slots: usize,
    /// `DGQ_KV_RING_UNCAPPED`: linear (no-wrap) sliding KV storage
    /// (ring-read isolation). Any set value enables.
    pub kv_ring_uncapped: bool,
    /// `DGQ_ARENA_F16_ALL`: build the MAIN activation set fp16 too (bring-up
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
    pub pack: PackFlags,
}

impl RuntimeConfig {
    /// Parse every `DGQ_*` env flag once. The single source of env truth.
    pub fn from_env() -> Self {
        let (cfg, errors) = Self::from_reader_checked(&REAL_ENV);
        if !errors.is_empty() {
            eprintln!("fatal: invalid DGQ_* flag value(s):");
            for e in &errors {
                eprintln!("  {e}");
            }
            eprintln!(
                "A set-but-invalid flag is NOT silently ignored: it would run with the \
                 default while reading like it was configured (e.g. DGQ_PREFIX_EXIT=1 \
                 used to DISABLE prefix-exit). Fix or unset the flag."
            );
            std::process::exit(2);
        }
        cfg
    }

    /// Parse an EXPLICIT set of `DGQ_*` pairs exactly as if they were the
    /// process env — same helpers, same validation, same defaults for
    /// anything absent. This is how a census ARM is built: an arm states
    /// only what it overrides, and a typo in an arm is caught by the same
    /// rules that would kill the process at startup, before any GPU time is
    /// spent on it. Returns the rejections instead of exiting so the caller
    /// can attribute them to the arm that owns them.
    pub fn from_pairs(pairs: &[(String, String)]) -> (Self, Vec<String>) {
        Self::from_reader_checked(&EnvReader {
            real: false,
            pairs: Some(pairs),
        })
    }

    /// [`from_reader`](Self::from_reader) plus the rejections it recorded —
    /// the testable half of the hard-kill validation, so tests can assert the
    /// diagnosis without the process exit.
    fn from_reader_checked(r: &EnvReader<'_>) -> (Self, Vec<String>) {
        let _ = take_flag_errors(); // discard anything a prior parse left
        let cfg = Self::from_reader(r);
        (cfg, take_flag_errors())
    }

    /// The one parse. `Default` goes through this with [`EMPTY_ENV`].
    fn from_reader(r: &EnvReader<'_>) -> Self {
        let parse_usize = |name: &str, default: usize| -> usize {
            let Ok(raw) = r.var(name) else { return default };
            match raw.parse::<usize>() {
                Ok(v) => v,
                Err(_) => {
                    reject(name, &raw, "a non-negative integer");
                    default
                }
            }
        };
        let quiet_from_env = r.var_os("DGQ_QUIET").is_some();

        RuntimeConfig {
            sampler: SamplerFlags {
                freeze: r.on_if_one("DGQ_FREEZE"),
                denoiser_argmax: r.on_unless_zero("DGQ_DENOISER_ARGMAX"),
                early_stop_mean_ent: r.ranged_f32(
                    "DGQ_EARLY_STOP_MEAN_ENT",
                    0.05,
                    0.0,
                    f32::MAX,
                ),
                empty_reply_retry: parse_usize("DGQ_EMPTY_REPLY_RETRY", 3) as u32,
                ws_block_stop: r.on_if_one("DGQ_WS_BLOCK_STOP"),
                block_commit_max_ent: r
                    .ranged_f32("DGQ_BLOCK_COMMIT_MAX_ENT", 0.0, 0.0, f32::MAX),
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
                commit_conf_trim: r
                    .ranged_f32("DGQ_COMMIT_CONF_TRIM", 0.9, 0.0, 1.0),
                commit_conf_hard: r.ranged_f32("DGQ_COMMIT_CONF_HARD", 0.0, 0.0, 1.0),
                prefix_exit: r
                    // 0.0 disables; the old filter REJECTED 1 and silently
                    // fell back to 0.0 — i.e. `=1` disabled the lever it read
                    // like it was enabling. Now `=1` is fatal with a message.
                    .ranged_f32("DGQ_PREFIX_EXIT", 0.0, 0.0, 0.5),
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
                gemm_w32: r.on_if_one("DGQ_GEMM_W32"),
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
            pack: PackFlags {
                hf_home: r.var_os("DGQ_HF_HOME").map(PathBuf::from),
            },
            server: ServerFlags {
                paced_stream: r.on_unless_zero("DGQ_PACED_STREAM"),
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
                trace_pmax_jsonl: r.var("DGQ_TRACE_PMAX_JSONL").ok().filter(|v| !v.is_empty()),
                token_class_probe: r.var("DGQ_TOKEN_CLASS").ok().filter(|v| !v.is_empty()),
                delim_check: r.on_if_one("DGQ_DELIM_CHECK"),
                delim_check_jsonl: r.var("DGQ_DELIM_CHECK_JSONL").ok().filter(|v| !v.is_empty()),
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

#[cfg(test)]
#[must_use = "the override is reverted when the guard drops"]
pub fn install_for_test(cfg: RuntimeConfig) -> ScopedGuard {
    install_scoped(cfg)
}

/// Install `cfg` as the config for the CURRENT THREAD's scope; the returned
/// guard restores the prior thread-local override on drop.
///
/// Thread-local by design, so a scoped override can never leak to another
/// reader. Used by tests, and by `census` to run one ARM of a campaign
/// without re-launching the process.
#[must_use = "the override is reverted when the guard drops"]
pub fn install_scoped(cfg: RuntimeConfig) -> ScopedGuard {
    let prev = OVERRIDE.with(|o| o.borrow_mut().replace(Arc::new(cfg)));
    ScopedGuard { prev }
}

/// RAII guard from [`install_scoped`]: restores the prior thread-local
/// override (possibly `None`) on drop.
pub struct ScopedGuard {
    prev: Option<Arc<RuntimeConfig>>,
}

impl Drop for ScopedGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        OVERRIDE.with(|o| *o.borrow_mut() = prev);
    }
}

mod accessors;
pub use accessors::*;

#[cfg(test)]
mod validation_tests {
    use super::*;

    /// Parse an explicit pair set, returning the recorded rejections.
    fn parse_with(vars: &[(&str, &str)]) -> (RuntimeConfig, Vec<String>) {
        let owned: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        RuntimeConfig::from_pairs(&owned)
    }

    /// THE footgun this exists for: `DGQ_PREFIX_EXIT=1` is out of the lever's
    /// (0, 0.5] range, so it used to fall back to 0.0 — DISABLING the very
    /// thing the operator was switching on, silently.
    #[test]
    fn out_of_range_float_is_rejected_not_silently_defaulted() {
        let (cfg, errs) = parse_with(&[("DGQ_PREFIX_EXIT", "1")]);
        assert_eq!(cfg.sampler.prefix_exit, 0.0, "value must not be adopted");
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("DGQ_PREFIX_EXIT"), "{errs:?}");
        assert!(errs[0].contains("[0, 0.5]"), "range must be stated: {errs:?}");
        // In range still works, and reports nothing.
        let (cfg, errs) = parse_with(&[("DGQ_PREFIX_EXIT", "0.05")]);
        assert_eq!(cfg.sampler.prefix_exit, 0.05);
        assert!(errs.is_empty(), "{errs:?}");
    }

    /// A typo'd bool used to be read as its "other" value, and which value
    /// depended on whether the flag was opt-in or opt-out.
    #[test]
    fn unparseable_bool_is_rejected_under_both_patterns() {
        // on_if_one (default OFF) and on_unless_zero (default ON).
        let (_, errs) = parse_with(&[("DGQ_FREEZE", "yep"), ("DGQ_DENOISER_ARGMAX", "nope")]);
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("DGQ_FREEZE")), "{errs:?}");
        assert!(
            errs.iter().any(|e| e.contains("DGQ_DENOISER_ARGMAX")),
            "{errs:?}"
        );
        // The documented spellings all parse, both directions.
        for v in ["1", "true", "yes", "on", "0", "false", "no", "off", "TRUE"] {
            let (_, errs) = parse_with(&[("DGQ_FREEZE", v)]);
            assert!(errs.is_empty(), "{v} rejected: {errs:?}");
        }
    }

    #[test]
    fn unparseable_integer_is_rejected() {
        let (_, errs) = parse_with(&[("DGQ_EMPTY_REPLY_RETRY", "lots")]);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("DGQ_EMPTY_REPLY_RETRY"), "{errs:?}");
    }

    /// Every bad flag in one run, so an operator fixes them in one pass
    /// instead of one-per-restart.
    #[test]
    fn all_bad_flags_are_reported_together() {
        let (_, errs) = parse_with(&[
            ("DGQ_PREFIX_EXIT", "1"),
            ("DGQ_FREEZE", "maybe"),
            ("DGQ_EMPTY_REPLY_RETRY", "x"),
        ]);
        assert_eq!(errs.len(), 3, "{errs:?}");
    }

    /// Unset stays default and silent — the whole point is that only a
    /// PRESENT-but-invalid value is fatal.
    #[test]
    fn unset_env_is_default_and_reports_nothing() {
        let (cfg, errs) = parse_with(&[]);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(cfg, RuntimeConfig::default());
    }
}

#[cfg(test)]
mod arm_tests {
    use super::*;

    /// A census ARM states only its overrides; everything else is the
    /// documented default, so arms compose without restating the world.
    #[test]
    fn arm_overrides_only_what_it_names() {
        let (cfg, errs) = RuntimeConfig::from_pairs(&[(
            "DGQ_COMMIT_CONF_TRIM".into(),
            "0.9".into(),
        )]);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(cfg.sampler.commit_conf_trim, 0.9);
        // Untouched flags keep their defaults.
        let base = RuntimeConfig::default();
        assert_eq!(cfg.sampler.prefix_exit, base.sampler.prefix_exit);
        assert_eq!(cfg.perf.attn_topk, base.perf.attn_topk);
        // An empty arm IS the default config.
        let (empty, errs) = RuntimeConfig::from_pairs(&[]);
        assert!(errs.is_empty());
        assert_eq!(empty, base);
    }

    /// A typo in an arm must be caught by the SAME rules that guard the
    /// process env — before the campaign spends GPU hours on it.
    #[test]
    fn arm_validation_matches_process_env_rules() {
        let (_, errs) = RuntimeConfig::from_pairs(&[
            ("DGQ_PREFIX_EXIT".into(), "1".into()),
            ("DGQ_FREEZE".into(), "sometimes".into()),
        ]);
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("DGQ_PREFIX_EXIT")), "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("DGQ_FREEZE")), "{errs:?}");
    }
}
