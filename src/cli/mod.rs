//! Command-line surface: the `Cli`/`Command` types and the `parse_cli` parser.
//! Extracted from the former monolithic main.rs; the command bodies live in
//! `crate::commands`.

mod parse;
pub(crate) use parse::parse_cli;

use std::path::PathBuf;

#[derive(Debug)]
pub(crate) struct Cli {
    pub(crate) model_dir: PathBuf,
    pub(crate) command: Command,
    /// When true, `-p` text is BPE-encoded as-is (no chat template).
    pub(crate) raw_prompt: bool,
}
#[derive(Debug)]
pub(crate) enum Command {
    Summary,
    Config,
    Weights(String),
    Layer0,
    GenerateMonolithic {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        kernel_assert: bool,
        kernel_debug_deep: bool,
        write_golden: Option<String>,
        write_trace: Option<PathBuf>,
        /// `--verbose`: show the full generation telemetry. Default prints only
        /// the clean reply (one-shot chat).
        verbose: bool,
    },
    GenerateMonolithicParity {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        golden: Option<String>,
        write_golden: Option<String>,
    },
    Tokenize(String),
    Gemm {
        size: usize,
    },
    Attention,
    ProbeDevice,
    BenchGemm {
        shapes: String,
        oracle: Option<String>,
        iters: usize,
    },
    /// Tunable E17 prefill-attention bench for sweeps (task #87): compiles the
    /// kernels for a tile/HC/TPG config and prints RESULT {json: ms, tf_s}.
    BenchPrefillAttn {
        kv_len: u32,
        qk_bm: usize,
        qk_bn: usize,
        pv_bm: usize,
        pv_bn: usize,
        hc: usize,
        sm_tpg: usize,
        side: bool,
        causal: bool,
        iters: usize,
    },
    /// Holistic prefill proxy (task #87): time one real M=1024 super-chunk (all
    /// stages, real weights) at --kv-len; prints RESULT {json: ms}. Honors the
    /// tunable tile flags — the objective for a whole-model BO sweep.
    BenchPrefillSuper {
        kv_len: u32,
        iters: usize,
        stages: bool,
        n_subs: usize,
    },
    Quantize {
        output: PathBuf,
        profile: String,
    },
    StepSmoke {
        layers: usize,
        steps: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        forward_only: bool,
        prompt: Option<String>,
    },
    StepProbe {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        prompt: Option<String>,
    },
    StepKvCheck {
        kv_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
    },
    StepKvParity {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepKvBf16Cross {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        bf16_ref_dir: PathBuf,
    },
    StepAttnProbe {
        prompt: Option<String>,
        prompt_len: usize,
        layer: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepLogitsDump {
        prompt: Option<String>,
        layers: usize,
        steps: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        positions: String,
        top_k: usize,
    },
    StepBf16OracleLogitsDump {
        prompt: Option<String>,
        layers: usize,
        steps: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        positions: String,
        top_k: usize,
        bf16_ref_dir: PathBuf,
        gpu_kv: bool,
    },
    StepLayerProbe {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        position: usize,
    },
    StepAttnDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        position: usize,
    },
    /// E22-M0: dump the full canvas Q plane + layer K cache (raw f32) for
    /// offline block-mass analysis.
    StepAttnQkDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
    },
    StepMoeDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        position: usize,
    },
    StepMoeRouteDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        run_grouped: bool,
    },
    StepMoeBatchedPinDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
    },
    StepMoeSingleDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        position: usize,
        expert: u32,
    },
    StepPreambleDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        position: usize,
    },
    EmbedRowDump {
        token: u32,
        layers: usize,
        max_seq: usize,
        prompt: Option<String>,
        raw_prompt: bool,
        output: PathBuf,
        bf16_ref_dir: Option<PathBuf>,
        gpu: bool,
    },
    StepVerify {
        layers: usize,
    },
    StepCi {
        layers: usize,
    },
    StepParity {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
    },
    BenchStepKernel {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        iters: usize,
        forward_only: bool,
        profile: bool,
        profile_steps: usize,
        layer_profile: bool,
    },
    BenchGemmFusion {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        iters: usize,
    },
    Chat {
        seed: u64,
        steps: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        initial_prompt: Option<String>,
        verbose: bool,
        events_path: Option<PathBuf>,
        json: bool,
        /// Context window (max_seq) override; None = default 8192.
        ctx: Option<usize>,
    },
    /// OpenAI-compatible HTTP server (`serve`). Single-GPU-queue chat completions.
    Serve {
        addr: String,
        seed: u64,
        steps: usize,
        max_layers: Option<usize>,
        /// Context window (max_seq); default 8192.
        ctx: usize,
        /// Tool-output compaction (KV rewinder). Also `DGQ_TOOL_COMPACT=1`.
        tool_compact: bool,
        /// Model-guided tool-call repair (error response -> corrected call ->
        /// corrupt exchange rewound out of KV). Also `DGQ_TOOL_REPAIR=1`.
        tool_repair: bool,
        /// Rewind + regenerate on malformed tool-call grammar. Also
        /// `DGQ_TOOL_VALIDATE=1`.
        tool_validate: bool,
        /// Directory for per-turn JSONL debug logs (`--log-dir`).
        log_dir: Option<PathBuf>,
        /// Default thought-channel enablement when the request does not set
        /// `enable_thinking` or a `model:think…` suffix. CLI `--think` / `--think=false`.
        think: bool,
    },
    Smoketest {
        prompts_path: Option<PathBuf>,
        /// None = spec-pinned gate seed; Some = explicit CLI --seed sweep.
        seed: Option<u64>,
        steps: usize,
        max_layers: Option<usize>,
        /// Substring (case-insensitive) on prompt id; only matching prompts run.
        filter: Option<String>,
        /// Repeat the whole (filtered) prompt sequence N times in ONE session
        /// (no re-warmup) — surfaces reset_kv session-state carryover.
        repeat: usize,
        /// Run ONLY the long-context doc-QA tier (bigger session; E13).
        longctx: bool,
    },
    /// Flag-arm x battery campaign with explicit acceptance gates.
    Census {
        /// `NAME:KEY=VAL,...` per arm; `NAME:` is a no-override baseline.
        arms: Vec<String>,
        /// `smoke` and/or `longctx`.
        batteries: Vec<String>,
        seeds: Vec<u64>,
        /// `METRIC<OP>VALUE`, value may be `baseline[*factor]`.
        gates: Vec<String>,
        baseline: Option<String>,
        out_dir: Option<PathBuf>,
        /// Dup-tier p_max threshold used when COUNTING warts (not a lever).
        tau: f32,
        steps: usize,
        /// Report on an existing trace dir instead of running (no GPU).
        analyze: Option<PathBuf>,
    },
    /// Golden byte-identity pack — the Tier-1 refactor gate (task #73).
    /// Print the generated kernel FC-axis manifest (TOML) to stdout.
    Manifest,
    /// Re-execute an `ops.jsonl` op-log (serve `--log-dir`) against a fresh
    /// pipeline and diff every event against the recorded one.
    Replay {
        /// Path to the ops.jsonl to replay.
        log_path: PathBuf,
        /// Spawn overrides; default to the log's `meta` header.
        ctx: Option<usize>,
        steps: Option<usize>,
    },
    Golden {
        /// Pack spec (default `fixtures/golden/golden.json`).
        pack_path: Option<PathBuf>,
        /// Re-record every case's golden output instead of checking it.
        bless: bool,
        /// Substring (case-insensitive) on case id; only matching cases run.
        filter: Option<String>,
    },
}
#[cfg(target_os = "macos")]
pub(crate) fn default_generate_command(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
) -> Command {
    let _ = model_dir;
    // The non-monolithic generate surface is retired; `ask`/`generate` always
    // run the monolithic step path.
    Command::GenerateMonolithic {
        prompt,
        seed,
        steps,
        prompt_len,
        max_new_tokens,
        max_layers,
        no_early_stop,
        kernel_assert: false,
        kernel_debug_deep: false,
        write_golden: None,
        write_trace: None,
        verbose: false,
    }
}
/// Production generate/chat default is 48 (model card); parity/bench default is 2.
pub(crate) fn resolve_steps(override_steps: Option<usize>, parity_default: bool) -> usize {
    override_steps.unwrap_or(if parity_default { 2 } else { 48 })
}
