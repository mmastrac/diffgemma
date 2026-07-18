use crate::metal::step_kernel::{N_LAYERS, StepFinishMode, StepSmokeConfig};
use crate::sample::SamplerConfig;

/// Per-denoise-step progress snapshot for live UIs (chat streaming/spinner).
/// `argmax` is the **active** canvas slice only (not stale rows past
/// `active_canvas` after E6 shrink-on-retry). Under the MLX-exact sampler the
/// block's final commit IS that slice, so a stable prefix is a faithful preview.
pub struct StepProgressEvent<'a> {
    /// 1-based committed-block index this step belongs to.
    pub block_idx: usize,
    /// Total blocks budgeted for this generate (`max_new_tokens / CANVAS`).
    pub max_blocks: usize,
    /// 1-based denoise step within the block.
    pub step_in_block: u32,
    pub max_steps: usize,
    /// Active-canvas argmax of this step (`active_canvas` entries).
    pub argmax: &'a [u32],
    /// Positions accepted by the entropy-bound rule this step.
    pub accept_count: u32,
    /// Mean per-position entropy (nats) this step.
    pub mean_entropy: f32,
    /// True when this block's argmax is finalized and will not be E6-retried.
    /// Mid-attempt "would-stop" steps stay false so stream mappers never commit
    /// a discarded empty/degenerate canvas (which used to splice a second
    /// `<|channel>thought` into `reasoning_content`).
    pub block_done: bool,
}

/// Called after every denoise step with the current canvas snapshot. Must be
/// cheap; runs on the generation thread.
pub type StepObserver = std::sync::Arc<dyn Fn(&StepProgressEvent<'_>) + Send + Sync>;

/// Called once per committed denoise block after its stats (incl. token ids and
/// stop metadata) are finalized. Serve uses this to flush `model-*.json` to
/// disk before the next block so a killed process still leaves partial logs.
pub type BlockCommitObserver =
    std::sync::Arc<dyn Fn(&crate::generate::BlockDenoiseStats) + Send + Sync>;

/// Cooperative cancellation for an in-flight generate. Checked between denoise
/// steps and between blocks: the current canvas is abandoned uncommitted (never
/// reaches a streamer as final) and generation returns the blocks committed so
/// far with [`crate::generate::GenerateOutput::cancelled`] set. The KV stays
/// consistent with the returned token log. Clone freely; any holder — a
/// client-side thread, serve's stream observer on a dead socket — can cancel.
#[derive(Clone, Default)]
pub struct CancelToken(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Clone)]
pub struct StepGenerateConfig {
    pub seed: u64,
    pub max_new_tokens: usize,
    pub max_seq: usize,
    pub layers: usize,
    pub sampler: SamplerConfig,
    pub no_early_stop: bool,
    /// Override random canvas (256 ids) for parity with MLX/HF traces.
    pub initial_canvas_ids: Option<Vec<u32>>,
    /// End-of-turn / EOS ids that terminate a "full message". When non-empty,
    /// generation stops (and the sequence is truncated) as soon as a committed
    /// block emits any of these. Empty preserves the fixed `max_new_tokens`
    /// budget behavior used by parity/golden paths.
    pub stop_token_ids: Vec<u32>,
    /// When set (serve tool-mode), a stop token does not end the turn if the
    /// reply still looks unfinished: open `call:NAME{…}`, or non-empty prose
    /// after a closed tool call (`…}<tool_call|>Wait, I`). Stop is trimmed and
    /// generation continues into the next block.
    pub continue_incomplete_tool_calls: bool,
    /// E6 empty/degenerate-reply canvas re-roll (with `DGQ_EMPTY_REPLY_RETRY>0`).
    /// Given the first block's committed argmax, returns true when it renders as
    /// an empty user-facing reply (eos-first canvas OR `<|channel>thought`
    /// ceremony) — checked against the real decoded+sanitized output, not a
    /// token blocklist. `None` disables the re-roll. Built by
    /// `chat_template::empty_reply_check`.
    pub degenerate_reply_check: Option<std::sync::Arc<dyn Fn(&[u32]) -> bool + Send + Sync>>,
    /// Optional per-step progress callback (chat streaming / spinner).
    pub step_observer: Option<StepObserver>,
    /// Optional per-block commit callback (serve model-log flush).
    pub block_commit_observer: Option<BlockCommitObserver>,
    /// Optional cooperative cancel (serve wires its stream observer's dead
    /// socket to this; pipeline clients keep a clone across the channel).
    pub cancel: Option<CancelToken>,
    /// Optional status hook: message-layer stages (tool repair / validator)
    /// call this with a short first-person status when they intervene
    /// mid-turn; serve wires it to a synthetic `reasoning_content` delta so
    /// streaming clients see a thinking block instead of a silent do-over.
    /// Wire-only — the text never enters the token log or canonical KV.
    pub status_notify: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,
}

impl std::fmt::Debug for StepGenerateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepGenerateConfig")
            .field("seed", &self.seed)
            .field("max_new_tokens", &self.max_new_tokens)
            .field("max_seq", &self.max_seq)
            .field("layers", &self.layers)
            .field("sampler", &self.sampler)
            .field("no_early_stop", &self.no_early_stop)
            .field("stop_token_ids", &self.stop_token_ids)
            .field(
                "continue_incomplete_tool_calls",
                &self.continue_incomplete_tool_calls,
            )
            .field(
                "degenerate_reply_check",
                &self.degenerate_reply_check.is_some(),
            )
            .field("step_observer", &self.step_observer.is_some())
            .field(
                "block_commit_observer",
                &self.block_commit_observer.is_some(),
            )
            .field("cancel", &self.cancel.is_some())
            .field("status_notify", &self.status_notify.is_some())
            .finish()
    }
}

impl StepGenerateConfig {
    pub fn from_generate(
        seed: u64,
        max_new_tokens: usize,
        max_seq: usize,
        layers: usize,
        sampler: SamplerConfig,
        no_early_stop: bool,
    ) -> Self {
        Self {
            seed,
            max_new_tokens,
            max_seq,
            layers,
            sampler,
            no_early_stop,
            initial_canvas_ids: None,
            stop_token_ids: Vec::new(),
            continue_incomplete_tool_calls: false,
            degenerate_reply_check: None,
            step_observer: None,
            block_commit_observer: None,
            cancel: None,
            status_notify: None,
        }
    }

    pub(super) fn cancel_requested(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
    }
}

pub(super) fn smoke_config(
    cfg: &StepGenerateConfig,
    prefill_token_ids: Option<Vec<u32>>,
) -> StepSmokeConfig {
    StepSmokeConfig {
        layers: cfg.layers.clamp(1, N_LAYERS),
        steps: cfg.sampler.max_denoising_steps.max(1),
        kv_len: 0,
        seed: cfg.seed,
        max_seq: cfg.max_seq,
        finish: StepFinishMode::Full,
        prefill_token_ids,
        no_early_stop: false,
    }
}
