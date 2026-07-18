use crate::tokenizer::Tokenizer;
use std::time::Duration;

use crate::flags::progress_enabled;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DenoiseStopReason {
    None,
    Confident,
    Plateau,
    MaxSteps,
    /// `DGQ_PREFIX_EXIT`: a settled head was committed early; the tail
    /// re-denoises next block.
    PrefixExit,
    Cancelled,
}

impl DenoiseStopReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Confident => "confident",
            Self::Plateau => "plateau",
            Self::MaxSteps => "max_steps",
            Self::PrefixExit => "prefix_exit",
            Self::Cancelled => "cancelled",
        }
    }
}

pub(super) fn log_denoise_step_progress(
    block_idx: usize,
    max_blocks: usize,
    step_idx: u32,
    max_steps: usize,
    stats: &crate::sample::StepEntropyStats,
    mean_entropy_gpu: f32,
    prefix_mean: Option<f32>,
    region_end: Option<usize>,
    answer_text: Option<&str>,
    canvas_stable: u32,
    prefix_stable: u32,
    full_argmax_diff: Option<usize>,
    prefix_argmax_diff: Option<usize>,
    argmax_hist_len: u32,
    accept_plateau: u32,
    step_elapsed: Duration,
    block_elapsed: Duration,
    denoise_elapsed: Duration,
    stop: DenoiseStopReason,
) {
    if !progress_enabled() {
        return;
    }
    let stop_note = match stop {
        DenoiseStopReason::None => "",
        DenoiseStopReason::Confident => " confident_stop",
        DenoiseStopReason::Plateau => " plateau_stop",
        DenoiseStopReason::MaxSteps => " max_steps",
        DenoiseStopReason::PrefixExit => " prefix_exit",
        DenoiseStopReason::Cancelled => " cancelled",
    };
    let mut extra = String::new();
    if let Some(pm) = prefix_mean {
        extra.push_str(&format!(" prefix_mean={pm:.4}"));
    }
    if let Some(re) = region_end {
        extra.push_str(&format!(" ans_len={re}"));
    }
    if let Some(text) = answer_text {
        extra.push_str(&format!(" text={:?}", condense_step_text(text, 80)));
    }
    if let Some(d) = full_argmax_diff {
        extra.push_str(&format!(" full_diff={d}"));
    }
    if let Some(d) = prefix_argmax_diff {
        extra.push_str(&format!(" prefix_diff={d}"));
    }
    eprintln!(
        "step-generate: block {block_idx}/{max_blocks} step {step_idx}/{max_steps} accept={} low_ent={} min_ent={:.4} mean_ent={mean_entropy_gpu:.4} canvas_stable={canvas_stable} prefix_stable={prefix_stable} hist_len={argmax_hist_len} plateau={accept_plateau}{extra} step={step_elapsed:.2?} block={block_elapsed:.2?} denoise={denoise_elapsed:.2?}{stop_note}",
        stats.accept_count, stats.low_entropy_positions, stats.min_entropy,
    );
}

/// Condense a decoded canvas for the per-step progress line. A raw mid-denoise
/// canvas is mostly noise: pad/space floods and an `<eos>` tail run drown the
/// few tokens that matter. Three transforms: (1) collapse every whitespace run
/// (incl. newlines) to one space, (2) drop every `<eos>` except the last, and
/// (3) middle-clip to ~`max` chars — the head shows where the reply starts,
/// the tail is where the churn is (so it gets the larger share).
pub(super) fn condense_step_text(text: &str, max: usize) -> String {
    const EOS: &str = "<eos>";

    let mut collapsed = String::with_capacity(text.len());
    let mut in_ws = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            in_ws = true;
        } else {
            if in_ws && !collapsed.is_empty() {
                collapsed.push(' ');
            }
            in_ws = false;
            collapsed.push(ch);
        }
    }

    // Keep only the LAST <eos>: interior ones are denoise churn, the tail run
    // is redundant, but one marks "the model thinks it is done here".
    if let Some(last) = collapsed.rfind(EOS) {
        let tail = collapsed.split_off(last);
        collapsed = collapsed.replace(EOS, "");
        collapsed.push_str(&tail);
    }
    // The eos removal can leave doubled spaces behind ("a <eos> b" → "a  b").
    while collapsed.contains("  ") {
        collapsed = collapsed.replace("  ", " ");
    }
    let collapsed = collapsed.trim();

    let total = collapsed.chars().count();
    if total <= max {
        return collapsed.to_string();
    }
    let head_n = max * 2 / 5;
    let tail_n = max - head_n;
    let head: String = collapsed.chars().take(head_n).collect();
    let tail: String = collapsed.chars().skip(total - tail_n).collect();
    format!(
        "{head}<... [{}] chars clipped>{tail}",
        total - head_n - tail_n
    )
}

pub(super) fn step_answer_text(
    tokenizer: Option<&Tokenizer>,
    prev_argmax: &[u32],
    ids: &[u32],
    eos_token_id: u32,
) -> (usize, Option<String>) {
    let region_end = crate::sample::answer_region_end(ids, eos_token_id);
    let prefix = crate::sample::answer_prefix_ids(prev_argmax, ids, eos_token_id);
    let text = tokenizer.map(|tok| tok.decode(prefix));
    (region_end, text)
}
