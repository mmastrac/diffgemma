//! OpenAI SSE / JSON envelope framing for the [`WireDelta`]s the decoder's
//! serve mapper produces.

use serde::Serialize;

use crate::decoder::WireDelta;

#[derive(Serialize)]
pub(crate) struct DeltaBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(rename = "x-diffusion-draft", skip_serializing_if = "Option::is_none")]
    pub(crate) draft: Option<DraftBody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
pub(crate) struct DraftBody {
    pub(crate) text: String,
    pub(crate) committed: usize,
    pub(crate) block: usize,
    pub(crate) step: u32,
}

#[derive(Serialize)]
pub(crate) struct ChunkChoice {
    pub(crate) index: u32,
    pub(crate) delta: DeltaBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
pub(crate) struct Chunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChunkChoice>,
}

pub(crate) fn empty_delta() -> DeltaBody {
    DeltaBody {
        role: None,
        content: None,
        reasoning_content: None,
        draft: None,
        tool_calls: None,
    }
}

impl WireDelta {
    pub(crate) fn into_delta_body(self) -> DeltaBody {
        match self {
            WireDelta::Reasoning(s) => DeltaBody {
                reasoning_content: Some(s),
                ..empty_delta()
            },
            WireDelta::Content(s) => DeltaBody {
                content: Some(s),
                ..empty_delta()
            },
            WireDelta::Draft {
                text,
                committed,
                block,
                step,
            } => DeltaBody {
                draft: Some(DraftBody {
                    text,
                    committed,
                    block,
                    step,
                }),
                ..empty_delta()
            },
        }
    }
}

pub(crate) fn finish_reason_for(tool_calls: &[serde_json::Value], stopped: bool) -> &'static str {
    if !tool_calls.is_empty() {
        "tool_calls"
    } else if stopped {
        "stop"
    } else {
        "length"
    }
}

/// Truncate or suppress wire deltas that would leak native tool-call markup into
/// OpenAI `content`. Returns `None` when the delta should be dropped.
pub(crate) fn filter_tool_markup_delta(
    d: WireDelta,
    strip: bool,
    suppress: &std::sync::atomic::AtomicBool,
) -> Option<WireDelta> {
    use std::sync::atomic::Ordering;
    if !strip {
        return Some(d);
    }
    match d {
        WireDelta::Content(s) => {
            if suppress.load(Ordering::Relaxed) {
                return None;
            }
            match s.find("<|tool_call>") {
                Some(i) => {
                    suppress.store(true, Ordering::Relaxed);
                    let keep = s[..i].trim_end().to_string();
                    (!keep.is_empty()).then_some(WireDelta::Content(keep))
                }
                None => Some(WireDelta::Content(s)),
            }
        }
        WireDelta::Draft {
            mut text,
            committed,
            block,
            step,
        } => {
            if let Some(i) = text.find("<|tool_call>") {
                text.truncate(i);
                while text.ends_with(char::is_whitespace) {
                    text.pop();
                }
            }
            Some(WireDelta::Draft {
                text,
                committed,
                block,
                step,
            })
        }
        other => Some(other),
    }
}
