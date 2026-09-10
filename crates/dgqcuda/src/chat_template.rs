//! DiffusionGemma / Gemma 4 text-only chat formatting.
//!
//! Token assembly matches HuggingFace `apply_chat_template` for the simple
//! text path: special tokens (`<bos>`, `<|turn>`, `<turn|>`, `<|channel>`,
//! `<channel|>`) are inserted by ID; role lines and content are BPE-encoded.
//!
//! The engine carries the same renderer; this is the CUDA path's copy so the
//! CLI can take a text prompt without depending on the macOS-only crate.

use crate::config::Error;
use crate::tokenizer::Tokenizer;

pub const BOS_TOKEN: &str = "<bos>";
const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const CHANNEL_OPEN: &str = "<|channel>";
const CHANNEL_CLOSE: &str = "<channel|>";
const THINK: &str = "<|think|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Model,
    System,
}

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

impl ChatTurn {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn model(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Model,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChatFormatOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
}

impl Default for ChatFormatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
        }
    }
}

fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::User => "user",
        ChatRole::Model => "model",
        ChatRole::System => "system",
    }
}

fn push_special(out: &mut Vec<u32>, tok: &Tokenizer, token: &str) -> Result<(), Error> {
    let id = tok
        .special_token_id(token)
        .ok_or_else(|| Error::Msg(format!("tokenizer has no {token}")))?;
    out.push(id);
    Ok(())
}

fn append_turns(out: &mut Vec<u32>, tok: &Tokenizer, turns: &[ChatTurn]) -> Result<(), Error> {
    for turn in turns {
        push_special(out, tok, TURN_OPEN)?;
        tok.encode_append(out, &format!("{}\n", role_name(turn.role)));
        tok.encode_append(out, turn.content.trim());
        push_special(out, tok, TURN_CLOSE)?;
        tok.encode_append(out, "\n");
    }
    Ok(())
}

/// Build chat prompt token ids (HF `apply_chat_template` compatible).
pub fn format_chat_token_ids(
    tok: &Tokenizer,
    turns: &[ChatTurn],
    opts: &ChatFormatOptions,
) -> Result<Vec<u32>, Error> {
    let mut out = Vec::new();
    push_special(&mut out, tok, BOS_TOKEN)?;
    let think_mode = opts.enable_thinking;
    let first_is_system = turns.first().is_some_and(|t| t.role == ChatRole::System);
    let rest = if think_mode || first_is_system {
        push_special(&mut out, tok, TURN_OPEN)?;
        tok.encode_append(&mut out, "system\n");
        if think_mode {
            push_special(&mut out, tok, THINK)?;
            tok.encode_append(&mut out, "\n");
        }
        let rest = if first_is_system {
            tok.encode_append(&mut out, turns[0].content.trim());
            &turns[1..]
        } else {
            turns
        };
        push_special(&mut out, tok, TURN_CLOSE)?;
        tok.encode_append(&mut out, "\n");
        rest
    } else {
        turns
    };
    append_turns(&mut out, tok, rest)?;
    if opts.add_generation_prompt {
        push_special(&mut out, tok, TURN_OPEN)?;
        tok.encode_append(&mut out, "model\n");
        if !think_mode {
            push_special(&mut out, tok, CHANNEL_OPEN)?;
            tok.encode_append(&mut out, "thought\n");
            push_special(&mut out, tok, CHANNEL_CLOSE)?;
        }
    }
    Ok(out)
}

/// One user message rendered as a prompt.
pub fn user_prompt_ids(tok: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    format_chat_token_ids(tok, &[ChatTurn::user(text)], &ChatFormatOptions::default())
}

/// Turn decoded model text into the user-facing reply: drop the turn ceremony
/// the model may re-emit, then remove thought-channel spans.
pub fn sanitize_model_reply(text: &str) -> String {
    let mut s = text.trim_start().to_string();
    while let Some(rest) = s.strip_prefix(TURN_OPEN) {
        s = match rest.split_once('\n') {
            Some((_, body)) => body.trim_start().to_string(),
            None => String::new(),
        };
    }
    if let Some(idx) = s.find(TURN_CLOSE) {
        s.truncate(idx);
    }
    if let Some(idx) = s.find(TURN_OPEN) {
        s.truncate(idx);
    }
    // Thought spans: everything from an opening channel marker to its close.
    while let Some(open) = s.find(CHANNEL_OPEN) {
        match s[open..].find(CHANNEL_CLOSE) {
            Some(rel) => {
                let end = open + rel + CHANNEL_CLOSE.len();
                s.replace_range(open..end, "");
            }
            None => {
                s.truncate(open);
                break;
            }
        }
    }
    // Orphan closes.
    s = s.replace(CHANNEL_CLOSE, "");
    s.trim().to_string()
}
