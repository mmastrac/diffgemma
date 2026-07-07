//! DiffusionGemma / Gemma 4 text-only chat formatting.
//!
//! Token assembly matches HuggingFace `apply_chat_template` for the simple
//! text path: special tokens (`<bos>`, `<|turn>`, `<turn|>`, `<|channel>`,
//! `<channel|>`) are inserted by ID; role lines and content are BPE-encoded.

use crate::safetensors::Error;
use crate::tokenizer::Tokenizer;

pub const BOS_TOKEN: &str = "<bos>";
const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const CHANNEL_OPEN: &str = "<|channel>";
const CHANNEL_CLOSE: &str = "<channel|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Model,
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

/// Predicate for the E6 empty/degenerate-reply canvas re-roll: given a
/// committed block's full-canvas argmax, returns true when it would render as
/// an EMPTY user-facing reply. It checks the actual output STATE — take the
/// answer region (up to the first stop/eos token), strip pad/filler, decode,
/// and `sanitize_model_reply` — rather than blocklisting leading token ids.
/// This catches both shapes of the attractor with one rule: the eos-first
/// "empty" canvas AND the `<|channel>thought` ceremony (sanitize erases the
/// thought-channel scaffold to nothing). Owns the tokenizer by move so it can
/// live in `StepGenerateConfig` as a `Send + Sync` closure.
pub fn empty_reply_predicate(
    tok: Tokenizer,
    stop_ids: Vec<u32>,
    eos_token_id: u32,
) -> impl Fn(&[u32]) -> bool + Send + Sync {
    move |argmax: &[u32]| {
        let end = argmax
            .iter()
            .position(|id| *id == eos_token_id || stop_ids.contains(id))
            .unwrap_or(argmax.len());
        let region = crate::sample::strip_degenerate_token_ids(&argmax[..end]);
        sanitize_model_reply(&tok.decode(&region)).is_empty()
    }
}

/// Build the E6 empty-reply predicate for `StepGenerateConfig`, or `None` when
/// the retry is disabled (`DGQ_EMPTY_REPLY_RETRY=0`) or the tokenizer/config
/// can't be loaded — so the re-roll only ever activates when enabled.
pub fn empty_reply_check(
    model_dir: &std::path::Path,
    stop_ids: Vec<u32>,
) -> Option<std::sync::Arc<dyn Fn(&[u32]) -> bool + Send + Sync>> {
    if crate::flags::empty_reply_retry() == 0 {
        return None;
    }
    let tok = Tokenizer::load(model_dir.join("tokenizer.json")).ok()?;
    let eos = crate::config::ModelConfig::load(model_dir)
        .ok()?
        .eos_token_id_u32();
    Some(std::sync::Arc::new(empty_reply_predicate(tok, stop_ids, eos)))
}

fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::User => "user",
        ChatRole::Model => "model",
    }
}

fn push_special(out: &mut Vec<u32>, tok: &Tokenizer, token: &str) -> Result<(), Error> {
    let id = tok
        .special_token_id(token)
        .ok_or(Error::Format("missing chat special token"))?;
    out.push(id);
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
    for turn in turns {
        push_special(&mut out, tok, TURN_OPEN)?;
        tok.encode_append(&mut out, &format!("{}\n", role_name(turn.role)));
        tok.encode_append(&mut out, turn.content.trim());
        push_special(&mut out, tok, TURN_CLOSE)?;
        tok.encode_append(&mut out, "\n");
    }
    if opts.add_generation_prompt {
        push_special(&mut out, tok, TURN_OPEN)?;
        tok.encode_append(&mut out, "model\n");
        if !opts.enable_thinking {
            push_special(&mut out, tok, CHANNEL_OPEN)?;
            tok.encode_append(&mut out, "thought\n");
            push_special(&mut out, tok, CHANNEL_CLOSE)?;
        }
    }
    Ok(out)
}

/// Format a conversation string for debugging (not used for encode).
pub fn format_chat_prompt(turns: &[ChatTurn], opts: &ChatFormatOptions) -> String {
    let mut s = String::from(BOS_TOKEN);
    for turn in turns {
        s.push_str(TURN_OPEN);
        s.push_str(role_name(turn.role));
        s.push('\n');
        s.push_str(turn.content.trim());
        s.push_str(TURN_CLOSE);
        s.push('\n');
    }
    if opts.add_generation_prompt {
        s.push_str(TURN_OPEN);
        s.push_str("model\n");
        if !opts.enable_thinking {
            s.push_str(CHANNEL_OPEN);
            s.push_str("thought\n");
            s.push_str(CHANNEL_CLOSE);
        }
    }
    s
}

/// Wrap a single user message (debug string).
pub fn format_user_prompt(user_text: &str) -> String {
    format_chat_prompt(
        &[ChatTurn::user(user_text)],
        &ChatFormatOptions::default(),
    )
}

/// Strip control tokens from decoded model output before storing in history.
pub fn sanitize_model_reply(text: &str) -> String {
    let mut s = text.to_string();
    if let Some(idx) = s.find(TURN_CLOSE) {
        s.truncate(idx);
    }
    if let Some(idx) = s.find(TURN_OPEN) {
        s.truncate(idx);
    }
    s = s.replace(&format!("{CHANNEL_OPEN}thought\n{CHANNEL_CLOSE}"), "");
    s = s.replace(&format!("{CHANNEL_OPEN}thought\n"), "");
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_tokenizer() -> Option<Tokenizer> {
        let path = PathBuf::from("model/transformer/tokenizer.json");
        if path.exists() {
            Tokenizer::load(path).ok()
        } else {
            None
        }
    }

    #[test]
    fn single_user_turn_includes_bos_and_gen_prompt() {
        let formatted = format_user_prompt("Why is the sky blue?");
        assert!(formatted.starts_with("<bos>"));
        assert!(formatted.contains("<|turn>user\nWhy is the sky blue?<turn|>\n"));
        assert!(formatted.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn multi_turn_history() {
        let turns = [
            ChatTurn::user("Hi"),
            ChatTurn::model("Hello!"),
            ChatTurn::user("Tell me more"),
        ];
        let formatted = format_chat_prompt(&turns, &ChatFormatOptions::default());
        assert!(formatted.contains("<|turn>user\nHi<turn|>\n"));
        assert!(formatted.contains("<|turn>model\nHello!<turn|>\n"));
        assert!(formatted.contains("<|turn>user\nTell me more<turn|>\n"));
        assert!(formatted.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
    }

    #[test]
    fn sanitize_strips_turn_close() {
        assert_eq!(sanitize_model_reply("Hello world<turn|>\n"), "Hello world");
    }

    #[test]
    fn sky_blue_token_ids_match_hf_reference_when_model_present() {
        let Some(tok) = model_tokenizer() else {
            return;
        };
        let ids = format_chat_token_ids(
            &tok,
            &[ChatTurn::user("Why is the sky blue?")],
            &ChatFormatOptions::default(),
        )
        .expect("format");
        let expected = [
            2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105, 4368, 107,
            100, 45518, 107, 101,
        ];
        assert_eq!(ids, expected);
    }
}
