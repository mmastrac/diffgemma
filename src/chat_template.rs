//! DiffusionGemma / Gemma 4 text-only chat formatting.
//!
//! Token assembly matches HuggingFace `apply_chat_template` for the simple
//! text path: special tokens (`<bos>`, `<|turn>`, `<turn|>`, `<|channel>`,
//! `<channel|>`) are inserted by ID; role lines and content are BPE-encoded.

use crate::Error;
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

/// Predicate for empty/degenerate-reply canvas re-roll: given a
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

/// Build the empty-reply predicate for `StepGenerateConfig`, or `None` when
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
    Some(std::sync::Arc::new(empty_reply_predicate(
        tok, stop_ids, eos,
    )))
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
        .ok_or(Error::Runtime("missing chat special token"))?;
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
    format_chat_prompt(&[ChatTurn::user(user_text)], &ChatFormatOptions::default())
}

/// Strip control tokens from decoded model output before storing in history.
///
/// Also removes thought-channel spans and orphan `<channel|>` / `<|channel>`
/// markers. The model sometimes re-emits a close after the answer
/// (`…rustc.<channel|><|tool_call>…`); stream split only consumes the first
/// close, so leftovers must die here or they leak into OpenAI `content`.
///
/// Marker scanning is masked (`tools::masked_ranges`): a turn or channel
/// marker inside a complete call's quoted args or a closed code fence is
/// literal data — truncating there corrupted tool args whose content quoted
/// the chat grammar (e.g. an edit to a jinja chat template).
pub fn sanitize_model_reply(text: &str) -> String {
    // Absorb LEADING turn ceremony first: after a tool-response prompt the
    // generation opener is not seeded, so the model legitimately begins its
    // reply with `<|turn>model\n` — truncating at that turn-open would erase
    // the whole reply. Only trailing turn markers are hallucinated next
    // turns and get cut below.
    let mut s = text.trim_start().to_string();
    while let Some(rest) = s.strip_prefix(TURN_OPEN) {
        // Skip the role line ("model\n"); a marker with no newline yet is a
        // still-converging fragment — drop what's there.
        s = match rest.split_once('\n') {
            Some((_, body)) => body.trim_start().to_string(),
            None => String::new(),
        };
    }
    let masks = crate::tools::masked_ranges(&s);
    // Truncation at an unmasked position keeps every remaining mask intact
    // (a mask never straddles an unmasked byte), so one mask pass serves
    // both cuts.
    if let Some(idx) = crate::tools::find_unmasked(&s, &masks, TURN_CLOSE, 0) {
        s.truncate(idx);
    }
    if let Some(idx) = crate::tools::find_unmasked(&s, &masks, TURN_OPEN, 0) {
        s.truncate(idx);
    }
    // Thought spans + orphan channel markers (masked): same machine as the
    // tools-side splitter, one source of truth.
    crate::tools::strip_thinking(&s)
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
    fn system_prompt_renders_as_a_leading_block() {
        let Some(tok) = model_tokenizer() else {
            return;
        };
        let turns = [ChatTurn::system("You are a pirate."), ChatTurn::user("Hi")];
        let plain =
            tok.decode(&format_chat_token_ids(&tok, &turns, &ChatFormatOptions::default()).unwrap());
        assert!(plain.contains("<|turn>system\nYou are a pirate.<turn|>"));
    }

    #[test]
    fn sanitize_strips_turn_close() {
        assert_eq!(sanitize_model_reply("Hello world<turn|>\n"), "Hello world");
    }

    /// Post-tool-response turns get no seeded generation opener, so the model
    /// legitimately begins its reply with `<|turn>model\n` — that leading
    /// ceremony is absorbed, while a trailing turn-open (hallucinated next
    /// turn) still truncates.
    #[test]
    fn sanitize_absorbs_leading_turn_open_keeps_trailing_cut() {
        assert_eq!(sanitize_model_reply("<|turn>model\nHello"), "Hello");
        assert_eq!(sanitize_model_reply("Hi<|turn>user\nnext"), "Hi");
        assert_eq!(sanitize_model_reply("<|turn>model\nOk<|turn>user\nx"), "Ok");
        // Still-converging bare opener fragment: nothing usable yet.
        assert_eq!(sanitize_model_reply("<|turn>model"), "");
    }

    /// Turn/channel markers inside a complete call's quoted args are literal
    /// content (a written file quoting the chat grammar) — sanitize must not
    /// truncate or strip them, while the same markers OUTSIDE quotes still die.
    #[test]
    fn sanitize_keeps_quoted_markers_cuts_unquoted() {
        let call = "<|tool_call>call:edit{path:<|\"|>t.jinja<|\"|>,new:<|\"|>\
                    <|turn>user\n<turn|><|channel>x<channel|><|\"|>}<tool_call|>";
        let reply = format!("Updating.{call}");
        assert_eq!(sanitize_model_reply(&reply), reply);
        // Trailing hallucinated turn AFTER the call still truncates there.
        let with_tail = format!("{reply}<|turn>user\nnext");
        assert_eq!(sanitize_model_reply(&with_tail), reply);
    }

    #[test]
    fn sanitize_strips_trailing_channel_close() {
        assert_eq!(
            sanitize_model_reply("then I will run it using rustc.<channel|>"),
            "then I will run it using rustc."
        );
    }

    #[test]
    fn sanitize_strips_empty_thought_ceremony_and_orphan_close() {
        assert_eq!(
            sanitize_model_reply("<|channel>thought\n<channel|>Answer.<channel|>"),
            "Answer."
        );
        assert_eq!(
            sanitize_model_reply("<|channel>\n\n<channel|>Plan.<channel|>"),
            "Plan."
        );
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
            2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105, 4368, 107, 100,
            45518, 107, 101,
        ];
        assert_eq!(ids, expected);
    }
}
