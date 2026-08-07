use serde_json::{Map, Value};

use super::Q;

// ===========================================================================
// Masked-marker scanning: which bytes of a reply are DATA, not protocol.
//
// Model text carries protocol markers (`<|channel>`, `<|tool_call>`, `call:`,
// turn tokens) *and* arbitrary literal content that may contain the same byte
// sequences — a tool call editing this repo's own `chat_template.jinja`, or
// prose quoting the grammar in a code fence. Every marker scan in this module
// (and `chat_template::sanitize_model_reply`) goes through these helpers so
// markers inside literal regions are inert instead of corrupting the reply.
// ===========================================================================

/// Byte ranges of `text` that are literal data:
///
/// - The braced ARG BODY (delimiters included) of every complete
///   `call:NAME{…}` — file contents and strings ride inside `<|"|>`-quoted
///   args, and `read_braced` already treats those runs as opaque, so this
///   pass is anchored at the grammar and immune to stray quote literals in
///   prose (global `<|"|>` parity is deliberately NOT used: one displayed
///   quote token in prose would flip it and hide every later real marker).
/// - CLOSED ``` code fences whose delimiters sit outside those arg bodies
///   and which do not swallow one (prose quoting markers or example calls).
///   An unclosed fence stays ordinary text: failing open there would let a
///   single stray fence hide every later real marker.
///
/// Ranges are ascending and non-overlapping.
pub(crate) fn masked_ranges(text: &str) -> Vec<(usize, usize)> {
    const FENCE: &str = "```";
    // Pass 1: complete call arg bodies, by the same consumption scan
    // `parse_tool_calls` uses.
    let mut masks: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = text[pos..].find("call:") {
        let at = pos + rel;
        let after = &text[at + "call:".len()..];
        let Some(brace) = after.find('{') else { break };
        match read_braced(&after[brace..]) {
            Some((_, consumed)) => {
                let start = at + "call:".len() + brace;
                masks.push((start, start + consumed));
                pos = start + consumed;
            }
            None => pos = at + "call:".len(),
        }
    }
    // Pass 2: closed fences outside the arg bodies.
    let args = masks.clone();
    let mut at = 0usize;
    while let Some(open) = find_unmasked(text, &args, FENCE, at) {
        let Some(close) = find_unmasked(text, &args, FENCE, open + FENCE.len()) else {
            break; // unclosed fence: ordinary text
        };
        let end = close + FENCE.len();
        // A pair straddling a real call's args would hide that call; treat
        // this opener as text and try the next one.
        if args.iter().any(|&(s, e)| open < s && e <= end) {
            at = open + FENCE.len();
            continue;
        }
        masks.push((open, end));
        at = end;
    }
    masks.sort_unstable();
    masks
}

/// Length of the prefix of `text` whose interpretation no appended suffix can
/// change. Both mask passes fail open on an unfinished construct (an unclosed
/// fence is ordinary text, an unbalanced call body is no call), so a later
/// chunk that completes the construct retroactively re-masks bytes from where
/// it started. Everything before the earliest unfinished construct is final:
/// masks are built left-to-right and a suffix can only start new constructs
/// at or after it. The chat renderer uses this to decide how much
/// block-committed text may be flushed into terminal scrollback, where it can
/// never be revised.
pub(crate) fn settled_prefix_len(text: &str) -> usize {
    let mut settled = text.len();
    // Pass-1 replica: a `call:` whose braced body never balances is a call
    // still forming; everything from the marker on may become an arg mask.
    let mut args: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = text[pos..].find("call:") {
        let at = pos + rel;
        let after = &text[at + "call:".len()..];
        let Some(brace) = after.find('{') else { break };
        match read_braced(&after[brace..]) {
            Some((_, consumed)) => {
                let start = at + "call:".len() + brace;
                args.push((start, start + consumed));
                pos = start + consumed;
            }
            None => {
                settled = settled.min(at);
                break;
            }
        }
    }
    // Pass-2 replica (same walk as `masked_ranges`, including the straddle
    // skip): a fence opener with no closer yet may become a mask spanning
    // from the opener once the closing ``` arrives.
    const FENCE: &str = "```";
    let mut at = 0usize;
    while let Some(open) = find_unmasked(text, &args, FENCE, at) {
        let Some(close) = find_unmasked(text, &args, FENCE, open + FENCE.len()) else {
            settled = settled.min(open);
            break;
        };
        let end = close + FENCE.len();
        if args.iter().any(|&(s, e)| open < s && e <= end) {
            at = open + FENCE.len();
            continue;
        }
        at = end;
    }
    settled
}

/// First occurrence of `needle` at/after byte `from` that does not start
/// inside a masked range. Masks must come from `masked_ranges(text)`.
pub(crate) fn find_unmasked(
    text: &str,
    masks: &[(usize, usize)],
    needle: &str,
    from: usize,
) -> Option<usize> {
    let mut at = from;
    while at <= text.len() {
        let rel = text[at..].find(needle)?;
        let pos = at + rel;
        match masks.iter().find(|&&(s, e)| pos >= s && pos < e) {
            Some(&(_, e)) => at = e,
            None => return Some(pos),
        }
    }
    None
}

/// Last unmasked occurrence of `needle` in `text`.
fn rfind_unmasked(text: &str, masks: &[(usize, usize)], needle: &str) -> Option<usize> {
    let mut last = None;
    let mut at = 0usize;
    while let Some(pos) = find_unmasked(text, masks, needle, at) {
        last = Some(pos);
        at = pos + needle.len();
    }
    last
}

/// The user-facing content of a tool-call response: the preamble before the
/// first `<|tool_call>` (the between-call / trailing text is dropped — clients
/// treat `content` as the preamble when `tool_calls` are present).
pub fn content_before_tool_calls(text: &str) -> String {
    let masks = masked_ranges(text);
    // The model may skip the `<|tool_call>` opener and go straight to the
    // grammar — a bare `call:` starts the calls just the same.
    let cut = [
        find_unmasked(text, &masks, "<|tool_call>", 0),
        find_unmasked(text, &masks, "call:", 0),
    ]
    .into_iter()
    .flatten()
    .min();
    match cut {
        Some(i) => text[..i].trim().to_string(),
        None => text.trim().to_string(),
    }
}

/// Parsed calls -> OpenAI `tool_calls` array (`arguments` serialized as a string).
///
/// Includes `index` on every entry so the same array is valid for both
/// `message.tool_calls` and streaming `delta.tool_calls`. OpenCode's chat
/// protocol keys stream deltas by `index`; omitting it leaves `undefined` and
/// breaks tool-call assembly.
pub fn to_openai_tool_calls(calls: &[ParsedToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "index": i,
                "id": format!("call_{i}"),
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments.to_string()},
            })
        })
        .collect()
}

/// Flatten a message's `content` (string or content-parts array) to text.
pub fn message_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Remove `<|channel>…<channel|>` thought spans (jinja `strip_thinking`).
///
/// Marker scanning is masked: a `<|channel>`/`<channel|>` inside a
/// `<|"|>`-quoted tool arg or a closed code fence is literal content and
/// neither opens nor closes a span. Unmasked semantics are unchanged from
/// the historical split-based version: stray closes are dropped, an
/// UNCLOSED span drops its text to the end, nested opens don't stack.
pub(crate) fn strip_thinking(text: &str) -> String {
    const OPEN: &str = "<|channel>";
    const CLOSE: &str = "<channel|>";
    let masks = masked_ranges(text);
    let mut out = String::new();
    let mut pos = 0usize;
    let mut in_thought = false;
    loop {
        if in_thought {
            match find_unmasked(text, &masks, CLOSE, pos) {
                Some(c) => {
                    pos = c + CLOSE.len();
                    in_thought = false;
                }
                None => break, // unclosed span: tail is thought, dropped
            }
        } else {
            let open = find_unmasked(text, &masks, OPEN, pos);
            let close = find_unmasked(text, &masks, CLOSE, pos);
            match (open, close) {
                (Some(o), Some(c)) if c < o => {
                    out.push_str(&text[pos..c]);
                    pos = c + CLOSE.len();
                }
                (Some(o), _) => {
                    out.push_str(&text[pos..o]);
                    pos = o + OPEN.len();
                    in_thought = true;
                }
                (None, Some(c)) => {
                    out.push_str(&text[pos..c]);
                    pos = c + CLOSE.len();
                }
                (None, None) => {
                    out.push_str(&text[pos..]);
                    break;
                }
            }
        }
    }
    out.trim().to_string()
}

// ===========================================================================
// Calls: the model's `<|tool_call>call:NAME{…}<tool_call|>` output → OpenAI shape
// ===========================================================================

/// A tool call parsed from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    /// Arguments as a JSON object (serialized to a string for the OpenAI field).
    pub arguments: Value,
}

/// True when `text` has a `call:NAME{…}` that is still open (no closing `}`),
/// or a `<|tool_call>` opener with no `call:` yet. Used to keep generating
/// blocks instead of ending the turn on a premature stop token.
pub fn has_incomplete_tool_call(text: &str) -> bool {
    let masks = masked_ranges(text);
    if let Some(i) = rfind_unmasked(text, &masks, "<|tool_call>")
        && find_unmasked(text, &masks, "call:", i).is_none()
    {
        return true;
    }
    let mut pos = 0usize;
    while let Some(at) = find_unmasked(text, &masks, "call:", pos) {
        let after = &text[at + "call:".len()..];
        let Some(brace) = after.find('{') else {
            return true;
        };
        match read_braced(&after[brace..]) {
            Some((_, consumed)) => {
                pos = at + "call:".len() + brace + consumed;
            }
            None => return true,
        }
    }
    false
}

/// True when a completed `call:NAME{…}` is followed by non-empty prose (after
/// optional `<tool_call|>` closers). The serve client drops that trailing text
/// when packaging `tool_calls`, so ending the turn there abandons mid-thought
/// self-talk (`…}<tool_call|>Wait, I` + `<eos>`) and the next request often
/// empty-eos's. Caller should keep denoising instead.
pub fn has_trailing_after_tool_calls(text: &str) -> bool {
    let masks = masked_ranges(text);
    let mut last_complete_end: Option<usize> = None;
    let mut base = 0usize;
    while base < text.len() {
        let Some(abs) = find_unmasked(text, &masks, "call:", base) else {
            break;
        };
        let after = &text[abs + "call:".len()..];
        let Some(brace) = after.find('{') else {
            break;
        };
        match read_braced(&after[brace..]) {
            Some((_, consumed)) => {
                let end = abs + "call:".len() + brace + consumed;
                last_complete_end = Some(end);
                base = end;
            }
            None => break,
        }
    }
    let Some(end) = last_complete_end else {
        return false;
    };
    let mut t = text[end..].trim_start();
    loop {
        let before = t;
        for closer in [
            "<tool_call|>",
            "</tool_call>",
            "</tool_call|>",
            "<|tool_call|>",
        ] {
            if let Some(s) = t.strip_prefix(closer) {
                t = s.trim_start();
            }
        }
        if t == before {
            break;
        }
    }
    !t.is_empty()
}

/// Tool-mode stop deferral: open tool call, or finished tool call(s) with
/// abandoned trailing prose that would be dropped from the OpenAI payload.
pub fn should_continue_past_stop(text: &str) -> bool {
    has_incomplete_tool_call(text) || has_trailing_after_tool_calls(text)
}

/// Every `call:` attempt in `text`, in order, with its name (may be empty
/// when unrecoverable) and whether it parses as a complete call. The repair
/// stage emits one error tool-response per invalid attempt.
pub fn scan_call_attempts(text: &str) -> Vec<(String, bool)> {
    let masks = masked_ranges(text);
    let mut out = Vec::new();
    let mut pos = 0usize;
    while let Some(at) = find_unmasked(text, &masks, "call:", pos) {
        let after = &text[at + "call:".len()..];
        let Some(brace) = after.find('{') else {
            out.push((
                after.split_whitespace().next().unwrap_or("").to_string(),
                false,
            ));
            break;
        };
        let name = after[..brace].trim().to_string();
        match read_braced(&after[brace..]) {
            Some((_, consumed)) => {
                out.push((name, true));
                pos = at + "call:".len() + brace + consumed;
            }
            None => {
                out.push((name, false));
                break; // unbalanced tail: nothing further can parse
            }
        }
    }
    out
}

/// Complete tool calls emitted INSIDE the thought channel while the visible
/// reply has none: the model acted in the wrong channel, and the thought
/// strip would silently discard its action. Rehearsing a call in thought
/// while ALSO emitting it visibly is common and fine — only the lost case
/// flags.
pub fn tool_call_lost_in_thinking(text: &str) -> bool {
    if !parse_tool_calls(&strip_thinking(text)).is_empty() {
        return false; // a visible call exists; thought content is moot
    }
    thinking_spans(text)
        .into_iter()
        .any(|(s, e)| !parse_tool_calls(&text[s..e]).is_empty())
}

/// Names of complete calls found inside thought spans (for the repair
/// stage's error responses).
pub fn thinking_call_names(text: &str) -> Vec<String> {
    thinking_spans(text)
        .into_iter()
        .flat_map(|(s, e)| parse_tool_calls(&text[s..e]).into_iter().map(|c| c.name))
        .collect()
}

/// Byte ranges of the BODIES of `<|channel>…<channel|>` thought spans (an
/// unclosed span runs to the end), with masked markers inert.
fn thinking_spans(text: &str) -> Vec<(usize, usize)> {
    const OPEN: &str = "<|channel>";
    const CLOSE: &str = "<channel|>";
    let masks = masked_ranges(text);
    let mut spans = Vec::new();
    let mut pos = 0usize;
    while let Some(open) = find_unmasked(text, &masks, OPEN, pos) {
        let body = open + OPEN.len();
        match find_unmasked(text, &masks, CLOSE, body) {
            Some(close) => {
                spans.push((body, close));
                pos = close + CLOSE.len();
            }
            None => {
                spans.push((body, text.len()));
                break;
            }
        }
    }
    spans
}

/// Strict reply-level grammar verdict for the validator stage: a reply that
/// speaks the tool grammar at all must parse cleanly and in the VISIBLE
/// channel. Plain prose is `Ok`. (Prose containing a bare `call:` is
/// flagged, consistent with `should_continue_past_stop` treating it as an
/// unfinished tool turn.)
pub fn validate_tool_reply(text: &str) -> Result<(), &'static str> {
    if has_incomplete_tool_call(text) {
        return Err("incomplete tool call");
    }
    if tool_call_lost_in_thinking(text) {
        return Err("tool call inside thinking block");
    }
    // Masked find: a `call:` inside a quoted arg or code fence is quoted
    // DATA, not an attempt to speak the grammar.
    let masks = masked_ranges(text);
    if find_unmasked(text, &masks, "call:", 0).is_some() && parse_tool_calls(text).is_empty() {
        return Err("tool-call grammar present but nothing parseable");
    }
    Ok(())
}

/// Parse every `call:NAME{ARGS}` in model output into structured tool calls,
/// tolerant of the exact wrapper tokens around/between them.
/// Recover tool calls the stream split dropped: parse the thought-stripped raw
/// decode when it is available, else the round reasoning. A call misclassified
/// into the reasoning must not silently end the turn as a plain answer.
pub(crate) fn salvage_lost_calls(raw_decode: Option<&str>, reasoning: &str) -> Vec<ParsedToolCall> {
    if let Some(raw) = raw_decode {
        let calls = parse_tool_calls(&strip_thinking(raw));
        if !calls.is_empty() {
            return calls;
        }
    }
    parse_tool_calls(reasoning)
}

pub fn parse_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let masks = masked_ranges(text);
    let mut out = Vec::new();
    let mut pos = 0usize;
    while let Some(at) = find_unmasked(text, &masks, "call:", pos) {
        let after = &text[at + "call:".len()..];
        let Some(brace) = after.find('{') else { break };
        // A malformed/unbalanced call must not drop the calls that follow it:
        // skip past this `call:` and keep scanning.
        let Some((inner, consumed)) = read_braced(&after[brace..]) else {
            pos = at + "call:".len();
            continue;
        };
        let name = after[..brace].trim().to_string();
        out.push(ParsedToolCall {
            name,
            arguments: parse_object(inner, 0),
        });
        pos = at + "call:".len() + brace + consumed;
    }
    out
}

/// Given a string starting with '{', return (inner content, bytes consumed incl.
/// both braces), respecting nested braces and `<|"|>`-quoted runs (braces inside
/// quotes do not affect depth -- otherwise `call:edit{oldString:<|"|>...{...}<|"|>}`
/// is reported unbalanced and the whole tool call is dropped).
fn read_braced(s: &str) -> Option<(&str, usize)> {
    let qb = Q.as_bytes();
    let b = s.as_bytes();
    if b.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < b.len() {
        if b[i..].starts_with(qb) {
            let after = i + qb.len();
            if let Some(rel) = find_bytes(&b[after..], qb) {
                i = after + rel + qb.len();
                continue;
            }
            // Unclosed quote: remaining bytes are opaque (no brace/comma effects).
            break;
        }
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[1..i], i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Cap on argument nesting depth — past it, a value is kept as a raw string
/// rather than recursed into (guards against runaway `{{{…}}}` output).
const MAX_ARG_DEPTH: u32 = 32;

/// Parse a `key:value,key:value` body (bare keys, `<|"|>`-quoted strings) into a
/// JSON object.
fn parse_object(inner: &str, depth: u32) -> Value {
    let mut map = Map::new();
    for field in split_top_level(inner) {
        let Some(colon) = field.find(':') else {
            continue;
        };
        let key = field[..colon].trim().trim_matches('"').to_string();
        if key.is_empty() {
            continue;
        }
        map.insert(key, parse_value(field[colon + 1..].trim(), depth));
    }
    Value::Object(map)
}

fn parse_value(s: &str, depth: u32) -> Value {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix(Q).and_then(|x| x.strip_suffix(Q)) {
        return Value::String(inner.to_string());
    }
    if let Some(inner) = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        return Value::String(inner.to_string());
    }
    if depth < MAX_ARG_DEPTH {
        if s.starts_with('{')
            && let Some((body, _)) = read_braced(s)
        {
            return parse_object(body, depth + 1);
        }
        if let Some(body) = s.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
            return Value::Array(
                split_top_level(body)
                    .into_iter()
                    .map(|v| parse_value(v, depth + 1))
                    .collect(),
            );
        }
    }
    match s {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    if let Ok(n) = serde_json::from_str::<serde_json::Number>(s) {
        return Value::Number(n);
    }
    Value::String(s.to_string())
}

/// Byte-substring search (needle in haystack) → start index.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Split on top-level commas, respecting `{}`/`[]` nesting and `<|"|>` strings.
///
/// Byte-safe: scans `s.as_bytes()` and only ever slices `s` at ASCII delimiter
/// positions (commas/brackets), which are always char boundaries — so multi-byte
/// UTF-8 anywhere in the input can never split mid-character (the previous
/// `s[i..]` scan panicked on e.g. a bare `naïve`).
fn split_top_level(s: &str) -> Vec<&str> {
    let qb = Q.as_bytes();
    let bytes = s.as_bytes();
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    let mut i = 0;
    while i < bytes.len() {
        // Skip a whole `<|"|>…<|"|>` quoted run so delimiters inside don't count.
        if bytes[i..].starts_with(qb) {
            let after = i + qb.len();
            if let Some(rel) = find_bytes(&bytes[after..], qb) {
                i = after + rel + qb.len();
                continue;
            }
            // Unclosed quote: remainder is opaque — do not split on commas/braces.
            break;
        }
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < s.len() || !s.is_empty() && start == s.len() {
        let tail = s[start..].trim();
        if !tail.is_empty() {
            parts.push(tail);
        }
    }
    parts
}
