//! Tool-output context compaction for `serve` (the KV rewinder, M1+M2).
//!
//! The OpenAI protocol is stateless: clients re-send the full verbose history
//! every turn, so a huge tool response would be re-prefixed forever. When
//! compaction is on (`--tool-compact` / `DGQ_TOOL_COMPACT=1`), the server
//! instead:
//!
//! 1. detects over-threshold `tool` messages ([`find_compactable`]),
//! 2. asks the model itself for a `<summarize>…</summarize>` digest (a
//!    checkpoint→generate→rollback pass in server.rs — the KV never keeps the
//!    verbose text or the summarize scaffold),
//! 3. substitutes `{summary, full_output_id, note}` for the verbose content at
//!    the *messages* level ([`compact_messages`]) — the canonical grammar
//!    renders object content natively, and because the id is derived from a
//!    content hash the substitution is deterministic across turns, keeping the
//!    rendered prompt a prefix-match for the conversation's compacted KV,
//! 4. stores the full output on disk ([`ToolOutputStore`]) for on-demand
//!    retrieval via the built-in [`expand_summary_tool`], executed server-side
//!    ([`dispatch_expand`]) so excerpts stay turn-scoped (finalize evicts
//!    them from KV).
//!
//! Everything in this module is pure or filesystem-only — no GPU, fully
//! unit-tested. The generation plumbing lives in server.rs.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;

/// Name of the built-in retrieval tool appended to the user's tool list.
pub const EXPAND_TOOL_NAME: &str = "expand_summary";

// ---------------------------------------------------------------------------
// Content-derived ids
// ---------------------------------------------------------------------------

/// FNV-1a 64 (hand-rolled: restart-stable, unlike `DefaultHasher`). The id
/// must be reproducible across server runs so a client re-sending the same
/// verbose history keeps hitting the same stored output.
pub fn fnv1a64(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `tr_xxxxxxxx` — 8 hex chars folded from the 64-bit hash.
pub fn output_id(hash: u64) -> String {
    format!("tr_{:08x}", (hash >> 32) as u32 ^ hash as u32)
}

// ---------------------------------------------------------------------------
// Full-output store
// ---------------------------------------------------------------------------

/// One compacted tool output: its stable id, the model's summary, and where
/// the full text lives on disk.
pub struct StoredOutput {
    pub id: String,
    pub summary: String,
    pub path: PathBuf,
}

/// Disk store for full tool outputs, keyed by content hash. Mirrors the
/// conversation snapshot store's lifecycle: startup sweep for files left by a
/// hard-killed run, per-file cleanup on Drop (graceful exit).
pub struct ToolOutputStore {
    dir: PathBuf,
    by_hash: HashMap<u64, StoredOutput>,
    by_id: HashMap<String, u64>,
}

impl ToolOutputStore {
    pub fn new(dir: PathBuf) -> Self {
        remove_stale_outputs(&dir);
        Self {
            dir,
            by_hash: HashMap::new(),
            by_id: HashMap::new(),
        }
    }

    pub fn get(&self, hash: u64) -> Option<&StoredOutput> {
        self.by_hash.get(&hash)
    }

    /// Store a full output + its summary. Idempotent per hash (a re-put keeps
    /// the existing entry). The file write must succeed before the entry is
    /// recorded — a missing blob would make `expand_summary` lie.
    pub fn put(
        &mut self,
        hash: u64,
        full_text: &str,
        summary: String,
    ) -> std::io::Result<&StoredOutput> {
        if self.by_hash.contains_key(&hash) {
            return Ok(&self.by_hash[&hash]);
        }
        let id = output_id(hash);
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{id}.txt"));
        std::fs::write(&path, full_text)?;
        self.by_id.insert(id.clone(), hash);
        self.by_hash
            .insert(hash, StoredOutput { id, summary, path });
        Ok(&self.by_hash[&hash])
    }

    /// Full text by id (as the model references it in an `expand_summary` call).
    pub fn read_full(&self, id: &str) -> Option<String> {
        let hash = self.by_id.get(id)?;
        let out = self.by_hash.get(hash)?;
        std::fs::read_to_string(&out.path).ok()
    }
}

impl Drop for ToolOutputStore {
    fn drop(&mut self) {
        for out in self.by_hash.values() {
            let _ = std::fs::remove_file(&out.path);
        }
        let _ = std::fs::remove_dir(&self.dir); // only if now empty
    }
}

/// Best-effort sweep of `tr_*.txt` blobs left by a previous hard-killed server
/// (Drop never ran). Same convention as the conversation store's `*.kv` sweep.
fn remove_stale_outputs(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("tr_") && name.ends_with(".txt") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    let _ = std::fs::remove_dir(dir);
}

// ---------------------------------------------------------------------------
// Message-level detection + substitution
// ---------------------------------------------------------------------------

/// A tool message worth compacting.
pub struct CompactCandidate {
    /// Index into the request's `messages` array.
    pub idx: usize,
    /// Content hash of the flattened text (the store key).
    pub hash: u64,
    /// The flattened verbose text.
    pub text: String,
}

/// Flatten a tool message's content to text, or None when it isn't
/// compactable: only string / text-parts content qualifies. Object content is
/// skipped — user-supplied structured payloads render field-by-field, and our
/// own substituted `{summary, full_output_id}` objects must never be
/// re-compacted (idempotence).
fn compactable_text(msg: &Value) -> Option<String> {
    if msg.get("role").and_then(Value::as_str) != Some("tool") {
        return None;
    }
    match msg.get("content") {
        Some(Value::String(_)) | Some(Value::Array(_)) => {
            let text = crate::tools::message_text(msg);
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Over-threshold tool messages, in message order. `count` (a token counter)
/// is injected so the pure logic tests without a tokenizer.
pub fn find_compactable(
    messages: &[Value],
    threshold: usize,
    count: &dyn Fn(&str) -> usize,
) -> Vec<CompactCandidate> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            let text = compactable_text(m)?;
            (count(&text) > threshold).then(|| CompactCandidate {
                idx,
                hash: fnv1a64(&text),
                text,
            })
        })
        .collect()
}

/// The object that replaces a verbose tool response. Renders canonically via
/// the existing object branch of the tool-response formatter.
pub fn substituted_content(id: &str, summary: &str) -> Value {
    json!({
        "summary": summary,
        "full_output_id": id,
        "note": "full output available via expand_summary",
    })
}

/// Substitute every over-threshold tool message whose hash `resolve` knows
/// (id, summary). Unknown hashes pass through verbatim (they get summarized —
/// and become resolvable — before the final render). Pure; only `content` is
/// replaced, so `tool_call_id`/`name` pairing is preserved.
pub fn compact_messages(
    messages: &[Value],
    threshold: usize,
    count: &dyn Fn(&str) -> usize,
    resolve: &dyn Fn(u64) -> Option<(String, String)>,
) -> Vec<Value> {
    let mut out = messages.to_vec();
    for cand in find_compactable(messages, threshold, count) {
        if let Some((id, summary)) = resolve(cand.hash) {
            out[cand.idx]["content"] = substituted_content(&id, &summary);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Summarize pass (pure parts)
// ---------------------------------------------------------------------------

/// The instruction turn appended after the verbose tool response for the
/// summarize pass.
pub fn summarize_instruction() -> String {
    "Summarize the critical information from this tool response, delimited \
     with XML <summarize></summarize> tags. The tool response will be replaced \
     with your summary, so make it short and compact to conserve context \
     space. A follow-up tool call (expand_summary) will be available to search \
     the full, unsummarized tool output if needed."
        .to_string()
}

/// First `<summarize>…</summarize>` span, trimmed. Total: any malformed input
/// (missing/unterminated tags, empty body) yields `None`.
pub fn extract_summary(text: &str) -> Option<String> {
    let start = text.find("<summarize>")? + "<summarize>".len();
    let end = start + text[start..].find("</summarize>")?;
    let body = text[start..end].trim();
    (!body.is_empty()).then(|| body.to_string())
}

/// Fallback when the model produced no usable summary: head + tail of the raw
/// text, capped at ~`max_chars`. Char-boundary safe.
pub fn mechanical_summary(text: &str, max_chars: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let half = max_chars / 2;
    let head_end = text.char_indices().nth(half).map_or(text.len(), |(i, _)| i);
    let tail_start = text
        .char_indices()
        .rev()
        .nth(half.saturating_sub(1))
        .map_or(0, |(i, _)| i);
    format!(
        "{}\n[…truncated…]\n{}",
        text[..head_end].trim_end(),
        text[tail_start..].trim_start()
    )
}

// ---------------------------------------------------------------------------
// expand_summary: declaration + server-side dispatch
// ---------------------------------------------------------------------------

/// OpenAI-shape declaration for the built-in retrieval tool. Appended (last)
/// to the user's tool list on every render so the system-turn prefix stays
/// stable for KV reuse.
pub fn expand_summary_tool() -> Value {
    json!({"type": "function", "function": {
        "name": EXPAND_TOOL_NAME,
        "description": "Retrieve part of a full, unsummarized tool output that was replaced by a summary. Use the full_output_id from the summary object.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "description": "the full_output_id of the stored output"},
            "mode": {"type": "string", "description": "how to slice the output", "enum": ["head", "tail", "lines", "grep"]},
            "pattern": {"type": "string", "description": "substring to search for (grep mode)"},
            "start": {"type": "integer", "description": "first line, 1-indexed (lines mode)"},
            "count": {"type": "integer", "description": "max lines to return (default 40)"}},
            "required": ["id", "mode"]}}})
}

/// Model-visible error text for a failed expand (unknown id, bad args, …).
pub fn expand_error(msg: &str) -> String {
    format!("expand_summary error: {msg}")
}

/// Execute an `expand_summary` call against the stored full text. Pure; every
/// failure mode returns a model-visible string, never an Err.
pub fn dispatch_expand(args: &Value, full_text: &str) -> String {
    let mode = args.get("mode").and_then(Value::as_str).unwrap_or("");
    let count = args
        .get("count")
        .and_then(Value::as_u64)
        .map(|c| c as usize)
        .filter(|&c| c > 0)
        .unwrap_or(40);
    let lines: Vec<&str> = full_text.lines().collect();
    let n = lines.len();
    match mode {
        "head" => {
            let take = count.min(n);
            let mut out = lines[..take].join("\n");
            if take < n {
                out.push_str(&format!("\n[{} more lines]", n - take));
            }
            out
        }
        "tail" => {
            let take = count.min(n);
            let mut out = String::new();
            if take < n {
                out.push_str(&format!("[{} earlier lines]\n", n - take));
            }
            out.push_str(&lines[n - take..].join("\n"));
            out
        }
        "lines" => {
            let start = args
                .get("start")
                .and_then(Value::as_u64)
                .map(|s| s as usize)
                .filter(|&s| s > 0)
                .unwrap_or(1);
            if start > n {
                return expand_error(&format!("start {start} is past the end ({n} lines)"));
            }
            let end = (start - 1 + count).min(n);
            let mut out = lines[start - 1..end].join("\n");
            if end < n {
                out.push_str(&format!("\n[{} more lines]", n - end));
            }
            out
        }
        "grep" => {
            let Some(pattern) = args
                .get("pattern")
                .and_then(Value::as_str)
                .filter(|p| !p.is_empty())
            else {
                return expand_error("grep mode requires a non-empty 'pattern'");
            };
            let matches: Vec<(usize, &str)> = lines
                .iter()
                .enumerate()
                .filter(|(_, l)| l.contains(pattern))
                .map(|(i, l)| (i + 1, *l))
                .collect();
            if matches.is_empty() {
                return format!("no lines match {pattern:?} ({n} lines searched)");
            }
            let take = count.min(matches.len());
            let mut out = matches[..take]
                .iter()
                .map(|(i, l)| format!("{i}:{l}"))
                .collect::<Vec<_>>()
                .join("\n");
            if take < matches.len() {
                out.push_str(&format!("\n[{} more matches]", matches.len() - take));
            }
            out
        }
        other => expand_error(&format!(
            "unknown mode {other:?} (expected head|tail|lines|grep)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn words(s: &str) -> usize {
        s.split_whitespace().count()
    }

    // -- ids ----------------------------------------------------------------

    #[test]
    fn hash_and_id_are_stable_literals() {
        // Restart determinism: these values must never change.
        assert_eq!(fnv1a64(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64("hello"), 0xa430_d846_80aa_bd0b);
        assert_eq!(output_id(fnv1a64("hello")), "tr_249a654d");
    }

    // -- detection ----------------------------------------------------------

    fn tool_msg(text: &str) -> Value {
        json!({"role": "tool", "tool_call_id": "call_0", "content": text})
    }

    #[test]
    fn find_compactable_gates_on_threshold_and_shape() {
        let big = "word ".repeat(20);
        let messages = vec![
            json!({"role": "user", "content": big.clone()}), // wrong role
            tool_msg("short"),                               // under threshold
            tool_msg(&big),                                  // YES
            json!({"role": "tool", "content": {"already": "object"}}), // object skipped
            json!({"role": "tool", "content": substituted_content("tr_x", "s")}), // idempotence
            tool_msg(&big),                                  // YES (same hash as idx 2)
        ];
        let found = find_compactable(&messages, 10, &words);
        assert_eq!(found.iter().map(|c| c.idx).collect::<Vec<_>>(), vec![2, 5]);
        assert_eq!(found[0].hash, found[1].hash);
        // message_text passes string content through verbatim (no trim) — the
        // hash must key on exactly what the client keeps re-sending.
        assert_eq!(found[0].text, big);
    }

    #[test]
    fn find_compactable_flattens_parts_content() {
        let msg = json!({"role": "tool", "content": [
            {"type": "text", "text": "one two three"},
            {"type": "text", "text": "four five six"},
        ]});
        let found = find_compactable(&[msg], 4, &words);
        assert_eq!(found.len(), 1);
        assert!(found[0].text.contains("four five six"));
    }

    // -- substitution ---------------------------------------------------------

    #[test]
    fn compact_messages_substitutes_known_and_keeps_pairing() {
        let big = "word ".repeat(20);
        let hash = fnv1a64(&big);
        let messages = vec![
            json!({"role": "assistant", "tool_calls": [{"id": "call_0"}]}),
            tool_msg(&big),
        ];
        let resolve = |h: u64| (h == hash).then(|| ("tr_test".into(), "the gist".into()));
        let out = compact_messages(&messages, 10, &words, &resolve);
        assert_eq!(
            out[1]["content"],
            substituted_content("tr_test", "the gist")
        );
        // Pairing metadata untouched.
        assert_eq!(out[1]["tool_call_id"], "call_0");
        assert_eq!(out[0], messages[0]);
    }

    #[test]
    fn compact_messages_is_idempotent_and_passes_unresolved() {
        let big = "word ".repeat(20);
        let messages = vec![tool_msg(&big)];
        // Unknown hash: verbatim passthrough.
        let none = |_: u64| None;
        assert_eq!(compact_messages(&messages, 10, &words, &none), messages);
        // Known: substituted once; a second pass is a no-op (object content).
        let some = |_: u64| Some(("tr_a".to_string(), "sum".to_string()));
        let once = compact_messages(&messages, 10, &words, &some);
        assert_eq!(compact_messages(&once, 10, &words, &some), once);
    }

    // -- summary extraction ---------------------------------------------------

    #[test]
    fn extract_summary_happy_and_malformed() {
        assert_eq!(
            extract_summary("ok <summarize> the gist </summarize> tail"),
            Some("the gist".to_string())
        );
        // First span wins.
        assert_eq!(
            extract_summary("<summarize>a</summarize><summarize>b</summarize>"),
            Some("a".to_string())
        );
        assert_eq!(extract_summary("no tags at all"), None);
        assert_eq!(extract_summary("<summarize>unterminated"), None);
        assert_eq!(extract_summary("</summarize><summarize>"), None);
        assert_eq!(extract_summary("<summarize>   </summarize>"), None);
        // Multi-byte safety.
        assert_eq!(
            extract_summary("naïve <summarize>résumé</summarize>"),
            Some("résumé".to_string())
        );
    }

    #[test]
    fn mechanical_summary_head_tail_shape() {
        let text = (1..=100)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let s = mechanical_summary(&text, 40);
        assert!(s.contains("[…truncated…]"));
        assert!(s.starts_with("line1"));
        assert!(s.ends_with("line100"));
        // Short text passes through whole.
        assert_eq!(mechanical_summary(" short ", 40), "short");
        // Multi-byte boundary safety.
        let uni = "é".repeat(100);
        let s = mechanical_summary(&uni, 10);
        assert!(s.contains("[…truncated…]"));
    }

    // -- declaration ----------------------------------------------------------

    #[test]
    fn expand_declaration_renders_byte_canonically() {
        let expected = "<|tool>declaration:expand_summary{description:<|\"|>Retrieve part of a full, unsummarized tool output that was replaced by a summary. Use the full_output_id from the summary object.<|\"|>,parameters:{properties:{count:{description:<|\"|>max lines to return (default 40)<|\"|>,type:<|\"|>INTEGER<|\"|>},id:{description:<|\"|>the full_output_id of the stored output<|\"|>,type:<|\"|>STRING<|\"|>},mode:{description:<|\"|>how to slice the output<|\"|>,enum:[<|\"|>head<|\"|>,<|\"|>tail<|\"|>,<|\"|>lines<|\"|>,<|\"|>grep<|\"|>],type:<|\"|>STRING<|\"|>},pattern:{description:<|\"|>substring to search for (grep mode)<|\"|>,type:<|\"|>STRING<|\"|>},start:{description:<|\"|>first line, 1-indexed (lines mode)<|\"|>,type:<|\"|>INTEGER<|\"|>}},required:[<|\"|>id<|\"|>,<|\"|>mode<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
        assert_eq!(
            crate::tools::format_tool_declarations(&[expand_summary_tool()]),
            expected
        );
    }

    // -- dispatch ---------------------------------------------------------------

    const TEN_LINES: &str = "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10";

    #[test]
    fn dispatch_head_tail() {
        let head = dispatch_expand(&json!({"mode": "head", "count": 3}), TEN_LINES);
        assert_eq!(head, "l1\nl2\nl3\n[7 more lines]");
        let tail = dispatch_expand(&json!({"mode": "tail", "count": 2}), TEN_LINES);
        assert_eq!(tail, "[8 earlier lines]\nl9\nl10");
        // count >= n: whole text, no marker.
        let all = dispatch_expand(&json!({"mode": "head", "count": 99}), TEN_LINES);
        assert_eq!(all, TEN_LINES);
    }

    #[test]
    fn dispatch_lines() {
        let mid = dispatch_expand(&json!({"mode": "lines", "start": 4, "count": 2}), TEN_LINES);
        assert_eq!(mid, "l4\nl5\n[5 more lines]");
        // start past end → model-visible error.
        let err = dispatch_expand(&json!({"mode": "lines", "start": 42}), TEN_LINES);
        assert!(err.starts_with("expand_summary error:"));
        // default start=1.
        let first = dispatch_expand(&json!({"mode": "lines", "count": 1}), TEN_LINES);
        assert_eq!(first, "l1\n[9 more lines]");
    }

    #[test]
    fn dispatch_grep() {
        let text = "alpha\nbeta\nalpha beta\ngamma";
        let hits = dispatch_expand(&json!({"mode": "grep", "pattern": "alpha"}), text);
        assert_eq!(hits, "1:alpha\n3:alpha beta");
        let capped = dispatch_expand(
            &json!({"mode": "grep", "pattern": "alpha", "count": 1}),
            text,
        );
        assert_eq!(capped, "1:alpha\n[1 more matches]");
        let none = dispatch_expand(&json!({"mode": "grep", "pattern": "zeta"}), text);
        assert!(none.starts_with("no lines match"));
        let bad = dispatch_expand(&json!({"mode": "grep"}), text);
        assert!(bad.starts_with("expand_summary error:"));
    }

    #[test]
    fn dispatch_bad_mode_and_args() {
        assert!(
            dispatch_expand(&json!({"mode": "explode"}), TEN_LINES)
                .starts_with("expand_summary error:")
        );
        assert!(dispatch_expand(&json!({}), TEN_LINES).starts_with("expand_summary error:"));
        // count=0 falls back to the default rather than an empty slice.
        let head = dispatch_expand(&json!({"mode": "head", "count": 0}), "a\nb");
        assert_eq!(head, "a\nb");
    }

    // -- store -------------------------------------------------------------------

    fn temp_store_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dgq-toolcompact-test-{tag}-{}", std::process::id()))
    }

    #[test]
    fn store_roundtrip_and_cleanup() {
        let dir = temp_store_dir("roundtrip");
        let hash = fnv1a64("full text");
        let path;
        {
            let mut store = ToolOutputStore::new(dir.clone());
            let out = store.put(hash, "full text", "sum".into()).unwrap();
            let id = out.id.clone();
            path = out.path.clone();
            assert_eq!(store.get(hash).unwrap().summary, "sum");
            assert_eq!(store.read_full(&id).as_deref(), Some("full text"));
            assert!(store.read_full("tr_bogus").is_none());
            // Idempotent re-put keeps the original summary.
            let again = store.put(hash, "full text", "other".into()).unwrap();
            assert_eq!(again.summary, "sum");
        }
        // Drop removed the blob (and the dir, since it's empty).
        assert!(!path.exists());
        assert!(!dir.exists());
    }

    #[test]
    fn store_startup_sweeps_stale_blobs() {
        let dir = temp_store_dir("sweep");
        std::fs::create_dir_all(&dir).unwrap();
        let stale = dir.join("tr_deadbeef.txt");
        let other = dir.join("keep.log");
        std::fs::write(&stale, "old").unwrap();
        std::fs::write(&other, "keep").unwrap();
        let _store = ToolOutputStore::new(dir.clone());
        assert!(!stale.exists());
        assert!(other.exists()); // only our naming pattern is swept
        std::fs::remove_file(&other).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    // -- renderer invariants -------------------------------------------------------

    /// Locks the KV-reuse invariant at the render level: the canonical render
    /// of a completed compacted turn is a byte prefix of the next turn's
    /// generation render.
    #[test]
    fn substituted_canonical_is_prefix_of_next_turn_render() {
        let tools = vec![
            json!({"type": "function", "function": {"name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties":
                    {"path": {"type": "string", "description": "path"}},
                    "required": ["path"]}}}),
            expand_summary_tool(),
        ];
        let mut messages = vec![
            json!({"role": "user", "content": "read /etc/hosts"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\":\"/etc/hosts\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "call_0",
                   "content": substituted_content("tr_ab12cd34", "hosts file: localhost entries")}),
            json!({"role": "assistant", "content": "The hosts file maps localhost."}),
        ];
        let canonical = crate::tools::render_conversation(&messages, &tools, false, false);
        messages.push(json!({"role": "user", "content": "anything else?"}));
        let next = crate::tools::render_conversation(&messages, &tools, true, false);
        assert!(
            next.starts_with(&canonical),
            "canonical must prefix the next render:\n{canonical}\nvs\n{next}"
        );
        // And the substituted object renders in the canonical grammar.
        assert!(canonical.contains("full_output_id:<|\"|>tr_ab12cd34<|\"|>"));
    }
}
