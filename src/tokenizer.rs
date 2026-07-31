//! Gemma BPE tokenizer loaded from HuggingFace `tokenizer.json`.

use crate::Error;
use std::collections::HashMap;
use std::path::Path;

const SPACE_REPLACEMENT: char = '\u{2581}';

/// Sentinel bracketing client-supplied text in a GUARDED prompt render
/// (private-use codepoint; carries no meaning in real text). Emitted by the
/// tool-aware renderer around every client-controlled insertion — message
/// content, tool names/descriptions/keys, argument and response values — and
/// consumed by [`Tokenizer::encode_prompt`], which refuses to special-match
/// inside guarded ranges. Never reaches the model or the display render.
pub const CLIENT_GUARD: char = '\u{E000}';

#[derive(Debug, serde::Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    model: BpeModelJson,
}

#[derive(Debug, serde::Deserialize)]
struct BpeModelJson {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    merges: Vec<[String; 2]>,
    #[serde(default)]
    byte_fallback: bool,
    unk_token: Option<String>,
}

#[derive(Debug)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    byte_len: usize,
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    special_tokens: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merge_ranks: HashMap<(u32, u32), (u32, u32)>,
    byte_fallback: bool,
    unk_id: Option<u32>,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)?;
        let parsed: TokenizerJson = serde_json::from_str(&json)?;
        if parsed.model.model_type != "BPE" {
            return Err(Error::Format("unsupported tokenizer model type"));
        }

        let vocab = parsed.model.vocab;
        let mut special_tokens = HashMap::new();
        let mut max_id = vocab.values().copied().max().unwrap_or(0);
        for tok in &parsed.added_tokens {
            special_tokens.insert(tok.content.clone(), tok.id);
            max_id = max_id.max(tok.id);
        }
        let max_id = max_id as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in &vocab {
            if id as usize <= max_id {
                id_to_token[id as usize] = token.clone();
            }
        }
        for tok in &parsed.added_tokens {
            if tok.id as usize <= max_id {
                id_to_token[tok.id as usize] = tok.content.clone();
            }
        }

        let unk_id = parsed
            .model
            .unk_token
            .as_ref()
            .and_then(|t| vocab.get(t).copied());

        let mut merge_ranks = HashMap::new();
        for (rank, pair) in parsed.model.merges.iter().enumerate() {
            let left = &pair[0];
            let right = &pair[1];
            let merged = format!("{left}{right}");
            if let (Some(&l), Some(&r), Some(&m)) =
                (vocab.get(left), vocab.get(right), vocab.get(&merged))
            {
                merge_ranks.insert((l, r), (rank as u32, m));
            }
        }

        Ok(Self {
            vocab,
            special_tokens,
            id_to_token,
            merge_ranks,
            byte_fallback: parsed.model.byte_fallback,
            unk_id,
        })
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    /// Append BPE-encoded text (no added-token splitting).
    pub fn encode_append(&self, out: &mut Vec<u32>, text: &str) {
        out.extend(self.encode(text, false));
    }

    /// Encode `text` that may contain special-token *literals* (e.g. `<|turn>`,
    /// `<|tool_call>`, `<|"|>`) as their ids, BPE-encoding the runs between. Used
    /// to tokenize prompt strings built by the tool-aware renderer, which emits
    /// special tokens inline as text. Longest special match wins at each position.
    pub fn encode_with_specials(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut text_start = 0;
        let mut i = 0;
        while i < text.len() {
            if let Some((lit_len, id)) = self.match_special_at(&text[i..]) {
                if text_start < i {
                    self.encode_append(&mut out, &text[text_start..i]);
                }
                out.push(id);
                i += lit_len;
                text_start = i;
            } else {
                // Advance one char (keeps `i` on a UTF-8 boundary).
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if text_start < text.len() {
            self.encode_append(&mut out, &text[text_start..]);
        }
        out
    }

    /// Encode a GUARDED prompt render (client-supplied text wrapped in
    /// [`CLIENT_GUARD`] sentinel pairs by the tool-aware renderer): identical
    /// scan to [`encode_with_specials`](Self::encode_with_specials), except a
    /// special-token literal whose match STARTS inside a client-guarded range
    /// is NOT matched — it stays in the surrounding text run and BPE-encodes
    /// as plain characters. This closes the token-injection hole where a file
    /// body or web page containing `<|tool_response>`/`<|turn>user` literals
    /// became real protocol tokens. Guard-free input takes the exact same
    /// code path with the same run boundaries, so benign prompts encode
    /// bit-identically to `encode_with_specials`.
    ///
    /// Returns `(ids, neutralized)` — the count of special literals refused.
    pub fn encode_prompt(&self, guarded: &str) -> (Vec<u32>, usize) {
        // Strip sentinels, recording the client byte-ranges they delimited.
        let mut text = String::with_capacity(guarded.len());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut open: Option<usize> = None;
        for ch in guarded.chars() {
            if ch == CLIENT_GUARD {
                match open.take() {
                    Some(s) => ranges.push((s, text.len())),
                    None => open = Some(text.len()),
                }
            } else {
                text.push(ch);
            }
        }
        if let Some(s) = open {
            // Unbalanced guard: treat everything to the end as client text
            // (fail closed — over-guarding only costs special recognition).
            ranges.push((s, text.len()));
        }

        let mut out = Vec::new();
        let mut neutralized = 0usize;
        let mut text_start = 0;
        let mut i = 0;
        let mut r = 0;
        while i < text.len() {
            while r < ranges.len() && ranges[r].1 <= i {
                r += 1;
            }
            let in_client = r < ranges.len() && ranges[r].0 <= i && i < ranges[r].1;
            if let Some((lit_len, id)) = self.match_special_at(&text[i..]) {
                if in_client {
                    // Refuse the match: the literal stays in the pending text
                    // run; skip its bytes so nested shorter specials inside it
                    // are not re-matched.
                    neutralized += 1;
                    i += lit_len;
                    continue;
                }
                if text_start < i {
                    self.encode_append(&mut out, &text[text_start..i]);
                }
                out.push(id);
                i += lit_len;
                text_start = i;
            } else {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if text_start < text.len() {
            self.encode_append(&mut out, &text[text_start..]);
        }
        (out, neutralized)
    }

    /// Longest special-token literal that `s` starts with → (byte length, id).
    fn match_special_at(&self, s: &str) -> Option<(usize, u32)> {
        // Special tokens begin with '<'; cheap reject otherwise.
        if !s.starts_with('<') {
            return None;
        }
        self.special_tokens
            .iter()
            .filter(|(lit, _)| s.starts_with(lit.as_str()))
            .max_by_key(|(lit, _)| lit.len())
            .map(|(lit, id)| (lit.len(), *id))
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token
            .get(id as usize)
            .map(String::as_str)
            .filter(|s| !s.is_empty())
    }

    fn normalize(&self, text: &str) -> String {
        text.replace(' ', SPACE_REPLACEMENT.to_string().as_str())
    }

    fn pretokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut start = 0usize;
        for (idx, ch) in text.char_indices() {
            if ch == ' ' {
                if idx > start {
                    out.push(text[start..idx].to_string());
                }
                if let Some(last) = out.last_mut() {
                    last.push(' ');
                } else {
                    out.push(" ".to_string());
                }
                start = idx + ch.len_utf8();
            }
        }
        if start < text.len() {
            out.push(text[start..].to_string());
        }
        if out.is_empty() {
            out.push(text.to_string());
        }
        out
    }

    fn byte_token_id(&self, byte: u8) -> Option<u32> {
        let key = format!("<0x{byte:02X}>");
        self.vocab.get(&key).copied()
    }

    fn piece_to_symbols(&self, piece: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let char_indices: Vec<(usize, char)> = piece.char_indices().collect();
        let mut pending_unk: Option<(u32, usize)> = None;

        for (i, &(start, _ch)) in char_indices.iter().enumerate() {
            let end = char_indices
                .get(i + 1)
                .map(|(j, _)| *j)
                .unwrap_or(piece.len());
            let chunk = &piece[start..end];
            let byte_len = chunk.len();

            if let Some(&id) = self.vocab.get(chunk) {
                if let Some((unk_id, unk_len)) = pending_unk.take() {
                    Self::push_symbol(&mut symbols, unk_id, unk_len);
                }
                Self::push_symbol(&mut symbols, id, byte_len);
                continue;
            }

            if self.byte_fallback {
                let mut ids = Vec::new();
                let mut ok = true;
                for b in chunk.bytes() {
                    match self.byte_token_id(b) {
                        Some(id) => ids.push((id, 1usize)),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    if let Some((unk_id, unk_len)) = pending_unk.take() {
                        Self::push_symbol(&mut symbols, unk_id, unk_len);
                    }
                    for (id, len) in ids {
                        Self::push_symbol(&mut symbols, id, len);
                    }
                    continue;
                }
            }

            if let Some(unk_id) = self.unk_id {
                pending_unk = match pending_unk {
                    Some((id, len)) => Some((id, len + byte_len)),
                    None => Some((unk_id, byte_len)),
                };
            }
        }

        if let Some((unk_id, unk_len)) = pending_unk {
            Self::push_symbol(&mut symbols, unk_id, unk_len);
        }
        symbols
    }

    fn push_symbol(symbols: &mut Vec<Symbol>, id: u32, byte_len: usize) {
        let len = symbols.len() as isize;
        if let Some(last) = symbols.last_mut() {
            last.next = len;
        }
        symbols.push(Symbol {
            id,
            prev: if len == 0 { -1 } else { len - 1 },
            next: -1,
            byte_len,
        });
    }

    fn merge_symbols(&self, symbols: &mut Vec<Symbol>) {
        #[derive(Eq, PartialEq)]
        struct MergeJob {
            rank: u32,
            pos: usize,
            new_id: u32,
        }

        impl Ord for MergeJob {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other
                    .rank
                    .cmp(&self.rank)
                    .then_with(|| other.pos.cmp(&self.pos))
            }
        }
        impl PartialOrd for MergeJob {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = std::collections::BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            let left = symbols[i].id;
            let right = symbols[i + 1].id;
            if let Some(&(rank, new_id)) = self.merge_ranks.get(&(left, right)) {
                heap.push(MergeJob {
                    rank,
                    pos: i,
                    new_id,
                });
            }
        }

        while let Some(top) = heap.pop() {
            if top.pos >= symbols.len() || symbols[top.pos].byte_len == 0 {
                continue;
            }
            let next_pos = symbols[top.pos].next;
            if next_pos < 0 {
                continue;
            }
            let next_pos = next_pos as usize;
            if next_pos >= symbols.len() || symbols[next_pos].byte_len == 0 {
                continue;
            }

            let left_id = symbols[top.pos].id;
            let right_id = symbols[next_pos].id;
            let expected = self
                .merge_ranks
                .get(&(left_id, right_id))
                .map(|&(_, new_id)| new_id);
            if expected != Some(top.new_id) {
                continue;
            }

            let right_next = symbols[next_pos].next;
            symbols[top.pos].id = top.new_id;
            symbols[top.pos].byte_len += symbols[next_pos].byte_len;
            symbols[top.pos].next = right_next;
            symbols[next_pos].byte_len = 0;

            if right_next >= 0 && (right_next as usize) < symbols.len() {
                symbols[right_next as usize].prev = top.pos as isize;
            }

            if symbols[top.pos].prev >= 0 {
                let prev_pos = symbols[top.pos].prev as usize;
                let pair = (symbols[prev_pos].id, symbols[top.pos].id);
                if let Some(&(rank, new_id)) = self.merge_ranks.get(&pair) {
                    heap.push(MergeJob {
                        rank,
                        pos: prev_pos,
                        new_id,
                    });
                }
            }
            if symbols[top.pos].next >= 0 {
                let next = symbols[top.pos].next as usize;
                let pair = (symbols[top.pos].id, symbols[next].id);
                if let Some(&(rank, new_id)) = self.merge_ranks.get(&pair) {
                    heap.push(MergeJob {
                        rank,
                        pos: top.pos,
                        new_id,
                    });
                }
            }
        }

        symbols.retain(|s| s.byte_len > 0);
    }

    fn encode_piece(&self, piece: &str) -> Vec<u32> {
        let mut symbols = self.piece_to_symbols(piece);
        self.merge_symbols(&mut symbols);
        symbols.into_iter().map(|s| s.id).collect()
    }

    pub fn encode(&self, text: &str, _add_special_tokens: bool) -> Vec<u32> {
        let normalized = self.normalize(text);
        let pieces = self.pretokenize(&normalized);
        let mut ids = Vec::new();
        for piece in pieces {
            ids.extend(self.encode_piece(&piece));
        }
        ids
    }

    /// Decode token ids to text (Gemma SentencePiece: ▁ marks word starts).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        self.decode_append(&mut out, ids);
        out
    }

    /// Incremental decode: appends the text for `ids` to `out`. This mirrors
    /// [`decode`](Self::decode) but lets callers reuse a buffer across calls so
    /// streaming UIs don't re-decode the whole committed prefix every step.
    ///
    /// Matches `tokenizer.json`'s Replace decoder: every U+2581 (`▁`) becomes a
    /// space. Collapsing a leading-only `▁` (older behavior) turns indent tokens
    /// like `▁▁▁▁` (id 140) into a single space and loses nesting in tool writes.
    pub fn decode_append(&self, out: &mut String, ids: &[u32]) {
        for &id in ids {
            let Some(piece) = self.id_to_token(id) else {
                continue;
            };
            for ch in piece.chars() {
                if ch == SPACE_REPLACEMENT {
                    out.push(' ');
                } else {
                    out.push(ch);
                }
            }
        }
    }
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
    fn hello_matches_reference_when_model_present() {
        let Some(tok) = model_tokenizer() else {
            return;
        };
        assert_eq!(tok.encode("Hello", false), vec![9259]);
    }

    #[test]
    fn hello_world_matches_reference_when_model_present() {
        let Some(tok) = model_tokenizer() else {
            return;
        };
        assert_eq!(tok.encode("Hello world", false), vec![9259, 1902]);
    }

    #[test]
    fn decode_preserves_multi_space_indent() {
        let Some(tok) = model_tokenizer_q4_or_transformer() else {
            return;
        };
        // HF: "    return" -> [140, 2060] tokens ['▁▁▁▁','return'] -> "    return"
        let ids = tok.encode("    return", false);
        assert_eq!(ids, vec![140, 2060], "unexpected encode of 4-space indent");
        assert_eq!(tok.decode(&ids), "    return");
        assert_eq!(tok.decode(&[144, 584]), "        if"); // 8 spaces + if
        assert_eq!(tok.decode(&tok.encode("\treturn", false)), "\treturn");
    }

    fn model_tokenizer_q4_or_transformer() -> Option<Tokenizer> {
        for path in [
            PathBuf::from("model/diffgemma-26b-a4b-it-q4/tokenizer.json"),
            PathBuf::from("model/transformer/tokenizer.json"),
        ] {
            if path.exists() {
                return Tokenizer::load(path).ok();
            }
        }
        None
    }

    /// Guard-free input: `encode_prompt` == `encode_with_specials`, 0
    /// neutralized. This is the transparency contract for non-tool paths.
    #[test]
    fn encode_prompt_transparent_without_guards() {
        let Some(tok) = model_tokenizer_q4_or_transformer() else {
            return;
        };
        for s in [
            "Hello world",
            "<|turn>user\nhi<turn|>",
            "<|tool_call>call:x{a:<|\"|>b<|\"|>}<tool_call|>",
            "",
        ] {
            let (ids, n) = tok.encode_prompt(s);
            assert_eq!(n, 0, "unguarded input tripped the guard: {s:?}");
            assert_eq!(ids, tok.encode_with_specials(s), "drift on {s:?}");
        }
    }

    /// A special literal inside a guarded range encodes as PLAIN TEXT (byte-
    /// identical to BPE-ing the raw string), while an IDENTICAL literal
    /// outside the guard becomes its special id. Same string, opposite fate —
    /// the whole point of the mechanism.
    #[test]
    fn encode_prompt_guarded_literal_is_plain_text() {
        let Some(tok) = model_tokenizer_q4_or_transformer() else {
            return;
        };
        let g = CLIENT_GUARD;
        let turn_id = tok.encode_with_specials("<|turn>")[0];

        // Guarded: no special id, and the bytes equal a plain BPE of the text.
        let (guarded, n) = tok.encode_prompt(&format!("{g}see <|turn> here{g}"));
        assert_eq!(n, 1);
        assert!(
            !guarded.contains(&turn_id),
            "guarded literal leaked a special"
        );
        assert_eq!(guarded, tok.encode("see <|turn> here", false));

        // Unguarded: the same literal is the special id.
        let (open, _) = tok.encode_prompt("see <|turn> here");
        assert!(
            open.contains(&turn_id),
            "unguarded literal was not promoted"
        );

        // Mixed: guarded client text between two REAL template specials — the
        // structure survives, the injected middle does not.
        let (mixed, n) = tok.encode_prompt(&format!("<|turn>{g}<|turn>{g}<turn|>"));
        assert_eq!(n, 1);
        assert_eq!(
            mixed.iter().filter(|&&x| x == turn_id).count(),
            1,
            "exactly the one real leading <|turn> should be special"
        );
    }

    /// Unbalanced guard fails closed: everything after a lone open guard is
    /// treated as client text (over-guarding, never under-guarding).
    #[test]
    fn encode_prompt_unbalanced_guard_fails_closed() {
        let Some(tok) = model_tokenizer_q4_or_transformer() else {
            return;
        };
        let turn_id = tok.encode_with_specials("<|turn>")[0];
        let (ids, n) = tok.encode_prompt(&format!("<|turn>{}tail <|turn>", CLIENT_GUARD));
        assert_eq!(n, 1, "the post-open <|turn> must be neutralized");
        assert_eq!(
            ids.iter().filter(|&&x| x == turn_id).count(),
            1,
            "only the pre-guard <|turn> stays special"
        );
    }
}
