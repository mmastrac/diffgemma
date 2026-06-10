//! Gemma BPE tokenizer loaded from HuggingFace `tokenizer.json`.

use crate::safetensors::Error;
use std::collections::HashMap;
use std::path::Path;

const SPACE_REPLACEMENT: char = '\u{2581}';

#[derive(Debug, serde::Deserialize)]
struct TokenizerJson {
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

#[allow(dead_code)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
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
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in &vocab {
            if id as usize <= max_id {
                id_to_token[id as usize] = token.clone();
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
            id_to_token,
            merge_ranks,
            byte_fallback: parsed.model.byte_fallback,
            unk_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
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
                heap.push(MergeJob { rank, pos: i, new_id });
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
}
