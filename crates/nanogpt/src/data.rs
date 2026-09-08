//! Character-level corpus and tokenizer.
//!
//! The default corpus is a 120 KB slice of tiny shakespeare
//! (karpathy/char-rnn, public domain). Pass --data to use another text file;
//! the vocabulary is whatever bytes it contains.

use std::collections::HashMap;

pub const DEFAULT_TEXT: &str = include_str!("../data/tiny.txt");

pub struct Corpus {
    bytes: Vec<u8>,
    vocab: Vec<u8>,
    stoi: HashMap<u8, u32>,
}

impl Corpus {
    pub fn from_str(text: &str) -> Self {
        let bytes = text.as_bytes().to_vec();
        let mut vocab: Vec<u8> = bytes.clone();
        vocab.sort_unstable();
        vocab.dedup();
        let stoi = vocab
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, i as u32))
            .collect();
        Self { bytes, vocab, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Encode a string; unknown bytes are skipped.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        s.bytes().filter_map(|b| self.stoi.get(&b).copied()).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&i| self.vocab.get(i as usize).copied())
            .map(char::from)
            .collect()
    }

    /// A deterministic training crop: batch windows of block tokens plus their
    /// one-token-shifted targets.
    pub fn batch(&self, start: usize, batch: usize, block: usize) -> (Vec<u32>, Vec<u32>) {
        let n = self.bytes.len();
        let mut x = Vec::with_capacity(batch * block);
        let mut y = Vec::with_capacity(batch * block);
        for b in 0..batch {
            let off = (start + b * block * 7) % (n - block - 1);
            for t in 0..block {
                x.push(self.stoi[&self.bytes[off + t]]);
                y.push(self.stoi[&self.bytes[off + t + 1]]);
            }
        }
        (x, y)
    }
}
