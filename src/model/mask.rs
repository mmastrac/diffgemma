/// Decoder attention mask over `[canvas_len, kv_cache_len + canvas_len]`.
/// `attend[query * total_kv + key] == true` means the query may attend to that key.
#[derive(Debug, Clone)]
pub struct DecoderAttnMask {
    pub canvas_len: usize,
    pub kv_cache_len: usize,
    pub attend: Vec<bool>,
}

impl DecoderAttnMask {
    pub fn total_kv_len(&self) -> usize {
        self.kv_cache_len + self.canvas_len
    }

    /// All valid cache tokens + bidirectional canvas (DiffusionGemma default when unpadded).
    pub fn all_valid(canvas_len: usize, kv_cache_len: usize) -> Self {
        Self {
            canvas_len,
            kv_cache_len,
            attend: vec![true; canvas_len * (kv_cache_len + canvas_len)],
        }
    }

    pub fn can_attend(&self, query: usize, key: usize) -> bool {
        self.attend[query * self.total_kv_len() + key]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid_mask_canvas_bidirectional() {
        let mask = DecoderAttnMask::all_valid(4, 3);
        assert_eq!(mask.total_kv_len(), 7);
        // canvas query 0 attends to canvas key 3
        assert!(mask.can_attend(0, 6));
        // canvas query 3 attends to canvas key 0
        assert!(mask.can_attend(3, 3));
        // attends to kv
        assert!(mask.can_attend(2, 1));
    }
}
