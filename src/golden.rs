//! Golden byte-identity pack — the Tier-1 refactoring gate.
//!
//! Generation is deterministic per seed (SC races fixed), so the strongest and
//! cheapest evidence that a refactor changed nothing is *byte identity*: the
//! exact token-id sequence AND the exact causal-KV bytes, for a fixed
//! (prompt, seed, config). This module holds the data model + compare logic;
//! the GPU harness lives in `main.rs::run_golden_cmd`, which drives the SAME
//! production path chat/serve use (`StepGenerateSession` + `generate_with_session`).
//!
//! The case matrix is chosen so each distinct code path is load-bearing in at
//! least one case — engine prefill (≤256), fast prefill (>256 super-chunk),
//! sliding-ring wrap (>2048), multi-block reply + fast block extend, cross-turn
//! KV reuse (delta prefill), the max-steps path (`no_early_stop`), tool-call
//! grammar emission, and full-layer attention at depth (longctx re-entry) —
//! plus a same-process determinism double-run that catches within-run
//! nondeterminism the cross-open golden can't.
//!
//! Byte-identity is BRITTLE BY DESIGN: any float reassociation (loop reorder,
//! accumulator change) breaks it even when harmless. For refactoring that is a
//! feature — it forces the "was this supposed to change numerics?" question —
//! but it means the pack is a refactor gate, not a portable CI oracle: fixtures
//! are blessed on one machine class (the 36GB M3 Pro dev box) and a genuine
//! numeric change is re-blessed only via `--bless-golden` after the Tier-2/3
//! ritual (multi-seed smoketest + longctx + census + sign-off).

use crate::safetensors::Error;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// FNV-1a over raw bytes (restart-stable change detector for the KV snapshot).
/// Mirrors `toolcompact::fnv1a64` but takes bytes, not `&str`.
pub fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `kv_<16 hex>` digest of a `snapshot_kv` byte blob.
pub fn kv_digest(bytes: &[u8]) -> String {
    format!("kv_{:016x}", fnv1a64_bytes(bytes))
}

/// One golden case: an input spec plus (once blessed) the recorded output. The
/// prompt-shape fields are mutually layered by the harness: `tool` → tool-call
/// render; else `doc` → "excerpt + question" (longctx); else `turns` → chat
/// (2+ turns exercise cross-turn KV reuse, no reset between them).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenCase {
    pub id: String,
    /// Human note: which code path this case makes load-bearing.
    pub note: String,
    /// User message(s). >1 entry = cross-turn KV reuse (the harness appends the
    /// model's own reply between turns, exactly like chat). Ignored when `doc`
    /// or `tool` is set (those use `turns[0]` as the single question/message).
    #[serde(default)]
    pub turns: Vec<String>,
    /// Longctx: document file (relative to the spec dir) whose first
    /// `doc_tokens` tokens are prepended to the question. Drives fast prefill /
    /// ring wrap / full-layer depth by `doc_tokens` alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(default)]
    pub doc_tokens: usize,
    /// Tool-call path: a single tool JSON declaration rendered alongside the
    /// message via `tools::render_conversation`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<serde_json::Value>,
    pub seed: u64,
    pub max_new_tokens: usize,
    /// Disable the entropy/plateau early stop → the max-steps denoise path.
    #[serde(default)]
    pub no_early_stop: bool,
    /// Run this case twice in one process and assert the two runs are identical
    /// (catches within-run nondeterminism, e.g. an SC race, that a single run
    /// might average past the golden).
    #[serde(default)]
    pub check_determinism: bool,
    /// Recorded golden — machine-owned, rewritten only by `--bless-golden`.
    /// `None` = never blessed → the gate fails loudly rather than passing empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub golden: Option<GoldenRecord>,
}

/// The recorded output of one case's production run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldenRecord {
    /// Final turn's prompt length in tokens (the prefill work under test).
    pub prompt_len: usize,
    /// `kv_valid_tokens().len()` after the run: prompt + causally-extended
    /// committed blocks (the final bidirectional canvas block is excluded).
    pub kv_valid_len: usize,
    /// The full emitted sequence (prompt + generated), byte-for-byte.
    pub token_ids: Vec<u32>,
    pub denoise_steps_run: usize,
    pub blocks_committed: usize,
    pub stopped_on_eot: bool,
    pub block_steps_eff: Vec<u32>,
    /// FNV digest of `snapshot_kv().kv_bytes` — the causal KV state a
    /// prefill/extend/ring refactor would perturb even if token_ids survive.
    pub kv_hash: String,
}

impl GoldenRecord {
    /// First divergence (most load-bearing invariant first: token_ids, then the
    /// KV hash, then the metadata). `Ok` = byte-identical.
    pub fn compare(&self, observed: &GoldenRecord) -> Result<(), String> {
        if self.token_ids != observed.token_ids {
            let at = self
                .token_ids
                .iter()
                .zip(observed.token_ids.iter())
                .position(|(a, b)| a != b);
            return Err(match at {
                Some(i) => format!(
                    "token_ids diverge at index {i}: golden={:?} got={:?} (len golden={} got={})",
                    self.token_ids.get(i),
                    observed.token_ids.get(i),
                    self.token_ids.len(),
                    observed.token_ids.len()
                ),
                None => format!(
                    "token_ids length differs: golden={} got={}",
                    self.token_ids.len(),
                    observed.token_ids.len()
                ),
            });
        }
        if self.kv_hash != observed.kv_hash {
            return Err(format!(
                "KV hash differs (token_ids identical): golden={} got={} (kv_valid_len golden={} got={}) — a KV-state change the argmax path didn't surface",
                self.kv_hash, observed.kv_hash, self.kv_valid_len, observed.kv_valid_len
            ));
        }
        if self.denoise_steps_run != observed.denoise_steps_run {
            return Err(format!(
                "denoise_steps_run: golden={} got={}",
                self.denoise_steps_run, observed.denoise_steps_run
            ));
        }
        if self.blocks_committed != observed.blocks_committed {
            return Err(format!(
                "blocks_committed: golden={} got={}",
                self.blocks_committed, observed.blocks_committed
            ));
        }
        if self.block_steps_eff != observed.block_steps_eff {
            return Err(format!(
                "block_steps_eff: golden={:?} got={:?}",
                self.block_steps_eff, observed.block_steps_eff
            ));
        }
        if self.stopped_on_eot != observed.stopped_on_eot {
            return Err(format!(
                "stopped_on_eot: golden={} got={}",
                self.stopped_on_eot, observed.stopped_on_eot
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenPack {
    pub cases: Vec<GoldenCase>,
}

impl GoldenPack {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(Error::Io)?;
        serde_json::from_str(&text).map_err(Error::Json)
    }

    pub fn write(&self, path: &Path) -> Result<(), Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(Error::Io)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(Error::Json)?;
        std::fs::write(path, format!("{text}\n")).map_err(Error::Io)
    }
}

pub fn default_pack_path() -> &'static Path {
    Path::new("fixtures/golden/golden.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(token_ids: Vec<u32>, kv_hash: &str) -> GoldenRecord {
        GoldenRecord {
            prompt_len: 3,
            kv_valid_len: 3,
            token_ids,
            denoise_steps_run: 4,
            blocks_committed: 1,
            stopped_on_eot: true,
            block_steps_eff: vec![4],
            kv_hash: kv_hash.into(),
        }
    }

    #[test]
    fn fnv_pins_algorithm_constants() {
        // Empty input = the FNV-1a offset basis; a single byte must fold via the
        // prime. Pinning both catches an accidental algorithm/constant change
        // that would silently invalidate every blessed kv_hash.
        assert_eq!(fnv1a64_bytes(b""), 0xcbf2_9ce4_8422_2325);
        // One zero byte = (basis ^ 0) * prime.
        assert_eq!(
            fnv1a64_bytes(&[0x00]),
            0xcbf2_9ce4_8422_2325u64.wrapping_mul(0x0000_0100_0000_01b3)
        );
        // Order sensitivity: [1,2] != [2,1].
        assert_ne!(fnv1a64_bytes(&[1, 2]), fnv1a64_bytes(&[2, 1]));
    }

    #[test]
    fn identical_records_compare_ok() {
        let a = rec(vec![1, 2, 3], "kv_dead");
        assert!(a.compare(&a).is_ok());
    }

    #[test]
    fn token_divergence_is_reported_first() {
        let g = rec(vec![1, 2, 3], "kv_a");
        let o = rec(vec![1, 9, 3], "kv_b"); // both token + hash differ
        let err = g.compare(&o).unwrap_err();
        assert!(err.contains("token_ids diverge at index 1"), "{err}");
    }

    #[test]
    fn kv_hash_divergence_reported_when_tokens_match() {
        let g = rec(vec![1, 2, 3], "kv_a");
        let o = rec(vec![1, 2, 3], "kv_b");
        let err = g.compare(&o).unwrap_err();
        assert!(err.contains("KV hash differs"), "{err}");
    }

    #[test]
    fn digest_shape() {
        let d = kv_digest(&[0u8, 1, 2, 3]);
        assert!(d.starts_with("kv_") && d.len() == 3 + 16);
    }
}
