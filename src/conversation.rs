//! Multi-conversation KV manager (v1): keep N conversations' KV resident in an
//! LRU pool and swap them through the single hot GPU buffer, so interleaved
//! clients don't re-prefill on every request.
//!
//! **Identity is implicit.** A request routes to the conversation whose canonical
//! token log is the longest prefix of the incoming prompt. This is
//! OpenAI-compatible: the client resends the whole history each turn, so the
//! shared prefix *is* the conversation. No match (a fresh chat, or an edited
//! history) opens a new conversation.
//!
//! **Thought-safety.** Each turn is finalized to the *canonical completed-turns*
//! token log — the sanitized answer, with no `thought` channel and no
//! generation-prompt scaffold. That log is a genuine prefix of the next turn's
//! prompt, so (a) the snapshot survives a swap and reuse works, and (b) reasoning
//! never persists into the KV. See [`ConversationManager::finalize`].
//!
//! v1 keeps snapshots in RAM (a byte budget, LRU-evicted → re-prefill). v2 swaps
//! the backing store for SSD + a disk cap; the manager API is unchanged.

use crate::metal::{KvSnapshot, StepGenerateSession};
use crate::safetensors::Error;
use std::collections::HashMap;

/// Longest common prefix length (in tokens).
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

struct Conversation {
    /// Canonical completed-turns token log the snapshot represents — a prefix of
    /// the next turn's prompt. Empty until the first turn is finalized.
    tokens: Vec<u32>,
    /// Resident KV, or `None` when evicted (LRU) → re-prefill from `tokens`.
    snapshot: Option<KvSnapshot>,
    /// Monotonic access stamp for LRU.
    last_used: u64,
}

/// Pick the conversation to continue for `prompt`: the one whose (non-empty)
/// token log is a prefix of `prompt`, longest first. `None` → open a new one.
fn route(convs: &HashMap<u64, Conversation>, prompt: &[u32]) -> Option<u64> {
    convs
        .iter()
        .filter(|(_, c)| !c.tokens.is_empty() && prompt.starts_with(&c.tokens))
        .max_by_key(|(_, c)| c.tokens.len())
        .map(|(id, _)| *id)
}

/// Least-recently-used conversation that still holds an evictable snapshot,
/// skipping `keep` and `current`. `None` → nothing left to free.
fn lru_victim(convs: &HashMap<u64, Conversation>, keep: u64, current: Option<u64>) -> Option<u64> {
    convs
        .iter()
        .filter(|(id, c)| **id != keep && Some(**id) != current && c.snapshot.is_some())
        .min_by_key(|(_, c)| c.last_used)
        .map(|(id, _)| *id)
}

pub struct ConversationManager {
    session: StepGenerateSession,
    convs: HashMap<u64, Conversation>,
    /// Conversation whose KV is currently in the hot buffer.
    current: Option<u64>,
    budget_bytes: usize,
    used_bytes: usize,
    tick: u64,
    next_id: u64,
}

impl ConversationManager {
    pub fn new(session: StepGenerateSession, budget_bytes: usize) -> Self {
        Self {
            session,
            convs: HashMap::new(),
            current: None,
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            next_id: 1,
        }
    }

    /// The hot session, for the caller to drive generation
    /// (`generate_with_session`) after [`activate`](Self::activate).
    pub fn session_mut(&mut self) -> &mut StepGenerateSession {
        &mut self.session
    }

    /// Number of conversations currently tracked (for logging/metrics).
    pub fn len(&self) -> usize {
        self.convs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.convs.is_empty()
    }

    /// Route `prompt` to a conversation, loading its KV into the hot buffer, and
    /// return its id. After this the session's `kv_valid_tokens` is set so
    /// `generate_with_session(prompt)` reuses the conversation's prefix and
    /// prefills only the new-turn delta (or re-prefills whole if evicted/new).
    pub fn activate(&mut self, prompt: &[u32]) -> u64 {
        self.tick += 1;
        let id = route(&self.convs, prompt).unwrap_or_else(|| self.new_conversation());
        self.switch_to(id);
        id
    }

    fn new_conversation(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.convs.insert(
            id,
            Conversation {
                tokens: Vec::new(),
                snapshot: None,
                last_used: self.tick,
            },
        );
        id
    }

    fn switch_to(&mut self, id: u64) {
        if self.current == Some(id) {
            self.touch(id);
            return;
        }
        // Stash the outgoing conversation's finalized KV before overwriting it.
        if let Some(cur) = self.current {
            self.stash(cur);
        }
        // Bring the target in: restore its snapshot, or reset (new/evicted →
        // the upcoming generate re-prefills the whole prompt).
        match self.convs.get(&id).and_then(|c| c.snapshot.as_ref()) {
            Some(snap) => self.session.restore_kv(snap),
            None => self.session.reset_kv(),
        }
        self.current = Some(id);
        self.touch(id);
    }

    fn touch(&mut self, id: u64) {
        if let Some(c) = self.convs.get_mut(&id) {
            c.last_used = self.tick;
        }
    }

    /// Snapshot the session's current (finalized) KV into `id`'s pool slot,
    /// replacing any prior snapshot, then evict LRU snapshots to stay in budget.
    fn stash(&mut self, id: u64) {
        let snap = self.session.snapshot_kv();
        let bytes = snap.byte_len();
        let prev = self
            .convs
            .get(&id)
            .and_then(|c| c.snapshot.as_ref())
            .map_or(0, |s| s.byte_len());
        self.used_bytes = self.used_bytes - prev + bytes;
        if let Some(c) = self.convs.get_mut(&id) {
            c.snapshot = Some(snap);
        }
        self.evict_to_budget(id);
    }

    /// Drop LRU snapshots (their token logs stay → re-prefill on next access)
    /// until back under budget. Never evicts `keep` or the current conversation.
    fn evict_to_budget(&mut self, keep: u64) {
        while self.used_bytes > self.budget_bytes {
            let Some(victim) = lru_victim(&self.convs, keep, self.current) else {
                break; // nothing evictable (only keep/current hold snapshots)
            };
            if let Some(c) = self.convs.get_mut(&victim)
                && let Some(s) = c.snapshot.take()
            {
                self.used_bytes -= s.byte_len();
            }
        }
    }

    /// Finalize turn `id`: rebuild the resident KV to the canonical completed-turns
    /// `canonical_tokens` (sanitized answer, no `thought` channel, no
    /// generation-prompt scaffold) by rolling back to the shared prefix and
    /// re-deriving the tail. That log is a clean prefix of the next turn's prompt,
    /// so the snapshot stays reusable across the swap and reasoning never persists.
    pub fn finalize(&mut self, id: u64, canonical_tokens: &[u32]) -> Result<(), Error> {
        let reuse = common_prefix_len(self.session.kv_valid_tokens(), canonical_tokens);
        self.session.truncate_kv_to(reuse);
        self.session.extend_kv(&canonical_tokens[reuse..])?;
        if let Some(c) = self.convs.get_mut(&id) {
            c.tokens = canonical_tokens.to_vec();
            c.last_used = self.tick;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(tokens: &[u32], bytes: usize, last_used: u64) -> Conversation {
        Conversation {
            tokens: tokens.to_vec(),
            // A zero-length snapshot with a recorded byte cost is enough to
            // exercise routing/LRU without a GPU session.
            snapshot: (bytes > 0).then(|| KvSnapshot::for_test(bytes)),
            last_used,
        }
    }

    #[test]
    fn route_prefers_longest_prefix_match() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[10, 20], 1, 0)); // prefix of the prompt
        convs.insert(2, conv(&[10, 20, 30], 1, 0)); // longer prefix — wins
        convs.insert(3, conv(&[10, 99], 1, 0)); // diverges
        assert_eq!(route(&convs, &[10, 20, 30, 40]), Some(2));
    }

    #[test]
    fn route_none_when_no_conversation_is_a_prefix() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[10, 20], 1, 0));
        // Prompt shares only [10]; conv 1's log [10,20] is not a prefix.
        assert_eq!(route(&convs, &[10, 99]), None);
        // Empty logs never match (a brand-new, not-yet-finalized conversation).
        convs.insert(2, conv(&[], 0, 0));
        assert_eq!(route(&convs, &[]), None);
    }

    #[test]
    fn lru_victim_picks_oldest_and_skips_keep_and_current() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[1], 1, 5)); // oldest
        convs.insert(2, conv(&[2], 1, 9));
        convs.insert(3, conv(&[3], 1, 7));
        assert_eq!(lru_victim(&convs, 99, None), Some(1));
        // Skips `keep` (1) → next oldest is 3.
        assert_eq!(lru_victim(&convs, 1, None), Some(3));
        // Skips current (3) and keep (1) → only 2 left.
        assert_eq!(lru_victim(&convs, 1, Some(3)), Some(2));
    }

    #[test]
    fn lru_victim_none_when_nothing_evictable() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[1], 0, 5)); // no snapshot
        assert_eq!(lru_victim(&convs, 99, None), None);
    }

    #[test]
    fn common_prefix_len_basic() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
    }
}
