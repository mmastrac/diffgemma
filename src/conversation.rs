//! Multi-conversation KV manager: keep recent conversations' KV out of the
//! prefill path by holding it in a tiered pool and swapping it through the single
//! hot GPU buffer, so interleaved clients don't re-prefill every request.
//!
//! **Identity is implicit.** A request routes to the conversation whose canonical
//! token log is the longest prefix of the incoming prompt. This is
//! OpenAI-compatible: the client resends the whole history each turn, so the
//! shared prefix *is* the conversation. No match opens a new conversation.
//!
//! **Thought-safety.** Each turn is finalized to the *canonical completed-turns*
//! token log — the sanitized answer, no `thought` channel, no generation-prompt
//! scaffold. That log is a genuine prefix of the next turn's prompt, so the
//! snapshot survives a swap and reasoning never persists. See [`finalize`].
//! Truncating past a sliding-ring wrap re-prefills the kept prefix (clamping
//! `kv_len` alone would leave early ring slots holding post-wrap K/V).
//!
//! **Tiers** (v2). A conversation's KV snapshot lives in exactly one place:
//! - **RAM** — recent snapshots, bounded by a byte budget.
//! - **SSD** — RAM overflow spills here (a compact blob) rather than being
//!   dropped; restoring streams it back, far cheaper than re-prefilling a long
//!   conversation.
//! - **nowhere** — over the disk budget too, the snapshot is dropped and the
//!   conversation re-prefills from its (always-retained) token log.
//!
//! [`finalize`]: ConversationManager::finalize

use crate::Error;
use crate::metal::{KvSnapshot, StepGenerateSession};
use std::collections::HashMap;
use std::path::PathBuf;

/// Longest common prefix length (in tokens).
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Best-effort removal of this manager's `*.kv` blobs from `dir` (nothing else),
/// then the dir if it ends up empty.
fn remove_stale_blobs(dir: &std::path::Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "kv") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let _ = std::fs::remove_dir(dir); // only succeeds if now empty
}

/// Where a conversation's KV snapshot currently lives.
enum Residence {
    /// No snapshot — restoring this conversation re-prefills from `tokens`.
    None,
    /// Resident in the RAM pool.
    Ram(KvSnapshot),
    /// Spilled to an SSD blob at `path` (`bytes` long).
    Disk { path: PathBuf, bytes: usize },
}

impl Residence {
    fn ram_bytes(&self) -> usize {
        match self {
            Residence::Ram(s) => s.byte_len(),
            _ => 0,
        }
    }
    fn disk_bytes(&self) -> usize {
        match self {
            Residence::Disk { bytes, .. } => *bytes,
            _ => 0,
        }
    }
}

struct Conversation {
    /// Canonical completed-turns token log the snapshot represents — a prefix of
    /// the next turn's prompt. Empty until the first turn is finalized.
    tokens: Vec<u32>,
    residence: Residence,
    /// Monotonic access stamp for LRU.
    last_used: u64,
}

/// Tail-salvage window (KV-reuse-first): a conversation whose log diverges
/// from the prompt only in its last `TAIL_SLACK` tokens is still routed to —
/// the caller truncates the resident KV to the common prefix, which is O(1)
/// as long as the cut stays inside the sliding-ring slack (~769 tokens at
/// ring 2048 / window 1024 / CANVAS 256; 512 keeps margin). Deeper
/// divergence would force a ring rebuild ≈ a fresh prefill, so a new
/// conversation is genuinely equivalent — that is the smart decline.
const TAIL_SLACK: usize = 512;

/// Minimum shared prefix before tail-salvage may claim a conversation:
/// prevents two unrelated short prompts that share a few opening tokens from
/// merging. Real clients share multi-hundred-token system/tool preambles.
const MIN_SALVAGE_PREFIX: usize = 256;

/// Pick the conversation to continue for `prompt`, longest common prefix
/// first: an exact-prefix match (the log extends into the prompt) always
/// qualifies; a tail-divergent log qualifies when the shared prefix is
/// substantial (>= `MIN_SALVAGE_PREFIX`) and the divergence is confined to
/// the last `TAIL_SLACK` tokens (clients re-render turn tails; the grammar
/// moves a tool turn's prose across the response splice). `None` → new one.
fn route(convs: &HashMap<u64, Conversation>, prompt: &[u32]) -> Option<u64> {
    convs
        .iter()
        .filter_map(|(id, c)| {
            if c.tokens.is_empty() {
                return None;
            }
            let lcp = common_prefix_len(&c.tokens, prompt);
            let exact = lcp == c.tokens.len();
            // Rewind: the prompt is a prefix of the canonical log (repeat,
            // retry, regenerate, edited last turn) — the log's first `lcp`
            // positions ARE the prompt, so the conversation's KV prefix is
            // fully reusable after a truncate (begin_turn's rewind guard
            // handles O(1)-vs-reset).
            let rewind = lcp == prompt.len() && lcp >= MIN_SALVAGE_PREFIX;
            let salvage = lcp >= MIN_SALVAGE_PREFIX && c.tokens.len() - lcp <= TAIL_SLACK;
            (exact || rewind || salvage).then_some((*id, lcp))
        })
        .max_by_key(|&(_, lcp)| lcp)
        .map(|(id, _)| id)
}

/// Least-recently-used conversation (excluding `current`) for which `pick`
/// returns true — used to choose a RAM snapshot to spill or a disk blob to drop.
fn lru_victim(
    convs: &HashMap<u64, Conversation>,
    current: Option<u64>,
    pick: impl Fn(&Conversation) -> bool,
) -> Option<u64> {
    convs
        .iter()
        .filter(|(id, c)| Some(**id) != current && pick(c))
        .min_by_key(|(_, c)| c.last_used)
        .map(|(id, _)| *id)
}

pub struct ConversationManager {
    session: StepGenerateSession,
    convs: HashMap<u64, Conversation>,
    /// Conversation whose KV is currently in the hot buffer.
    current: Option<u64>,
    ram_budget: usize,
    ram_used: usize,
    disk_budget: usize,
    disk_used: usize,
    disk_dir: PathBuf,
    tick: u64,
    next_id: u64,
    /// Monotonic counter for unique SSD blob filenames.
    write_seq: u64,
}

impl ConversationManager {
    pub fn new(
        session: StepGenerateSession,
        ram_budget: usize,
        disk_budget: usize,
        disk_dir: PathBuf,
    ) -> Self {
        // Clear any stale `*.kv` blobs left in this dir by a prior run that was
        // hard-killed (SIGTERM/SIGKILL skip `Drop`). Only our own files are
        // touched, so a user-pointed dir with other content is safe.
        remove_stale_blobs(&disk_dir);
        Self {
            session,
            convs: HashMap::new(),
            current: None,
            ram_budget,
            ram_used: 0,
            disk_budget,
            disk_used: 0,
            disk_dir,
            tick: 0,
            next_id: 1,
            write_seq: 0,
        }
    }

    /// The hot session, for the caller to drive generation
    /// (`generate_with_session`) after [`activate`](Self::activate).
    pub fn session_mut(&mut self) -> &mut StepGenerateSession {
        &mut self.session
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
                residence: Residence::None,
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
        // Stash the outgoing conversation's finalized KV (to RAM) before
        // overwriting it, then bring the target in and make it current.
        if let Some(cur) = self.current {
            self.stash(cur);
        }
        self.materialize(id);
        self.current = Some(id);
        self.touch(id);
        // Rebalance only now that `current` is the target: the just-stashed
        // outgoing conversation is no longer protected, so it can spill to disk
        // (otherwise it would linger in RAM over budget until it cycled out).
        self.rebalance();
    }

    fn touch(&mut self, id: u64) {
        if let Some(c) = self.convs.get_mut(&id) {
            c.last_used = self.tick;
        }
    }

    /// Load conversation `id`'s KV into the hot buffer: from RAM, else stream it
    /// back from SSD, else reset (the upcoming generate re-prefills the prompt).
    fn materialize(&mut self, id: u64) {
        match self.convs.get(&id).map(|c| &c.residence) {
            Some(Residence::Ram(snap)) => self.session.restore_kv(snap),
            Some(Residence::Disk { path, .. }) => {
                let tokens = self.convs[&id].tokens.clone();
                match std::fs::read(path) {
                    Ok(bytes) => {
                        let snap = KvSnapshot::from_parts(bytes, tokens);
                        self.session.restore_kv(&snap);
                    }
                    Err(err) => {
                        eprintln!("serve: conversation disk read failed ({err}); re-prefilling");
                        self.session.reset_kv();
                    }
                }
            }
            _ => self.session.reset_kv(),
        }
    }

    /// Snapshot the session's current (finalized) KV into `id`'s RAM slot,
    /// replacing any prior residence. The caller runs `rebalance` afterwards
    /// (once `current` has moved on, so `id` can itself be spilled if needed).
    fn stash(&mut self, id: u64) {
        self.free_residence(id);
        // With no snapshot pool at all, rebalance would drop the snapshot
        // immediately — skip the full-KV gather (a conversation switch then
        // costs only the target's re-prefill, not a wasted copy-out too).
        if self.ram_budget == 0 && self.disk_budget == 0 {
            self.touch(id);
            return;
        }
        let snap = self.session.snapshot_kv();
        self.ram_used += snap.byte_len();
        if let Some(c) = self.convs.get_mut(&id) {
            c.residence = Residence::Ram(snap);
        }
        self.touch(id);
    }

    /// Release whatever tier `id` occupies (drop RAM / delete the SSD blob),
    /// updating the used counters.
    fn free_residence(&mut self, id: u64) {
        if let Some(c) = self.convs.get_mut(&id) {
            match std::mem::replace(&mut c.residence, Residence::None) {
                Residence::Ram(s) => self.ram_used -= s.byte_len(),
                Residence::Disk { path, bytes } => {
                    let _ = std::fs::remove_file(&path);
                    self.disk_used -= bytes;
                }
                Residence::None => {}
            }
        }
    }

    /// Enforce both budgets: spill LRU RAM snapshots to SSD, then drop LRU SSD
    /// blobs. The just-stashed conversation is the most-recently-used, so LRU
    /// protects it until everything older is gone.
    fn rebalance(&mut self) {
        // RAM → SSD (or drop if there is no disk tier / the write fails).
        while self.ram_used > self.ram_budget {
            let Some(victim) =
                lru_victim(&self.convs, self.current, |c| c.residence.ram_bytes() > 0)
            else {
                break;
            };
            self.spill_to_disk(victim);
        }
        // SSD → dropped (re-prefill from the token log next time).
        while self.disk_used > self.disk_budget {
            let Some(victim) =
                lru_victim(&self.convs, self.current, |c| c.residence.disk_bytes() > 0)
            else {
                break;
            };
            self.free_residence(victim);
        }
    }

    /// Move `id`'s RAM snapshot to an SSD blob. On any failure (no disk budget,
    /// write error) the snapshot is dropped instead — the conversation will
    /// re-prefill, never serve stale KV.
    fn spill_to_disk(&mut self, id: u64) {
        let bytes = match self.convs.get(&id).map(|c| &c.residence) {
            Some(Residence::Ram(s)) => s.byte_len(),
            _ => return, // not RAM-resident — nothing to spill
        };
        if self.disk_budget == 0 || bytes > self.disk_budget {
            self.free_residence(id); // no room on disk either → drop
            return;
        }
        self.write_seq += 1;
        let path = self.disk_dir.join(format!("{id}-{}.kv", self.write_seq));
        // Write the snapshot bytes straight from the RAM residence (no clone).
        let write_result = match self.convs.get(&id).map(|c| &c.residence) {
            Some(Residence::Ram(snap)) => std::fs::create_dir_all(&self.disk_dir)
                .and_then(|_| std::fs::write(&path, snap.kv_bytes())),
            _ => return,
        };
        match write_result {
            Ok(()) => {
                self.ram_used -= bytes;
                self.disk_used += bytes;
                if let Some(c) = self.convs.get_mut(&id) {
                    c.residence = Residence::Disk { path, bytes };
                }
            }
            Err(err) => {
                eprintln!("serve: conversation disk spill failed ({err}); dropping snapshot");
                let _ = std::fs::remove_file(&path);
                self.free_residence(id);
            }
        }
    }

    /// Finalize turn `id`: rebuild the resident KV to the canonical completed-turns
    /// `canonical_tokens` (sanitized answer, no `thought` channel, no
    /// generation-prompt scaffold) by rolling back to the shared prefix and
    /// re-deriving the tail. That log is a clean prefix of the next turn's prompt,
    /// so the snapshot stays reusable across the swap and reasoning never persists.
    ///
    /// After the hot sequence wraps a sliding ring, truncate rebuilds the kept
    /// prefix by re-prefill (ring slots for early absolute positions were
    /// overwritten). Below the wrap this stays an O(1) `kv_len` clamp.
    ///
    /// A near-context-full turn can produce a canonical log longer than the KV
    /// can hold with canvas headroom (the last block may overshoot the token
    /// budget by up to a block): the KV keeps the longest fitting *prefix* —
    /// still a valid reuse prefix of the next prompt — while the routing log
    /// keeps the full canonical.
    pub fn finalize(&mut self, id: u64, canonical_tokens: &[u32]) -> Result<(), Error> {
        let fit = canonical_tokens.len().min(self.session.extend_capacity());
        let reuse = common_prefix_len(self.session.kv_valid_tokens(), canonical_tokens).min(fit);
        if self.session.truncate_needs_rebuild(reuse) {
            // Deep thought-strip rewind (a long-thinking reply overflowed the
            // ring's O(1) slack): a rebuild here costs ≈ a fresh prefill of
            // the kept prefix (measured 39s at 14.7k) and yields lineage KV
            // (see memory kv-reuse-delta-divergence). Drop the hot KV instead
            // — the canonical token log still routes, and the next activation
            // re-prefills lazily, which costs the same only when the
            // conversation actually continues.
            self.session.reset_kv();
        } else {
            self.session.truncate_kv_to(reuse)?;
            self.session.extend_kv(&canonical_tokens[reuse..fit])?;
        }
        if let Some(c) = self.convs.get_mut(&id) {
            c.tokens = canonical_tokens.to_vec();
            c.last_used = self.tick;
        }
        Ok(())
    }
}

impl Drop for ConversationManager {
    fn drop(&mut self) {
        // Graceful-exit cleanup (signal kills skip this — startup covers those).
        for c in self.convs.values() {
            if let Residence::Disk { path, .. } = &c.residence {
                let _ = std::fs::remove_file(path);
            }
        }
        remove_stale_blobs(&self.disk_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(tokens: &[u32], ram: usize, last_used: u64) -> Conversation {
        Conversation {
            tokens: tokens.to_vec(),
            residence: if ram > 0 {
                Residence::Ram(KvSnapshot::for_test(ram))
            } else {
                Residence::None
            },
            last_used,
        }
    }

    #[test]
    fn route_prefers_longest_prefix_match() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[10, 20], 1, 0));
        convs.insert(2, conv(&[10, 20, 30], 1, 0)); // longer prefix — wins
        convs.insert(3, conv(&[10, 99], 1, 0)); // diverges
        assert_eq!(route(&convs, &[10, 20, 30, 40]), Some(2));
    }

    /// KV-reuse-first tail salvage: a log diverging from the prompt only in
    /// its tail (client re-rendered the last turn / the grammar moved a tool
    /// turn's prose) still routes, provided the shared prefix is substantial.
    #[test]
    fn route_salvages_tail_divergence() {
        let base: Vec<u32> = (0..1000u32).collect();
        let mut convs = HashMap::new();
        convs.insert(1, conv(&base, 1, 0));

        // Prompt agrees except the log's last 74 tokens (the OpenCode case).
        let mut prompt = base[..926].to_vec();
        prompt.extend(5000..5400u32);
        assert_eq!(route(&convs, &prompt), Some(1));

        // Divergence deeper than TAIL_SLACK: salvage would need a ring
        // rebuild, so decline (fresh conversation is equivalent cost).
        let mut deep = base[..300].to_vec();
        deep.extend(5000..5400u32);
        assert_eq!(route(&convs, &deep), None);

        // Substantial-prefix floor: short logs never salvage on a few shared
        // opening tokens (two unrelated conversations must not merge).
        let mut short_convs = HashMap::new();
        short_convs.insert(1, conv(&[10, 20, 30, 40], 1, 0));
        assert_eq!(route(&short_convs, &[10, 20, 99, 98, 97]), None);

        // Exact-prefix match still wins over a salvage candidate.
        let mut both = HashMap::new();
        both.insert(1, conv(&base, 1, 0)); // needs salvage vs `prompt`
        both.insert(2, conv(&prompt[..990], 1, 0)); // exact prefix, longer lcp
        assert_eq!(route(&both, &prompt), Some(2));
    }

    #[test]
    fn route_accepts_rewind_prompt_prefix_of_canonical() {
        // Repeat/retry: the prompt is a prefix of the canonical log.
        let mut convs = HashMap::new();
        let canonical: Vec<u32> = (0..1000).collect();
        convs.insert(7, conv(&canonical, 0, 1));
        let prompt: Vec<u32> = (0..600).collect();
        assert_eq!(route(&convs, &prompt), Some(7));
        // Below the salvage floor a rewind does NOT route (fresh conv).
        let tiny: Vec<u32> = (0..100).collect();
        assert_eq!(route(&convs, &tiny), None);
    }

    #[test]
    fn route_none_when_no_conversation_is_a_prefix() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[10, 20], 1, 0));
        assert_eq!(route(&convs, &[10, 99]), None);
        convs.insert(2, conv(&[], 0, 0)); // empty log never matches
        assert_eq!(route(&convs, &[]), None);
    }

    #[test]
    fn lru_victim_picks_oldest_and_skips_current() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[1], 1, 5)); // oldest
        convs.insert(2, conv(&[2], 1, 9));
        convs.insert(3, conv(&[3], 1, 7));
        let has_ram = |c: &Conversation| c.residence.ram_bytes() > 0;
        assert_eq!(lru_victim(&convs, None, has_ram), Some(1));
        assert_eq!(lru_victim(&convs, Some(1), has_ram), Some(3)); // skip current
    }

    #[test]
    fn lru_victim_none_when_nothing_matches() {
        let mut convs = HashMap::new();
        convs.insert(1, conv(&[1], 0, 5)); // no RAM snapshot
        assert_eq!(
            lru_victim(&convs, None, |c| c.residence.ram_bytes() > 0),
            None
        );
    }

    #[test]
    fn common_prefix_len_basic() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
    }
}
