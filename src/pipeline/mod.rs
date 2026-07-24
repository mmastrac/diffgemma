//! Token pipeline — the serialized op-stream core.
//!
//! One dedicated thread owns the model session AND the multi-conversation
//! registry (GPU + KV); clients speak token IDS only — never strings —
//! through an input queue of ops and an output queue of events. Current
//! clients: `ask` (Generate), `chat` (Generate per turn), `serve`
//! (Activate / Generate / Finalize / AlignTo / Mark+Rewind for
//! tool-compact). The per-block protocol
//! (BeginTurn/ProposeBlock/CommitBlock/DiscardBlock/EndTurn) exposes each
//! uncommitted canvas for inspection: partial commit covers
//! good-prefix/bad-tail, discard re-rolls from fresh noise, and `Generate`
//! remains the whole-turn composite over the identical primitives
//! (equivalence pinned byte-exactly by `per_block_ops_match_monolithic_generate`).
//!
//! KV identity: [`KvId`] = (epoch, position). Lineage-invalidating ops
//! (today: `Rewind`) bump the epoch; a rewind to a stale-epoch id fails
//! loudly instead of silently landing on different-lineage KV — the
//! OpenCode-collapse drift class, type-checked.

mod event;
mod op;
mod runner;
mod stages;

pub use event::PipelineEvent;
pub use op::PipelineOp;
pub use runner::Pipeline;
pub use stages::{OpLogStage, ToolRepairStage, ToolValidatorStage};

#[cfg(test)]
mod tests;

/// A KV position bound to its lineage epoch. Issued by pipeline events;
/// consumed by `Rewind`. Stale-epoch ids are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvId {
    pub epoch: u64,
    pub pos: usize,
}

/// One link in the op chain. The terminal stage is [`Pipeline`] (the model
/// thread); wrappers — tool compaction, span handles, validators, the op-log —
/// implement the same trait around an inner stage and compose freely. Clients
/// (ask/chat/serve) talk to the top of whatever chain they were given.
pub trait PipelineStage {
    fn call(&self, op: PipelineOp) -> PipelineEvent;
}

impl<S: PipelineStage + ?Sized> PipelineStage for Box<S> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        (**self).call(op)
    }
}

/// FNV-1a over token ids (the event digest `replay` diffs against).
pub(crate) fn ids_fnv(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x1_0000_0000_01b3);
        }
    }
    h
}
