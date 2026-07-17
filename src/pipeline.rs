//! Token pipeline (P0) — the serialized op-stream core (design: PLAN.md
//! "Token pipeline").
//!
//! One dedicated thread owns the model session (GPU + KV); clients speak
//! token IDS only — never strings — through an input queue of ops and an
//! output queue of events. P0 scope: `Extend` / `Generate` (whole-reply via
//! `generate_with_session`; the per-block propose/commit protocol is P2) /
//! `Rewind` / `KvFingerprint`, plus the standing rewind byte-consistency
//! gate. Nothing is rerouted through this yet.
//!
//! KV identity: [`KvId`] = (epoch, position). Lineage-invalidating ops
//! (today: `Rewind`) bump the epoch; a rewind to a stale-epoch id fails
//! loudly instead of silently landing on different-lineage KV — the
//! OpenCode-collapse drift class, type-checked.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

/// A KV position bound to its lineage epoch. Issued by pipeline events;
/// consumed by `Rewind`. Stale-epoch ids are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvId {
    pub epoch: u64,
    pub pos: usize,
}

/// Input ops, applied strictly in order by the pipeline thread.
pub enum PipelineOp {
    /// Append tokens to the causal KV (prefill/extend path).
    Extend(Vec<u32>),
    /// Generate a reply continuing the resident context. The prompt is the
    /// session's own causal token log (zero-delta reuse), so callers stage
    /// context exclusively through `Extend`/`Rewind`.
    Generate {
        seed: u64,
        max_new_tokens: usize,
    },
    /// Truncate the causal KV to `id.pos` (token-granular; O(1) below a
    /// sliding-ring wrap, ring-rebuild re-prefill past one). Bumps the epoch.
    Rewind(KvId),
    /// FNV-1a of the valid KV snapshot + causal token log — the rewind
    /// byte-consistency probe.
    KvFingerprint,
    Shutdown,
}

/// One event per op, in op order.
#[derive(Debug)]
pub enum PipelineEvent {
    Extended {
        kv: KvId,
    },
    /// `ids` = the full committed reply; `kv` = the causally-resident KV
    /// position (the final block is committed to the reply but only causally
    /// extended when a later op needs it — same contract as the session).
    Generated {
        ids: Vec<u32>,
        kv: KvId,
    },
    Rewound {
        kv: KvId,
    },
    Fingerprint {
        fnv: u64,
        kv: KvId,
    },
    Error(String),
    ShutDown,
}

pub struct Pipeline {
    tx: Sender<PipelineOp>,
    rx: Receiver<PipelineEvent>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Pipeline {
    /// Spawn the pipeline thread and open the model session on it (Metal
    /// objects are created and stay on that thread; nothing GPU-owned
    /// crosses the channel).
    pub fn spawn(model_dir: PathBuf, max_seq: usize, steps: usize) -> Self {
        let (tx, op_rx) = channel();
        let (ev_tx, rx) = channel();
        let join = std::thread::Builder::new()
            .name("token-pipeline".into())
            .spawn(move || run_pipeline(&model_dir, max_seq, steps, &op_rx, &ev_tx))
            .expect("spawn token-pipeline thread");
        Self {
            tx,
            rx,
            join: Some(join),
        }
    }

    /// Send one op and wait for its event (ops are strictly ordered, so this
    /// is a synchronous facade over the queue pair).
    pub fn call(&self, op: PipelineOp) -> PipelineEvent {
        if self.tx.send(op).is_err() {
            return PipelineEvent::Error("pipeline thread gone".into());
        }
        self.rx
            .recv()
            .unwrap_or_else(|_| PipelineEvent::Error("pipeline thread gone".into()))
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.tx.send(PipelineOp::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn run_pipeline(
    model_dir: &std::path::Path,
    max_seq: usize,
    steps: usize,
    op_rx: &Receiver<PipelineOp>,
    ev_tx: &Sender<PipelineEvent>,
) {
    use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};

    let layers = match crate::commands::resolve_model_layers(model_dir, None) {
        Ok(n) => n,
        Err(err) => {
            let _ = ev_tx.send(PipelineEvent::Error(format!("model layers: {err}")));
            return;
        }
    };
    let open_cfg = StepGenerateConfig::from_generate(
        7,
        256,
        max_seq,
        layers,
        crate::sample::sampler_for_steps(steps, false),
        false,
    );
    let mut session = match StepGenerateSession::open(model_dir, &open_cfg, None) {
        Ok((s, _)) => s,
        Err(err) => {
            let _ = ev_tx.send(PipelineEvent::Error(format!("session open: {err}")));
            return;
        }
    };

    let mut epoch: u64 = 0;
    let kv_id = |epoch: u64, session: &StepGenerateSession| KvId {
        epoch,
        pos: session.kv_valid_tokens().len(),
    };

    while let Ok(op) = op_rx.recv() {
        let event = match op {
            PipelineOp::Extend(ids) => match session.extend_kv(&ids) {
                Ok(()) => PipelineEvent::Extended {
                    kv: kv_id(epoch, &session),
                },
                Err(err) => PipelineEvent::Error(format!("extend: {err}")),
            },
            PipelineOp::Generate {
                seed,
                max_new_tokens,
            } => {
                let cfg = StepGenerateConfig::from_generate(
                    seed,
                    max_new_tokens,
                    max_seq,
                    layers,
                    crate::sample::sampler_for_steps(steps, false),
                    false,
                );
                let prompt = session.kv_valid_tokens().to_vec();
                match generate_with_session(&mut session, &prompt, &cfg, "pipeline") {
                    Ok(out) => PipelineEvent::Generated {
                        ids: out.token_ids[prompt.len()..].to_vec(),
                        kv: kv_id(epoch, &session),
                    },
                    Err(err) => PipelineEvent::Error(format!("generate: {err}")),
                }
            }
            PipelineOp::Rewind(id) => {
                if id.epoch != epoch {
                    PipelineEvent::Error(format!(
                        "rewind: stale epoch {} (current {epoch})",
                        id.epoch
                    ))
                } else {
                    match session.truncate_kv_to(id.pos) {
                        Ok(()) => {
                            epoch += 1;
                            PipelineEvent::Rewound {
                                kv: kv_id(epoch, &session),
                            }
                        }
                        Err(err) => PipelineEvent::Error(format!("rewind: {err}")),
                    }
                }
            }
            PipelineOp::KvFingerprint => PipelineEvent::Fingerprint {
                fnv: session.snapshot_kv().fnv64(),
                kv: kv_id(epoch, &session),
            },
            PipelineOp::Shutdown => {
                let _ = ev_tx.send(PipelineEvent::ShutDown);
                return;
            }
        };
        if ev_tx.send(event).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standing rewind byte-consistency gate (PLAN "Token pipeline").
    /// Seeded generate -> rewind loops must (a) restore the KV snapshot hash
    /// exactly after every rewind and (b) regenerate bit-identical replies at
    /// the same seed. Any lineage residue a rewind leaves behind fails one of
    /// the two. Runs below the sliding-ring wrap (prompt 400 + canvas 256
    /// excursions stay inside the window), so every rewind takes the O(1)
    /// truncate path; a wrap-crossing variant is a follow-up.
    #[test]
    fn pipeline_rewind_kv_byte_consistency() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 7919) % 30000).collect();

        let PipelineEvent::Extended { kv: mut mark } = p.call(PipelineOp::Extend(ids)) else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };

        let mut first_reply: Option<Vec<u32>> = None;
        for round in 0..3 {
            let ev = p.call(PipelineOp::Generate {
                seed: 42,
                max_new_tokens: 192,
            });
            let PipelineEvent::Generated { ids, .. } = ev else {
                panic!("generate failed at round {round}: {ev:?}");
            };
            match &first_reply {
                None => first_reply = Some(ids),
                Some(f) => assert_eq!(
                    f, &ids,
                    "regeneration diverged after rewind (round {round})"
                ),
            }
            let ev = p.call(PipelineOp::Rewind(mark));
            let PipelineEvent::Rewound { kv } = ev else {
                panic!("rewind failed at round {round}: {ev:?}");
            };
            mark = kv;
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed at round {round}");
            };
            assert_eq!(fnv, fp0, "KV bytes diverged after rewind (round {round})");
        }
    }

    /// Stale-epoch rewinds must fail loudly (the drift class, type-checked):
    /// after a rewind bumps the epoch, an id captured before it is dead.
    #[test]
    fn pipeline_rejects_stale_epoch_rewind() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 2048, 24);
        let ids: Vec<u32> = (0..64u32).map(|i| 2000 + i * 3).collect();
        let PipelineEvent::Extended { kv: old } = p.call(PipelineOp::Extend(ids)) else {
            panic!("extend failed");
        };
        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(old)) else {
            panic!("first rewind failed");
        };
        // `old` belongs to the pre-rewind epoch now.
        match p.call(PipelineOp::Rewind(old)) {
            PipelineEvent::Error(msg) => assert!(msg.contains("stale epoch"), "wrong error: {msg}"),
            other => panic!("stale-epoch rewind must fail, got {other:?}"),
        }
    }
}
