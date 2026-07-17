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
    /// Generate a reply from `prompt` (prefill delta + denoise). `cfg.layers`
    /// and `cfg.max_seq` are session-owned and overwritten with the pipeline's
    /// open-time values — the session's buffers are sized once at spawn.
    Generate {
        prompt: Vec<u32>,
        cfg: Box<crate::metal::StepGenerateConfig>,
        label: String,
    },
    /// Truncate the causal KV to `id.pos` (token-granular; O(1) below a
    /// sliding-ring wrap, ring-rebuild re-prefill past one). Bumps the epoch.
    Rewind(KvId),
    /// Replace the session state with a deterministic pseudorandom KV declared
    /// `tokens` long (~1 s for 100k vs a ~7-minute real prefill). The bytes are
    /// meaningless but finite and bit-deterministic — the substrate for the
    /// long-context order-of-operations gates. Bumps the epoch (the previous
    /// lineage is destroyed). Test infrastructure.
    SyntheticFill {
        tokens: usize,
        seed: u64,
    },
    /// FNV-1a of the READABLE state (causal token log + live KV: linear
    /// layers whole prefix, ring layers window-only) — the rewind
    /// byte-consistency probe. Ring residue no future read can observe is
    /// deliberately excluded.
    KvFingerprint,
    Shutdown,
}

/// One event per op, in op order.
pub enum PipelineEvent {
    Extended {
        kv: KvId,
    },
    /// `out` = the full generate output (token_ids includes the prompt); `kv`
    /// = the causally-resident KV position (the final block is committed to
    /// the reply but only causally extended when a later op needs it — same
    /// contract as the session).
    Generated {
        out: Box<crate::generate::GenerateOutput>,
        kv: KvId,
    },
    Rewound {
        kv: KvId,
    },
    Filled {
        kv: KvId,
    },
    Fingerprint {
        fnv: u64,
        kv: KvId,
    },
    Error(String),
    ShutDown,
}

impl std::fmt::Debug for PipelineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extended { kv } => write!(f, "Extended {{ kv: {kv:?} }}"),
            Self::Generated { out, kv } => write!(
                f,
                "Generated {{ tokens: {}, blocks: {}, kv: {kv:?} }}",
                out.token_ids.len(),
                out.blocks_committed
            ),
            Self::Rewound { kv } => write!(f, "Rewound {{ kv: {kv:?} }}"),
            Self::Filled { kv } => write!(f, "Filled {{ kv: {kv:?} }}"),
            Self::Fingerprint { fnv, kv } => {
                write!(f, "Fingerprint {{ fnv: {fnv:#x}, kv: {kv:?} }}")
            }
            Self::Error(msg) => write!(f, "Error({msg:?})"),
            Self::ShutDown => write!(f, "ShutDown"),
        }
    }
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
            PipelineOp::Generate { prompt, cfg, label } => {
                let mut cfg = *cfg;
                cfg.layers = layers;
                cfg.max_seq = max_seq;
                match generate_with_session(&mut session, &prompt, &cfg, &label) {
                    Ok(out) => PipelineEvent::Generated {
                        out: Box::new(out),
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
            PipelineOp::SyntheticFill { tokens, seed } => {
                match session.synthetic_fill_kv(tokens, seed) {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Filled {
                            kv: kv_id(epoch, &session),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("synthetic fill: {err}")),
                }
            }
            PipelineOp::KvFingerprint => PipelineEvent::Fingerprint {
                fnv: session.live_kv_fingerprint(),
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

        let PipelineEvent::Extended { kv: mut mark } = p.call(PipelineOp::Extend(ids.clone()))
        else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };

        let mut first_reply: Option<Vec<u32>> = None;
        for round in 0..3 {
            let cfg = crate::metal::StepGenerateConfig::from_generate(
                42,
                192,
                4096,
                0, // session-owned; overwritten by the pipeline
                crate::sample::sampler_for_steps(24, false),
                false,
            );
            let ev = p.call(PipelineOp::Generate {
                prompt: ids.clone(),
                cfg: Box::new(cfg),
                label: "rewind-gate".into(),
            });
            let PipelineEvent::Generated { out, .. } = ev else {
                panic!("generate failed at round {round}: {ev:?}");
            };
            match &first_reply {
                None => first_reply = Some(out.token_ids.clone()),
                Some(f) => assert_eq!(
                    f, &out.token_ids,
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

    /// Long-context extension byte-consistency on a synthetic 100k KV (PLAN
    /// "Token pipeline"): a pseudorandom KV declared 100k tokens costs ~1 s
    /// instead of a ~7-minute prefill, and the gates only assert
    /// order-of-operations bit-identity, never semantics. For each delta
    /// (1 token, then 256 = one full chunk): extend → fingerprint; rewind →
    /// must restore the base fingerprint (O(1) truncate — deltas stay inside
    /// the sliding-ring slack, ~769 tokens at ring 2048/window 1024, so no
    /// rebuild replaces the synthetic bytes); re-extend the same ids → must
    /// reproduce the post-extend fingerprint (extend-path recompute
    /// determinism at a 100k offset).
    #[test]
    fn synthetic_kv_extension_byte_consistency_100k() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 101_000, 24);
        let PipelineEvent::Filled { kv: mut base } = p.call(PipelineOp::SyntheticFill {
            tokens: 100_000,
            seed: 7,
        }) else {
            panic!("synthetic fill failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp_base, .. } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };

        for delta in [1usize, 256] {
            let ids: Vec<u32> = (0..delta as u32).map(|i| 5000 + i * 11).collect();
            let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids.clone())) else {
                panic!("extend {delta} failed");
            };
            let PipelineEvent::Fingerprint { fnv: fp_add, .. } = p.call(PipelineOp::KvFingerprint)
            else {
                panic!("fingerprint failed");
            };
            assert_ne!(fp_add, fp_base, "extend {delta} did not change the KV");

            let PipelineEvent::Rewound { kv } = p.call(PipelineOp::Rewind(base)) else {
                panic!("rewind after extend {delta} failed");
            };
            base = kv;
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed");
            };
            assert_eq!(fnv, fp_base, "rewind after extend {delta} left residue");

            let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids)) else {
                panic!("re-extend {delta} failed");
            };
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed");
            };
            assert_eq!(
                fnv, fp_add,
                "re-extend {delta} at 100k offset was not bit-deterministic"
            );
            let PipelineEvent::Rewound { kv } = p.call(PipelineOp::Rewind(base)) else {
                panic!("rewind back to base failed");
            };
            base = kv;
        }
    }

    /// Deep-extension determinism at 100k (ignored: ~2× 10k-token extends).
    /// A 10k truncate would cross the ring slack and REBUILD (re-embedding the
    /// synthetic ids — premise destroyed), so determinism is asserted by
    /// refilling the identical synthetic base and re-running the identical
    /// extend: same op sequence, same bytes. Exercises the batched super-chunk
    /// extend path (M=1024) at a 100k offset.
    /// Run: `cargo test --release synthetic_kv_deep_extend -- --ignored`
    #[test]
    #[ignore = "long: two 10k-token extends at 100k offset"]
    fn synthetic_kv_deep_extend_redo_determinism() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 111_000, 24);
        let ids: Vec<u32> = (0..10_000u32).map(|i| 3000 + (i * 97) % 150_000).collect();
        let mut fps = Vec::new();
        for round in 0..2 {
            let PipelineEvent::Filled { .. } = p.call(PipelineOp::SyntheticFill {
                tokens: 100_000,
                seed: 7,
            }) else {
                panic!("synthetic fill failed (round {round})");
            };
            let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids.clone())) else {
                panic!("10k extend failed (round {round})");
            };
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed (round {round})");
            };
            fps.push(fnv);
        }
        assert_eq!(
            fps[0], fps[1],
            "identical synthetic-fill + 10k-extend op sequences diverged"
        );
    }

    /// The ask reroute's equivalence gate: `generate_monolithic_gpu` (direct:
    /// session-open prefill) and `generate_monolithic_gpu_pipeline` (pipeline
    /// thread: generate-time prefill) must produce identical token ids for the
    /// same inputs. Pins the first production client of the pipeline to the
    /// path it replaced.
    #[test]
    fn ask_via_pipeline_matches_direct() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let gen_cfg = crate::generate::GenerateConfig {
            seed: 42,
            max_new_tokens: 64,
            full_message_stop: true,
            ..Default::default()
        };
        let ids: Vec<u32> = (0..48u32).map(|i| 1000 + i * 13).collect();
        let direct =
            crate::generate::generate_monolithic_gpu(&dir, &ids, &gen_cfg, 1024, "eq-gate")
                .expect("direct generate");
        let piped = crate::generate::generate_monolithic_gpu_pipeline(
            &dir, &ids, &gen_cfg, 1024, "eq-gate",
        )
        .expect("pipeline generate");
        assert_eq!(
            direct.token_ids, piped.token_ids,
            "pipeline ask path diverged from the direct path"
        );
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
