//! Token pipeline — the serialized op-stream core (design: PLAN.md
//! "Token pipeline").
//!
//! One dedicated thread owns the model session AND the multi-conversation
//! registry (GPU + KV); clients speak token IDS only — never strings —
//! through an input queue of ops and an output queue of events. Current
//! clients: `ask` (Generate), `chat` (Generate per turn), `serve`
//! (Activate / Generate / Finalize / AlignTo / Mark+Rewind for
//! tool-compact). The per-block propose/commit protocol is P2.
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
    /// Readiness / liveness probe. The first `Pong` proves the model session
    /// opened (serve blocks its "listening" print on it).
    Ping,
    /// Capture the current lineage position (the checkpoint half of the
    /// checkpoint→generate→rollback pattern; the rollback half is `Rewind`).
    Mark,
    /// Route `prompt` to its conversation (longest-prefix match), swapping KV
    /// through the multi-conversation registry as needed. Bumps the epoch
    /// (activation may replace the whole resident state).
    Activate {
        prompt: Vec<u32>,
    },
    /// Finalize a conversation to its canonical (thought-free) token log:
    /// truncate to the common prefix, extend the canonical tail, record the
    /// log in the registry. Bumps the epoch.
    Finalize {
        conv_id: u64,
        canonical: Vec<u32>,
    },
    /// Align the resident KV to `target`: truncate to the longest common
    /// prefix with the causal log, extend the remainder. The tool-compact
    /// expand loop's primitive (and the session-level core of `Finalize`,
    /// without the registry write). Bumps the epoch.
    AlignTo {
        target: Vec<u32>,
    },
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
    Pong,
    Marked {
        kv: KvId,
    },
    /// `reused` = longest common prefix of the resident causal log and the
    /// activating prompt (the cross-turn reuse the serve log reports).
    Activated {
        conv_id: u64,
        kv: KvId,
        reused: usize,
    },
    Finalized {
        kv: KvId,
    },
    Aligned {
        kv: KvId,
        reused: usize,
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
            Self::Pong => write!(f, "Pong"),
            Self::Marked { kv } => write!(f, "Marked {{ kv: {kv:?} }}"),
            Self::Activated {
                conv_id,
                kv,
                reused,
            } => write!(
                f,
                "Activated {{ conv: {conv_id}, kv: {kv:?}, reused: {reused} }}"
            ),
            Self::Finalized { kv } => write!(f, "Finalized {{ kv: {kv:?} }}"),
            Self::Aligned { kv, reused } => {
                write!(f, "Aligned {{ kv: {kv:?}, reused: {reused} }}")
            }
            Self::Error(msg) => write!(f, "Error({msg:?})"),
            Self::ShutDown => write!(f, "ShutDown"),
        }
    }
}

/// One link in the op chain. The terminal stage is [`Pipeline`] (the model
/// thread); wrappers — tool compaction, span handles, validators, the op-log —
/// implement the same trait around an inner stage and compose freely. Clients
/// (ask/chat/serve) talk to the top of whatever chain they were given.
pub trait PipelineStage {
    fn call(&self, op: PipelineOp) -> PipelineEvent;
}

/// The first wrapper stage: append every op + event to a JSONL op-log (the
/// design's durable replay artifact). Deliberately minimal — it demonstrates
/// the chain shape and gives field sessions a replayable trace.
pub struct OpLogStage<S> {
    inner: S,
    log: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
}

impl<S: PipelineStage> OpLogStage<S> {
    pub fn new(inner: S, path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self {
            inner,
            log: std::sync::Mutex::new(std::io::BufWriter::new(
                std::fs::File::options()
                    .create(true)
                    .append(true)
                    .open(path)?,
            )),
        })
    }
}

impl<S: PipelineStage> PipelineStage for OpLogStage<S> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        use std::io::Write;
        let op_line = op.log_line();
        let event = self.inner.call(op);
        if let Ok(mut w) = self.log.lock() {
            let _ = writeln!(
                w,
                "{{\"op\":{op_line},\"event\":{:?}}}",
                format!("{event:?}")
            );
            let _ = w.flush();
        }
        event
    }
}

impl PipelineOp {
    /// Compact JSON-ish description for the op-log (ids elided past a prefix —
    /// the log records the op SHAPE; full replay logging is P4).
    fn log_line(&self) -> String {
        match self {
            Self::Extend(ids) => format!("{{\"extend\":{}}}", ids.len()),
            Self::Generate { prompt, .. } => format!("{{\"generate\":{}}}", prompt.len()),
            Self::Rewind(id) => format!("{{\"rewind\":[{},{}]}}", id.epoch, id.pos),
            Self::SyntheticFill { tokens, seed } => {
                format!("{{\"synthetic_fill\":[{tokens},{seed}]}}")
            }
            Self::KvFingerprint => "\"fingerprint\"".into(),
            Self::Ping => "\"ping\"".into(),
            Self::Mark => "\"mark\"".into(),
            Self::Activate { prompt } => format!("{{\"activate\":{}}}", prompt.len()),
            Self::Finalize { conv_id, canonical } => {
                format!("{{\"finalize\":[{conv_id},{}]}}", canonical.len())
            }
            Self::AlignTo { target } => format!("{{\"align_to\":{}}}", target.len()),
            Self::Shutdown => "\"shutdown\"".into(),
        }
    }
}

pub struct Pipeline {
    tx: Sender<PipelineOp>,
    rx: Receiver<PipelineEvent>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl PipelineStage for Pipeline {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        Pipeline::call(self, op)
    }
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
    let session = match StepGenerateSession::open(model_dir, &open_cfg, None) {
        Ok((s, _)) => s,
        Err(err) => {
            let _ = ev_tx.send(PipelineEvent::Error(format!("session open: {err}")));
            return;
        }
    };
    // The multi-conversation registry lives ON the pipeline thread with the
    // session it wraps (serve's Activate/Finalize become ops; single-client
    // callers like ask/chat simply never Activate).
    let mut manager = crate::conversation::ConversationManager::new(
        session,
        crate::flags::conv_cache_bytes(),
        crate::flags::conv_disk_bytes(),
        crate::flags::conv_cache_dir(),
    );

    let mut epoch: u64 = 0;
    let kv_id = |epoch: u64, manager: &mut crate::conversation::ConversationManager| KvId {
        epoch,
        pos: manager.session_mut().kv_valid_tokens().len(),
    };

    while let Ok(op) = op_rx.recv() {
        let event = match op {
            PipelineOp::Extend(ids) => match manager.session_mut().extend_kv(&ids) {
                Ok(()) => PipelineEvent::Extended {
                    kv: kv_id(epoch, &mut manager),
                },
                Err(err) => PipelineEvent::Error(format!("extend: {err}")),
            },
            PipelineOp::Generate { prompt, cfg, label } => {
                let mut cfg = *cfg;
                cfg.layers = layers;
                cfg.max_seq = max_seq;
                match generate_with_session(manager.session_mut(), &prompt, &cfg, &label) {
                    Ok(out) => PipelineEvent::Generated {
                        out: Box::new(out),
                        kv: kv_id(epoch, &mut manager),
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
                    match manager.session_mut().truncate_kv_to(id.pos) {
                        Ok(()) => {
                            epoch += 1;
                            PipelineEvent::Rewound {
                                kv: kv_id(epoch, &mut manager),
                            }
                        }
                        Err(err) => PipelineEvent::Error(format!("rewind: {err}")),
                    }
                }
            }
            PipelineOp::SyntheticFill { tokens, seed } => {
                match manager.session_mut().synthetic_fill_kv(tokens, seed) {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Filled {
                            kv: kv_id(epoch, &mut manager),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("synthetic fill: {err}")),
                }
            }
            PipelineOp::KvFingerprint => PipelineEvent::Fingerprint {
                fnv: manager.session_mut().live_kv_fingerprint(),
                kv: kv_id(epoch, &mut manager),
            },
            PipelineOp::Ping => PipelineEvent::Pong,
            PipelineOp::Mark => PipelineEvent::Marked {
                kv: kv_id(epoch, &mut manager),
            },
            PipelineOp::Activate { prompt } => {
                let conv_id = manager.activate(&prompt);
                epoch += 1;
                let reused = manager
                    .session_mut()
                    .kv_valid_tokens()
                    .iter()
                    .zip(prompt.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                PipelineEvent::Activated {
                    conv_id,
                    kv: kv_id(epoch, &mut manager),
                    reused,
                }
            }
            PipelineOp::Finalize { conv_id, canonical } => {
                match manager.finalize(conv_id, &canonical) {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Finalized {
                            kv: kv_id(epoch, &mut manager),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("finalize: {err}")),
                }
            }
            PipelineOp::AlignTo { target } => {
                let session = manager.session_mut();
                let reused = session
                    .kv_valid_tokens()
                    .iter()
                    .zip(target.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                let aligned = session
                    .truncate_kv_to(reused)
                    .and_then(|()| session.extend_kv(&target[reused..]));
                match aligned {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Aligned {
                            kv: kv_id(epoch, &mut manager),
                            reused,
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("align: {err}")),
                }
            }
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
