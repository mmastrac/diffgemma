use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

use super::{KvId, PipelineEvent, PipelineOp, PipelineStage};

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
        // Quiet is thread-local: inherit the spawner's effective choice, or
        // serve/chat's set_quiet never reaches the thread that actually runs
        // generate (which is how the per-step debug spew came back in serve).
        let inherit_quiet = !crate::flags::progress_enabled();
        let join = std::thread::Builder::new()
            .name("token-pipeline".into())
            .spawn(move || {
                crate::flags::set_quiet(inherit_quiet);
                run_pipeline(&model_dir, max_seq, steps, &op_rx, &ev_tx)
            })
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

    // Per-block turn state: at most one open turn, at most one pending
    // proposal. Lineage-mutating ops are rejected while a turn is open —
    // the turn's TurnState and the session would silently diverge otherwise.
    let mut turn: Option<Box<crate::metal::TurnState>> = None;
    let mut turn_cfg: Option<Box<crate::metal::StepGenerateConfig>> = None;
    let mut pending: Option<Box<crate::metal::ProposedBlock>> = None;
    fn conflicts_with_open_turn(op: &PipelineOp) -> bool {
        matches!(
            op,
            PipelineOp::Extend(_)
                | PipelineOp::Generate { .. }
                | PipelineOp::Rewind(_)
                | PipelineOp::SyntheticFill { .. }
                | PipelineOp::Activate { .. }
                | PipelineOp::Finalize { .. }
                | PipelineOp::AlignTo { .. }
                | PipelineOp::Splice { .. }
                | PipelineOp::BeginTurn { .. }
        )
    }

    while let Ok(op) = op_rx.recv() {
        if turn.is_some() && conflicts_with_open_turn(&op) {
            let refused = PipelineEvent::Error(format!(
                "op rejected while a turn is open (end or discard first): {}",
                op.name()
            ));
            if ev_tx.send(refused).is_err() {
                return;
            }
            continue;
        }
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
                let session = manager.session_mut();
                let reused = session
                    .kv_valid_tokens()
                    .iter()
                    .zip(prompt.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                // KV-reuse-first tail salvage: routing may match a
                // conversation whose log diverges from the prompt in its
                // tail. Truncate the resident KV to the common prefix so
                // every downstream consumer (reuse guard, delta prefill)
                // sees a clean extend — O(1) inside the routing slack.
                let salvage = if reused < session.kv_valid_tokens().len() {
                    session.truncate_kv_to(reused).err()
                } else {
                    None
                };
                match salvage {
                    None => PipelineEvent::Activated {
                        conv_id,
                        kv: kv_id(epoch, &mut manager),
                        reused,
                    },
                    Some(err) => PipelineEvent::Error(format!("activate salvage: {err}")),
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
            PipelineOp::Splice {
                start,
                end,
                replacement,
            } => {
                let session = manager.session_mut();
                let log_len = session.kv_valid_tokens().len();
                if start > end || end > log_len {
                    PipelineEvent::Error(format!(
                        "splice: range {start}..{end} out of bounds (log {log_len})"
                    ))
                } else {
                    let inserted = replacement.len();
                    let removed = end - start;
                    let mut tail = replacement;
                    tail.extend_from_slice(&session.kv_valid_tokens()[end..]);
                    let spliced = session
                        .truncate_kv_to(start)
                        .and_then(|()| session.extend_kv(&tail));
                    match spliced {
                        Ok(()) => {
                            epoch += 1;
                            PipelineEvent::Spliced {
                                kv: kv_id(epoch, &mut manager),
                                removed,
                                inserted,
                            }
                        }
                        Err(err) => PipelineEvent::Error(format!("splice: {err}")),
                    }
                }
            }
            PipelineOp::BeginTurn { prompt, cfg, label } => {
                let mut cfg = *cfg;
                cfg.layers = layers;
                cfg.max_seq = max_seq;
                match crate::metal::begin_turn(manager.session_mut(), &prompt, &cfg, &label) {
                    Ok(ts) => {
                        turn = Some(Box::new(ts));
                        turn_cfg = Some(Box::new(cfg));
                        PipelineEvent::TurnStarted {
                            kv: kv_id(epoch, &mut manager),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("begin_turn: {err}")),
                }
            }
            PipelineOp::ProposeBlock => match (turn.as_mut(), turn_cfg.as_ref()) {
                _ if pending.is_some() => PipelineEvent::Error(
                    "propose_block: proposal pending (commit or discard first)".into(),
                ),
                (Some(ts), Some(cfg)) => {
                    match crate::metal::propose_block(manager.session_mut(), cfg, ts) {
                        Ok(crate::metal::BlockOutcome::Proposal(pb)) => {
                            let stop = pb
                                .token_ids
                                .iter()
                                .position(|id| cfg.stop_token_ids.contains(id))
                                .map(|off| (off, pb.token_ids[off]));
                            let ev = PipelineEvent::Proposed {
                                ids: pb.token_ids.clone(),
                                stop,
                                steps_eff: pb.stats.steps_eff,
                                late_mean_ent: pb.stats.late_mean_ent,
                                kv: kv_id(epoch, &mut manager),
                            };
                            pending = Some(pb);
                            ev
                        }
                        Ok(outcome) => PipelineEvent::TurnStalled {
                            reason: match outcome {
                                crate::metal::BlockOutcome::Exhausted => "exhausted",
                                crate::metal::BlockOutcome::Abandoned => "abandoned",
                                crate::metal::BlockOutcome::Cancelled => "cancelled",
                                crate::metal::BlockOutcome::Proposal(_) => unreachable!(),
                            },
                            kv: kv_id(epoch, &mut manager),
                        },
                        Err(err) => PipelineEvent::Error(format!("propose_block: {err}")),
                    }
                }
                _ => PipelineEvent::Error("propose_block: no open turn".into()),
            },
            PipelineOp::CommitBlock { kept_len, extend } => {
                if pending
                    .as_ref()
                    .is_some_and(|pb| kept_len > pb.token_ids.len())
                {
                    PipelineEvent::Error(format!(
                        "commit_block: kept_len {kept_len} exceeds proposal ({})",
                        pending.as_ref().map_or(0, |pb| pb.token_ids.len())
                    ))
                } else {
                    match (turn.as_mut(), turn_cfg.as_ref(), pending.take()) {
                        (Some(ts), Some(cfg), Some(pb)) => match crate::metal::commit_block(
                            manager.session_mut(),
                            cfg,
                            ts,
                            *pb,
                            kept_len,
                            extend,
                        ) {
                            Ok(()) => PipelineEvent::BlockCommitted {
                                kv: kv_id(epoch, &mut manager),
                                new_tokens: turn.as_ref().expect("turn open").new_tokens(),
                            },
                            Err(err) => PipelineEvent::Error(format!("commit_block: {err}")),
                        },
                        (_, _, None) => {
                            PipelineEvent::Error("commit_block: no pending proposal".into())
                        }
                        _ => PipelineEvent::Error("commit_block: no open turn".into()),
                    }
                }
            }
            PipelineOp::DiscardBlock => {
                if pending.take().is_some() {
                    PipelineEvent::BlockDiscarded {
                        kv: kv_id(epoch, &mut manager),
                    }
                } else {
                    PipelineEvent::Error("discard_block: no pending proposal".into())
                }
            }
            PipelineOp::EndTurn => {
                if pending.is_some() {
                    PipelineEvent::Error(
                        "end_turn: proposal pending (commit or discard first)".into(),
                    )
                } else if let (Some(ts), Some(cfg)) = (turn.take(), turn_cfg.take()) {
                    match crate::metal::finish_turn(manager.session_mut(), &cfg, *ts) {
                        Ok(out) => PipelineEvent::Generated {
                            out: Box::new(out),
                            kv: kv_id(epoch, &mut manager),
                        },
                        Err(err) => PipelineEvent::Error(format!("end_turn: {err}")),
                    }
                } else {
                    PipelineEvent::Error("end_turn: no open turn".into())
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
