use super::{KvId, ids_fnv};

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
    /// `removed`/`inserted` = the splice delta; `kv` = the new log end.
    Spliced {
        kv: KvId,
        removed: usize,
        inserted: usize,
    },
    /// A turn is open; block ops drive it until `EndTurn`.
    TurnStarted {
        kv: KvId,
    },
    /// An uncommitted block awaits the commit decision. `stop` is the
    /// advisory stop-token scan over `cfg.stop_token_ids`: (offset, id).
    Proposed {
        ids: Vec<u32>,
        stop: Option<(usize, u32)>,
        steps_eff: u32,
        late_mean_ent: f32,
        kv: KvId,
    },
    /// No further proposals will come (budget spent, commit-guard abandon, or
    /// cancel); `EndTurn` collects the output.
    TurnStalled {
        reason: &'static str,
        kv: KvId,
    },
    BlockCommitted {
        kv: KvId,
        new_tokens: usize,
    },
    BlockDiscarded {
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
                "Generated {{ tokens: {}, blocks: {}, cancelled: {}, kv: {kv:?} }}",
                out.token_ids.len(),
                out.blocks_committed,
                out.cancelled
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
            Self::Spliced {
                kv,
                removed,
                inserted,
            } => write!(
                f,
                "Spliced {{ kv: {kv:?}, removed: {removed}, inserted: {inserted} }}"
            ),
            Self::TurnStarted { kv } => write!(f, "TurnStarted {{ kv: {kv:?} }}"),
            Self::Proposed {
                ids,
                stop,
                steps_eff,
                late_mean_ent,
                kv,
            } => write!(
                f,
                "Proposed {{ tokens: {}, stop: {stop:?}, steps_eff: {steps_eff}, late_mean_ent: {late_mean_ent:.3}, kv: {kv:?} }}",
                ids.len()
            ),
            Self::TurnStalled { reason, kv } => {
                write!(f, "TurnStalled {{ reason: {reason:?}, kv: {kv:?} }}")
            }
            Self::BlockCommitted { kv, new_tokens } => {
                write!(
                    f,
                    "BlockCommitted {{ kv: {kv:?}, new_tokens: {new_tokens} }}"
                )
            }
            Self::BlockDiscarded { kv } => write!(f, "BlockDiscarded {{ kv: {kv:?} }}"),
            Self::Error(msg) => write!(f, "Error({msg:?})"),
            Self::ShutDown => write!(f, "ShutDown"),
        }
    }
}

impl PipelineEvent {
    /// Digest JSON for the op-log: enough to DIFF a replay against (token
    /// counts, ids FNV, KV fingerprints, lineage ids) without storing every
    /// generated id.
    pub fn log_json(&self) -> serde_json::Value {
        use serde_json::json;
        fn kv_json(kv: &KvId) -> serde_json::Value {
            json!({"epoch": kv.epoch, "pos": kv.pos})
        }
        match self {
            Self::Extended { kv } => json!({"extended": {"kv": kv_json(kv)}}),
            Self::Generated { out, kv } => json!({"generated": {
                "tokens": out.token_ids.len(),
                "blocks": out.blocks_committed,
                "cancelled": out.cancelled,
                "ids_fnv": format!("{:#x}", ids_fnv(&out.token_ids)),
                "kv": kv_json(kv),
            }}),
            Self::Rewound { kv } => json!({"rewound": {"kv": kv_json(kv)}}),
            Self::Filled { kv } => json!({"filled": {"kv": kv_json(kv)}}),
            Self::Fingerprint { fnv, kv } => {
                json!({"fingerprint": {"fnv": format!("{fnv:#x}"), "kv": kv_json(kv)}})
            }
            Self::Pong => json!("pong"),
            Self::Marked { kv } => json!({"marked": {"kv": kv_json(kv)}}),
            Self::Activated {
                conv_id,
                kv,
                reused,
            } => json!({"activated": {"conv_id": conv_id, "reused": reused, "kv": kv_json(kv)}}),
            Self::Finalized { kv } => json!({"finalized": {"kv": kv_json(kv)}}),
            Self::Aligned { kv, reused } => {
                json!({"aligned": {"reused": reused, "kv": kv_json(kv)}})
            }
            Self::Spliced {
                kv,
                removed,
                inserted,
            } => json!({"spliced": {"removed": removed, "inserted": inserted, "kv": kv_json(kv)}}),
            Self::TurnStarted { kv } => json!({"turn_started": {"kv": kv_json(kv)}}),
            Self::Proposed {
                ids,
                stop,
                steps_eff,
                late_mean_ent,
                kv,
            } => json!({"proposed": {
                "tokens": ids.len(),
                "ids_fnv": format!("{:#x}", ids_fnv(ids)),
                "stop": stop.map(|(off, id)| json!([off, id])),
                "steps_eff": steps_eff,
                "late_mean_ent": late_mean_ent,
                "kv": kv_json(kv),
            }}),
            Self::TurnStalled { reason, kv } => {
                json!({"turn_stalled": {"reason": reason, "kv": kv_json(kv)}})
            }
            Self::BlockCommitted { kv, new_tokens } => {
                json!({"block_committed": {"new_tokens": new_tokens, "kv": kv_json(kv)}})
            }
            Self::BlockDiscarded { kv } => json!({"block_discarded": {"kv": kv_json(kv)}}),
            Self::Error(msg) => json!({"error": msg}),
            Self::ShutDown => json!("shutdown"),
        }
    }
}
