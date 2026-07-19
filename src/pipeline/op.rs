use super::KvId;

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
    /// Surgically replace `[start..end)` of the causal token log with
    /// `replacement`, re-encoding the tail at its new offset. The
    /// anti-collapse move: failed attempts, malformed calls, and expired
    /// transient hints are REMOVED from context instead of accumulating
    /// (tool-compact substitution and span handles are Splice clients).
    /// Composite of Rewind(start) + Extend(replacement + tail). Bumps the
    /// epoch.
    Splice {
        start: usize,
        end: usize,
        replacement: Vec<u32>,
    },
    /// Open a per-block turn: prefill `prompt`, then drive generation with
    /// `ProposeBlock`/`CommitBlock`/`DiscardBlock`, closing with `EndTurn`.
    /// While a turn is open, lineage-mutating ops are rejected. `Generate`
    /// remains the whole-turn composite (identical machinery, default policy).
    BeginTurn {
        prompt: Vec<u32>,
        cfg: Box<crate::metal::StepGenerateConfig>,
        label: String,
    },
    /// Denoise one block WITHOUT committing it. The `Proposed` event carries
    /// the canvas argmax for inspection; nothing lands in the token log or
    /// causal KV until `CommitBlock`.
    ProposeBlock,
    /// Commit `kept_len` tokens of the pending proposal (partial commit =
    /// good-prefix/bad-tail). `extend: false` skips the causal KV re-encode —
    /// the monolithic contract for turn-final blocks.
    CommitBlock {
        kept_len: usize,
        extend: bool,
    },
    /// Drop the pending proposal; the next `ProposeBlock` re-rolls the canvas
    /// from fresh noise (the rng advances every attempt).
    DiscardBlock,
    /// Close the open turn: record `kv_valid_tokens` and return the assembled
    /// `Generated` output (same shape as the whole-turn op).
    EndTurn,
    Shutdown,
}

impl PipelineOp {
    /// Short op name for error messages and progress lines.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Extend(_) => "extend",
            Self::Generate { .. } => "generate",
            Self::Rewind(_) => "rewind",
            Self::SyntheticFill { .. } => "synthetic_fill",
            Self::KvFingerprint => "fingerprint",
            Self::Ping => "ping",
            Self::Mark => "mark",
            Self::Activate { .. } => "activate",
            Self::Finalize { .. } => "finalize",
            Self::AlignTo { .. } => "align_to",
            Self::Splice { .. } => "splice",
            Self::BeginTurn { .. } => "begin_turn",
            Self::ProposeBlock => "propose_block",
            Self::CommitBlock { .. } => "commit_block",
            Self::DiscardBlock => "discard_block",
            Self::EndTurn => "end_turn",
            Self::Shutdown => "shutdown",
        }
    }

    /// Full-fidelity JSON for the op-log: every id, plus the replayable core
    /// of a Generate/BeginTurn cfg. Observers and cancel tokens don't
    /// serialize — they never alter the token trajectory. `replay`
    /// reconstructs ops from exactly this shape (see `op_from_log_json`).
    pub fn log_json(&self) -> serde_json::Value {
        use serde_json::json;
        fn cfg_json(cfg: &crate::metal::StepGenerateConfig) -> serde_json::Value {
            json!({
                "seed": cfg.seed,
                "max_new_tokens": cfg.max_new_tokens,
                "steps": cfg.sampler.max_denoising_steps,
                "no_early_stop": cfg.no_early_stop,
                "sampler_no_early_stop": cfg.sampler.confidence_threshold == f32::MAX,
                "stop_token_ids": cfg.stop_token_ids,
                "continue_incomplete_tool_calls": cfg.continue_incomplete_tool_calls,
                "quote_token_id": cfg.quote_token_id,
                "degenerate_reply_check": cfg.degenerate_reply_check.is_some(),
            })
        }
        match self {
            Self::Extend(ids) => json!({"extend": {"ids": ids}}),
            Self::Generate { prompt, cfg, label } => {
                json!({"generate": {"prompt": prompt, "label": label, "cfg": cfg_json(cfg)}})
            }
            Self::Rewind(id) => json!({"rewind": {"epoch": id.epoch, "pos": id.pos}}),
            Self::SyntheticFill { tokens, seed } => {
                json!({"synthetic_fill": {"tokens": tokens, "seed": seed}})
            }
            Self::KvFingerprint => json!("fingerprint"),
            Self::Ping => json!("ping"),
            Self::Mark => json!("mark"),
            Self::Activate { prompt } => json!({"activate": {"prompt": prompt}}),
            Self::Finalize { conv_id, canonical } => {
                json!({"finalize": {"conv_id": conv_id, "canonical": canonical}})
            }
            Self::AlignTo { target } => json!({"align_to": {"target": target}}),
            Self::Splice {
                start,
                end,
                replacement,
            } => json!({"splice": {"start": start, "end": end, "replacement": replacement}}),
            Self::BeginTurn { prompt, cfg, label } => {
                json!({"begin_turn": {"prompt": prompt, "label": label, "cfg": cfg_json(cfg)}})
            }
            Self::ProposeBlock => json!("propose_block"),
            Self::CommitBlock { kept_len, extend } => {
                json!({"commit_block": {"kept_len": kept_len, "extend": extend}})
            }
            Self::DiscardBlock => json!("discard_block"),
            Self::EndTurn => json!("end_turn"),
            Self::Shutdown => json!("shutdown"),
        }
    }
}

impl PipelineOp {
    /// Reconstruct an op from its [`Self::log_json`] form (the replay half).
    /// `model_dir` rebuilds the non-serializable cfg parts (the
    /// degenerate-reply check needs the tokenizer). `None` = unknown shape.
    pub fn from_log_json(v: &serde_json::Value, model_dir: &std::path::Path) -> Option<PipelineOp> {
        fn ids(v: &serde_json::Value) -> Option<Vec<u32>> {
            v.as_array()?
                .iter()
                .map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        }
        fn cfg_from(
            v: &serde_json::Value,
            model_dir: &std::path::Path,
        ) -> Option<crate::metal::StepGenerateConfig> {
            let seed = v.get("seed")?.as_u64()?;
            let max_new = v.get("max_new_tokens")?.as_u64()? as usize;
            let steps = v.get("steps")?.as_u64()? as usize;
            let no_early = v.get("no_early_stop")?.as_bool()?;
            let sampler_ne = v.get("sampler_no_early_stop")?.as_bool()?;
            let stop_ids = ids(v.get("stop_token_ids")?)?;
            let cont = v.get("continue_incomplete_tool_calls")?.as_bool()?;
            let degen = v.get("degenerate_reply_check")?.as_bool()?;
            let sampler = crate::sample::sampler_for_steps(steps, sampler_ne);
            // layers/max_seq are session-owned and overwritten by the pipeline.
            let mut cfg = crate::metal::StepGenerateConfig::from_generate(
                seed, max_new, 0, 0, sampler, no_early,
            );
            cfg.stop_token_ids = stop_ids.clone();
            cfg.continue_incomplete_tool_calls = cont;
            // Absent in pre-2026-07-18 logs → None (the plain stop-scan),
            // which is exactly what those sessions ran.
            cfg.quote_token_id = v
                .get("quote_token_id")
                .and_then(|q| q.as_u64())
                .map(|q| q as u32);
            if degen {
                cfg.degenerate_reply_check =
                    crate::chat_template::empty_reply_check(model_dir, stop_ids);
            }
            Some(cfg)
        }
        if let Some(s) = v.as_str() {
            return match s {
                "fingerprint" => Some(Self::KvFingerprint),
                "ping" => Some(Self::Ping),
                "mark" => Some(Self::Mark),
                "propose_block" => Some(Self::ProposeBlock),
                "discard_block" => Some(Self::DiscardBlock),
                "end_turn" => Some(Self::EndTurn),
                "shutdown" => Some(Self::Shutdown),
                _ => None,
            };
        }
        let obj = v.as_object()?;
        let (key, body) = obj.iter().next()?;
        match key.as_str() {
            "extend" => Some(Self::Extend(ids(body.get("ids")?)?)),
            "generate" => Some(Self::Generate {
                prompt: ids(body.get("prompt")?)?,
                cfg: Box::new(cfg_from(body.get("cfg")?, model_dir)?),
                label: body.get("label")?.as_str()?.to_string(),
            }),
            "rewind" => Some(Self::Rewind(KvId {
                epoch: body.get("epoch")?.as_u64()?,
                pos: body.get("pos")?.as_u64()? as usize,
            })),
            "synthetic_fill" => Some(Self::SyntheticFill {
                tokens: body.get("tokens")?.as_u64()? as usize,
                seed: body.get("seed")?.as_u64()?,
            }),
            "activate" => Some(Self::Activate {
                prompt: ids(body.get("prompt")?)?,
            }),
            "finalize" => Some(Self::Finalize {
                conv_id: body.get("conv_id")?.as_u64()?,
                canonical: ids(body.get("canonical")?)?,
            }),
            "align_to" => Some(Self::AlignTo {
                target: ids(body.get("target")?)?,
            }),
            "splice" => Some(Self::Splice {
                start: body.get("start")?.as_u64()? as usize,
                end: body.get("end")?.as_u64()? as usize,
                replacement: ids(body.get("replacement")?)?,
            }),
            "begin_turn" => Some(Self::BeginTurn {
                prompt: ids(body.get("prompt")?)?,
                cfg: Box::new(cfg_from(body.get("cfg")?, model_dir)?),
                label: body.get("label")?.as_str()?.to_string(),
            }),
            "commit_block" => Some(Self::CommitBlock {
                kept_len: body.get("kept_len")?.as_u64()? as usize,
                extend: body.get("extend")?.as_bool()?,
            }),
            _ => None,
        }
    }
}
