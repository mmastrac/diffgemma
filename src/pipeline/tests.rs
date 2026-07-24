use super::*;

/// A scripted inner stage for validator tests: returns canned events in
/// order and records the op names it saw.
struct ScriptedStage {
    script: std::cell::RefCell<Vec<PipelineEvent>>,
    seen: std::cell::RefCell<Vec<&'static str>>,
}

impl PipelineStage for ScriptedStage {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        self.seen.borrow_mut().push(op.name());
        self.script.borrow_mut().remove(0)
    }
}

fn generated_event(token_ids: Vec<u32>, kv: KvId) -> PipelineEvent {
    PipelineEvent::Generated {
        out: Box::new(crate::generate::GenerateOutput {
            token_ids,
            denoise_steps_run: 1,
            blocks_committed: 1,
            cancelled: false,
            stopped_on_eot: true,
            stop_token_id: None,
            stop_block_idx: None,
            stop_offset: None,
            block_stats: Vec::new(),
            block_steps_eff: vec![1],
            last_block_accept_hist: Vec::new(),
            last_block_min_entropy_hist: Vec::new(),
            prefill_elapsed: std::time::Duration::ZERO,
            denoise_elapsed: std::time::Duration::ZERO,
            extend_elapsed: std::time::Duration::ZERO,
            #[cfg(target_os = "macos")]
            session_telemetry: crate::metal::SessionTelemetry::default(),
            #[cfg(target_os = "macos")]
            denoise_trace: None,
        }),
        kv,
    }
}

/// Scripted inner for the repair test: echoes each Generate's prompt with
/// a canned reply suffix (so the stage's self-built repair prompt flows
/// through naturally), and records ops + Generate prompt lengths.
struct RepairScript {
    replies: std::cell::RefCell<Vec<Vec<u32>>>,
    seen: std::cell::RefCell<Vec<&'static str>>,
    gen_prompt_lens: std::cell::RefCell<Vec<usize>>,
    rewind_pos: std::cell::RefCell<Vec<usize>>,
}

impl PipelineStage for RepairScript {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        self.seen.borrow_mut().push(op.name());
        // Fresh conversation: the pre-generate mark sits at position 0.
        let kv = KvId { epoch: 0, pos: 0 };
        match op {
            PipelineOp::Mark => PipelineEvent::Marked { kv },
            PipelineOp::Rewind(id) => {
                self.rewind_pos.borrow_mut().push(id.pos);
                PipelineEvent::Rewound { kv: id }
            }
            PipelineOp::Generate { prompt, .. } => {
                self.gen_prompt_lens.borrow_mut().push(prompt.len());
                let reply = self.replies.borrow_mut().remove(0);
                let mut ids = prompt;
                ids.extend(reply);
                generated_event(ids, kv)
            }
            _ => PipelineEvent::Error("unexpected op".into()),
        }
    }
}

/// A call emitted inside the thinking block gets the tailored error
/// (naming the lost call), not the generic malformed-call text.
#[test]
fn repair_error_responses_for_thinking_call() {
    let lost =
        "<|channel>thought\n<|tool_call>call:write{path:<|\"|>/tmp/x<|\"|>}<tool_call|><channel|>";
    let errs =
        crate::tools::strip_client_guards(&ToolRepairStage::<RepairScript>::error_responses(lost));
    assert!(
        errs.contains("inside your thinking block"),
        "missing thinking-specific error: {errs}"
    );
    assert!(
        errs.contains("response:write"),
        "error must name the lost call: {errs}"
    );

    let malformed = "<|tool_call>call:write{path:";
    let errs = crate::tools::strip_client_guards(
        &ToolRepairStage::<RepairScript>::error_responses(malformed),
    );
    assert!(
        errs.contains("was malformed"),
        "generic path broken: {errs}"
    );
}

/// The repair stage's choreography, pinned against a scripted inner: an
/// invalid first reply (incomplete call) triggers error-response feedback
/// (the second Generate's prompt = the failed turn + the rendered error
/// tool response), then a Rewind — and the caller receives the ORIGINAL
/// prompt + the corrected reply, with the corrupt exchange gone.
#[test]
fn tool_repair_feeds_error_and_rewinds_corrupt_call() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let tokenizer = std::sync::Arc::new(
        crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer"),
    );
    let prompt: Vec<u32> = vec![10, 11, 12];
    let bad_reply = tokenizer.encode_with_specials("<|tool_call>call:write{path:");
    let good_reply = tokenizer
        .encode_with_specials("<|tool_call>call:write{path:<|\"|>/tmp/x.rs<|\"|>}<tool_call|>");
    // The regeneration runs to eos, so it may append a hallucinated tool
    // response — the stage must trim from the opener on.
    let mut regen_reply = good_reply.clone();
    regen_reply.extend(
        tokenizer.encode_with_specials("<|tool_response>response:write{value:<|\"|>fake<|\"|>}"),
    );
    let inner = RepairScript {
        replies: std::cell::RefCell::new(vec![bad_reply.clone(), regen_reply]),
        seen: std::cell::RefCell::new(Vec::new()),
        gen_prompt_lens: std::cell::RefCell::new(Vec::new()),
        rewind_pos: std::cell::RefCell::new(Vec::new()),
    };
    let repair = ToolRepairStage::new(inner, tokenizer, 1);
    let mut cfg = crate::metal::StepGenerateConfig::from_generate(
        42,
        64,
        1024,
        0,
        crate::sample::sampler_for_steps(8, false),
        false,
    );
    // The repair must announce itself through the status hook exactly
    // once (serve renders this as a thinking block on the stream).
    let statuses = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let statuses_sink = std::sync::Arc::clone(&statuses);
    cfg.status_notify = Some(std::sync::Arc::new(move |msg: &str| {
        statuses_sink.lock().unwrap().push(msg.to_string());
    }));
    let ev = repair.call(PipelineOp::Generate {
        prompt: prompt.clone(),
        cfg: Box::new(cfg),
        label: "repair-test".into(),
    });
    assert_eq!(
        *statuses.lock().unwrap(),
        vec!["I made a mistake with my tool call, let me try again.".to_string()],
        "repair must emit exactly one status notification"
    );
    let PipelineEvent::Generated { out, .. } = ev else {
        panic!("repair swallowed the Generated event: {ev:?}");
    };
    let mut expect = prompt.clone();
    expect.extend(good_reply);
    assert_eq!(
        out.token_ids, expect,
        "caller must receive original prompt + the corrected reply, hallucinated response trimmed"
    );
    assert_eq!(
        *repair.inner.seen.borrow(),
        vec!["mark", "generate", "generate", "rewind"],
        "repair choreography mismatch"
    );
    let lens = repair.inner.gen_prompt_lens.borrow();
    assert!(
        lens[1] > prompt.len() + bad_reply.len(),
        "repair prompt must include the failed turn + the error response ({} vs {})",
        lens[1],
        prompt.len() + bad_reply.len()
    );
    // KV-reuse-first: the rewind must land at the end of the prompt (its
    // prefill is a valid causal prefix), not the position-0 mark.
    assert_eq!(*repair.inner.rewind_pos.borrow(), vec![prompt.len()]);
}

/// The validator's retry choreography, pinned against a scripted inner
/// stage: malformed first reply -> Mark, Generate, Rewind, Generate — and
/// the second (clean) reply is what the caller receives. No GPU; the
/// tokenizer comes from the model dir.
#[test]
fn tool_validator_rewinds_and_retries_malformed_reply() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let tokenizer = std::sync::Arc::new(
        crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer"),
    );
    let prompt: Vec<u32> = vec![10, 11, 12];
    let malformed: Vec<u32> = {
        let mut ids = prompt.clone();
        ids.extend(tokenizer.encode_with_specials("<|tool_call>call:list_dir{path:"));
        ids
    };
    let clean: Vec<u32> = {
        let mut ids = prompt.clone();
        ids.extend(tokenizer.encode_with_specials("All done."));
        ids
    };
    let kv = |pos| KvId { epoch: 0, pos };
    let inner = ScriptedStage {
        script: std::cell::RefCell::new(vec![
            PipelineEvent::Marked { kv: kv(3) },
            generated_event(malformed, kv(3)),
            PipelineEvent::Rewound { kv: kv(3) },
            generated_event(clean.clone(), kv(3)),
        ]),
        seen: std::cell::RefCell::new(Vec::new()),
    };
    let validator = ToolValidatorStage::new(inner, tokenizer, 1);
    let cfg = crate::metal::StepGenerateConfig::from_generate(
        42,
        64,
        1024,
        0,
        crate::sample::sampler_for_steps(8, false),
        false,
    );
    let ev = validator.call(PipelineOp::Generate {
        prompt,
        cfg: Box::new(cfg),
        label: "validator-test".into(),
    });
    let PipelineEvent::Generated { out, .. } = ev else {
        panic!("validator swallowed the Generated event: {ev:?}");
    };
    assert_eq!(
        out.token_ids, clean,
        "caller must receive the retried reply"
    );
    assert_eq!(
        *validator.inner.seen.borrow(),
        vec!["mark", "generate", "rewind", "generate"],
        "retry choreography mismatch"
    );
}

/// The op-log round trip: ops journaled by OpLogStage reconstruct via
/// `from_log_json` and re-execute on a fresh pipeline to bit-identical
/// event digests (fingerprints, generated ids FNV, lineage ids). This is
/// the standing replay gate — a field ops.jsonl IS a repro artifact.
#[test]
fn oplog_roundtrip_replays_bit_identically() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let log_path =
        std::env::temp_dir().join(format!("dgq-oplog-gate-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&log_path);
    let ids: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 5087) % 30000).collect();
    let cfg = || {
        crate::metal::StepGenerateConfig::from_generate(
            42,
            192,
            4096,
            0,
            crate::sample::sampler_for_steps(24, false),
            false,
        )
    };

    // Record a session through the op-log.
    {
        let stage = OpLogStage::new(Pipeline::spawn(dir.clone(), 4096, 24), &log_path)
            .expect("op-log open");
        let PipelineEvent::Extended { .. } = stage.call(PipelineOp::Extend(ids.clone())) else {
            panic!("extend failed");
        };
        let PipelineEvent::Marked { kv: mark } = stage.call(PipelineOp::Mark) else {
            panic!("mark failed");
        };
        let PipelineEvent::Fingerprint { .. } = stage.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };
        let PipelineEvent::Generated { .. } = stage.call(PipelineOp::Generate {
            prompt: ids.clone(),
            cfg: Box::new(cfg()),
            label: "oplog-gate".into(),
        }) else {
            panic!("generate failed");
        };
        let PipelineEvent::Rewound { .. } = stage.call(PipelineOp::Rewind(mark)) else {
            panic!("rewind failed");
        };
        let PipelineEvent::Fingerprint { .. } = stage.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };
        // Stage (and its pipeline) drop here — the GPU session closes
        // before the replay pipeline opens.
    }

    // Replay the journal on a fresh pipeline; every event digest must match.
    let text = std::fs::read_to_string(&log_path).expect("read op-log");
    let p = Pipeline::spawn(dir.clone(), 4096, 24);
    for (lineno, line) in text.lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(line).expect("op-log line parses");
        let (Some(op_json), Some(recorded)) = (v.get("op"), v.get("event")) else {
            panic!("line {} has no op/event: {line}", lineno + 1);
        };
        let op = PipelineOp::from_log_json(op_json, &dir)
            .unwrap_or_else(|| panic!("line {}: op did not reconstruct", lineno + 1));
        let got = p.call(op).log_json();
        assert_eq!(
            &got,
            recorded,
            "replay diverged at line {} ({})",
            lineno + 1,
            op_json
        );
    }
    let _ = std::fs::remove_file(&log_path);
}

/// Wrap-crossing rewind gate (now load-bearing):
/// every other byte-consistency gate runs below the sliding-ring wrap,
/// but real sessions rewind — guard re-rolls, cancels, salvage truncates
/// — at 5-13k tokens, past it. At ring 4096: (a) generate → rewind at
/// ~5k takes the past-wrap O(1) truncate and must restore the base
/// fingerprint; (b) a rewind deeper than the slack takes the ring-REBUILD
/// path and must land bit-identical to a fresh build of the same prefix.
#[test]
fn wrap_crossing_rewind_restores_state() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 8192, 8);
    // 5,000 tokens: past the default 4096 ring wrap.
    let ids: Vec<u32> = (0..5000u32).map(|i| 1000 + (i * 2477) % 30000).collect();
    let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
        panic!("extend failed");
    };
    let PipelineEvent::Fingerprint { fnv: fp_base, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };

    let cfg = crate::metal::StepGenerateConfig::from_generate(
        42,
        384,
        8192,
        0,
        crate::sample::sampler_for_steps(8, false),
        false,
    );
    let PipelineEvent::Generated { .. } = p.call(PipelineOp::Generate {
        prompt: ids.clone(),
        cfg: Box::new(cfg),
        label: "wrap-gate".into(),
    }) else {
        panic!("generate failed");
    };
    // (a) Past-wrap rewind of the generated turn: O(1) truncate territory.
    let ev = p.call(PipelineOp::Rewind(mark));
    let PipelineEvent::Rewound { kv } = ev else {
        panic!("rewind failed: {ev:?}");
    };
    let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };
    assert_eq!(fnv, fp_base, "past-wrap generate->rewind left residue");

    // (b) Deep rewind (5000 -> 1500 = 3500 back, beyond the 2817 slack):
    // the ring-rebuild path. Must equal a fresh build of ids[..1500].
    let deep = KvId {
        epoch: kv.epoch,
        pos: 1500,
    };
    let ev = p.call(PipelineOp::Rewind(deep));
    let PipelineEvent::Rewound { kv } = ev else {
        panic!("deep rewind failed: {ev:?}");
    };
    let PipelineEvent::Fingerprint {
        fnv: fp_rebuilt, ..
    } = p.call(PipelineOp::KvFingerprint)
    else {
        panic!("fingerprint failed");
    };
    let zero = KvId {
        epoch: kv.epoch,
        pos: 0,
    };
    let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(zero)) else {
        panic!("rewind to 0 failed");
    };
    let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids[..1500].to_vec())) else {
        panic!("fresh extend failed");
    };
    let PipelineEvent::Fingerprint { fnv: fp_fresh, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };
    assert_eq!(
        fp_rebuilt, fp_fresh,
        "ring-rebuild rewind diverged from a fresh build of the same prefix"
    );
}

/// In-block re-roll residue gate: a discarded canvas attempt must leave NO
/// state the retry or later rewinds can observe. Forces exactly
/// one degenerate-reply re-roll per round (via cfg.degenerate_reply_check)
/// inside the standing generate → rewind loop: regeneration must stay
/// bit-identical across rounds and every rewind must restore the base
/// fingerprint. Also exercises shrink-on-retry (the retry canvas is 128).
#[test]
fn forced_reroll_leaves_no_residue() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 4096, 8);
    let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 6659) % 30000).collect();
    let PipelineEvent::Extended { kv: mut mark } = p.call(PipelineOp::Extend(ids.clone())) else {
        panic!("extend failed");
    };
    let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };

    let mut first_reply: Option<Vec<u32>> = None;
    for round in 0..2 {
        let mut cfg = crate::metal::StepGenerateConfig::from_generate(
            42,
            192,
            4096,
            0,
            crate::sample::sampler_for_steps(8, false),
            false,
        );
        // Fires once per round: attempt 0's canvas is declared degenerate
        // and discarded; attempt 1 (shrunk canvas) is the reply.
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        cfg.degenerate_reply_check = Some(std::sync::Arc::new(move |_ids: &[u32]| {
            !fired.swap(true, std::sync::atomic::Ordering::Relaxed)
        }));
        let ev = p.call(PipelineOp::Generate {
            prompt: ids.clone(),
            cfg: Box::new(cfg),
            label: "reroll-gate".into(),
        });
        let PipelineEvent::Generated { out, .. } = ev else {
            panic!("generate failed at round {round}: {ev:?}");
        };
        match &first_reply {
            None => first_reply = Some(out.token_ids.clone()),
            Some(f) => assert_eq!(
                f, &out.token_ids,
                "re-rolled regeneration diverged (round {round})"
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
        assert_eq!(
            fnv, fp0,
            "discarded re-roll attempt left KV residue (round {round})"
        );
    }
}

/// The KV-reuse-first salvage gate: a follow-up prompt that diverges from
/// the finalized conversation only in its tail must route BACK to that
/// conversation with the shared prefix reused (truncate-to-LCP on
/// activate) — not fork a fresh conversation with reused: 0, which was
/// the OpenCode full-re-prefill-per-tool-turn bug. Deep divergence still
/// declines (a rebuild-depth truncate costs a fresh prefill anyway).
#[test]
fn activate_salvages_tail_divergent_conversation() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 4096, 24);
    let canonical: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 3571) % 30000).collect();

    let ev = p.call(PipelineOp::Activate {
        prompt: canonical.clone(),
    });
    let PipelineEvent::Activated {
        conv_id, reused, ..
    } = ev
    else {
        panic!("first activate failed: {ev:?}");
    };
    assert_eq!(reused, 0);
    let PipelineEvent::Finalized { kv } = p.call(PipelineOp::Finalize {
        conv_id,
        canonical: canonical.clone(),
    }) else {
        panic!("finalize failed");
    };
    assert_eq!(kv.pos, canonical.len());

    // Follow-up whose last 40 canonical tokens were re-rendered.
    let mut next = canonical[..560].to_vec();
    next.extend((0..240u32).map(|i| 7000 + i));
    let ev = p.call(PipelineOp::Activate { prompt: next });
    let PipelineEvent::Activated {
        conv_id: id2,
        reused,
        kv,
    } = ev
    else {
        panic!("salvage activate failed: {ev:?}");
    };
    assert_eq!(id2, conv_id, "tail divergence must salvage, not fork");
    assert_eq!(reused, 560, "shared prefix not reused");
    assert_eq!(
        kv.pos, 560,
        "resident KV not truncated to the shared prefix"
    );

    // Divergence far above the tail slack: fresh conversation.
    let mut deep = canonical[..100].to_vec();
    deep.extend((0..300u32).map(|i| 8000 + i));
    let ev = p.call(PipelineOp::Activate { prompt: deep });
    let PipelineEvent::Activated {
        conv_id: id3,
        reused,
        ..
    } = ev
    else {
        panic!("deep activate failed: {ev:?}");
    };
    assert_ne!(
        id3, conv_id,
        "deep divergence must not steal the conversation"
    );
    assert_eq!(reused, 0);
}

/// The Splice gate: surgically replacing a mid-log span must leave state
/// byte-identical to building the target log fresh — no residue from the
/// removed span, and the tail re-encode at its new offset must be
/// bit-deterministic (the offset-resume property, exercised at a splice
/// boundary instead of a turn boundary).
#[test]
fn splice_matches_fresh_build() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 4096, 24);
    let ids: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 6151) % 30000).collect();
    let replacement: Vec<u32> = (0..50u32).map(|i| 9000 + i * 17).collect();
    let mut target = ids[..200].to_vec();
    target.extend_from_slice(&replacement);
    target.extend_from_slice(&ids[300..]);

    let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids.clone())) else {
        panic!("extend failed");
    };
    let ev = p.call(PipelineOp::Splice {
        start: 200,
        end: 300,
        replacement,
    });
    let PipelineEvent::Spliced {
        kv,
        removed,
        inserted,
    } = ev
    else {
        panic!("splice failed: {ev:?}");
    };
    assert_eq!((removed, inserted), (100, 50));
    assert_eq!(kv.pos, target.len(), "spliced log length wrong");
    let PipelineEvent::Fingerprint {
        fnv: fp_spliced, ..
    } = p.call(PipelineOp::KvFingerprint)
    else {
        panic!("fingerprint failed");
    };

    // Fresh build of the identical target log on the same session.
    let zero = KvId {
        epoch: kv.epoch,
        pos: 0,
    };
    let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(zero)) else {
        panic!("rewind to 0 failed");
    };
    let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(target)) else {
        panic!("fresh extend failed");
    };
    let PipelineEvent::Fingerprint { fnv: fp_fresh, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };
    assert_eq!(
        fp_spliced, fp_fresh,
        "splice left residue vs a fresh build of the same log"
    );

    // Range validation fails loudly.
    match p.call(PipelineOp::Splice {
        start: 10_000,
        end: 10_001,
        replacement: vec![1],
    }) {
        PipelineEvent::Error(msg) => {
            assert!(msg.contains("out of bounds"), "wrong error: {msg}")
        }
        ev => panic!("out-of-range splice must fail, got {ev:?}"),
    }
}

/// The P2 equivalence gate: a client driving
/// BeginTurn/ProposeBlock/CommitBlock/EndTurn with the default policy
/// (commit everything, mirror the monolithic extend rule) must produce
/// byte-identical token ids to the whole-turn Generate op at the same
/// seed. Pins the per-block protocol to the path it decomposed.
#[test]
fn per_block_ops_match_monolithic_generate() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 4096, 24);
    let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 7919) % 30000).collect();
    let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
        panic!("extend failed");
    };

    let max_new = 512usize; // two blocks
    let cfg = || {
        crate::metal::StepGenerateConfig::from_generate(
            42,
            max_new,
            4096,
            0, // session-owned; overwritten by the pipeline
            crate::sample::sampler_for_steps(24, false),
            false,
        )
    };
    let ev = p.call(PipelineOp::Generate {
        prompt: ids.clone(),
        cfg: Box::new(cfg()),
        label: "p2-mono".into(),
    });
    let PipelineEvent::Generated { out: mono, .. } = ev else {
        panic!("monolithic generate failed: {ev:?}");
    };

    let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(mark)) else {
        panic!("rewind failed");
    };

    let ev = p.call(PipelineOp::BeginTurn {
        prompt: ids.clone(),
        cfg: Box::new(cfg()),
        label: "p2-ops".into(),
    });
    let PipelineEvent::TurnStarted { .. } = ev else {
        panic!("begin_turn failed: {ev:?}");
    };
    let mut new_tokens = 0usize;
    loop {
        match p.call(PipelineOp::ProposeBlock) {
            PipelineEvent::Proposed {
                ids: block, stop, ..
            } => {
                let kept = stop.map_or(block.len(), |(off, _)| off);
                let end = stop.is_some();
                let remaining = max_new - new_tokens;
                let is_last = remaining <= 256;
                let extend = !end && !is_last && kept > 0;
                let ev = p.call(PipelineOp::CommitBlock {
                    kept_len: kept,
                    extend,
                });
                let PipelineEvent::BlockCommitted { new_tokens: n, .. } = ev else {
                    panic!("commit failed: {ev:?}");
                };
                new_tokens = n;
                if end {
                    break;
                }
            }
            PipelineEvent::TurnStalled { .. } => break,
            ev => panic!("unexpected propose event: {ev:?}"),
        }
    }
    let ev = p.call(PipelineOp::EndTurn);
    let PipelineEvent::Generated { out: ops, .. } = ev else {
        panic!("end_turn failed: {ev:?}");
    };
    assert_eq!(
        mono.token_ids, ops.token_ids,
        "per-block ops diverged from the monolithic path"
    );
    assert_eq!(mono.blocks_committed, ops.blocks_committed);
}

/// Partial commit (good-prefix/bad-tail) + discard leave a consistent
/// lineage: EndTurn reports exactly the committed prefix, a second
/// proposal generates beyond it, lineage-mutating ops are rejected while
/// the turn is open, and a rewind past the whole turn restores the base
/// fingerprint bit-exactly.
#[test]
fn per_block_partial_commit_and_discard_consistency() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 4096, 24);
    let ids: Vec<u32> = (0..400u32).map(|i| 2000 + (i * 4093) % 30000).collect();
    let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
        panic!("extend failed");
    };
    let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };

    let cfg = crate::metal::StepGenerateConfig::from_generate(
        7,
        512,
        4096,
        0,
        crate::sample::sampler_for_steps(24, false),
        false,
    );
    let ev = p.call(PipelineOp::BeginTurn {
        prompt: ids.clone(),
        cfg: Box::new(cfg),
        label: "p2-partial".into(),
    });
    let PipelineEvent::TurnStarted { .. } = ev else {
        panic!("begin_turn failed: {ev:?}");
    };

    // Rewind is a lineage op: rejected while the turn is open.
    match p.call(PipelineOp::Rewind(mark)) {
        PipelineEvent::Error(msg) => {
            assert!(msg.contains("turn is open"), "wrong error: {msg}")
        }
        ev => panic!("rewind during open turn must fail, got {ev:?}"),
    }

    let PipelineEvent::Proposed { ids: block, .. } = p.call(PipelineOp::ProposeBlock) else {
        panic!("propose failed");
    };
    assert!(!block.is_empty());
    let kept = block.len() / 2;
    let ev = p.call(PipelineOp::CommitBlock {
        kept_len: kept,
        extend: true,
    });
    let PipelineEvent::BlockCommitted { new_tokens, .. } = ev else {
        panic!("partial commit failed: {ev:?}");
    };
    assert_eq!(new_tokens, kept);

    // The next proposal builds on the committed prefix; reject it.
    let PipelineEvent::Proposed { .. } = p.call(PipelineOp::ProposeBlock) else {
        panic!("second propose failed");
    };
    let PipelineEvent::BlockDiscarded { .. } = p.call(PipelineOp::DiscardBlock) else {
        panic!("discard failed");
    };

    let PipelineEvent::Generated { out, .. } = p.call(PipelineOp::EndTurn) else {
        panic!("end_turn failed");
    };
    assert_eq!(
        out.token_ids.len(),
        ids.len() + kept,
        "committed prefix mismatch"
    );
    assert_eq!(out.blocks_committed, 1);
    assert_eq!(&out.token_ids[ids.len()..], &block[..kept]);

    let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(mark)) else {
        panic!("rewind after turn failed");
    };
    let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };
    assert_eq!(fnv, fp0, "partial-commit turn left KV residue");
}

/// The standing rewind byte-consistency gate.
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

    let PipelineEvent::Extended { kv: mut mark } = p.call(PipelineOp::Extend(ids.clone())) else {
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

/// Long-context extension byte-consistency on a synthetic 100k KV: a
/// pseudorandom KV declared 100k tokens costs ~1 s
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
    let PipelineEvent::Fingerprint { fnv: fp_base, .. } = p.call(PipelineOp::KvFingerprint) else {
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
    let direct = crate::generate::generate_monolithic_gpu(&dir, &ids, &gen_cfg, 1024, "eq-gate")
        .expect("direct generate");
    let piped =
        crate::generate::generate_monolithic_gpu_pipeline(&dir, &ids, &gen_cfg, 1024, "eq-gate")
            .expect("pipeline generate");
    assert_eq!(
        direct.token_ids, piped.token_ids,
        "pipeline ask path diverged from the direct path"
    );
}

/// The cancel gate: a [`crate::metal::CancelToken`] riding in the Generate
/// cfg (the same way observer Arcs do) stops an in-flight generation
/// between denoise steps, and the abandoned canvas leaves NO residue — a
/// rewind to the pre-generate mark must restore the base fingerprint
/// exactly. This is the serve dead-socket fix's core contract (the stream
/// observer cancels when the client hangs up).
#[test]
fn pipeline_cancel_stops_generate_kv_clean() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let p = Pipeline::spawn(dir, 4096, 24);
    let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 7919) % 30000).collect();
    let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
        panic!("extend failed");
    };
    let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };

    let mut cfg = crate::metal::StepGenerateConfig::from_generate(
        42,
        2048, // 8 blocks — far more than can denoise before the cancel lands
        4096,
        0, // session-owned; overwritten by the pipeline
        crate::sample::sampler_for_steps(24, false),
        false,
    );
    let cancel = crate::metal::CancelToken::new();
    cfg.cancel = Some(cancel.clone());
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1500));
        cancel.cancel();
    });
    let ev = p.call(PipelineOp::Generate {
        prompt: ids.clone(),
        cfg: Box::new(cfg),
        label: "cancel-gate".into(),
    });
    canceller.join().expect("canceller thread");
    let PipelineEvent::Generated { out, .. } = ev else {
        panic!("generate failed: {ev:?}");
    };
    assert!(
        out.cancelled,
        "generate ran to completion before the cancel"
    );
    assert!(
        out.token_ids.len() < ids.len() + 2048,
        "cancelled generate still produced the full budget"
    );

    let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(mark)) else {
        panic!("rewind after cancel failed");
    };
    let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
        panic!("fingerprint failed");
    };
    assert_eq!(fnv, fp0, "cancelled generate left KV residue past the mark");
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
