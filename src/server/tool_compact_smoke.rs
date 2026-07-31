//! Tests for `tool_compact_smoke`, extracted from server.rs (backlog item 3).

use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
use crate::pipeline::{KvId, PipelineEvent, PipelineOp, PipelineStage};
use crate::toolcompact as tc;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Minimal stage over a test-owned session (the ops `run_summarize_pass`
/// uses: Mark / Generate / Rewind) — the reference shape for wrapping an
/// existing session in the op vocabulary without spawning a pipeline thread.
struct SessionStage<'a>(std::cell::RefCell<&'a mut crate::conversation::ConversationManager>);

impl PipelineStage for SessionStage<'_> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        let mut m = self.0.borrow_mut();
        let session = m.session_mut();
        match op {
            PipelineOp::Mark => PipelineEvent::Marked {
                kv: KvId {
                    epoch: 0,
                    pos: session.kv_valid_tokens().len(),
                },
            },
            PipelineOp::Rewind(id) => match session.truncate_kv_to(id.pos) {
                Ok(()) => PipelineEvent::Rewound { kv: id },
                Err(err) => PipelineEvent::Error(format!("rewind: {err}")),
            },
            PipelineOp::Generate { prompt, cfg, label } => {
                match generate_with_session(session, &prompt, &cfg, &label) {
                    Ok(out) => PipelineEvent::Generated {
                        out: Box::new(out),
                        kv: KvId {
                            epoch: 0,
                            pos: session.kv_valid_tokens().len(),
                        },
                    },
                    Err(err) => PipelineEvent::Error(format!("generate: {err}")),
                }
            }
            _ => PipelineEvent::Error("unsupported op in SessionStage".into()),
        }
    }
}

// The verbose fixture renders ~1800 tokens; 4096 leaves room for the
// summarize prompt + a canvas block at every step.
const MAX_SEQ: usize = 4096;
const THRESHOLD: usize = 64;

fn model_dir() -> Option<PathBuf> {
    for p in ["model/diffgemma-26b-a4b-it-q4", "/tmp/quantized-weights"] {
        let d = PathBuf::from(p);
        if d.join("model.dgq.json").exists() {
            return Some(d);
        }
    }
    None
}

fn open(
    dir: &Path,
) -> (
    crate::tokenizer::Tokenizer,
    StepGenerateConfig,
    StepGenerateSession,
) {
    let tok = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let layers = crate::commands::resolve_model_layers(dir, None).unwrap();
    let stop = crate::config::load_generation_stop_tokens(dir);
    let sampler = crate::sample::sampler_for_steps(24, false);
    let mut cfg = StepGenerateConfig::from_generate(7, 512, MAX_SEQ, layers, sampler, false);
    cfg.stop_token_ids = stop.clone();
    cfg.degenerate_reply_check = crate::chat_template::empty_reply_check(dir, stop);
    let (session, _) = StepGenerateSession::open(dir, &cfg, None).unwrap();
    (tok, cfg, session)
}

fn read_file_tool() -> Value {
    json!({"type":"function","function":{
            "name":"read_file",
            "description":"Read a file from disk",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"absolute path"}},
                "required":["path"]}}})
}

/// A fake verbose tool output, comfortably over THRESHOLD tokens.
fn big_output() -> String {
    let mut s = String::from("directory listing of /data:\n");
    for i in 1..=60 {
        s.push_str(&format!(
            "file_{i:03}.txt  {}00 bytes  2026-07-0{}\n",
            i,
            i % 9 + 1
        ));
    }
    s.push_str("secret_marker=zebra42\n");
    s
}

fn base_messages(output: &str) -> Vec<Value> {
    vec![
        json!({"role":"user","content":"Read /data and tell me what's inside."}),
        json!({"role":"assistant","content":"","tool_calls":[
                {"id":"call_0","type":"function",
                 "function":{"name":"read_file","arguments":"{\"path\":\"/data\"}"}}]}),
        json!({"role":"tool","tool_call_id":"call_0","content":output}),
    ]
}

/// All three compaction smokes as phases of ONE test: the three separate
/// model loads dominated the entire default suite, and every phase after A
/// starts from a state the manager/session APIs guarantee (activate() of a
/// new conversation resets the KV). Phases:
///   A) raw extend past capacity = typed error, no partial write;
///   B) M2 expand re-entry + finalize eviction (raw session);
///   C) M1 summarize-rewind + substituted finalize (via the manager);
///   D) overlong finalize trims to capacity instead of panicking.
#[test]
fn tool_compact_m1_m2_and_overlong_smoke() {
    let Some(dir) = model_dir() else {
        eprintln!("skip tool_compact_m1_m2_and_overlong_smoke: quantized model not present");
        return;
    };
    let (tok, cfg, mut session) = open(&dir);

    // --- Phase A: raw extend past capacity (fresh session → no partial write).
    let capacity = session.extend_capacity();
    assert_eq!(capacity, MAX_SEQ - crate::metal::CANVAS);

    // Raw extend past capacity: typed error, no partial write, no panic.
    let overlong: Vec<u32> = tok
        .encode(&"inventory line item alpha beta ".repeat(900), false)
        .into_iter()
        .take(capacity + 100)
        .collect();
    assert!(overlong.len() > capacity, "fixture must exceed capacity");
    assert!(session.extend_kv(&overlong).is_err());
    assert_eq!(session.kv_valid_tokens().len(), 0);

    // Finalize with the same overlong canonical: Ok, KV holds the longest
    // fitting prefix, routing log keeps the full canonical.

    // --- Phase B: M2 expand re-entry + eviction, on the raw session.
    {
        let output = big_output();
        let messages = base_messages(&output);
        let mut tools = vec![read_file_tool()];
        tools.push(tc::expand_summary_tool());
        let count = |s: &str| tok.encode(s, false).len();
        let hash = tc::fnv1a64(&output);
        let resolve = |h: u64| {
            (h == hash).then(|| ("tr_smoke".to_string(), "a directory listing".to_string()))
        };
        let messages_sub = tc::compact_messages(&messages, THRESHOLD, &count, &resolve);

        let prompt = tok.encode_with_specials(&crate::tools::render_conversation(
            &messages_sub,
            &tools,
            true,
            false,
        ));
        let mut cfg_r0 = cfg.clone();
        cfg_r0.max_new_tokens = 256;
        let out = generate_with_session(&mut session, &prompt, &cfg_r0, "expand-smoke").unwrap();

        // Hand-build the expand round exactly as handle_tool_compact does:
        // model's own token ids + the canonical rendered tool response.
        let excerpt =
            tc::dispatch_expand(&json!({"mode":"grep","pattern":"secret_marker"}), &output);
        assert!(excerpt.contains("zebra42"));
        let resp_text =
            crate::tools::render_tool_response(tc::EXPAND_TOOL_NAME, &json!({"content": excerpt}));
        let mut ext = out.token_ids.clone();
        ext.extend(tok.encode_with_specials(&resp_text));
        assert!(ext.len() + crate::metal::CANVAS < MAX_SEQ);
        let reuse = ext
            .iter()
            .zip(session.kv_valid_tokens())
            .take_while(|(a, b)| a == b)
            .count();
        session.truncate_kv_to(reuse).unwrap();
        session.extend_kv(&ext[reuse..]).unwrap();
        assert_eq!(session.kv_valid_tokens(), &ext[..]);

        // Re-entry: prompt == kv_valid_tokens → no prefill, fresh block denoise.
        let round_prompt = session.kv_valid_tokens().to_vec();
        let mut cfg_r1 = cfg.clone();
        cfg_r1.max_new_tokens =
            256.min(MAX_SEQ.saturating_sub(round_prompt.len() + crate::metal::CANVAS));
        let out2 = generate_with_session(&mut session, &round_prompt, &cfg_r1, "expand-smoke")
            .expect("re-entry after expand extension must succeed");
        assert!(
            out2.token_ids.len() > round_prompt.len(),
            "re-entry must denoise new tokens"
        );

        // Finalize-equivalent: rebuild the canonical (no expand round in it) and
        // verify the excerpt tokens are gone from the causal KV.
        let mut completed = messages_sub.clone();
        completed.push(json!({"role":"assistant","content":"done"}));
        let canonical = tok.encode_with_specials(&crate::tools::render_conversation(
            &completed, &tools, false, false,
        ));
        let reuse = canonical
            .iter()
            .zip(session.kv_valid_tokens())
            .take_while(|(a, b)| a == b)
            .count();
        session.truncate_kv_to(reuse).unwrap();
        session.extend_kv(&canonical[reuse..]).unwrap();
        assert_eq!(session.kv_valid_tokens(), &canonical[..]);
        assert!(
            !tok.decode(session.kv_valid_tokens()).contains("zebra42"),
            "expand excerpt must be evicted from the canonical KV"
        );
    }

    // --- Phase C: M1 summarize-rewind + substituted finalize (manager owns
    // the session from here; activate() resets the KV for the new conv).
    let conv_dir = std::env::temp_dir().join(format!("dgq-compact-smoke-{}", std::process::id()));
    let mut manager = crate::conversation::ConversationManager::new(session, 0, 0, conv_dir);

    let output = big_output();
    let messages = base_messages(&output);
    let mut tools = vec![read_file_tool()];
    tools.push(tc::expand_summary_tool());
    let count = |s: &str| tok.encode(s, false).len();
    assert!(
        count(&output) > THRESHOLD,
        "fixture must exceed the threshold"
    );

    // Route + prefill the (still verbose) routing prompt, as the server does.
    let routing = tok.encode_with_specials(&crate::tools::render_conversation(
        &messages, &tools, true, false,
    ));
    let conv_id = manager.activate(&routing);
    manager.session_mut().extend_kv(&routing).unwrap();
    let before = manager.session_mut().kv_valid_tokens().to_vec();
    assert!(!before.is_empty());

    // Summarize pass: generates, then must roll the KV back exactly.
    let mut ctx = messages.clone();
    ctx.push(json!({"role":"user","content": tc::summarize_instruction()}));
    let stop = crate::config::load_generation_stop_tokens(&dir);
    let summary_opt = {
        let stage = SessionStage(std::cell::RefCell::new(&mut manager));
        super::run_summarize_pass(
            &stage, &tok, &cfg, 24, &stop, &dir, MAX_SEQ, 256, &ctx, &tools,
        )
    };
    assert_eq!(
        manager.session_mut().kv_valid_tokens(),
        &before[..],
        "summarize pass must leave the KV exactly as it found it"
    );
    // Extracted or mechanical — either way compaction proceeds.
    let summary = summary_opt.unwrap_or_else(|| tc::mechanical_summary(&output, 1024));
    assert!(!summary.is_empty());

    // Substituted main generation (delta-prefills from the divergence point).
    let hash = tc::fnv1a64(&output);
    let resolve = |h: u64| (h == hash).then(|| ("tr_smoke".to_string(), summary.clone()));
    let messages_sub = tc::compact_messages(&messages, THRESHOLD, &count, &resolve);
    assert_ne!(messages_sub[2]["content"], messages[2]["content"]);
    let prompt = tok.encode_with_specials(&crate::tools::render_conversation(
        &messages_sub,
        &tools,
        true,
        false,
    ));
    let mut cfg_main = cfg.clone();
    cfg_main.max_new_tokens = 256.min(MAX_SEQ.saturating_sub(prompt.len() + crate::metal::CANVAS));
    let out = generate_with_session(manager.session_mut(), &prompt, &cfg_main, "compact-smoke")
        .expect("substituted generation must succeed");
    let reply = tok.decode(&crate::sample::strip_degenerate_token_ids(
        out.token_ids.get(prompt.len()..).unwrap_or(&[]),
    ));
    let content = crate::tools::content_before_tool_calls(
        &crate::chat_template::sanitize_model_reply(&reply),
    );

    // Finalize with the substituted canonical: the resident KV becomes
    // exactly the compacted completed-turns log (verbose text evicted).
    let mut completed = messages_sub.clone();
    completed.push(json!({"role":"assistant","content": content}));
    let canonical = tok.encode_with_specials(&crate::tools::render_conversation(
        &completed, &tools, false, false,
    ));
    manager.finalize(conv_id, &canonical).unwrap();
    assert_eq!(manager.session_mut().kv_valid_tokens(), &canonical[..]);
    // The canonical log must not contain the verbose response (it holds the
    // substituted summary object instead) — the whole point of compaction.
    let canonical_text = tok.decode(&canonical);
    assert!(
        !canonical_text.contains("file_042.txt"),
        "verbose output leaked into canonical KV"
    );
    assert!(
        canonical_text.contains("tr_smoke"),
        "substituted id missing from canonical KV"
    );

    // --- Phase D: overlong finalize trims to the longest fitting prefix.
    let conv_id = manager.activate(&overlong);
    manager.finalize(conv_id, &overlong).unwrap();
    assert_eq!(
        manager.session_mut().kv_valid_tokens(),
        &overlong[..capacity]
    );
}
