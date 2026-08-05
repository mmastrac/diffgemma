//! Unit tests for the serve HTTP layer (envelope framing, log writers, request
//! parsing, tool-turn continuation).

use super::*;

#[test]
fn tool_markup_strip_truncates_content_at_opener() {
    use std::sync::atomic::AtomicBool;
    let suppress = AtomicBool::new(false);
    let out = filter_tool_markup_delta(
        WireDelta::Content("I'll check.<|tool_call>call:x{}<tool_call|>".into()),
        true,
        &suppress,
    );
    assert_eq!(out, Some(WireDelta::Content("I'll check.".into())));
    assert!(suppress.load(std::sync::atomic::Ordering::Relaxed));
    assert_eq!(
        filter_tool_markup_delta(WireDelta::Content("trailing".into()), true, &suppress),
        None
    );
}

#[test]
fn tool_markup_strip_noop_when_disabled() {
    use std::sync::atomic::AtomicBool;
    let suppress = AtomicBool::new(false);
    let raw = WireDelta::Content("<|tool_call>call:x{}<tool_call|>".into());
    assert_eq!(
        filter_tool_markup_delta(raw.clone(), false, &suppress),
        Some(raw)
    );
    assert!(!suppress.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn trunc_preview_flattens_and_caps() {
    assert_eq!(trunc_preview("hi\nthere", 120), "hi there");
    let long: String = "x".repeat(200);
    let out = trunc_preview(&long, 120);
    assert_eq!(out.chars().count(), 123); // 120 + "..."
    assert!(out.ends_with("..."));
}

#[test]
fn write_serve_and_model_logs() {
    let dir = std::env::temp_dir().join(format!("dgq-serve-log-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let serve = serde_json::json!({"seq": 7, "finish_reason": "stop"});
    let path = write_serve_log(&dir, 7, &serve).unwrap();
    assert_eq!(path.file_name().unwrap(), "serve-00007.json");
    let body: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(body["seq"], 7);

    let stats = crate::generate::BlockDenoiseStats {
        block_idx: 2,
        max_blocks: 4,
        steps_eff: 7,
        denoise_secs: 1.25,
        accept_per_step: vec![1, 5, 256],
        min_ent_per_step: vec![0.2, 0.01, 0.0],
        mean_ent_per_step: vec![1.0, 0.1, 0.0],
        low_ent_per_step: vec![0, 10, 256],
        late_accept_sum: 256,
        late_min_ent: 0.0,
        late_mean_ent: 0.0,
        late_max_low_ent: 256,
        denoise_stop: "confident".into(),
        kept_tokens: 54,
        token_ids: vec![1, 2],
        stop_token_id: Some(1),
        stop_offset: Some(54),
        continued_past_stop: false,
    };
    // The writer paths under test only exercise filenames, so build the record
    // by hand rather than through a tokenizer.
    let model = serde_json::json!({
        "tokens": [[1, "<eos>"], [2, "hi"]],
        "stop_token": "<eos>",
        "steps_eff": stats.steps_eff,
    });
    let mpath = write_model_block_log(&dir, 7, 2, &model).unwrap();
    assert_eq!(mpath.file_name().unwrap(), "model-00007-00002.json");
    let mbody: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&mpath).unwrap()).unwrap();
    assert_eq!(mbody["stop_token"], "<eos>");
    assert_eq!(mbody["steps_eff"], 7);
    assert_eq!(mbody["tokens"][0][0], 1);
    assert_eq!(mbody["tokens"][0][1], "<eos>");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn serve_progress_pct_rises_as_entropy_falls() {
    let early = serve_progress_pct(10, 2.5, 256);
    let late = serve_progress_pct(60, 0.02, 256);
    assert!(late > early, "early={early} late={late}");
    assert!(late <= 100);
}

#[test]
fn model_think_suffix_parses() {
    assert_eq!(parse_model_think_suffix("diffusiongemma"), None);
    assert_eq!(parse_model_think_suffix("diffusiongemma:think"), Some(true));
    assert_eq!(
        parse_model_think_suffix("diffusiongemma:think=true"),
        Some(true)
    );
    assert_eq!(
        parse_model_think_suffix("diffusiongemma:Think=FALSE"),
        Some(false)
    );
    assert_eq!(parse_model_think_suffix("foo:bar"), None);
    assert_eq!(parse_model_think_suffix("foo:think=maybe"), None);
}

#[test]
fn resolve_thinking_precedence() {
    assert!(!resolve_enable_thinking(
        Some("m:think=false"),
        Some(true),
        None,
        true
    ));
    assert!(resolve_enable_thinking(
        Some("m:think"),
        Some(false),
        None,
        false
    ));
    assert!(!resolve_enable_thinking(Some("m"), Some(false), None, true));
    assert!(resolve_enable_thinking(None, None, Some(true), false));
    assert!(resolve_enable_thinking(None, None, None, true));
    assert!(!resolve_enable_thinking(None, None, None, false));
}

fn pending_fixture() -> PendingToolTurn {
    let messages = vec![serde_json::json!({"role": "user", "content": "list the files"})];
    let tool_calls = vec![
        serde_json::json!({"index": 0, "id": "call_0", "type": "function",
            "function": {"name": "ls", "arguments": "{\"path\":\".\"}"}}),
        serde_json::json!({"index": 1, "id": "call_1", "type": "function",
            "function": {"name": "read", "arguments": "{\"path\":\"a\"}"}}),
    ];
    PendingToolTurn {
        messages,
        tool_calls,
        raw_log: vec![1, 2, 3],
        tools: vec![serde_json::json!({"type": "function", "function": {"name": "ls"}})],
        thinking: true,
    }
}

fn continuation_messages(p: &PendingToolTurn) -> Vec<serde_json::Value> {
    let mut m = p.messages.clone();
    m.push(serde_json::json!({
        "role": "assistant", "content": "",
        "tool_calls": p.tool_calls.clone(),
    }));
    m.push(serde_json::json!({
        "role": "tool", "tool_call_id": "call_1", "name": "read", "content": "aaa"
    }));
    m.push(serde_json::json!({
        "role": "tool", "tool_call_id": "call_0", "name": "ls", "content": "a b"
    }));
    m
}

#[test]
fn continuation_matches_and_orders_responses_by_call() {
    let p = pending_fixture();
    let msgs = continuation_messages(&p);
    // Responses arrive out of order, and the match returns them in our call order.
    let got = match_tool_continuation(&p, &msgs, &p.tools.clone(), true).unwrap();
    assert_eq!(
        got,
        vec![
            ("ls".to_string(), "a b".to_string()),
            ("read".to_string(), "aaa".to_string()),
        ]
    );
}

#[test]
fn continuation_rejects_divergence() {
    let p = pending_fixture();
    let msgs = continuation_messages(&p);
    // Thinking flip, tool-set change, edited history, missing/extra responses:
    // all fall back to the re-render path.
    assert!(match_tool_continuation(&p, &msgs, &p.tools.clone(), false).is_none());
    assert!(match_tool_continuation(&p, &msgs, &[], true).is_none());
    let mut edited = msgs.clone();
    edited[0] = serde_json::json!({"role": "user", "content": "different"});
    assert!(match_tool_continuation(&p, &edited, &p.tools.clone(), true).is_none());
    let missing = &msgs[..msgs.len() - 1];
    assert!(match_tool_continuation(&p, missing, &p.tools.clone(), true).is_none());
    let mut extra = msgs.clone();
    extra.push(serde_json::json!({"role": "user", "content": "and now this"}));
    assert!(match_tool_continuation(&p, &extra, &p.tools.clone(), true).is_none());
    // A response answering an unknown call id.
    let mut wrong = msgs.clone();
    wrong[2]["tool_call_id"] = serde_json::json!("call_9");
    assert!(match_tool_continuation(&p, &wrong, &p.tools.clone(), true).is_none());
}

#[test]
fn continuation_matches_by_position_without_ids() {
    let p = pending_fixture();
    let mut msgs = p.messages.clone();
    msgs.push(serde_json::json!({
        "role": "assistant", "content": "",
        "tool_calls": p.tool_calls.clone(),
    }));
    msgs.push(serde_json::json!({"role": "tool", "name": "ls", "content": "a b"}));
    msgs.push(serde_json::json!({"role": "tool", "name": "read", "content": "aaa"}));
    let got = match_tool_continuation(&p, &msgs, &p.tools.clone(), true).unwrap();
    assert_eq!(got[0].0, "ls");
    assert_eq!(got[1].1, "aaa");
}
