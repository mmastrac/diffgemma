//! Tests for `tool_smoke`, extracted from server.rs (backlog item 3).

use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
use crate::tools::ParsedToolCall;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn model_dir() -> Option<PathBuf> {
    for p in ["model/diffgemma-26b-a4b-it-q4", "/tmp/quantized-weights"] {
        let d = PathBuf::from(p);
        if d.join("model.dgq.json").exists() {
            return Some(d);
        }
    }
    None
}

const MAX_SEQ: usize = 2048;

/// One session shared by every scenario in this module: the model load
/// dominated these tests as separate #[test] fns, so they now run as
/// phases of a single test against one session.
fn open_rig(
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

/// One tool turn, driven exactly like the server worker: render the messages
/// + tools in the model's canonical format, tokenize the specials, generate
/// (stop at eos), then parse the calls back. Returns (raw reply, calls).
/// `generate_with_session` resets the KV when the prompt doesn't extend it.
fn tool_turn(
    tok: &crate::tokenizer::Tokenizer,
    cfg: &StepGenerateConfig,
    session: &mut StepGenerateSession,
    messages: &[Value],
    tools: &[Value],
) -> (String, Vec<ParsedToolCall>) {
    let prompt_str = crate::tools::render_conversation(messages, tools, true, false);
    let prompt = tok.encode_with_specials(&prompt_str);
    let mut cfg = cfg.clone();
    cfg.max_new_tokens = 512.min(MAX_SEQ.saturating_sub(prompt.len()).max(1));
    let out = generate_with_session(session, &prompt, &cfg, "tool-smoke").unwrap();
    let new_ids =
        crate::sample::strip_degenerate_token_ids(out.token_ids.get(prompt.len()..).unwrap_or(&[]));
    let text = tok.decode(&new_ids);
    let calls = crate::tools::parse_tool_calls(&text);
    (text, calls)
}

fn list_dir_tool() -> Value {
    json!({"type":"function","function":{
            "name":"list_dir",
            "description":"Lists all files and subdirectories in a specified directory path",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"The absolute path to the directory to list (e.g., /tmp)"}},
                "required":["path"]}}})
}

fn search_tool() -> Value {
    json!({"type":"function","function":{
            "name":"search",
            "description":"Search the web",
            "parameters":{"type":"object","properties":{
                "query":{"type":"string","description":"the query"},
                "count":{"type":"integer","description":"number of results"},
                "safe":{"type":"boolean","description":"safe search"},
                "sort":{"type":"string","description":"sort order","enum":["relevance","date"]}},
                "required":["query"]}}})
}

/// Both tool-grammar scenarios as phases of one test (single model load):
/// a well-formed call with a string arg, then schema-typed argument kinds.
#[test]
fn tool_smoke_call_shape_and_argument_types() {
    let Some(dir) = model_dir() else {
        eprintln!("skip tool_smoke_call_shape_and_argument_types: quantized model not present");
        return;
    };
    let (tok, cfg, mut session) = open_rig(&dir);

    // --- Phase 1: a well-formed call with the expected name + path arg.
    let msgs = [json!({"role":"user","content":"List the files in /tmp using the tool."})];
    let (text, calls) = tool_turn(&tok, &cfg, &mut session, &msgs, &[list_dir_tool()]);
    assert!(
        !calls.is_empty(),
        "expected a tool call, got reply: {text:?}"
    );
    assert_eq!(calls[0].name, "list_dir", "reply: {text:?}");
    let path = calls[0].arguments.get("path").and_then(Value::as_str);
    assert!(
        path.is_some_and(|p| p.contains("tmp")),
        "path arg should reference /tmp, got {:?} (reply: {text:?})",
        calls[0].arguments
    );

    // --- Phase 2: schema-typed arguments (integer/boolean/enum, not strings).
    let msgs = [json!({"role":"user","content":
            "Search for \"rust async\" with 5 results, safe search on, sorted by date. Use the tool."})];
    let (text, calls) = tool_turn(&tok, &cfg, &mut session, &msgs, &[search_tool()]);
    assert!(
        !calls.is_empty(),
        "expected a tool call, got reply: {text:?}"
    );
    let a = &calls[0].arguments;
    assert_eq!(calls[0].name, "search", "reply: {text:?}");
    // The stable invariant (per the degradation experiment): the model emits
    // schema-typed args — integer/boolean/valid-enum, not stringified.
    assert!(
        a.get("count").is_some_and(Value::is_number),
        "count should parse as a number, got {a:?} (reply: {text:?})"
    );
    assert!(
        a.get("safe").is_some_and(Value::is_boolean),
        "safe should parse as a bool, got {a:?} (reply: {text:?})"
    );
    if let Some(sort) = a.get("sort").and_then(Value::as_str) {
        assert!(
            ["relevance", "date"].contains(&sort),
            "sort should be a declared enum member, got {sort:?} (reply: {text:?})"
        );
    }
}
