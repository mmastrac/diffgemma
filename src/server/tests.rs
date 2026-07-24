//! Tests for `tests`, extracted from server.rs.

use super::*;
use crate::metal::StepProgressEvent;

/// Trivial decoder: printable-ASCII ids map to their char; others (special
/// tokens) decode to nothing. Enough to exercise the id-level split logic.
struct FakeDecoder;
impl TextDecoder for FakeDecoder {
    fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        self.decode_append(&mut s, ids);
        s
    }
    fn decode_append(&self, out: &mut String, ids: &[u32]) {
        for &id in ids {
            if (32..127).contains(&id) {
                out.push(id as u8 as char);
            }
        }
    }
}

fn step<'a>(block: usize, step: u32, argmax: &'a [u32], done: bool) -> StepProgressEvent<'a> {
    StepProgressEvent {
        block_idx: block,
        max_blocks: 4,
        step_in_block: step,
        max_steps: 48,
        argmax,
        accept_count: 0,
        mean_entropy: 0.0,
        block_done: done,
    }
}

fn c(ch: char) -> u32 {
    ch as u32
}

#[test]
fn content_only_commits_answer_text() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, None, false, false, false, false);
    let hi = [c('H'), c('i')];
    // Not committed yet: no content deltas.
    assert!(m.on_step(&step(1, 1, &hi, false)).is_empty());
    // Commit the block -> content delta "Hi".
    let out = m.on_step(&step(1, 2, &hi, true));
    assert_eq!(out, vec![WireDelta::Content("Hi".to_string())]);
    assert_eq!(m.content(), "Hi");
}

#[test]
fn draft_streams_stable_prefix_before_commit() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, None, false, false, true, false);
    let hi = [c('H'), c('i')];
    // Streak must reach STABLE_STREAK before the draft shows anything.
    assert!(draft_of(&m.on_step(&step(1, 1, &hi, false))).is_none());
    let _ = m.on_step(&step(1, 2, &hi, false));
    let d = draft_of(&m.on_step(&step(1, 3, &hi, false)));
    assert_eq!(d.as_deref(), Some("Hi"));
}

#[test]
fn thinking_splits_reasoning_from_content() {
    // channel_open=1, channel_close=2. Canvas: <open> t h o u g h t <close> H i
    let open = 1u32;
    let close = 2u32;
    let mut m = DiffusionStreamMapper::new(
        FakeDecoder,
        vec![],
        Some(open),
        Some(close),
        None,
        false,
        true,
        false,
        false,
    );
    let canvas = [
        open,
        c('t'),
        c('h'),
        c('o'),
        c('u'),
        c('g'),
        c('h'),
        c('t'),
        close,
        c('H'),
        c('i'),
    ];
    let out = m.on_step(&step(1, 2, &canvas, true));
    // "thought" opener scaffold stripped from reasoning; answer is "Hi".
    assert!(out.contains(&WireDelta::Content("Hi".to_string())));
    assert_eq!(m.content(), "Hi");
    assert!(m.reasoning().is_empty() || !m.reasoning().contains("Hi"));
}

#[test]
fn thinking_drops_trailing_channel_close_from_content() {
    // Real log shape: <open> … <close> Answer <close> (extra close before tools).
    let open = 1u32;
    let close = 2u32;
    let mut m = DiffusionStreamMapper::new(
        FakeDecoder,
        vec![],
        Some(open),
        Some(close),
        None,
        false,
        true,
        false,
        false,
    );
    let canvas = [
        open,
        c('x'),
        close,
        c('O'),
        c('k'),
        close, // must not leak into content
        c('!'),
    ];
    let _ = m.on_step(&step(1, 2, &canvas, true));
    assert_eq!(m.content(), "Ok!");
}

/// Decoder that surfaces channel specials as their literal markup (like the
/// real SentencePiece specials), so mid-thought re-opens stay detectable.
struct ChannelDecoder;
impl TextDecoder for ChannelDecoder {
    fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        self.decode_append(&mut s, ids);
        s
    }
    fn decode_append(&self, out: &mut String, ids: &[u32]) {
        for &id in ids {
            match id {
                1 => out.push_str("<|channel>"),
                2 => out.push_str("<channel|>"),
                3 => out.push_str("thought"),
                4 => out.push_str("<|\"|>"),
                10 => out.push('\n'),
                id if (32..127).contains(&id) => out.push(id as u8 as char),
                _ => {}
            }
        }
    }
}

#[test]
fn thinking_strips_nested_channel_reopen_from_reasoning() {
    // OpenCode nests a Thought UI when `<|channel>thought` appears mid-reasoning.
    let open = 1u32;
    let close = 2u32;
    let thought = 3u32;
    let mut m = DiffusionStreamMapper::new(
        ChannelDecoder,
        vec![],
        Some(open),
        Some(close),
        None,
        false,
        true,
        false,
        false,
    );
    // <open> thought \n Hello <open> thought \n world <close>
    let canvas = [
        open,
        thought,
        c('\n'),
        c('H'),
        c('i'),
        open,
        thought,
        c('\n'),
        c('!'),
        close,
        c('A'),
    ];
    let _ = m.on_step(&step(1, 2, &canvas, true));
    assert!(
        !m.reasoning().contains("<|channel>"),
        "nested open leaked: {:?}",
        m.reasoning()
    );
    assert!(
        !m.reasoning().contains("<channel|>"),
        "close leaked: {:?}",
        m.reasoning()
    );
    // The re-open's NAME must go with it — a bare "thought" line otherwise
    // leaks into the client's Thought UI.
    assert!(
        !m.reasoning().contains("thought"),
        "channel name leaked: {:?}",
        m.reasoning()
    );
    assert_eq!(m.content(), "A");
}

/// Field regression: thinking mode ON but the model skips the thought
/// ceremony and answers with a bare tool call —
/// no channel markers anywhere in the reply. The old "everything is
/// reasoning until a close appears" rule streamed the whole call as
/// reasoning_content and the client got an empty message. Classification
/// must follow emission: no thought span ⇒ all content.
#[test]
fn thinking_bare_reply_without_thought_span_is_content() {
    let open = 1u32;
    let close = 2u32;
    let mut m = DiffusionStreamMapper::new(
        FakeDecoder,
        vec![],
        Some(open),
        Some(close),
        None,
        false,
        true,
        false,
        false,
    );
    let canvas: Vec<u32> = "call:edit{x:1}".chars().map(|ch| ch as u32).collect();
    let _ = m.on_step(&step(1, 2, &canvas, true));
    assert_eq!(m.content(), "call:edit{x:1}");
    assert_eq!(m.reasoning(), "");
}

/// Tokens before the first explicit thought open are content (turn-opener
/// ceremony etc.), not retroactively reasoning.
#[test]
fn thinking_pre_open_tokens_stay_content() {
    let open = 1u32;
    let close = 2u32;
    let mut m = DiffusionStreamMapper::new(
        FakeDecoder,
        vec![],
        Some(open),
        Some(close),
        None,
        false,
        true,
        false,
        false,
    );
    let canvas = [c('A'), open, c('z'), close, c('B')];
    let _ = m.on_step(&step(1, 2, &canvas, true));
    assert_eq!(m.content(), "AB");
    assert!(
        m.reasoning().contains('z'),
        "thought span lost: {:?}",
        m.reasoning()
    );
}

/// A channel-open id inside a `<|"|>` quote run is literal tool-arg content:
/// it must stay in content verbatim (not toggle the thought state, not get
/// filtered) so the parsed call's args round-trip byte-exact.
#[test]
fn quoted_channel_id_stays_in_content_verbatim() {
    let (open, close, quote) = (1u32, 2u32, 4u32);
    let mut m = DiffusionStreamMapper::new(
        ChannelDecoder,
        vec![],
        Some(open),
        Some(close),
        Some(quote),
        false,
        true,
        false,
        false,
    );
    // call:e{x:<Q> <open> <Q>}   — the quoted open is data.
    let mut canvas: Vec<u32> = "call:e{x:".chars().map(|ch| ch as u32).collect();
    canvas.extend([quote, open, quote]);
    canvas.push(c('}'));
    let _ = m.on_step(&step(1, 2, &canvas, true));
    assert_eq!(m.content(), "call:e{x:<|\"|><|channel><|\"|>}");
    assert_eq!(m.reasoning(), "");
    let calls = crate::tools::parse_tool_calls(m.content());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments["x"].as_str().unwrap(), "<|channel>");
}

/// With stop-skip enabled (mirroring the engine's continue-past-stop
/// policy), a stop id inside an open quote run is content, not a stop —
/// including when the quote opened in an earlier committed block.
#[test]
fn quoted_stop_id_streams_through_when_skip_enabled() {
    let quote = 4u32;
    let mk = |skip: bool| {
        DiffusionStreamMapper::new(
            ChannelDecoder,
            vec![99],
            None,
            None,
            Some(quote),
            skip,
            false,
            false,
            false,
        )
    };
    // Block 1 leaves the quote OPEN; block 2 carries the stop id inside it.
    let block1: Vec<u32> = vec![c('a'), quote, c('b')];
    let block2: Vec<u32> = vec![c('c'), 99, quote, c('d')];
    let mut m = mk(true);
    let _ = m.on_step(&step(1, 2, &block1, true));
    let _ = m.on_step(&step(2, 2, &block2, true));
    assert!(!m.ended(), "quoted stop must not end the stream");
    assert!(m.content().contains('d'), "content: {:?}", m.content());
    // Same stream without the skip: the stop cuts inside the quote.
    let mut m = mk(false);
    let _ = m.on_step(&step(1, 2, &block1, true));
    let _ = m.on_step(&step(2, 2, &block2, true));
    assert!(m.ended());
    assert!(!m.content().contains('d'));
}

#[test]
fn stop_token_ends_and_cuts() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![99], None, None, None, false, false, false, false);
    let canvas = [c('O'), c('k'), 99, c('X')];
    let out = m.on_step(&step(1, 2, &canvas, true));
    assert_eq!(out, vec![WireDelta::Content("Ok".to_string())]);
    assert!(m.ended());
    // Further steps are inert.
    assert!(m.on_step(&step(1, 3, &canvas, true)).is_empty());
}

#[test]
fn append_delta_semantics() {
    assert_eq!(append_delta("", "Hi"), Some("Hi".to_string()));
    assert_eq!(append_delta("Hi", "Hi there"), Some(" there".to_string()));
    assert_eq!(append_delta("Hi", "Hi"), None);
    assert_eq!(append_delta("Hello", "Help"), Some("Help".to_string()));
}

fn draft_of(deltas: &[WireDelta]) -> Option<String> {
    deltas.iter().find_map(|d| match d {
        WireDelta::Draft { text, .. } => Some(text.clone()),
        _ => None,
    })
}

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
    // Tokenizer not needed for field presence in unit test of writer paths that
    // only exercise filenames — build JSON with a stub via empty id map is hard.
    // Instead check the tokens array shape through a hand-built record.
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

#[test]
fn paced_stream_holds_commit_and_releases_during_next_block() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, None, false, false, false, true);
    let text: Vec<u32> = "abcdefghij".chars().map(|ch| ch as u32).collect();
    // Block 1 commits after 16 steps (seeds the EMA at (16+16)/2 = 16):
    // paced mode holds the text — no Content delta at commit.
    let out = m.on_step(&step(1, 16, &text, true));
    assert!(
        out.iter().all(|d| !matches!(d, WireDelta::Content(_))),
        "paced commit must hold content: {out:?}"
    );
    // content() still reports the full committed text (final-response path).
    assert_eq!(m.content(), "abcdefghij");
    // Block 2 progresses: half the estimated steps releases ~half the text.
    let tail = [c('z')];
    let mut streamed = String::new();
    for s in 1..=8 {
        for d in m.on_step(&step(2, s, &tail, false)) {
            if let WireDelta::Content(t) = d {
                streamed.push_str(&t);
            }
        }
    }
    assert!(
        !streamed.is_empty() && streamed.len() < 10,
        "partial release expected, got {streamed:?}"
    );
    assert!("abcdefghij".starts_with(&streamed));
    // Block 2 commits with a stop -> turn ends -> everything flushes.
    let stop = [c('z'), 999];
    let mut m2 = DiffusionStreamMapper::new(FakeDecoder, vec![999], None, None, None, false, false, false, true);
    let _ = m2.on_step(&step(1, 16, &text, true));
    let out = m2.on_step(&step(2, 4, &stop, true));
    let content: String = out
        .iter()
        .filter_map(|d| match d {
            WireDelta::Content(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(content, "abcdefghijz", "turn end must flush all held text");
}

#[test]
fn paced_stream_final_flush_releases_held_text() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, None, false, false, false, true);
    let text: Vec<u32> = "hold me".chars().map(|ch| ch as u32).collect();
    let _ = m.on_step(&step(1, 16, &text, true));
    // Turn ends without a stop token (budget): final_flush drains the rest.
    let out = m.final_flush();
    let content: String = out
        .iter()
        .filter_map(|d| match d {
            WireDelta::Content(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(content, "hold me");
    assert!(m.final_flush().is_empty(), "second flush is a no-op");
}

#[test]
fn paced_stream_tool_call_disables_pacing() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, None, false, false, false, true);
    let text: Vec<u32> = "call:read{".chars().map(|ch| ch as u32).collect();
    // A committed block containing a tool call flushes immediately.
    let out = m.on_step(&step(1, 8, &text, true));
    let content: String = out
        .iter()
        .filter_map(|d| match d {
            WireDelta::Content(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(content, "call:read{");
    // ...and pacing stays off: the next block's commit also emits directly.
    let more: Vec<u32> = "}".chars().map(|ch| ch as u32).collect();
    let out = m.on_step(&step(2, 4, &more, true));
    assert!(
        out.contains(&WireDelta::Content("}".to_string())),
        "pace_off must persist: {out:?}"
    );
}
