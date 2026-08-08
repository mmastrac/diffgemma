use super::testutil::{FakeDecoder, step};
use super::*;
use crate::chat::ChatEvent;

fn texts(events: &[ChatEvent]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn thoughts(events: &[ChatEvent]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::Thought { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn stable_prefix_requires_two_sightings() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![]);
    let hi = [b'H' as u32, b'i' as u32];
    // Step 1: fresh, streak 0, nothing stable yet.
    let e1 = d.on_step(&step(1, 1, &hi, false));
    assert_eq!(texts(&e1), Vec::<&str>::new());
    // Step 2: streak 1, still not stable.
    let e2 = d.on_step(&step(1, 2, &hi, false));
    assert_eq!(texts(&e2), Vec::<&str>::new());
    // Step 3: streak 2, both positions stream.
    let e3 = d.on_step(&step(1, 3, &hi, false));
    assert_eq!(texts(&e3), vec!["Hi"]);
}

#[test]
fn prethink_streams_seed_then_thought() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![]).with_thinking(false, Some("SEED ".into()));
    let hi = [b'H' as u32, b'i' as u32];
    // The reasoning surfaces as `Thought`, never as `Text`. The injected seed
    // shows immediately, before any generated thought stabilizes.
    let e1 = d.on_step(&step(1, 1, &hi, false));
    assert_eq!(thoughts(&e1), vec!["SEED "]);
    assert_eq!(texts(&e1), Vec::<&str>::new());
    assert_eq!(
        thoughts(&d.on_step(&step(1, 2, &hi, false))),
        Vec::<&str>::new()
    );
    // Then the thought streams in, still prefixed by the seed.
    assert_eq!(
        thoughts(&d.on_step(&step(1, 3, &hi, false))),
        vec!["SEED Hi"]
    );
}

/// Sentinel channel ids for exercising the token-level split. 200 opens the
/// thought channel and 201 closes it. The decoder renders neither, since the
/// walk consumes them as toggles like the real specials.
const OPEN: u32 = 200;
const CLOSE: u32 = 201;

fn channel_ids() -> ChannelIds {
    ChannelIds {
        open: Some(OPEN),
        close: Some(CLOSE),
        quote: None,
    }
}

#[test]
fn split_consumes_open_marker_and_role_name() {
    // FakeDecoder renders "thought" and "\n" name tokens as chars, so use
    // explicit ids here to exercise the name-skip logic.
    struct NameDecoder;
    impl TextDecoder for NameDecoder {
        fn decode(&self, ids: &[u32]) -> String {
            let mut s = String::new();
            self.decode_append(&mut s, ids);
            s
        }
        fn decode_append(&self, out: &mut String, ids: &[u32]) {
            for &id in ids {
                match id {
                    300 => out.push_str("thought"),
                    301 => out.push('\n'),
                    32..=127 => out.push(id as u8 as char),
                    _ => {}
                }
            }
        }
    }
    let ids = [OPEN, 300, 301, b'H' as u32, b'i' as u32, CLOSE, b'O' as u32];
    let split = channel_ids().split(&NameDecoder, &ids, false);
    assert_eq!(split.reasoning, vec![b'H' as u32, b'i' as u32]);
    assert_eq!(split.content, vec![b'O' as u32]);
    // A prethink continuation starts in-thought with no marker at all.
    let split = channel_ids().split(&NameDecoder, &[b'H' as u32, CLOSE, b'O' as u32], true);
    assert_eq!(split.reasoning, vec![b'H' as u32]);
    assert_eq!(split.content, vec![b'O' as u32]);
}

#[test]
fn split_keeps_quoted_channel_markup_as_data() {
    const QUOTE: u32 = 202;
    let ids = ChannelIds {
        open: Some(OPEN),
        close: Some(CLOSE),
        quote: Some(QUOTE),
    };
    // Inside an open quote run, a `<channel|>` id is literal arg content. It
    // must not close the (never-opened) thought or vanish from content.
    let stream = [b'A' as u32, QUOTE, CLOSE, OPEN, QUOTE, b'B' as u32];
    let split = ids.split(&FakeDecoder, &stream, false);
    assert_eq!(
        split.content,
        vec![b'A' as u32, QUOTE, CLOSE, OPEN, QUOTE, b'B' as u32]
    );
    assert!(split.reasoning.is_empty());
}

#[test]
fn show_thinking_splits_reasoning_from_answer() {
    // Normal turn (no seed) with show_thinking: the model opens its own
    // thought, reasons, closes, then answers.
    let mut d = StreamDecoder::new(FakeDecoder, vec![])
        .with_channels(channel_ids())
        .with_thinking(true, None);
    // Reasoning canvas: <|channel> 'H' 'i', stabilizes by step 3.
    let reasoning = [OPEN, b'H' as u32, b'i' as u32];
    let _ = d.on_step(&step(1, 1, &reasoning, false));
    let _ = d.on_step(&step(1, 2, &reasoning, false));
    let e = d.on_step(&step(1, 3, &reasoning, false));
    // The reasoning surfaced as a `Thought` event, never as answer `Text`.
    assert_eq!(thoughts(&e), vec!["Hi"]);
    assert_eq!(texts(&e), Vec::<&str>::new());
    // Now the thought closes and the answer appears (committed on block_done).
    let full = [
        OPEN,
        b'H' as u32,
        b'i' as u32,
        CLOSE,
        b'O' as u32,
        b'k' as u32,
    ];
    let done = d.on_step(&step(1, 4, &full, true));
    assert_eq!(texts(&done), vec!["Ok"]);
}

/// A text-form `<|channel>` marker a differently-BPE'd scaffold leaves in the
/// reasoning ids is scrubbed from the displayed thought (the token-level split
/// only drops the exact `thought` token, so `decode_reasoning` catches the rest).
#[test]
fn thinking_display_scrubs_leaked_channel_marker() {
    struct LeakyDecoder;
    impl TextDecoder for LeakyDecoder {
        fn decode(&self, ids: &[u32]) -> String {
            let mut s = String::new();
            self.decode_append(&mut s, ids);
            s
        }
        fn decode_append(&self, out: &mut String, ids: &[u32]) {
            for &id in ids {
                match id {
                    400 => out.push_str("<|channel>"),
                    32..=127 => out.push(id as u8 as char),
                    _ => {}
                }
            }
        }
    }
    let mut d = StreamDecoder::new(LeakyDecoder, vec![])
        .with_channels(channel_ids())
        .with_thinking(true, None);
    // <open> <leaked-marker> H i <close> O k
    let full = [
        OPEN,
        400,
        b'H' as u32,
        b'i' as u32,
        CLOSE,
        b'O' as u32,
        b'k' as u32,
    ];
    let done = d.on_step(&step(1, 1, &full, true));
    assert_eq!(thoughts(&done), vec!["Hi"]);
    assert_eq!(texts(&done), vec!["Ok"]);
}

/// Regression: a multi-block answer's committed prefix must grow monotonically.
/// A decrease reset the renderer's print cursor and re-emitted the reply.
#[test]
fn thinking_answer_committed_is_monotonic_across_blocks() {
    let pairs = |events: &[ChatEvent]| -> Vec<(usize, String)> {
        events
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Text { committed, text } => Some((*committed, text.clone())),
                _ => None,
            })
            .collect()
    };
    let mut d = StreamDecoder::new(FakeDecoder, vec![])
        .with_channels(channel_ids())
        .with_thinking(true, None);
    // Block 1 completes: thought "Hi" closes, answer "Ab" is committed (len 2).
    let b1 = [
        OPEN,
        b'H' as u32,
        b'i' as u32,
        CLOSE,
        b'A' as u32,
        b'b' as u32,
    ];
    assert_eq!(
        pairs(&d.on_step(&step(1, 1, &b1, true))),
        vec![(2, "Ab".into())]
    );
    // Block 2 extends the answer with a still-speculative "cd". The committed
    // prefix stays at the finalized length 2, never dropping to 0.
    let b2 = [b'c' as u32, b'd' as u32];
    let _ = d.on_step(&step(2, 1, &b2, false));
    let _ = d.on_step(&step(2, 2, &b2, false));
    assert_eq!(
        pairs(&d.on_step(&step(2, 3, &b2, false))),
        vec![(2, "Abcd".into())]
    );
}

#[test]
fn stop_token_cuts_the_visible_prefix() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![99]);
    let canvas = [b'O' as u32, b'k' as u32, 99, b'X' as u32];
    for s in 1..=3 {
        let _ = d.on_step(&step(1, s, &canvas, false));
    }
    let e = d.on_step(&step(1, 4, &canvas, false));
    // Everything after the stop token (99) is dropped, even though "X" is also
    // stable.
    assert_eq!(texts(&e), Vec::<&str>::new()); // unchanged: step 3 already showed "Ok"
    assert_eq!(d.committed_text(), "");
    let e3 = {
        let mut d2 = StreamDecoder::new(FakeDecoder, vec![99]);
        for s in 1..=2 {
            let _ = d2.on_step(&step(1, s, &canvas, false));
        }
        d2.on_step(&step(1, 3, &canvas, false))
    };
    assert_eq!(texts(&e3), vec!["Ok"]);
}

#[test]
fn block_done_commits_full_canvas_and_ends_on_stop() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![99]);
    let canvas = [b'H' as u32, b'i' as u32, 99];
    let e = d.on_step(&step(1, 5, &canvas, true));
    assert_eq!(d.committed_text(), "Hi");
    assert!(
        e.iter()
            .any(|ev| matches!(ev, ChatEvent::BlockCommit { .. }))
    );
    // Ended: a further step yields nothing.
    assert!(d.on_step(&step(1, 6, &canvas, true)).is_empty());
}

#[test]
fn rewind_emitted_when_a_stable_position_revises() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![]);
    let cat = [b'C' as u32, b'a' as u32, b't' as u32];
    for s in 1..=3 {
        let _ = d.on_step(&step(1, s, &cat, false));
    }
    // Now the middle position flips: the stable "Cat" draft rewinds to "C" on
    // the flip step, then re-stabilizes to "Cot".
    let cot = [b'C' as u32, b'o' as u32, b't' as u32];
    let mut all = Vec::new();
    all.extend(d.on_step(&step(1, 4, &cot, false)));
    all.extend(d.on_step(&step(1, 5, &cot, false)));
    let last = d.on_step(&step(1, 6, &cot, false));
    all.extend(last.clone());
    assert!(
        all.iter()
            .any(|ev| matches!(ev, ChatEvent::Rewind { common: 1 }))
    );
    assert_eq!(texts(&last), vec!["Cot"]);
}

#[test]
fn salvage_answer_takes_the_trailing_block() {
    // The model reasoned, then stated the answer after a blank line.
    assert_eq!(
        salvage_answer("The project 'Bluebird' ships in March.\n\nMarch."),
        "March."
    );
    // No separator: the whole completion is the recovered answer.
    assert_eq!(
        salvage_answer("just reasoning, ending in 56"),
        "just reasoning, ending in 56"
    );
    // Empty or whitespace: nothing to salvage.
    assert_eq!(salvage_answer("  \n\n  "), "");
}

#[test]
fn settle_prethink_splits_at_close_and_salvages_without_one() {
    let ids = channel_ids();
    // Closed: seed plus completion before the close, then the answer after.
    let closed = [b'S' as u32, b'1' as u32, CLOSE, b'4' as u32, b'2' as u32];
    assert_eq!(
        ids.settle_prethink(&FakeDecoder, "Think. ", &closed),
        ("Think. S1".to_string(), "42".to_string())
    );
    // Never closed: all thought, so the answer is salvaged from its tail (here
    // with no blank-line separator, the whole completion).
    let open = [b'S' as u32, b'1' as u32];
    assert_eq!(
        ids.settle_prethink(&FakeDecoder, "seed ", &open),
        ("seed S1".to_string(), "S1".to_string())
    );
}

#[test]
fn settle_tool_reply_keeps_calls_without_a_channel_close() {
    let ids = channel_ids();
    let calls = [b'c' as u32, b'a' as u32, b'l' as u32, b'l' as u32];
    // The round begins inside an open thought (the prompt seeds the marker), and
    // the close separates reasoning from the calls.
    let stream = [
        b'H' as u32,
        b'i' as u32,
        CLOSE,
        calls[0],
        calls[1],
        calls[2],
        calls[3],
    ];
    let (thought, content) = ids.settle_tool_reply(&FakeDecoder, "", &stream);
    assert_eq!(thought.as_deref(), Some("Hi"));
    assert_eq!(content, "call");
    // A seed prefixes the surfaced reasoning.
    let (thought, _) = ids.settle_tool_reply(&FakeDecoder, "Plan: ", &stream);
    assert_eq!(thought.as_deref(), Some("Plan: Hi"));
    // Never closed: the whole text is kept so calls the model left inside its
    // thought still parse, the regression this guards against.
    let (thought, content) = ids.settle_tool_reply(&FakeDecoder, "", &calls);
    assert_eq!(thought, None);
    assert_eq!(content, "call");
}

/// A tool round's forced-open thought splits from the first id even when nothing
/// displays it: no `Thought` events, no reasoning leaking into `Text`, and the
/// answer starts after the model's close.
#[test]
fn start_in_thought_hides_undisplayed_reasoning() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![])
        .with_channels(channel_ids())
        .starting_in_thought();
    let reasoning = [b'H' as u32, b'i' as u32];
    for s in 1..=3 {
        let e = d.on_step(&step(1, s, &reasoning, false));
        assert_eq!(thoughts(&e), Vec::<&str>::new());
        assert_eq!(texts(&e), Vec::<&str>::new());
    }
    // The thought closes and the answer appears, reasoning never shown.
    let full = [b'H' as u32, b'i' as u32, CLOSE, b'O' as u32, b'k' as u32];
    let done = d.on_step(&step(1, 4, &full, true));
    assert_eq!(thoughts(&done), Vec::<&str>::new());
    assert_eq!(texts(&done), vec!["Ok"]);
    assert!(
        done.iter()
            .any(|ev| matches!(ev, ChatEvent::BlockCommit { .. }))
    );
}

#[test]
fn stabilizer_skips_quoted_stops_and_carries_parity() {
    const QUOTE: u32 = 7;
    const STOP: u32 = 99;
    let mut st = Stabilizer::new(vec![STOP], Some(QUOTE), true);
    // Block 1 commits with an open quote run (one quote id).
    let b1 = [b'A' as u32, QUOTE, b'B' as u32];
    let sp = st.on_step(&step(1, 1, &b1, true));
    assert_eq!(sp.ids, b1);
    assert!(!sp.hit_stop);
    // Block 2: still inside the quote, so the stop id is literal content.
    let b2 = [STOP, QUOTE, STOP];
    let sp = st.on_step(&step(2, 1, &b2, true));
    assert!(sp.new_block);
    assert_eq!(sp.ids, vec![STOP, QUOTE]);
    assert!(sp.hit_stop); // the second STOP falls outside the closed quote
}

#[test]
fn first_unquoted_stop_respects_quote_parity() {
    let stops = [99u32];
    // No quote id: plain scan.
    assert_eq!(
        first_unquoted_stop(&[7, 99, 8], &stops, None, false),
        Some(1)
    );
    // Quoted stop skipped, the next unquoted one found.
    assert_eq!(
        first_unquoted_stop(&[4, 99, 4, 99], &stops, Some(4), false),
        Some(3)
    );
    // Starting inside a quote run (odd parity carried in from the reply so far).
    assert_eq!(
        first_unquoted_stop(&[99, 4, 99], &stops, Some(4), true),
        Some(2)
    );
    // Fully quoted block: no stop at all.
    assert_eq!(first_unquoted_stop(&[99, 99], &stops, Some(4), true), None);
}

fn ids(s: &str) -> Vec<u32> {
    s.chars().map(|c| c as u32).collect()
}

#[test]
fn unpaced_reveals_a_committed_block_whole() {
    // The default (pacing off) surfaces a finished block's answer at once.
    let mut d = StreamDecoder::new(FakeDecoder, vec![]);
    let e = d.on_step(&step(1, 10, &ids("Hello"), true));
    assert_eq!(texts(&e), vec!["Hello"]);
}

#[test]
fn paced_holds_a_block_then_dribbles_it_over_the_next() {
    let mut d = StreamDecoder::new(FakeDecoder, vec![]).paced(true);
    // Block 1 commits "Hello", but paced holds it: nothing visible yet. This
    // also seeds the pacing denominator (EMA of 16 and this block's 10 -> 13).
    let e = d.on_step(&step(1, 10, &ids("Hello"), true));
    assert_eq!(texts(&e), Vec::<&str>::new());
    // Block 2's denoise dribbles the held text out, growing monotonically and
    // reaching the whole block by the time this one is as long as the estimate.
    let mid = texts(&d.on_step(&step(2, 5, &ids("!"), false)))
        .last()
        .unwrap()
        .to_string();
    assert!(!mid.is_empty() && mid.len() < 5, "partial reveal: {mid:?}");
    assert!("Hello".starts_with(&mid), "prefix of the block: {mid:?}");
    let full = texts(&d.on_step(&step(2, 13, &ids("!"), false)))
        .last()
        .unwrap()
        .to_string();
    // The still-speculative "!" of block 2 is not shown; only block 1's text is.
    assert_eq!(full, "Hello");
}

#[test]
fn paced_flushes_and_stops_once_a_tool_call_commits() {
    // An agent consumes a tool round whole, so a committed `call:` ends pacing
    // and surfaces the block immediately rather than dribbling it.
    let mut d = StreamDecoder::new(FakeDecoder, vec![]).paced(true);
    let e = d.on_step(&step(1, 8, &ids("call:{x}"), true));
    assert_eq!(texts(&e), vec!["call:{x}"]);
}
