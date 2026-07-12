//! Tests for `tests`, extracted from server.rs (backlog item 3).

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
        step_in_block: step,
        max_steps: 48,
        argmax,
        accept_count: 0,
        block_done: done,
    }
}

fn c(ch: char) -> u32 {
    ch as u32
}

#[test]
fn content_only_commits_answer_text() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, false, false);
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
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, false, true);
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
    let mut m =
        DiffusionStreamMapper::new(FakeDecoder, vec![], Some(open), Some(close), true, false);
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
fn stop_token_ends_and_cuts() {
    let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![99], None, None, false, false);
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
