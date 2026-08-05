//! Live chat rendering, driven by the `ChatEvent` protocol (`chat_protocol`).
//!
//! Two sinks consume the event stream:
//! - **JSONL** (optional): every event as one JSON line, to a file (`--events`)
//!   or stdout (`--json`). This is the observable ground truth.
//! - **Terminal** (default, interactive tty): a spinner + streamed text.
//!
//! The terminal renderer is robust against the failure that plagued the old
//! append-based version (corruption once text wraps across terminal rows):
//! - A background **ticker** thread is the *sole* writer to stdout and repaints
//!   on a fixed timer, so the spinner keeps moving even across the silent
//!   KV-extend / prefill gaps between blocks (the previous "lockup").
//! - Only **immutable committed whole lines** are ever printed permanently
//!   (append-only, may wrap freely — never re-cleared). The speculative draft
//!   and the status live in a single transient status row that is truncated to
//!   the terminal width and cleared with `\r\x1b[2K` — one row, no wrap, no
//!   cursor-up gymnastics.

use crate::chat_protocol::{ChatEvent, StreamDecoder};
use crate::metal::{StepObserver, StepProgressEvent};
use crate::tokenizer::Tokenizer;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const DIM: &str = "\x1b[2m";
const UNDIM: &str = "\x1b[0m";
/// Grey (bright-black): the reasoning colour, matching the `think>` lines the
/// chat command prints elsewhere (`/thought`, forced replies).
const GREY: &str = "\x1b[90m";

/// Terminal (cols, rows) via macOS TIOCGWINSZ, falling back to $COLUMNS/80 × 24.
fn terminal_size() -> (usize, usize) {
    #[repr(C)]
    struct WinSize {
        row: u16,
        col: u16,
        xpix: u16,
        ypix: u16,
    }
    unsafe extern "C" {
        fn ioctl(fd: i32, request: u64, ...) -> i32;
    }
    const TIOCGWINSZ: u64 = 0x4008_7468; // macOS
    let mut ws = WinSize {
        row: 0,
        col: 0,
        xpix: 0,
        ypix: 0,
    };
    let ok = unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut WinSize) } == 0;
    if ok && ws.col > 0 {
        return (ws.col as usize, (ws.row as usize).max(2));
    }
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&c| c > 0)
        .unwrap_or(80);
    (cols, 24)
}

/// Terminal rows a string occupies at `cols` width (counting wrapping; a
/// trailing display-width approximated by char count, adequate for our text).
fn rows_for(text: &str, cols: usize) -> usize {
    let cols = cols.max(1);
    text.split('\n')
        .map(|line| {
            let w = line.chars().count();
            if w == 0 { 1 } else { w.div_ceil(cols) }
        })
        .sum()
}

/// Keep only the last `max_rows` wrapped rows of `text`, prefixing "…\n" when
/// content was dropped. Bounds the redrawable region to the screen so the
/// relative-cursor-up clear never runs off the top.
fn clamp_tail_rows(text: &str, cols: usize, max_rows: usize) -> String {
    if max_rows == 0 {
        return String::new();
    }
    if rows_for(text, cols) <= max_rows {
        return text.to_string();
    }
    let lines: Vec<&str> = text.split('\n').collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut rows = 1; // reserve one row for the "…" marker
    for &line in lines.iter().rev() {
        let r = rows_for(line, cols);
        if rows + r > max_rows && !kept.is_empty() {
            break;
        }
        kept.push(line);
        rows += r;
    }
    kept.reverse();
    format!("…\n{}", kept.join("\n"))
}

/// Clip each line of `text` to `width` characters (no wrapping), so every line
/// occupies exactly one physical terminal row. This keeps the transient's row
/// count exact, so the cursor-up erase never leaves a stray row when a line
/// would otherwise land on the terminal's wrap column.
fn clip_lines(text: &str, width: usize) -> String {
    text.split('\n')
        .map(|line| line.chars().take(width).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Terminal render state: a live dimmed preview plus a one-shot final print.
///
/// The forming answer shows in a **transient** region (dimmed, clipped to the
/// screen), redrawn each tick via relative cursor-up (`\x1b[1A\r\x1b[2K` per row)
/// and cleared before the next repaint. The answer is printed permanently only
/// once, by `finish`, from the authoritative reply. The stream's stable prefix is
/// speculative and can still be revised, so permanently flushing it would leave
/// garbled or duplicated text in the scrollback that `finish` can no longer
/// reach. The reasoning is the exception: it freezes to a permanent grey block
/// above the answer (there is no separate authoritative copy to defer to).
///
/// Two orthogonal display toggles:
/// - `show_thinking`: when set, the reasoning (`Thought` events) streams grey
///   under `think>` during the thought phase, then freezes as one permanent grey
///   block above the answer. When clear, the reasoning stays hidden (spinner).
/// - `stream_output`: when set (default), the forming answer is previewed dimmed;
///   when clear, only a spinner runs. Either way the final answer prints at
///   `finish`.
struct Render {
    /// Latest visible answer preview (whole speculative draft).
    text: String,
    /// Terminal rows the transient region currently occupies.
    transient_rows: usize,
    prefill: bool,
    block: usize,
    step: u32,
    max_steps: usize,
    canvas: usize,
    locked: usize,
    spinner: usize,
    cols: usize,
    rows: usize,
    /// Surface the reasoning live (`--show-thinking` / `/prethink`).
    show_thinking: bool,
    /// Stream the answer as it forms (default); else hold it until `finish`.
    stream_output: bool,
    /// Latest reasoning text (from `Thought` events); shown while thinking.
    thought: String,
    /// The answer has begun (a `Text` event arrived), so the thought is settled.
    answer_started: bool,
    /// The reasoning has been frozen to a permanent grey block already.
    thought_printed: bool,
}

impl Render {
    fn new(cols: usize, rows: usize, show_thinking: bool, stream_output: bool) -> Self {
        Render {
            text: String::new(),
            transient_rows: 0,
            prefill: true,
            block: 0,
            step: 0,
            max_steps: 0,
            canvas: 0,
            locked: 0,
            spinner: 0,
            cols,
            rows,
            show_thinking,
            stream_output,
            thought: String::new(),
            answer_started: false,
            thought_printed: false,
        }
    }

    fn apply(&mut self, ev: &ChatEvent) {
        match ev {
            ChatEvent::BlockStart { block } => self.block = *block,
            ChatEvent::Status {
                block,
                step,
                max_steps,
                canvas,
                locked,
                ..
            } => {
                self.prefill = false;
                self.block = *block;
                self.step = *step;
                self.max_steps = *max_steps;
                self.canvas = *canvas;
                self.locked = *locked;
            }
            ChatEvent::Thought { text } => {
                if self.show_thinking {
                    self.thought = text.clone();
                }
            }
            ChatEvent::Text { text, .. } => {
                // An answer `Text` means the thought has closed: settle it so the
                // reasoning freezes above the answer on the next paint.
                self.answer_started = true;
                self.text = text.clone();
            }
            _ => {}
        }
    }

    /// Erase the transient region, leaving the cursor at its top (== end of the
    /// permanent committed text, always column 0 since committed ends at '\n').
    fn erase_transient(&mut self) {
        if self.transient_rows == 0 {
            return;
        }
        // `\x1b[G` (cursor to column 1) rather than `\r`: a CSI sequence is
        // immune to CR→NL output translation (OCRNL), which would otherwise
        // walk the cursor down and stack the redraws instead of clearing.
        print!("\x1b[G\x1b[2K");
        for _ in 1..self.transient_rows {
            print!("\x1b[1A\x1b[G\x1b[2K");
        }
        self.transient_rows = 0;
    }

    /// Freeze the reasoning to one permanent grey `think>` block above the
    /// answer, once (when the answer starts, or at `finish`). No-op unless
    /// `show_thinking` and there is a thought to show.
    fn freeze_thought(&mut self) {
        if self.thought_printed || !self.show_thinking {
            return;
        }
        let t = self.thought.trim();
        if !t.is_empty() {
            println!("{GREY}think> {t}{UNDIM}");
        }
        self.thought_printed = true;
    }

    /// Repaint. Sole stdout writer during a turn.
    fn paint(&mut self) {
        self.erase_transient();
        if self.answer_started {
            self.freeze_thought();
        }
        // Transient region, redrawn each tick. Every line is clipped to a single
        // physical row (no wrapping), so `transient_rows` is exact and the
        // cursor-up erase never leaves a stray dimmed row when a line lands on the
        // wrap column. Clamped to the screen height so the redraw stays on-screen.
        let frame = SPINNER[self.spinner % SPINNER.len()];
        let status = |r: &Self| {
            clip_lines(
                &format!(
                    "{frame} thinking · block {} · step {}/{} · {}/{} locked",
                    r.block, r.step, r.max_steps, r.locked, r.canvas
                ),
                r.cols.saturating_sub(1),
            )
        };
        // Leave room on a body line for a `think>`/`model>` prefix and the spinner.
        let body_w = self.cols.saturating_sub(8).max(1);
        let clamp = self.rows.saturating_sub(1);
        let tail = self.text.trim();
        let (styled, trows) = if self.prefill {
            (format!("{frame} thinking…"), 1)
        } else if self.show_thinking && !self.answer_started {
            // Reasoning phase: stream the thought grey (spinner alone if empty).
            let t = self.thought.trim();
            if t.is_empty() {
                (status(self), 1)
            } else {
                let body = clip_lines(&clamp_tail_rows(t, self.cols, clamp), body_w);
                let rows = body.split('\n').count();
                (format!("think> {GREY}{body}{UNDIM} {frame}"), rows)
            }
        } else if !self.stream_output || tail.is_empty() {
            // Answer withheld (or none stable yet): a working spinner.
            (status(self), 1)
        } else {
            let prefix = "model> ";
            let body = clip_lines(&clamp_tail_rows(tail, self.cols, clamp), body_w);
            let rows = body.split('\n').count();
            (format!("{prefix}{DIM}{body}{UNDIM} {frame}"), rows)
        };
        print!("{styled}");
        self.transient_rows = trows;
        let _ = std::io::stdout().flush();
        self.spinner = self.spinner.wrapping_add(1);
    }

    /// Erase the preview and print the authoritative reply — the sole permanent
    /// answer write, so the speculative draft can never reach the scrollback.
    fn finish(&mut self, reply: &str) {
        self.erase_transient();
        // Ensure the reasoning is shown even when no answer ever streamed (a
        // thought-only turn, or a salvaged answer).
        self.freeze_thought();
        if reply.is_empty() {
            println!("model> (empty response)");
        } else {
            println!("model> {reply}");
        }
    }
}

/// Shared state behind one mutex: the decoder + JSON sink (touched by the
/// generation thread via the observer) and the terminal render state (painted
/// by the ticker). Only the ticker and `finish` write to stdout.
struct Shared {
    decoder: StreamDecoder<Arc<Tokenizer>>,
    json: Option<Box<dyn Write + Send>>,
    render: Render,
    interactive: bool,
}

impl Shared {
    fn emit(&mut self, ev: &ChatEvent) {
        if let Some(w) = self.json.as_mut()
            && serde_json::to_writer(&mut *w, ev).is_ok()
        {
            let _ = w.write_all(b"\n");
            let _ = w.flush();
        }
        if self.interactive {
            self.render.apply(ev);
        }
    }
}

pub struct ChatStream {
    shared: Arc<Mutex<Shared>>,
    done: Arc<AtomicBool>,
    interactive: bool,
    ticker: Option<std::thread::JoinHandle<()>>,
}

/// Live-display options for a streamed turn.
pub struct StreamDisplay {
    /// Surface the reasoning live (grey `think>`); implied by `prethink_seed`.
    pub show_thinking: bool,
    /// `/prethink` continuation prefix (the reasoning the prompt opened with).
    pub prethink_seed: Option<String>,
    /// Stream the answer as it forms (default) vs print it whole at `finish`.
    pub stream_output: bool,
}

impl ChatStream {
    /// `interactive` enables the terminal spinner/streaming (an interactive tty).
    /// `json` is an optional JSONL sink (file or stdout); `turn` / `prompt_tokens`
    /// seed the opening `TurnStart` event.
    pub fn start(
        tokenizer: Arc<Tokenizer>,
        stop_token_ids: Vec<u32>,
        interactive: bool,
        display: StreamDisplay,
        json: Option<Box<dyn Write + Send>>,
        turn: u64,
        prompt_tokens: usize,
    ) -> Self {
        let show_thinking = display.show_thinking || display.prethink_seed.is_some();
        let shared = Arc::new(Mutex::new(Shared {
            decoder: StreamDecoder::new(Arc::clone(&tokenizer), stop_token_ids)
                .with_thinking(show_thinking, display.prethink_seed),
            json,
            render: {
                let (cols, rows) = terminal_size();
                Render::new(cols, rows, show_thinking, display.stream_output)
            },
            interactive,
        }));
        {
            let mut s = shared.lock().unwrap();
            s.emit(&ChatEvent::TurnStart {
                turn,
                prompt_tokens,
            });
        }
        let done = Arc::new(AtomicBool::new(false));
        let ticker = if interactive {
            let shared_t = Arc::clone(&shared);
            let done_t = Arc::clone(&done);
            Some(std::thread::spawn(move || {
                loop {
                    if done_t.load(Ordering::Relaxed) {
                        break;
                    }
                    shared_t.lock().unwrap().render.paint();
                    std::thread::sleep(Duration::from_millis(100));
                }
            }))
        } else {
            None
        };
        Self {
            shared,
            done,
            interactive,
            ticker,
        }
    }

    /// Observer to install as `StepGenerateConfig::step_observer`.
    pub fn observer(&self) -> StepObserver {
        let shared = Arc::clone(&self.shared);
        Arc::new(move |ev: &StepProgressEvent<'_>| {
            let mut s = shared.lock().unwrap();
            let events = s.decoder.on_step(ev);
            for e in &events {
                s.emit(e);
            }
        })
    }

    /// Reconcile the stream against the authoritative reply, terminate the
    /// line, and close the JSON stream with a `Done` event.
    pub fn finish(mut self, reply: &str, tokens: usize, steps: usize, secs: f64, stopped: bool) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.ticker.take() {
            let _ = t.join();
        }
        let mut s = self.shared.lock().unwrap();
        if self.interactive {
            s.render.finish(reply);
        }
        s.emit(&ChatEvent::Done {
            tokens,
            steps,
            secs,
            stopped,
            text: reply.to_string(),
        });
    }
}
