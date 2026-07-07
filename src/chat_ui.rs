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

/// Terminal render state: inline dimmed streaming with no status line.
///
/// Layout: committed (block-final) text is printed permanently as NORMAL text,
/// append-only, flushed only on complete lines so it always ends at a line
/// boundary — it may wrap and scroll freely and is never re-touched. Below it, a
/// **transient** region shows the current block's still-forming tail DIMMED,
/// exactly where the text will land; when the block commits, that text is
/// re-emitted permanently as normal (un-dimmed). The transient is redrawn each
/// tick via relative cursor-up (`\x1b[1A\r\x1b[2K` per row) and is clamped to
/// the screen height, so the clear never runs off the top — scroll-safe with no
/// DECSC/DECRC (which don't survive scrolling).
struct Render {
    /// Full visible answer text; `[0, committed_len)` is immutable/committed.
    text: String,
    committed_len: usize,
    /// Bytes of committed text already flushed permanently (ends at '\n' or 0).
    printed: usize,
    /// True once the permanent "model> " prefix has been printed.
    prefixed: bool,
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
}

impl Render {
    fn new(cols: usize, rows: usize) -> Self {
        Render {
            text: String::new(),
            committed_len: 0,
            printed: 0,
            prefixed: false,
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
            ChatEvent::Text { committed, text } => {
                self.text = text.clone();
                self.committed_len = *committed;
                if self.printed > self.committed_len {
                    self.printed = self.committed_len; // committed prefix shrank (rare)
                }
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

    /// Repaint. Sole stdout writer during a turn.
    fn paint(&mut self) {
        self.erase_transient();
        // Flush newly-committed COMPLETE lines permanently (normal text). This
        // keeps the permanent region ending at a line boundary so the transient
        // below always starts at column 0.
        let region_end = self.committed_len.min(self.text.len());
        if self.printed < region_end {
            if let Some(rel) = self.text[self.printed..region_end].rfind('\n') {
                let end = self.printed + rel + 1;
                if !self.prefixed {
                    print!("model> ");
                    self.prefixed = true;
                }
                print!("{}", &self.text[self.printed..end]);
                self.printed = end;
            }
        }
        // Transient: the still-forming tail (uncommitted + any partial committed
        // line), dimmed, clamped to the screen so its redraw stays on-screen.
        // `styled` carries the dim/spinner escapes; `visible` is the same text
        // without escapes, used to count the rows to clear next tick.
        let frame = SPINNER[self.spinner % SPINNER.len()];
        let tail = &self.text[self.printed..];
        let (styled, visible) = if self.prefill {
            let s = format!("{frame} thinking…");
            (s.clone(), s)
        } else if tail.trim().is_empty() {
            let s = format!(
                "{frame} thinking · block {} · step {}/{} · {}/{} locked",
                self.block, self.step, self.max_steps, self.locked, self.canvas
            );
            (s.clone(), s)
        } else {
            let prefix = if self.prefixed { "" } else { "model> " };
            let body = clamp_tail_rows(tail, self.cols, self.rows.saturating_sub(1));
            (
                format!("{prefix}{DIM}{body}{UNDIM} {frame}"),
                format!("{prefix}{body} {frame}"),
            )
        };
        print!("{styled}");
        self.transient_rows = rows_for(&visible, self.cols);
        let _ = std::io::stdout().flush();
        self.spinner = self.spinner.wrapping_add(1);
    }

    /// Reconcile against the authoritative reply and end the line.
    fn finish(&mut self, reply: &str) {
        self.erase_transient();
        let shown = &self.text[..self.printed.min(self.text.len())];
        if reply.is_empty() {
            if self.prefixed {
                println!();
            } else {
                println!("model> (empty response)");
            }
            return;
        }
        if !self.prefixed {
            println!("model> {reply}");
        } else if let Some(rest) = reply.strip_prefix(shown) {
            // Re-emit the remaining (previously-dimmed) text permanently, normal.
            print!("{rest}");
            println!();
        } else {
            // Streamed prefix diverged from the final commit (rare); reprint.
            println!();
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
        if let Some(w) = self.json.as_mut() {
            if serde_json::to_writer(&mut *w, ev).is_ok() {
                let _ = w.write_all(b"\n");
                let _ = w.flush();
            }
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

impl ChatStream {
    /// `interactive` enables the terminal spinner/streaming (an interactive
    /// tty). `json` is an optional JSONL sink (file or stdout). `turn` /
    /// `prompt_tokens` seed the opening `TurnStart` event.
    pub fn start(
        tokenizer: Arc<Tokenizer>,
        stop_token_ids: Vec<u32>,
        interactive: bool,
        json: Option<Box<dyn Write + Send>>,
        turn: u64,
        prompt_tokens: usize,
    ) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            decoder: StreamDecoder::new(Arc::clone(&tokenizer), stop_token_ids),
            json,
            render: {
                let (cols, rows) = terminal_size();
                Render::new(cols, rows)
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
