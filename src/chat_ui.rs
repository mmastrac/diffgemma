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

/// Terminal columns (macOS TIOCGWINSZ), falling back to `$COLUMNS` then 80.
fn terminal_cols() -> usize {
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
        return ws.col as usize;
    }
    std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&c| c > 0)
        .unwrap_or(80)
}

/// Truncate to at most `cols` chars (approx display width; good enough for the
/// mostly-ASCII status row).
fn truncate_cols(s: &str, cols: usize) -> String {
    if s.chars().count() <= cols {
        s.to_string()
    } else {
        s.chars().take(cols.saturating_sub(1)).collect::<String>() + "…"
    }
}

/// Terminal render state, mutated by events and painted by the ticker.
///
/// Layout invariant: immutable **committed** text is printed permanently and
/// append-only (it may wrap freely; never re-cleared). A single transient
/// status row lives below it, drawn via cursor save/restore (`\x1b7`/`\x1b8`):
/// each repaint restores to the saved end-of-permanent, clears everything below
/// (`\x1b[0J`), flushes any new committed text, re-saves, and prints one status
/// line. This is prose-safe (committed streams immediately, not only on
/// newlines) and wrap-safe (clear-to-end-of-screen handles any width).
struct Render {
    /// Full visible text; `[0, committed_len)` is immutable/committed.
    text: String,
    committed_len: usize,
    /// Bytes of the committed region already printed permanently.
    printed_upto: usize,
    /// True once the permanent "model> " prefix has been printed.
    prefixed: bool,
    /// True once a transient status row has been drawn (guards `\x1b8`).
    drawn: bool,
    // Status telemetry.
    prefill: bool,
    block: usize,
    step: u32,
    max_steps: usize,
    canvas: usize,
    locked: usize,
    spinner: usize,
    cols: usize,
}

impl Render {
    fn new(cols: usize) -> Self {
        Render {
            text: String::new(),
            committed_len: 0,
            printed_upto: 0,
            prefixed: false,
            drawn: false,
            prefill: true,
            block: 0,
            step: 0,
            max_steps: 0,
            canvas: 0,
            locked: 0,
            spinner: 0,
            cols,
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
                if self.printed_upto > self.committed_len {
                    // Committed prefix shrank (rare re-sanitize divergence);
                    // never re-print, just stop advancing past the new bound.
                    self.printed_upto = self.committed_len;
                }
            }
            // Rewind only affects the transient draft (redrawn each tick);
            // BlockCommit/TurnStart/Done carry no extra terminal state.
            _ => {}
        }
    }

    /// Erase the transient region, returning the cursor to end-of-permanent.
    fn erase_transient(&self) {
        if self.drawn {
            print!("\x1b8\x1b[0J"); // restore to saved end-of-permanent, clear below
        } else {
            print!("\r\x1b[2K"); // first paint: just clear the current row
        }
    }

    /// Repaint: flush any new committed text permanently, then draw the single
    /// transient status row. Sole stdout writer during a turn.
    fn paint(&mut self) {
        self.erase_transient();
        // Flush all newly-committed (immutable) text permanently.
        let region_end = self.committed_len.min(self.text.len());
        if self.printed_upto < region_end {
            if !self.prefixed {
                print!("model> ");
                self.prefixed = true;
            }
            print!("{}", &self.text[self.printed_upto..region_end]);
            self.printed_upto = region_end;
        }
        print!("\x1b7"); // save cursor at end of permanent text
        self.drawn = true;
        // Transient status: drawn at the cursor (NO leading newline — a leading
        // `\n` scrolls at the screen bottom and, because DECSC/DECRC don't track
        // scrolling, leaves stale saved positions that pile up as blank lines).
        // A separator space keeps the spinner off any mid-line committed text.
        let frame = SPINNER[self.spinner % SPINNER.len()];
        let at_line_start =
            self.printed_upto == 0 || self.text[..self.printed_upto].ends_with('\n');
        let sep = if at_line_start { "" } else { " " };
        let draft = self.text[self.printed_upto..].replace('\n', " ");
        let draft = draft.trim();
        let body = if self.prefill {
            "prefilling…".to_string()
        } else if draft.is_empty() {
            format!(
                "thinking · block {} · step {}/{} · {}/{} locked",
                self.block, self.step, self.max_steps, self.locked, self.canvas
            )
        } else {
            // Tail of the forming draft, so long drafts stay on one row.
            let cap = self.cols.saturating_sub(6);
            if draft.chars().count() > cap {
                let tail: String = draft
                    .chars()
                    .rev()
                    .take(cap)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                format!("…{tail}")
            } else {
                draft.to_string()
            }
        };
        print!(
            "{sep}{}",
            truncate_cols(&format!("{frame} {body}"), self.cols.saturating_sub(1))
        );
        let _ = std::io::stdout().flush();
        self.spinner = self.spinner.wrapping_add(1);
    }

    /// Reconcile against the authoritative reply and end the line.
    fn finish(&mut self, reply: &str) {
        self.erase_transient(); // cursor now at end of permanent committed text
        let shown = &self.text[..self.printed_upto.min(self.text.len())];
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
            // Continue the committed line with the authoritative remainder.
            print!("{rest}");
            println!();
        } else {
            // The streamed prefix diverged from the final commit (rare late
            // revision); reprint the authoritative reply fresh.
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
            render: Render::new(terminal_cols()),
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
            Some(std::thread::spawn(move || loop {
                if done_t.load(Ordering::Relaxed) {
                    break;
                }
                shared_t.lock().unwrap().render.paint();
                std::thread::sleep(Duration::from_millis(100));
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
