//! The interactive terminal renderer, driven by the `ChatEvent` protocol.
//!
//! [`ChatStream`] runs one generation's stream: its observer decodes each
//! denoise step, pumps the events into the session sinks, and (on an
//! interactive tty) applies them to the live display. The renderer reserves a
//! pane at the bottom of the screen (a DECSTBM scroll region) while the
//! generation runs:
//! - A background **ticker** thread is the *sole* writer to stdout, redrawing the
//!   pane on a fixed timer so the spinner keeps moving across the silent
//!   KV-extend / prefill gaps between blocks.
//! - The speculative preview lives only in the fixed pane and is wiped when the
//!   turn commits; the normal scrollback above it is untouched. The authoritative
//!   reply is printed once, into real scrollback, at `finish` (see [`Viewport`]).

use super::ChatEvent;
use crate::decoder::{ChannelIds, StepProgressEvent, StreamDecoder};
use crate::metal::StepObserver;
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

/// Disable / enable terminal auto-wrap (DECAWM). A preview line is drawn with
/// wrap OFF so it stays on one physical row — the terminal truncates overflow at
/// the right margin instead of wrapping — keeping the fixed pane's rows aligned
/// without the renderer ever assuming a column count.
const WRAP_OFF: &str = "\x1b[?7l";
const WRAP_ON: &str = "\x1b[?7h";
/// Reset scroll region to the full screen, then show the cursor. Emitted on
/// every teardown path (commit, Drop, panic, signal) so the terminal is never
/// left stuck in a restricted scrolling region.
const REGION_RESET: &[u8] = b"\x1b[r\x1b[?25h";

// macOS libc bindings (hand-rolled to stay dependency-free, matching the ioctl
// style below). Used for the cursor-position query and the signal safety net.
#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: u64,
    c_oflag: u64,
    c_cflag: u64,
    c_lflag: u64,
    c_cc: [u8; 20],
    c_ispeed: u64,
    c_ospeed: u64,
}
#[repr(C)]
struct Pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}
unsafe extern "C" {
    fn ioctl(fd: i32, request: u64, ...) -> i32;
    fn tcgetattr(fd: i32, termios: *mut Termios) -> i32;
    fn tcsetattr(fd: i32, opt: i32, termios: *const Termios) -> i32;
    fn cfmakeraw(termios: *mut Termios);
    fn poll(fds: *mut Pollfd, nfds: u32, timeout: i32) -> i32;
    fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn signal(sig: i32, handler: usize) -> usize;
    fn raise(sig: i32) -> i32;
}

/// Terminal (cols, rows) via macOS TIOCGWINSZ, falling back to (80, 24). Width is
/// used only to wrap the preview into the pane; a wrong value can at worst clip a
/// line (the pane draws with wrap off), never corrupt the layout — the row count
/// is fixed, not derived from it.
fn winsize() -> (usize, usize) {
    #[repr(C)]
    struct WinSize {
        row: u16,
        col: u16,
        xpix: u16,
        ypix: u16,
    }
    const TIOCGWINSZ: u64 = 0x4008_7468; // macOS
    let mut ws = WinSize {
        row: 0,
        col: 0,
        xpix: 0,
        ypix: 0,
    };
    if unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut WinSize) } == 0 && ws.row > 0 {
        return ((ws.col as usize).max(1), (ws.row as usize).max(2));
    }
    (80, 24)
}

fn terminal_rows() -> usize {
    winsize().1
}

/// Wrap `text` to `width` columns: split on existing newlines, hard-break long
/// lines. Char-based (not display-width), adequate for the preview.
fn wrap_to_width(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows = Vec::new();
    for line in text.split('\n') {
        let chars: Vec<char> = line.chars().collect();
        if chars.is_empty() {
            rows.push(String::new());
            continue;
        }
        for chunk in chars.chunks(width) {
            rows.push(chunk.iter().collect());
        }
    }
    rows
}

/// Ask the terminal for the cursor row (1-based) via a Cursor-Position Report
/// (`\x1b[6n` -> `\x1b[row;colR` on stdin). Flips stdin to raw so the reply is
/// readable without a newline, with a short poll timeout; `None` if the terminal
/// does not answer. This is the one dimension the renderer needs but cannot get
/// from the winsize: where the live pane should sit.
fn query_cursor_row() -> Option<u16> {
    const STDIN: i32 = 0;
    let mut orig: Termios = unsafe { std::mem::zeroed() };
    if unsafe { tcgetattr(STDIN, &mut orig) } != 0 {
        return None;
    }
    let mut raw = orig;
    unsafe { cfmakeraw(&mut raw) };
    if unsafe { tcsetattr(STDIN, 0, &raw) } != 0 {
        return None;
    }
    let row = read_cpr(STDIN);
    unsafe { tcsetattr(STDIN, 0, &orig) };
    row
}

fn read_cpr(fd: i32) -> Option<u16> {
    print!("\x1b[6n");
    std::io::stdout().flush().ok()?;
    let mut buf = [0u8; 32];
    let mut len = 0usize;
    loop {
        let mut pfd = Pollfd {
            fd,
            events: 0x0001, // POLLIN
            revents: 0,
        };
        if unsafe { poll(&mut pfd, 1, 200) } <= 0 {
            break;
        }
        let mut b = 0u8;
        if unsafe { read(fd, &mut b, 1) } != 1 {
            break;
        }
        if len < buf.len() {
            buf[len] = b;
            len += 1;
        }
        if b == b'R' {
            break;
        }
    }
    let s = std::str::from_utf8(&buf[..len]).ok()?;
    let s = s.strip_prefix('\x1b')?.strip_prefix('[')?;
    s.split(';').next()?.parse().ok()
}

/// True while a scroll region is set. The signal handler and panic hook read it
/// to decide whether the terminal needs a region reset before the process dies.
static VIEWPORT_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Install the region-reset safety net once: a SIGINT/SIGTERM handler and a
/// panic hook that both emit [`REGION_RESET`] (a `write` syscall is
/// async-signal-safe) so a crash mid-generation cannot leave the shell stuck
/// scrolling only the bottom few lines.
fn install_guards() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        let handler = on_fatal_signal as *const () as usize;
        unsafe {
            signal(SIGINT, handler);
            signal(SIGTERM, handler);
        }
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            reset_region_if_active();
            prev(info);
        }));
    });
}

fn reset_region_if_active() {
    if VIEWPORT_ACTIVE.load(Ordering::Relaxed) {
        unsafe {
            write(1, REGION_RESET.as_ptr(), REGION_RESET.len());
        }
    }
}

extern "C" fn on_fatal_signal(sig: i32) {
    reset_region_if_active();
    unsafe {
        signal(sig, 0); // SIG_DFL
        raise(sig);
    }
}

/// A reserved live region at the bottom of the terminal, above which the normal
/// scrollback keeps flowing. The generation preview redraws into the fixed
/// bottom pane (rows `anchor+1 ..= anchor+n`); the main content scrolls in the
/// region `[1, anchor]`. On [`Viewport::commit`] the region is reset and the
/// authoritative reply is printed into normal flow starting at `anchor`, so it
/// joins real scrollback. The speculative preview lives only in the fixed pane
/// and is wiped at commit — it can never reach scrollback.
struct Viewport {
    anchor: u16,
    n: u16,
    active: bool,
}

impl Viewport {
    /// Reserve `n` lines below the cursor and lock the scrolling region above
    /// them. `None` if the terminal will not report the cursor (see
    /// [`query_cursor_row`]) or there is no room.
    fn enter(n: u16) -> Option<Viewport> {
        install_guards();
        // Reserve n lines (scroll up if at the bottom), then step the cursor back
        // to the reply anchor so the reserved rows sit directly below the input.
        print!("{}\x1b[{n}A", "\n".repeat(n as usize));
        let _ = std::io::stdout().flush();
        let anchor = query_cursor_row()?;
        let h = terminal_rows() as u16;
        let n = n.min(h.saturating_sub(anchor));
        if n == 0 {
            return None;
        }
        // Region = the top `anchor` lines; reposition to the anchor (setting the
        // region homes the cursor) and hide it while the pane redraws.
        print!("\x1b[1;{anchor}r\x1b[{anchor};1H\x1b[?25l");
        let _ = std::io::stdout().flush();
        VIEWPORT_ACTIVE.store(true, Ordering::Relaxed);
        Some(Viewport {
            anchor,
            n,
            active: true,
        })
    }

    /// Redraw the pane: each of the `n` rows is positioned absolutely, cleared,
    /// and (if it has content) written with wrap off so it stays one row. The
    /// cursor returns to the anchor.
    fn draw(&self, lines: &[String]) {
        let mut out = String::new();
        for i in 0..self.n {
            out.push_str(&format!("\x1b[{};1H\x1b[2K", self.anchor + 1 + i));
            if let Some(line) = lines.get(i as usize) {
                out.push_str(WRAP_OFF);
                out.push_str(line);
                out.push_str(WRAP_ON);
            }
        }
        out.push_str(&format!("\x1b[{};1H", self.anchor));
        print!("{out}");
        let _ = std::io::stdout().flush();
    }

    /// Reset the region, wipe the pane, and print `permanent` (which should end
    /// with a newline) into normal flow from the anchor, where it scrolls into
    /// real scrollback. Idempotent teardown is left to `Drop`.
    fn commit(&mut self, permanent: &str) {
        print!("\x1b[r\x1b[{};1H\x1b[J{permanent}\x1b[?25h", self.anchor);
        let _ = std::io::stdout().flush();
        self.active = false;
        VIEWPORT_ACTIVE.store(false, Ordering::Relaxed);
    }
}

impl Drop for Viewport {
    fn drop(&mut self) {
        if self.active {
            let _ = std::io::stdout().write_all(REGION_RESET);
            let _ = std::io::stdout().flush();
            VIEWPORT_ACTIVE.store(false, Ordering::Relaxed);
        }
    }
}

/// Terminal render state: a live preview in a reserved bottom pane plus a
/// one-shot authoritative print at `finish`.
///
/// The forming answer/reasoning streams into a fixed [`Viewport`] pane; the
/// authoritative reply is printed once, at `finish`, into normal scrollback. The
/// stream's stable prefix is speculative and can still be revised, so it is never
/// flushed permanently — only the settled reply is. When there is no viewport
/// (no cursor report, or a non-tty), the preview degrades to a single spinner
/// line and the reply still prints at `finish`.
///
/// Two orthogonal display toggles:
/// - `show_thinking`: stream the reasoning (`Thought` events) grey under `think>`
///   during the thought phase; when clear it stays hidden (spinner).
/// - `stream_output`: preview the forming answer (default) vs a spinner. Either
///   way the final answer prints at `finish`.
struct Render {
    text: String,
    prefill: bool,
    block: usize,
    step: u32,
    max_steps: usize,
    canvas: usize,
    locked: usize,
    spinner: usize,
    show_thinking: bool,
    stream_output: bool,
    thought: String,
    answer_started: bool,
    /// The reserved bottom pane, when the terminal supports it.
    viewport: Option<Viewport>,
}

impl Render {
    fn new(show_thinking: bool, stream_output: bool, interactive: bool) -> Self {
        // A multi-line preview earns a tall pane; a spinner-only turn takes one row.
        let viewport = interactive.then(|| {
            let want = if show_thinking || stream_output {
                10
            } else {
                1
            };
            let n = (terminal_rows().saturating_sub(2)).min(want).max(1) as u16;
            Viewport::enter(n)
        });
        Render {
            text: String::new(),
            prefill: true,
            block: 0,
            step: 0,
            max_steps: 0,
            canvas: 0,
            locked: 0,
            spinner: 0,
            show_thinking,
            stream_output,
            thought: String::new(),
            answer_started: false,
            viewport: viewport.flatten(),
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
                self.answer_started = true;
                self.text = text.clone();
            }
            _ => {}
        }
    }

    fn status_line(&self, frame: &str) -> String {
        format!(
            "{frame} thinking · block {} · step {}/{} · {}/{} locked",
            self.block, self.step, self.max_steps, self.locked, self.canvas
        )
    }

    /// The pane's content: the current phase (reasoning, answer, or a status
    /// spinner) wrapped to `cols` and cut to the last `pane` rows.
    fn preview_lines(&self, frame: &str, pane: usize, cols: usize) -> Vec<String> {
        if self.prefill {
            return vec![format!("{frame} thinking…")];
        }
        let reasoning = self.show_thinking && !self.answer_started;
        let (marker, body, color) = if reasoning {
            ("think> ", self.thought.trim(), GREY)
        } else if self.stream_output {
            ("model> ", self.text.trim(), DIM)
        } else {
            return vec![self.status_line(frame)]; // answer withheld
        };
        if body.is_empty() {
            return vec![self.status_line(frame)];
        }
        let wrapped = wrap_to_width(&format!("{marker}{body}"), cols);
        let start = wrapped.len().saturating_sub(pane);
        wrapped[start..]
            .iter()
            .map(|r| format!("{color}{r}{UNDIM}"))
            .collect()
    }

    /// Repaint. Sole stdout writer during a turn.
    fn paint(&mut self) {
        let frame = SPINNER[self.spinner % SPINNER.len()];
        self.spinner = self.spinner.wrapping_add(1);
        match &self.viewport {
            Some(vp) => {
                let lines = self.preview_lines(frame, vp.n as usize, winsize().0);
                vp.draw(&lines);
            }
            None => {
                // No reserved pane: a single in-place status line.
                print!("\r\x1b[2K{}", self.status_line(frame));
                let _ = std::io::stdout().flush();
            }
        }
    }

    /// Settle the turn into normal scrollback: the `think>` block (reasoning
    /// turns) above the reply, then `stats` as a trailing dim line. `reply` is
    /// the answer text (empty renders as `(empty response)`); `None` prints no
    /// `model>` line at all, for a tool-call round that only reasoned before
    /// calling.
    fn finish(&mut self, reply: Option<&str>, stats: Option<&str>) {
        let mut permanent = String::new();
        if self.show_thinking {
            let t = self.thought.trim();
            if !t.is_empty() {
                permanent.push_str(&format!("{GREY}think> {t}{UNDIM}\n"));
            }
        }
        match reply {
            Some(r) if !r.is_empty() => permanent.push_str(&format!("model> {r}\n")),
            Some(_) => permanent.push_str("model> (empty response)\n"),
            None => {}
        }
        if let Some(line) = stats {
            permanent.push_str(&format!("  {line}\n"));
        }
        match self.viewport.as_mut() {
            Some(vp) => vp.commit(&permanent),
            None => print!("\r\x1b[2K{permanent}"),
        }
        let _ = std::io::stdout().flush();
    }
}

/// No-GPU renderer exercise for terminal verification (`DGQ_RENDER_DEMO`): drives
/// the real `Render`/`Viewport` with synthetic events so the escape output can be
/// captured (e.g. under tmux) without loading a model. `stage` is `preview` (hold
/// mid-reasoning) or anything else (run through commit); it then sleeps so the
/// final screen can be captured.
#[doc(hidden)]
pub fn render_demo(stage: &str) {
    for i in 0..22 {
        println!("hist{i:02}");
    }
    println!("you> demo: explain the plan");
    let _ = std::io::stdout().flush();
    let mut r = Render::new(true, true, true);
    r.apply(&ChatEvent::Status {
        block: 1,
        step: 1,
        max_steps: 24,
        accepted: 0,
        canvas: 256,
        locked: 0,
    });
    let mut thought = String::new();
    for i in 0..12 {
        thought.push_str(&format!(
            "reasoning step {i}: consider this approach and the trade-offs it carries here\n"
        ));
        r.apply(&ChatEvent::Thought {
            text: thought.clone(),
        });
        r.paint();
        std::thread::sleep(Duration::from_millis(120));
    }
    if stage == "preview" {
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    for i in 0..6 {
        let text = (0..=i)
            .map(|j| {
                format!("answer line {j}: content long enough that it might overflow the pane")
            })
            .collect::<Vec<_>>()
            .join("\n");
        r.apply(&ChatEvent::Text { committed: 0, text });
        r.paint();
        std::thread::sleep(Duration::from_millis(120));
    }
    r.finish(
        Some("The final settled answer, line one.\nAnd line two of the reply."),
        Some("(42 tok · 12 steps · 3.1s · 13.5 tok/s)"),
    );
    std::thread::sleep(Duration::from_secs(30));
}

/// Shared state behind one mutex: the decoder (driven by the generation
/// thread via the observer) and the terminal render state (painted by the
/// ticker). Only the ticker and `finish` write to stdout. Protocol events
/// fan out through the session [`super::SharedSinks`].
struct Shared {
    decoder: StreamDecoder<Arc<Tokenizer>>,
    sinks: super::SharedSinks,
    render: Render,
    interactive: bool,
}

impl Shared {
    fn emit(&mut self, ev: &ChatEvent) {
        super::sink::emit(&self.sinks, ev);
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
    /// The generation begins inside an already-open thought channel (a tool
    /// round's forced-open thought) even without a seed.
    pub start_in_thought: bool,
    /// Stream the answer as it forms (default) vs print it whole at `finish`.
    pub stream_output: bool,
}

impl ChatStream {
    /// `interactive` enables the terminal spinner/streaming (an interactive tty).
    /// Protocol events flow through `sinks`; `round` / `prompt_tokens` seed the
    /// opening `RoundStart` event.
    pub fn start(
        tokenizer: Arc<Tokenizer>,
        stop_token_ids: Vec<u32>,
        interactive: bool,
        display: StreamDisplay,
        sinks: super::SharedSinks,
        round: usize,
        prompt_tokens: usize,
    ) -> Self {
        let show_thinking = display.show_thinking || display.prethink_seed.is_some();
        let channels = ChannelIds::from_tokenizer(&tokenizer);
        let mut decoder = StreamDecoder::new(Arc::clone(&tokenizer), stop_token_ids)
            .with_channels(channels)
            .with_thinking(show_thinking, display.prethink_seed);
        if display.start_in_thought {
            decoder = decoder.starting_in_thought();
        }
        let shared = Arc::new(Mutex::new(Shared {
            decoder,
            sinks,
            render: Render::new(show_thinking, display.stream_output, interactive),
            interactive,
        }));
        {
            let mut s = shared.lock().unwrap();
            s.emit(&ChatEvent::RoundStart {
                round,
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

    /// Settle the interactive display against the authoritative reply:
    /// stop the ticker, commit the viewport, and print `reply` (with `stats`
    /// as a trailing dim line, if given) into real scrollback. `reply` =
    /// `None` settles with no `model>` line (a tool-call round that only
    /// narrated). Display-only — the caller emits the protocol's `RoundEnd`
    /// or `Done` through the sinks.
    pub fn finish(mut self, reply: Option<&str>, stats: Option<&str>) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.ticker.take() {
            let _ = t.join();
        }
        let mut s = self.shared.lock().unwrap();
        if self.interactive {
            s.render.finish(reply, stats);
        }
    }
}
