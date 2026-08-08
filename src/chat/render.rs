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

    /// Print `chunk` (whole lines, each ending with a newline) at the anchor
    /// row while the region is still active: the region scrolls it up into
    /// real scrollback and the pane below is untouched. Only text that can
    /// never be revised may go through here: scrollback cannot be unprinted.
    fn flush_permanent(&self, chunk: &str) {
        print!("\x1b[{};1H{chunk}", self.anchor);
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

/// The flushable byte cut of `text`: the end of the last complete line that
/// is both block-committed (`committed`, from the `Text` event) and settled
/// (`tools::settled_prefix_len`: no unclosed fence or forming tool call whose
/// completion could re-mask earlier bytes). Text before the cut can never
/// change, so it is safe to print into scrollback.
fn flush_cut(text: &str, committed: usize) -> usize {
    let committed = committed.min(text.len());
    let settled = crate::tools::settled_prefix_len(&text[..committed]);
    text[..settled].rfind('\n').map_or(0, |i| i + 1)
}

/// Bytes to newly reveal on a paced repaint: a fraction of the backlog so a
/// block that lands whole drains fast at first then eases in, floored and capped
/// so the rate stays legible either way. Tuned against the ~100 ms repaint tick.
fn reveal_step(remaining: usize) -> usize {
    (remaining / 8).clamp(6, 40).min(remaining)
}

/// Printable characters per group, and the pause between groups, for the
/// no-viewport typewriter below.
const TYPE_GROUP: usize = 4;
const TYPE_DELAY_MS: u64 = 22;

/// Print `s` to stdout as if typed: printable characters in small groups with a
/// short pause between them, ANSI escape sequences passed through whole (never
/// split, no pause). Used when a paced turn has no reserved pane to animate, so
/// the settled reply streams into plain scrollback instead of landing at once.
fn type_to_scrollback(s: &str) {
    let mut out = std::io::stdout();
    let mut buf = String::new();
    let mut pending = 0usize;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        buf.push(c);
        if c == '\x1b' {
            // Pass the escape through atomically, up to its final letter.
            while let Some(&n) = chars.peek() {
                buf.push(n);
                chars.next();
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        pending += 1;
        if pending >= TYPE_GROUP || c == '\n' {
            let _ = write!(out, "{buf}");
            let _ = out.flush();
            std::thread::sleep(Duration::from_millis(TYPE_DELAY_MS));
            buf.clear();
            pending = 0;
        }
    }
    if !buf.is_empty() {
        let _ = write!(out, "{buf}");
        let _ = out.flush();
    }
}

/// Terminal render state: a live preview in a reserved bottom pane plus an
/// authoritative print at `finish`.
///
/// The forming answer/reasoning streams into a fixed [`Viewport`] pane. The
/// stream's stable-prefix tail is speculative and can still be revised, so it
/// lives only in the pane. Text behind [`flush_cut`] is final, and a
/// `progressive` turn prints those whole lines into real scrollback as they
/// settle, keeping only the still-converging tail in the pane. `finish`
/// prints whatever the authoritative reply adds beyond the flushed prefix
/// (all of it, on a non-progressive turn). When there is no viewport (no
/// cursor report, or a non-tty), the preview degrades to a single spinner
/// line and the reply still prints whole at `finish`.
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
    /// Flush committed settled lines into scrollback mid-turn. Off for tool
    /// rounds, whose settled text is reshaped at round end (salvage, preamble
    /// extraction), so their preview stays pane-only.
    progressive: bool,
    /// Pace the display: reveal the answer at a bounded rate per repaint so it
    /// types out smoothly even when the model lands a whole block at once (the
    /// held tail finishes typing in `finish`). Off shows every committed byte
    /// immediately.
    paced: bool,
    /// Bytes of `text` currently allowed on screen while `paced`, advanced toward
    /// `text.len()` each repaint. Unused when pacing is off.
    revealed: usize,
    /// Committed byte prefix of `text` (from the latest `Text` event).
    committed: usize,
    /// Answer bytes already printed into scrollback; always a prefix of the
    /// `text` they were cut from, ending at a newline.
    flushed: String,
    /// `text` stopped extending `flushed` (a rare committed-text revision):
    /// flushing stops and `finish` reconciles against the authoritative reply.
    frozen: bool,
    /// The `think>` block printed ahead of the first flushed answer line, so
    /// `finish` prints only what the thought gained since (usually nothing).
    thought_flushed: String,
}

impl Render {
    fn new(
        show_thinking: bool,
        stream_output: bool,
        interactive: bool,
        progressive: bool,
        paced: bool,
    ) -> Self {
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
            progressive: progressive && stream_output,
            paced: paced && stream_output && interactive,
            revealed: 0,
            committed: 0,
            flushed: String::new(),
            frozen: false,
            thought_flushed: String::new(),
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
            ChatEvent::Text { committed, text } => {
                self.answer_started = true;
                self.committed = *committed;
                self.text = text.clone();
                // A revision may shrink the text under the reveal cursor; never
                // let it point past the end.
                self.revealed = self.revealed.min(self.text.len());
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

    /// The answer text currently allowed on screen: capped to the reveal cursor
    /// while paced (so it types out), the whole of `text` otherwise.
    fn visible(&self) -> &str {
        if self.paced {
            &self.text[..self.revealed.min(self.text.len())]
        } else {
            &self.text
        }
    }

    /// Advance the reveal cursor one repaint's worth toward `text.len()`, snapped
    /// up to a char boundary. A no-op unless pacing is on and text remains.
    fn advance_reveal(&mut self) {
        if !self.paced || self.revealed >= self.text.len() {
            return;
        }
        self.revealed += reveal_step(self.text.len() - self.revealed);
        while self.revealed < self.text.len() && !self.text.is_char_boundary(self.revealed) {
            self.revealed += 1;
        }
    }

    /// Type out the remainder of a paced answer before it settles: adopt the
    /// authoritative reply (it holds the final block the stream never revealed)
    /// and repaint at the tick cadence until the reveal cursor catches up. A
    /// no-op unless paced with a viewport and the reply extends what streamed; a
    /// revision falls through to `finish`'s correction path.
    fn drain_reveal(&mut self, reply: Option<&str>) {
        let Some(r) = reply else { return };
        if !self.paced || self.viewport.is_none() || !r.starts_with(self.flushed.as_str()) {
            return;
        }
        self.answer_started = true;
        self.prefill = false;
        self.text = r.to_string();
        self.committed = r.len();
        self.revealed = self.revealed.min(self.text.len());
        while self.revealed < self.text.len() {
            self.paint(); // advances the reveal cursor, then redraws
            std::thread::sleep(Duration::from_millis(55));
        }
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
            // After a progressive flush the pane holds only the unflushed
            // tail, markerless: it continues the flushed lines above. A
            // frozen turn falls back to previewing the whole text.
            match self.visible().strip_prefix(self.flushed.as_str()) {
                Some(tail) if !self.flushed.is_empty() => ("", tail.trim(), DIM),
                _ => ("model> ", self.visible().trim(), DIM),
            }
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

    /// Flush newly settled committed lines into scrollback (progressive turns
    /// with a viewport only). The one rare failure mode, `text` no longer
    /// extending what was flushed, freezes flushing for the turn instead of
    /// printing a revision.
    fn flush_committed(&mut self) {
        if !self.progressive || self.frozen || self.viewport.is_none() || !self.answer_started {
            return;
        }
        if !self.text.starts_with(self.flushed.as_str()) {
            self.frozen = true;
            return;
        }
        // While paced, only flush whole lines the reveal cursor has reached, so
        // scrollback fills at the same typed rate as the pane.
        let cap = if self.paced {
            self.revealed
        } else {
            self.text.len()
        };
        let cut = flush_cut(&self.text, self.committed.min(cap));
        if cut <= self.flushed.len() {
            return;
        }
        let mut chunk = String::new();
        if self.flushed.is_empty() {
            // The settled reasoning precedes the first answer line, exactly
            // where `finish` would have put it.
            if self.show_thinking {
                let t = self.thought.trim();
                if !t.is_empty() {
                    chunk.push_str(&format!("{GREY}think> {t}{UNDIM}\n"));
                    self.thought_flushed = t.to_string();
                }
            }
            chunk.push_str("model> ");
        }
        chunk.push_str(&self.text[self.flushed.len()..cut]);
        self.flushed = self.text[..cut].to_string();
        if let Some(vp) = &self.viewport {
            vp.flush_permanent(&chunk);
        }
    }

    /// Repaint. Sole stdout writer during a turn.
    fn paint(&mut self) {
        let frame = SPINNER[self.spinner % SPINNER.len()];
        self.spinner = self.spinner.wrapping_add(1);
        self.advance_reveal();
        self.flush_committed();
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
    /// calling. A progressive turn already flushed a prefix of the reply (and
    /// its `think>` block), so only the remainder prints here. An
    /// authoritative reply that does not extend the flushed prefix (the rare
    /// committed-text revision) reprints whole under a correction notice.
    fn finish(&mut self, reply: Option<&str>, stats: Option<&str>) {
        let mut permanent = String::new();
        if self.show_thinking {
            let t = self.thought.trim();
            if !t.is_empty() && self.thought_flushed.is_empty() {
                permanent.push_str(&format!("{GREY}think> {t}{UNDIM}\n"));
            } else if let Some(gained) = t.strip_prefix(self.thought_flushed.as_str())
                && !gained.trim().is_empty()
            {
                permanent.push_str(&format!("{GREY}think> {}{UNDIM}\n", gained.trim()));
            }
        }
        match reply {
            Some(r) if self.flushed.is_empty() => {
                if !r.is_empty() {
                    permanent.push_str(&format!("model> {r}\n"));
                } else {
                    permanent.push_str("model> (empty response)\n");
                }
            }
            Some(r) => match r.strip_prefix(self.flushed.as_str()) {
                Some(rest) => {
                    if !rest.is_empty() {
                        permanent.push_str(&format!("{rest}\n"));
                    }
                }
                None => permanent.push_str(&format!(
                    "{DIM}(the streamed reply above was revised, final version below){UNDIM}\nmodel> {r}\n"
                )),
            },
            None => {}
        }
        if let Some(line) = stats {
            permanent.push_str(&format!("  {line}\n"));
        }
        let paced = self.paced;
        match self.viewport.as_mut() {
            Some(vp) => vp.commit(&permanent),
            // No reserved pane (e.g. a terminal that never answered the cursor
            // query): a paced turn types the settled reply into scrollback
            // instead of dropping it whole; otherwise print it at once.
            None => {
                print!("\r\x1b[2K");
                let _ = std::io::stdout().flush();
                if paced {
                    type_to_scrollback(&permanent);
                } else {
                    print!("{permanent}");
                }
            }
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
    let mut r = Render::new(true, true, true, true, false);
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
        // All but the newest line reads as block-committed, so the demo
        // exercises the progressive scrollback flush.
        let committed = text.rfind('\n').map_or(0, |n| n + 1);
        r.apply(&ChatEvent::Text { committed, text });
        r.paint();
        std::thread::sleep(Duration::from_millis(120));
    }
    // The authoritative reply extends the streamed text (the normal case), so
    // finish prints only the last, never-flushed line.
    let reply = (0..6)
        .map(|j| format!("answer line {j}: content long enough that it might overflow the pane"))
        .collect::<Vec<_>>()
        .join("\n");
    r.finish(
        Some(&reply),
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
    /// Flush committed settled lines into real scrollback as the answer forms
    /// (plain turns). Off for tool rounds: their settled text is reshaped at
    /// round end (salvage, preamble extraction) and, under the opt-in
    /// validator/repair stages, a whole streamed draft can evaporate. The
    /// pane-only preview already shows those rounds streaming.
    pub progressive: bool,
    /// Dribble each committed block's answer out over the next block's denoise so
    /// it types out smoothly instead of landing a block at a time. Serve's paced
    /// streaming, in the terminal.
    pub paced: bool,
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
            .with_thinking(show_thinking, display.prethink_seed)
            .paced(display.paced);
        if display.start_in_thought {
            decoder = decoder.starting_in_thought();
        }
        let shared = Arc::new(Mutex::new(Shared {
            decoder,
            sinks,
            render: Render::new(
                show_thinking,
                display.stream_output,
                interactive,
                display.progressive,
                display.paced,
            ),
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
            // Type out any paced remainder (the held final block) before settling.
            s.render.drain_reveal(reply);
            s.render.finish(reply, stats);
        }
    }
}
