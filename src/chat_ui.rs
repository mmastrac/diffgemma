//! Live chat rendering: thinking spinner + stable-prefix token streaming.
//!
//! Streaming rule: a canvas position's argmax counts as *stable* once it has
//! held the same value for 2 consecutive denoise steps. The longest stable
//! prefix of the answer (cut at the first stop token) is decoded and printed
//! incrementally. Under the MLX-exact sampler the block's final commit IS the
//! last step's argmax, so this preview converges to the real reply; printing
//! is monotone (text is only ever appended when the new decode extends what
//! is already on screen) and `finish()` reconciles against the authoritative
//! final reply.

use crate::metal::{StepObserver, StepProgressEvent};
use crate::tokenizer::Tokenizer;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Consecutive-step repeats required before a position streams (3 sightings —
/// 2 proved too eager on fast-converging replies).
const STABLE_STREAK: u32 = 2;

struct StreamInner {
    last_argmax: Vec<u32>,
    streak: Vec<u32>,
    block_idx: usize,
    /// Final ids of committed blocks (already cut at the first stop token).
    committed_ids: Vec<u32>,
    /// Stop token seen — no further text will be shown.
    ended: bool,
    /// Text already written to stdout (after "model> ").
    printed: String,
    started_text: bool,
    status: String,
}

pub struct ChatStream {
    inner: Arc<Mutex<StreamInner>>,
    done: Arc<AtomicBool>,
    tokenizer: Arc<Tokenizer>,
    stop_token_ids: Vec<u32>,
    interactive: bool,
    ticker: Option<std::thread::JoinHandle<()>>,
}

fn clear_line() {
    print!("\r\x1b[2K");
}

impl ChatStream {
    /// `interactive=false` (piped stdout / --verbose) disables the spinner and
    /// streaming; only the final reply is printed by `finish`.
    pub fn start(
        tokenizer: Arc<Tokenizer>,
        stop_token_ids: Vec<u32>,
        interactive: bool,
    ) -> Self {
        let inner = Arc::new(Mutex::new(StreamInner {
            last_argmax: Vec::new(),
            streak: Vec::new(),
            block_idx: 0,
            committed_ids: Vec::new(),
            ended: false,
            printed: String::new(),
            started_text: false,
            status: "prefilling…".to_string(),
        }));
        let done = Arc::new(AtomicBool::new(false));

        let ticker = if interactive {
            let inner_t = Arc::clone(&inner);
            let done_t = Arc::clone(&done);
            Some(std::thread::spawn(move || {
                let mut frame = 0usize;
                loop {
                    if done_t.load(Ordering::Relaxed) {
                        break;
                    }
                    {
                        let st = inner_t.lock().unwrap();
                        if st.started_text {
                            break; // text is streaming; spinner retired
                        }
                        clear_line();
                        print!(
                            "{} thinking · {}",
                            SPINNER_FRAMES[frame % SPINNER_FRAMES.len()],
                            st.status
                        );
                        let _ = std::io::stdout().flush();
                    }
                    frame += 1;
                    std::thread::sleep(std::time::Duration::from_millis(120));
                }
            }))
        } else {
            None
        };

        Self {
            inner,
            done,
            tokenizer,
            stop_token_ids,
            interactive,
            ticker,
        }
    }

    /// The observer to install as `StepGenerateConfig::step_observer`.
    pub fn observer(&self) -> StepObserver {
        let inner = Arc::clone(&self.inner);
        let tokenizer = Arc::clone(&self.tokenizer);
        let stops = self.stop_token_ids.clone();
        let interactive = self.interactive;
        Arc::new(move |ev: &StepProgressEvent<'_>| {
            let mut st = inner.lock().unwrap();
            if st.ended {
                return;
            }
            if ev.block_idx != st.block_idx {
                st.block_idx = ev.block_idx;
                st.last_argmax.clear();
                st.streak.clear();
            }
            if st.last_argmax.len() != ev.argmax.len() {
                st.last_argmax = ev.argmax.to_vec();
                st.streak = vec![0; ev.argmax.len()];
            } else {
                for i in 0..ev.argmax.len() {
                    if ev.argmax[i] == st.last_argmax[i] {
                        st.streak[i] = st.streak[i].saturating_add(1);
                    } else {
                        st.streak[i] = 0;
                        st.last_argmax[i] = ev.argmax[i];
                    }
                }
            }
            st.status = format!(
                "step {}/{} · {}/{} locked",
                ev.step_in_block,
                ev.max_steps,
                ev.accept_count,
                ev.argmax.len()
            );

            // Display ids = committed blocks + this block's stable prefix
            // (block_done: the whole block is final), cut at the first stop.
            let mut ids = st.committed_ids.clone();
            let prefix_end = if ev.block_done {
                ev.argmax.len()
            } else {
                st.streak
                    .iter()
                    .position(|&k| k < STABLE_STREAK)
                    .unwrap_or(ev.argmax.len())
            };
            let mut hit_stop = false;
            for &id in &ev.argmax[..prefix_end] {
                if stops.contains(&id) {
                    hit_stop = true;
                    break;
                }
                ids.push(id);
            }
            if ev.block_done {
                st.committed_ids = ids.clone();
                st.last_argmax.clear();
                st.streak.clear();
                if hit_stop {
                    st.ended = true;
                }
            }

            if !interactive {
                return;
            }
            let shown = crate::sample::strip_degenerate_token_ids(&ids);
            if shown.is_empty() {
                return;
            }
            let text = crate::chat_template::sanitize_model_reply(&tokenizer.decode(&shown));
            if text.len() > st.printed.len() && text.starts_with(&st.printed) {
                let delta = text[st.printed.len()..].to_string();
                if !st.started_text {
                    clear_line();
                    print!("model> ");
                    st.started_text = true;
                }
                print!("{delta}");
                let _ = std::io::stdout().flush();
                st.printed = text;
            }
        })
    }

    /// Reconcile the streamed preview against the authoritative final reply
    /// and terminate the line. Returns nothing; prints the remainder (or the
    /// whole reply when nothing streamed / the preview diverged).
    pub fn finish(mut self, reply: &str) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.ticker.take() {
            let _ = t.join();
        }
        let st = self.inner.lock().unwrap();
        if self.interactive && !st.started_text {
            clear_line();
        }
        if reply.is_empty() {
            println!("model> (empty response)");
            return;
        }
        if st.started_text {
            if reply.starts_with(&st.printed) {
                println!("{}", &reply[st.printed.len()..]);
            } else {
                // The preview diverged from the final commit (rare: a late
                // argmax revision). Erase the draft (cursor-up per printed
                // newline; wrapped long lines may leave residue — acceptable)
                // and reprint authoritatively.
                clear_line();
                for _ in 0..st.printed.matches('\n').count() {
                    print!("\x1b[1A\x1b[2K");
                }
                print!("\r");
                println!("model> {reply}");
            }
        } else {
            println!("model> {reply}");
        }
    }
}
