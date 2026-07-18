//! Process- and machine-wide memory budget for model-loading code paths.
//!
//! The suite historically ran `--test-threads=1` because N runtimes × ~20 GiB
//! cannot coexist on a 36 GB machine, and "never two model-loading processes"
//! was enforced only by discipline (pgrep before timing). This module turns
//! both rules into a byte-denominated permit:
//!
//! - [`MemBudget::global`] holds the machine budget (`DGQ_TEST_MEM_BUDGET`
//!   bytes, default physical RAM minus headroom).
//! - [`MemBudget::acquire`] blocks (FIFO) until `bytes` fit, then returns a
//!   [`MemPermit`] that releases on `Drop`.
//! - Cross-process: each permit holds an `flock(EX)` on a size-stamped file in
//!   `DGQ_MEM_LOCK_DIR`; other processes count live-locked files against the
//!   budget. The OS drops the flock on any death, so stale holdings self-heal.
//!
//! Misuse resistance: the permit is acquired INSIDE the runtime builder (the
//! one place that allocates model-scale memory) and stored in the runtime, so
//! release is tied to the allocation's own lifetime. Nothing outside this
//! module and that builder ever handles a permit. A nested acquire on one
//! thread panics (it would deadlock once budgets bind) — a second runtime on
//! the same thread must be built after dropping the first.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

/// Non-budget headroom left for the OS + everything untracked (Metal shader
/// heaps, tokenizer, test scaffolding).
const DEFAULT_HEADROOM_BYTES: usize = 6 << 30;

/// Poll interval while waiting on memory held by OTHER processes (in-process
/// waits use a condvar and don't poll).
const CROSS_PROCESS_POLL: Duration = Duration::from_millis(200);

/// Default [`MemBudget::acquire`] wait deadline. Long enough for any healthy
/// runtime turnover on this machine (the whole suite is ~9 min), short enough
/// to surface a wedged or forgotten holder (an idle `serve` next to the
/// suite) as an error naming the holder instead of an indefinite silent
/// stall. `DGQ_MEM_LOCK_TIMEOUT` seconds overrides; `0` waits forever.
const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(600);

struct State {
    used: usize,
    /// Live in-process grants `(ticket, label, bytes)` — named in the
    /// timeout error so the blocker is identifiable without a debugger.
    grants: Vec<(u64, String, usize)>,
    /// FIFO tickets: head-of-line acquires first, so one large request cannot
    /// be starved by a stream of small ones.
    queue: std::collections::VecDeque<u64>,
    next_ticket: u64,
}

struct Inner {
    total: usize,
    state: Mutex<State>,
    cv: Condvar,
    lock_dir: Option<PathBuf>,
    /// Acquire deadline; `None` waits forever.
    timeout: Option<Duration>,
}

/// `acquire` gave up waiting. Carries a holder listing for the error message.
#[derive(Debug)]
pub struct MemWaitTimeout {
    pub waited: Duration,
    pub wanted: usize,
    /// One line per live holder: in-process grants and other processes'
    /// lock files (`pid-label.mem`).
    pub holders: Vec<String>,
}

impl std::fmt::Display for MemWaitTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "timed out after {:.0}s waiting for a {:.1} GiB memory grant; live holders: [{}] \
             (kill the holder, raise DGQ_TEST_MEM_BUDGET, or set DGQ_MEM_LOCK_TIMEOUT=0 to wait)",
            self.waited.as_secs_f64(),
            self.wanted as f64 / (1024.0 * 1024.0 * 1024.0),
            if self.holders.is_empty() {
                "none visible — budget smaller than the request while busy".to_string()
            } else {
                self.holders.join(", ")
            }
        )
    }
}

pub struct MemBudget {
    inner: Arc<Inner>,
}

/// A granted slice of the budget. Dropping it releases the bytes (and the
/// cross-process lock file). Not `Clone`: one grant, one release.
pub struct MemPermit {
    bytes: usize,
    /// Grant id in `State::grants` (removed on drop).
    ticket: u64,
    inner: Arc<Inner>,
    /// Held `flock(EX)` advertising this grant to other processes.
    lock_file: Option<(fs::File, PathBuf)>,
    /// Thread that acquired (and armed the nested-acquire tripwire). A permit
    /// dropped on a DIFFERENT thread leaves the owner's tripwire armed — the
    /// next acquire there panics loudly rather than risking a silent nested
    /// deadlock; runtimes are built and dropped on one thread today.
    owner: std::thread::ThreadId,
}

std::thread_local! {
    /// Nested-acquire tripwire (see module doc).
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn env_bytes(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

impl MemBudget {
    /// The machine-wide budget. `DGQ_TEST_MEM_BUDGET` (bytes) overrides;
    /// default is physical RAM minus a fixed headroom. `DGQ_MEM_LOCK_DIR`
    /// overrides the cross-process lock directory ("" disables the layer).
    pub fn global() -> &'static MemBudget {
        static GLOBAL: OnceLock<MemBudget> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            let ram = crate::metal::memwatch::physical_ram_bytes() as usize;
            let total = env_bytes("DGQ_TEST_MEM_BUDGET")
                .unwrap_or_else(|| ram.saturating_sub(DEFAULT_HEADROOM_BYTES).max(1 << 30));
            let lock_dir = match std::env::var("DGQ_MEM_LOCK_DIR") {
                Ok(s) if s.is_empty() => None,
                Ok(s) => Some(PathBuf::from(s)),
                Err(_) => Some(std::env::temp_dir().join("diffgemma-memlock")),
            };
            let timeout = match env_bytes("DGQ_MEM_LOCK_TIMEOUT") {
                Some(0) => None,
                Some(secs) => Some(Duration::from_secs(secs as u64)),
                None => Some(DEFAULT_ACQUIRE_TIMEOUT),
            };
            MemBudget::with_timeout(total, lock_dir, timeout)
        })
    }

    /// In-process-or-given-dir budget that waits forever (test constructor;
    /// the global gets the env-configured timeout).
    pub fn new(total: usize, lock_dir: Option<PathBuf>) -> Self {
        Self::with_timeout(total, lock_dir, None)
    }

    pub fn with_timeout(
        total: usize,
        lock_dir: Option<PathBuf>,
        timeout: Option<Duration>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                total,
                state: Mutex::new(State {
                    used: 0,
                    grants: Vec::new(),
                    queue: std::collections::VecDeque::new(),
                    next_ticket: 0,
                }),
                cv: Condvar::new(),
                lock_dir,
                timeout,
            }),
        }
    }

    /// Block until `bytes` fit (FIFO among in-process waiters, polling against
    /// other processes), then grant. Oversized requests (> total) are granted
    /// when the budget is otherwise idle rather than deadlocking — the budget
    /// is a scheduler, not a hard allocator, and refusing would turn an
    /// under-provisioned machine into a hang. Gives up after the budget's
    /// timeout (global: `DGQ_MEM_LOCK_TIMEOUT`, default 600 s) with an error
    /// naming the live holders — a wedged or forgotten holder should be a
    /// loud failure, not an indefinite stall.
    pub fn acquire(&self, bytes: usize, label: &str) -> Result<MemPermit, MemWaitTimeout> {
        HOLDING.with(|h| {
            assert!(
                !h.get(),
                "membudget: nested acquire on one thread ({label}) — this deadlocks \
                 once budgets bind; drop the first runtime before building a second"
            );
            h.set(true);
        });
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        let ticket = st.next_ticket;
        st.next_ticket += 1;
        st.queue.push_back(ticket);
        let wait_started = std::time::Instant::now();
        let mut announced = false;
        loop {
            let external = inner.lock_dir.as_deref().map_or(0, external_used);
            let fits = st.used + external + bytes <= inner.total || (st.used == 0 && external == 0); // oversized-when-idle
            if st.queue.front() == Some(&ticket) && fits {
                break;
            }
            if let Some(limit) = inner.timeout
                && wait_started.elapsed() >= limit
            {
                st.queue.retain(|&t| t != ticket);
                // A later ticket may be head-of-line now.
                inner.cv.notify_all();
                let mut holders: Vec<String> = st
                    .grants
                    .iter()
                    .map(|(_, l, b)| format!("{l} ({:.1} GiB, this process)", gib(*b)))
                    .collect();
                if let Some(dir) = inner.lock_dir.as_deref() {
                    holders.extend(external_holders(dir));
                }
                drop(st);
                HOLDING.with(|h| h.set(false));
                return Err(MemWaitTimeout {
                    waited: wait_started.elapsed(),
                    wanted: bytes,
                    holders,
                });
            }
            if !announced && wait_started.elapsed() > Duration::from_secs(2) {
                announced = true;
                eprintln!(
                    "membudget: {label} waiting for {:.1} GiB (in-process {:.1} GiB, \
                     other processes {:.1} GiB, budget {:.1} GiB)",
                    gib(bytes),
                    gib(st.used),
                    gib(external),
                    gib(inner.total)
                );
            }
            // Cross-process holdings (and the deadline) can change without
            // waking our condvar — always wake to re-poll.
            let poll = match inner.timeout {
                Some(limit) => CROSS_PROCESS_POLL.min(limit.saturating_sub(wait_started.elapsed())),
                None if inner.lock_dir.is_some() => CROSS_PROCESS_POLL,
                None => {
                    st = inner.cv.wait(st).unwrap();
                    continue;
                }
            };
            let (g, _) = inner
                .cv
                .wait_timeout(st, poll.max(Duration::from_millis(1)))
                .unwrap();
            st = g;
        }
        st.queue.pop_front();
        st.used += bytes;
        st.grants.push((ticket, label.to_string(), bytes));
        // Wake siblings: the next ticket may also fit alongside this one.
        inner.cv.notify_all();
        drop(st);
        let lock_file = inner
            .lock_dir
            .as_ref()
            .and_then(|d| advertise(d, bytes, label));
        Ok(MemPermit {
            bytes,
            ticket,
            inner: Arc::clone(inner),
            lock_file,
            owner: std::thread::current().id(),
        })
    }
}

fn gib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

impl MemPermit {
    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for MemPermit {
    fn drop(&mut self) {
        if let Some((file, path)) = self.lock_file.take() {
            // Unlink before releasing the flock so scanners never see an
            // unlocked-but-present file as a dead holder they must clean up.
            let _ = fs::remove_file(&path);
            drop(file);
        }
        let mut st = self.inner.state.lock().unwrap();
        st.used = st.used.saturating_sub(self.bytes);
        st.grants.retain(|&(t, _, _)| t != self.ticket);
        self.inner.cv.notify_all();
        if std::thread::current().id() == self.owner {
            HOLDING.with(|h| h.set(false));
        }
    }
}

/// Create a size-stamped, `flock(EX)`-held file advertising a grant. Best
/// effort: on any error the cross-process layer degrades to in-process-only.
fn advertise(dir: &std::path::Path, bytes: usize, label: &str) -> Option<(fs::File, PathBuf)> {
    fs::create_dir_all(dir).ok()?;
    let path = dir.join(format!(
        "{}-{}.mem",
        std::process::id(),
        sanitize_label(label)
    ));
    let mut f = fs::File::create(&path).ok()?;
    flock_ex_nonblock(&f)?;
    writeln!(f, "{bytes} {label}").ok()?;
    f.flush().ok()?;
    Some((f, path))
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(40)
        .collect()
}

/// Sum the byte stamps of OTHER processes' still-flocked files. A file whose
/// `flock(SH)` we can take belongs to a dead process: unlink and skip it.
fn external_used(dir: &std::path::Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let me = format!("{}-", std::process::id());
    let mut sum = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "mem") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(&me))
        {
            continue; // our own grants are already in `used`
        }
        let Ok(f) = fs::File::open(&path) else {
            continue;
        };
        if flock_sh_nonblock(&f).is_some() {
            // Holder died; the OS released its flock. Self-heal.
            let _ = fs::remove_file(&path);
            continue;
        }
        if let Ok(s) = fs::read_to_string(&path)
            && let Some(n) = s
                .split_whitespace()
                .next()
                .and_then(|w| w.parse::<usize>().ok())
        {
            sum += n;
        }
    }
    sum
}

/// Describe OTHER processes' live grants for the timeout error: one line per
/// still-flocked file, `pid <pid>: <label> (<GiB>)`. Dead holders are skipped
/// (and reaped by the regular scan), not blamed.
fn external_holders(dir: &std::path::Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let me = format!("{}-", std::process::id());
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "mem") {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with(&me) {
            continue;
        }
        let Ok(f) = fs::File::open(&path) else {
            continue;
        };
        if flock_sh_nonblock(&f).is_some() {
            continue; // dead holder
        }
        let pid = name.split('-').next().unwrap_or("?");
        if let Ok(s) = fs::read_to_string(&path) {
            let mut words = s.split_whitespace();
            let bytes = words.next().and_then(|w| w.parse::<usize>().ok());
            let label = words.next().unwrap_or("?");
            out.push(match bytes {
                Some(b) => format!("pid {pid}: {label} ({:.1} GiB)", gib(b)),
                None => format!("pid {pid}: {label}"),
            });
        }
    }
    out
}

// Direct FFI: the crate is macOS-only and keeps its dependency list minimal
// (no libc). BSD flock(2) constants.
const LOCK_SH: i32 = 1;
const LOCK_EX: i32 = 2;
const LOCK_NB: i32 = 4;
unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

fn flock_ex_nonblock(f: &fs::File) -> Option<()> {
    flock_op(f, LOCK_EX | LOCK_NB)
}

fn flock_sh_nonblock(f: &fs::File) -> Option<()> {
    flock_op(f, LOCK_SH | LOCK_NB)
}

fn flock_op(f: &fs::File, op: i32) -> Option<()> {
    use std::os::fd::AsRawFd;
    (unsafe { flock(f.as_raw_fd(), op) } == 0).then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn budget(total: usize) -> MemBudget {
        MemBudget::new(total, None)
    }

    #[test]
    fn serial_grants_within_budget_do_not_block() {
        let b = budget(100);
        let p1 = b.acquire(40, "a").unwrap();
        drop(p1);
        let p2 = b.acquire(100, "b").unwrap();
        drop(p2);
    }

    #[test]
    fn oversized_request_grants_when_idle() {
        let b = budget(10);
        let p = b.acquire(50, "huge").unwrap();
        assert_eq!(p.bytes(), 50);
    }

    #[test]
    fn concurrent_grants_never_exceed_total() {
        let b = Arc::new(budget(100));
        let peak = Arc::new(AtomicUsize::new(0));
        let cur = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for i in 0..16 {
            let (b, peak, cur) = (Arc::clone(&b), Arc::clone(&peak), Arc::clone(&cur));
            handles.push(std::thread::spawn(move || {
                for _ in 0..20 {
                    let p = b.acquire(30 + (i % 3) * 10, "t").unwrap();
                    let now = cur.fetch_add(p.bytes(), Ordering::SeqCst) + p.bytes();
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::yield_now();
                    cur.fetch_sub(p.bytes(), Ordering::SeqCst);
                    drop(p);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(
            peak.load(Ordering::SeqCst) <= 100,
            "peak {} exceeded budget",
            peak.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn fifo_head_of_line_large_request_is_not_starved() {
        // One large waiter behind a granted small one must beat later smalls.
        let b = Arc::new(budget(100));
        let first = b.acquire(80, "first").unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let big = {
            let (b, order) = (Arc::clone(&b), Arc::clone(&order));
            std::thread::spawn(move || {
                let _p = b.acquire(100, "big").unwrap();
                order.lock().unwrap().push("big");
            })
        };
        // Give `big` time to enqueue ahead of the smalls.
        std::thread::sleep(Duration::from_millis(50));
        let smalls: Vec<_> = (0..3)
            .map(|_| {
                let (b, order) = (Arc::clone(&b), Arc::clone(&order));
                std::thread::spawn(move || {
                    let _p = b.acquire(10, "small").unwrap();
                    order.lock().unwrap().push("small");
                })
            })
            .collect();
        std::thread::sleep(Duration::from_millis(50));
        drop(first);
        big.join().unwrap();
        for s in smalls {
            s.join().unwrap();
        }
        assert_eq!(
            order.lock().unwrap().first(),
            Some(&"big"),
            "large head-of-line waiter was starved by later small requests"
        );
    }

    #[test]
    fn timeout_errors_name_the_holder_and_leave_budget_healthy() {
        let b = Arc::new(MemBudget::with_timeout(
            100,
            None,
            Some(Duration::from_millis(100)),
        ));
        let first = b.acquire(80, "holder-a").unwrap();
        let (timed_out_tx, timed_out_rx) = std::sync::mpsc::channel();
        let waiter = {
            let b = Arc::clone(&b);
            std::thread::spawn(move || {
                let err = match b.acquire(80, "waiter") {
                    Ok(_) => panic!("acquire should have timed out"),
                    Err(e) => e,
                };
                timed_out_tx.send(err).unwrap();
                // The failed wait must not wedge the queue or THIS thread's
                // tripwire: once the holder releases, the same thread can
                // acquire again.
                b.acquire(80, "retry").map(|p| p.bytes())
            })
        };
        let err = timed_out_rx.recv().unwrap();
        assert!(err.waited >= Duration::from_millis(100));
        assert_eq!(err.wanted, 80);
        let msg = err.to_string();
        assert!(msg.contains("holder-a"), "holder not named: {msg}");
        drop(first);
        assert_eq!(waiter.join().unwrap().unwrap(), 80);
    }

    #[test]
    fn timeout_does_not_fire_when_grant_arrives_in_time() {
        let b = Arc::new(MemBudget::with_timeout(
            100,
            None,
            Some(Duration::from_secs(30)),
        ));
        let first = b.acquire(100, "short-holder").unwrap();
        let waiter = {
            let b = Arc::clone(&b);
            std::thread::spawn(move || b.acquire(50, "patient").map(|p| p.bytes()))
        };
        std::thread::sleep(Duration::from_millis(50));
        drop(first);
        assert_eq!(waiter.join().unwrap().unwrap(), 50);
    }

    #[test]
    #[should_panic(expected = "nested acquire")]
    fn nested_acquire_panics() {
        let b = budget(100);
        let _p1 = b.acquire(10, "outer").unwrap();
        let _p2 = b.acquire(10, "inner");
    }

    #[test]
    fn cross_process_layer_counts_live_locks_and_reaps_dead_ones() {
        let dir = std::env::temp_dir().join(format!("dgq-membudget-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        // A live foreign holding: fake another pid so it is not skipped as ours.
        fs::create_dir_all(&dir).unwrap();
        let foreign = dir.join("999999999-live.mem");
        let f = fs::File::create(&foreign).unwrap();
        flock_ex_nonblock(&f).unwrap();
        {
            use std::io::Write;
            let mut fw = fs::File::options().write(true).open(&foreign).unwrap();
            writeln!(fw, "60 live").unwrap();
        }
        assert_eq!(external_used(&dir), 60);
        // A dead holding: present but unlocked. Scanning reaps it.
        let dead = dir.join("999999998-dead.mem");
        fs::write(&dead, "40 dead\n").unwrap();
        assert_eq!(external_used(&dir), 60, "dead holder must not count");
        assert!(!dead.exists(), "dead holder file must be reaped");
        drop(f);
        let _ = fs::remove_dir_all(&dir);
    }
}
