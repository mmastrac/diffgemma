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

struct State {
    used: usize,
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
}

pub struct MemBudget {
    inner: Arc<Inner>,
}

/// A granted slice of the budget. Dropping it releases the bytes (and the
/// cross-process lock file). Not `Clone`: one grant, one release.
pub struct MemPermit {
    bytes: usize,
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
            MemBudget::new(total, lock_dir)
        })
    }

    pub fn new(total: usize, lock_dir: Option<PathBuf>) -> Self {
        Self {
            inner: Arc::new(Inner {
                total,
                state: Mutex::new(State {
                    used: 0,
                    queue: std::collections::VecDeque::new(),
                    next_ticket: 0,
                }),
                cv: Condvar::new(),
                lock_dir,
            }),
        }
    }

    /// Block until `bytes` fit (FIFO among in-process waiters, polling against
    /// other processes), then grant. Oversized requests (> total) are granted
    /// when the budget is otherwise idle rather than deadlocking — the budget
    /// is a scheduler, not a hard allocator, and refusing would turn an
    /// under-provisioned machine into a hang.
    pub fn acquire(&self, bytes: usize, label: &str) -> MemPermit {
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
            if !announced && wait_started.elapsed() > Duration::from_secs(2) {
                announced = true;
                let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
                eprintln!(
                    "membudget: {label} waiting for {:.1} GiB (in-process {:.1} GiB, \
                     other processes {:.1} GiB, budget {:.1} GiB)",
                    gib(bytes),
                    gib(st.used),
                    gib(external),
                    gib(inner.total)
                );
            }
            if inner.lock_dir.is_some() {
                // Cross-process holdings can change without waking our condvar.
                let (g, _) = inner.cv.wait_timeout(st, CROSS_PROCESS_POLL).unwrap();
                st = g;
            } else {
                st = inner.cv.wait(st).unwrap();
            }
        }
        st.queue.pop_front();
        st.used += bytes;
        // Wake siblings: the next ticket may also fit alongside this one.
        inner.cv.notify_all();
        drop(st);
        let lock_file = inner
            .lock_dir
            .as_ref()
            .and_then(|d| advertise(d, bytes, label));
        MemPermit {
            bytes,
            inner: Arc::clone(inner),
            lock_file,
            owner: std::thread::current().id(),
        }
    }
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
        let p1 = b.acquire(40, "a");
        drop(p1);
        let p2 = b.acquire(100, "b");
        drop(p2);
    }

    #[test]
    fn oversized_request_grants_when_idle() {
        let b = budget(10);
        let p = b.acquire(50, "huge");
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
                    let p = b.acquire(30 + (i % 3) * 10, "t");
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
        let first = b.acquire(80, "first");
        let order = Arc::new(Mutex::new(Vec::new()));
        let big = {
            let (b, order) = (Arc::clone(&b), Arc::clone(&order));
            std::thread::spawn(move || {
                let _p = b.acquire(100, "big");
                order.lock().unwrap().push("big");
            })
        };
        // Give `big` time to enqueue ahead of the smalls.
        std::thread::sleep(Duration::from_millis(50));
        let smalls: Vec<_> = (0..3)
            .map(|_| {
                let (b, order) = (Arc::clone(&b), Arc::clone(&order));
                std::thread::spawn(move || {
                    let _p = b.acquire(10, "small");
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
    #[should_panic(expected = "nested acquire")]
    fn nested_acquire_panics() {
        let b = budget(100);
        let _p1 = b.acquire(10, "outer");
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
