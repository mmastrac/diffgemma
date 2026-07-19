//! `programmatic` battery — generate a program, EXECUTE it, judge stdout and
//! exit code.
//!
//! Every other gate in this repo judges text: does the reply contain the right
//! keyword, did it converge in budget. None of them answer whether the model's
//! output is *executably correct* rather than textually plausible. This one
//! compiles (or syntax-checks) the reply and runs it against fixture cases.
//!
//! THREE outcome states, deliberately not collapsed:
//!
//!   compile_fail  — the reply is not a well-formed program at all
//!   wrong_output  — it builds and runs, but stdout/exit disagree with the case
//!   pass          — stdout and exit code both match
//!
//! "Well-formed but wrong" is a different failure from "unparseable", and that
//! distinction is the whole diagnostic value: collapsing them turns a legible
//! signal ("the model writes valid Rust but can't sieve primes") into a bare
//! failure count.
//!
//! The sharpness of that boundary is language-dependent, and comparing
//! `compile_fail` ACROSS languages is therefore a mistake. `rustc` rejects
//! anything that is not a Rust program. `python3 -m py_compile` rejects prose.
//! `bash -n` accepts almost any text, because a bare word is a valid command
//! invocation — so a bash reply that is not a program at all typically scores
//! `wrong_output` (empty stdout, exit 127) rather than `compile_fail`.
//! Verified by a fixture-mutation control, not assumed. Read `compile_fail`
//! within a language, not between them.
//!
//! Instruction-following is judged, not just measured. Every prompt ends with
//! "no markdown code fences", so a fenced reply disobeyed an explicit
//! instruction and FAILS its probe. The fence is still stripped and the program
//! still built and run, because the two signals are independent and both worth
//! having: a fenced-but-correct probe reports all cases passing and a failed
//! verdict, which reads as "can write the program, didn't follow the format".
//! `fenced` counts the probes, so the rate stays visible next to the verdict.
//!
//! Safety. The engine is trusted, but its *output* is not a program anyone
//! reviewed, and a gate that can wedge the machine is a gate nobody runs. So:
//! a fresh temp cwd per case, a hard timeout with a kill, stdout/stderr always
//! captured (never inherited from the terminal), and no probe that needs the
//! network. A hanging generation is a failed case, not a hung campaign.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Hard per-case wall clock. A generated program that blocks (reads stdin it
/// was never given, loops forever) gets killed and scored `wrong_output`.
const CASE_TIMEOUT: Duration = Duration::from_secs(10);
/// Build/syntax-check budget. Separate from the run budget because a cold
/// `rustc` legitimately takes seconds, while a *running* program should not.
const BUILD_TIMEOUT: Duration = Duration::from_secs(60);
/// Captured stderr shown on a compile failure — enough to see the error, not
/// enough to drown the report.
const STDERR_PREVIEW: usize = 160;
/// Lines of the rejected source echoed under a `compile_fail`. Enough to see
/// whether the program is malformed or merely cut off; compile failures are
/// rare enough that this does not flood a campaign log.
const SOURCE_PREVIEW_LINES: usize = 14;

/// One executable probe: a prompt plus the cases its program must satisfy.
#[derive(serde::Deserialize)]
pub(crate) struct ProgProbe {
    pub(crate) id: String,
    pub(crate) lang: ProgLang,
    pub(crate) prompt: String,
    /// The SAME generated program re-invoked with different inputs. One case
    /// tests whether it works; several test whether it generalizes past the
    /// example baked into the prompt.
    pub(crate) cases: Vec<ProgCase>,
    /// Runaway guard, not a ratchet. This battery judges execution, not
    /// convergence, so the budget is loose on purpose — a probe that blows it
    /// is looping, not merely slow.
    pub(crate) max_steps: usize,
}

#[derive(serde::Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ProgLang {
    Rust,
    Python,
    Bash,
}

impl ProgLang {
    fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::Bash => "bash",
        }
    }
    /// Toolchain probe, run once per language BEFORE any GPU time is spent —
    /// a missing `rustc` must fail the campaign in the first second, not score
    /// every Rust probe as a model compile failure three hours in.
    fn toolchain(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::Rust => ("rustc", &["--version"]),
            Self::Python => ("python3", &["--version"]),
            Self::Bash => ("bash", &["--version"]),
        }
    }
    fn source_name(self) -> &'static str {
        match self {
            Self::Rust => "main.rs",
            Self::Python => "prog.py",
            Self::Bash => "prog.sh",
        }
    }
}

/// One invocation of the generated program.
#[derive(serde::Deserialize)]
pub(crate) struct ProgCase {
    #[serde(default)]
    pub(crate) args: Vec<String>,
    #[serde(default)]
    pub(crate) stdin: Option<String>,
    pub(crate) stdout: String,
    #[serde(default)]
    pub(crate) exit: i32,
}

/// Battery tallies. `cases` == `pass + compile_fail + wrong_output` by
/// construction; `fenced` and `probes` count PROBES, not cases (a fence is a
/// property of one generation, not of each re-invocation of its program).
#[derive(Default, Clone, Debug)]
pub(crate) struct ProgCounts {
    pub(crate) probes: u64,
    pub(crate) cases: u64,
    pub(crate) pass: u64,
    pub(crate) compile_fail: u64,
    pub(crate) wrong_output: u64,
    pub(crate) fenced: u64,
}

impl ProgCounts {
    /// The ONLY place a case verdict is tallied, so
    /// `pass + compile_fail + wrong_output == cases` holds by construction
    /// rather than by three call sites agreeing to increment in pairs.
    fn record(&mut self, o: CaseOutcome) {
        self.cases += 1;
        match o {
            CaseOutcome::Pass => self.pass += 1,
            CaseOutcome::CompileFail => self.compile_fail += 1,
            CaseOutcome::WrongOutput => self.wrong_output += 1,
        }
    }
}

/// Per-case verdict. The three states of the battery.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub(crate) enum CaseOutcome {
    CompileFail,
    WrongOutput,
    Pass,
}

/// Strip ONE leading and ONE trailing markdown fence, reporting whether a fence
/// was there. Only a fence that opens the (trimmed) reply counts: a reply that
/// explains itself first and fences second is not a program, and scoring it
/// `compile_fail` is the honest answer — the model was asked for bare source.
///
/// An unterminated opening fence still strips and still counts, because the
/// generation was cut off mid-block, not written without a fence.
pub(crate) fn strip_fence(reply: &str) -> (String, bool) {
    let trimmed = reply.trim();
    let mut lines: Vec<&str> = trimmed.lines().collect();
    let is_fence = |l: &str| l.trim_start().starts_with("```");
    if lines.first().is_none_or(|l| !is_fence(l)) {
        return (trimmed.to_string(), false);
    }
    lines.remove(0);
    // Closing fence: the last line, if it is one. Anything after a mid-reply
    // fence is prose the model appended, and dropping it silently would hide
    // exactly the instruction-following miss we are trying to measure — so
    // only a TRAILING fence is removed.
    if lines.last().is_some_and(|l| is_fence(l)) {
        lines.pop();
    }
    (lines.join("\n").trim().to_string(), true)
}

/// Compare program stdout to the fixture's expectation: trailing whitespace on
/// each line and at the end of the stream is ignored, everything else is exact.
/// Line COUNT and inner spacing are part of the contract — several probes exist
/// precisely to catch a stray separator or a doubled newline.
fn stdout_matches(actual: &str, expected: &str) -> bool {
    fn norm(s: &str) -> String {
        s.lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n")
            .trim_end()
            .to_string()
    }
    norm(actual) == norm(expected)
}

/// What a finished (or killed) child process produced.
struct Ran {
    code: Option<i32>,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

/// Run a child with captured pipes and a hard deadline.
///
/// stdout/stderr are drained on their own threads: polling `try_wait` while a
/// child fills a pipe buffer deadlocks, which would turn the timeout this
/// function exists to enforce into the hang it exists to prevent.
fn run_capture(cmd: &mut Command, stdin_data: Option<&str>, timeout: Duration) -> std::io::Result<Ran> {
    cmd.stdin(if stdin_data.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    })
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;

    if let (Some(data), Some(mut sink)) = (stdin_data, child.stdin.take()) {
        let data = data.to_string();
        // Own thread: a program that never reads stdin would otherwise block
        // us forever on the write.
        std::thread::spawn(move || {
            let _ = sink.write_all(data.as_bytes());
        });
    }
    fn drain<R: Read + Send + 'static>(r: Option<R>) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut r) = r {
                let _ = r.read_to_end(&mut buf);
            }
            String::from_utf8_lossy(&buf).into_owned()
        })
    }
    let out_t = drain(child.stdout.take());
    let err_t = drain(child.stderr.take());

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait()? {
            Some(s) => break Some(s),
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                timed_out = true;
                break None;
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    Ok(Ran {
        code: status.and_then(|s| s.code()),
        stdout: out_t.join().unwrap_or_default(),
        stderr: err_t.join().unwrap_or_default(),
        timed_out,
    })
}

/// A program ready to invoke: the command and the fixed args that precede the
/// case's own arguments.
struct Built {
    program: std::path::PathBuf,
    prefix: Vec<String>,
}

/// Write the source and make it runnable: compile it (rust) or syntax-check it
/// (python, bash).
///
/// The syntax check is the deliberate analogue of compilation — it separates
/// "not a program" from "a program that computes the wrong thing", which is the
/// compile_fail/wrong_output split. It does NOT catch a `NameError`; that is
/// correct, because an interpreter reaching a NameError at runtime produced
/// wrong output, and that is what the case will score it as.
fn build(lang: ProgLang, source: &str, dir: &Path) -> std::io::Result<Result<Built, String>> {
    let src = dir.join(lang.source_name());
    std::fs::write(&src, source)?;
    let (check, built) = match lang {
        ProgLang::Rust => {
            let bin = dir.join("prog");
            let mut c = Command::new("rustc");
            c.current_dir(dir)
                .arg("--edition")
                .arg("2021")
                .arg("-o")
                .arg(&bin)
                .arg(&src);
            (
                c,
                Built {
                    program: bin,
                    prefix: Vec::new(),
                },
            )
        }
        ProgLang::Python => {
            let mut c = Command::new("python3");
            c.current_dir(dir).arg("-m").arg("py_compile").arg(&src);
            (
                c,
                Built {
                    program: "python3".into(),
                    prefix: vec![src.to_string_lossy().into_owned()],
                },
            )
        }
        ProgLang::Bash => {
            let mut c = Command::new("bash");
            c.current_dir(dir).arg("-n").arg(&src);
            (
                c,
                Built {
                    program: "bash".into(),
                    prefix: vec![src.to_string_lossy().into_owned()],
                },
            )
        }
    };
    let mut check = check;
    let ran = run_capture(&mut check, None, BUILD_TIMEOUT)?;
    if ran.timed_out {
        return Ok(Err("build timed out".to_string()));
    }
    if ran.code != Some(0) {
        let msg = ran
            .stderr
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("(no stderr)")
            .chars()
            .take(STDERR_PREVIEW)
            .collect::<String>();
        return Ok(Err(msg));
    }
    Ok(Ok(built))
}

/// Judge one case: fresh cwd, hard timeout, captured pipes.
fn run_case(built: &Built, case: &ProgCase, cwd: &Path) -> std::io::Result<(CaseOutcome, String)> {
    let mut c = Command::new(&built.program);
    c.current_dir(cwd).args(&built.prefix).args(&case.args);
    let ran = run_capture(&mut c, case.stdin.as_deref(), CASE_TIMEOUT)?;
    if ran.timed_out {
        // Built fine, never terminated. Not a compile failure; a failed case.
        return Ok((CaseOutcome::WrongOutput, "TIMEOUT".to_string()));
    }
    let out_ok = stdout_matches(&ran.stdout, &case.stdout);
    let exit_ok = ran.code == Some(case.exit);
    if out_ok && exit_ok {
        return Ok((CaseOutcome::Pass, String::new()));
    }
    let mut why = Vec::new();
    if !out_ok {
        why.push(format!(
            "stdout {:?} != {:?}",
            ran.stdout.trim_end().chars().take(48).collect::<String>(),
            case.stdout.chars().take(48).collect::<String>(),
        ));
    }
    if !exit_ok {
        why.push(format!("exit {:?} != {}", ran.code, case.exit));
    }
    Ok((CaseOutcome::WrongOutput, why.join("; ")))
}

/// Fail the battery before it starts if a language's toolchain is missing.
/// A missing `rustc` is an infrastructure fault; scoring it as the model's
/// compile failure would be a fabricated result.
pub(crate) fn preflight(probes: &[ProgProbe]) -> Result<(), String> {
    let mut seen: Vec<ProgLang> = Vec::new();
    for p in probes {
        if seen.contains(&p.lang) {
            continue;
        }
        seen.push(p.lang);
        let (bin, args) = p.lang.toolchain();
        let mut c = Command::new(bin);
        c.args(args);
        match run_capture(&mut c, None, BUILD_TIMEOUT) {
            Ok(r) if r.code == Some(0) => {}
            Ok(r) => {
                return Err(format!(
                    "programmatic: `{bin}` for lang {} exited {:?} — toolchain unusable",
                    p.lang.name(),
                    r.code
                ));
            }
            Err(e) => {
                return Err(format!(
                    "programmatic: cannot run `{bin}` for lang {}: {e}",
                    p.lang.name()
                ));
            }
        }
    }
    Ok(())
}

/// Unique-per-run scratch root. No wall-clock in the name: the pid plus a
/// monotonic counter is enough, and keeps the path reproducible in a log.
fn scratch_root() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "dgq-prog-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Run every probe: generate, build, execute each case, tally.
///
/// `run_one` is the caller's single-turn generation (the battery does not own a
/// session — it borrows `smoketest`'s, so all batteries share one warm model).
/// Returns the counts plus the per-probe pass/total/failures the outcome needs.
pub(crate) fn run_probes(
    probes: &[ProgProbe],
    run_one: &mut dyn FnMut(&str) -> Result<(usize, String), crate::Error>,
) -> (ProgCounts, usize, usize, Vec<String>) {
    let mut counts = ProgCounts::default();
    let (mut passed, mut total) = (0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    let root = scratch_root();
    if let Err(e) = std::fs::create_dir_all(&root) {
        eprintln!("programmatic: cannot create {}: {e}", root.display());
        return (counts, 0, probes.len(), probes.iter().map(|p| p.id.clone()).collect());
    }

    for (pi, p) in probes.iter().enumerate() {
        total += 1;
        counts.probes += 1;
        let (steps, reply) = match run_one(&p.prompt) {
            Ok(v) => v,
            Err(err) => {
                println!("  {:<22} ERROR  {err}", p.id);
                failures.push(p.id.clone());
                continue;
            }
        };
        let (source, fenced) = strip_fence(&reply);
        if fenced {
            counts.fenced += 1;
        }
        let probe_dir = root.join(format!("p{pi}"));
        if let Err(e) = std::fs::create_dir_all(&probe_dir) {
            println!("  {:<22} ERROR  mkdir: {e}", p.id);
            failures.push(p.id.clone());
            continue;
        }

        let n_cases = p.cases.len() as u64;
        let built = match build(p.lang, &source, &probe_dir) {
            Ok(Ok(b)) => b,
            Ok(Err(why)) => {
                // Every case of an unbuildable probe is a compile_fail — the
                // program never got to be wrong, it never got to run.
                for _ in &p.cases {
                    counts.record(CaseOutcome::CompileFail);
                }
                failures.push(p.id.clone());
                println!(
                    "  {id:<22} FAIL  {lang:<6} steps {steps:>3}/{max:<3} {f} compile_fail: {why}",
                    id = p.id,
                    lang = p.lang.name(),
                    max = p.max_steps,
                    f = if fenced { "fenced" } else { "bare  " },
                );
                // The rejected source IS the diagnostic. Without it a
                // compile_fail cannot be told apart from a harness artifact —
                // "unexpected EOF" reads identically whether the model wrote
                // an unbalanced quote or the token cap truncated it mid-line.
                // HEAD AND TAIL, not just head: whether a rejected program was
                // truncated is read off its LAST line. A generation that ran
                // into the token cap ends mid-statement, and scoring that as
                // the model's compile failure would be measuring our own
                // ceiling — head-only output cannot tell the two apart.
                let lines: Vec<&str> = source.lines().collect();
                let show = |range: std::ops::Range<usize>| {
                    for i in range {
                        println!("      {:>3} | {}", i + 1, lines[i]);
                    }
                };
                if lines.len() <= SOURCE_PREVIEW_LINES * 2 {
                    show(0..lines.len());
                } else {
                    show(0..SOURCE_PREVIEW_LINES);
                    println!("      ... {} line(s) elided ...", lines.len() - SOURCE_PREVIEW_LINES * 2);
                    show(lines.len() - SOURCE_PREVIEW_LINES..lines.len());
                }
                continue;
            }
            Err(e) => {
                println!("  {:<22} ERROR  build: {e}", p.id);
                failures.push(p.id.clone());
                continue;
            }
        };

        let mut case_pass = 0u64;
        let mut detail = String::new();
        for (ci, case) in p.cases.iter().enumerate() {
            // Fresh cwd per case: a program that writes a file must not be
            // able to change what the next case observes.
            let cwd = probe_dir.join(format!("case{ci}"));
            let judged = match std::fs::create_dir_all(&cwd) {
                Ok(()) => run_case(&built, case, &cwd)
                    .unwrap_or_else(|e| (CaseOutcome::WrongOutput, format!("spawn: {e}"))),
                Err(e) => (CaseOutcome::WrongOutput, format!("mkdir: {e}")),
            };
            counts.record(judged.0);
            if judged.0 == CaseOutcome::Pass {
                case_pass += 1;
            } else if detail.is_empty() {
                detail = format!("case {ci}: {}", judged.1);
            }
        }
        // A fence FAILS the probe: every prompt says "no markdown code
        // fences", so a fenced reply disobeyed an explicit instruction, and a
        // verdict that ignored it would let that pass silently. The program is
        // still built and run regardless, so the correctness diagnostic is
        // kept — a fenced-but-correct probe reports 2/2 cases AND fails.
        let ok = case_pass == n_cases && steps <= p.max_steps && !fenced;
        if ok {
            passed += 1;
        } else {
            failures.push(p.id.clone());
        }
        println!(
            "  {id:<22} {mark:<4}  {lang:<6} steps {steps:>3}/{max:<3} {f} cases {cp}/{n}  {detail}",
            id = p.id,
            mark = if ok { "PASS" } else { "FAIL" },
            lang = p.lang.name(),
            max = p.max_steps,
            f = if fenced { "fenced" } else { "bare  " },
            cp = case_pass,
            n = n_cases,
        );
    }

    let _ = std::fs::remove_dir_all(&root);
    (counts, passed, total, failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_is_stripped_but_counted() {
        let (s, f) = strip_fence("```rust\nfn main() {}\n```");
        assert_eq!(s, "fn main() {}");
        assert!(f, "instruction-following miss must be reported, not hidden");

        let (s, f) = strip_fence("fn main() {}\n");
        assert_eq!(s, "fn main() {}");
        assert!(!f);

        // Unterminated fence: the generation was cut off, but it WAS fenced.
        let (s, f) = strip_fence("```\nprint(1)");
        assert_eq!(s, "print(1)");
        assert!(f);

        // Prose first is not a bare program; leaving it in scores compile_fail,
        // which is the honest verdict for "did not follow the instruction".
        let (s, f) = strip_fence("Here you go:\n```\nprint(1)\n```");
        assert!(s.starts_with("Here you go:"));
        assert!(!f);
    }

    /// Fence shapes a real reply plausibly takes.
    #[test]
    fn fence_edges() {
        // Language tag plus trailing whitespace on the closing fence.
        let (s, f) = strip_fence("```python  \nprint(1)\n```   ");
        assert_eq!(s, "print(1)");
        assert!(f);

        // A triple-backtick INSIDE the program is content, not a delimiter:
        // only the first and last lines are ever treated as fences, so the
        // body survives intact.
        let (s, f) = strip_fence("```\ns = \"```\"\nprint(s)\n```");
        assert_eq!(s, "s = \"```\"\nprint(s)");
        assert!(f);

        // Trailing prose after the closing fence is NOT dropped — silently
        // discarding it would hide the instruction-following miss, so it stays
        // in the source and scores compile_fail.
        let (s, f) = strip_fence("```\nprint(1)\n```\nHope that helps!");
        assert!(s.ends_with("Hope that helps!"), "got {s:?}");
        assert!(f);

        // An empty fenced block yields an empty program, not a panic.
        let (s, f) = strip_fence("```\n```");
        assert_eq!(s, "");
        assert!(f);
    }

    #[test]
    fn stdout_compare_ignores_only_trailing_space() {
        assert!(stdout_matches("a\nb\n", "a\nb"));
        assert!(stdout_matches("a  \nb\n\n", "a\nb"));
        // Line count and inner spacing are part of the contract.
        assert!(!stdout_matches("a\n\nb", "a\nb"));
        assert!(!stdout_matches("a b", "a  b"));
        assert!(!stdout_matches("", "0"));
        // A probe may legitimately expect NO output (see rust_exit_by_input,
        // which is judged entirely on its exit status).
        assert!(stdout_matches("", ""));
        assert!(stdout_matches("\n", ""));
        assert!(!stdout_matches("0", ""));
        // \r\n normalizes: `lines()` splits it and trim_end eats the \r, so a
        // program emitting CRLF is not failed for its line endings.
        assert!(stdout_matches("a\r\nb\r\n", "a\nb"));
    }

    /// The battery's own machinery, exercised without the model: a correct
    /// program passes, a well-formed-but-wrong one is `wrong_output`, and an
    /// unparseable one is `compile_fail`. Bash only — always present, and no
    /// compile step to slow the suite.
    #[test]
    fn three_outcome_states_are_distinguished() {
        let root = scratch_root();
        std::fs::create_dir_all(&root).unwrap();
        let case = ProgCase {
            args: vec!["x".into(), "y".into()],
            stdin: None,
            stdout: "x_y".into(),
            exit: 0,
        };

        let judge = |name: &str, src: &str| -> CaseOutcome {
            let d = root.join(name);
            std::fs::create_dir_all(&d).unwrap();
            match build(ProgLang::Bash, src, &d).unwrap() {
                Err(_) => CaseOutcome::CompileFail,
                Ok(b) => run_case(&b, &case, &d).unwrap().0,
            }
        };

        assert_eq!(
            judge("good", "IFS=_; echo \"$*\""),
            CaseOutcome::Pass,
        );
        assert_eq!(
            judge("wrong", "echo \"$*\""),
            CaseOutcome::WrongOutput,
            "well-formed but wrong is NOT a compile failure"
        );
        assert_eq!(
            judge("broken", "if [ 1 -eq 1 ]; then\necho unterminated"),
            CaseOutcome::CompileFail,
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Exit code is judged, not just stdout — `rust_hello_42` exists to catch
    /// a program that prints the right thing and exits 0 anyway.
    #[test]
    fn exit_code_is_part_of_the_verdict() {
        let root = scratch_root();
        let d = root.join("exit");
        std::fs::create_dir_all(&d).unwrap();
        let built = build(ProgLang::Bash, "echo hello\nexit 42", &d).unwrap().unwrap();
        let want = |exit| ProgCase {
            args: Vec::new(),
            stdin: None,
            stdout: "hello".into(),
            exit,
        };
        assert_eq!(run_case(&built, &want(42), &d).unwrap().0, CaseOutcome::Pass);
        assert_eq!(
            run_case(&built, &want(0), &d).unwrap().0,
            CaseOutcome::WrongOutput
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A program that never terminates must cost one case, not the machine.
    #[test]
    fn a_hang_is_a_failed_case_not_a_wedged_machine() {
        let root = scratch_root();
        let d = root.join("hang");
        std::fs::create_dir_all(&d).unwrap();
        let built = build(ProgLang::Bash, "while true; do :; done", &d).unwrap().unwrap();
        let case = ProgCase {
            args: Vec::new(),
            stdin: None,
            stdout: String::new(),
            exit: 0,
        };
        let start = Instant::now();
        let (outcome, why) = run_case(&built, &case, &d).unwrap();
        assert_eq!(outcome, CaseOutcome::WrongOutput);
        assert_eq!(why, "TIMEOUT");
        assert!(
            start.elapsed() < CASE_TIMEOUT * 3,
            "the kill must actually fire"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The SCORER end-to-end with no model: a fake generator returns canned
    /// replies, so every tallying rule is pinned deterministically instead of
    /// being inferred from a campaign that happened to pass. Bash only — no
    /// compile step, so the whole probe set scores in well under a second.
    ///
    /// Pins, in one pass: case-level partial credit under a binary probe
    /// verdict, compile_fail fanning out to EVERY case of an unbuildable
    /// probe, and THREE independent failure axes — wrong cases, a blown step
    /// budget, and a markdown fence — plus the
    /// `pass + compile_fail + wrong_output == cases` invariant.
    #[test]
    fn scorer_tallies_a_mixed_probe_set_without_the_model() {
        // Deserialized from JSON rather than built inline, so this also pins
        // the fixture schema the real prompts.json must satisfy.
        let probes: Vec<ProgProbe> = serde_json::from_str(
            r#"[
              {"id":"all_ok","lang":"bash","prompt":"","max_steps":10,
               "cases":[{"args":["a"],"stdout":"a","exit":0},
                        {"args":["b"],"stdout":"b","exit":0}]},
              {"id":"partial","lang":"bash","prompt":"","max_steps":10,
               "cases":[{"args":["a"],"stdout":"a","exit":0},
                        {"args":["b"],"stdout":"bb","exit":0}]},
              {"id":"broken","lang":"bash","prompt":"","max_steps":10,
               "cases":[{"args":[],"stdout":"x","exit":0},
                        {"args":[],"stdout":"y","exit":0}]},
              {"id":"fenced_correct","lang":"bash","prompt":"","max_steps":10,
               "cases":[{"args":[],"stdout":"hi","exit":0}]},
              {"id":"overbudget","lang":"bash","prompt":"","max_steps":1,
               "cases":[{"args":[],"stdout":"hi","exit":0}]}
            ]"#,
        )
        .unwrap();

        let replies = [
            "echo \"$1\"",                     // correct for both cases
            "echo \"$1\"",                     // correct for case 0 only
            "if [ 1 -eq 1 ]; then\necho x",    // unterminated -> compile_fail
            "```bash\necho hi\n```",           // correct, but fenced
            "echo hi",                         // correct, but over the budget
        ];
        let mut n = 0usize;
        let mut fake = |_: &str| -> Result<(usize, String), crate::Error> {
            let r = replies[n].to_string();
            n += 1;
            // 5 steps: inside every budget above except `overbudget`'s 1.
            Ok((5, r))
        };

        let (c, passed, total, failures) = run_probes(&probes, &mut fake);

        assert_eq!(c.cases, 8);
        assert_eq!(
            c.cases,
            c.pass + c.compile_fail + c.wrong_output,
            "the three states must partition the cases"
        );
        assert_eq!(
            c.pass, 5,
            "2 all_ok + 1 partial + fenced_correct + overbudget: a fenced or \
             over-budget probe still gets full CASE credit for running right"
        );
        assert_eq!(c.compile_fail, 2, "BOTH cases of the unbuildable probe");
        assert_eq!(c.wrong_output, 1, "only partial's second case");
        assert_eq!(c.fenced, 1);
        assert_eq!(c.probes, 5);

        // Probe verdict is binary even though case credit is partial, and it
        // fails on ANY of the three axes — so `fenced_correct` and
        // `overbudget` both fail despite every one of their cases passing.
        assert_eq!((passed, total), (1, 5), "only all_ok clears every axis");
        assert_eq!(
            failures,
            ["partial", "broken", "fenced_correct", "overbudget"]
        );
    }

    /// stdin is piped, and a program that ignores it does not deadlock the
    /// runner on the write side.
    #[test]
    fn stdin_is_piped_and_ignoring_it_does_not_hang() {
        let root = scratch_root();
        let d = root.join("reader");
        let e = root.join("deaf");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::create_dir_all(&e).unwrap();
        let counter = build(ProgLang::Bash, "wc -l | tr -d ' '", &d).unwrap().unwrap();
        let case = ProgCase {
            args: Vec::new(),
            stdin: Some("a\nb\nc\n".into()),
            stdout: "3".into(),
            exit: 0,
        };
        assert_eq!(run_case(&counter, &case, &d).unwrap().0, CaseOutcome::Pass);

        // Bigger than a pipe buffer, and never read: the write must not block
        // the runner (it lives on its own thread).
        let deaf = build(ProgLang::Bash, "echo done", &e).unwrap().unwrap();
        let ignore = ProgCase {
            args: Vec::new(),
            stdin: Some("x".repeat(1 << 20)),
            stdout: "done".into(),
            exit: 0,
        };
        assert_eq!(run_case(&deaf, &ignore, &e).unwrap().0, CaseOutcome::Pass);
        let _ = std::fs::remove_dir_all(&root);
    }
}
