# diffgemma — how to work on this codebase

Read this before writing kernels, tests, or chasing bugs. It is **how to work
here without repeating the mistakes that have already cost days.**

The project: a Rust + Metal inference engine for DiffusionGemma (Gemma-4
26B-A4B MoE, discrete block diffusion) on Apple Silicon.

Doc map — three documents, each with one job:
- **ARCHITECTURE.md** — the design and the implemented generation contract,
  including the *Negative Knowledge* section (disproven approaches and the
  physics that blocked them).
- **PLAN.md** — open work only.
- **AGENTS.md** (this file) — working discipline.
- Everything else lives in **git history** (thorough commit messages are the
  changelog) and the agent memory. Do NOT add changelog-style banners,
  dated status sections, or "history of this fix" narratives to code or
  docs — describe the present; commit messages record the past.

Authoritative numeric behavior: the CPU reference (per-kernel `cpu.rs`
oracles under `src/shaders/`, shared ops in `src/shaders/cpu/`, plus
`sample.rs`) and the weight manifest (`model.dgq.json`).

---

## 1. The one thing to internalize

**Every serious bug in this project has lived in a fused or accelerated GPU
path that a slower reference path got right.** MPS-Q4 producing uniform
logits, the SC GEMM transpose, the softmax grid collapse, the MoE
route-garbage from a last-expert `n_tok` bug, the fast-prefill encoder
running a denoise-only norm — all the same shape: the optimized path computes
a different function than the reference, parity is green because the
optimized path has no golden, and the symptom only appears downstream
(layer 2+, entropy collapse, pad output, >2.5k-token collapse) far from the
cause.

The corollary that governs everything below: **an untested path is where the
next bug is.** Speed without a per-path correctness gate is how bugs ship.
So the strategy is not "go fast"; it is "make divergence impossible to
introduce silently, then go fast."

---

## 2. Diagnostic discipline (how to chase a bug)

When output is wrong, follow this order. Do not skip to optimization or to
rewriting a kernel you can't explain.

1. **Localize before you theorize.** Find the *smallest* unit that diverges
   from the reference. A 30-layer entropy collapse is not a bug location;
   it's a symptom. Bisect: which layer, which kernel, which stage. The MoE
   hunt took ten turns because we theorized about kernel math for rounds
   before dumping the one value that localized it in a single read.

2. **Same input, same weights, two paths.** Run the suspect GPU path and the
   CPU reference on *byte-identical* input, compare cosine, and read the
   magnitude as a diagnostic: >0.999 = that stage is fine; ~0.5 =
   correlated-but-wrong (a swap, a scale, partial corruption); ~0 = reading
   the wrong data entirely.

3. **Dump what the kernel actually reads, not what you think it reads.** A
   CPU transliteration of the *source* can match the reference while the
   *GPU execution* diverges — it can't reproduce threadgroup semantics, arena
   bindings, or route resolution. Write the kernel's real inputs to a scratch
   buffer and read them back.

4. **Impossible numbers mean wrong N or wrong normalization.** Entropy >
   ln(N), Z=0, cos > 1, values at ~1e38 are never "the model is just bad" —
   they are indexing, normalization, or overflow bugs with a specific cause.

5. **Two paths failing differently is a gift.** The *difference* localizes the
   bug to path-specific code. One exploding (inf) and one inert (zero) meant
   two bugs in the same conceptual spot, and finding one explained the other.

6. **A contradiction is a second bug, not an anomaly.** If a fix changes an
   intermediate but not the output, you are measuring two different code
   paths (probe vs production) or reading a stale buffer. Resolve it.

7. **Don't let a workaround end the investigation.** Falling back to a slow
   reference path unblocks convergence but leaves the latent bug in every
   other kernel sharing the flawed pattern.

8. **Reproduce a gap across inputs before calling it a bug, and change one
   variable at a time.** In the denoise loop a sub-1e-4 per-step difference
   can flip an accept decision and cascade into a different but equally valid
   trajectory, so a single-prompt delta can be chaos rather than a defect —
   check it is systematic before chasing it. When it is real, bisect to the
   axis (sampler / forward precision / a specific quantized tensor) before
   rebuilding; "rebuild with everything different" tells you nothing.

9. **Measure value/type ranges before calling a precision experiment failed**
   (`DGQ_TRACE_RANGES`, value-cos), and check ARCHITECTURE.md's Negative
   Knowledge + agent memory for the lever family BEFORE planning perf work —
   task descriptions carry stale premises.

---

## 3. Measure before optimizing — always

- **Get a clean measurement first.** Compile out probes/readbacks before
  timing anything. A readback is a GPU pipeline stall; instrumentation can
  dominate a step.
- **Attribute the time before reducing it.** Per-dispatch timing on one clean
  step. Do not guess which stage dominates; the guesses have been wrong.
- **Know the regime** — ARCHITECTURE.md's "Performance regime" has the
  measured classification and the numbers. The short version: nothing here is
  bandwidth-bound, so byte-cutting levers do not pay. Check Negative Knowledge
  before reaching for one.
- **Sequence optimizations to the bottleneck.** At multi-second steps the win
  is "stop doing the slow/temporary thing" (remove probes, use the fused
  kernel, re-enable the fast path). Don't reach for architecture when the
  cost is a left-on debug path.
- **Timing methodology**: separate retrieval from perf runs; perf A/Bs are
  isolated benches on representative subsets, within-process adjacent
  (`bench-step-kernel`, `bench-gemm`), never full-wall-clock comparisons of
  mixed work.
- **Lever hygiene**: a lever disproven on one shape is not disproven
  forever — when the shape changes, cheaply re-test the levers it could
  have unblocked. But gate every re-test on an output oracle FIRST: a
  re-test that reports a speedup with no correctness check at the new
  config is not a vindication (the fake `DGQ_MOE_PREFILL_BM` win). And a
  physics story that "matches exactly" is the failure mode to distrust —
  decompose the claimed mechanism one variable at a time before recording
  it (the int8 9× that was really 1.7×bug × 2.2× × 2.5×).

---

## 4. Kernel organization: one body, compile-time specialization

Kernel code lives in **`src/shaders/<group>/<kernel>/`** — the Rust wrapper
(`mod.rs`), the Metal source (`<kernel>.metal`), the CPU oracle (`cpu.rs`),
and the manifest registration (`SPEC`) are colocated. Only `src/shaders/`
knows `.metal` paths; `src/metal/` (the runtime) consumes pipelines via
`shaders::<kernel>::pipeline_for*` / `{SHADER, ENTRY}`.

- **One source body per logical kernel.** A kernel = the operation + its
  tiling. Variant axes — weight format (q4/q8/nvfp4/raw), fusion/output mode,
  dtype, dump depth, even divergent buffer signatures — are **function
  constants** selected at pipeline-compile time. A "variant" is a tuple of
  constant values, not a file. Never fork `k_foo` into `k_foo_fp8`,
  `k_foo_debug`.
- **Intermediate dumps are a compile-time mode of the production kernel**
  (`K_DUMP_STAGE`), not a separate probe kernel. Fold hunt-time probes back
  behind the dump flag once a bug is found.
- FC 1–3 are global (shape-assert, dump, quant-format); local axes are
  registered in each kernel's `SPEC` (see `diffgemma manifest`).
- Quoted `#include "x.metal"` resolves from `src/shaders/include/` at
  runtime (`common/expand.rs`); the pipeline binary archive keys on the
  whole-tree hash, so shader edits can never be served stale — but if golden
  regresses right after a `.metal` edit, **rebuild clean before diagnosing**
  (a mid-sequence binary can still be stale).

---

## 5. Testing: three tiers, push assertions down

**Tier 1 — per-kernel unit tests.** Synthetic, blob-free, milliseconds. One
test per kernel against its CPU oracle on a tiny hand-built fixture — never
the 19 GiB blob. Fixtures must exceed the worst tile (e.g. grouped
`rows_per_expert ≥ 33`). Every kernel has a permanent CPU twin; the twin is
the oracle forever.

**Tier 2 — staged comparison.** Real weights, reduced stages, dump depth on.
Catches integration bugs between correct kernels.

**Tier 3 — end-to-end goldens.** `golden` (byte-identity, 8-case path
matrix), `smoketest` (17/17 gate), full suite. Regression gates, **not**
debugging tools.

**Transitivity is the speed win**: pin both engines to the same CPU oracle in
Tier 1 and engine-vs-engine agreement is automatic.

**Recurring failure class**: stale test dispatch grids after kernel grid
rewrites — a cos≈0.34 oracle failure usually means the HARNESS dispatches an
old shape, not that the kernel math broke. Check the harness grid first.

**Suite mechanics**: `cargo test --release`. Model-gated tests key off
`test_util::dgq_model_dir()`. `--test-threads=1` is no longer required —
`membudget` gates concurrent model loads with byte permits (a second loader
waits on a condvar instead of OOMing), and the pipeline archive is behind a
`Mutex`. Expect only ~20-30% off the wall, since the ~9 heavy model-gated
tests still serialize on memory and dominate it.

If parallelism regresses it fails LOUD (SIGSEGV, or a membudget timeout that
names the holder) — revert to `--test-threads=1` and say why. `membudget`
PANICS on a nested acquire on ONE thread: build a second runtime only after
dropping the first. The separate hard rule stands regardless of thread count:
never two model-loading PROCESSES at once.

---

## 6. Non-negotiable invariants

- **Finite:** no NaN/Inf in logits, activations, attention output, SC signal.
- **Softmax rows sum to 1.0** over their actual support.
- **Entropy ≤ ln(N)**; SC signal finite and non-zero on step ≥ 2.
- **Determinism:** same seed → same tokens on the deterministic path.
- **Not all-pad/all-filler** on a converged block before early-stop fires.
- **Offsets in `ulong`** for all blob addressing (blob > uint32).
- **Quant K-order:** sequential decode == indexed decode in K-order, not just
  as a set.
- **Tile-bound dimensions:** for every kernel, each compile-time tile must be
  either grid-tiled, striped in-kernel, or ranged-dispatched if a
  data-dependent dimension can exceed it; Tier-1 fixtures must exceed the
  worst tile. Audit mechanically: grep fixed-size arrays, check each index's
  runtime bound.
- **Ring KV writes respect `StepParams.kv_write_end`.** Past-the-end ≠ dead
  on a ring: pad/garbage rows wrap onto `pos & ring_mask` and clobber the
  oldest live window slots. Any kernel writing KV by absolute position must
  honor it.
- **Producer/consumer dtype mismatch is the real precision hazard**, not the
  choice of dtype. When changing a plane's dtype, convert every writer and
  reader together and audit the toggleable loaders.
- **KV-reuse-first** (user directive): lean toward reusing 100%
  of the KV cache — these are small machines. Any path that discards
  resident KV (fresh-conversation fork, deep truncate, canonical that
  isn't a prefix of the next request) must be a smart, explicit,
  cost-justified decision, and rewinds land at `max(mark, prompt end)`,
  never below a live prefill.

---

## 7. Authority & sources of truth

- **Numeric behavior:** the CPU reference is the oracle. When GPU and CPU
  disagree, the CPU is right and the GPU path is the suspect.
- **Weight layout:** the `model.dgq.json` manifest is authoritative for
  shapes, offsets, orientation. When a kernel's addressing assumption is in
  question, the manifest decides — not memory, not the comment.
- **Model semantics:** ARCHITECTURE.md Part II (forward-pass details: RoPE
  split-half pairing, proportional-RoPE full-head-dim denominator,
  temperature count-down, prefix-sum accept rule, QK-norm folding the
  attention scale, V aliased from raw k_proj on full layers). These are
  checkpoint-specific and several are counterintuitive — do not "correct"
  them from general Gemma knowledge without checking the reference.
- **Kernel FC registration:** each kernel's `SPEC` const;
  `diffgemma manifest` renders the full TOML view.
- **Env flags:** `src/flags/` (mod.rs = registry + parse, accessors.rs =
  read surface) is the single registry — check it before inventing any
  `DGQ_` flag. Parsed once into `RuntimeConfig`; scoped overrides via
  `install_scoped(cfg)`, and `RuntimeConfig::from_pairs` parses an explicit
  set (census arms) through the same helpers and validation.
- **A set-but-invalid `DGQ_*` value is FATAL** (exit 2, every offender
  named), never a silent fallback to the default — `DGQ_PREFIX_EXIT=1` used
  to DISABLE the lever it read like it was enabling. Unset is still the
  documented default. Add new flags via the checked helpers
  (`on_if_one`/`on_unless_zero`/`parse_usize`/`ranged_f32`), not a bespoke
  `var(...)` chain, or the value silently stops being validated.
- **Do not trust comments over code/manifest.** A stale comment has already
  misled. Verify against the authoritative source.

---

## 8. Ship gates & operational rules

- **Quality never ratchets without human sign-off.** Bit-identical changes
  ship on identity evidence (`golden` 8/8 + suite). Anything
  trajectory-affecting needs the multi-seed smoketest aggregate ({7,42,123})
  + wart census + explicit user approval. Single-seed results are arbitrary
  for trajectory-reshuffling changes.
- **`golden` is the Tier-1 refactor gate**: run before/after every refactor;
  `--bless` only after Tier-2/3 sign-off.
- **Long-context claims are judged on real-document Q&A ladders
  (`smoketest --longctx`), never needle probes alone** — needles ride a few
  sharp attention edges and stay exact while document comprehension is fully
  hallucinated.
- **NEVER run two model-loading processes in parallel** (ours ~19 GiB + MLX
  ~15 GiB = machine crash). `pgrep -f "diffgemma -m"` before any GPU run;
  serialize every bench.
- **Wart census** (10-seed greentext) is the sensitive sampler probe.
- **A battery can only evaluate a lever whose TRIGGER its outputs actually
  produce.** Check `trims > 0` (or the equivalent activation count) in the
  treatment arm before reading any arm comparison — a battery that never fires
  the lever reports a clean null three times in a row and means nothing. Judge
  by the multi-seed aggregate; use PROBE-level counts for arm comparisons and
  case-level only for diagnosing what failed.
- **Pass `--seed` EXPLICITLY on every arm of a comparison.** `smoketest
  --longctx` defaults to seed 7, not 42, so a defaulted run and a
  `--seed 42` run are different experiments. This cost a whole false
  "long-context nondeterminism" finding (later retracted): the two
  differing runs were seeds 7 and 42. The tool prints `seed N` — read it.
- **Replicate BOTH arms before believing a difference, and run the cheapest
  control first**: re-run the exact failing command verbatim. A control on
  only the arm you suspect proves nothing about the one you trust.
- **State assumptions as checks, not beliefs.** "The stride is K+2" → assert
  it against the manifest and a byte dump.
- **One bisecting measurement beats three rounds of theory.**
- **Every bug fixed gets a regression test at the lowest tier that would have
  caught it.**
- **No path ships without a golden.** New kernel variant / code path → a
  matrix row before it's "done."
- **Don't optimize against a dirty baseline.**
- **Write thorough commit messages** — they are the project changelog and
  the disproof ledger's primary record. Keep PLAN.md forward-only and
  ARCHITECTURE.md present-tense; move anything historical into the commit
  that changed it.
- `cargo fmt` is safe to run freely (repo is rustfmt-normalized; big
  reformats go in their own blame-ignored commit).

---

## 9. Command reference

`WEIGHTS=model/diffgemma-26b-a4b-it-q4`; binary at `target/release/diffgemma`
(build: `cargo build --release`).

```bash
# Generate / chat / serve
diffgemma ask  -m $WEIGHTS -p "Hello" --seed 42
diffgemma chat -m $WEIGHTS                  # --ctx N for long context
#   chat extras: --tool NAME[:DESC]=CMD; --harness FILE.json (prompt + prethink +
#   file-backed vars + multi-param tools in one file; examples in harness/)
diffgemma serve -m $WEIGHTS                 # OpenAI-compatible HTTP; --ctx N (default 128k)
#   serve extras: --log-dir DIR (op-log ops.jsonl), --tool-repair, --tool-validate
diffgemma replay /path/ops.jsonl -m $WEIGHTS  # re-execute + diff an op-log

# Gates (run before commit)
diffgemma smoketest -m $WEIGHTS             # 17/17 required
diffgemma smoketest -m $WEIGHTS --longctx   # doc-QA ladder
diffgemma golden -m $WEIGHTS                # byte-identity 8/8
cargo test --release

# Campaigns: flag ARMS x BATTERIES with explicit gates, one process, stats
# to a dir. This is how a quality lever is decided — not a shell loop.
diffgemma census -m $WEIGHTS \
  --arm 'off:' --arm 'trim:DGQ_COMMIT_CONF_TRIM=0.9' \
  --battery smoke,longctx --seeds 7,42,123 --baseline off \
  --out runs/c1 --gate 'passed==1' --gate 'contested_per_1k<=baseline'
diffgemma census -m $WEIGHTS --analyze runs/c1   # re-report, no GPU
#   batteries: smoke | longctx | programmatic (generate a program, RUN it,
#   judge stdout + exit code; metrics prog_pass_pct, compile_fail,
#   wrong_output, fenced_pct) | soft (indirect retrieval + hallucination
#   rates, non-blocking)

# Bench / diagnostics
diffgemma bench-step-kernel -m $WEIGHTS --profile-steps 8
diffgemma bench-gemm --shapes 256x2816x2816 --oracle mps --iters 10
diffgemma step-probe -m $WEIGHTS --layers 3 --kv-len 64 --seed 42
diffgemma step-parity -m $WEIGHTS           # engine-vs-monolith oracle
diffgemma probe-device; diffgemma summary; diffgemma config
diffgemma manifest                          # kernel FC-axis TOML

# -m accepts a local directory OR a bare HF repo id (org/name): an existing
# directory is used as-is; otherwise the newest matching snapshot in the
# local HF cache is resolved, or the exact `hf download` remedy is printed.
# Works for every command that takes -m, including quantize/repack source.
diffgemma chat -m google/diffusiongemma-26B-A4B-it
diffgemma chat -m mmastrac/diffgemma-26b-a4b-it-q4   # a downloaded pack, read-only

# Fetch a monolithic pack from HF — the distribution format (layered/overlay
# packs are local-only dev tooling, never what ships to users). Verifies the
# transfer (manifest version, blob length) and prints a one-line pack
# summary once done.
diffgemma download --repo mmastrac/diffgemma-26b-a4b-it-q4 -o model/diffgemma-26b-a4b-it-q4

# Requantize from HF safetensors (source: a local dir or a repo id, per
# above — a repo id ALSO pins (repo, revision) exactly for --overlay below,
# in place of the single-symlink-hop auto-detect).
diffgemma quantize -m model/transformer -o model/diffgemma-26b-a4b-it-q4 --profile q4

# Custom quantization classes (ARCHITECTURE.md §8.2): --set class=format
# overrides classify_tensor's per-class output on top of --profile. Classes:
# experts (or experts.gate_up / experts.down separately), attn, dense,
# vision, sc. Formats: raw, q4, q6, q8, nvfp4 (per-class support varies —
# an unsupported combo is a fatal error before anything is written).
diffgemma quantize -m model/transformer -o model/pack-mixed --profile q4 \
  --set experts=q6 --set attn=nvfp4
# nvfp4x is sugar for --profile q4 --set experts=nvfp4.
diffgemma quantize -m model/transformer -o model/pack-nvfp4x --profile nvfp4x

# Layered/overlay packs (ARCHITECTURE.md §8.1) — LOCAL-ONLY DEV TOOLING for
# experiment arms, never the distribution format: raw tensors ref the HF
# base in ~/.cache/huggingface instead of being copied into the pack, served
# zero-copy via VA-splice (no head cache). Composes with --set: a class
# switched from raw to quantized just moves from an external ref to a local
# blob entry.
diffgemma quantize -m model/transformer -o model/pack-overlay --profile nvfp4 --overlay
diffgemma repack --overlay -m model/diffgemma-26b-a4b-it-q4 -o model/pack-overlay
# The dual: flatten a layered pack back to a self-contained (monolithic) one
# — no requantization, just a byte-copy driven by the manifest's own offsets.
diffgemma repack --monolithic -m model/pack-overlay -o model/diffgemma-26b-a4b-it-q4

# MLX reference comparison (SERIALIZE with our runs — never in parallel)
python/.venv/bin/python python/scripts/mlx_generate.py -p "..." -o /tmp/mlx.json
```

MLX parity tooling (`python/scripts/`): `mlx_generate.py`,
`compare_generation.py`, layer-cos + denoise-trace dump/compare pairs.
ALWAYS prompt-match layer-cos comparisons.

Debug/probe env flags (`DGQ_TRACE_*`, `DGQ_LOG_*`, `DGQ_MEM_WATCH`, …) are
documented in `src/flags/`.
