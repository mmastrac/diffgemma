# diffgemma-mps — engineering strategy for agents

Read this before writing kernels, tests, or chasing bugs. It is not a task list (that's `PLAN.md`) or a data archive (that's `NOTES.md`). It is **how to work on this codebase without repeating the mistakes that have already cost days.**

The project: a Rust + Metal inference engine for DiffusionGemma (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon. Model semantics are in `ARCHITECTURE.md`; authoritative numeric behavior is in the CPU reference (`src/kernels/cpu.rs`, `sample.rs`) and the manifest (`model.dgq.json`).

---

## 1. The one thing to internalize

**Every serious bug in this project has lived in a fused or accelerated GPU path that a slower reference path got right.** MPS-Q4 producing uniform logits, the SC GEMM transpose, the softmax grid collapse, the MoE route-garbage from a last-expert `n_tok` bug — all the same shape: the optimized path computes a different function than the reference, parity is green because the optimized path has no golden, and the symptom only appears downstream (layer 2+, entropy collapse, pad output) far from the cause.

The corollary that governs everything below: **an untested path is where the next bug is.** Speed without a per-path correctness gate is how bugs ship. So the strategy is not "go fast"; it is "make divergence impossible to introduce silently, then go fast."

---

## 2. Diagnostic discipline (how to chase a bug)

When output is wrong, follow this order. Do not skip to optimization or to rewriting a kernel you can't explain.

1. **Localize before you theorize.** Find the *smallest* unit that diverges from the reference. A 30-layer entropy collapse is not a bug location; it's a symptom. Bisect: which layer, which kernel, which stage within the kernel. The MoE hunt took ten turns because we theorized about kernel math for several rounds before dumping the one value (`x`/`tok` as the kernel actually read it) that localized it in one read.

2. **Same input, same weights, two paths.** The fastest localization is always: run the suspect GPU path and the CPU reference on *byte-identical* input and weights, compare cosine. cos > 0.999 = that stage is fine. cos ~0.5 = correlated-but-wrong (often a swap, a scale, or partial corruption). cos ~0 = orthogonal, reading the wrong data entirely. The cosine *magnitude* is a diagnostic, not just pass/fail — read it.

3. **Dump the actual bytes/values the kernel reads, not what you think it reads.** Repeatedly, the bug was "kernel reads a different thing than the reference" — wrong row, wrong tensor, wrong activation. A CPU transliteration of the *source* can match the reference while the *GPU execution* diverges, because the transliteration can't reproduce threadgroup semantics, arena bindings, or route resolution. When in doubt, write the kernel's actual inputs to a scratch buffer and read them back.

4. **Impossible numbers mean wrong N or wrong normalization.** Entropy > ln(N), Z=0, cos > 1, values at ~1e38 — these are never "the model is just bad." They are indexing, normalization, or precision-overflow bugs with a specific cause. Chase them as such. (Softmax entropy 3.84 > ln(14) turned out to be the reader confusing 14 prompt tokens with 270 attention keys — a presentation bug, but the "impossible" was the tell.)

5. **Two paths failing differently is a gift, not noise.** When the engine and monolithic paths diverge on the same input, the *difference* localizes the bug to the path-specific code. One exploding (inf), one inert (zero) meant two different bugs in the same conceptual spot — and finding one explained the other.

6. **Don't let a workaround end the investigation.** Swapping a broken fused kernel for a slow reference path unblocks convergence but leaves the latent bug in every *other* kernel that shares the flawed pattern. Root-cause it, then decide whether to fix or replace. (The fused-MoE bug, had it been swapped-not-fixed, would have left the same `dequant_q4_group` usage suspect everywhere it appears.)

7. **A contradiction is a second bug, not an anomaly.** "act went 0.015 → 1.0 but gpu_out is bit-identical" cannot happen in a correct pipeline. When a fix changes an intermediate but not the output, you are measuring two different code paths (probe vs production) or reading a stale buffer. Resolve contradictions; do not note-and-move-on.

---

## 3. Measure before optimizing — always

Several regressions came from optimizing against an unmeasured or instrumented baseline:
- The SC GEMM "fast path" regressed to ~130 s/step because it reorganized a 262144-long contraction into a cache-hostile orientation.
- A "correct now" step measured 12 s — worse than the prior 4.8 s — because debug probes with full-buffer readbacks were still live, the slow decomposed MoE fallback was active, and native-Q4 dense (0.18 TFLOP/s) had replaced MPS dense (2.22 TFLOP/s).

**Rules:**
- **Get a clean measurement first.** Compile out probes/readbacks before timing anything. A readback is a GPU pipeline stall; instrumentation can dominate a step.
- **Attribute the time before reducing it.** Per-dispatch timing on one clean step. Do not guess which stage dominates; the guesses have been wrong.
- **Know the regime.** On M3-class at canvas=256 the step is **compute-bound on f16 matmul**, not bandwidth-bound. Weight read is ~70 ms; the GEMMs are the cost. This means: (a) smaller quant formats (fp4/fp8) buy ~nothing in speed here — they're dequant-to-f16 anyway, no native low-bit compute on any Mac; (b) MFU and dispatch/round-trip overhead are the levers, not bandwidth tricks.
- **Sequence optimizations to the bottleneck.** ICB, step-distillation, canvas-128 are *sub-second-step* optimizations. At multi-second steps the win is "stop doing the slow/temporary thing" (remove probes, use the fused kernel not the reference, re-enable MPS dense if its correctness bug is fixed). Don't reach for architecture when the cost is a left-on debug path.

---

## 4. Kernel variants: one body, compile-time specialization

Never fork a kernel into `k_foo`, `k_foo_fp8`, `k_foo_debug`, `k_foo_mps`. Forks drift, and drift is the bug source. Instead:

- **One source body per logical kernel.** Variant axes (dequant format q4/mxfp4/nvfp4/q8, accumulation dtype, dense backend, dump-depth) are **function constants** selected at pipeline-compile time. A "variant" is a tuple of constant values, not a file. The matmul loop exists once; a format bug cannot exist in one variant and not another.
- **Intermediate dumps are a compile-time mode of the production kernel,** not a separate probe kernel. A `DUMP_STAGE` function constant writes a chosen intermediate to a scratch buffer; production compiles it out (writes vanish). This prevents probe-vs-production drift — the exact failure behind the "act fixed, output unchanged" contradiction.
- **Fold hunt-time probe kernels back into the main bodies** behind the dump flag once a bug is found. Do not leave parallel probe kernels in the tree.

---

## 5. Testing: three tiers, push assertions down

The tests were slow *and* missed the bugs because they ran whole pipelines to exercise small logic. Fix both at once by pushing assertions to the smallest unit with the smallest fixture.

**Tier 1 — per-kernel unit tests. Synthetic, blob-free, milliseconds. Run on every save.**
One test per kernel against a CPU transliteration of *that kernel*, on a tiny hand-built fixture (e.g. 2 experts, 64 hidden, 4 tokens) — **never the 15 GiB blob**. The moment a "unit" test mmaps the real model it leaves the inner loop. This tier catches the entire class of bugs that took multi-turn hunts (route resolution, decode K-order, transpose, grid collapse) in milliseconds. Every kernel has a permanent CPU twin; the twin is the oracle forever. Promote ad-hoc hunt transliterations (e.g. the MoE mirror) into permanent Tier-1 references.

**Tier 2 — staged comparison. Real weights, reduced stages, flagged dump depth. Seconds. On demand / pre-push.**
2–3 layers, 1 step, intermediate-dump ON, comparing GPU vs CPU (vs MLX where available) at each stage. Catches *integration* bugs between correct kernels — wrong wiring, wrong buffer, stale read — that unit tests can't. Bounded (few layers/steps), not the full matrix.

**Tier 3 — end-to-end goldens. Full stack. Minutes. CI only, not the inner loop.**
Full 30L, real prompt, token-id match; the ship gates. These are regression gates, **not** debugging tools. Using Tier 3 to localize a bug is the slow path that caused the hunts.

**Transitivity is the speed win:** if monolithic `k_moe_grouped` and engine `f32_q4_linear_grouped` are each pinned to the *same* CPU oracle in Tier 1, "do the two engines agree" is automatic — you never need a slow engine-vs-engine end-to-end comparison to find a divergence. The 476 s engine-vs-monolithic trace becomes unnecessary.

**Variant matrix:** Tier-1 + Tier-2 run as a cross product of variant tuples ({format} × {accum} × {dense backend}) against the oracle, each cell with a characterized tolerance (fp4 looser than q4). A new variant is not "done" until it has a passing matrix row. `bench-matrix` mirrors this for perf. This makes "should we use fp4" a measured table cell, not a debate — and makes a silently-wrong fast variant impossible to ship.

---

## 6. Non-negotiable invariants (cheap checks that catch catastrophes)

Property assertions need no oracle and catch the "catastrophic but novel" class that no fixture anticipated. Wire these into the cheapest tier that can run them:

- **Finite:** no NaN/Inf in logits, activations, attention output, SC signal. (Caught: NaN-from-unzeroed-buffers, inf-from-bad-GEMM.)
- **Softmax rows sum to 1.0** over their actual support, every softmax kernel. (Would have caught the grid-collapse Z=0.)
- **Entropy ≤ ln(N)** over N keys/classes. (Would have flagged the impossible-entropy presentation bug.)
- **SC signal finite and non-zero on step ≥ 2.** (Would have caught both SC bugs — inf and inert.)
- **Determinism:** same seed → same tokens across runs on the deterministic path. (Caught the original MPS nondeterminism.)
- **Not all-pad / all-filler** on a converged block before early-stop fires. (The premature-commit quality bug.)
- **Offsets in `ulong`** for all blob addressing — the blob exceeds uint32; a uint intermediate truncates silently.
- **Quant K-order:** sequential `dequant_q4_group[m]` == `q4_weight_at(row, base+m)` in K-order, not just as a set (VERIFY-K — the blind spot that hid behind VERIFY-N).

---

## 7. Authority & sources of truth

- **Numeric behavior:** the CPU reference (`sample.rs`, `kernels/cpu.rs`) is the oracle. When GPU and CPU disagree and parity-vs-HF historically passed, the CPU is right and the GPU path is the suspect.
- **Weight layout:** `model.dgq.json` manifest is authoritative for shapes, offsets, and tensor orientation. When a kernel's addressing assumption (stride, transpose, fused-vs-separate tensors) is in question, the manifest decides — not memory, not the comment. (The SC GEMM `A@W` vs `A@W^T` and the MoE fused-1408 questions were both settled by the manifest.)
- **Model semantics:** `ARCHITECTURE.md` and the spec section of `NOTES.md` (RoPE split-half pairing, proportional-RoPE full-head-dim denominator, temperature count-down, test-before-add accept rule, QK-norm folding the attention scale, no separate `query_pre_attn_scalar`). These are checkpoint-specific and several are counterintuitive — do not "correct" them from general Gemma knowledge without checking the reference.
- **Do not trust comments over code/manifest.** A stale comment ("entropy before temperature") has already misled. Verify against the authoritative source.

---

## 8. Working rules for agents

- **State assumptions as checks, not beliefs.** "The stride is K+2" → assert it against the manifest and a byte dump, don't assume it.
- **One bisecting measurement beats three rounds of theory.** When stuck, find the cheapest dump that splits the hypothesis space in half, and take it before proposing fixes.
- **Every bug fixed gets a regression test at the lowest tier that would have caught it.** This is how the test suite stops missing the bug class it keeps missing.
- **No path ships without a golden.** If you add a kernel variant, dense backend, or code path, it gets a matrix row before it's "done." Untested path = next bug.
- **Don't optimize against a dirty baseline.** Probes out, path confirmed, time attributed — then optimize.
- **Keep `PLAN.md` (forward work) and `NOTES.md` (data/history) current;** record resolved bugs in `NOTES.md` so they can't be re-litigated, and re-file open issues in `PLAN.md` by their P-number.
