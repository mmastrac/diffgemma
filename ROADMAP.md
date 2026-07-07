# diffgemma-mps — 4-week road to production-ready

Written 2026-07-07 from a deep pass over PLAN.md / SPEC.md / KERNELS.md /
STRATEGY.md / NOTES.md, the agent-memory disproof ledger, the mlx_vlm
reference source, and the TurboQuant literature. Everything cited here is
measured on this machine (M3 Pro 36 GB) unless marked *estimate*.

---

## 0. Where the engine actually is (evidence snapshot)

| Axis | State | Evidence |
|---|---|---|
| Short/medium chat wall-clock | **Beats MLX-4bit (their fastest config) on every probe** | capital 2.9 vs 3.7s; sky ~410tok 22.0 vs 27.6s; transformer ~840tok 50.2 vs 59.2s |
| Denoise convergence | Parity-class: ~1.15× steps vs MLX best config, ~1.6× faster than mxfp4 | 7 matched-canvas pairs, multi-seed |
| Long context | **Works to 105k, exact needle retrieval**; KV 2.4 GiB; ~4.5 s/step at 100k | needle probes 6.5k/33k/105k |
| 100k prefill | ~15.5 min (batched; was 18) | 33k = 142 s measured, linear-ish extrapolation |
| Quality gates | spec-seed 17/17; genuine multi-seed 47/51; wart census 0/10 | gate re-run after every landing |
| Test suite | 708 unit + oracle matrix green; 7 ignored (model-dir) | `cargo test` |
| Known quality debt | seed-123 empty-reply artifact (5-6 short factual prompts at that seed) | pre-existing, trajectory-level |
| Product surface | CLI `ask`/`chat` (+`--ctx`, JSONL events); no server, no vision, single session | — |
| Ops surface | no CI, no packaged release, quantize pipeline is a CLI subcommand, first run pays Metal JIT | — |

**Perf levers already BUILT AND DISPROVEN — do not re-chase** (each has
machinery + a recorded verdict in flags.rs/memory):
sequential-kv-block "flash decode" (SLC already lockstep-served, 171 > 150
GB/s); q8 KV for speed (+9% prefill / +54% denoise at 33k — kernels are
issue-bound, not byte-starved; memory lever only, `DGQ_KV_Q8=1`);
TurboQuant-class low-bit KV *for speed* (same physics); MoE tiny-M tiles;
partial-forward; f32 hidden; f16 arena; ICB.

---

## 1. What "production-ready" means here (proposed acceptance)

1. **Install-to-first-token < 30 min on a clean 36 GB Apple Silicon Mac**
   (documented model fetch + quantize + warm), < 10 s model load thereafter.
2. **A local HTTP server** (OpenAI-ish chat completions + streaming) in
   addition to the CLI; concurrent-request safe (serialized generation,
   queued requests — the GPU is single-tenant by hard rule).
3. **No panics reachable from user input.** Context overflow, over-long
   prompts, bad UTF-8, missing files → typed errors with actionable messages.
4. **Quality**: spec-seed gate 17/17 in CI nightly; multi-seed aggregate ≥
   the 47/51 baseline; needle exact at 33k in nightly.
5. **Perf floor** (regression-gated numbers, not aspirations): ≥ 15 tok/s
   mid-length replies; ≤ 1.1 s/step at chat lengths; 33k prefill ≤ 140 s;
   100k functional ≤ 16 min prefill, ≤ 5 s/step.
6. **Docs**: README quickstart, SPEC/KERNELS current, model card + license
   notes, a benchmark page with the MLX head-to-head methodology.
7. **Scope decision recorded**: vision (VLM) in or out for v1 (recommend OUT,
   see §5.3 — it's the single biggest unscoped item).

---

## 2. Week 1 — finish the perf story (kernel work, all gate-bounded)

### 2.1 Weight-stationary expert GEMM — DONE 2026-07-07: DISPROVEN
Built (`DGQ_MOE_PREFILL_BM=64|128`, default 32/off, machinery kept): the
batched-prefill block list is built at a taller block height and consumed by
the same tunable sparse kernel compiled at that TUNE_BM, so one threadgroup
owns all (most) of an expert's ~64 rows = one dequantized weight stream.
Bit-identical PROVEN (6.5k needle: full 30-layer KV dump byte-equal and
tokens identical at 32/64/128); suite 708/708; spec gate 17/17.
- **Result**: BM=64 is a WASH (21.1s vs 21.1s 6.5k prefill, alternating warm
  runs); BM=128 is 3.6x SLOWER (76s — TM=8 means 64 f32 accumulators/lane,
  the known register-spill regime).
- **Why (roofline, real dims)**: expert W per layer = 128 x 3.72 MB q4 =
  476 MB → ~7.6 ms/layer at DRAM even WITH the 2.4x per-block re-reads; MoE
  MMA at M=1024 ≈ 110 GFLOP/layer ÷ ~2.3 TF/s ≈ 48 ms/layer. The expert
  GEMM is COMPUTE-bound by ~6x — cutting W bytes can't help. The premise
  ("win capped by weight re-reads", eb0edba) was a wrong attribution; the
  cap is kernel TF/s. Raising prefill MoE throughput now means raising the
  GEMM's TF/s (E5-class fragment work), not byte reduction.

### 2.2 Retire the f32 engine prefill — DONE 2026-07-07: DISPROVEN, keep the engine
Forced `DGQ_FAST_PREFILL=1` for ALL lengths, multi-seed gate {7,42,123}:
**41/51 (16/15/10) vs engine baseline 47/51 (17/17/13)** — a 6-point
regression, all in the EMPTY-REPLY / non-convergence class on short factual
prompts (blue_plus_yellow, opposite_of_hot, one_plus_one … "answer BAD",
blank output). Fast prefill's bf16-perturbed prompt KV aggravates the
seed-123 empty-reply artifact and spreads it to seed 42. Fails the ≥47/51
ship gate on quality alone (wall-clock moot). **VERDICT: keep the f32 engine
for short prompts — the length heuristic is a QUALITY mitigation, not just
wall-clock. Strong mechanistic clue for E6: the empty-reply artifact is
prefill-KV-precision-sensitive (engine f32 forward = partial cure).**

### 2.2b (superseded) Retire the f32 engine prefill for short prompts — 1 day
Short prompts (≤256) still run the legacy f32 engine at session open
(~0.85 s steady + one-time shader JIT; ~2 s cold). Post-no-freeze evidence
says fast prefill is adherence-clean (17/17 forced at seeds 7/42); the
length heuristic survives on wall-clock grounds only (+convergence steps on
some prompts).
- **Experiment**: multi-seed gate + census with `DGQ_FAST_PREFILL` forced for
  ALL lengths; measure convergence-step delta across the gate set and total
  turn wall-clock (fast prefill ~0.8 s floor vs engine 0.85 s + occasional
  cold JIT).
- **Ship if**: aggregate ≥ 47/51 and mean turn time not worse. Payoff: delete
  an entire engine dependency from the hot path (engine becomes pure
  validation), kill the cold-start JIT wart, simplify session open.

### 2.3 Canvas shrink near max_tokens (MLX parity, divergence #5) — 1 day
MLX shrinks the canvas to `max(remaining, 64)`; we always run 256-wide.
Matters at reply tails and tight `--max-new-tokens`: the finish stages
(lm_head + SC softembed, ~0.25 s/step) scale with canvas rows.
- **Experiment**: implement denoise-side M=`min(CANVAS, remaining)` rounded
  to 64; gate multi-seed (trajectory-affecting: canvas width changes the
  entropy pool → NOT bit-identical).
- **Expected**: minor wall win on capped replies; main value is closing a
  deliberate divergence before v1 and matching MLX's `min_canvas_length`
  semantics. Low risk, bounded.

### 2.4 q8-KV at 100k A/B — DONE 2026-07-07: table CLOSED, no default change
Measured per the timing methodology (isolated `bench-step-kernel
--layers 30 --kv-len 105000 --profile-steps 4`, full KV allocated, 3
interleaved rounds per arm — NOT full-prefill wall-clock):
- **The sign flips at 100k but only to −6%**: pre_moe (attention)
  4.14 → 3.87 s/step, total 4.66 → 4.39 (every q8 step beat every f16 step
  across all rounds). Under the ≥15% adaptive-flip gate → default stays f16
  at every length; q8 remains opt-in.
- **Why so small**: even at kv=105k the full-layer sweep reads ~430 MB/layer
  in ~740 ms ≈ 580 GB/s effective — still SLC-served lockstep, still
  issue-bound; unique DRAM traffic (~430 MB once) is milliseconds. The
  33k physics holds at 105k.
- Updated guidance: q8 = the KV MEMORY lever (halves KV), and at 100k+ it's
  additionally ~6% faster — win-win for very long contexts.
- Retrieval at 105k under q8 run separately (untimed, correctness only).
- **SHIPPED 2026-07-07: q8 auto-enables at long context** (`kv_q8(max_seq)`
  scales with the captured working-set cap; `DGQ_KV_Q8=1/0` overrides). f16
  @262k SWAPS (91% + 704 MB, pressure WARN via DGQ_MEM_WATCH); auto-q8 keeps
  it at 82%. Threshold ~178k on 36 GB, lower on smaller Macs. Short prompts
  stay f16 → gate 17/17 unchanged.

### 2.5 Attention issue-bound follow-up (timeboxed research, ≤ 1 day)
Full-layer attention at 100k ≈ 3 s/step is instruction-issue-bound. The one
untried shape: fewer, larger MMA fragments per stream (steel-style per-lane
`vec<T,2>` register tiles instead of `simdgroup_load`, as KERNELS.md maps
for GEMMs). Timebox a prototype on `attention_mma_full` only; if the bench
row doesn't show ≥ 1.5× at kv 32k, record and move on. Do NOT ship without
the full multi-seed gate (f16 math order changes).

---

## 3. Week 2 — correctness debt, robustness, CI

### 3.1 Seed-123 empty-reply artifact — 2 days, investigation playbook
The one standing quality bug: at seed 123, 5-6 short factual prompts emit
empty replies (both prefill paths; pre-existing across many changes).
- Localize per STRATEGY §2: trace a failing prompt (`--write-trace`,
  `DGQ_TRACE_ENTROPY=full`) and diff the first-step entropy/accept pattern
  vs a passing seed. Hypotheses to bisect: (a) initial-canvas collision with
  the stop-token region (canvas RNG at that seed seeds an `<eos>`-heavy
  region → stop-token trim fires on a garbage block), (b) first-block
  early-stop misfire (stop conditions on a canvas that converged to pad+eos),
  (c) chat-template/stop-token interaction.
  The stop-token TRIM path (`sequences.truncate(block_base + rel)`) is the
  prime suspect: an `<eos>` at offset 0 of block 1 yields an empty reply by
  construction.
- **Fix bar**: whatever changes is trajectory-affecting → multi-seed gate +
  census; target aggregate > 47/51 with seed-123 specifically improving.
- If root cause is "model genuinely emits eos-first on these canvases", the
  fix is a retry-with-reseed on empty first block (bounded, deterministic
  seed bump) — a product-level guard, gated the same way.

### 3.2 Panic-to-error audit — 1 day
Grep-driven: every `assert!`/`panic!`/`unwrap()` reachable from user input
becomes a typed error (the `set_kv_len` overflow assert, prompt >
context, tokenizer file missing, model dir malformed, `--ctx` beyond
memory). Add a `--ctx` memory estimator (we already print the budget table)
that refuses configs that exceed physical RAM with a clear message.

### 3.3 CI: three tiers wired to actual runners — 1-2 days
- **Tier 1 (every push, GitHub-runner-safe, no model)**: `cargo test`
  (708) minus `metal`-gated GPU tests… on a mac runner if available;
  otherwise compile + CPU-oracle subset. Clippy + fmt.
- **Tier 2 (self-hosted mac nightly)**: full suite + spec-seed smoketest
  17/17 + needle_3k exact + `--profile-steps` perf floor assertions
  (step ≤ 1.1 s at kv64, prefill 6.5k ≤ 21 s — 10% margins over current).
- **Tier 3 (weekly)**: multi-seed gate aggregate ≥ 47/51, wart census 0/10,
  33k needle, MLX head-to-head refresh (serialized!).
- Add `smoketest --json` output for CI parsing if not already parseable.

### 3.4 Session/robustness tests — 1 day
Multi-turn chat property tests: cross-turn KV reuse vs reset equivalence
(exists as ignored test — promote to Tier 2), context-full behavior, repeat
`--repeat` session-carryover loop (exists), kill/resume, empty prompt,
maximum prompt.

---

## 4. Week 3 — product surface

### 4.1 HTTP server mode — 3 days
`diffgemma-mps serve -m <model> [--ctx N] [--port]`:
- OpenAI-compatible `/v1/chat/completions` (stream + non-stream). The chat
  JSONL event protocol (`ChatEvent`) already models streaming with rewinds;
  map `Text{committed,..}` to SSE deltas (only emit committed-monotonic
  prefixes; diffusion rewinds stay internal — emit on commit, or expose a
  `x-diffusion-draft` extension event for UIs that want the shimmer).
- Single generation queue (GPU is single-tenant — hard rule); per-request
  seeds; cancellation (client disconnect → abort block loop between steps —
  needs a cancellation check in `generate_with_session`'s step loop, ~1 line
  per step).
- Reuse cross-turn KV: key sessions by conversation prefix (the
  `kv_valid_tokens` diff already handles arbitrary prefix reuse).

### 4.2 Sampler/feature parity with MLX — 1-2 days
From the mlx_vlm source diff (measured surface, not guesses):
- **`temperature` user knob**: expose categorical sampling (machinery exists
  behind `DGQ_DENOISER_ARGMAX=0`) as a per-request parameter; default stays
  argmax (their default is temperature 0 too).
- **`confidence-threshold` sampler** (their alternate accept rule,
  `diffusion_threshold=0.9`): implement in the GPU sampler as a variant of
  the accept kernel; compare convergence on the gate set. Ship only if
  neutral-or-better; otherwise document as unsupported.
- **`min/max_canvas_length`** — covered by §2.3.
- **NOT porting**: `diffusion_compile`, `static_cache`, unmasking display
  (we have inline dimmed streaming), `prefill_step_size` (ours is batched
  differently and faster per token).

### 4.3 Install & model pipeline UX — 1-2 days
- `diffgemma-mps fetch` (or documented `huggingface-cli` recipe) +
  `quantize --profile q4` one-liner; verify the 15 GiB disk headroom check
  and resume-safety of the converter.
- Warm-start: persist the Metal pipeline archive (exists —
  `DGQ_METAL_PIPELINE_CACHE`) by default; document first-run JIT.
- Homebrew formula / notarized binary decision (stretch).

### 4.4 Docs — 1 day
README quickstart (install → chat → serve → long-context flags), perf page
with the MLX methodology (matched canvas, temp 0, natural finish,
serialized runs), flag reference generated from `flags.rs` doc comments.

---

## 5. Week 4 — release engineering + the two big scope calls

### 5.1 Release hardening — 2 days
Version tagging, CHANGELOG from the commit narrative, binary size pass,
`--version` with model-format compatibility check (`.dgq` manifest version
gate — old q8-embed blobs still load, verify), final full-matrix run:
suite + gate 3 seeds + census + needle {6.5k, 33k, 105k} + MLX head-to-head.

### 5.2 TurboQuant / rotated-KV — IN PROGRESS (moved up from Week 4, 2026-07-07)
TurboQuant (arXiv 2504.19874) / QuaRot: an orthogonal rotation → near-uniform
coordinates → outlier-free low-bit quant. Our measured physics says low-bit
KV is a MEMORY lever only (issue-bound kernels), so this targets **262k
contexts** (f16 KV 5.3 GiB → ~1.4 GiB at q4) / smaller Macs. **Do not do
this for speed** — the disproof stands (E4: q8@100k only −6%).

**PREMISE CORRECTED 2026-07-07 (the old text below the line was wrong).**
The earlier claim "fold R into (W_q,W_k) and (W_v,W_o) offline, zero runtime
cost" is FALSE for K, and the V half needs rework too:
- **K cannot fold offline.** RoPE sits between W_k and the KV store
  (`qk_rope_kv.metal` applies RoPE, then writes K), and a Hadamard doesn't
  commute with position-dependent RoPE. K needs a RUNTIME Walsh-Hadamard on
  `head[]` AFTER RoPE, plus the same H on Q after its RoPE — `(Hq)·(Hk)=q·k`
  keeps scores exact. Cheap: O(hd·log hd), hd=256/512 both pow2.
- **The offline W_v fold is nearly worthless.** Full layers (5,11,17,23,29)
  have NO separate `v_proj` — V aliases the raw k_proj output — and those
  full layers store KV LINEARLY (sliding layers ring-cap at 2048), so they
  hold ~2.1 of the 2.4 GB at 105k. The memory that matters is exactly the
  layers with no W_v to fold.
- **Winning design = uniform runtime rotation, only W_o folded offline.** In
  `qk_rope_kv`, after RoPE apply H to q / k / v (v has no RoPE); store q4 K+V.
  Attention reads rotated q (arena) + rotated q4 K/V — scores and output
  `Σattn·(Hv)=H·o` come out right with NO rotation logic in the attention
  kernels (they need only q4 dequant). Unrotate V via W_o offline: fold Hᵀ
  into each q-head hd-block of W_o. Q rotation self-cancels in the QK dot.
- **Milestones**: M1 q4-KV format end-to-end, no rotation (mechanical, mirror
  the KV_Q8 lattice → KV_Q4; establishes the degraded baseline) → M2 runtime
  Hadamard (q/k/v) + offline W_o Hᵀ fold, prove rotated-q4 recovers quality →
  M3 non-expert bf16 weights → M4 rotated experts (§5.4, E9, v2+). Full
  corrected math + code map: memory `turboquant-rotated-kv`. WHT helper
  shipped (`src/kernels/sub/hadamard.rs`, 834cc1d). Gate as usual.

### 5.3 Vision (VLM) — decision, not implementation
The checkpoint ships a vision tower (mlx_vlm serves images; our config.rs
parses `vision_config` but the engine is decoder-only). Porting = SigLIP
encoder + image token splicing + mm masks — realistically 2+ weeks alone.
**Recommendation: v1 is text-only, stated in the README; vision is the v2
headline.** If overruled, this displaces §4.1-4.3 and the month becomes
perf-freeze + vision.

### 5.4 Rotated experts (E9) — far-future / v2+, experts LAST
The same Hadamard rotation applied to the MoE expert weights. Value is
**near-bf16 fidelity within the mandatory 4-bit budget** (bf16 experts don't
fit; the q6 experiment showed spending bits barely moves warts, so rotation
attacks CONDITIONING — a different axis — not size). Only start after the
KV rotation infra (E8) is proven bit-exact; build experts last; quality
change → full multi-seed gate + wart census + sign-off.

Two sub-cases, different cost — and the "gate/up is free" framing needs a
correction (verified against the code 2026-07-07):
- **gate/up.** Both read the pre_ff_ln_2-normed hidden (`moein`, one shared
  buffer). The rotation folds into the WEIGHTS offline (W_gate·Rᵀ, W_up·Rᵀ),
  but feeding the rotated input `R·moein` is NOT zero-cost: it needs either
  (a) ONE runtime WHT on `moein` per token — cheap, shared across all 128
  experts (moein is expert-branch-local; the router uses a *separate* norm
  `router_scale`, so it's not a shared fold) — or (b) a global
  residual-stream rotation (QuaRot computational invariance: rotate once at
  embed, unrotate at lm_head) which makes it truly free but is a much bigger
  change. So gate/up = "one cheap shared WHT," not "free." Covers 2/3 of
  expert params; the whole bet until proven insufficient.
- **down_proj.** `out = h·W_downᵀ`, `h = silu(gate)·up` — the nonlinearity
  blocks any offline fold; needs a runtime WHT on `h` (moe_ff 704 → pad
  1024) every denoise step. Contingent, hot-path.

Milestones: M1 gate/up-only (shared WHT + offline weight fold, gated vs a
bf16-expert reference — STOP here if it reaches near-bf16) → M2 down_proj
WHT (only if M1 short; measure with `DGQ_MEM_WATCH` + `bench-step-kernel`
A/B first, abort if it regresses the short-context, non-memory-bound case) →
M3 optional sub-q4 gate/up for memory (separate gate). Headroom evidence
(q6 = warts unchanged) shows expert-quant error isn't today's wart driver
but does NOT prove rotated-q4 recovers bf16 or that q3 is safe.

**Prior-art specifics + build-order correction (2026-07-07).** "TurboQuant
for weights" is NOT novel: TurboQuant (2504.19874) never quantizes weights
(KV + nearest-neighbor only), and its stage-2 QJL-residual trick only buys
*inner-product* unbiasedness — an attention concern, irrelevant to weights
(we want MSE). The real weight-domain twins are **PolarQuant** (2603.29078:
128-block L2-norm → Walsh–Hadamard → Lloyd–Max centroids fit to 𝒩(0,1),
calibration-free) and **QAM-W** (2605.26339: Hadamard + 2D codebook +
activation-aware scaling); lineage QuIP/QuIP#/QuaRot/SpinQuant. So E9 is a
*port*, not research — benchmark to beat is PolarQuant/QuIP#, not "is it new."
The one actionable result they surface: **PolarQuant measured Hadamard
rotation = ~98% of the quality gain (PPL 6.90→6.40 at Q5); optimal Lloyd–Max
centroids add only ~1%.** This matches our q6 finding (bits ≈ no wart
movement) — the lever is CONDITIONING, not the codebook. So **M1 should prove
the rotation with PLAIN absmax/uniform q4 first** and treat the good
quantizer as a near-non-lever; only revisit optimal 𝒩(0,1)/Beta centroids
for the M3 sub-q4 case, where the centroid share grows (TurboQuant's whole
point is centroids dominate at low bit — 98% is a Q5 number, not a q4/q3
one). PolarQuant's 128-elt intra-block matches our existing WHT helper block
size; all of it is calibration-free (no new data pipeline).

### 5.5 Slack (2-3 days)
History says every week here surfaces one unplanned wall (this week it was
the sampler-struct growth ripple). The slack is the plan.

---

## 6. Experiment playbook index (single table)

| # | Experiment | Hypothesis | Cost | Ship gate | Abort criterion |
|---|---|---|---|---|---|
| E1 | Weight-stationary expert GEMM | ~~33k prefill −20%~~ **DISPROVEN 2026-07-07**: BM=64 wash, BM=128 3.6x slower; expert GEMM at M=1024 is compute-bound (~6x margin over W bytes) | done | bit-identity held (KV byte-equal), gate 17/17 | hit: < 5% → machinery kept, default off |
| E2 | Fast prefill for short prompts | turn time neutral, engine retired | 1 d | multi-seed ≥ 47/51 + census | adherence regression |
| E3 | Canvas shrink at tail | MLX parity, small tail win | 1 d | multi-seed gate | quality regression |
| E4 | q8 KV @ 100k | ~~q8 wins where DRAM-bound~~ **CLOSED 2026-07-07**: −6% at 105k (issue-bound); but f16 @262k SWAPS (91%) → **q8 now AUTO-enables at long ctx** so long sessions stay resident | done | q8 = memory lever; auto above ~178k (scales with cap); gate 17/17 (short prompts stay f16) | hit: table closed |
| E5 | Fragment-tile attention | ≥ 1.5× at kv 32k | 1 d timebox | bench row, then full gate | < 1.5× on bench |
| E6 | seed-123 artifact root-cause | ~~stop-token trim~~ **CLOSED 2026-07-07**: NOT a forward bug — MLX-faithful (empties on the identical canvas too); intrinsic (prompt,canvas)→eos/ceremony attractor. **FIX SHIPPED default-on**: `DGQ_EMPTY_REPLY_RETRY=3` re-rolls the first block's canvas on a degenerate pos-0 (eos/stop/control token) | done | seed-123 answers 13→17; seeds 7/42 unchanged 17/17 (no-op on good replies); suite 714/0; deterministic per --seed | hit: fix landed |
| E7 | confidence-threshold sampler | parity feature, maybe faster stops | 1 d | gate neutral | worse convergence |
| E8 | Hadamard-rotated q4 KV (§5.2) | **IN PROGRESS 2026-07-07**: 262k KV 5.3→~1.4 GiB, quality-neutral. Premise corrected — K rotates at RUNTIME after RoPE (can't fold), V rotates at runtime too (full layers alias V/no W_v), only W_o folds offline | ~3 d | rotated-q4 passes multi-seed gate + needle 33k/100k exact + DGQ_MEM_WATCH < 90% @ 262k | rotated-q4 fails gate (rotation doesn't recover q4 quality) |
| E9 | Rotated experts (near-bf16 fidelity @ q4) (§5.4) | M1 gate/up (shared WHT + offline fold) recovers ~bf16 forward; M2 down_proj WHT only if needed | v2+ | full gate vs bf16-expert ref + census + sign-off | M1 no better than plain q4, or M2 WHT regresses short-context |
| E10 | Precision-decay KV (recent f16, aged rotated-q4) | segmented full-layer KV: f16 recent window + q4 aged bulk, demoted+rotated as tokens age past the window; better quality/memory than uniform q4 | v2+ | needle 262k exact + DGQ_MEM_WATCH under budget + gate | on 36 GB marginal over q8-auto (which already fits 262k) — value is 18-24 GB / >262k / long-range quality |

Every experiment observes the standing rules: bit-identical ships on
identity evidence; anything else needs multi-seed gate + census + explicit
sign-off; serialize all model-loading runs; check the disproof ledger first.

---

## 7. Risk register

1. **Single-machine evidence.** All numbers are one M3 Pro. Prod claims need
   at least one M-series variant re-run (M1/M2/M4 differ in SLC + core
   counts — the SLC-locality disproof might not hold on M1). Mitigation:
   recruit one external config in week 3; gate perf floors per-machine.
2. **36 GB floor.** 19 GiB weights + scratch. 18 GB Macs need q6/q4 embed +
   rotated-q4 KV (E8) — currently out of scope; state minimum RAM in README.
3. **Upstream drift.** mlx_vlm's mxfp4 production bug got fixed upstream? Our
   head-to-head cites their 4bit; refresh before publishing benchmarks.
4. **Gate is 17 prompts.** It has been sensitive so far, but v1 confidence
   wants a broader eval (e.g. 100-prompt adherence set, run weekly, not
   blocking). Cheap to add in Tier 3.
5. **The seed-123 class** may be a model property, not a bug — the retry
   guard (§3.1) is the containment plan either way.
