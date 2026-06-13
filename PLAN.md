# diffgemma-mps — production plan

Low-dependency Rust inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon. Iris-style: mmap weights, CPU reference path, Metal acceleration.

This is the single forward-looking plan. Historical context, measured data, and resolved-issue archaeology live in `NOTES.md`. Model semantics live in `ARCHITECTURE.md`.

---

## Where we are (2026-06)

End-to-end generation works on GPU through **two** engines that share weights, tokenizer, and sampler semantics:

- **Engine** (`generate-gpu`): multi-kernel `.dgq` path. Proven-correct, golden-locked. ~3.89 tok/s @ 30L (M3 Pro), ~7.87 s/step forward.
- **Monolithic** (`generate-monolithic`): single-encoder step kernel (`shaders/diffgemma_step.metal`). **~6.58 tok/s @ 30L, ~39% faster forward (4.80 s/step).** Prefill/extend writers, GPU sampler, chat template all landed.

The monolithic path is the chosen production target. It is faster, and its single-command-buffer-per-step structure is the only path to interactivity.

**The honest status: we are "sort of working" but not interactive, and output quality on templated chat prompts is not yet a ship gate.** The two things between us and a shippable interactive tool are (1) a **quality bug** — premature block commit on degenerate argmax — and (2) **per-step latency** still in seconds, not the sub-second we need for chat. Throughput groundwork (quantize-to-residency, grouped MoE kernel, GPU sampler) is largely done; what remains is finishing GPU round-trip elimination and fixing the stop logic.

### Target

Sustained **>= 8 tok/s end-to-end on 24 GiB base M-series; 25+ on Pro/Max 36 GiB**, with *readable* chat output. Per-step latency <= 1.8 s on M3 Pro near-term, <= 1.4 s stretch.

### Non-goals (this cycle)

Training/fine-tuning/LoRA, multi-user serving, CUDA/Linux GPU, vision/multimodal (deferred; 355 vision tensors skippable via `--skip-vision`).

---

## The two-stage commit model (read this before touching the loop)

The single most expensive misunderstanding in this codebase. DiffusionGemma has **two distinct "commit" stages** and they are not the same number:

1. **Per denoise step — ~15-20 canvas positions.** Every forward computes logits for all 256 canvas positions. The entropy-bound sampler (`entropy_bound = 0.1`) freezes only the lowest-entropy positions whose cumulative entropy stays <= 0.1 — typically ~15-20 — and **re-noises the rest**. This is the intra-block convergence mechanism. It is *not* an "emit 16 tokens" mode.

2. **Per block end — 256 tokens.** When the block converges (early stop) or hits `max_denoising_steps`, the model takes the final argmax over all 256 positions, **emits all 256** to output, and runs causal encoder-extend on all 256 into the KV cache. Then a fresh canvas starts if more output is needed.

So: ~15-20 positions freeze per step; 256 tokens emit per finished block. **Do not "strip pads and continue" as a fix for bad output** — that diverges from official KV semantics. The correct fix is to stop the block from committing *before the canvas has converged*. See P1 below.

---

## Architecture (production target)

```
generate-monolithic
--------------------
prompt -> chat template -> tokenize
  -> GPU encoder prefill --> b4 kvcache (half, [pos][K|V][kv_head][dim] per layer)
  -> block loop (Rust driver, src/metal/step_generate.rs):
      CanvasState.ids init (256 random)
      for step in 1..=max_steps:
        encode_step (ICB replay): preamble (SC) + 30 layers + finish (lm_head/softcap/sampler)
        read stop_flag                       <- 1 sync/step, ~3 KB readback
      commit: read CanvasState.prev_argmax (256) -> emit -> GPU encoder-extend --> b4
  -> detokenize, strip pads for display only
```

Buffer ABI (fixed — version-bump to change), documented in `diffgemma_step.metal`:

| b | buffer | role |
|---|--------|------|
| 0 | `.dgq` blob | mmap quantized weights (zero-copy `MTLBuffer`) |
| 1 | `ModelLayout` | per-tensor byte offsets + per-layer metadata |
| 2 | `StepParams` | kv_len, sampler thresholds, temperature schedule |
| 3 | arena | f16 activations at byte offsets (`A_*`) |
| 4 | kvcache | half, `[pos][K\|V][kv_head][dim]` per layer region (NOT the legacy f32 `GpuKvCache`) |
| 5 | `CanvasState` | ids, entropy, accept, rng, step, stop_flag |
| 6 | logits | half `[256][262144]` (step N write -> step N+1 SC read) |
| 7 | `RouteScratch` | MoE routing buckets |

---

## What's already done (do not redo)

| Area | State |
|------|-------|
| Weight loading, config, CPU reference kernels, 30L decoder + encoder, KV cache | done |
| Entropy-bound block sampler (CPU reference, authoritative in `sample.rs`) | done |
| Tokenizer + chat template (HF-matched, `python/` uv parity tests) | done |
| Metal GEMM / attention / MoE; fused norm+QKV+RoPE+GQA+o_proj; fused FF | done |
| `.dgq` quantizer: 15.35 GiB q4 blob, q8 embed/lm_head/SC, mmap zero-copy load | done |
| Dequant-in-kernel residency: 0 expert evictions on `.dgq` (was 106k) | done |
| Grouped MoE kernel (gather + 2 dispatches/layer); dense via MPS | done |
| GPU router top-8 (f32 logits, parity tie-break); GPU full sampler (~3 KB/step) | done |
| Monolithic step kernel: 30L forward, GPU sampler, softcap, first-step SC skip | done |
| Monolithic KV: prefill writer, extend writer, b4 layout, `step-kv-check` | done |
| Monolithic generate loop + chat REPL; feature flag `DGQ_MONOLITHIC` / `--monolithic` | done |
| Determinism: bit-stable `generate-parity` with `DGQ_MPS_Q4=0` (CPU router/MoE/sampler) | done |
| Monolithic M0-M5 core gates, raw-prompt goldens | done |

**Measured baseline (M3 Pro 36 GiB, `/tmp/quantized-weights`):** `bench-step` engine 7.87 s/step (152 syncs, ~1.3 GiB readback); monolithic `bench-step-kernel` 4.80 s/step forward; `generate-monolithic` 6.58 tok/s (8 steps/block); pipeline compile cached ~3 ms. Full Q0 device numbers and the throughput model are in `NOTES.md`.

---

## Plan forward

Three tracks. **Quality (P1) is the ship blocker and comes first** — a fast engine that emits pads is not shippable. Perf (P2) and hardening (P3) proceed after the loop produces readable text.

### P1 — Make output correct & readable (ship blocker)

Root cause, confirmed: with early stop on and `--steps` defaulting to 2, blocks commit after ~2 steps when argmax stabilizes on **degenerate tokens** (`<pad>`=0 or filler 262143) — confident, but before the canvas has converged to real text. The ~15-20/step freeze never gets enough steps to fill the canvas, so the 256-argmax emit is mostly pad.

| # | Task | Exit |
|---|------|------|
| P1.1 | **Pad-aware early stop** — do not count all-pad / all-filler stable argmax as convergence. Require a minimum of real (non-pad) committed positions and/or `steps_eff >= ~12` before early stop may fire. | Stop never fires on an all-pad canvas |
| P1.2 | **Chat-oriented CLI defaults** — `chat` / interactive generate default to `--steps 48` (or until convergence) with the model-card stop, not the parity default of 2. Keep `--steps 2` only for bench/parity. | Default `generate-monolithic -p "..."` yields readable text |
| P1.3 | **`steps_eff` + accepted-positions telemetry** — per-block effective steps and a histogram of accepted positions/step, so "did the canvas converge" is observable, not guessed. | Telemetry visible in generate output |
| P1.4 | **Decode hygiene** — display only non-pad new tokens; block commit still emits the full 256 argmax per official KV semantics. | Chat preview shows clean text; KV unchanged |
| P1.5 | **Templated-chat quality gate** — add a templated-prompt golden or small eval harness; promote chat text from "not gated" to a ship gate. | CI fails on pad-heavy chat regression |

**P1 exit:** `generate-monolithic -p "Hello"` with default flags produces a coherent reply, no manual `--no-early-stop --steps 48` incantation required.

### P2 — Close the latency gap to interactive

Per-step is ~4.8 s forward on M3 Pro; interactive chat needs <= 1.8 s near-term. The remaining cost is GPU round-trips and dispatch overhead, not raw GEMM (residency + grouped MoE already landed). Sequence:

| # | Task | Impact | Exit |
|---|------|--------|------|
| P2.1 | **Finish GPU round-trip elimination** — keep forward readback at ids+scalars only (engine still does ~1.3 GiB/step; monolithic is close). Merge attention command batches. | High | <= 3 syncs/step, <= 1 MB readback/step |
| P2.2 | **ICB record/replay** — record the step once, replay with param patches; no per-kernel CPU encode in steady state. (Pipeline compile already cached ~3 ms; full ICB deferred from M3.) | High | steady-state encode cost ~= 0 |
| P2.3 | **SC softembed fast path** — current `k_sc_softembed` is O(vocab x hidden)/step; the experimental GEMM variant *regressed* to ~130 s/step. Needs tiled vocab chunks or an MPS matmul over materialized softmax probs. | High @ step>0 | SC step <= a few % of step time |
| P2.4 | **Dispatch fusion** — ~130 dispatches/step/layer; fuse `bucket_count`+`fill`, hoist invariant binds. | Medium | dispatch count down measurably |
| P2.5 | **lm_head + sampler over uncommitted positions only** — committed tokens are frozen; lm_head is ~19% of step FLOPs and shrinks toward 0 late in the loop. Mathematically identical for frozen positions — verify, else gate behind flag. | Medium | +>=15% tok/s, no golden change |
| P2.6 | **MPS Q8 for lm_head**; **f16 vs f32 activation** sweep; residual MoE/attention tile retuning. | Medium | step <= 1.4 s stretch |

**P2 exit:** `bench-step-kernel` <= 1.8 s/step @ 30L (M3 Pro); end-to-end >= 8 tok/s with P1 stop logic active.

### P3 — Harden & ship

| # | Task | Exit |
|---|------|------|
| P3.1 | **Multi-block extend** at scale — `max_new_tokens > 256` is implemented but lightly tested; verify KV extend across >= 3 blocks. | Multi-block coherent output |
| P3.2 | **kv>0 parity** — `step-parity` is kv_len=0 only; extend to real prompt context. | Parity holds at kv_len=64 |
| P3.3 | **MoE determinism for goldens** — f32 atomic scatter is ~1 ulp nondeterministic; keep the ordered/CPU variant for golden runs, document tolerance. | Goldens reproducible |
| P3.4 | **Memory budget on 24 GiB** — verify peak RSS <= budget with q4 `.dgq` on base M4; `--skip-vision` saves ~1.1 GiB; document `iogpu.wired_limit_mb`. | Runs on 24 GiB without OOM |
| P3.5 | **`q5` profile** for 36 GiB devices — ~free in wall-clock on compute-bound M3, buys quality headroom. Ablate embed/lm_head precision first if quality drops. | Optional higher-quality profile |
| P3.6 | **CI**: `step-ci` (config + verify + parity + generate smoke), graceful skip without weights; retire engine path only after monolithic sustains the perf win through dogfooding. | Green CI; monolithic is default |

**Ship definition:** `generate-monolithic` is the default macOS+metal path; readable chat at default flags; >= 8 tok/s on 24 GiB; CI green; engine retained as fallback until the win holds for ~2 weeks of dogfooding.

---

## Execution order

```
P1 (quality)  -->  P2 (latency)  -->  P3 (harden/ship)
   |                  |
   +-- P1.1-P1.2 unblock interactive use immediately;
       P2 can begin in parallel once P1.2 proves readable output
```

**Critical path to interactive:** P1.1 (pad-aware stop) -> P1.2 (chat defaults) -> P2.1 (round-trips) -> P2.2 (ICB).

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Pad-aware stop changes token-id goldens | Gate behind chat-mode; parity goldens keep `--raw` + fixed `--steps` |
| q4 quality loss (MoE ~= 4B-dense sensitivity) | q8 embed/lm_head/SC; bump shared expert + first/last layers before routed experts; q5 on 36 GiB |
| 24 GiB working-set cap (~18 GiB) tight | q4 + `--skip-vision` + scratch diet; document `iogpu.wired_limit_mb` |
| SC fast-path stays slow | tiled vocab chunks or MPS over materialized probs; worst case keep O(vocab) but it gates step>0 latency |
| ICB complexity (ICB vs Metal command replay) | prototype both, pick lower complexity |
| Two KV layouts (engine f32 vs monolithic half) diverge | monolithic b4 is canonical; do not force-fit legacy `GpuKvCache`; unify only post-ship |
| MoE f32 scatter nondeterminism masks real bugs | golden-match with ordered/CPU MoE *before* trusting the atomic path |
| M3 Pro lands at the 8 tok/s floor with thin margin | if step stalls, recover via P2.5 (uncommitted lm_head) + steps_eff tuning before quality-gated approximations |

---

## Deferred / explicitly out of scope

Vision (355 tensors, 280 soft tokens/image, bidirectional vision attention); step distillation (training, upstream concern); canvas=128 latency mode (measure, don't assume — worse weight-read amortization); Fast-dLLM-style committed-K/V freezing (approximation; bidirectional reps legitimately evolve — gate on fixtures only if needed).

---

## Command reference

`WEIGHTS=/tmp/quantized-weights` (or your `.dgq` dir). Prompts use the Gemma 4 chat template by default; pass **`--raw`** for bare BPE (legacy goldens, token-id parity).

```bash
# Quantize bf16 safetensors -> .dgq (once)
cargo run --release -- quantize -m model/transformer -o $WEIGHTS --profile q4

# One-shot generate (monolithic = production path)
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "Hello" --layers 30 --seed 42

# Interactive chat REPL (monolithic .dgq; optional -p for first turn)
cargo run --release --features metal -- -m $WEIGHTS chat --layers 30 --steps 48 --seed 42
cargo run --release --features metal -- -m $WEIGHTS chat -p "Hello" --layers 30 --steps 48 --seed 42

# Quality workaround until P1.1/P1.2 land (readable templated output)
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic \
  -p "Hello" --layers 30 --steps 48 --no-early-stop --seed 42

# Engine fallback (multi-kernel path; golden-locked baseline)
cargo run --release --features metal -- -m $WEIGHTS generate-gpu -p "Hello" --layers 30 --steps 2 --seed 42

# Chat template / tokenization debug
cargo run --release -- tokenize "Hello"
cargo run --release -- tokenize "Hello" --raw

# Bench (apples-to-apples: both kv prefill @ 64, 30 layers)
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --kv-len 64 --iters 5
cargo run --release --features metal -- -m $WEIGHTS bench-step --layers 30 --iters 5

# Step-kernel smoke / KV extend check
cargo run --release --features metal -- -m $WEIGHTS step-smoke --layers 3 --steps 4 --kv-len 64 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-kv-check --kv-len 64 --layers 30 --seed 42

# Parity / correctness gates (DGQ_MPS_Q4=0 for deterministic Q4 on goldens)
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS step-verify --layers 30
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS step-parity --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-ci --layers 3
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS generate-monolithic-parity \
  -p hello --raw --layers 3 --steps 4 --seed 42 --no-early-stop
cargo run --release --features metal -- -m $WEIGHTS generate-parity -p "Hello" --raw --layers 3 --steps 1 --seed 42
```

## Auxiliary commands

Bring-up, micro-benches, and unit tests — useful when debugging a subsystem, not for everyday generate/chat.

```bash
# Default entrypoint (no subcommand): same as generate-gpu on metal; CPU-only build uses generate
cargo run --release --features metal -- -m $WEIGHTS -p "Hello" --seed 42 --steps 2

# Route default to monolithic on .dgq (env or flag; same as generate-monolithic)
DGQ_MONOLITHIC=1 cargo run --release --features metal -- -m $WEIGHTS -p "Hello" --seed 42
cargo run --release --features metal -- -m $WEIGHTS --monolithic -p "Hello" --seed 42

# Unit / integration tests (CI runs both)
cargo test
cargo test --features metal
cargo test --features metal -- --skip gpu_determinism   # skip slow full-GPU determinism loops

# Step-kernel activation probe: one forward, checkpoint max_abs per stage (finer than step-smoke)
cargo run --release --features metal -- -m $WEIGHTS step-probe --layers 3 --kv-len 64 --seed 42

# Engine encoder prefill bench (prompt -> KV only; isolates prefill from denoise loop)
cargo run --release --features metal -- -m $WEIGHTS bench-prefill --prefill-len 64 --layers 30 --iters 5

# Legacy weight prep: bf16 safetensors -> iris.pack (pre-transposed GEMM); prefer quantize -> .dgq
cargo run --release -- convert-model -m model/transformer -o model/packed

# Metal device info (name, recommended limits)
cargo run --release --features metal -- probe-device

# Custom GEMM kernel micro-bench vs MPS oracle on shape list
cargo run --release --features metal -- bench-gemm --shapes 256x2816x2816 --oracle mps --iters 10

# Slow CPU-vs-GPU generate parity (optional; default generate-parity is golden-only)
cargo run --release --features metal -- -m $WEIGHTS generate-parity -p "Hello" --raw --layers 3 --steps 1 --seed 42 --compare-cpu

# Weight / config introspection (no GPU)
cargo run --release -- summary
cargo run --release -- config
```

| Command | What it does |
|---------|--------------|
| *(no subcommand)* | Shorthand for `generate-gpu` with `-p` / `--steps` / etc. On `.dgq` + `DGQ_MONOLITHIC=1` or `--monolithic`, routes to `generate-monolithic` instead. |
| `cargo test` | Rust unit tests on Linux CI; kernel/sampler logic without Metal. |
| `cargo test --features metal` | Full suite including Metal paths; add `-- --skip gpu_determinism` locally for faster iteration. |
| `step-probe` | Single monolithic forward with per-stage activation checkpoints (`finite`, `max_abs`). Use when `step-smoke` passes but you need to localize a bad layer/stage. |
| `bench-prefill` | Times engine-path encoder prefill only (`--prefill-len N`). Compare against `bench-step` / `bench-step-kernel` to see prefill vs denoise share. |
| `convert-model` | One-time pack of bf16 shards into `iris.pack` (transposed weights). Superseded for production by `quantize` → `.dgq` mmap blob. |
| `probe-device` | Prints Metal GPU identity and memory guidance; sanity check before long bench runs. |
| `bench-gemm` | Benchmarks custom QGEMM shapes; `--oracle mps` adds Apple's MPS matmul baseline on the same shapes. |
| `generate-parity --compare-cpu` | Runs CPU reference generate then GPU and diffs token streams — very slow; use for deep regressions, not CI smoke. |
| `summary` / `config` | Load weights and print shard/tensor summary or parsed `config.json` fields — quick sanity after download or quantize. |
