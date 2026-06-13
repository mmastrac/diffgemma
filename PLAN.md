# diffgemma-mps — production plan

Low-dependency Rust inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon. Iris-style: mmap weights, CPU reference path, Metal acceleration.

This is the single forward-looking plan. Historical context, measured data, and resolved-issue archaeology live in `NOTES.md`. Model semantics live in `ARCHITECTURE.md`.

---

## Where we are (2026-06)

End-to-end generation works on GPU through **two** engines that share weights, tokenizer, and sampler semantics:

- **Engine** (`generate-gpu`): multi-kernel `.dgq` path. Proven-correct, golden-locked. ~3.89 tok/s @ 30L (M3 Pro), ~7.87 s/step forward.
- **Monolithic** (`generate-monolithic`): single-encoder step kernel (`shaders/diffgemma_step.metal`). **~39% faster forward (4.80 s/step @ 30L).** Prefill/extend writers, GPU sampler, chat template all landed.

The monolithic path is the production target.

**Honest status (post-P1.1–P1.5):** Infrastructure for readable chat is in place, but **`generate-monolithic -p "Hello"` @ 30L q4 still emits gibberish** after 48 steps (~320 s). Telemetry (seed 42) shows the accept rule is working as designed — the model rarely produces enough low-entropy positions per step.

### P1.6 telemetry findings (2026-06, seed 42, templated "Hello")

| Run | min_ent (early → late) | accept/step (late) | Output |
|-----|------------------------|--------------------|--------|
| **30L**, kv=14 | 0.87 → **0.008** by step 46 | 1–2 for steps 1–27; ramps to **5** at step 48 | Gibberish |
| **3L**, kv=14 | **~9.6–10.0** entire run | **1** every step | Mask/filler tokens |

Interpretation:

- **`min_ent` is the bottleneck**, not the accept implementation. With `entropy_bound=0.1`, a second accept requires the lowest-entropy position to have H ≤ 0.1 nats. Early steps (high temperature + random canvas) correctly get ~1 accept.
- **30L eventually sharpens** (min_ent < 0.1 after ~step 28), but even then only **~5 positions** have H < 0.1 simultaneously — far below the ~15–20 needed for fast canvas fill. Late-window accept_sum=26 over 8 steps.
- **3L + prompt KV is dead** (min_ent ≈ uniform ≈ 10 nats). Parity goldens use **kv_len=0**; chat always prefills kv≈14. Do not judge chat quality at 3L.
- **`step-kv-check` passes** @ kv=64, 3L — KV pack is not totally zeroed, but 3L cannot use prompt context effectively.
- **Prefill ~100 s** in `generate-monolithic` was initially blamed on cold reload (P1.8). **`bench-prefill` isolates the engine path at ~2.6 s/run** for 14 tokens @ 30L — so most of the gap is monolithic-specific overhead, not intrinsic encoder forward cost.
- **Hypotheses (P1.8):** duplicate `.dgq` GPU blob fixed via `load_weight_cache_opt`, but **prefill still ~155 s gpu_forward** when step runtime is resident vs `bench-prefill` ~2.6 s — likely GPU memory pressure, not reload alone. KV pack ~9–74 ms.
- **MTLBinaryArchive** (P2.0): runtime pipeline ISA cache at `~/.cache/diffgemma-mps/metal-pipelines/` (`DGQ_METAL_PIPELINE_CACHE=0` to disable). Skips recompiling our `.metal` kernels on restart; MPS matmul internals remain uncached.

**Next P1.6 experiments:** `--steps 96`; q5 profile; f32 rowstats; HF/Python accept count on same logits; engine vs monolithic @ 30L text compare.

### Target

Sustained **>= 8 tok/s end-to-end on 24 GiB base M-series; 25+ on Pro/Max 36 GiB**, with *readable* chat output. Per-step latency <= 1.8 s on M3 Pro near-term, <= 1.4 s stretch.

### Non-goals (this cycle)

Training/fine-tuning/LoRA, multi-user serving, CUDA/Linux GPU, vision/multimodal (deferred; 355 vision tensors skippable via `--skip-vision`).

---

## The two-stage commit model (read this before touching the loop)

DiffusionGemma has **two distinct "commit" stages**:

1. **Per denoise step — ~15-20 canvas positions.** Entropy-bound sampler (`entropy_bound = 0.1`) freezes lowest-entropy positions while `sum(entropy of prior accepted) <= 0.1` (HuggingFace mutual-information bound). **Re-noises the rest.**

2. **Per block end — 256 tokens.** Early stop or `max_denoising_steps` → argmax all 256 → emit → encoder-extend → fresh canvas.

**Do not strip pads from KV commit** — fix convergence before block commit. See open P1.6.

---

## Architecture (production target)

```
generate-monolithic
--------------------
prompt -> chat template -> tokenize
  -> GPU encoder prefill --> b4 kvcache (half, [pos][K|V][kv_head][dim] per layer)
  -> block loop (src/metal/step_generate.rs):
      CanvasState.ids init (256 random)
      for step in 1..=max_steps:
        encode_step: preamble (SC) + 30 layers + finish (lm_head/softcap/sampler)
        read stop_flag                       <- 1 sync/step
      commit: prev_argmax (256) -> emit -> GPU encoder-extend --> b4
  -> detokenize, strip pads for display only
```

Buffer ABI in `diffgemma_step.metal` (version-bump to change).

---

## What's already done (do not redo)

| Area | State |
|------|-------|
| Weight loading, config, CPU reference kernels, 30L decoder + encoder, KV cache | done |
| Entropy-bound block sampler (CPU + GPU; HF accept rule) | done |
| Tokenizer + chat template (HF-matched, `python/` uv parity tests) | done |
| `.dgq` quantizer, grouped MoE, MPS dense, GPU router/sampler | done |
| Monolithic step kernel, KV prefill/extend, generate loop, chat REPL | done |
| Determinism goldens (`DGQ_MPS_Q4=0`), M0–M5 gates | done |
| **P1.1** Pad-aware early stop (CPU + `k_sample_commit`) | done |
| **P1.2** Production `--steps 48` default; parity/bench stay at 2 | done |
| **P1.3** `steps_eff` + accept/step telemetry | done |
| **P1.4** Display strips pad/filler; KV commit unchanged | done |
| **P1.5** Templated-chat quality gate in `step-ci` | done |
| Plan consolidation (`NOTES.md`, retired PLAN2/MONOLITHIC) | done |

**Measured baseline (M3 Pro, `/tmp/quantized-weights`):** monolithic forward ~4.8 s/step; generate with early stop ~6.58 tok/s (8 steps/block). User run @ 30L, 48 steps: ~4.7 s/step denoise, **prefill ~98 s wall** (see P1.8). `bench-prefill --prefill-len 14 --layers 30`: **~2.6 s/run** engine-only.

---

## Plan forward

### P1 — Make output correct & readable (ship blocker)

| # | Task | Status | Exit |
|---|------|--------|------|
| P1.1 | Pad-aware early stop | **done** | No stop on all-pad/filler argmax before min steps |
| P1.2 | Chat-oriented CLI defaults (`--steps 48`) | **done** | Production paths default 48; parity uses 2 |
| P1.3 | `steps_eff` + accept/step telemetry | **done** | Visible in generate output |
| P1.4 | Decode hygiene (display only) | **done** | Chat preview strips pad/filler |
| P1.5 | Templated-chat quality gate (`step-ci`) | **done** | CI fails pad-heavy regression |
| P1.6 | **Canvas convergence** — see telemetry findings above. Open: raise simultaneous low-H positions (forward/quant/KV/SC) or validate HF parity on same weights. | **open** | `low_ent` ≥ 15 late-block OR readable Hello @ 30L |
| P1.7 | HF accept parity (unit + `sampler_accept_entropy.json`) | **done** | Fixture tests pass; Metal uses equivalent prefix rule |
| P1.8 | **Encoder prefill latency** — shared blob, phased timing, session reuse. Open: ~155 s gpu_forward when step runtime resident. | **partial** | Generate prefill ≤3 s @ 14 tok |
| P2.0 | **MTLBinaryArchive pipeline cache** — persist compiled compute pipeline ISA across restarts. | **done** | Archive load/save under `~/.cache/diffgemma-mps/metal-pipelines/` |

**P1 exit (unchanged):** `generate-monolithic -p "Hello"` default flags → coherent reply.

### P2 — Close the latency gap to interactive

Blocked on P1.6 (no point optimizing gibberish). Sequence unchanged:

| # | Task | Impact | Exit |
|---|------|--------|------|
| P2.1 | GPU round-trip elimination | High | <= 3 syncs/step, <= 1 MB readback/step |
| P2.2 | ICB record/replay | High | steady-state encode ~= 0 |
| P2.3 | SC softembed fast path | High @ step>0 | SC <= few % of step |
| P2.4 | Dispatch fusion | Medium | dispatch count down |
| P2.5 | lm_head over uncommitted positions only | Medium | +>=15% tok/s |
| P2.6 | MPS Q8 lm_head; f16/f32 sweep | Medium | step <= 1.4 s stretch |

**P2 exit:** `bench-step-kernel` <= 1.8 s/step @ 30L; >= 8 tok/s e2e with P1 active.

### P3 — Harden & ship

Unchanged — multi-block extend, kv>0 parity, MoE determinism docs, 24 GiB budget, q5 profile, CI default monolithic.

---

## Execution order

```
P1.6 (convergence)  -->  P1.8 (prefill)  -->  P2  -->  P3
     P1.7 done
```

**Critical path to interactive:** P1.6 → P1.8 → P2.1 → P2.2.

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Accept/entropy fix changes token goldens | Synthetic-entropy fixtures only; token goldens keep `--raw` + fixed `--steps` |
| q4 quality insufficient for 30L chat | q5 on 36 GiB; ablate embed/lm_head; CPU MoE parity isolate forward |
| Half logits rowstats numerically wrong @ 262K vocab | P1.6: f32 rowstats experiment behind flag |
| Prefill dominates short prompts (~98 s vs ~2.6 s bench) | P1.8: phased timing, share dgq blob, avoid duplicate GPU weight resident |
| Prefill KV pack wrong → flat logits | `step-kv-check` + prefix max_abs audit before sampler work |
| 24 GiB cap tight | q4 + `--skip-vision`; document `iogpu.wired_limit_mb` |

---

## Command reference

`WEIGHTS=/tmp/quantized-weights`. Chat template by default; **`--raw`** for parity goldens.

```bash
# Production generate (48 steps default)
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "Hello" --layers 30 --seed 42

# Interactive chat
cargo run --release --features metal -- -m $WEIGHTS chat -p "Hello" --layers 30 --seed 42

# CI / parity
cargo run --release --features metal -- -m $WEIGHTS step-ci --layers 3
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS generate-monolithic-parity \
  -p hello --raw --layers 3 --steps 4 --seed 42 --no-early-stop

# KV + sampler + prefill diagnostics
cargo run --release --features metal -- -m $WEIGHTS bench-prefill --prefill-len 14 --layers 30 --iters 3
cargo run --release --features metal -- -m $WEIGHTS step-kv-check --kv-len 64 --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-smoke --layers 3 --steps 4 --kv-len 64 --seed 42

# Bench
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --kv-len 64 --iters 5
```
