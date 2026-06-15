# diffgemma-mps — production plan

Low-dependency Rust inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon. Iris-style: mmap weights, CPU reference path, Metal acceleration.

This is the single forward-looking plan. Historical context, measured data, and resolved-issue archaeology live in `NOTES.md`. Model semantics live in `ARCHITECTURE.md`.

---

## Where we are (2026-06)

End-to-end generation works on GPU through **two** engines that share weights, tokenizer, and sampler semantics:

- **Engine** (`generate-gpu`): multi-kernel `.dgq` path. Proven-correct, golden-locked. ~3.89 tok/s @ 30L (M3 Pro), ~7.87 s/step forward.
- **Monolithic** (`generate-monolithic`): single-encoder step kernel (`shaders/diffgemma_step.metal`). **~39% faster forward (4.80 s/step @ 30L).** Prefill/extend writers, GPU sampler, chat template all landed.

The monolithic path is the production target.

**Honest status (post-P1.1–P1.9):** Infrastructure for readable chat is in place. **`generate-monolithic` defaults to native Q4 for both step kernel and encoder prefill** (see P1.9). With those defaults, templated `"Hello"` @ 30L shows real entropy (`min_ent` ≪ ln vocab) but output can still be gibberish after 48 steps — canvas convergence (P1.6) remains the quality bottleneck.

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
- **Hypotheses (P1.8):** ~~duplicate `.dgq` GPU blob~~ fixed; monolithic encoder now respects `encoder_use_mps_q4` / env. **Fast encoder prefill** landed in PREF-2 (GPU grouped MoE default).
- **P1.9 findings (2026-06):** `step-kv-parity` @ 30L now **passes** (`max_kv_diff≈0.008`). Root cause was transposed `dequant_q4_matrix` grid + GPU grouped MoE diverging from CPU MoE; encoder uses MPS dense + CPU MoE.
- **PREF-2 (2026-06):** Encoder prefill now defaults to **grouped GPU MoE** (`experts_forward_gpu_batched` → `gemm_linear_grouped`). ~**1.4 s** @ 22 tok / 30L (was ~30–62 s on CPU MoE). `step-kv-parity` and CPU-vs-GPU-MoE KV test pass @ 22 tok Calgary prompt (`max_kv_diff` < 0.02). Opt out: `DGQ_ENCODER_GPU_MOE=0`. Longer-prompt repetition was partly **`qk_rope_kv` V-alias bug** on full layers 5/11/17/23/29 (fixed 2026-06); remainder is P1.6 q4 convergence.
- **P1.10 (2026-06):** `dequant_q4_matrix` dispatch had `width`/`height` swapped vs shader `row=gid.y, col=gid.x` for `[N,K]` weights — transposed dequant broke all MPS Q4 dense paths. Fixed; `step-q4-parity` @ 30L passes. Step + encoder MPS Q4 re-enabled by default (`DGQ_*_Q4=0` to opt out).
- **MTLBinaryArchive** (P2.0): runtime pipeline ISA cache at `~/.cache/diffgemma-mps/metal-pipelines/` (`DGQ_METAL_PIPELINE_CACHE=0` to disable). Skips recompiling our `.metal` kernels on restart; MPS matmul internals remain uncached.

**Next P1.6 experiments:** `--steps 96`; q5 profile; f32 rowstats; HF/Python accept count on same logits; engine vs monolithic @ 30L text compare.

**P1.6 trace tooling (2026-06):** Per-step denoise JSON (`--write-trace`) from `generate-monolithic`; HuggingFace reference via `python/scripts/dump_denoise_trace.py`; MLX via `python/scripts/dump_mlx_denoise_trace.py`; compare with `python/scripts/compare_denoise_trace.py`. Fast iteration: `--steps 2` + `DGQ_LOG_DENOISE=1` + `DGQ_TRACE_ENTROPY=1`. **Canvas RNG:** monolithic uses Rust LCG (`sample::Rng`); MLX default uses `mx.random` — use `dump_mlx_denoise_trace.py --canvas-rng rust` for matched-canvas compares. Step-1 parity gap (2026-06): same canvas+prefill, MLX mxfp4 entropies ~0.04–1.5 nats vs `.dgq` ~0.5–3.1 nats → MLX accepts ~236 positions, mono ~1; argmax matches pos 0–1 then diverges (quant/forward, not accept rule).

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
| **P1.7–P1.10** HF accept parity; MPS Q4 fix; parity gates | done |
| **P2.1** Generate hot path: 1 sync/step, ~12 KiB host readback/step | done |
| **P2.1** Monolithic generate hot path: 1 sync/step, ~12 KiB readback/step | done |
| Plan consolidation (`NOTES.md`, retired PLAN2/MONOLITHIC) | done |

**Measured baseline (M3 Pro, `/tmp/quantized-weights`):** monolithic forward ~1.8 s/step; encoder prefill ~1.4 s @ 22 tok / 30L (GPU grouped MoE); `step-kv-parity` + encoder MoE KV test pass @ 30L.

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
| P1.8 | **Encoder prefill path** — respects `StepGenerateConfig.encoder_use_mps_q4`; generate defaults native Q4. | **done** | Correct KV + readable entropy @ 30L with defaults |
| P1.9 | **MPS encoder KV parity** — `step-kv-parity`; MPS dense + GPU grouped MoE | **done** | `max_kv_diff` < 0.5 @ 30L |
| P1.10 | **Step-kernel MPS Q4** — fix `dequant_q4_matrix` grid; `step-q4-parity` gate | **done** | MPS min_ent ≪ ln vocab; |Δmin_ent| < 3 vs native @ 30L |
| P1.11 | **Step-kernel MPS NVFP4** — `step-nvfp4-parity` gate; fused half-dequant in `gemm_block` | **done** | Parity passes @ 30L; MPS ≈ fused speed on M3 (dequant-bound) |
| P2.0 | **MTLBinaryArchive pipeline cache** — persist compiled compute pipeline ISA across restarts. | **done** | Archive load/save under `~/.cache/diffgemma-mps/metal-pipelines/` |

**P1 exit (unchanged):** `generate-monolithic -p "Hello"` default flags → coherent reply.

### P2 — Close the latency gap to interactive

P1.6 (convergence) remains the quality gate; P2.1 latency work can proceed in parallel.

| # | Task | Impact | Exit |
|---|------|--------|------|
| P2.1 | GPU round-trip elimination | High | **done** | ≤3 syncs/step, ≤1 MB readback/step on generate hot path |
| P2.2 | ICB record/replay | High | steady-state encode ~= 0 |
| P2.3 | SC softembed fast path | High @ step>0 | SC <= few % of step |
| P2.4 | Dispatch fusion | Medium | dispatch count down |
| P2.5 | lm_head over uncommitted positions only | Medium | +>=15% tok/s |
| P2.6 | MPS Q8 lm_head; f16/f32 sweep | Medium | step <= 1.4 s stretch |

**P2 exit:** `bench-step-kernel` <= 1.8 s/step @ 30L; >= 8 tok/s e2e with P1 active.

### P3 — Harden & ship

| # | Task | Impact | Exit |
|---|------|--------|------|
| P3.1 | Multi-block extend + kv>0 golden parity | High | `generate-monolithic` multi-block matches engine on fixed seed |
| P3.2 | MoE determinism policy documented + tested | Medium | Atomic scatter vs CPU scatter tradeoffs in `NOTES.md`; engine parity path explicit |
| P3.3 | 24 GiB memory budget enforcement | Medium | `--skip-vision` + q4 documented; wired-limit guidance |
| P3.4 | q5 profile on 36 GiB | Low | Optional quant profile; quality A/B vs q4 @ 30L |
| P3.5 | CI default monolithic | Medium | `step-ci` + templated gate on monolithic path |
| P3.6 | **Declarative step dispatch schedule** | High (maintainability) | See [P3.6 detail](#p36-declarative-step-dispatch-schedule) |
| P3.7 | **GPU debug status / invariant flag** | High (debuggability) | See [P3.7 detail](#p37-gpu-debug-status--invariant-flag) |
| P3.8 | Subkernel extraction completion | Medium | All monolithic stage bodies in `shaders/kernels/` + Tier-1 oracles; `qgemm.metal` retired |

**P3 exit:** ship-quality chat @ 30L on 24 GiB; orchestration drift class prevented by schedule asserts; §6 invariants enforced in debug builds.

#### P3.6 Declarative step dispatch schedule

**Problem:** `step_kernel.rs` (~4k lines) encodes the denoise-step schedule imperatively (~20 `encode_*` methods, ~216 hand bind/dispatch calls, ~91 arena-offset references). The intended schedule also exists as a *comment* in `shaders/monolithic/diffgemma_step.metal`. Two representations of one program → probe/production forks (`encode_layer_moe_grouped` vs `encode_layer_moe_grouped_act_probe`), silent arena aliasing, and ICB that only replays a host-recorded imperative trace instead of a canonical schedule.

**Not in scope:** moving orchestration into Metal shaders — pipeline construction and buffer binding are host-only; ICB is replay of host-encoded commands, not shader-side control flow.

**Target architecture:**

```rust
/// One logical stage in a denoise step (layer-local or global).
enum StepStage {
    Memzero { buf: BufferSlot, bytes: u64 },
    RmsNormRows { in: Arena, out: Arena, weight: TensorRef, eps: f32 },
    GemmBlock { x: Arena, out: Arena, w: TensorRef, m: u32, n: u32, k: u32 },
    GemmLinearGrouped { a: BufferSlot, c: BufferSlot, jobs: RouteJobs, k: u32, n: u32 },
    QkRopeKv { /* layer, head layout from ModelLayout */ },
    Attention { layer: u32, mask: AttnMaskKind },
    ResidualHalf { a: Arena, b: Arena },
    Router { layer: u32 },
    MoeRouterBucket { phase: u32 },
    GatherRows { src: BufferSlot, indices: RouteField, dst: BufferSlot, hidden: u32 },
    SwigluMoeGateUp { gu: BufferSlot, act: BufferSlot, moe_ff: u32 },
    MoeScatterWeighted { expert_out: BufferSlot, moe_out: BufferSlot },
    MoeGrouped { format: QuantFormat, probe: Option<DumpTarget> },  // one stage, not 6 encode methods
    ScSoftembed { /* step>0 only */ },
    LmHeadSoftcapSampler { finish: FinishMode },
    // ...
}

struct StepSchedule {
    preamble: Vec<StepStage>,
    per_layer: Vec<StepStage>,   // repeated `layers` times with layer index injected
    finish: Vec<StepStage>,
}

fn build_step_schedule(layout: &ModelLayout, profile: StepBlockProfile) -> StepSchedule;
```

**Single interpreter** (`StepInterpreter` or extend `StepEnc`):

1. Walk `Vec<StepStage>`; for each stage resolve buffer handles from `StepBuffers` + arena table.
2. Select pipeline from manifest variant tuple (`KernelVariant` + per-kernel FC axes — same vocabulary as Tier-1 subkernels).
3. Bind, dispatch, record to ICB when `record: bool`.
4. **Liveness check (debug):** each stage declares read/write arena ranges; interpreter asserts no read of a range that is not yet written in the current step, and no concurrent live aliases (catches `A_DENSE` reuse class *before* GPU run).
5. **Dump/probe mode:** `Stage::MoeGrouped { probe: Some(A_MOEOUT) }` uses the *same* stage list with `KernelVariant { dump_stage: N }` — no parallel `encode_*_probe` methods (STRATEGY.md §4: dump is a mode, not a fork).

**ICB on-ramp:** interpreter with `record=true` populates `IcbRecorder` + `StepReplayOp` list; replay path is `interpret(schedule, record=false, replay=Some(plan))`. Unblocks P2.2 for prefilled KV (`kv_len>0`) because the schedule is data, not hard-coded `kv_len==0` gate.

**Migration plan (incremental, post-P1.6):**

| Phase | Work | Risk |
|-------|------|------|
| P3.6a | Inventory: map each `encode_*` → proposed `StepStage`; generate schedule print/diff tool | Read-only |
| P3.6b | Dual-run: interpreter drives one layer while imperative path remains default; compare buffer checksums | Medium |
| P3.6c | Replace layer loop with interpreter; delete redundant `encode_*` | High — do one layer type at a time |
| P3.6d | Fold probe/capture/single-expert forks into stage flags; delete duplicate encode methods | Medium |
| P3.6e | Wire ICB record/replay through interpreter; lift `kv_len==0` restriction | Medium |

**Exit criteria:**

- `build_step_schedule()` is the sole schedule source; metal-file dispatch comment deleted or generated.
- Unit tests on schedule data: arena liveness, no probe/production stage-list divergence.
- `icb_replay_matches_live_tier2` passes via interpreter record path.
- Engine vs monolithic schedule diff is a data diff (optional cross-check), not manual audit.

**Explicit non-goals for P3.6:** changing kernel math; merging engine and monolithic into one binary schedule (they may differ in buffer ABI — diff tool only).

---

#### P3.7 GPU debug status / invariant flag

**Problem:** GPU kernels cannot return errors. Precondition violations (index OOB, impossible softmax norm, bad route token) produce finite garbage discovered layers later (cosine hunts). STRATEGY.md §6 lists invariants that should fire *inside* kernels but currently have no reporting channel.

**Target:** debug-gated `DebugStatus` buffer + shared error codes (manifest-owned), complementary to Tier-1 CPU oracles (oracles catch wrong-but-plausible math; flag catches impossible values).

**Metal layout** (`shaders/include/debug_status.metal`, included only when needed):

```metal
struct DebugStatus {
    atomic_uint code;       // 0 = ok; else shared ErrorCode enum
    atomic_uint kernel_id;  // manifest kernel index or hash
    atomic_uint threadgroup; // tg.x | (tg.y << 16)
    atomic_uint value;      // offending index / raw bits / count
};

/// First writer wins — later errors must not clobber root cause.
inline void debug_set_error(device DebugStatus *st, uint code, uint kernel_id,
                            uint tg, uint value) {
    if (!K_SHAPE_ASSERT) return;
    uint expected = 0u;
    atomic_compare_exchange_weak_explicit(&st->code, &expected, code,
        memory_order_relaxed, memory_order_relaxed);
    if (expected == 0u) {
        atomic_store_explicit(&st->kernel_id, kernel_id, memory_order_relaxed);
        atomic_store_explicit(&st->threadgroup, tg, memory_order_relaxed);
        atomic_store_explicit(&st->value, value, memory_order_relaxed);
    }
}
```

**Error code enum** (shared Rust + Metal via `build/manifest.toml` `[debug_errors]` or kernel manifest extension):

| Code | Meaning | Example kernel |
|------|---------|----------------|
| 1 | Index OOB | `gather_rows`, `moe_grouped` |
| 2 | Route token out of range | `moe_scatter_weighted` |
| 3 | Softmax normalizer zero | `softmax_rows` |
| 4 | Non-finite output | any stage output |
| 5 | Entropy > ln(N) | `sample_rowstats` |
| 6 | Quant format unsupported | block GEMM stages |
| 7 | Arena offset OOB (tier-2) | interpreter-only on host; optional GPU bounds |

**Rust wiring:**

- `StepBuffers.debug_status: Option<MTLBuffer>` — 16 bytes, zeroed at step start in debug builds only.
- `KernelVariant { shape_assert: true }` pipelines compile in `debug_set_error` calls; production `PRODUCTION` variant compiles them out (**zero atomics on hot path**).
- After debug step: read back; if `code != 0`, panic with decoded `{code, kernel, tg, value}` (same ergonomics as Rust assert).
- Tier-2 tests: inject bad route fixture → expect panic code 2, not cos 0.02.

**Relationship to existing FC axes:**

- `K_SHAPE_ASSERT` (FC1) gates both bounds checks and error-flag writes — already compile-out in production pipelines.
- `K_DUMP_STAGE` (FC2) remains separate (diagnostic capture); error flag is not a dump substitute.

**Exit criteria:**

- Manifest lists error codes; Rust decoder unit-tested.
- At least three hot kernels wired (router/scatter, grouped GEMM bounds, softmax).
- Debug `step-probe` / `step-smoke` with `KernelVariant::TEST_ASSERT` panics on injected bad fixture.
- Production `bench-step-kernel` / `generate-monolithic` show no measurable regression (flag path compiled out).

**Explicit non-goals:** replacing Tier-1 cosine oracles; catching wrong GEMM math that stays finite (e.g. the historical `col=gid.x` grouped GEMM bug).

---


## Execution order

```
P1.6 (convergence)  -->  ship quality
P2.1 done; P2.2–P2.6 (latency) in parallel with P1.6 experiments
     P1.7–P1.10 done (accept rule, MPS Q4 fix, parity gates)
```

**Critical path to interactive:** P1.6 → P2.2 (ICB) → P2.3 (SC fast path).

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Accept/entropy fix changes token goldens | Synthetic-entropy fixtures only; token goldens keep `--raw` + fixed `--steps` |
| q4 quality insufficient for 30L chat | q5 on 36 GiB; ablate embed/lm_head; CPU MoE parity isolate forward |
| Half logits rowstats numerically wrong @ 262K vocab | P1.6: f32 rowstats experiment behind flag |
| Prefill dominates short prompts | Encoder prefill ~1.4–2.7 s @ 14–22 tok / 30L (PREF-2 GPU MoE default); denoise still ~1.8 s/step |
| MPS encoder KV wrong → flat logits | `step-kv-parity` gate (passes @ 30L) |
| MPS step Q4 wrong → flat logits | `step-q4-parity` gate (passes @ 30L) |
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
cargo run --release --features metal -- -m $WEIGHTS step-kv-parity -p "Hello" --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-q4-parity --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-smoke --layers 3 --steps 4 --kv-len 64 --seed 42

# Bench
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --kv-len 64 --iters 5
```
