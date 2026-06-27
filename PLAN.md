# diffgemma-mps — production plan

Low-dependency Rust + Metal inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (Gemma-4 26B-A4B MoE, discrete block diffusion) on Apple Silicon.

This is the single forward-looking plan: **open work only.** Resolved work, measured data, and bug archaeology live in `NOTES.md`. How to work on the code lives in `STRATEGY.md`. Model semantics live in `ARCHITECTURE.md`.

---

## Where we are (2026-06)

`generate-monolithic` (single-encoder step kernel, `shaders/monolithic/diffgemma_step.metal`) is the production path: GPU encoder prefill → block denoise loop → encoder-extend, sharing weights/tokenizer/sampler with the legacy multi-kernel `generate-gpu` engine.

**Mixed-precision `.dgq` (current default profile):** bf16 attention + dense FFN, **bf16 embed** (tied lm_head + SC), q8 self-conditioning MLP, **q4 experts** (the only bulk-quantized tensors — memory constraint). Blob ~18.9 GiB.

**Quality vs MLX (matched-canvas "sky blue", the reference is `mlx-community/diffusiongemma-26B-A4B-it-4bit`):** Rust now converges in **16 denoise steps vs MLX-4bit's 13** (was 31 before the bf16-embed fix). Full chat ~37 steps / ~93 s; output reads like MLX with only minor residual token-doubling. See `NOTES.md` §10 for the equivalence investigation.

**Latency:** per denoise step ~1.7–1.9 s @ 30L (M3 Pro); still ~1.9× slower per step than MLX (MoE q4 + CPU↔GPU sync dominate now that the attention GEMM is fast).

### Target

Sustained **≥ 8 tok/s end-to-end on 24 GiB base M-series; 25+ on Pro/Max 36 GiB**, MLX-quality chat. Per-step ≤ 1.8 s on M3 Pro near-term, ≤ 1.4 s stretch.

### Non-goals (this cycle)

Training/fine-tuning/LoRA, multi-user serving, CUDA/Linux GPU, vision/multimodal (deferred; 355 vision tensors skippable via `--skip-vision`).

---

## Q — MLX generation equivalence (active focus)

Close the remaining quality + convergence gap to MLX-4bit. The bf16-embed fix (commit `d61cc24`) removed the tail-convergence stall — matched-canvas 31→16 steps; gemm_bf16 tiling (`f1900f4`) and SC fp16 probs (`0042d92`) also landed (see `NOTES.md` §10). What's left:

| # | Task | Exit |
|---|------|------|
| Q4 | Close last ~3 steps to MLX-4bit (16 vs 13) + residual "as as" doublings in long chat | matched-canvas ≤ 14 steps; no visible doubling |
| Q5 | Confirm equivalence end-to-end vs MLX-4bit on long generations (not just matched single-block) | side-by-side long-generation quality table |

**Q4 leads:** experts (q4 group-32 vs MLX 4-bit group-64 — but memory-neutral nvfp4 experts did *not* help, so likely not the lever); residual chunked-SC trajectory chaos; or a small remaining forward/precision diff. Investigate memory-neutral only — **no resident bf16 experts** (blob budget; see `NOTES.md` §6). Tooling: `generate-monolithic --write-trace` + `DGQ_TRACE_ENTROPY=1` → `python/scripts/dump_mlx_denoise_trace.py --canvas-ids` → `compare_denoise_trace.py`.

---

## P2 — Close the latency gap to interactive

Per-step is now bounded by MoE (q4 grouped GEMM) + dispatch/sync overhead, not the attention GEMM.

| # | Task | Impact | Status | Exit |
|---|------|--------|--------|------|
| P2.2 | ICB record/replay | High | open | steady-state encode ≈ 0; lift `kv_len==0` gate |
| P2.4 | Dispatch fusion | Medium | open | dispatch count down |
| P2.6 | MPS Q8 lm_head; f16/f32 sweep | Medium | open | step ≤ 1.4 s stretch |

### P2.7 open profile targets (`bench-step-kernel --layer-profile`)

Tier-1 GEMM K-tile double-buffering, stacked QKV/gate-up, SC chunked f32-accumulate, partial lm_head, and bf16/fp16 unification are **done** (see `NOTES.md` §4). Remaining hotspots:

| Priority | Stage | ~Cost | Options |
|----------|-------|-------|---------|
| 1 | MoE `gate_up` + `down` | ~33% | Grouped-GEMM occupancy for small per-expert M (~33 tok/expert); larger `tpg`/head tiling; MPS grouped (NOTES: ~3.7k encodes/step tradeoff) |
| 2 | attention | ~20% | Flash-style KV tiling (amortize softmax over T); head tiling |
| 3 | MoE scatter | ~5% | Smaller grid or fuse with down output |
| 4 | qkv / o_proj / dense | ~part | Fuse QK-norm + RoPE adjacency into QKV dispatch; fuse pre-FF RMSNorm/GLU |
| 5 | Preamble (SC) / finish (lm_head) / sync | ~12% / ~11% | ICB fast path (P2.2); Q8 lm_head tuning (P2.6); fewer command buffers/step |

**Suggested order:** MoE small-M occupancy → attention tiling → ICB (multi-step denoise) → cross-stage fusion (QKV+rope).

**P2 exit:** `bench-step-kernel` ≤ 1.8 s/step @ 30L; ≥ 8 tok/s e2e.

---

## P3 — Harden & ship

| # | Task | Impact | Exit |
|---|------|--------|------|
| P3.1 | Multi-block extend + `kv>0` golden parity | High | `generate-monolithic` multi-block matches engine on fixed seed |
| P3.2 | MoE determinism policy documented + tested | Medium | Atomic-scatter vs CPU-scatter tradeoffs in `NOTES.md`; engine parity path explicit |
| P3.3 | 24 GiB memory budget enforcement | Medium | `--skip-vision` + q4 documented; `iogpu.wired_limit_mb` guidance |
| P3.4 | q5 / alternate quant profile on 36 GiB | Low | Optional profile; quality A/B vs current default |
| P3.5 | CI default monolithic | Medium | `step-ci` + templated gate on monolithic path |
| P3.6 | Declarative step dispatch schedule | High (maintainability) | See [P3.6 detail](#p36-declarative-step-dispatch-schedule) |
| P3.7 | GPU debug status / invariant flag | High (debuggability) | See [P3.7 detail](#p37-gpu-debug-status--invariant-flag) |
| P3.8 | Subkernel extraction completion | Medium | All monolithic stage bodies in `shaders/kernels/` + Tier-1 oracles; legacy `qgemm.metal` retired |

**Loose ends to fold into P3:**
- `gemm_q8` / `gemm_q8_rowk` standalone GPU oracle tests fail (32-tile kernel vs `dispatch_shape` n_tile=128 mismatch); production dispatch is correct. Either migrate those kernels to the fast 128-tile (like `gemm_bf16`) or fix the test dispatch. (7 pre-existing test failures total incl. one sampler test.)
- RNG-1: canvas init vs `Rng::new(seed+1)` stream alignment (minor).

**P3 exit:** ship-quality chat @ 30L on 24 GiB; orchestration drift prevented by schedule asserts; `STRATEGY.md` §6 invariants enforced in debug builds.

#### P3.6 Declarative step dispatch schedule

**Problem:** `step_kernel.rs` (~4k lines) encodes the denoise schedule imperatively (~20 `encode_*` methods, hundreds of hand bind/dispatch calls, ~91 arena-offset references). The schedule also exists as a *comment* in `diffgemma_step.metal`. Two representations → probe/production forks, silent arena aliasing, and ICB that replays a host-recorded trace rather than a canonical schedule.

**Not in scope:** moving orchestration into Metal shaders — pipeline construction and buffer binding are host-only; ICB replays host-encoded commands, not shader control flow.

**Target:** a `StepSchedule { preamble, per_layer, finish }` of `StepStage` enum values (`Memzero`, `RmsNormRows`, `GemmBlock`, `GemmLinearGrouped`, `QkRopeKv`, `Attention`, `Router`, `MoeGrouped { probe: Option<DumpTarget> }`, `ScSoftembed`, `LmHeadSoftcapSampler`, …) built by `build_step_schedule(layout, profile)`. A single `StepInterpreter`:
1. Walks the stage list; resolves buffer handles from `StepBuffers` + arena table.
2. Selects pipeline from the manifest variant tuple (`KernelVariant` + FC axes).
3. Binds, dispatches, records to ICB when `record: bool`.
4. **Liveness check (debug):** each stage declares read/write arena ranges; interpreter asserts no read-before-write and no live aliasing (catches arena reuse *before* GPU run).
5. **Dump = a mode, not a fork:** `MoeGrouped { probe: Some(...) }` uses the same stage with `KernelVariant { dump_stage: N }` — no parallel `encode_*_probe` methods.

**ICB on-ramp:** `record=true` populates the replay op list; replay is `interpret(schedule, record=false, replay=Some(plan))`. Unblocks P2.2 for prefilled KV because the schedule is data, not a hard-coded `kv_len==0` gate.

**Migration (incremental):** (a) inventory `encode_*` → `StepStage` + schedule print/diff tool; (b) dual-run interpreter on one layer vs imperative, compare checksums; (c) replace layer loop, delete redundant `encode_*` one layer type at a time; (d) fold probe/capture forks into stage flags; (e) wire ICB through interpreter, lift `kv_len==0`.

**Exit:** `build_step_schedule()` is the sole schedule source; unit tests on schedule data (arena liveness, no probe/production divergence); `icb_replay_matches_live` passes via interpreter record path.

**Non-goals:** changing kernel math; merging engine + monolithic into one schedule (diff tool only).

#### P3.7 GPU debug status / invariant flag

**Problem:** GPU kernels can't return errors. Precondition violations (index OOB, zero softmax norm, bad route token) produce finite garbage discovered layers later. `STRATEGY.md` §6 lists invariants that should fire *inside* kernels but have no reporting channel.

**Target:** debug-gated `DebugStatus` buffer + shared error codes (manifest-owned), complementary to Tier-1 CPU oracles (oracles catch wrong-but-plausible math; the flag catches impossible values).

**Metal** (`shaders/include/debug_status.metal`, first-writer-wins via `atomic_compare_exchange`): `struct DebugStatus { atomic_uint code, kernel_id, threadgroup, value; }`; `debug_set_error()` gated on `K_SHAPE_ASSERT` (FC1, compiled out in production).

**Error codes** (shared Rust+Metal): 1 index OOB · 2 route token out of range · 3 softmax normalizer zero · 4 non-finite output · 5 entropy > ln(N) · 6 quant format unsupported · 7 arena offset OOB.

**Rust wiring:** `StepBuffers.debug_status: Option<MTLBuffer>` (16 B, zeroed at step start, debug only); `KernelVariant { shape_assert: true }` compiles in the checks, production compiles them out (zero atomics on hot path); after a debug step, read back and panic with `{code, kernel, tg, value}`; Tier-2 tests inject bad fixtures → expect a specific code, not cos 0.02.

**Exit:** manifest lists codes; Rust decoder unit-tested; ≥3 hot kernels wired (router/scatter, grouped GEMM bounds, softmax); production bench shows no regression.

**Non-goals:** replacing Tier-1 cosine oracles; catching wrong-but-finite GEMM math.

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Accept/entropy changes shift token goldens | Synthetic-entropy fixtures only; token goldens keep `--raw` + fixed `--steps` |
| Quant quality insufficient for 30L chat | bf16 attention + embed landed; experts stay q4 (memory); q5/alt-profile on 36 GiB; isolate forward with CPU MoE parity |
| Prefill dominates short prompts | Encoder prefill ~1.4–2.7 s @ 14–22 tok / 30L (GPU MoE); denoise still ~1.7–1.9 s/step |
| 24 GiB cap tight (blob now ~18.9 GiB) | q4 experts + `--skip-vision`; document `iogpu.wired_limit_mb`; bf16 embed is +0.74 GiB over q8 |
| MoE scatter nondeterminism regresses | `moe_scatter_weighted` TG-reduce (no float atomics); determinism golden |

---

## Command reference

`WEIGHTS=model/diffusiongemma-q4emb` (current default profile). Chat template by default; **`--raw`** for parity goldens.

```bash
# Production generate / chat
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "Hello" --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS chat -p "Hello" --layers 30 --seed 42

# CI / parity
cargo run --release --features metal -- -m $WEIGHTS step-ci --layers 3
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS generate-monolithic-parity \
  -p hello --raw --layers 3 --steps 4 --seed 42 --no-early-stop

# KV + parity diagnostics
cargo run --release --features metal -- -m $WEIGHTS step-kv-parity -p "Hello" --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-q4-parity --layers 30 --seed 42

# Bench
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --kv-len 64 --iters 5
cargo run --release --features metal -- bench-gemm --shapes 256x2816x8192 --oracle mps --iters 5

# MLX equivalence trace (matched canvas)
DGQ_TRACE_ENTROPY=1 ./target/release/diffgemma-mps -m $WEIGHTS -p "Why is the sky blue?" \
  generate-monolithic --steps 40 --write-trace /tmp/rust.json
cd python && uv run python scripts/dump_mlx_denoise_trace.py \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit -p "Why is the sky blue?" \
  --steps 40 --canvas-ids /tmp/canvas.json -o /tmp/mlx.json
```

Auxiliary/bring-up commands (`step-probe`, `bench-prefill`, `convert-model`, `probe-device`, `summary`, `config`) are documented in `NOTES.md` (Auxiliary commands section).
