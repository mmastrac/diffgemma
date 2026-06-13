# Monolithic step kernel — productionization plan

Roadmap to take the **rev2 monolithic GPU denoise path** (`shaders/diffgemma_step.metal` + `src/metal/step_kernel.rs`) from a parallel smoke/bench harness to a **shippable replacement** for the multi-kernel `generate-gpu` engine.

See `ARCHITECTURE.md` for DiffusionGemma semantics (block diffusion, entropy sampler, causal prefill vs bidirectional denoise). See `PLAN2.md` for the broader quantized `.dgq` engine that this path must eventually match or beat.

---

## Goal

| Target | Notes |
|--------|--------|
| **End-to-end text generation** | Prompt → prefill → denoise loop → block commit → KV extend → decoded text |
| **Parity with engine** | Token-id goldens within agreed fp tolerance; same sampler semantics as `src/sample.rs` |
| **Performance** | ≥ `generate-gpu` throughput at 30 layers on M3-class hardware |
| **Single command buffer per denoise step** | Record once (ICB), replay with param patches — no per-kernel CPU encode in steady state |
| **Production CLI** | `generate-monolithic` (or feature-flagged default) with telemetry matching `generate-gpu` |

**Non-goals (initial ship):** vision/multimodal, CPU fallback for the monolithic path, multi-GPU.

---

## Current state (2026-06, M3 Pro, `/tmp/quantized-weights`)

### What works

| Capability | Status |
|------------|--------|
| `.dgq` blob mmap + `ModelLayout` ABI | ✅ |
| Full denoise step encode (~130 dispatches/layer stack) | ✅ 3 layers tested |
| Q4 dense GEMM via MPS (`DGQ_MPS_Q4=1`, default) | ✅ ~2.33 s/step forward-only @ 3L |
| Fused Q4 simdgroup fallback (`DGQ_MPS_Q4=0`) | ✅ ~2.80 s/step @ 3L |
| GPU softcap (Metal `tanh` clamp fix) | ✅ `max_abs=30`, finite logits |
| GPU sampler (rowstats → commit → apply → write) | ✅ early stop, entropy |
| First-step SC skip (no rowstats / softembed / SC MLP when `step==0`) | ✅ |
| MoE grouped kernel + fused `k_rmsnorm_f32` tail | ✅ |
| Tooling | ✅ `step-smoke`, `step-probe`, `bench-step-kernel` |

### Measured vs engine

| Benchmark | Monolithic | Engine (`bench-step`) | Caveat |
|-----------|------------|----------------------|--------|
| Forward-only @ 3L | **2.33 s/step** | 2.44 s/step | step-kernel `kv_len=0`; engine prefill @ 64 |
| Full smoke (1 step, sampler) | ~2.7 s | — | Random canvas, no prompt |
| 30 layers | **not tested** | ~5.3 s/step (PLAN2) | IS_FULL pipelines exist but unverified |

### What does **not** work yet

- **No prompt / no KV context** — `step-smoke` starts from a random 256-token canvas with `kv_len=0`.
- **No encoder prefill or extend** — shader comment `NOTE-KV`: monolithic KV layout ≠ `GpuKvCache`.
- **Not wired to `generate`** — no block loop, no token output decode, no multi-block chaining.
- **Layer coverage** — smoke/probe default to 3 layers; full 30L + 5 full-attention layers unverified.
- **Parity gaps** — open `VERIFY-*` items in shader header (nibble order, SC scale, rng init).
- **Perf debt** — ~130 dispatches/step/layer; `k_sc_softembed` is O(vocab×hidden) per step>0; MPS scratch ~92 MiB for largest Q4 shapes.

---

## Architecture snapshot

```
generate-gpu (today)                    monolithic target (future)
─────────────────────                   ───────────────────────────
Encoder prefill (causal, GpuKvCache)  → NEW: prefill writer → b4 kvcache (half layout)
Block loop:                             → step-generate driver (Rust)
  initialize_canvas                     → CanvasState.ids init (GPU or CPU)
  for step in 1..max:                   → encode_step (ICB replay)
    decoder_forward (multi-kernel)      →   preamble + layers + finish
    gpu_sampler                         →   (sampler already in finish)
  commit argmax block                   → read CanvasState after stop
  extend_encoder_kv                     → NEW: extend writer → b4 kvcache
decode + print                          → same tokenizer path as generate-gpu
```

### Buffer ABI (fixed — do not change without version bump)

Documented in `shaders/diffgemma_step.metal` lines 4–15:

| Index | Buffer | Role |
|-------|--------|------|
| b0 | `.dgq` blob | mmap weights |
| b1 | `ModelLayout` | tensor byte offsets + per-layer metadata |
| b2 | `StepParams` | kv_len, sampler thresholds, temperature schedule |
| b3 | arena | f16 activations @ byte offsets (`A_*` in `step_kernel.rs`) |
| b4 | kvcache | **half**, layout `[pos][K\|V][kv_head][dim]` per layer region |
| b5 | `CanvasState` | ids, entropy, accept, rng, step, stop_flag |
| b6 | logits | half `[256][262144]` |
| b7 | `RouteScratch` | MoE routing buckets |

---

## Phase M0 — Correctness & parity gates

**Deliverable:** monolithic forward+sampler matches engine on fixed seeds for N layers, before any generate integration.

| Task | Files | Exit |
|------|-------|------|
| **M0.1 VERIFY-N** — Q4 nibble parity vs `q4_weight_at` in `qgemm.metal` | `diffgemma_step.metal` `dequant_q4_group` | ✅ `step-verify` (CPU dequant_row_q4 vs metal layout) |
| **M0.2 VERIFY-SC** — softembed `EMBED_SCALE = sqrt(hidden)`; softmax over post-softcap logits | `k_sc_softembed`, `k_logit_rowstats` | ✅ `step-verify` (scale + rowstats CPU ref) |
| **M0.3 RNG init** — `CanvasState.rng_state` matches `Rng::new(seed)` / `sample.rs` | `init_canvas_state` | ✅ `step-verify` (1000 LCG draws) |
| **M0.4 Full-layer paths** — layers 5,11,17,23,29: V aliased from K, partial RoPE, q/k/o GEMM shapes | `encode_layer`, `k_qk_rope_kv` | ✅ `step-probe` finite @ 30L |
| **M0.5 Sampler parity** — temperature schedule, entropy-bound accept, stable+confident stop | `k_sample_*`, `StepParams` | ✅ `step-verify` (deterministic 3L×4 steps, 3 seeds) |
| **M0.6 Decoder parity** — hidden/logits vs `decoder-gpu` forward-only @ same canvas+kv | `step-parity` CLI | ✅ @ 3L/30L vs `fixtures/generate/monolithic_parity_*.json` limits |
| **M0.7 MoE determinism policy** — document ~1 ulp f32 scatter variance; optional ordered-reduce kernel for goldens | `k_moe_grouped` comment | ✅ shader NOTE + relaxed parity tolerances |

**Commands to add:**

```bash
# M0 unit + integration gates (Q4, RNG, sampler determinism, 30L finite)
cargo run --release --features metal -- -m $WEIGHTS step-verify --layers 30

# Layer checkpoints + sampler stats
cargo run --release --features metal -- -m $WEIGHTS step-probe --layers 30 --seed 42

# Forward-only parity vs engine (same canvas; kv_len=0 until M1)
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS step-parity --layers 3 --seed 42
DGQ_MPS_Q4=0 cargo run --release --features metal -- -m $WEIGHTS step-parity --layers 30 --seed 42
```

**Exit:** `step-parity` passes @ 3L and 30L; goldens checked in under `fixtures/generate/` (new profile `monolithic`).

---

## Phase M1 — KV cache & encoder integration

**Deliverable:** monolithic denoise step runs with **real prompt context** (`kv_len > 0`).

The monolithic KV layout is **intentionally different** from `GpuKvCache` (half, unified per-layer region). Do **not** force-fit the old cache; add writers that target b4.

| Task | Notes |
|------|-------|
| **M1.1 Layout spec** | ✅ `src/metal/step_kv.rs` module docs + `kv_cache_total_bytes()` |
| **M1.2 Prefill writer** | ✅ GPU encoder prefill → readback → pack b4 (`prefill_monolithic_kv`); CPU path for bf16 |
| **M1.3 Extend writer** | ✅ `extend_monolithic_kv`: hydrate b4 prefix → GPU extend → pack suffix |
| **M1.4 Attention read path audit** | ✅ `step-kv-check` verifies b4 prefix + forward/extend vs kv_len=0 |
| **M1.5 Mask semantics** | ✅ Monolithic `k_attention` uses full `T=kv_len+CANVAS` (matches `DecoderAttnMask::all_valid`) |
| **M1.6 CLI** | ✅ `step-smoke --kv-len 64` runs GPU prefill writer; `-p` for real prompt tokens |

**Exit:** `step-smoke --kv-len 64 --layers 30` finite logits; attention output changes vs kv_len=0 (sanity: not identical).

```bash
cargo run --release --features metal -- -m $WEIGHTS step-kv-check --kv-len 64 --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS step-smoke --kv-len 64 --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS -p "Hello" step-smoke --kv-len 64 --layers 3
cargo run --release --features metal -- -m $WEIGHTS step-probe --kv-len 64 --layers 30 --seed 42
```

---

## Phase M2 — Full generate loop

**Deliverable:** `generate-monolithic -p "hello"` prints decoded text; structurally matches `generate_inner` in `src/generate.rs`.

| Task | Notes |
|------|-------|
| **M2.1 `StepGenerateConfig`** | Mirror `GenerateConfig`: max_new_tokens, max_denoising_steps, entropy_bound, seed, layers |
| **M2.2 Block outer loop** | Same as engine: prefill prompt → repeat { denoise until stop → commit argmax canvas → extend KV } |
| **M2.3 Canvas lifecycle** | Init from `initialize_canvas`; after stop read `CanvasState.ids`; commit **argmax** (not raw ids — match engine `argmax_canvas_tokens`) |
| **M2.4 Step loop** | `StepParams.kv_len` updated each block; `CanvasState.step` reset per block; SC logits from prior step via b6 |
| **M2.5 Early stop** | Honor `stop_flag`; respect max steps (48 default); linear temperature via existing `temp_at()` |
| **M2.6 Output** | `GenerateOutput`-compatible struct; decode via `tokenizer.json`; print like `print_generate_output` |
| **M2.7 CLI** | `generate-monolithic` flags aligned with `generate-gpu` (`-p`, `--seed`, `--steps`, `--layers`) |

**Exit:** `generate-monolithic -p "hello" --seed 42 --layers 30` produces readable text; token stream starts with prompt ids; blocks_committed ≥ 1.

---

## Phase M3 — Performance (production speed)

**Deliverable:** 30-layer end-to-end ≥ `generate-gpu` on M3 Pro (target: match PLAN2 ~5.3 s/step → improve toward ≤ 4 s/step).

| Task | Impact | Notes |
|------|--------|-------|
| **M3.1 ICB record/replay** | High | Encode dispatch schedule once at load; replay per step with patched b2/b5/b4 offsets only |
| **M3.2 SC softembed fast path** | High @ step>0 | Replace O(vocab×hidden) loop with prob materialization + `k_gemm_q8` (noted in shader) |
| **M3.3 Drop dispatch overhead** | Medium | Fuse bucket_count+fill; batch pipeline binds; invariant b0/b1 bind once per step |
| **M3.4 MPS Q8 for lm_head** | Medium | 256×262144×2816 — evaluate MPS vs fused `k_gemm_q8` |
| **M3.5 MPS scratch pooling** | Medium | Reuse ~92 MiB weight scratch across GEMMs; avoid alloc pressure |
| **M3.6 f16 vs f32 activations** | TBD | Engine uses f32; monolithic uses f16 arena — quantify quality/speed tradeoff |
| **M3.7 MoE path** | Medium | Evaluate MPS grouped Q4 vs `k_moe_grouped` native; expert traffic ~4 GiB/step @ 30L |
| **M3.8 Single sync per block** | High | One `waitUntilCompleted` per denoise step (already true); zero CPU readback in hot path |
| **M3.9 Memory budget doc** | Required | Arena 25 MiB + KV + logits 134 MiB + MPS scratch ~108 MiB + blob 15 GiB — print at load |

**Benchmark discipline** (same as PLAN2):

```bash
# Apples-to-apples: both with kv prefill @ 64, 30 layers
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --kv-len 64 --iters 5
cargo run --release --features metal -- -m $WEIGHTS bench-step --layers 30 --iters 5

# End-to-end
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "hello" --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS generate-gpu -p "hello" --layers 30 --seed 42
```

**Exit:** `bench-step-kernel` ≤ `bench-step` @ 30L, kv=64; `generate-monolithic` tok/s ≥ `generate-gpu` ±10%.

---

## Phase M4 — Production integration

**Deliverable:** safe to run as default GPU path behind a flag; operable in CI and interactive use.

| Task | Notes |
|------|-------|
| **M4.1 Feature flag** | `DGQ_MONOLITHIC=1` or `--monolithic` on default generate; fallback to engine on error |
| **M4.2 Telemetry** | Port `ForwardTelemetry` / `SessionTelemetry` hooks; print same summary as `generate-gpu` |
| **M4.3 Weight hot-reload** | Keep transformer loaded across prompts (match PLAN2 interactive lesson) |
| **M4.4 Error surfaces** | Non-finite logits → abort with checkpoint dump; Metal API validation clean |
| **M4.5 Config from model** | Read canvas_length, vocab, layer types, rms_eps from `config.json` — no hardcoded 2816/30 |
| **M4.6 `--forward-only` bench mode** | Keep for perf isolation (no sampler) |
| **M4.7 Docs** | Update README usage section; cross-link this plan |

**Exit:** one-command generate works on fresh `.dgq` download; CI job runs `step-smoke` + `generate-monolithic` parity fixture.

---

## Phase M5 — Quality, CI, and ship criteria

| Gate | Requirement |
|------|-------------|
| **Golden parity** | `generate-monolithic-parity` vs `fixtures/generate/dgq_hello_*` (new or extended) |
| **Regression** | `step-smoke --layers 30` in CI (skip if no weights); `make test` green |
| **Memory** | Peak RSS ≤ 24 GiB budget on base M4 with q4 `.dgq` (document sysctl if needed) |
| **Determinism** | Same seed → same tokens with `DGQ_MPS_Q4=0`; document MPS nondeterminism if any |
| **License / weights** | Same as engine — no new deps |

**Ship definition:** `generate-monolithic` replaces `generate-gpu` as default on macOS+metal when `DGQ_MONOLITHIC=1`, with engine fallback until M5 gates pass.

---

## Known issues & technical debt (track explicitly)

| ID | Issue | Severity | Phase |
|----|-------|----------|-------|
| KV-1 | `GpuKvCache` incompatible with b4 layout | Blocker for generate | M1 |
| SC-1 | `k_sc_softembed` O(vocab×hidden) | Perf @ step>0 | M3 |
| DISPATCH-1 | ~130 encoder calls/step/layer | Perf | M3 |
| MOE-1 | Atomic f32 scatter nondeterministic | Parity | M0 |
| METAL-1 | `tanh` large-input NaN (fixed via clamp) | Done | — |
| MEM-1 | MPS Q4 weight scratch up to ~92 MiB | Memory | M3 |
| BENCH-1 | Historical benches used kv_len=0 vs engine 64 | Misleading | M1 |
| FULL-1 | 5 full-attention layers need shape-specialized pipelines | Correctness | M0 |
| RNG-1 | `init_canvas_state` may not match `Rng::new(seed+1)` | Parity | M0 |

---

## Suggested execution order

```
M0 (parity) ──► M1 (KV) ──► M2 (generate loop) ──► M3 (perf) ──► M4 (integrate) ──► M5 (ship)
     │              │              │                    │
     └──────────────┴──────────────┴── can overlap M3.1–M3.3 after M2.1 proves loop
```

**Critical path:** M0.4 (30L finite) → M1.2 (prefill writer) → M2.2 (block loop) → M3.1 (ICB).

**Quick win already done:** MPS Q4 (26c3a62), GPU softcap (dde56a6), first-step SC skip (dde56a6).

---

## File map

| Path | Role |
|------|------|
| `shaders/diffgemma_step.metal` | All step kernels + dispatch schedule comment |
| `shaders/qgemm.metal` | Authoritative Q4 dequant reference |
| `src/metal/step_kernel.rs` | ABI, layout builder, encode driver, CLI backends |
| `src/metal/mod.rs` | Public exports |
| `src/main.rs` | `step-smoke`, `step-probe`, `bench-step-kernel` |
| `src/generate.rs` | Reference block loop to mirror |
| `src/sample.rs` | Authoritative sampler semantics |
| `src/metal/kv_cache.rs` | **Legacy** — do not use for monolithic b4 |
| `src/metal/encoder_extend.rs` | Reference for extend prefill behavior |

---

## Open questions (resolve in M0/M1)

1. **Unify KV layouts?** Long-term, one layout for engine + monolithic reduces code — but migration cost is high; monolithic-first writer may be faster to ship.
2. **Keep f16 arena?** Saves bandwidth; may complicate parity — decide before M5 golden lock.
3. **ICB vs Metal 3 command replay?** Prototype both in M3.1; pick lower complexity.
4. **When to delete engine path?** Only after M5 + 30L perf win sustained for 2 weeks of dogfooding.

---

*Last updated: 2026-06 — reflects commits through `26c3a62` (MPS Q4, softcap, SC skip, MoE rmsnorm fuse).*
