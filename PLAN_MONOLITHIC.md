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
| Tooling | ✅ `step-smoke`, `step-probe`, `step-kv-check`, `bench-step-kernel`, `generate-monolithic` |

### Measured vs engine

| Benchmark | Monolithic | Engine (`bench-step`) | Caveat |
|-----------|------------|----------------------|--------|
| Forward-only @ 3L | **2.33 s/step** | 2.44 s/step | step-kernel `kv_len=0`; engine prefill @ 64 |
| Full smoke (1 step, sampler) | ~2.7 s | — | Random canvas, no prompt |
| 30 layers forward-only | **4.80 s/step** (kv=64) | **7.87 s/step** | step-kernel beats engine ~39% |
| 30L generate end-to-end | **6.58 tok/s** (8 steps/block) | **3.89 tok/s** (~33 s/step) | monolithic ~69% faster denoise |

### What does **not** work yet

- **Multi-block extend** — single 256-token block verified; `max_new_tokens > 256` extend path implemented but not heavily tested.
- **Parity vs engine @ kv>0** — `step-parity` still kv_len=0 only.
- **Layer coverage** — full 30L generate verified @ hello prompt; 5 full-attention layers exercised in forward path.
- **Perf debt** — ~130 dispatches/step/layer; recompiles runtime per generate invocation (no cross-prompt reuse yet).
- **Chat-quality generate** — chat template + default early-stop often commits all-pad blocks before canvas converges; see [Inference semantics](#inference-semantics-first-principles) below. Parity goldens use `--raw`; templated prompts are not yet a ship gate.

---

## Inference semantics (first principles)

DiffusionGemma has **two different “commit” stages**. Conflating them explains empty/pad output and the “~16 tokens” question.

### Per denoise step (~15–20 canvas positions)

Each decoder forward still computes logits for all **256** canvas positions. The **entropy-bound sampler** (`entropy_bound = 0.1` in `src/sample.rs`) only **freezes** the lowest-entropy positions whose cumulative entropy stays ≤ 0.1; the rest are re-noised. In practice that is **~15–20 positions per step** — not a separate “emit 16 tokens to the user” mode.

```
denoise step:  forward(256) → accept ~15–20 low-entropy slots → renoise rest
               (repeat until converged or step limit)
```

Both engine (`generate.rs` + GPU sampler) and monolithic (`k_sample_*` in `diffgemma_step.metal`) implement this intra-step accept/renoise loop.

### Per block end (256 tokens to user / KV)

When the block finishes (adaptive early stop or `max_denoising_steps`), official behavior (model card, vLLM) is:

1. Take final **argmax** over all 256 canvas positions.
2. **Emit all 256** to output.
3. Run causal **encoder extend** on all 256 into KV (b4 monolithic / `GpuKvCache` engine).
4. Start a fresh 256-token canvas for the next block (if `max_new_tokens > 256`).

Our block drivers match this shape (`src/generate.rs`: `sequences.extend_from_slice(&argmax_canvas_tokens)` then `extend_encoder_kv` on the same 256 tokens).

**Do not** “strip pads and continue” as the primary fix for bad output — that diverges from official KV semantics. Fix **premature block commit** (stop before the canvas has converged to real tokens).

### Adaptive early stopping (why we stop at all)

Early stop is an **optimization**, not the core algorithm. Model card / `ARCHITECTURE.md`:

| Criterion | Value |
|-----------|--------|
| Max denoising steps | **48** (upper bound) |
| Typical effective steps | **12–16** (task-dependent) |
| Early stop | mean canvas entropy **< 0.005** AND argmax **stable** for 2 consecutive steps |
| Entropy bound (per-step accept) | **0.1** |
| Temperature | linear **0.8 → 0.4** |

Implemented in `StableConfidentStopper` (`src/sample.rs`) and monolithic `StepParams.conf_threshold` / `stop_flag`.

**CLI mismatch:** `main.rs` defaults `--steps` to **2** for fast parity/bench — not model-card production. With early stop **on** (default), runs often exit at step 2 when argmax stabilizes on **degenerate tokens** (`<pad>` id 0 or filler `262143`), which is confident but not meaningful text.

**Recommended for chat / quality debugging:**

```bash
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic \
  -p "Hello" --layers 30 --steps 48 --no-early-stop --seed 42

# Parity / legacy goldens (bare BPE, not chat template):
... generate-monolithic-parity -p hello --raw --layers 3 --steps 4 --seed 42 --no-early-stop
```

### Chat templating (2026-06)

| Item | Status |
|------|--------|
| `src/chat_template.rs` — HF-matched token assembly (`<bos>`, `<\|turn>`, `<turn\|>`, `<\|channel>`, `<channel\|>`) | ✅ |
| Default `-p` wraps user text; `--raw` for bare BPE (parity goldens) | ✅ |
| `chat` REPL on `StepGenerateSession` | ✅ |
| `tokenize` shows `formatted` + ids; Python `test_chat_template.py` parity | ✅ |
| Readable chat output @ default `--steps 2` + early stop | ❌ open (STOP-1 / QUAL-1) |

Chat prompt ends at empty thought channel (`<|channel>thought\n<channel|>`); generation needs enough denoise steps for the canvas to converge past that structure.

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
| **M2.1 `StepGenerateConfig`** | ✅ `src/metal/step_generate.rs` mirrors `GenerateConfig` fields |
| **M2.2 Block outer loop** | ✅ prefill → denoise until stop → commit argmax → extend KV |
| **M2.3 Canvas lifecycle** | ✅ `reset_block` + read `CanvasState.prev_argmax` for commit |
| **M2.4 Step loop** | ✅ `StepParams.kv_len` patched per block; SC via b6 logits on step>0 |
| **M2.5 Early stop** | ✅ GPU `stop_flag` + `max_steps`; `--no-early-stop` sets `conf_threshold=MAX`. ⚠️ No guard against degenerate all-pad/stable argmax — see [Inference semantics](#inference-semantics-first-principles) |
| **M2.6 Output** | ✅ returns `GenerateOutput`; decode/print via existing `print_generate_output` |
| **M2.7 CLI** | ✅ `generate-monolithic` (`-p`, `--seed`, `--steps`, `--layers`) |
| **M2.8 Chat template** | ✅ `chat_template.rs`, `--raw`, `chat` REPL; default `-p` uses Gemma 4 turn format |

**Exit:** `generate-monolithic -p "hello" --seed 42 --layers 30` produces readable text; token stream starts with prompt ids; blocks_committed ≥ 1.

**Note:** Exit was written for raw/`hello` smoke. Templated chat at `--steps 2` + early stop may still emit pad-heavy blocks — use `--steps 48 --no-early-stop` until STOP-1/QUAL-1 land.

```bash
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "hello" --seed 42 --layers 30
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "hello" --layers 3 --steps 4 --seed 42
```

---

## Phase M3 — Performance (production speed)

**Deliverable:** 30-layer end-to-end ≥ `generate-gpu` on M3 Pro (target: match PLAN2 ~5.3 s/step → improve toward ≤ 4 s/step).

| Task | Impact | Notes |
|------|--------|-------|
| **M3.1 ICB record/replay** | High | ✅ Partial: `OnceLock` pipeline compile cache (~65ms → ~4ms on repeat loads). Full ICB deferred. |
| **M3.2 SC softembed fast path** | High @ step>0 | ⚠️ Experimental `DGQ_SC_GEMM=1` (probs + `k_gemm_q8_rowk`); **regresses** (~130s/step) — default **off**. Needs tiled vocab chunks or MPS matmul. |
| **M3.3 Drop dispatch overhead** | Medium | Not started (fuse bucket_count+fill; hoist invariant binds) |
| **M3.4 MPS Q8 for lm_head** | Medium | Not started |
| **M3.5 MPS scratch pooling** | Medium | ✅ Scratch buffers allocated once per `StepRuntime` (reuse across steps) |
| **M3.6 f16 vs f32 activations** | TBD | Not started |
| **M3.7 MoE path** | Medium | Not started |
| **M3.8 Single sync per block** | High | ✅ One `waitUntilCompleted` per denoise step; no hot-path CPU readback |
| **M3.9 Memory budget doc** | Required | ✅ `log_step_memory_budget()` printed at `build_step_runtime` |

**Measured (M3 Pro class, `/tmp/quantized-weights`, `DGQ_MPS_Q4=1`):**

| Benchmark | Monolithic | Engine | Notes |
|-----------|------------|--------|-------|
| `bench-step-kernel` 30L kv64 forward-only | **4.80 s/step** | — | 3 iters, warmup 5.07s |
| `bench-step` 30L | — | **7.87 s/step** | 152 syncs, 1.3 GiB readback/step |
| `generate-monolithic` 30L hello seed42 | **6.58 tok/s** (38.9s/8 steps) | — | 1 block, 256 new tokens |
| `generate-gpu` 30L hello seed42 | — | **3.89 tok/s** (65.9s/2 steps) | early-stop @ step 2; ~33 s/step wall |
| Pipeline compile (repeat load) | **~3 ms** | — | `OnceLock` cache (was ~65 ms) |

**Exit (M3):** ✅ Met — see measured table. Remaining items (ICB replay, SC GEMM, dispatch fusion, MPS lm_head) are incremental; not blocking M4.

```bash
# Apples-to-apples: both with kv prefill @ 64, 30 layers
cargo run --release --features metal -- -m $WEIGHTS bench-step-kernel --layers 30 --kv-len 64 --iters 5
cargo run --release --features metal -- -m $WEIGHTS bench-step --layers 30 --iters 5

# End-to-end
cargo run --release --features metal -- -m $WEIGHTS generate-monolithic -p "hello" --layers 30 --seed 42
cargo run --release --features metal -- -m $WEIGHTS generate-gpu -p "hello" --layers 30 --seed 42

# Experimental SC GEMM (not recommended — slow)
DGQ_SC_GEMM=1 cargo run --release --features metal -- -m $WEIGHTS step-smoke --layers 30 --steps 4
```

---

## Phase M4 — Production integration

**Deliverable:** safe to run as default GPU path behind a flag; operable in CI and interactive use.

| Task | Notes |
|------|-------|
| **M4.1 Feature flag** | ✅ `DGQ_MONOLITHIC=1` or `--monolithic` routes default `generate` / `generate-gpu` to monolithic on `.dgq`; engine fallback on error |
| **M4.2 Telemetry** | ✅ Per-step `SessionTelemetry` (1 sync, 0 readback); same summary format as `generate-gpu` |
| **M4.3 Weight hot-reload** | ✅ `StepGenerateSession` holds runtime across `generate_with_session` calls |
| **M4.4 Error surfaces** | ✅ `check_logits_finite()` after each denoise step |
| **M4.5 Config from model** | ✅ `validate_step_model()` at runtime; layers default from `config.json` |
| **M4.6 `--forward-only` bench mode** | ✅ Already in `bench-step-kernel` |
| **M4.7 Docs** | ✅ `step-ci` + `fixtures/generate/README.md` cross-link |

**Exit:** one-command generate works on fresh `.dgq` download; CI job runs `step-ci` (verify + generate smoke).

```bash
# CI gate (skips gracefully if no .dgq weights)
cargo run --release --features metal -- -m $WEIGHTS step-ci --layers 3
```

---

## Phase M5 — Quality, CI, and ship criteria

| Gate | Requirement |
|------|-------------|
| **Golden parity** | ✅ `generate-monolithic-parity` vs `fixtures/generate/monolithic_hello_steps4_layers3.json` (`DGQ_MPS_Q4=0`, **`--raw`**) |
| **Chat template parity** | ✅ Rust `format_chat_token_ids` vs HF `apply_chat_template` (`python/tests/test_chat_template.py`) |
| **End-to-end chat text quality** | ❌ Not gated — engine/monolithic with templated `-p` + default early-stop often all-pad; distinct from token-id parity |
| **Regression** | ✅ `step-ci --layers 3` (config + verify + parity); GitHub Actions `ci.yml` |
| **Memory** | Peak RSS ≤ 24 GiB budget on base M4 with q4 `.dgq` (document sysctl if needed) |
| **Determinism** | ✅ Same seed → same tokens with `DGQ_MPS_Q4=0` @ `monolithic_hello_steps4_layers3` |
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
| STOP-1 | Early stop fires on degenerate all-pad / filler argmax (confident but not text) | Quality | M5+ |
| QUAL-1 | Templated chat + default `--steps 2` → no readable reply; need pad-aware stop + chat defaults | Quality | M5+ |
| CLI-1 | `--steps` default is 2 (parity); model card recommends up to 48 | UX | M5+ |
| CHAT-1 | Display decodes full 256 block incl. pads; strip pads in `print_generate_output` / `chat` only | UX | M5+ |

### Planned fixes (STOP-1 / QUAL-1 / CLI-1)

1. **Pad-aware early stop** — do not treat all-pad (or all-filler) stable argmax as convergence.
2. **Chat-oriented CLI defaults** — e.g. `chat` / default generate: `--steps 48`, stricter or disabled early-stop until steps_eff ≥ ~12.
3. **Decode hygiene** — show only non-pad new tokens in text preview; block commit still emits 256 argmax per official semantics.
4. **Optional telemetry** — `steps_eff` per block + histogram of accepted positions/step (`PLAN2.md` Q4).

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
| `src/chat_template.rs` | Gemma 4 turn formatting + HF-matched token ids |
| `src/tokenizer.rs` | BPE + `added_tokens` for chat special tokens |
| `fixtures/generate/README.md` | Golden commands (`--raw` for legacy prompts) |
| `src/metal/kv_cache.rs` | **Legacy** — do not use for monolithic b4 |
| `src/metal/encoder_extend.rs` | Reference for extend prefill behavior |

---

## Open questions (resolve in M0/M1 / M5+)

1. **Unify KV layouts?** Long-term, one layout for engine + monolithic reduces code — but migration cost is high; monolithic-first writer may be faster to ship.
2. **Keep f16 arena?** Saves bandwidth; may complicate parity — decide before M5 golden lock.
3. **ICB vs Metal 3 command replay?** Prototype both in M3.1; pick lower complexity.
4. **When to delete engine path?** Only after M5 + 30L perf win sustained for 2 weeks of dogfooding.
5. **Block commit shape?** Official emit is always 256 argmax; per-step accept is ~15–20 positions only inside the denoise loop — do not confuse the two when debugging output.
6. **Chat ship gate?** Add templated-prompt golden or eval harness once STOP-1/QUAL-1 fixed; until then ship parity on `--raw` only.

---

*Last updated: 2026-06 — M0–M5 core gates, chat template (`chat_template.rs`, `--raw`, `chat`), inference-semantics notes (two-level commit, early-stop pitfalls).*
