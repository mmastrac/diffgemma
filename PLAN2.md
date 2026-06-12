# diffgemma-mps — implementation plan (v2: near-real-time on Apple Silicon)

Low-dependency Rust inference engine for [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it), Iris-style (mmap weights, CPU reference path, Metal acceleration).

See `ARCHITECTURE.md` for model semantics. Phases 0–11 of PLAN v1 are **done** (weight loading → end-to-end GPU generation with golden parity); summarized below, history in git.

**Target:** sustained ≥ 8 tok/s end-to-end on 24 GiB base M-series; 25+ tok/s on Pro/Max with 36 GiB. Baseline today: 0.03 tok/s (`--no-early-stop --steps 48`), ~2.2 tok/s (steps=1), 18.7 s prefill.

---

## Why v2 exists: the working-set wall

Measured (steps=48, canvas=256): 106,480 expert-cache evictions, ~78 ms each (CPU bf16 transpose + upload) = 8,374 s of 8,393 s total. The LRU cannot work:

| Quantity | Value |
|---|---|
| Unique experts touched / layer / step (measured) | ~74 of 128 (uniform-routing bound: ~all 128) |
| Per-step expert working set @ bf16 | 30 × 74–128 × 11.3 MiB ≈ **25–42 GiB** |
| Expert LRU budget | 6.6 GiB (584 slots vs ~2,200+ touches/step) |
| Conclusion | ~100% miss rate **by construction** |

At canvas=256, top-8/128 routing activates most experts every step: **MoE is effectively dense for canvas-sized batches.** Per-step traffic ≈ full expert pool, every step, forever.

**bf16 is unshippable on target hardware.** 48 GiB > 24–36 GiB unified memory; even with free paging, SSD (~5–7 GB/s) floors a step at ~5–7 s. No kernel work changes this.

**Therefore: quantize until resident.** At ~4.5 bpw the whole model is ~14 GiB → GPU-resident on 24 GiB. The LRU, the transpose cache, and per-layer weight paging get **deleted**, not improved.

### Throughput model (calibrate in Q0, don't trust the constants)

```
step_time ≈ max( W_step / BW ,  F_step / (TFLOPS × MFU) ) + sync + sampler
F_step    ≈ 2 × 3.8e9 active params × 256 tokens ≈ 1.95 TFLOP
W_step    ≈ resident quantized bytes actually touched ≈ 8–14 GiB (q4, measured expert uniqueness)
block     ≈ (steps_eff ≈ 20 w/ early stop + 1 incremental prefill) × step_time
tok/s     ≈ 256 / block
```

Estimated landing zones (assumptions: 0.3–0.5 MFU, q4.5, steps_eff=20 — **verify**):

| Device class | BW (GB/s) | est. step | est. tok/s |
|---|---|---|---|
| **M3 Pro (measured BW=112 GiB/s, compute ~0.18 TFLOP/s @ dense shape)** | **112** | **~11 s compute floor; ~108 s w/ LRU today** | **~0.03 today → ~8+ post-quant** |
| M4 Max / M3 Max (36 GiB+) | 400–546 | 150–300 ms | 30–80 |
| M1/M2 Max | ~400 | 250–450 ms | 25–50 |
| M*-Pro (24–36 GiB) | 150–273 | 300–700 ms | 15–40 |
| Base M4 (24 GiB) | ~120 | 0.8–1.5 s | 8–15 |

Honest caveat (was in v1, still true): canvas=256 makes each step dense-cost, so Apple Silicon won't see the H100 multiples from the model card. Reading-speed-plus is the realistic win.

**M3-class is compute-bound, not bandwidth-bound.** At canvas=256: ~1.95 TFLOP/step vs M3 Pro ~5–7 TFLOPS (no fp16 rate-doubling on Apple GPUs) → ~0.8–1.2 s/step compute vs ~55–90 ms q4 weight read @ 150 GB/s. M3 Max: ~400 ms compute vs ~30–45 ms bandwidth. Consequences: (a) q4 vs q5 is a **residency/quality** choice on M3, not speed — q5 is ~free in wall-clock on 36 GiB; (b) Q5 (MoE MFU) outranks bandwidth tricks; (c) get activation precision right early — fp16 math, f32 accumulate.

---

## Done (PLAN v1, phases 0–11) — compressed

| # | Milestone | Status |
|---|---|---|
| 0–5 | mmap shards, config, CPU kernels, decoder/encoder stacks, KV cache | ✅ |
| 6–7 | Entropy-bound block sampler, tokenizer (+ uv parity tests) | ✅ |
| 8–10 | Metal GEMM/attention/MoE, fused norm+QKV+RoPE+GQA+o_proj, fused FF | ✅ |
| 11 | `generate-gpu` / `generate-parity`, GPU KV prefill/extend, goldens | ✅ |
| 12 | FastSlice, paging, pool/trim discipline, bench-decoder | ✅ (superseded in part by v2) |

Keep: golden-parity discipline, `bench-decoder` regression tables, CPU/BLAS oracle path, fused-submit lessons (no per-layer `pool.trim`, no chained readback on same buffer).

---

## Phase Q0 — Instrument the model ✅-gate for everything else

**Deliverable:** numbers that replace the estimates above.

| Task | Notes |
|---|---|
| Per-step phase timers in `generate-gpu` | weights-bytes touched, unique experts/layer (histogram), sync count, sampler ms, logits-readback bytes |
| `bench-step` CLI | full 30L, canvas=256, real routing (prompt-seeded), reports ms/step + bytes/step |
| Device probe | BW estimate (memcpy kernel), fp16/fp32 GEMM TFLOPS probe at sizes (256×2816×2816, 16×2816×~1400) |
| Record table in PLAN | this machine (36 GiB, working_set 27 GiB) first |

**Exit:** measured BW, MFU at MoE-relevant GEMM shapes, expert-uniqueness distribution. Update the landing-zone table.

### Measured on dev box (Apple M3 Pro, 27 GiB working set, 2026-06-03)

| Metric | Value |
|---|---|
| `probe-device` memcpy BW | **112 GiB/s** |
| GEMM dense 256×2816×2816 | **184 GFLOP/s** (~10.6 s for 1.95 TFLOP/step, kernel-only) |
| GEMM MoE 16×2816×1408 | **110 GFLOP/s** |
| `bench-step` wall (30L, canvas=256, 1 iter) | **107.5 s/step** |
| GPU syncs/step | **151** |
| GPU readback/step | **1.24 GiB** (residual/norm/MoE scatter; logits separate) |
| lm_head CPU logits/step | **256 MiB** |
| Expert unique/layer (mean/min/max) | **62 / 33 / 96** |
| Expert bf16 touched/step | **20.6 GiB** |
| Expert LRU misses/step (cold-ish) | **1858** (~21 GiB uploaded) |

**Takeaways:** M3 Pro is **compute-bound** at canvas=256 for the dense GEMM probe (~0.18 TFLOP/s vs ~1.95 TFLOP/step → ~11 s compute floor), but the **dominant wall today is expert LRU thrashing** (20+ GiB/step touched, 6.6 GiB budget). Quantized full residency (Q1/Q2) is mandatory; instrumentation confirms PLAN v2 estimates.

---

## Phase Q1 — Offline quantizer + `.dgq` format (the long pole)

**Deliverable:** `diffgemma-mps quantize model/transformer -o model.dgq --profile q4` produces a single resident-friendly file.

| Task | Notes |
|---|---|
| Group quantization | **affine int4** (groups of 32 along K; fp16 scale+offset, Q4_1-style). Not literal FP4/e2m1 — Metal has no native 4-bit type, everything is nibbles dequantized in-kernel; int4 affine is better matched to weight distributions and better characterized. k-quant later only if quality demands |
| Mixed-precision profile | experts q4; dense attn/MLP q4–q5; **embed/lm_head (tied) q8**; self-conditioning proj q8; **router f16 weights, f32 accum + f32 logits** (routing = control flow: near-boundary logit noise flips experts discretely; f16 compare breaks CPU/GPU tie-break parity); norms + layer_scalars f16/f32 |
| **Checkpoint-orientation layout** | quantized blocks stay `[out, in]` row-major in HF tensor coordinates; kernels read it natively (dot-product rows, à la llama.cpp/MLX). Quant groups run along K → nibbles+scale contiguous per row already. No transpose at conversion or runtime — the 78 ms/miss was a kernel-orientation artifact, not a layout necessity. Wins: streaming converter (no reorder), oracle shares layout, parity indices match checkpoint. Row-interleaving (8-row simdgroup blocks) reserved as a converter-side **format version bump** iff Q5 profiling shows load-bound expert GEMMs |
| Page-aligned tensors + header index | mmap file → `makeBuffer(bytesNoCopy:)` → zero-copy `MTLBuffer`(s), `StorageModeShared`. Watch max buffer length; shard if needed |
| CPU dequant reference | `q*` → f32 on CPU so the oracle can run the *same* quantized weights |
| Two profiles | `q4` (~13–14 GiB, 24 GiB floor) and `q5` (~16–18 GiB, 32–36 GiB devices; ~free in speed on compute-bound M3 — see model above) |

**MoE quant sensitivity:** only ~3.8B params active per token — 4-bit damage behaves like quantizing a ~4B dense model, not a 26B one; don't import 70B-dense Q4 intuitions. If fixtures drift: bump shared expert + first/last layers to q5/q6 **before** touching routed experts (cheap; routed experts are where the bytes are).

**Memory budget @ q4 (verify):** experts ≈ 12.2 GiB, dense ≈ 1.0, embed q8 ≈ 0.75, KV (sliding-window capped: 25 layers × 1024 tok + 5 full-attn layers × T) ≈ 0.4 @ T=4K, activations/logits scratch ≈ 1.5–2 → **~16 GiB peak**. On 24 GiB machines the default working-set cap (~75%) is tight: document `sysctl iogpu.wired_limit_mb` and/or default to q4 with smaller scratch.

**Exit:** file loads in <2 s (mmap, no copy); per-tensor CPU dequant matches Python-quantized reference; total resident bytes printed and ≤ budget.

---

## Phase Q2 — Dequant-in-kernel GEMMs (residency goes live)

**Deliverable:** all weight-consuming GPU paths read quantized blocks directly; **never** materialize bf16/f32 weight buffers (that recreates the bandwidth problem).

| Task | Notes |
|---|---|
| Quantized matvec/GEMM kernels | consume `[out, in]` row-major quantized blocks **as stored** (no runtime relayout); load nibbles + scales, dequant in registers/threadgroup; fp16 math, f32 accumulate. Prior art: llama.cpp / MLX Metal kernels (reference, not deps) |
| MoE grouped GEMM | sort/bucket tokens by expert (GPU, Q3 finishes this; CPU-sorted interim ok), one batched dispatch over present experts; reuse existing arena 2-sync structure |
| Dense layers + lm_head on quantized weights | embed gather stays CPU (tiny); lm_head must be GPU (q8 GEMM, 378 GFLOP/step) |
| **Delete** | expert LRU, transpose cache, `GpuLayerWeightCache` paging, resident-cap/eviction logic, bf16 GPU weight upload path |
| Regenerate goldens | quantized goldens (CPU-dequant oracle on same `.dgq`); keep bf16 CPU fixtures separately for quality eval |

**Exit / gates (dev box):**
- `bench-step` full 30L canvas=256 ≤ **500 ms** (vs ~175 s today implied).
- Zero evictions, zero transposes (counters must read 0).
- Prefill (1 tok) ≤ 0.5 s (vs 18.7 s).
- `generate-parity` passes on quantized goldens; bf16-vs-q4 divergence documented on fixtures.

---

## Phase Q3 — Kill the CPU round-trips

Once weights are resident, the per-step serial CPU work becomes the wall: logits readback [256 × 262,144] ≈ 134–268 MB/step + CPU softmax/entropy over 67M values; CPU router = 30 sync round-trips/step.

| Task | Notes |
|---|---|
| GPU router top-8 | finish `route_gpu`; **f32 logits into top-k** (f16 ≈ 3 sig. digits → adjacent experts within epsilon → CPU/GPU ordering flakiness); tie-break key = (logit, expert index), bit-exact vs CPU |
| GPU sampler | softmax → per-position entropy → commit mask (entropy_bound) → categorical sample for re-noised positions → write canvas in place |
| Counter-based RNG | Philox/Threefry keyed (seed, block, step, position); implement identically in CPU oracle; regenerate goldens once |
| GPU early-stop reduction | avg-entropy scalar + argmax-stable-2-steps flag; CPU reads 2 scalars + committed ids per step |
| Self-conditioning on GPU | sc probs/logits never leave GPU; feed prev-step distribution path entirely on-device |
| Command-buffer structure | encode whole step into 1–3 command buffers; no mid-step `waitUntilCompleted`; sampler dependency stays on-GPU |

**Exit / gates:**
- Readback/step ≤ 1 MB (ids + scalars), down from ~134–268 MB.
- Syncs/step ≤ 3 (was ≥ 30 router + sampler).
- `bench-step` ≤ **350 ms**; `generate-gpu` end-to-end ≥ 10 tok/s on dev box.

---

## Phase Q4 — Step-count and step-shape efficiency

| Task | Notes |
|---|---|
| lm_head + sampler over **uncommitted positions only** | committed tokens are frozen; lm_head is ~19% of step FLOPs and shrinks toward 0 late in the loop |
| Early-stop defaults on | model-card config (entropy_bound 0.1, temp 0.8→0.4, avg-entropy 0.005 + stable-argmax); `--no-early-stop` stays a stress bench only |
| Incremental prefill reuse | committed block causal pass shares decoder weights/kernels; confirm it costs ≈ 1 step, no special-case paging |
| steps_eff telemetry | per-block effective steps + committed/step histogram in output |

**Exit:** tok/s improves ≥ 15% over Q3 on standard prompts; no golden change (exact-skip is mathematically identical for frozen positions' logits — verify, else gate behind flag).

---

## Phase Q5 — MoE kernel MFU

The per-expert GEMM is M≈16 rows (2048 draws / ~128 experts) — occupancy-hostile. This phase is profile-driven; only after Q2/Q3 numbers exist.

| Task | Notes |
|---|---|
| GPU token sort/scatter | bucket by expert on GPU, fuse weighted scatter-add tail (open item from v1) |
| Tile/threadgroup tuning at M=8..32 | simdgroup_matrix ops; benchmark per shape, table in repo. If load-bound here → row-interleaved format bump in converter (see Q1), never runtime relayout |
| Fuse activation into gate_up→down | gelu/swiglu in-kernel (Metal `tanh` clamp lesson stands) |
| Attention sync merges | carry-over from v1 ideas list |

**Exit:** MoE MFU ≥ 2× Q2 measurement or documented why not; `bench-step` ≤ **250 ms** on dev box (stretch).

---

## Phase Q6 — Quality-gated approximations (stretch, default-off)

| Idea | Risk |
|---|---|
| Freeze K/V of committed canvas positions within a block (Fast-dLLM-style) | approximation — bidirectional means committed reps legitimately evolve; gate on fixture divergence + eval prompts |
| Smaller canvas (128) for latency-sensitive mode | fewer tokens/block, worse weight-read amortization; measure, don't assume |
| Step distillation | training; out of scope here, note for upstream |

---

## Parity & quality strategy (v2 additions)

| Level | What |
|---|---|
| Quant unit | per-tensor-class q⇄f32 round-trip error bounds |
| Oracle | CPU dequant path runs same `.dgq`; bf16 CPU retained for quality reference |
| Goldens | regenerated once at Q2 (quant) and once at Q3 (RNG); never silently |
| Quality eval | fixed prompt set, bf16 vs q4 vs q5 side-by-side text + logit-divergence stats; ablate embed/lm_head precision first if quality drops |
| Bench | `bench-step` table before/after every Q-phase commit (v1 discipline) |

---

## Risk register (v2)

| Risk | Mitigation |
|---|---|
| q4 quality loss (MoE ≈ 4B-dense sensitivity) | q8 embed/lm_head/SC; bump shared expert + first/last layers first; q5 profile on 36 GiB; k-quants as fallback |
| Metal max buffer length < file size | shard `.dgq` into per-group buffers; offsets in header |
| 24 GiB working-set cap (~18 GiB) too tight | q4 profile + scratch diet; document `iogpu.wired_limit_mb`; measure real peak in Q0 counters |
| GPU top-k / RNG parity drift | bit-exact tie-break spec + counter-based RNG mirrored in oracle; goldens regenerated deliberately |
| M=16 expert GEMMs stuck at low MFU | Q5 tuning; worst case still bandwidth-floored ≈ fine for ≥8 tok/s target |
| Estimates wrong | Q0 exists so every later gate uses measured constants |

---

## Milestones (v2)

| # | Milestone | Gate (dev box, 36 GiB) |
|---|---|---|
| Q0 | Instrumentation + device probes | measured BW/MFU/expert-uniqueness table |
| Q1 | `.dgq` quantizer + zero-copy load | ≤2 s load, ≤ budget resident, CPU dequant parity |
| Q2 | Quantized kernels, residency, deletions | step ≤ 500 ms; 0 evictions; prefill ≤ 0.5 s |
| Q3 | GPU router/sampler/early-stop | ≤3 syncs, ≤1 MB readback/step; ≥10 tok/s e2e |
| Q4 | Uncommitted-only lm_head, step telemetry | +15% tok/s |
| Q5 | MoE MFU | step ≤ 250 ms (stretch) |
| Q6 | Approximations | default-off, quality-gated |

---

## Immediate next step

**Q0**: add bytes/uniqueness/sync counters + `bench-step`, run device probes, replace the estimate table with measurements. Then start the Q1 quantizer (long pole — begin file format + CPU dequant while Q0 numbers come in).

```bash
cargo run --release --features metal -- bench-step --canvas 256 --layers 30 --iters 3
cargo run --release --features metal -- probe-device
cargo run --release -- quantize model/transformer -o model.dgq --profile q4   # Q1
```