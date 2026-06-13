# diffgemma-mps — engineering notes & history

Companion to `PLAN.md`. This file holds the things that aren't forward-looking tasks but that we'll regret losing: measured data, model-spec details, resolved-bug archaeology, and the reasoning behind decisions. When in doubt about "why is it this way," it should be answered here.

---

## 1. Project lineage

Three plans preceded this consolidation, archived in git:

- **PLAN.md (v1)** — original phased roadmap, phases 0-14. Got us from mmap weight loading to end-to-end GPU generation with golden parity (phases 0-11 done; 12 perf/FastSlice partial). The multi-kernel "engine" path.
- **PLAN2.md** — the near-real-time pivot. Diagnosed the working-set wall, defined the `.dgq` quantization program (Q0-Q6), shipped residency + grouped MoE + GPU sampler.
- **PLAN_MONOLITHIC.md** — the single-encoder step kernel (M0-M5). Took `diffgemma_step.metal` from sketch to a faster-than-engine path with prefill/extend writers and a chat loop.

The consolidated `PLAN.md` carries forward only open work. Everything below is the residue worth keeping.

---

## 2. Model spec (DiffusionGemma, Gemma-4 26B-A4B) — authoritative

From `config.json` + CPU reference (`kernels/cpu.rs`, `decoder_layer.rs`, `attention.rs`, `moe.rs`, `sample.rs`).

### Dimensions
- 30 layers, hidden 2816, vocab 262144, canvas 256.
- 16 Q heads / 8 KV heads (GQA); head_dim 256 (sliding), 512 (full).
- MoE: 128 experts, top-8, `moe_intermediate_size = 704`.
- Dense MLP `intermediate_size = 2112`. Total 25.2B params, 3.8B active.

### Norms
- Classic Gemma RMSNorm, eps 1e-6, `out = x/sqrt(mean(x^2)+eps) * w` — **not** AdaLN, **not** `(1+w)`.
- Pre-norm everywhere (residual + sublayer(norm(x))).
- QK-norm: per-head RMSNorm on Q/K with learned weights; V gets rms_norm_no_scale.
- Per-layer `layer_scalar` (bf16) multiplies the whole layer output.
- FF sandwich has many norms: input, post_attn, pre_ff, post_ff_1, pre_ff_2, post_ff_2, post_ff. Don't drop any.

### Attention
- Layer types: indices **5, 11, 17, 23, 29 = full_attention**; rest sliding.
- Sliding: head_dim 256, full rotary, theta 10000.
- Full: head_dim 512, partial_rotary_factor 0.25 (128 of 512 dims rotated), theta 1e6, rope_type "proportional".
- **RoPE rotation is split-half pairs** `(d, d+rotary_dim/2)`, NOT interleaved. (Cost us a rewrite — see section 5.)
- **Proportional-RoPE frequency exponent uses the FULL head_dim (512)** as denominator, not the rotary subset (128).
- No explicit 1/sqrt(d) attention scale — raw dot products -> mask -> softmax.
- Final logit softcapping: `tanh(x/30)*30` after the tied lm_head.
- **Sliding window (1024) applies to encoder prefill/extend only.** The decoder canvas path uses `DecoderAttnMask::all_valid` — full bidirectional, window NOT applied. This is non-obvious and matters for the mask kernel.
- **Full-attention layers have no v_proj tensor.** V is aliased from the k_proj output (rms_norm_no_scale, no RoPE). Confirmed against the checkpoint (layer 5 has only q/k/o; layer 0 has q/k/v/o).

### MoE
- Order: rms_norm_no_scale(stream) -> `router_scale[i] * hidden^-0.5` -> linear -> softmax over 128 -> top-8 -> renormalize top-8 to sum 1 -> multiply by `per_expert_scale[expert]`.
- Tie-break: higher prob wins; equal prob -> lower expert index (CPU and GPU must match).
- **No separate shared-expert tensor** — the dense MLP runs in parallel as the always-on path.
- Expert `gate_up_proj` trailing dim **1408 = fused [gate||up], 704 each**; split at 704. True inter = 704. (Reconciles the 11.3 MiB/expert figure and the 22.8B expert-param total; a per-branch 1408 would imply 45.7B params, impossible for a 25.2B model.)
- Activation: `gelu_pytorch_tanh` (Gemma), NOT SiLU.

### Embedding / lm_head
- Embed scale `sqrt(hidden) = sqrt(2816) ~= 53.066` on gather.
- Tied lm_head (`logits = final_hidden @ embed^T`), then softcap.

### Self-conditioning (the part not guessable from public material)
- 4 tensors: pre_norm, gate_proj, up_proj, down_proj (q8 in `.dgq`).
- Fed back: **softmax-weighted embedding mix of the previous step's processed logits** (`soft_embeddings_from_logits`) — not argmax, not raw logits.
- Injected before layer 0 only: `hidden = inputs_embeds + down(gelu_tanh(gate) * up(RMSNorm_w(soft_signal)))`, then rms_norm_no_scale.
- Step 1: self_conditioning_logits = None -> zero signal, but the SC MLP still runs on zeros.
- The soft mix gets the same sqrt(hidden) scale as token embeds (VERIFY-SC, confirmed in M0.2).

### Canvas / sampler
- Initial canvas: uniform random token ids (LCG seed).
- No noise-level / timestep embedding into the transformer.
- Canvas position ids: `kv_len .. kv_len+255` (continuation after prefill).
- Sampler defaults: entropy_bound 0.1, t_min 0.4 / t_max 0.8, stability_threshold 1, confidence_threshold 0.005, max_denoising_steps 48.
- **Temperature counts DOWN**: `cur_step` runs `max_steps..1`, `t = t_min + (t_max-t_min)*(cur/n)`. Step 1 -> t=0.8. (A count-up implementation is off by the full range — see section 5.)
- Entropy computed from temperature-scaled logits (despite a stale comment in `sample.rs` saying "before temperature").
- Denoiser sample: categorical draw from row softmax of tempered logits (inverse-CDF, `rng.next_f32()`).
- **Accept rule**: sort positions ascending entropy; `if prefix_sum <= entropy_bound { accept; prefix_sum += ent[idx] } else break`. Test-before-add, with break — the first position is always accepted when bound > 0. (An add-before-test implementation diverges — see section 5.)
- Renoise rejected positions in position order 0..255 (matches `renoise_canvas`).
- Early stop: mean_entropy < 0.005 AND argmax stable for `stability_threshold` prior steps.
- RNG: LCG `state = state*6966169279 + 1039523323`, uniform via high 32 bits. `Rng::new(seed)` => state = seed + 1. (VERIFY if `initialize_canvas` consumes from the same stream — then GPU state must start post-init, not seed+1; see RNG-1.)

---

## 3. The working-set wall (why .dgq exists)

The defining performance discovery. At canvas=256, top-8/128 routing touches most experts every step, so **MoE is effectively dense for canvas-sized batches**.

Measured bf16 (steps=48): 106,480 expert-cache evictions at ~78 ms each (CPU transpose + upload) = 8,374 s of 8,393 s total. Per-step expert working set 25-42 GiB vs a 6.6 GiB LRU = ~100% miss by construction. bf16 (48 GiB) can't be resident on 24-36 GiB at all; even free paging floors a step at ~5-7 s on SSD.

**Conclusion: quantize until resident.** q4 (~13-15 GiB) makes the whole model GPU-resident; the LRU, transpose cache, and per-layer paging get deleted, not optimized. This was an existence proof, not an optimization.

---

## 4. Measured data

### Q0 device probe (M3 Pro, 36 GiB, 2026-06)
| Metric | Value |
|--------|-------|
| Memcpy BW | 112 GiB/s |
| Dense GEMM 256x2816x2816 (custom naive) | 184 GFLOP/s (~3% of hw) |
| MoE GEMM 16x2816x1408 (custom naive) | 110 GFLOP/s |
| bench-step bf16 (30L, canvas=256) | 107.5 s/step |
| Syncs/step (engine) | 151-152 |
| Experts unique/layer | mean 62 (range 33-96) -> ~33 tokens/active-expert |
| Expert bytes/step bf16 | 20.6 GiB (~6 GiB @ q4) |
| LRU misses bf16 | 1858/step (~21 GiB re-upload) |

**Two walls:** (1) thrash = the 107.5 s; (2) GEMMs ran ~30x below the M3 Pro's ~5-6 TFLOPS class. M3 Pro is **compute-bound, not bandwidth-bound** — q4 weight read is only ~70 ms/step at 112 GiB/s. Kernel MFU mattered as much as residency.

### GEMM oracle (M3 Pro, re-bench)
| Path | Dense 256x2816x2816 | MoE M=33 |
|------|--------------------|---------| 
| Custom naive | 0.18 TFLOP/s | 0.15 TFLOP/s |
| MPS MPSMatrixMultiplication | 2.22 TFLOP/s | 0.35 TFLOP/s |

~12-18x on dense. Decision: dense linears go through MPS (per-layer dequant-to-scratch -> MPS); experts stay in the custom grouped kernel (per-expert MPS would be ~3,720 encodes/step; fp16-resident experts would cost 12+ GiB).

### Step-time progression (engine, 30L, M3 Pro)
107 s (bf16 thrash) -> 18.8 s (residency) -> 11.9 s (dense->MPS) -> **5.29 s (grouped MoE)**. Remaining wall after this is Q3: readback + sync + CPU sampler/router.

### Engine vs monolithic (M3 Pro, /tmp/quantized-weights, DGQ_MPS_Q4=1)
| Benchmark | Monolithic | Engine | Notes |
|-----------|-----------|--------|-------|
| Forward-only 3L | 2.33 s/step | 2.44 s/step | step-kernel kv=0; engine prefill @ 64 |
| 30L forward-only | **4.80 s/step** | 7.87 s/step | monolithic ~39% faster (152 syncs, 1.3 GiB readback engine-side) |
| 30L generate e2e | **6.58 tok/s** (8 steps/block) | 3.89 tok/s (~33 s/step, early-stop @ 2) | monolithic ~69% faster denoise |
| Pipeline compile (repeat) | ~3 ms | — | OnceLock cache (was ~65 ms) |
| Sampler readback | ~3 KB/step | — | full GPU sampler |

### .dgq blob facts
15.35 GiB, 1047 tensors: 454 q4_block (decoder matrices), 4 q8_row (embed + 3 SC projections), 589 raw (norms, router, scales, layer_scalars, vision). Vision = 356 tensors, ~0.37 GiB quantized. All offsets 64B-aligned. q4_block group = 20 B `[scale bf16:2][min bf16:2][nibbles:16]`, `w = scale*q + min`, ~5.0 bpw. q8_row = `[scale bf16:2][i8:K]`, `w = scale*q`. Convert ~214 s streaming; mmap load ~40 ms GPU cache init.

### Throughput model (calibrate, don't trust constants)
```
step_time ~= max( W_step / BW , F_step / (TFLOPS * MFU) ) + sync + sampler
F_step ~= 2 * 3.8e9 active * 256 tokens ~= 1.95 TFLOP
         (dense-shaped ~1.22 TFLOP + experts ~0.73 GFLOP)
```
M3 Pro forecast at MPS dense (3.35 historical / 2.22 re-bench) + grouped MoE target 0.8-1.5 TF/s: step ~0.9-1.3 s -> ~9-13 tok/s once GPU round-trips are gone. M3 Pro lands at the 8 tok/s floor with thin margin; it is the hardest target (weak GPU, irrelevant bandwidth advantage at canvas=256).

---

## 5. Resolved-bug archaeology (don't reintroduce)

### Monolithic kernel rewrite (rev1 -> rev2)
The first `diffgemma_step.metal` draft had silent-garbage bugs caught only by auditing against `qgemm.metal` / `cpu.rs`:
- **Q4 layout inverted** — draft put scale/min at group end; real layout is `[scale][min][nibbles]` at the front. Every matmul would've been garbage.
- **Q8 scale position** — assumed end-of-row; real is `[scale:2][i8:K]` front.
- **RoPE interleaved vs split-half** — draft used `(2j, 2j+1)`; Gemma uses `(d, d+rot/2)`.
- **Proportional-RoPE denominator** — must be full head_dim (512), not the 128 rotary subset.
- **Temperature index** — count-up vs the CPU's count-down; off by the whole 0.4-0.8 range on step 1.
- **Accept rule** — add-before-test vs CPU's test-before-add-with-break; different accept sets, different trajectories.
- **Missing `k_logit_rowstats`** — SC needs pre-temperature (t=1) stats separate from the tempered sampler stats; collapsing them broke SC at step >= 2.
- **Router RMS reduction** — only worked by accident of tpg=128; rewritten to explicit two-level.
- Missing `#include <metal_simdgroup>`; f32 regions addressed as half-element offsets in the arena (fixed to byte offsets).

### Nondeterminism hunt (2026-06)
- **Primary:** `MPSMatrixMultiplication` on `.dgq` Q4 dense linears changed tokens at canvas index >= 1 between runs. Fix: `use_mps_q4` toggle (`DGQ_MPS_Q4`) — on for bench, **off** for parity/deterministic.
- **NaN logits:** fresh/reused `BufferPool` MTLBuffers weren't zeroed -> stale bytes -> full-tensor NaN after lm_head -> argmax to token 262143. Fix: zero every buffer in `allocate`, zero KV at creation, `pool.clear()` at generate start.
- **GPU MoE Q4 scatter:** nondeterministic across fresh engines/repeats. Parity path uses CPU MoE.
- **GPU router top-k:** flaky readback (zeros on repeat) until typed u32 readback.
- **MoE stream input bug:** per-job path used pre-FF residual instead of `pre_ff_norm_2(stream)`.
- **Even-layer ping-pong:** final norm read the wrong hidden buffer.
- **Global canvas KV clear:** `clear_canvas_suffix` at forward start caused repeat drift; removed (canvas K/V is fully overwritten per layer anyway).
- Status: full drift surveys + `generate-parity` (steps 1/2, layers 3) pass; goldens regenerated.

### Other
- **Metal `tanh` NaN** on large gelu/softcap inputs — clamped argument to +/-15.
- **Old goldens encoded NaN logits** (odd-layer hidden bug + pool staleness) -> token 262143. Invalid; were discarded, not preserved.
- **OOM (bf16 era):** 30 persistent attention scratches + unbounded expert transpose cache + CPU||GPU peak. Fixed by two reusable layer scratches, per-layer eviction, GPU-first parity dropping CPU state.
- **`pool.trim` per layer caused 3x slowdown** — trim at section boundaries only.
- **Never chain readback on the same buffer within a batch** (Metal hazard).

---

## 6. Decisions & rationale

- **Checkpoint-orientation quant layout** (not pre-transposed): the 78 ms/miss transpose was a kernel-orientation artifact, not a layout necessity. Keeping `[out,in]` row-major gives a streaming converter, shared CPU/GPU layout, and parity indices that match the checkpoint. Row-interleaving reserved as a converter-side format-version bump iff profiling shows load-bound expert GEMMs.
- **int4 affine, not FP4/e2m1** — Metal has no native 4-bit; everything dequantizes to nibbles in-kernel anyway, and affine int4 matches weight distributions better and is better characterized.
- **Router stays f16 weights / f32 accum + f32 logits** — routing is control flow; near-boundary logit noise flips experts discretely and f16 compare breaks CPU/GPU tie-break parity. Quantizing the ~22 MB router saves nothing.
- **MoE quant sensitivity** — only 3.8B active, so q4 damage behaves like quantizing a ~4B dense model, not a 26B one. If quality drops, bump shared expert + first/last layers before routed experts (cheap; routed experts hold the bytes).
- **q4 vs q5 on M3 is a quality choice, not speed** — compute-bound at canvas=256, so q5 is ~free in wall-clock on 36 GiB.
- **Two KV layouts coexist** — monolithic b4 (half, unified per-layer region) is canonical; legacy f32 `GpuKvCache` retained for the engine path. Unify only post-ship; migration cost is high.

---

## 7. Known issues carried forward (IDs referenced in PLAN.md)

| ID | Issue | Lives in PLAN as |
|----|-------|------------------|
| STOP-1 | Early stop fires on degenerate all-pad/filler stable argmax | done (P1.1) |
| CONV-1 | ~1 accept/step; min_ent > 0.1 early, only ~5 positions H<0.1 late @ 30L q4 | P1.6 |
| PREF-1 | Monolithic encoder prefill forced CPU MoE (`use_mps_q4=false`); ~98–155 s → ~4 s after fix | done (P1.8) |
| QUAL-1 | Templated chat + default --steps 2 -> no readable reply | done (P1.2) |
| CLI-1 | `--steps` default 2 (parity) vs model-card up to 48 | P1.2 |
| CHAT-1 | Display decodes full 256 block incl. pads | P1.4 |
| SC-1 | `k_sc_softembed` O(vocab x hidden)/step | P2.3 |
| DISPATCH-1 | ~130 encoder calls/step/layer | P2.4 |
| MOE-1 | Atomic f32 scatter nondeterministic (~1 ulp) | P3.3 |
| MEM-1 | MPS Q4 weight scratch up to ~92 MiB | P3.4 |
| RNG-1 | `init_canvas_state` may not match `Rng::new(seed+1)` if canvas init shares the draw stream | section 2 / P3.2 |
| FULL-1 | 5 full-attention layers need shape-specialized pipelines | done (M0.4) |
| METAL-1 | `tanh` large-input NaN | done (clamp) |
| BENCH-1 | Historical benches used kv_len=0 vs engine 64 | reconciled in section 4 |

---

## 8. File map

| Path | Role |
|------|------|
| `shaders/diffgemma_step.metal` | All monolithic step kernels + dispatch schedule comment |
| `shaders/qgemm.metal` | **Authoritative** Q4/Q8 dequant reference |
| `shaders/gemm.metal`, `attention.metal`, `decoder.metal`, `sampler.metal`, `probe.metal` | Engine-path kernels |
| `src/metal/step_kernel.rs` | Monolithic ABI, layout builder, encode driver, CLI backends |
| `src/metal/step_generate.rs` | Monolithic generate loop + session |
| `src/metal/step_kv.rs` | b4 KV layout, prefill/extend writers |
| `src/metal/kv_cache.rs` | **Legacy** engine KV (f32) — do not use for monolithic b4 |
| `src/sample.rs` | **Authoritative** sampler semantics |
| `src/generate.rs` | Reference engine block loop |
| `src/chat_template.rs` | Gemma-4 turn formatting, HF-matched ids |
| `src/tokenizer.rs` | BPE + added_tokens for chat special tokens |
| `fixtures/generate/` | Goldens (`--raw` for legacy prompts; `monolithic` profile) |


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

---

*Consolidated 2026-06 from PLAN.md (v1), PLAN2.md, PLAN_MONOLITHIC.md. Forward-looking work lives in PLAN.md.*
