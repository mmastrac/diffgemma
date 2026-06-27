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
- **Full-attention layers have no v_proj tensor.** V is aliased from the **raw** k_proj output (`rms_norm_no_scale`, no RoPE) — same as MLX: `values = keys` before `k_norm`/`rope`, then `v_norm(values)`. The step-kernel `qk_rope_kv` must **not** mutate the `k` buffer on the K path when `v_proj==0` (V threads read raw k_proj concurrently). Bug (2026-06): K normed+rotated `k` in place before V read → wrong V on layers 5/11/17/23/29 + threadgroup race.

### MoE
- Order: rms_norm_no_scale(stream) -> `router_scale[i] * hidden^-0.5` -> linear -> softmax over 128 -> top-8 -> renormalize top-8 to sum 1 -> multiply by `per_expert_scale[expert]`.
- Tie-break: higher prob wins; equal prob -> lower expert index (CPU and GPU must match).
- **No separate shared-expert tensor** — the dense MLP runs in parallel as the always-on path.
- Expert `gate_up_proj` trailing dim **1408 = fused [gate||up], 704 each**; split at 704. True inter = 704. (Reconciles the 11.3 MiB/expert figure and the 22.8B expert-param total; a per-branch 1408 would imply 45.7B params, impossible for a 25.2B model.)
- Activation: `gelu_pytorch_tanh` (Gemma), NOT SiLU.

### Embedding / lm_head
- Embed scale `sqrt(hidden) = sqrt(2816) ~= 53.066` on gather.
- Tied lm_head (`logits = final_hidden @ embed^T`), then softcap.
- **`embed_tokens` is stored bf16 (Raw), not q8** (since the bf16-embed fix, §10): the tied lm_head and SC soft-embed both read it, and q8-per-row coarseness flattened logits on hard tail tokens → tail-convergence stall. bf16 is lossless and +0.74 GiB on one tensor. Kind is auto-detected (`embed_bf16` flag), so old q8-embed blobs still dispatch the q8 gather/lm_head/SC paths.
- **Partial lm_head (P2.5):** after positions are accepted (`frozen[]`), skip full `256×262144` GEMM on those rows; compact unfrozen rows → gather → `M_active×VOCAB` GEMM → scatter. Default on; `DGQ_PARTIAL_LM_HEAD=0` for A/B. Disables ICB replay when active (schedule differs).

### Self-conditioning (the part not guessable from public material)
- 4 tensors: pre_norm, gate_proj, up_proj, down_proj (gate/up/down q8 in `.dgq`; pre_norm raw). The SC soft-embed itself uses the (now bf16) `embed_tokens` table.
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
- **Frozen mask (P2.5):** after accept, `sample_write` sets `frozen[i]` sticky until block reset. Partial lm_head skips vocab GEMM + rowstats/apply on frozen rows; entropy forced to 0, `new_sample = ids`. Opt out: `DGQ_PARTIAL_LM_HEAD=0`.
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

### Engine vs monolithic (M3 Pro, model/q4, 2026-06)

Historical engine numbers used `/tmp/quantized-weights` and pre-P1.6 monolithic. **Current production defaults** (native Q4 step kernel + GPU grouped encoder MoE, partial lm_head on):

| Benchmark | Monolithic (current) | Engine (historical) | Notes |
|-----------|-------------------|---------------------|-------|
| Forward-only 3L | ~2.3 s/step | ~2.4 s/step | kv=0 smoke vs engine prefill @ 64 |
| 30L forward-only | ~4.8 s/step (pre-P2.5 baseline) | ~7.9 s/step | monolithic ~39% faster; 152 syncs engine-side |
| **Hello @ 30L, seed 42** | **4 denoise steps**, coherent reply | — | `accept/step` [239,252,256,256]; `confident_stop` |
| Denoise tok/s @ Hello | **~37 tok/s** (partial lm_head on) | ~3.9 tok/s | partial OFF ~28 tok/s same run class |
| Mean step @ Hello | **~1.74 s** (partial on) | ~33 s/step (early-stop @ 2, old) | P2.5: ~25% denoise vs full lm_head |
| Pipeline compile (repeat) | ~3 ms (archive cache) | — | `~/.cache/diffgemma-mps/metal-pipelines/` |
| Sampler readback | ~30 KiB/step (CanvasState) | — | P2.1 hot path; `DGQ_CHECK_LOGITS=1` opt-in full scan |

### .dgq blob facts
**Current default profile (mixed precision), ~18.9 GiB, 1047 tensors:** 60 q4_block (MoE experts only), 3 q8_row (SC gate/up/down), 984 raw — bf16 attention q/k/v/o + dense FFN gate/up/down + **bf16 embed_tokens** + norms/router/scales/layer_scalars/vision. Earlier uniform-q4 profile was 15.35 GiB (454 q4_block + q8 embed); the engine auto-detects each tensor's kind, so old blobs still load. Quant evolution: uniform q4 (15.35) → bf16 attn/FFN (16.1) → +q8→bf16 embed-attn upgrades → bf16 embed (18.9). All offsets 64B-aligned. q4_block group = 20 B `[scale bf16:2][min bf16:2][nibbles:16]`, `w = scale*q + min`, ~5.0 bpw. q8_row = `[scale bf16:2][i8:K]`, `w = scale*q`. nvfp4_block = E2M1 nibbles + FP8 E4M3 scale/16, ~4.5 bpw (supported, not in default profile — see §10). Convert ~210 s streaming; mmap load ~ms GPU cache init.

### Throughput model (calibrate, don't trust constants)
```
step_time ~= max( W_step / BW , F_step / (TFLOPS * MFU) ) + sync + sampler
F_step ~= 2 * 3.8e9 active * 256 tokens ~= 1.95 TFLOP
         (dense-shaped ~1.22 TFLOP + experts ~0.73 GFLOP)
```
M3 Pro forecast at MPS dense + grouped MoE + partial lm_head: Hello-class prompts already **~37 tok/s denoise-only** with 4-step early stop; stretch target remains **<= 1.4 s/step** mean (P2.6 / ICB / SC fast path).

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
- **GPU MoE Q4 scatter:** monolithic batched path had unreliable float atomics (2026-06, fixed — see MoE scatter entry). Engine parity path may still use CPU MoE for determinism.
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
- **Grouped MoE GEMM column index (`gemm_linear_grouped`, 2026-06):** dispatch is `n` threadgroups in X × 32 threads (one output column per threadgroup; lanes split K-groups and `simd_sum` within the simdgroup). Bug used `col = thread_position_in_grid.x`, so lanes 0..31 in a simdgroup mapped to columns 0..31 and `simd_sum` mixed partial dot products across columns. Symptom: batched `moe_out` cos ≈ 0.02–0.24 vs scalar `moe_grouped` ~0.99; tier-1 `gemm_linear_grouped` GPU tests failed the same way. Fix: `col = threadgroup_position_in_grid.x`, `global_row = threadgroup_position_in_grid.y`. Not a Q4 dequant / K-order issue — see subkernel `shaders/kernels/gemm_linear_grouped.metal`.
- **Grouped tiled MoE GEMM M-striping (`gemm_block_grouped`, 2026-06):** batched path uses `gemm_block_grouped` (simdgroup 32×32 tiles along N and K). Each threadgroup handles one expert (`tgid.y`) and one N-tile (`tgid.x`), but the kernel only loaded/stored A rows `m0+mm` for `mm ∈ [0,31]` with **no outer loop over M**. Production routing assigns up to **185 rows per expert** (L0 Calgary: expert 0 has M=96; 22 experts with M>32). Rows past 32 got zeros/garbage while CPU oracle computed them. Symptom: `step-moe-batched-pin` `gate_up_gemm` cos ≈ **0.76** (slots 32–39 cos **0.0**); `swiglu_isolated` cos **1.0** (swiglu cleared). Down GEMM uses the **same kernel** — `down` cos ≈ 0.52 was partly inherited gate_up error, partly the same M-drop (fixed together). Tier-1 fixtures missed it: `rows_per_expert` capped at **12** (`[8,12,4]`) — exercised N/K tiling only, not M. **`gemm_block` (dense) is fine:** `m0 = tgid.y * 32`, M tiled via grid height. **`gemm_linear_grouped` (scalar grouped) is fine:** one threadgroup per `global_row` (`grid.height = total_m`), no per-expert M cap. Fix: `for (m_base = 0; m_base < M; m_base += 32)` with `M_tile = min(32, M - m_base)` in `shaders/kernels/gemm_block_grouped.metal`. Fixtures bumped to **M=100** (`[100,4]` tiny, `[100,48,4]` tile) on both grouped kernels; real-weight harness `row_starts = [0,50,100]`.
- **`qk_rope_kv` V-alias on full layers (2026-06):** when `v_proj==0`, V must read **raw** k_proj and apply `rms_norm_no_scale` only (MLX: `values = keys` before `k_norm`/`rope`). GPU K path normed+rotated `k` **in place** while V threads read the same buffer (race + wrong semantics). CPU oracle had the same ordering bug (V after K in the head loop). Affects layers **5, 11, 17, 23, 29** every step — not prompt-length-specific, but can tip longer/harder prompts into repetition. Fix: K writes kvcache from thread-local head; skip in-place `k` mutation when `v_proj==0`. Tier-1 `full_attn_v_alias` fixture added (`v_proj=0`).
- **MoE weighted scatter float atomics (2026-06):** `moe_scatter_weighted` and fused `moe_grouped` used `atomic_add_f32` CAS on `device float*` — unreliable on MPS at scale (8 experts × 2816 dims). Symptom: layer-2 `moe_out` cos ~0.40 vs MLX, batched-pin scatter cos ~0.67. Fix: one threadgroup per `(tok, d)` with 8-thread reduction into `token_slot[tok][kk]` inverse map; scalar `moe_grouped` writes slot buffer then calls scatter kernel. GPU vs CPU moe oracle cos ~1.0; MLX layer hidden cos ~0.99 post-fix.

### Tile-bound dimension audit (2026-06)

Systematic check: for each kernel, find every dimension bounded by a tile constant; ask whether the real value can exceed it and whether a loop/grid tiles it.

| Kernel / dispatch | Tile bound | Real max (today) | Tiling mechanism | Status |
|-------------------|------------|------------------|------------------|--------|
| `gemm_block` | M tile 32 | arbitrary M | `grid.height = div_up(M,32)`, `m0 = tgid.y*32` | OK |
| `gemm_block_grouped` | M tile 32 in TG | **185 rows/expert** | was none; now `m_base` loop | **fixed** |
| `gemm_linear_grouped` | none on M | 2048 slots | `grid.height = total_m` | OK |
| `attention` KV loop | `acc[8]` for V accum | T = kv_len+256 ≤ 768; hd ≤ 512 | `for (t=0; t<T; t++)` streaming softmax — **KV axis bound-safe** | OK |
| `attention` head tiling | `acc[8]`, `red[8]` | hd=512, tpg=64 → **per=8 exact** | per-thread stride + simdgroup reduce | **exact-fit fragile** — `K_SHAPE_ASSERT per≤8, nsg≤8` added; not Calgary cause |
| `moe_router` / `bucket_*` | 128 experts | 128 | 128-wide TG, phase-1 loops `0..N_EXPERTS` | OK (`row_start` sum=2048 verified) |
| `dispatch_softcap` / rowstats | grid 65535 | CANVAS×VOCAB | `dispatch_1d_ranged` | OK (pattern to copy) |
| `gather_rows` / `gather_rows_bf16` | grid 65535 | MOE_SLOTS×HID/256 ≈ **22534** TG | `dispatch_1d_ranged` + `elem_base` | **ranged** |
| `moe_scatter_weighted` | TG width 8 | canvas×hidden TG | `(d, tok)` grid + per-slot reduction | **fixed** (was float atomics) |

**Smell:** vocab and logits already use `dispatch_1d_ranged`; grouped MoE M did not. Inconsistent overflow discipline caused the batched MoE bug class. Attention KV streaming was never the hazard — fixed score buffers would have been, but this kernel streams online softmax instead.

---

## 6. Decisions & rationale

- **Checkpoint-orientation quant layout** (not pre-transposed): the 78 ms/miss transpose was a kernel-orientation artifact, not a layout necessity. Keeping `[out,in]` row-major gives a streaming converter, shared CPU/GPU layout, and parity indices that match the checkpoint. Row-interleaving reserved as a converter-side format-version bump iff profiling shows load-bound expert GEMMs.
- **Mixed precision: bf16 where it's cheap, q4 only for the bulk.** Attention (q/k/v/o), dense FFN (gate/up/down), and `embed_tokens` are bf16 (Raw) — only ~2× q8 but lossless into the half GEMM tiles (source weights are bf16), beating MLX's q8 on those tensors. Only the MoE experts (the bulk) are q4. Driven by the equivalence work (§10): q8/q4 on attention and embed flattened logits and stalled convergence.
- **Experts stay ~4-bit — hard memory constraint.** bf16 experts ≈ 4× the q4 expert bytes, blowing the unified-memory budget; not acceptable without paged/swapped experts. MLX itself uses 4-bit experts and converges fine, so 4-bit is sufficient. Close any expert-quant gap memory-neutrally (better 4-bit scheme), never with resident bf16 experts.
- **int4 affine, not FP4/e2m1 for experts** — Metal has no native 4-bit; everything dequantizes to nibbles in-kernel anyway, and affine int4 matches weight distributions better and is better characterized. nvfp4 (microscaling fp4, MLX-mxfp4-compatible) is implemented but did **not** improve convergence over q4 group-32 in testing (§10).
- **Router stays f16 weights / f32 accum + f32 logits** — routing is control flow; near-boundary logit noise flips experts discretely and f16 compare breaks CPU/GPU tie-break parity. Quantizing the ~22 MB router saves nothing.
- **MoE quant sensitivity** — only 3.8B active, so q4 damage behaves like quantizing a ~4B dense model, not a 26B one. If quality drops, bump shared expert + first/last layers before routed experts (cheap; routed experts hold the bytes).
- **q4 vs q5 on M3 is a quality choice, not speed** — compute-bound at canvas=256, so q5 is ~free in wall-clock on 36 GiB.
- **Two KV layouts coexist** — monolithic b4 (half, unified per-layer region) is canonical; legacy f32 `GpuKvCache` retained for the engine path. Unify only post-ship; migration cost is high.

---

## 7. Known issues carried forward (IDs referenced in PLAN.md)

| ID | Issue | Status |
|----|-------|--------|
| STOP-1 | Early stop fires on degenerate all-pad/filler stable argmax | **done** (P1.1) |
| CONV-1 | ~1 accept/step; gibberish @ 48 steps @ 30L q4 | **done** (P1.6): MoE scatter fix + q4; Hello seed 42 → 4 steps, readable reply |
| PREF-1 | Monolithic encoder prefill forced CPU MoE; ~98–155 s | **done** (P1.8) |
| PREF-2 | Encoder prefill CPU MoE bottleneck | **done**: GPU grouped MoE default; ~1.4 s @ 22 tok / 30L |
| KV-MPS-1 | MPS encoder Q4 prefill KV ≠ native | **done** (P1.9–P1.10) |
| QUAL-1 | Templated chat + `--steps 2` → no readable reply | **done** (P1.2; production `--steps 48`) |
| CLI-1 | `--steps` default 2 vs 48 | **done** (P1.2) |
| CHAT-1 | Display decodes full 256 block incl. pads | **done** (P1.4) |
| SC-1 | `k_sc_softembed` O(vocab × hidden)/step | **done** (P2.3): chunked f32-accumulate GEMM is default; slow O(vocab) kept only as `DGQ_SC_GEMM=0` reference |
| DISPATCH-1 | ~130 encoder calls/step/layer | open (P2.4) |
| LM-PARTIAL-1 | Full 256×262k lm_head every step | **done** (P2.5): frozen mask + partial GEMM; ~25% denoise on Hello |
| MOE-1 | Float atomic scatter nondeterministic / wrong | **done** monolithic batched path (per-TG reduce); engine CPU MoE parity path unchanged |
| MEM-1 | MPS Q4 weight scratch up to ~92 MiB | open (P3.4) |
| RNG-1 | Canvas init vs `Rng::new(seed+1)` stream | open (section 2) |
| FULL-1 | 5 full-attention layers need shape-specialized pipelines | **done** (M0.4) |
| METAL-1 | `tanh` large-input NaN | **done** (clamp) |
| BENCH-1 | Historical benches used kv_len=0 vs engine 64 | reconciled (section 4) |
| RT-1 | Generate scanned 134 MiB logits/step for NaN guard | **done** (P2.1): opt-in `DGQ_CHECK_LOGITS=1` |
| TEST-1 | `gemm_q8` / `gemm_q8_rowk` standalone GPU oracle tests fail | open (P3): kernels are 32-N-tile but `dispatch_shape` uses `n_tile=128` → cols 32..127 unwritten. Production dispatch uses `div_up(n,32)` (correct); only the standalone tests are mis-dispatched. Fix by migrating to the fast 128-tile (`gemm_bf16` template) or correcting the test dispatch. |

---

## 8. File map

| Path | Role |
|------|------|
| `shaders/kernels/moe_scatter_weighted.metal` | Weighted expert → canvas scatter (per-(tok,d) TG reduce, no float atomics) |
| `shaders/kernels/compact_active_rows.metal` | Partial lm_head: list unfrozen canvas rows |
| `shaders/kernels/gather_rows_bf16.metal` | Gather bf16 arena rows for partial lm_head |
| `shaders/kernels/scatter_logits_rows.metal` | Scatter compact lm_head logits into full `[canvas,vocab]` buffer |
| `shaders/monolithic/diffgemma_step.metal` | Monolithic ABI + dispatch schedule comment |
| `shaders/kernels/gemm_linear_f32.metal` | Scalar Q4/NVFP4 f32 GEMM (`C = A @ W^T`) |
| `shaders/kernels/gemm_q8_linear_f32.metal` | Scalar Q8 f32 GEMM (`C = A @ W^T`) |
| `shaders/kernels/gemm_q8_linear_kxn_f32.metal` | Scalar Q8 f32 GEMM (`C = A @ W[K,N]`) |
| `shaders/kernels/gemm_linear_grouped.metal` | Grouped MoE block GEMM |
| `shaders/kernels/gemm_block_grouped.metal` | Tiled grouped MoE GEMM (batched path; M/N/K striping) |
| `shaders/kernels/dequant_block_matrix.metal` | Block matrix dequant (MPS fallback path) |
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
# Default on .dgq + metal: generate-monolithic
cargo run --release --features metal -- -m model/q4 -p "Hello" --seed 42

# Engine path (multi-kernel); slower but golden-locked
cargo run --release --features metal -- generate-gpu -m model/q4 -p "Hello" --seed 42 --steps 2

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
| *(no subcommand)* | On `.dgq` + metal: **`generate-monolithic`**. Otherwise `generate-gpu` or CPU `generate`. |
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

---

## 9. MLX vs `generate-monolithic` — failure modes & debug playbook

Use this when MLX (or HF bf16) traces disagree with Rust monolithic output. Most regressions we've hit before fall into one of these buckets — check the cheap mismatches first, then bisect forward vs sampler.

**Sources:** this thread ([d6390493](d6390493-6507-4641-8de8-b2607743d5ff)), monolithic bring-up ([bff0ada0 → 58497bad](58497bad-e96e-4335-8afe-c582af1b96e4)), `PLAN.md` P1.6 trace notes. Chat `0a8f1451-458f-4435-9b2e-99957a4919c4` was not present in local agent transcripts; overlap is covered by `PLAN.md` + section 5 here.

### A. Comparison setup traps (looks broken, engine is fine)

| Symptom | Likely cause | Fix / verify |
|---------|--------------|--------------|
| Traces diverge from step 0 | **Canvas RNG mismatch** — monolithic uses Rust LCG (`sample::Rng`); MLX default is `mx.random` | MLX dump: `dump_mlx_denoise_trace.py --canvas-rng rust`; Rust: same `--seed` |
| Token ids differ before denoise | **Chat template vs raw BPE** (`--raw` skips Gemma-4 turn formatting) | Both sides: templated `"Hello"` (not `--raw`) unless fixture says otherwise |
| MLX accepts ~200+ positions, mono ~1 at step 1 | **Quant format mismatch** — MLX mxfp4 vs `.dgq` q4_affine (different forward, not accept bug) | Expect divergence; compare **entropy curves** first. Documented: MLX ~0.04–1.5 nats vs `.dgq` ~0.5–3.1 nats @ step 1 @ 30L |
| Flat / nonsense logits @ 30L but OK @ 3L | **Prefill / KV path** — encoder wrote wrong KV (`KV-MPS-1`) or `kv_len=0` smoke vs real prompt | `step-kv-parity`, `step-kv-check`; monolithic prefill must use native Q4 (`encoder_use_mps_q4=false` default) |
| Different step counts in compare | **`--steps 2` parity default vs 48 production** | Match `--steps` and `--no-early-stop` on both sides for trace compares |
| `.dgq` in Python | **Format not loadable in HF/MLX** | MLX path uses `quantize_mlx.py` → `model/mlx-mxfp4`; Rust uses `/tmp/quantized-weights` |

### B. Monolithic forward bugs (garbled / flat entropy @ 30L)

See **section 5** for full archaeology. Highest-frequency when MLX/HF match each other but mono doesn't:

| ID / pattern | Symptom | Notes |
|--------------|---------|-------|
| **Q4 layout** | Silent garbage matmuls | `[scale][min][nibbles]` at group **front**; not tail |
| **Q8 scale** | Wrong embed / SC | `[scale:2][i8:K]` at row **front** |
| **VERIFY-N** | Q4 nibble parity | Even col = low nibble |
| **VERIFY-K** | MoE / grouped GEMM drift | Primary batched failures (2026-06): (1) **`col = gid.x` with `simd_sum`** in `gemm_linear_grouped` — use `threadgroup_position_in_grid.x`. (2) **M>32 per expert** in `gemm_block_grouped` — need `m_base` striping; tier-1 fixtures must use `rows_per_expert ≥ 33`. Also watch K-order in `dequant_q4_group` vs `q4_weight_at` |
| **RoPE** | Attention garbage | Split-half pairs; proportional-RoPE denominator = **full** head_dim (512) |
| **Temperature** | Wrong accept timing | Count-**down** schedule (`cur = max_steps - steps_done`) |
| **Accept rule** | Wrong frozen set | Test-before-add + break on prefix entropy (HF mutual-information bound) |
| **`k_logit_rowstats` / SC** | SC broken step ≥ 2 | SC needs **pre-temperature** rowstats; separate from sampler stats (`VERIFY-SC`) |
| **MoE stream input** | MoE uses wrong hidden | Must read **`pre_ff_norm_2(stream)`**, not pre-FF residual |
| **V aliasing** | Full-attn layers | No `v_proj`; V from k_proj + rms_norm_no_scale, no RoPE |
| **Prefill MoE** | 30L gibberish, 3L OK | Prefill forced CPU MoE / wrong Q4 path was ~98–155 s and bad KV (PREF-1, P1.8–P1.9) |

### C. Sampler / state bugs (timing OK, telemetry wrong)

| Symptom | Likely cause | Verify |
|---------|--------------|--------|
| Argmax all **262143** (vocab−1) | **NaN logits** from stale `BufferPool` / KV | Zero alloc path; `DGQ_CHECK_LOGITS=1`; golden discard |
| **`mean_ent` ~11+** (max confusion) | **In-process pollution** — `step-verify` then generate same process | Run generate in **fresh process**; step-ci golden may need regen after MoE changes |
| **`accept/step ≈ 1`**, gibberish @ 48 steps | **CONV-1 (historical)** — MoE scatter + forward; fixed P1.6 | If regression: layer bisect `compare_layer_moe.py`; check scatter cos |
| Early stop @ step 2, all `<pad>` | **STOP-1** degenerate argmax | Pad-aware stop (`MIN_EARLY_STOP_STEPS`); `--no-early-stop` for compares |
| Tokens drift run-to-run | **`DGQ_MPS_Q4=1`** MPS dense nondeterminism | Parity: `DGQ_MPS_Q4=0`; bench may use `=1` |
| MoE scatter wrong / layer cliff | Float atomics on scatter | **Fixed 2026-06** — use `moe_scatter_weighted` TG reduce; `token_slot` map |

### D. Recommended bisection (MLX vs mono)

1. **Matched setup:** same prompt (`Hello`), `--seed 42`, `--steps 8`, `--no-early-stop`, `--canvas-rng rust` on MLX, same layer count (start **3L** then **30L**).
2. **Dump traces:**
   ```bash
   # Rust (.dgq q4)
   ./target/release/diffgemma-mps -m /tmp/quantized-weights generate-monolithic \\
     -p Hello --seed 42 --steps 8 --layers 30 --no-early-stop \\
     --write-trace /tmp/mono_trace.json

   # MLX (mxfp4)
   cd python && uv run python scripts/dump_mlx_denoise_trace.py \\
     --model ../model/mlx-mxfp4 -p Hello --seed 42 --steps 8 --no-early-stop \\
     --canvas-rng rust -o /tmp/mlx_trace.json

   uv run python scripts/compare_denoise_trace.py /tmp/mono_trace.json /tmp/mlx_trace.json
   ```
3. **First diverging step:** if **`initial_canvas_ids` or step-0 entropy** differ → setup bug (A). If step 1 entropy differs but canvas matches → forward/quant (B). If entropy matches but **`accept_count` differs** → sampler (C).
4. **Layer bisect @ diverging step:** `dump_layer_hidden.py` / `compare_layer_hidden.py` (Rust dump + MLX/HF dump) — see `python/scripts/`.
5. **Rust-only gates before blaming quant:** `step-verify --layers 3`, `step-parity`, `step-kv-parity` on same weights; `step-probe` for stage max_abs.

### E. What we already know (2026-06 baseline)

- **30L q4 monolithic @ Hello, seed 42:** **4 denoise steps**, `accept/step` [239,252,256,256], text *"Hello! How can I help you today?"* — P1.6 gate met. Denoise **~7.0 s** / **~37 tok/s** with partial lm_head (P2.5); **~9.3 s** / **~28 tok/s** with `DGQ_PARTIAL_LM_HEAD=0` (same quality).
- **Early-run P1.6 telemetry (pre-scatter-fix):** 48-step runs showed min_ent eventually < 0.1 but only ~5 low-H positions late — superseded by scatter fix + 4-step `confident_stop` path above.
- **MLX mxfp4 vs `.dgq` q4 @ step 1:** argmax may match positions 0–1 then diverge; MLX accepts far more positions because entropies are much lower — **do not** treat as accept-rule regression without logit export parity.
- **NVFP4 grouped MoE:** large speedup; changes outputs — regen goldens / compare in fresh process after kernel changes.
- **Long prompts:** OOM (exit **137**) on full 48-step × 30L generate — use shorter `--max-new-tokens` or fewer steps for parity runs.

---

## 10. MLX generation equivalence — investigation (2026-06)

Goal: match MLX-4bit (`mlx-community/diffusiongemma-26B-A4B-it-4bit`) output quality. Reference chosen over vLLM (not integrated here). Tooling in `python/scripts/`: `mlx_generate.py` (ground-truth full-message gen), `dump_mlx_denoise_trace.py --canvas-ids` (per-step trace on a matched initial canvas), `compare_denoise_trace.py`, `compare_generation.py`.

**Baseline gap:** on matched-canvas "Why is the sky blue?", Rust took **31 denoise steps vs MLX-4bit 13** (mlx-mxfp4 17). Steps 1–8 matched MLX closely; divergence was entirely in the **tail (step ≥9)** — MLX entropy plunged to ~0, ours stalled/oscillated at 0.15–0.26. The hard, last-to-resolve canvas positions were systematically flatter (higher entropy) in our model.

**Root cause = embed quant, fixed.** lm_head is tied to `embed_tokens` (`logits = hidden @ embed^T`), and our embed was q8 **per-row symmetric** — measured relerr 0.0094 vs MLX's 8-bit **group-64 affine** 0.0053 (1.76× coarser; grouping is the win, asymmetry minor). Coarse embed → flat logits on hard tokens → tail stall. Storing `embed_tokens` as **bf16 (Raw, lossless)** dropped convergence to **16 steps** (commit `d61cc24`); the tail entropy now decays smoothly tracking MLX. Full chat: 59→37 steps, 137→93 s (~1.5×); the pervasive long-gen token-doubling ("the the", "pass pass", "been been") is largely gone.

**Ruled out — experts (decisively, all memory-neutral):**
- MLX-4bit (13 steps) uses 4-bit **group-64 affine** experts; ours are q4 **group-32** — *finer*, not under-resolved.
- Step-1 logits matched MLX (the quantized forward, incl. experts, is accurate; the gap was downstream/tail).
- Rebuilding with **nvfp4 experts** (fp4 group-16) gave 36 steps ≈ q4's 31, identical tail-divergence — swapping the expert quant format did nothing.

**Ruled out — chunked SC GEMM as a "bug":** the default chunked SC softembed vs the slow O(vocab) reference differs in step count (e.g. chunked 31 vs slow 22 on one prompt) but the order **reverses** on another prompt (chunked 26 vs slow 30) — it's **chaotic trajectory divergence** from tiny per-step soft-embed differences flipping entropy-boundary accept decisions (per-step parity is cos 0.999994), not a systematic precision bug. Do not "fix" chunked to match slow; both are valid. (A real precision bump was still made — SC probs fp16 not bf16, commit `0042d92` — but it does not materially change convergence.)

**Attention GEMM was the per-step bottleneck, fixed:** the bf16/q8 attention+FFN+lm_head kernel was a basic 32×32 simdgroup tile at ~185 GFLOP/s. Ported `gemm_block`'s tiling (128-N tile, double-buffered K, 16 accumulators) onto `gemm_bf16` → ~2400 GFLOP/s (commit `f1900f4`), removing the bf16-vs-q8 per-step penalty.

**Remaining (open, PLAN Q4/Q5):** bf16-embed 16 vs MLX-4bit 13 (~3 steps) + a few residual "as as" doublings — likely experts (q4 vs MLX 4-bit, but nvfp4 didn't help) or chunked-SC chaos. Per-step is also ~1.9× slower than MLX (MoE q4 + CPU↔GPU sync), separate from step count. Caveat: most matched-canvas comparisons used the correct MLX-4bit reference; one early trace used mlx-mxfp4 — prefer 4bit.


---

## 11. Block-sparse (megablocks-style) MoE GEMM (2026-06)

**Landed default-on (`DGQ_MOE_BLOCK_SPARSE`, opt-out with `=0`); ~6% per-step, bit-identical.**

Context: at canvas=256 top-8/128, ~76 experts are active/layer averaging ~33 tokens each (§3 working-set wall). The grouped GEMM (`gemm_block_grouped`) ran one threadgroup per active expert, each looping its rows in 32-row `m_base` tiles — variable-length threadgroups (load imbalance) + a partial last M-tile per expert.

Fix: pre-tile the ragged M into fixed ≤32-row blocks in `moe_bucket_fill` phase 1 (`route->block_expert[b]`, `block_row0[b]`, `num_blocks`; bounded by n_active + num_slots/32 ≤ 192, `MOE_MAX_BLOCKS=256`). New `gemm_block_sparse` kernel runs one block per threadgroup (grid `(N_tiles, num_blocks)` via indirect slots 2/3), no `m_base` loop — uniform, independently-schedulable work units. Same MMA math → bit-identical.

**Measured (M3 Pro, q4emb, "sky blue" seed 42):** grouped ~1758 ms/step vs block-sparse ~1653 ms/step (3 runs each, every on-run < every off-run). Bit-identical: exact same tokens + accept curve + 14 steps; all 11 smoketest prompts PASS with identical step counts.

**Why ~6% not the ~8–10% ceiling:** the restructure fixes *load balance / scheduling*, not the two remaining wastes — W is still re-dequantized per block, and the partial-last-tile padding is unchanged. Honest profiling (PLAN P2.7, `--profile-steps`) also showed the expert GEMM is ~17–21% of the step (not the ~35% an older note assumed), so the bit-identical ceiling here is modest.

**Ceiling-chase exhausted (2026-06) — ~6% is the practical max for this approach.** Both remaining levers were evaluated and rejected:
- *Hoist W dequant — infeasible.* Blocks are separate threadgroups (no shared dequant); an expert's full W (~7.9 MB half) won't fit threadgroup memory; materializing dequantized W to device costs ~4× the q4 bandwidth (half is 4× larger) — net loss. The 1.3–1.9× per-block W-redundancy is inherent.
- *Reduce partial-tile padding — measured net LOSS.* Padding is real (`DGQ_MOE_BLOCK_STATS`: ~50% of MMA rows at M_TILE=32, since experts average ~22 tokens < the 32-row tile). Only M_TILE=8 helps (M_TILE=16 doesn't — 22 rounds to 32 either way). Built + measured (`gemm_block_sparse_m8`, since reverted): bit-identical, padding → ~15%, **but +44% slower** (2412 vs 1677 ms/step, clean solo) — 32-thread/1-simdgroup TGs lose occupancy and W-dequant redundancy explodes to **2.8–5.3×** (~300 blocks for ~94 experts), which can't be amortized (see hoist). **When experts < the MMA tile, padding can't be cut without paying more in W-dequant + occupancy than is saved.** Next MoE-GEMM gains need a non-megablocks scheme; higher-confidence levers are ICB dispatch overhead + the ~24% pre_moe dense path.

**Guard methodology:** validated bit-identicality via the entropy-curve / step-count comparison (generate accept curve + smoketest step counts) per `STRATEGY.md` §8/§11 — a GEMM restructure that drifted would surface as changed accepts/step-count, not an obvious crash. Contrast `partial-forward` (separate branch), which the same guard *rejected* as irreducibly lossy.

---

## 12. GQA-grouped matrix-unit attention (`DGQ_ATTN_MMA`, 2026-06)

**Landed default-on; ~3–4.5% per step, non-bit-identical but quality-neutral.** `DGQ_ATTN_MMA=0` to opt out.

The scalar attention (`attention.metal`, one threadgroup per (token, q-head), streaming online softmax) was the one core GEMM not on the Apple matrix units. `attention_mma2` moves Q·K and P·V onto `simdgroup_float8x8`, exploiting GQA (16 Q / 8 KV → 2 Q heads share each K/V): one threadgroup (QG=2 simdgroups, 64 lanes) handles both Q heads of a KV group, staging K/V **once** for both. That shared staging is the win — the first attempt (`attention_mma`, 1 head/threadgroup) was *slower* than scalar (0.69–0.80×) because the bf16→half staging tax and occupancy dominated, not softmax cycles. mma2 hits **1.63× vs scalar on sliding layers (hd256)**. Full hd=512 layers keep scalar (two-head O = 32 KiB > threadgroup limit).

End-to-end: ~3% smoketest (1.70→1.65 s/step), ~4.5% on `--profile-steps` steady-state (1.55→1.48 s). Modest because attention is a small step fraction at short context; grows with multi-block context (P2.8).

**Non-bit-identical (f16 MMA vs f32 scalar): cos 0.9999 / max_tol 2e-2 parity — ~16× looser than the chunked-SC precedent (cos 0.999994).** The single precision source is the **f16 softmax-prob cast** (`ph = half(p)`) feeding P·V; Q·K is effectively lossless (bf16→f16 widens the mantissa). Despite the looser parity it is **quality-neutral**, verified two ways:
- *Multi-seed chaos symmetry:* across seeds 42/43/44 the scalar path glitches as much or more ("At Transformer is a type", "because to a phenomenon", "Transformer Transformer", channel-token leaks); step counts wobble both directions. The smoketest's "due of"/"neural learning" were chaotic samples, not deterministic MMA defects (didn't reproduce in fresh runs). The wobble is the model's existing trajectory chaos, not MMA degradation.
- *MLX bench (matched canvas, sky blue, seed 42):* Rust converges in 14 steps vs MLX-4bit 16; both diverge from MLX ~step 1–3 from the **quant gap, not attention**. MMA is **no further from MLX than scalar — marginally closer (60 vs 69 mismatches)**. Precision-tightening the f16-prob is therefore unwarranted: the attention error is below the dominant Rust-vs-MLX quant gap and doesn't register.

24 attention parity tests green; smoketest 16/16 both ways. `attention` (scalar) stays as the oracle/fallback and the full-layer path.

---

## 13. bf16 stacked QKV / gate+up GEMM (2026-06)

**Landed default-on; bit-identical; ~1.3% production (short ctx) / ~4.8% forward-only.** Fuses the per-layer QKV (q‖k‖v) and dense gate‖up projections into one stacked GEMM each on the **bf16 default path** (previously only the q4/nvfp4 path stacked; bf16 ran 3 separate q/k/v + 2 gate/up GEMMs). New `gemm_bf16_stacked.metal` = `gemm_block_stacked` minus dequant (reads bf16 [N,K] rows), reusing the FC segment table (`qkv_stacked_segments`/`gate_up_stacked_segments` are format-agnostic blob offsets) + output scatter. Wired in `encode_layer_qkv_gemm`/`encode_layer_dense_gate_up` under `attn_ffn_bf16`; gated by the existing `DGQ_FUSED_QKV`/`DGQ_FUSED_GATE_UP` (default on; `=0` opts each out).

**Win is GPU utilization, not bandwidth:** 5 small sequential GEMMs (q N=4096, k/v N=2048, gate/up N=2112) underutilize vs 2 fused (qkv N=8192, gate_up N=4224); weights are distinct so bandwidth is unchanged. Forward-only (`bench-step-kernel --profile-steps`, clean) = 1.475 vs 1.55 s/step (~4.8%); full-generate production ~1.3% (diluted by the constant preamble/SC/finish/sampler/readback overhead, which is a large fraction at short single-block context — the win grows with context). Bit-identical: same per-column dot products + tiling as the split path — verified same tokens + accept curve (scalar-attn A/B).

**Methodology note — don't trust `bench-gemm-fusion` as-is:** it reported "stacked +20%" but compares `gemm_q4_stacked` (q4, ¼ the weight bytes) vs the bf16 split — a q4-bandwidth confound. The real bf16-stacked-vs-bf16-split is ~1.3–4.8%, an order of magnitude less. Always measure the same dtype on both sides.

(Also added env-gated `DGQ_TIME_DISPATCH` in `dispatch_and_wait`: prints encode-vs-GPU split. Confirmed CPU encode is ~0.2 ms/step vs ~1600 ms GPU — so dispatch/sync overhead is negligible and the step is GPU-compute-bound; ICB replay (P2.2) would save nothing. The MLX per-step gap is real kernel compute, not overhead.)
