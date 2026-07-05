# Kernel audit (2026-07-02)

Full pass over the 72 Metal kernels: aggregation, genericization, perf, and
per-step precision. Verdicts below are the source of truth for "should this
kernel exist / merge / change dtype"; re-audit when a family gains members.

## Hot-path budget (steady denoise step ≈ 0.97 s since tunable GEMM, kv=25)

> **2026-07-02 REBASE: tunable GEMM default-ON** (phases 1-4, all BIT-EXACT +
> token-identical): pre_moe ~565→449 ms, MoE ~320→234, finish ~150→106.
> Step 1.22 → 0.97 s — at/under MLX's 0.94 per-step pace with fewer steps.
> The table below predates tunable; treat its ms as the LEGACY baseline and
> its "wall" verdicts as legacy-kernel walls.

## Legacy hot-path budget (pre-tunable, kv=25, all-GPU)

> **2026-07-02 CORRECTION: the GEMM "walls" below are OUR-KERNEL walls, not
> machine walls.** At our exact production shapes on this M3 Pro: MPS matmul
> 3.4–3.7 TF/s (f32!) / 3.7–4.2 (f16); MLX steel/qmm 4.0–4.4 (f16 and q4g32,
> verified CSE-defeated + per-eval). Our best: 2.4–2.9. ~1.5–1.7× kernel
> headroom across ~1.0s of the 1.22s step. See "GEMM headroom investigation".

| Bucket | ms | Rate vs roofline | Verdict |
|---|---|---|---|
| dense GEMMs (qkv / o_proj / FFN, `gemm_block*`) | ~380 | 1.8–2.3 TF/s ≈ our-kernel wall | ~1.5x headroom vs MPS/MLX (see correction above) |
| attention (`attention_mma2` / `attention_mma_full`) | ~190 | attention-typical | tuned levers disproven; window-clipped ≥1024 |
| MoE experts (`gemm_block_sparse` + scatter) | ~315 | ~2.3 TF/s on USEFUL flops ≈ dense wall | at the wall; M-tile levers disproven (see below) |
| SC preamble (`sc_sparse_*`, SC MLP q8, rowstats) | ~165 | occupancy/gather-bound | tiling levers disproven; NOT the vocab GEMM (sparse-SC active at default on bf16-embed; rowk chunked only runs with DGQ_SC_SPARSE=0) — rowk tunable port has NO production surface (2026-07-02) |
| finish (lm_head `gemm_block` Raw, softcap, sampler) | ~150 | at flops bound | at the wall |
| router (`gemm_block` + `moe_router_topk`) | ~22 | occupancy-limited N=128 | accepted (was 69 ms serial-dot) |

## Precision policy (settled empirically — see working notes for the data)

- **Weights**: bf16 lossless (attention / dense FFN / embed), q4 group-32
  experts (memory constraint), q8 SC-MLP. No changes.
- **Activation planes**: bf16 (`arena_load/store`, toggleable to f16 via
  `K_ACT_F16` for experiments only). The global-f16 flip, the f32 residual
  stream (`DGQ_HIDDEN_F32`, kept default-off), and finer expert formats were
  each built and measured: none improves quality; each just re-rolls
  borderline-prompt trajectories. Do not re-litigate without new evidence.
- **f16 where values are bounded [0,1]**: `sc_probs` (fp16 + GEMM prob scale),
  attention P tiles (half). Already done; nothing else qualifies.
- **Always-bf16 buffers** (range or cross-path layout): logits (`FC29` forced),
  KV cache (b4 layout shared with the engine), RouteScratch weights.
- **Always-f32**: moeout plane, rowstats planes, MoE grouped scratch.
- **The real precision hazard is producer/consumer dtype mismatch**, not the
  choice of dtype. Audit found exactly one (latent): `sc_probs` /
  `sc_softembed` read the always-bf16 logits via the *toggleable* loader —
  misread under `K_ACT_F16`. Fixed (always-bf16 reads, bit-identical at
  default). All other buffers verified consistent, including the previously
  bitten RouteScratch-weight path.

## Consolidation verdicts

Done already (no action): `gemm_bf16`→`gemm_block` (QUANT_RAW), moe grouped
q4/nvfp4 (QUANT_FORMAT), sc_softembed bf16 variant.

**Merge (mechanical, bit-identical, low priority)**
- `embed_gather` + `embed_gather_bf16` → one shader, quant FC (q8/raw) —
  halves the pipeline variants (now ×2 again for `K_HIDDEN_Y_F32`).
- `gather_rows` / `gather_rows_bf16` / `gather_rows_bf16_to_f32` → one shader,
  in/out dtype FCs.
- `f32_to_half` + `f32_to_half_scale` → scale FC (plain variant currently has
  no production dispatch).

**Keep separate (justified)**
- Attention family (scalar oracle / mma2 sliding / mma_full full): different
  algorithms and geometries; scalar is the causal-prefill + oracle path.
- `rms_norm_rows` vs `rms_norm_rows_tiled`: engine-vs-monolith dispatch
  patterns differ.
- `residual_half` (monolith, scalar+hidden-f32 axes) vs `vec_add_inplace`
  (engine): different paths, both production.

**Deleted (2026-07-02 cleanup after tunable landed)**
- `attention_mma` (1-head): superseded twice (mma2, mma_full); was test-only.
- `f32_to_half` plain: had no production dispatch (scale variant remains).
- `gemm_block_sq`: bench research prototype, superseded by `gemm_tunable`.

**Kept (correction: NOT dead)**
- `scatter_vocab_chunk`: the ENGINE-validation lm_head (lm_head.rs via
  sampler_kernels) dispatches it — engine-validation-only, not a placeholder.

**Deprecation candidates (need a stable tunable cycle first — do NOT delete yet)**
- Legacy `gemm_block` / `gemm_block_stacked` / `gemm_block_sparse`: still the
  DGQ_GEMM_TUNABLE=0 fallback, the non-bf16-checkpoint stacked-q4 path, the
  nvfp4 expert path, AND the bit-exactness ORACLE the tunable bench rows
  verify against. Revisit after a few stable weeks.
- Legacy adaptive-M machinery (`DGQ_MOE_TILE_ADAPT` + m16/m8 helpers in
  gemm_block_tile + classed pipelines): superseded by gemm_tunable_sparse at
  default (only live at DGQ_GEMM_TUNABLE=0). User signed it default-on
  2026-07-02; removal needs their nod.
- `gemm_block_grouped` (pre-sparse MoE): DGQ_MOE_BLOCK_SPARSE=0 fallback only.

**Env-gated graveyard (keep, documented as disproven):** partial-lm trio
(`compact_active_rows`, `scatter_logits_rows`, partial lm_head path),
`DGQ_ACCEPT_ROW_CAP`, `DGQ_HIDDEN_F32`.

## MoE tiny-M investigation (2026-07-02) — CLOSED, no lever

The old "1.3 TF/s tiny-M-bound" verdict was stale (isolated `bench-gemm` M=33
numbers, not the production step). Measured on the production step: useful MoE
flops = 24.5 GFLOP/layer at ~10.5 ms/layer = **~2.3 TF/s useful ≈ the
dense-GEMM wall** — there was never a 60-90 ms chunk to recover.

Two implementations built and measured (routing IS heavily skewed — step-1
sample: 38-71 active experts of 128, median m_e≈6, +59% padded rows at 32-row
tiles — the padding is real, it just isn't on the critical path):

- **M-tile classes** (3 pipelines, 32/16/8-row tails, class-partitioned block
  list): ~4% SLOWER. Cause isolated with an all-C32 diagnostic: the extra
  indirect dispatches cost ~0.12 ms each even at ZERO height (~0.5 ms/layer
  for 4 empty dispatches) — Metal hazard tracking serializes same-buffer
  dispatches. Machinery removed. **Rule: never split a hot GEMM into multiple
  same-encoder dispatches on this path.**
- **Adaptive-M** (`DGQ_MOE_TILE_ADAPT`, **default ON** since 2026-07-02 —
  user call; `=0` opts out): single dispatch, per-block runtime simdgroup
  remap (8/16/32 rows) — bit-identical (subkernel bitwise tests q4+nvfp4,
  E2E token-identical on/off, gate 17/17 at default). Perf: wash (adjacent
  A/Bs ±1.5%, overlapping). Removing 50-75% of the MMA work from ~1/3 of
  threadgroups moved wall time ~0% → per-TG cost is pipeline/fixed-dominated
  (see GEMM headroom investigation), not MMA-bound. Enabled anyway as a
  LATENT win: once the tunable GEMM port (task #19) makes per-TG cost
  compute-bound, the padding savings activate — carry the adaptive M-mapping
  into that port.

## GEMM headroom investigation (2026-07-02) — OPEN, the next big lever

MPS/MLX prove ~1.5–1.7× GEMM headroom at our shapes (numbers above). Root
cause hunted with a bench-only prototype (`gemm_block_sq`, wired into
`bench-gemm` as `gemm_block_sq/*` rows, correctness-checked vs gemm_block):

**Exonerated (each measured null on this machine):**
- Occupancy alone: 32x32x32 @ ~9KB tgmem (3 TGs/core vs production's 1 at
  20KB) TIES gemm_block (~2.3). But N_TILE=32 at UNCHANGED 20KB = 1.47 —
  narrow tiles need small tgmem to break even; occupancy matters, it's just
  not the MLX delta.
- bf16 A-load conversion: x_fp16 reinterpret variant = no change.
- W-dequant lane utilization (32→128 threads): no change.
- Transposed tgmem B loads (store-W-transposed variant): no change.
- f16 accumulation: MLX BlockMMA AccumType defaults to FLOAT — they
  accumulate f32 like us.

**Partial win, shippable:** 64x64x32 tile (4 simdgroups, 32x32 quadrant, 16
accs) at ~10KB via aliasing the f32 store tile over the dead load buffers:
+7% over production at MoE shapes (3.07 vs 2.88 @ 2048x704x2816), tie at
M=256, and **bit-exact vs gemm_block (max|d| = 0.0)** — the K-chain per
output is unchanged by tiling.

**TUNABLE GEMM SHIPPING (task #19, phase 1 landed 2026-07-02,
`DGQ_GEMM_TUNABLE=1` bring-up flag):** `gemm_tunable.metal` — fragment-level
(per-lane thread_elements() loads, mem_none hints, per-lane C store) +
vectorized loaders (4-wide bf16 converts, 8-nibble q4 decode), TUNE_BM/BN via
#define prepend; 64x64 won every production shape. Bench: q4 3.5-3.8 TF/s,
Raw 3.61 (production kernels: 2.2-2.9) — ALL configs BIT-EXACT vs gemm_block
per element. Phase 1 wires the plain Raw path (o_proj / dense down / router /
lm_head-bf16): pre_moe 564-567 → 524-530 ms/step, token-identical, gate
17/17. Remaining phases: stacked (qkv, dense gate/up), q8 (lm_head on q4emb
is q8-embed), rowk (SC softembed), block-sparse MoE (carry adaptive-M).

**Where the rest lives:** MLX steel's fragment-level machinery — per-lane
`vec<T,2>` register-tile loads with compile-time strides (never
`simdgroup_load` from tgmem), `simdgroup_barrier(mem_none)` scheduling hints,
software-pipelined loader/MMA overlap (BlockLoader/BlockMMA in the mlx wheel
headers, MIT). Capturing it = a real GEMM-engineering project: port
steel-style loaders+MMA tiles into our FC/dequant/bf16 framework. Payoff is
measured, not speculative: up to ~300ms/step (1.22 → ~0.9s, past MLX).
Bit-identity is plausibly preservable (steel's kk order is ascending 8-chunks
like ours; keep our dequant math + bf16 I/O rounding) — verify per element.

## Test-harness debt

- `gemm_q8_rowk` oracle tests dispatched 128-wide n-tiles against the 32×32
  kernel (cols 32–127 unwritten → the long-standing cos≈0.34 failures). Fixed:
  harness now dispatches the production 32-tile grid.
- `moe_grouped(+nvfp4)` oracle tests (8): the kernel is verified correct in
  production (`DGQ_MOE_BLOCK_SPARSE=0` generates correctly); the failures are
  test-side. Diagnosed so far: (a) the CPU oracle computed the OLD weighted
  per-token accumulate while the kernel emits unweighted per-SLOT rows
  (weighting moved to `moe_scatter_weighted`) — fixed; (b) a deeper residual
  remains: even at weight=1.0 (tiny fixture) GPU vs CPU cos is 0.34, so
  `expert_forward_q4_mirror` (CPU) has drifted from the kernel's expert math
  itself. Needs focused archaeology against moe_grouped.metal's gate/up/act/
  down sequence and scales.
- `sample::stopper_blocks_degenerate_all_pad_argmax`: pre-existing CPU test;
  undiagnosed.

## Notes

- The old `gemm_q8` 32-tile migration flagged in earlier notes is DONE
  (`gemm_q8.rs` compiles `gemm_block` with `QuantFormat::Q8`); stale note
  corrected.
- Every non-bit-identical change must be judged by MULTI-SEED gate aggregate +
  wart census, not a single-seed smoketest (see prompts.json seed comment).
