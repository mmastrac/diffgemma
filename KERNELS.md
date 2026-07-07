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
| attention (`attention_mma2` / `attention_mma_full`) | ~89 @kv64, ~372 @kv1024 (was ~212/~894) | 2.4–2.8x over prior | OCCUPANCY was the wall, not barriers/tuning (2026-07-06): mma2's 16 KiB tgmem O tile → 1 tg/core → kv-scaling latencies never overlapped. Fixes: mma2 O → registers (bit-identical, 2.7x); mma_full d-SPLIT across the 2 simdgroups (oreg 128→64/lane, ~10 KiB tgmem; QK needs a cross-half partial-sum → NOT bit-identical, gate-neutral 45/51, 1.6x). Full-tile K/V staging + fewer-barrier variants were TESTED AND SLOWER (wide-stride simdgroup_load + serial MMA chain); QG=1 mma_full also slower (register-bound, halved staging lanes). Window-clipped ≥1024 |
| MoE experts (`gemm_block_sparse` + scatter) | ~315 | ~2.3 TF/s on USEFUL flops ≈ dense wall | at the wall; M-tile levers disproven (see below) |
| SC preamble (`sc_sparse_*`, SC MLP q8, rowstats) | ~165 | occupancy/gather-bound | tiling levers disproven; NOT the vocab GEMM (sparse-SC active at default on bf16-embed; rowk chunked only runs with DGQ_SC_SPARSE=0) — rowk tunable port has NO production surface (2026-07-02) |
| finish (lm_head `gemm_block` Raw, softcap, sampler) | ~150 | at flops bound | at the wall |
| router (`gemm_block` + `moe_router_topk`) | ~22 | occupancy-limited N=128 | accepted (was 69 ms serial-dot) |

## Precision policy (settled empirically — see working notes for the data)

- **Weights**: bf16 lossless (attention / dense FFN / embed), q4 group-32
  experts (memory constraint), q8 SC-MLP. No changes.
- **Activation planes**: bf16 (`arena_load/store`). The global-f16 flip
  (K_ACT_F16), the f32 residual stream (DGQ_HIDDEN_F32), and finer expert
  formats were each built, measured, DISPROVEN, and their machinery DELETED
  in the 2026-07-05 flag cleanup: none improves quality; each just re-rolls
  borderline-prompt trajectories. Do not re-litigate without new evidence.
- **f16 where values are bounded [0,1]**: `sc_probs` (fp16 + GEMM prob scale),
  attention P tiles (half). Already done; nothing else qualifies.
- **Always-bf16 buffers** (range or cross-path layout): logits (`FC29` forced),
  RouteScratch weights.
- **KV cache: f16 (2026-07-06, was bf16).** Range-checked (max|KV| = 21.9 on a
  real prompt; f16 max 65504) — f16's 10 mantissa bits beat bf16's 7
  everywhere in the live range (gate went 45/51 -> 47/51 on the flip). The
  real motive: f16 lets the MMA attention kernels `simdgroup_load` K/V tiles
  straight from device memory, deleting the whole bf16->half staging pass —
  the long-context attention enabler. All writers (qk_rope_kv,
  pack_encoder_kv, CPU pack) and readers converted together; the
  producer/consumer-mismatch rule below applies doubly here.
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
  halves the pipeline variants (the hf32 variants are gone as of 2026-07-05).
- `gather_rows` / `gather_rows_bf16` / `gather_rows_bf16_to_f32` → one shader,
  in/out dtype FCs.
- `f32_to_half` + `f32_to_half_scale` → scale FC (plain variant currently has
  no production dispatch).

**Keep separate (justified)**
- Attention family (scalar oracle / mma2 sliding / mma_full full): different
  algorithms and geometries; scalar is the causal-prefill + oracle path.
- `rms_norm_rows` vs `rms_norm_rows_tiled`: engine-vs-monolith dispatch
  patterns differ.
- `residual_half` (monolith, scalar axis) vs `vec_add_inplace`
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

**Graveyard (disproven machinery DELETED in the 2026-07-05 flag cleanup):**
DGQ_HIDDEN_F32 (f32 hidden planes + hf32 kernel variants), K_ACT_F16 (f16
arena FC), ICB record/replay (step_icb), DGQ_ACCEPT_ROW_CAP (superseded by the
no-freeze fix), DGQ_SC_PRE_TEMP, the slow + full-prob SC softembed paths
(sc_softembed.metal deleted; chunked + sparse remain), scalar-per-expert MoE
env toggle, DGQ_GEMM_N_TILE (fixed 128), DGQ_MONOLITHIC, DGQ_TIME_DISPATCH,
DGQ_TRACE_BISECT. The partial-lm trio stays (live under DGQ_FREEZE=1).
**All surviving flags live in `src/flags.rs` — the single registry.**

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
- **Weight-stationary prefill blocks** (ROADMAP E1, 2026-07-07 —
  `DGQ_MOE_PREFILL_BM=64|128` opt-in, default off): batched-prefill block
  list built at a taller height, same tunable sparse kernel at that TUNE_BM,
  one weight stream per expert instead of ~2.4. BIT-IDENTICAL (6.5k needle
  KV dump byte-equal at 32/64/128) but DISPROVEN as perf: 64 = wash, 128 =
  3.6x slower (TM=8 → 64 f32 acc/lane register spill). Roofline: at M=1024
  the expert GEMM is COMPUTE-bound (~48 ms/layer MMA at 2.3 TF/s vs ~7.6
  ms/layer W bytes incl. re-reads) — byte-cutting is a non-lever here; the
  prefill MoE lever, if any, is GEMM TF/s (fragment-tile class).

## GEMM TF/s ceiling — SETTLED 2026-07-07 (bench: `bench-gemm --shapes sparse`)

Measured at the PREFILL shapes (M=8192-slot MoE, M=1024 dense), M3 Pro:
- **dense tunable 64x64 = 3.69-3.88 TF/s = AT the MPS matmul wall
  (3.65-3.69)** — the 2026-07-02 "1.5-1.7x headroom" is CLOSED; the tunable
  port captured it.
- **sparse @ prefill route distribution (128 experts x 64 rows): 3.33-3.64
  useful = 92-96% of its dense twin.** SHIPPED from the sweep: SPARSE_BN
  64→128 (gate_up +5.6%, down +8% within-run; both N dims divide 128; same
  block list; bit-identical — KV dump byte-equal, gate 17/17).
- **sparse @ denoise distribution (x16 rows): 1.73-1.87 useful** — the old
  "~2.3 TF/s useful wall" figure was THIS regime (per-TG fixed cost +
  padding at tiny M_tile), i.e. the CLOSED tiny-M family, not a kernel
  quality gap.
- **MLX steel qmm: 3.96-4.15 at the same shapes** (f16 dense 4.29-4.45).
  The remaining ~10-15% vs MLX is their software-pipelined loader/MMA;
  porting it is a real GEMM project worth <= ~2-3% end-to-end prefill →
  NON-LEVER at current ROI. GEMM TF/s is now a closed ledger; prefill
  gains must come from non-GEMM stages or fewer FLOPs.

## GEMM headroom investigation (2026-07-02) — CLOSED by the above (history)

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
- `moe_grouped(+nvfp4)` oracle tests (8): FIXED 2026-07-05. Root cause was the
  SAME class as the gemm_q8_rowk item above — stale test dispatch, not expert
  math: `moe_scatter_weighted` was rewritten to a (d-tile, token)×256-thread
  layout (`d = tgid.x*256 + tid`) but the sub-tests still dispatched the old
  (hidden, canvas)×8 grid, so only d<8 was ever written (hence cos≈0.34 at
  weight=1). Harnesses now dispatch the production grid, and the CPU side
  mirrors the full production chain (unweighted per-slot rows →
  `moe_scatter_weighted` CPU mirror). The scatter's own sub-test had the same
  stale shape but passed by fixture luck (hidden ≤ 8) — also fixed.
- `sample::stopper_blocks_degenerate_all_pad_argmax`: FIXED 2026-07-05. The
  pad-aware degenerate gate documented on `early_stop_allowed` was never wired
  into the stoppers; now enforced in BOTH `StableConfidentStopper` (CPU) and
  `sample_commit.metal` (GPU: all-pad/filler argmax suppresses confident/
  plateau stops; inert on normal prompts — gate 17/17 verified after).

## Notes

- The old `gemm_q8` 32-tile migration flagged in earlier notes is DONE
  (`gemm_q8.rs` compiles `gemm_block` with `QuantFormat::Q8`); stale note
  corrected.
- Every non-bit-identical change must be judged by MULTI-SEED gate aggregate +
  wart census, not a single-seed smoketest (see prompts.json seed comment).
