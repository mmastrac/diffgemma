# Kernel audit (2026-07-02)

Full pass over the 72 Metal kernels: aggregation, genericization, perf, and
per-step precision. Verdicts below are the source of truth for "should this
kernel exist / merge / change dtype"; re-audit when a family gains members.

## Hot-path budget (steady denoise step ≈ 1.22 s, kv=25, all-GPU)

| Bucket | ms | Rate vs roofline | Verdict |
|---|---|---|---|
| dense GEMMs (qkv / o_proj / FFN, `gemm_block*`) | ~380 | 1.8–2.3 TF/s ≈ peak | at the wall |
| attention (`attention_mma2` / `attention_mma_full`) | ~190 | attention-typical | tuned levers disproven; window-clipped ≥1024 |
| MoE experts (`gemm_block_sparse` + scatter) | ~315 | 1.3 TF/s, tiny-M-bound | ONLY remaining perf chunk (see PLAN / task queue) |
| SC preamble (`sc_sparse_*`, SC MLP q8, rowstats) | ~165 | occupancy-bound | tiling levers disproven |
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

**Dead / test-only (candidates for deletion)**
- `attention_mma` (1-head): superseded twice (mma2, mma_full); test-only.
- `f32_to_half` plain: no production dispatch (scale variant is used).
- `scatter_vocab_chunk`: validation placeholder.
- Env-gated graveyard (keep, documented as disproven): partial-lm trio
  (`compact_active_rows`, `scatter_logits_rows`, partial lm_head path),
  `DGQ_ACCEPT_ROW_CAP`, `DGQ_HIDDEN_F32`.

## Test-harness debt

- `gemm_q8_rowk` oracle tests dispatched 128-wide n-tiles against the 32×32
  kernel (cols 32–127 unwritten → the long-standing cos≈0.34 failures). Fixed:
  harness now dispatches the production 32-tile grid.
- `moe_grouped(+nvfp4)` fixtures went stale against the compact-dispatch
  RouteScratch changes (kernel verified correct in production); fix in flight.
- `sample::stopper_blocks_degenerate_all_pad_argmax`: pre-existing; fix in
  flight.

## Notes

- The old `gemm_q8` 32-tile migration flagged in earlier notes is DONE
  (`gemm_q8.rs` compiles `gemm_block` with `QuantFormat::Q8`); stale note
  corrected.
- Every non-bit-identical change must be judged by MULTI-SEED gate aggregate +
  wart census, not a single-seed smoketest (see prompts.json seed comment).
