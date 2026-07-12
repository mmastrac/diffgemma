# Validation-oracle GEMM kernels

These Metal kernels are **not on any production path** — production GEMM
(dense, stacked, and block-sparse MoE, every weight format) runs entirely
through `src/shaders/gemm/gemm_tunable/`.

They are kept as **independent bit-exact oracles**: the `gemm_tunable` kernels
are cross-checked against these legacy implementations (and their CPU mirrors)
in the unit tests and `bench-gemm`. Same math contract, different, simpler
tiling — a second implementation that would diverge if a tunable edit broke the
K-accumulation chain, dequant, or rounding.

| kernel | oracle for | Rust wrappers (`src/shaders/oracle/gemm/`) |
|--------|-----------|-------------------------------------|
| `gemm_block.metal` | dense raw/q8/q4/nvfp4 | `gemm_bf16` `gemm_q8` `gemm_q4` `gemm_nvfp4` |
| `gemm_block_stacked.metal` | fused QKV / gate+up | `gemm_block_stacked` `gemm_bf16_stacked` |
| `gemm_block_grouped.metal` | grouped MoE experts | `gemm_block_grouped` |

## Sampler / LM-head oracles

The decoder-engine (`GpuDecoderEngine::forward`) sampling + LM-head kernels are
also validation-only — production sampling runs the step-kernel sampler
(`sample_rowstats`/`sample_apply`/`sample_commit`/`sc_prob_cols`/…). These live
here too: `argmax_rows`, `logit_softcapping`, `scale_logits`, `scatter_vocab_chunk`,
`row_entropy`, `sample_from_probs_rows`, `gather_prob_cols`, `softmax_rows`.
(`convert_scale` lives in the production tree — it is also dispatched in the default ≤256-token engine prefill path.)

## GEMM oracles

`gemm_block_stacked` / `gemm_block_grouped` (Rust) also export **production**
types/helpers/CPU-oracles (`GemmStackedSeg`, `StackedSegFc`, `Fixture`,
`BlobGroupedParams`, `bind_gpu_buffers`, `cpu`) that `gemm_tunable` and its tests
depend on — only the GPU kernel entry points here are oracle-only.

Retired from production 2026-07-11 (task #74/#76). To fully delete these, first
add tunable-vs-CPU oracles for the raw/q8/q4 dense + stacked paths (the nvfp4
ones already live in `gemm_tunable.rs`), then remove the oracle test modules.
