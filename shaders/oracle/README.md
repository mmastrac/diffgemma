# Validation-oracle GEMM kernels

These Metal kernels are **not on any production path** — production GEMM
(dense, stacked, and block-sparse MoE, every weight format) runs entirely
through `shaders/kernels/gemm_tunable.metal`.

They are kept as **independent bit-exact oracles**: the `gemm_tunable` kernels
are cross-checked against these legacy implementations (and their CPU mirrors)
in the unit tests and `bench-gemm`. Same math contract, different, simpler
tiling — a second implementation that would diverge if a tunable edit broke the
K-accumulation chain, dequant, or rounding.

| kernel | oracle for | Rust wrappers (`src/kernels/sub/`) |
|--------|-----------|-------------------------------------|
| `gemm_block.metal` | dense raw/q8/q4/nvfp4 | `gemm_bf16` `gemm_q8` `gemm_q4` `gemm_nvfp4` |
| `gemm_block_stacked.metal` | fused QKV / gate+up | `gemm_block_stacked` `gemm_bf16_stacked` |
| `gemm_block_grouped.metal` | grouped MoE experts | `gemm_block_grouped` |

`gemm_block_stacked` / `gemm_block_grouped` (Rust) also export **production**
types/helpers/CPU-oracles (`GemmStackedSeg`, `StackedSegFc`, `Fixture`,
`BlobGroupedParams`, `bind_gpu_buffers`, `cpu`) that `gemm_tunable` and its tests
depend on — only the GPU kernel entry points here are oracle-only.

Retired from production 2026-07-11 (task #74/#76). To fully delete these, first
add tunable-vs-CPU oracles for the raw/q8/q4 dense + stacked paths (the nvfp4
ones already live in `gemm_tunable.rs`), then remove the oracle test modules.
