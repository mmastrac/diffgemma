//! Legacy block-GEMM family, retired 2026-07-11: kept as bit-exact validation
//! oracles for gemm_tunable (per-format fixtures + stacked/grouped variants).
pub mod gemm_bf16;
pub mod gemm_bf16_stacked;
pub mod gemm_block_grouped;
pub mod gemm_block_stacked;
pub mod gemm_nvfp4;
pub mod gemm_q4;
pub mod gemm_q8;
