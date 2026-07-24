//! Legacy block-GEMM family kept as bit-exact validation
//! oracles for gemm_tunable (per-format fixtures + stacked/grouped variants).
pub mod fixture;
pub mod gemm_bf16;
pub mod gemm_bf16_stacked;
pub mod gemm_block;
pub mod gemm_block_grouped;
pub mod gemm_block_stacked;
pub mod gemm_nvfp4;
pub mod gemm_q4;
pub mod gemm_q8;
