//! Dense / grouped / tunable GEMM kernels. `gemm.metal` holds the engine's
//! simple f32×bf16 linear entry points (compiled via include_str! in
//! src/metal/{engine,gemm}.rs).
pub mod common;
pub mod gemm_int_sparse;
pub mod gemm_linear_f32;
pub mod gemm_linear_grouped;
pub mod gemm_q8_linear_f32;
pub mod gemm_rowk;
pub mod gemm_tunable;
#[cfg(target_os = "macos")]
pub mod int_mma_probe;

/// Engine simple-linear entry points (`bf16_gemm` / `f32_bf16_linear` /
/// `f32_f32_linear` in `gemm.metal`) — compiled by src/metal/{engine,gemm}.rs.
/// System includes only, so no include_metal! expansion is needed.
pub const ENGINE_LINEAR_SHADER: &str = include_str!("gemm.metal");
