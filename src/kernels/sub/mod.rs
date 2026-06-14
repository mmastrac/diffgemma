//! Isolated subkernels: one Metal body + CPU oracle + tier-1 tests per module.

pub mod bf16;
pub mod f16;
pub mod f32_to_half;
pub mod gather_prob_cols;
pub mod gather_rows;
pub mod gelu;
pub mod gemm_common;
pub mod gemm_q4;
pub mod gpu_common;
pub mod half_scale;
pub mod half_to_f32;
pub mod manifest;
pub mod memzero_bytes;
pub mod residual_f32b;
pub mod residual_half;
pub mod rms_norm_rows;
pub mod rms_norm_rows_tiled;
pub mod router_scale_rows;
pub mod router_top_k_rows;
pub mod softmax_rows;
pub mod softcap_half;
pub mod swiglu;
pub mod test_util;
pub mod variant;
pub mod vec_add_inplace;
pub mod vec_fill_zero;
pub mod vec_mul_inplace;
pub mod vec_scale_inplace;

pub use variant::{ElemDtype, FcBool, FcUInt, KernelVariant, QuantFormat};
pub use manifest::{
    assert_no_fc_collisions, validate_shared, RmsNormRowsTiledVariant, RmsNormRowsVariant,
    SwigluSplitVariant,
};
