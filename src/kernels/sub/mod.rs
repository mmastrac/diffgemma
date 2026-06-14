//! Isolated subkernels: one Metal body + CPU oracle + tier-1 tests per module.

pub mod gelu_pytorch_tanh;
pub mod gelu_swiglu_gate_up;
pub mod rms_norm_rows;
pub mod rms_norm_rows_no_scale;
pub mod router_top_k_rows;
pub mod softmax_rows;
pub mod swiglu_mul;
pub mod test_util;
pub mod variant;

pub use variant::KernelVariant;
