//! One module per logical op; each holds its CPU reference, its Metal and CUDA
//! source bodies, and its tier-1 parity tests.

pub mod adamw;
pub mod fill_zero;
pub mod gather_rows;
pub mod gelu;
pub mod gelu_backward;
pub mod gemm;
pub mod rms_norm;
pub mod rms_norm_backward;
pub mod scatter_add_rows;
pub mod softmax;
pub mod softmax_backward;
pub mod vec_add;
pub mod vec_scale;
