//! One module per logical op; each holds its CPU reference, its Metal and CUDA
//! source bodies, and its tier-1 parity tests.

pub mod fill_zero;
pub mod gather_rows;
pub mod gelu;
pub mod gemm;
pub mod rms_norm;
pub mod softmax;
pub mod vec_add;
pub mod vec_scale;
