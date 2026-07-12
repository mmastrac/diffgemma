//! Elementwise / data-movement kernels: converts, norms, activations,
//! gathers/scatters, vec ops.
pub mod convert_scale;
pub mod gather_rows;
pub mod gelu;
pub mod memzero_bytes;
pub mod residual_half;
pub mod rms_norm_rows;
pub mod scatter_rows_weighted;
pub mod softcap_half;
pub mod swiglu;
pub mod vec_add_inplace;
pub mod vec_fill_zero;
pub mod vec_scale_inplace;
