//! Self-conditioning kernels: probability columns (+CPU oracle) and the
//! sparse select/gather subkernels (metal-only, dispatched from step_kernel.rs).
pub mod sc_prob_cols;
