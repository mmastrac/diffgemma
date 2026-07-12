//! Metal-only subkernel: sparse self-conditioning gather. Dispatched from
//! step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "sc_sparse_gather";
pub const SHADER: &str = include_str!("sc_sparse_gather.metal");
