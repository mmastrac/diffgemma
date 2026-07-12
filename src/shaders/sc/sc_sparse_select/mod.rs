//! Metal-only subkernel: sparse self-conditioning column select (prefix-sum
//! compaction). Dispatched from step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "sc_sparse_select";
pub const SHADER: &str =
    shader_include::include_metal!("sc/sc_sparse_select/sc_sparse_select.metal");
