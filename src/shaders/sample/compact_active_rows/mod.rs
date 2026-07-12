//! Metal-only subkernel: compacts active canvas rows ahead of the LM-head
//! GEMM. Dispatched from step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "compact_active_rows";
pub const SHADER: &str =
    shader_include::include_metal!("sample/compact_active_rows/compact_active_rows.metal");
