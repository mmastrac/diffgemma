//! Metal-only subkernel: scatters compacted logits rows back to canvas order.
//! Dispatched from step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "scatter_logits_rows";
pub const SHADER: &str = include_str!("scatter_logits_rows.metal");
