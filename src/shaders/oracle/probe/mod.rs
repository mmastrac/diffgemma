//! Device bandwidth probe kernel (diagnostics; src/metal/probe.rs).
pub const MEMCPY_ENTRY: &str = "memcpy_f32";
pub const SHADER: &str = include_str!("probe.metal");
