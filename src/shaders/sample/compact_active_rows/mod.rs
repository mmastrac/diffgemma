//! Metal-only subkernel: compacts active canvas rows ahead of the LM-head
//! GEMM. Dispatched from step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "compact_active_rows";
pub const SHADER: &str = include_str!("compact_active_rows.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "compact_active_rows",
        entry: "compact_active_rows",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
