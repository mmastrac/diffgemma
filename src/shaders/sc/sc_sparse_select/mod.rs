//! Metal-only subkernel: sparse self-conditioning column select (prefix-sum
//! compaction). Dispatched from step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "sc_sparse_select";
pub const SHADER: &str = include_str!("sc_sparse_select.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "sc_sparse_select",
        entry: "sc_sparse_select",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
