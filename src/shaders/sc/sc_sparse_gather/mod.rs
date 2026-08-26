//! Metal-only subkernel: sparse self-conditioning gather. Dispatched from
//! step_kernel.rs; no standalone wrapper/oracle.
pub const ENTRY: &str = "sc_sparse_gather";
pub const SHADER: &str = include_str!("sc_sparse_gather.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "sc_sparse_gather",
        entry: "sc_sparse_gather",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
