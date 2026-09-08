//! Metal-only subkernel: sparse self-conditioning gather. Dispatched from
//! step_kernel.rs; no standalone wrapper/oracle.
crate::shader_kernel! {
    name = "sc_sparse_gather",
    metal = "sc_sparse_gather.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}
