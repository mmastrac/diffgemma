//! Metal-only subkernel: sparse self-conditioning column select (prefix-sum
//! compaction). Dispatched from step_kernel.rs; no standalone wrapper/oracle.
crate::shader_kernel! {
    name = "sc_sparse_select",
    metal = "sc_sparse_select.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}
