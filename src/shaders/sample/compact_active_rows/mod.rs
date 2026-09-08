//! Metal-only subkernel: compacts active canvas rows ahead of the LM-head
//! GEMM. Dispatched from step_kernel.rs; no standalone wrapper/oracle.
crate::shader_kernel! {
    name = "compact_active_rows",
    metal = "compact_active_rows.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}
