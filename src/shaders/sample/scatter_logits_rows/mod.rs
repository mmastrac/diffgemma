//! Metal-only subkernel: scatters compacted logits rows back to canvas order.
//! Dispatched from step_kernel.rs; no standalone wrapper/oracle.
crate::shader_kernel! {
    name = "scatter_logits_rows",
    metal = "scatter_logits_rows.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}
