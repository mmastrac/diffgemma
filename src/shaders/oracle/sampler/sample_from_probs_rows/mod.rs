//! Oracle sampler kernel (engine categorical sampling). Compiled by
//! sampler_kernels.rs; no standalone wrapper.
crate::shader_kernel! {
    name = "sample_from_probs_rows",
    metal = "sample_from_probs_rows.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}
