//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
crate::shader_kernel! {
    name = "logit_softcapping",
    metal = "logit_softcapping.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}
