//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
pub const ENTRY: &str = "logit_softcapping";
pub const SHADER: &str = include_str!("logit_softcapping.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "logit_softcapping",
        entry: "logit_softcapping",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
