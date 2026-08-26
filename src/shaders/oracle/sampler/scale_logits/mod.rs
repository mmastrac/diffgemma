//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
pub const ENTRY: &str = "scale_logits";
pub const SHADER: &str = include_str!("scale_logits.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "scale_logits",
        entry: "scale_logits",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
