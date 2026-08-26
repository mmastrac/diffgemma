//! Oracle sampler kernel (engine categorical sampling). Compiled by
//! sampler_kernels.rs; no standalone wrapper.
pub const ENTRY: &str = "sample_from_probs_rows";
pub const SHADER: &str = include_str!("sample_from_probs_rows.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "sample_from_probs_rows",
        entry: "sample_from_probs_rows",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
