//! Oracle sampler kernel (engine categorical sampling). Compiled by
//! sampler_kernels.rs; no standalone wrapper.
pub const ENTRY: &str = "sample_from_probs_rows";
pub const SHADER: &str = shader_include::include_metal!(
    "oracle/sampler/sample_from_probs_rows/sample_from_probs_rows.metal"
);
