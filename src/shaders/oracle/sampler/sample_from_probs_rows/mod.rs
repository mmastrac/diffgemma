//! Oracle sampler kernel (engine categorical sampling). Compiled by
//! sampler_kernels.rs; no standalone wrapper.
pub const ENTRY: &str = "sample_from_probs_rows";
pub const SHADER: &str = include_str!("sample_from_probs_rows.metal");
