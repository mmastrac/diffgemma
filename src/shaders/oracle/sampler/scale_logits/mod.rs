//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
pub const ENTRY: &str = "scale_logits";
pub const SHADER: &str = include_str!("scale_logits.metal");
