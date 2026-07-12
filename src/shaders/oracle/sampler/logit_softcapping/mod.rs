//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
pub const ENTRY: &str = "logit_softcapping";
pub const SHADER: &str = include_str!("logit_softcapping.metal");
