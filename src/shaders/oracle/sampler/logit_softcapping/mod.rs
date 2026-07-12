//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
pub const ENTRY: &str = "logit_softcapping";
pub const SHADER: &str =
    shader_include::include_metal!("oracle/sampler/logit_softcapping/logit_softcapping.metal");
