//! Oracle sampler kernel (engine LM-head path). Compiled by
//! sampler_kernels.rs and the ranged sampler oracle.
pub const ENTRY: &str = "scale_logits";
pub const SHADER: &str =
    shader_include::include_metal!("oracle/sampler/scale_logits/scale_logits.metal");
