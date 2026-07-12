//! Validation-only kernels: bit-exact references the production paths are
//! tested against (legacy GEMM family, engine LM-head/sampler). Never
//! dispatched by production generation. See README.md.
pub mod gemm;
pub mod probe;
pub mod sampler;
