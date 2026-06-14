//! Isolated subkernels: one Metal body + CPU oracle + tier-1 tests per module.

pub mod rms_norm_rows;
pub mod test_util;
pub mod variant;

pub use variant::KernelVariant;
