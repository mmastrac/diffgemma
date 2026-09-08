//! Shared test helpers for the per-op oracle checks.

use crate::backend;

/// True when a GPU dispatch should actually run. On macOS a hosted CI runner
/// has no usable Metal device; a CUDA host is expected to have one.
pub fn gpu_available() -> bool {
    backend::available().is_some()
        && !(cfg!(target_os = "macos") && std::env::var_os("CI").is_some())
}

pub use gpukit::testing::{assert_oracle, cosine_f32, max_abs_diff};
