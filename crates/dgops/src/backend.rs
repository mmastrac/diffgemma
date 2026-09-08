//! Which GPU backend this build can dispatch to.

/// A backend that can actually run kernels here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    #[allow(dead_code)]
    Cuda,
}

/// The backend available to this build, if any. macOS always has Metal; every
/// other target needs the cuda feature and a working driver at dispatch time.
pub fn available() -> Option<Backend> {
    if cfg!(target_os = "macos") {
        Some(Backend::Metal)
    } else if cfg!(feature = "cuda") {
        Some(Backend::Cuda)
    } else {
        None
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Metal => write!(f, "metal"),
            Backend::Cuda => write!(f, "cuda"),
        }
    }
}
