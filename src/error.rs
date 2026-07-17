//! The crate-wide error type. Historically lived in `safetensors` (where the
//! first parser needed it); it is the de-facto error for every module, so it
//! lives here and is surfaced as `crate::Error`. `safetensors::Error` remains a
//! re-export for back-compat.

use crate::safetensors::DType;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// Model-file parse / layout / dtype problems (safetensors, `.dgq`).
    Format(&'static str),
    /// GPU / Metal failure: buffer allocation, encoder / command-buffer
    /// creation, pipeline compile, dispatch.
    Gpu(&'static str),
    /// Runtime / logic error: invalid argument, budget or limit exceeded,
    /// unsupported configuration, missing state.
    Runtime(&'static str),
    /// Token-pipeline op failure (the underlying error stringified on the
    /// pipeline thread, prefixed with the op).
    Pipeline(String),
    NotFound(String),
    DType {
        name: String,
        expected: DType,
        got: DType,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<i64>,
        got: Vec<i64>,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Pipeline(msg) => write!(f, "pipeline: {msg}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Gpu(msg) => write!(f, "gpu error: {msg}"),
            Self::Runtime(msg) => write!(f, "runtime error: {msg}"),
            Self::NotFound(name) => write!(f, "tensor not found: {name}"),
            Self::DType {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dtype mismatch for {name}: expected {}, got {}",
                    expected.as_str(),
                    got.as_str()
                )
            }
            Self::ShapeMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "shape mismatch for {name}: expected {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
