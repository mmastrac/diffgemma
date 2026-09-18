//! The pack reader's own error type. The engine converts it into
//! `crate::Error`. A backend crate with no engine dependency uses it as is.

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// Manifest parse, layout or dtype problem.
    Format(&'static str),
    /// Invalid argument or an offset the host cannot address.
    Runtime(&'static str),
    /// Pack integrity failure the manifest itself can prove: a blob file
    /// shorter than the tensors it describes, or one still carrying the
    /// download sentinel. Always names the shortfall and the remedy.
    Pack(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Runtime(msg) => write!(f, "runtime error: {msg}"),
            Self::Pack(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
