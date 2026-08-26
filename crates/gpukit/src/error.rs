/// Errors from device setup, shader compilation, and dispatch.
#[derive(Debug)]
pub enum Error {
    /// Device, queue, buffer, or encoder creation failed.
    Gpu(&'static str),
    /// Shader or pipeline compilation failed; carries the compiler message.
    Compile(String),
    /// Pipeline archive open or persistence failed.
    Cache(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(msg) => write!(f, "gpu error: {msg}"),
            Self::Compile(msg) => write!(f, "shader compile failed: {msg}"),
            Self::Cache(msg) => write!(f, "pipeline cache: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
