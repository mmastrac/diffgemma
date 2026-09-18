//! Reader for the `.dgq` weight pack: a JSON manifest plus one blob file.
//!
//! Lives below both backends so the engine and the CUDA side share one
//! definition of the format and one set of load-time gates. Writing packs
//! stays in the engine, which owns quantization policy.

pub mod error;
pub mod manifest;
pub mod pack;

pub use error::Error;
pub use manifest::*;
pub use pack::PackFile;
