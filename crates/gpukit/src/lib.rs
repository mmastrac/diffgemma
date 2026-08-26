//! GPU compute infrastructure with no kernel or model knowledge:
//! device/queue context, shader include expansion, function-constant
//! specialization with input-derived cache labels, an on-disk pipeline
//! archive, a buffer pool, and one-shot dispatch helpers.
//!
//! The crate holds mechanism only. Anything policy-shaped — which axes a
//! kernel specializes on, cache configuration, flag parsing — belongs to the
//! caller and arrives through [`metal::ContextConfig`] and plain arguments.

mod error;
pub use error::Error;

#[cfg(target_os = "macos")]
pub mod metal;
