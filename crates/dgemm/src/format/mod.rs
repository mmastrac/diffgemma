//! Weight storage formats and the decode paths that read them.
//!
//! A format is a byte layout plus the codecs that turn it back into f32:
//! `layout` holds the byte arithmetic (row/matrix sizes), `bf16` and `fp4` the
//! bit-level primitives those layouts are built from, and `block`/`nvfp4` the
//! per-format decode and CPU GEMM oracle bodies.
//!
//! Stage 2 covers the decode side; the kernel bodies that consume it arrive
//! with the fused GEMM.

pub mod bf16;
pub mod block;
pub mod fp4;
pub mod layout;
pub mod nvfp4;

/// Weight storage format: the axis key the fused bodies will specialize on.
/// Every variant's decode lives in the sibling modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Raw,
    Q4,
    Q6,
    Q8,
    Nvfp4,
}
