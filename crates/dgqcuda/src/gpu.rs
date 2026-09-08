//! GPU forward pass: CUDA when the feature is on, a clear error otherwise.

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::forward;

use crate::config::{Error, ModelConfig};
use crate::forward::Scratch;
use crate::weights::Weights;

#[cfg(not(feature = "cuda"))]
pub fn forward(
    _w: &Weights,
    _cfg: &ModelConfig,
    _ids: &[u32],
    _layers: Option<usize>,
    _rows: crate::forward::LogitRows,
    _sc: &mut Scratch,
) -> Result<crate::forward::ForwardOutput, Error> {
    Err(Error::Msg(
        "built without --features cuda: no GPU forward pass".into(),
    ))
}

#[cfg(feature = "cuda")]
pub use cuda::{attn_stage, cublas_probe, hidden_after};

#[cfg(not(feature = "cuda"))]
pub fn hidden_after(
    _w: &Weights,
    _cfg: &ModelConfig,
    _ids: &[u32],
    _layers: usize,
    _stop_at: u8,
    _sc: &mut Scratch,
) -> Result<Vec<f32>, Error> {
    Err(Error::Msg("built without --features cuda".into()))
}

#[cfg(not(feature = "cuda"))]
pub fn attn_stage(
    _w: &Weights,
    _cfg: &ModelConfig,
    _ids: &[u32],
    _sc: &mut Scratch,
) -> Result<Vec<f32>, Error> {
    Err(Error::Msg("built without --features cuda".into()))
}
