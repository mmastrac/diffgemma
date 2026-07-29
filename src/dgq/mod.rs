//! DiffusionGemma quantized weights (`.dgq`).

pub mod block;
pub mod convert;
pub mod dequant;
pub mod embed_row;
pub mod fp4;
pub mod hf_resolve;
pub mod layout;
pub mod nvfp4;
pub mod overlay;
#[cfg(test)]
pub mod spot_check;
pub mod store;

pub use convert::{QuantizeOptions, quantize_model};
pub use embed_row::{run_embed_row_dump, write_embed_row_dump};
pub use store::DgqStore;
