//! DiffusionGemma quantized weights (`.dgq`).

pub mod block;
pub mod convert;
pub mod dequant;
pub mod fp4;
pub mod layout;
pub mod nvfp4;
pub mod embed_row;
pub mod spot_check;
pub mod store;

pub use convert::{quantize_model, QuantizeOptions, QuantizeSummary};
pub use layout::{QuantKind, QuantProfile};
pub use embed_row::{
    run_embed_row_dump, write_embed_row_dump, EmbedRowDump, EMBED_SCALE as DGQ_EMBED_SCALE,
};
pub use store::DgqStore;
