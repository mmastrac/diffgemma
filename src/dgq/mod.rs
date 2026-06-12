//! DiffusionGemma quantized weights (`.dgq`).

pub mod block;
pub mod convert;
pub mod dequant;
pub mod layout;
pub mod spot_check;
pub mod store;

pub use convert::{quantize_model, QuantizeOptions, QuantizeSummary};
pub use layout::{QuantKind, QuantProfile};
pub use store::DgqStore;
