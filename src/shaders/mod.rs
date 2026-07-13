//! Unified kernel tree: `src/shaders/<group>/<kernel>/` colocates each
//! kernel's Rust wrapper (`mod.rs`), Metal source (`<kernel>.metal`), and CPU
//! oracle (`cpu.rs`; room for future CUDA equivalents beside them).
//!
//! - `include/` — shared .metal headers (every `#include "x.metal"` resolves
//!   here, regardless of the including file's directory).
//! - `common/`  — shared Rust test/dispatch helpers (no Metal).
//! - `cpu/`     — shared CPU reference ops (per-kernel oracles live beside
//!   their kernels; `cpu/mod.rs` re-exports them under the old names).
//! - `oracle/`  — validation-only kernels (bit-exact references for the
//!   production tunable/step paths; not dispatched in production).
//!
//! The flat re-exports below keep every kernel module addressable as
//! `shaders::<name>` (the historical flat namespace) so call sites stay
//! group-agnostic.

pub mod attn;
pub mod common;
pub mod cpu;
pub mod elementwise;
pub mod embed;
pub mod gemm;
pub mod kv;
pub mod moe;
pub mod oracle;
pub mod sample;
pub mod sc;

#[allow(unused_imports)]
pub use attn::{apply_rope_heads, attention, engine_gqa_common, gqa_attention, qk_rope_kv};
#[allow(unused_imports)]
pub use common::expand;
#[allow(unused_imports)]
pub use common::{bf16, dbg_kernel, f16, gpu_common, manifest, test_util, variant};
#[allow(unused_imports)]
pub use elementwise::rms_norm_rows::tiled as rms_norm_rows_tiled;
#[allow(unused_imports)]
pub use elementwise::{
    convert_scale, gather_rows, gelu, memzero_bytes, residual_half, rms_norm_rows,
    scatter_rows_weighted, softcap_half, swiglu, vec_add_inplace, vec_fill_zero, vec_scale_inplace,
};
#[allow(unused_imports)]
pub use embed::embed_gather;
#[allow(unused_imports)]
pub use gemm::common as gemm_common;
#[allow(unused_imports)]
pub use gemm::{gemm_linear_f32, gemm_linear_grouped, gemm_q8_linear_f32, gemm_rowk, gemm_tunable};
#[allow(unused_imports)]
pub use kv::{hadamard, kv_f32_side_hydrate, kv_quant, pack_encoder_kv, unpack_encoder_kv};
#[allow(unused_imports)]
pub use moe::batched_pin as moe_batched_pin;
#[allow(unused_imports)]
pub use moe::moe_grouped::nvfp4 as moe_grouped_nvfp4;
#[allow(unused_imports)]
pub use moe::{
    moe_bucket_count, moe_bucket_fill, moe_grouped, moe_router, moe_scatter_weighted,
    router_scale_rows, router_top_k_rows,
};
#[allow(unused_imports)]
pub use oracle::gemm::{
    gemm_bf16, gemm_bf16_stacked, gemm_block_grouped, gemm_block_stacked, gemm_nvfp4, gemm_q4,
    gemm_q8,
};
#[allow(unused_imports)]
pub use oracle::sampler::ranged as sampler_ranged;
#[allow(unused_imports)]
pub use oracle::sampler::{
    argmax_rows, gather_prob_cols, row_entropy, scatter_vocab_chunk, softmax_rows,
};
#[allow(unused_imports)]
pub use sample::{compact_active_rows, scatter_logits_rows};
#[allow(unused_imports)]
pub use sample::{logit_rowstats, sample_apply, sample_commit, sample_rowstats, sample_write};
#[allow(unused_imports)]
pub use sc::sc_prob_cols;
#[allow(unused_imports)]
pub use sc::sc_prob_cols::cpu as sc_probs;
#[allow(unused_imports)]
pub use sc::{sc_sparse_gather, sc_sparse_select};

#[allow(unused_imports)]
pub use manifest::SwigluSplitVariant;
#[allow(unused_imports)]
pub use variant::{KernelVariant, QuantFormat};
