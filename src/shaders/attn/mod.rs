//! Attention kernels: canvas/prefill (`attention` + mma variants), engine GQA,
//! and RoPE application.
pub mod apply_rope_heads;
pub mod attention;
pub mod attention_flash;
pub mod attention_gemm;
pub mod attention_topk;
pub mod cpu;
pub mod engine_gqa_common;
pub mod gqa_attention;
#[cfg(target_os = "macos")]
pub(crate) mod harness;
pub mod qk_rope_kv;
