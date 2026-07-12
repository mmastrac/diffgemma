//! Denoise sampler pipeline (rowstats → apply → commit → write), logit
//! rowstats, and the active-row compaction/scatter subkernels (metal-only,
//! dispatched from step_kernel.rs).
pub mod compact_active_rows;
pub mod cpu;
pub mod logit_rowstats;
pub mod sample_apply;
pub mod sample_commit;
pub mod sample_rowstats;
pub mod sample_write;
pub mod scatter_logits_rows;
