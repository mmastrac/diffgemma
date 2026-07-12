//! Engine LM-head / sampler kernels (GpuDecoderEngine::forward only — the
//! step-parity oracle path, not production generation).
pub mod argmax_rows;
pub mod gather_prob_cols;
pub mod logit_softcapping;
pub mod ranged;
pub mod row_entropy;
pub mod sample_from_probs_rows;
pub mod scale_logits;
pub mod scatter_vocab_chunk;
pub mod softmax_rows;
