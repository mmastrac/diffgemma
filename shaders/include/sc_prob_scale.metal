#pragma once

// Self-conditioning softembed: probs are stored bf16 then fed through fp16
// (`half`) GEMM tiles. fp16's normal range bottoms out at 2^-14, but softmax
// probs over the ~262k vocab reach ~2^-18 (denormal) for spread distributions,
// which loses precision and corrupts the soft-embedding. We scale probs up into
// fp16's normal range before the GEMM and divide the same factor back out of the
// final embed_scale on the Rust side (see SC_PROB_GEMM_SCALE in step_kernel.rs).
// 2^12 lifts a uniform prob (~2^-18) to ~2^-6 while a peaked prob (~1) stays
// well under fp16's max (65504).
constant float SC_PROB_GEMM_SCALE = 4096.0f;
