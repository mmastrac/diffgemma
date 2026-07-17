//! CPU oracle for top-k sparse attention — the authoritative numeric
//! reference for the `attention_topk` GPU kernels.
//!
//! Mirrors `attention_gemm::cpu_causal`'s rounding (Q -> bf16, KV -> f16 if
//! `round_kv_f16`, else raw f32; output bf16-rounded) so tolerances line up
//! with the E17 oracle. The top-k path is NOT bit-identical to dense
//! attention: only the top-k highest-scoring keys per (row, head) are kept
//! and renormalized. Quality-gated, not a refactor.
//!
//! Reference algorithm (per query row `tok`, head `qh`):
//!   1. Compute scores `s[t] = <Q_tok, K_t>` for all valid `t` (causal:
//!      `t <= kv_len + tok`).
//!   2. Take the top-`k` scores by value. If fewer than `k` keys are valid
//!      (causal early rows), keep all valid keys.
//!   3. Renormalize softmax over only those `k` scores: `p_j = exp(s_j -
//!      max) / sum_j exp(s_j - max)`.
//!   4. Output `O = sum_j p_j * V[idx_j]` and bf16-round at store (matching
//!      `cpu_causal`).
//!
//! Ties are broken by index (lower index wins) for determinism — the GPU
//! kernel must match this tie-break. The GPU kernel uses a stable
//! threadgroup-level radix tournament; the oracle uses a sort.

use crate::shaders::attention::Fixture;

/// Causal top-k attention CPU reference. Returns `[canvas][n_q_heads][hd]`
/// f32 (bf16-rounded at store, matching `cpu_causal`).
pub fn topk_causal(f: &Fixture, round_kv_f16: bool, k: usize) -> Vec<f32> {
    topk_attn(f, round_kv_f16, k, true)
}

/// Bidirectional (denoise) top-k attention CPU reference: every canvas row
/// sees all `kv_len + canvas` keys. Mirrors the decode arm
/// (`DGQ_ATTN_TOPK_DECODE`), where the kernel runs with `causal=0`.
pub fn topk_denoise(f: &Fixture, round_kv_f16: bool, k: usize) -> Vec<f32> {
    topk_attn(f, round_kv_f16, k, false)
}

fn topk_attn(f: &Fixture, round_kv_f16: bool, k: usize, causal: bool) -> Vec<f32> {
    use crate::shaders::{bf16, f16};
    let hd = f.head_dim();
    let nkv = f.n_kv();
    let n_q_heads = f.n_q_heads;
    let canvas = f.canvas;
    let kv_len = f.kv_len as usize;
    let group = n_q_heads / nkv;
    let q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    let kv: Vec<f32> = if round_kv_f16 {
        f.kvcache
            .iter()
            .map(|&v| f16::f16_bits_to_f32(f16::f32_to_f16_bits(v)))
            .collect()
    } else {
        f.kvcache.clone()
    };
    let k = k.max(1);
    let mut out = vec![0.0f32; f.out_len()];
    for tok in 0..canvas {
        for qh in 0..n_q_heads {
            let kvh = qh / group;
            let q_off = (tok * n_q_heads + qh) * hd;
            let qv = &q[q_off..q_off + hd];
            // Causal: inclusive cutoff kv_len + tok. Denoise: all keys valid.
            let n_valid = if causal {
                kv_len + tok + 1
            } else {
                kv_len + canvas
            };
            // Step 1: compute all valid scores.
            let mut scores: Vec<(f32, usize)> = Vec::with_capacity(n_valid);
            for t in 0..n_valid {
                let k_off = t * nkv * hd * 2 + kvh * hd;
                let d: f32 = qv
                    .iter()
                    .zip(kv[k_off..k_off + hd].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                scores.push((d, t));
            }
            // Step 2: top-k by (score desc, index asc) — stable tie-break.
            let kk = k.min(n_valid);
            if kk == 0 {
                continue;
            }
            // partial sort: get the top-k. n_valid can be large (long context);
            // use a partial selection (nth + sort the top slice).
            scores.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            let topk = &scores[..kk];
            // Step 3: renormalize softmax over the k selected.
            let mmax = topk[0].0;
            let mut l = 0.0f32;
            let mut probs = vec![0.0f32; kk];
            for (j, &(s, _)) in topk.iter().enumerate() {
                let p = (s - mmax).exp();
                probs[j] = p;
                l += p;
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            // Step 4: weighted sum of V at the selected indices.
            let mut acc = vec![0.0f32; hd];
            for (j, &(_, t)) in topk.iter().enumerate() {
                let p = probs[j] * inv;
                let v_off = t * nkv * hd * 2 + nkv * hd + kvh * hd;
                for (a, &vv) in acc.iter_mut().zip(kv[v_off..v_off + hd].iter()) {
                    *a += p * vv;
                }
            }
            // Step 5: bf16-round at store (matches cpu_causal output rounding).
            let o_off = (tok * n_q_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = bf16::store_bf16_round_half(*a);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::attention::Fixture;

    /// Sanity: topk with k >= n_valid must equal dense causal attention
    /// (re-normalized softmax over all keys == the same answer). Verifies the
    /// oracle's math against `attention_gemm::cpu_causal` in the k=∞ limit.
    /// macOS-only: `cpu_causal` is gated to `#[cfg(all(test, target_os="macos"))]`.
    #[cfg(all(test, target_os = "macos"))]
    #[test]
    fn topk_full_recovers_dense_when_k_ge_n_valid() {
        // kv_len=20 -> t_total=24; k=1024 >> n_valid=24 should match dense causal.
        let f = Fixture {
            q: (0..4 * 8 * 16)
                .map(|i| (i as f32 * 0.07).sin() * 0.5)
                .collect(),
            kvcache: (0..24 * 8 * 16 * 2)
                .map(|i| (i as f32 * 0.003).sin() * 0.3)
                .collect(),
            layer: crate::shaders::cpu::attention::LayerAttnParams {
                head_dim: 16,
                n_kv_heads: 8,
                is_full: true,
                v_proj: 0,
                kv_region: 0,
                q_norm_off: 0,
                k_norm_off: 0,
            },
            canvas: 4,
            n_q_heads: 8,
            kv_len: 20,
        };
        let topk = topk_causal(&f, false, 1024); // k >> n_valid=24
        let dense = crate::shaders::attention_gemm::cpu_causal(&f, false);
        let max = topk
            .iter()
            .zip(dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-4,
            "topk(k=∞) should equal dense causal; max_diff={max}"
        );
    }

    /// Sanity: denoise (causal=0) topk with k >= n_valid must equal dense
    /// bidirectional softmax attention over all kv_len + canvas keys,
    /// computed inline. Pure-CPU.
    #[test]
    fn topk_denoise_recovers_dense_when_k_ge_n_valid() {
        let f = Fixture {
            q: (0..4 * 8 * 16)
                .map(|i| (i as f32 * 0.07).sin() * 0.5)
                .collect(),
            kvcache: (0..24 * 8 * 16 * 2)
                .map(|i| (i as f32 * 0.003).sin() * 0.3)
                .collect(),
            layer: crate::shaders::cpu::attention::LayerAttnParams {
                head_dim: 16,
                n_kv_heads: 8,
                is_full: true,
                v_proj: 0,
                kv_region: 0,
                q_norm_off: 0,
                k_norm_off: 0,
            },
            canvas: 4,
            n_q_heads: 8,
            kv_len: 20,
        };
        let topk = topk_denoise(&f, false, 1024); // k >> n_valid = 24
        // Dense bidirectional reference, bf16-rounded Q + output like the oracle.
        let hd = f.head_dim();
        let nkv = f.n_kv();
        let group = f.n_q_heads / nkv;
        let n_valid = f.kv_len as usize + f.canvas;
        let q = crate::shaders::bf16::bf16_slice_to_f32(
            &crate::shaders::bf16::f32_slice_to_bf16_bits(&f.q),
        );
        let mut max_diff = 0.0f32;
        for tok in 0..f.canvas {
            for qh in 0..f.n_q_heads {
                let kvh = qh / group;
                let q_off = (tok * f.n_q_heads + qh) * hd;
                let scores: Vec<f32> = (0..n_valid)
                    .map(|t| {
                        let k_off = t * nkv * hd * 2 + kvh * hd;
                        q[q_off..q_off + hd]
                            .iter()
                            .zip(f.kvcache[k_off..k_off + hd].iter())
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();
                let mmax = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let l: f32 = scores.iter().map(|s| (s - mmax).exp()).sum();
                for d in 0..hd {
                    let o: f32 = (0..n_valid)
                        .map(|t| {
                            let v_off = t * nkv * hd * 2 + nkv * hd + kvh * hd;
                            (scores[t] - mmax).exp() / l * f.kvcache[v_off + d]
                        })
                        .sum();
                    let expected = crate::shaders::bf16::store_bf16_round_half(o);
                    max_diff =
                        max_diff.max((topk[(tok * f.n_q_heads + qh) * hd + d] - expected).abs());
                }
            }
        }
        assert!(
            max_diff < 1e-4,
            "denoise topk(k=∞) should equal dense bidirectional; max_diff={max_diff}"
        );
    }

    /// Sanity: topk with k=1 picks the argmax key per (row, head) and the
    /// output equals V[argmax] (renormalized to 1.0). Cheap deterministic
    /// check on a tiny fixture. Pure-CPU (no GPU dependency).
    #[test]
    fn topk_k1_returns_argmax_value() {
        let f = Fixture {
            q: (0..2 * 4 * 8)
                .map(|i| (i as f32 * 0.11).sin() * 0.4)
                .collect(),
            kvcache: (0..8 * 4 * 8 * 2)
                .map(|i| (i as f32 * 0.005).cos() * 0.3)
                .collect(),
            layer: crate::shaders::cpu::attention::LayerAttnParams {
                head_dim: 8,
                n_kv_heads: 4,
                is_full: true,
                v_proj: 0,
                kv_region: 0,
                q_norm_off: 0,
                k_norm_off: 0,
            },
            canvas: 2,
            n_q_heads: 4,
            kv_len: 6,
        };
        let topk = topk_causal(&f, false, 1);
        // For each (tok, qh), find argmax of <Q,K> over causal-valid keys,
        // and check O == V[argmax] (bf16-rounded).
        let hd = f.head_dim();
        let nkv = f.n_kv();
        let group = f.n_q_heads / nkv;
        for tok in 0..f.canvas {
            for qh in 0..f.n_q_heads {
                let kvh = qh / group;
                let q_off = (tok * f.n_q_heads + qh) * hd;
                let qv = &f.q[q_off..q_off + hd];
                let t_stop = f.kv_len as usize + tok;
                let mut best_t = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for t in 0..=t_stop {
                    let k_off = t * nkv * hd * 2 + kvh * hd;
                    let s: f32 = qv
                        .iter()
                        .zip(f.kvcache[k_off..k_off + hd].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    if s > best_s {
                        best_s = s;
                        best_t = t;
                    }
                }
                let v_off = best_t * nkv * hd * 2 + nkv * hd + kvh * hd;
                let o_off = (tok * f.n_q_heads + qh) * hd;
                for d in 0..hd {
                    let expected =
                        crate::shaders::bf16::store_bf16_round_half(f.kvcache[v_off + d]);
                    assert!(
                        (topk[o_off + d] - expected).abs() < 1e-5,
                        "topk_k1 mismatch at tok={tok} qh={qh} d={d}: got {} expected {expected}",
                        topk[o_off + d]
                    );
                }
            }
        }
    }

    /// Distribution sanity: top-k output should be finite on a model-shaped
    /// fixture. Pure-CPU.
    #[test]
    fn topk_model_shape_finite() {
        let f = crate::shaders::attention::model_bench_fixture(512, 64, true);
        let out = topk_causal(&f, false, 32);
        assert_eq!(out.len(), f.out_len());
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    /// Robustness: top-k should handle the ragged causal tail where `n_valid <
    /// k` for early query rows without panicking or producing non-finite
    /// output. canvas=256, kv_len=4 -> row 0 sees only 5 keys, row 255 sees 260.
    #[test]
    fn topk_handles_ragged_causal_tail() {
        let f = crate::shaders::attention::model_bench_fixture(512, 4, true);
        let out = topk_causal(&f, false, 64);
        assert_eq!(out.len(), f.out_len());
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }
}
