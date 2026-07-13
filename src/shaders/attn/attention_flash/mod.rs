//! E18: fused flash prefill attention (online softmax, register-resident O, no
//! device S/P traffic) — the steel-SDPA dataflow, in contrast to E17's
//! `attention_gemm` decomposition which materializes S=Q·Kᵀ and P to device.
//!
//! The Metal kernel (`attention_flash.metal`) streams the keys in BK-wide blocks
//! and keeps the O accumulator resident across the stream, combining each block
//! into the running (max, denom, O) with the standard flash rescale. This module
//! currently holds the **CPU block-streaming reference** used to pin the exact
//! blocking/masking/GQA scheme the kernel mirrors, validated against the E17
//! `attention_gemm::cpu_causal` batch reference (both compute the same softmax;
//! flash just blocks it). GPU pipeline + oracle land with the kernel.

/// CPU block-streaming flash reference. Mirrors the planned kernel exactly: per
/// query row, stream keys in `bk`-wide blocks, online-softmax combine each block
/// into (m, l, acc), divide at the end. Output layout matches `cpu_causal`
/// (`[canvas][n_q_heads][hd]`, bf16-rounded at store) so the two are directly
/// comparable. `round_kv_f16` rounds the KV to f16 (main-cache path) vs raw f32
/// (side-ring path) — same knob as `cpu_causal`.
pub fn cpu_flash_blocked(
    f: &crate::shaders::attention::Fixture,
    bk: usize,
    round_kv_f16: bool,
) -> Vec<f32> {
    use crate::shaders::bf16;
    use crate::shaders::f16;
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
    let mut out = vec![0.0f32; f.out_len()];
    for tok in 0..canvas {
        for qh in 0..n_q_heads {
            let kvh = qh / group;
            let q_off = (tok * n_q_heads + qh) * hd;
            let qv = &q[q_off..q_off + hd];
            let t_stop = kv_len + tok; // inclusive causal cutoff
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; hd];
            // Stream keys in BK-wide blocks (the kernel's inner loop).
            let mut t0 = 0usize;
            while t0 <= t_stop {
                let t1 = (t0 + bk).min(t_stop + 1);
                // Block scores, then block max over valid columns.
                let mut m_block = f32::NEG_INFINITY;
                let mut sblock = Vec::with_capacity(t1 - t0);
                for t in t0..t1 {
                    let k_off = t * nkv * hd * 2 + kvh * hd;
                    let d: f32 = qv
                        .iter()
                        .zip(kv[k_off..k_off + hd].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    m_block = m_block.max(d);
                    sblock.push(d);
                }
                // Online combine this block into (m, l, acc).
                let m_new = m.max(m_block);
                let corr = (m - m_new).exp();
                for a in acc.iter_mut() {
                    *a *= corr;
                }
                l *= corr;
                for (j, t) in (t0..t1).enumerate() {
                    let p = (sblock[j] - m_new).exp();
                    let v_off = t * nkv * hd * 2 + nkv * hd + kvh * hd;
                    for (a, &vv) in acc.iter_mut().zip(kv[v_off..v_off + hd].iter()) {
                        *a += p * vv;
                    }
                    l += p;
                }
                m = m_new;
                t0 = t1;
            }
            let o_off = (tok * n_q_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = bf16::store_bf16_round_half(a / l);
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;
