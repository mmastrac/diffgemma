//! CPU reference for monolithic step-kernel attention (QK-RoPE-KV + all-valid GQA).

pub const RMS_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct LayerAttnParams {
    pub head_dim: u32,
    pub n_kv_heads: u32,
    pub is_full: bool,
    pub v_proj: u64,
    pub kv_region: u64,
    pub q_norm_off: u64,
    pub k_norm_off: u64,
}

impl LayerAttnParams {
    pub fn rot_dim(&self) -> u32 {
        if self.is_full {
            self.head_dim / 4
        } else {
            self.head_dim
        }
    }

    pub fn theta(&self) -> f32 {
        if self.is_full { 1.0e6 } else { 1.0e4 }
    }
}

/// Standard split-half RoPE: rotate pairs `(d, d + rot/2)` within the first `rot` dims.
pub fn apply_split_half_rope(src: &mut [f32], rot: u32, head_dim: u32, theta: f32, pos: u32) {
    let half_rot = (rot / 2) as usize;
    let hd = head_dim as usize;
    for d in 0..half_rot {
        let inv_freq = theta.powf(-2.0 * d as f32 / head_dim as f32);
        let a = pos as f32 * inv_freq;
        let (c, s) = (a.cos(), a.sin());
        let x0 = src[d];
        let x1 = src[d + half_rot];
        src[d] = x0 * c - x1 * s;
        src[d + half_rot] = x0 * s + x1 * c;
    }
    let _ = hd;
}

/// Gemma proportional RoPE: rotate `left[i]` with `right[i]` for `i < rot/2`.
pub fn apply_proportional_rope(
    src: &mut [f32],
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    pos: u32,
) {
    let half_head = (head_dim / 2) as usize;
    let half_rot = (rotary_dim / 2) as usize;
    for d in 0..half_rot {
        let inv_freq = theta.powf(-2.0 * d as f32 / head_dim as f32);
        let a = pos as f32 * inv_freq;
        let (c, s) = (a.cos(), a.sin());
        let x0 = src[d];
        let x1 = src[half_head + d];
        src[d] = x0 * c - x1 * s;
        src[half_head + d] = x0 * s + x1 * c;
    }
}

pub fn rms_norm_head(src: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let hd = src.len();
    let ss: f32 = src.iter().map(|x| x * x).sum();
    let inv = (ss / hd as f32 + eps).sqrt().recip();
    for (i, v) in src.iter_mut().enumerate() {
        let scale = weight.map(|w| w[i]).unwrap_or(1.0);
        *v *= inv * scale;
    }
}

/// Monolithic `qk_rope_kv` for one layer (sliding or full via `layer`).
pub fn qk_rope_kv(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    kvcache: &mut [f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    layer: LayerAttnParams,
    canvas: usize,
    n_q_heads: usize,
    kv_len: u32,
) {
    let hd = layer.head_dim as usize;
    let nkv = layer.n_kv_heads as usize;
    let rot = layer.rot_dim();
    let theta = layer.theta();
    let kv_half_base = (layer.kv_region / 2) as usize;

    for tok in 0..canvas {
        let pos = kv_len + tok as u32;
        for h in 0..(n_q_heads + 2 * nkv) {
            let is_q = h < n_q_heads;
            let is_k = !is_q && h < n_q_heads + nkv;
            let hh = if is_q { h } else { (h - n_q_heads) % nkv };

            let head = if is_q {
                let off = (tok * n_q_heads + hh) * hd;
                &mut q[off..off + hd]
            } else if is_k {
                let off = (tok * nkv + hh) * hd;
                &mut k[off..off + hd]
            } else {
                let src = if layer.v_proj != 0 {
                    let off = (tok * nkv + hh) * hd;
                    &mut v[off..off + hd]
                } else {
                    let off = (tok * nkv + hh) * hd;
                    &mut k[off..off + hd]
                };
                let mut tmp = src.to_vec();
                rms_norm_head(&mut tmp, None, RMS_EPS);
                let dst_off = kv_half_base + (pos as usize * nkv * hd * 2) + nkv * hd + hh * hd;
                kvcache[dst_off..dst_off + hd].copy_from_slice(&tmp);
                continue;
            };

            let mut tmp = head.to_vec();
            let w = if is_q { q_norm_w } else { k_norm_w };
            rms_norm_head(&mut tmp, Some(w), RMS_EPS);
            if layer.is_full {
                apply_proportional_rope(&mut tmp, rot, layer.head_dim, theta, pos);
            } else {
                apply_split_half_rope(&mut tmp, rot, layer.head_dim, theta, pos);
            }

            if is_k {
                let dst_off = kv_half_base + (pos as usize * nkv * hd * 2) + hh * hd;
                kvcache[dst_off..dst_off + hd].copy_from_slice(&tmp);
                // Full-attn: V aliases raw k_proj — leave `k` untouched for the V pass.
                if layer.v_proj != 0 {
                    head.copy_from_slice(&tmp);
                }
            } else {
                head.copy_from_slice(&tmp);
            }
        }
    }
}

/// Monolithic `attention`: all-valid mask over T = kv_len + canvas positions.
pub fn attention(
    out: &mut [f32],
    q: &[f32],
    kvcache: &[f32],
    layer: LayerAttnParams,
    canvas: usize,
    n_q_heads: usize,
    kv_len: u32,
) {
    let hd = layer.head_dim as usize;
    let nkv = layer.n_kv_heads as usize;
    let n_groups = n_q_heads / nkv;
    let t_total = kv_len as usize + canvas;
    let kv_half_base = (layer.kv_region / 2) as usize;

    for tok in 0..canvas {
        for qh in 0..n_q_heads {
            let kvh = qh / n_groups;
            let q_off = (tok * n_q_heads + qh) * hd;
            let qv = &q[q_off..q_off + hd];
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; hd];

            for t in 0..t_total {
                let k_off = kv_half_base + t * nkv * hd * 2 + kvh * hd;
                let kk = &kvcache[k_off..k_off + hd];
                let d: f32 = qv.iter().zip(kk.iter()).map(|(a, b)| a * b).sum();
                let mn = m.max(d);
                let corr = (m - mn).exp();
                let p = (d - mn).exp();
                for a in acc.iter_mut() {
                    *a *= corr;
                }
                l = l * corr + p;
                m = mn;
                let v_off = kv_half_base + t * nkv * hd * 2 + nkv * hd + kvh * hd;
                let vv = &kvcache[v_off..v_off + hd];
                for (a, &vv_i) in acc.iter_mut().zip(vv.iter()) {
                    *a += p * vv_i;
                }
            }

            let o_off = (tok * n_q_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = a / l;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_split_half_identity_at_zero() {
        let mut v = vec![1.0, 0.0, 2.0, 0.0];
        apply_split_half_rope(&mut v, 4, 4, 1.0e4, 0);
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn proportional_rope_pairs_across_head_halves() {
        let hd = 8usize;
        let mut v = (0..hd).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
        let before = v.clone();
        apply_proportional_rope(&mut v, 4, hd as u32, 1.0e6, 1);
        assert!((v[0] - before[0]).abs() > 1e-6 || (v[4] - before[4]).abs() > 1e-6);
        assert!((v[2] - before[2]).abs() < 1e-6);
        assert!((v[6] - before[6]).abs() < 1e-6);
    }
}
