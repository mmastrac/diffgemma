use crate::buffer::Buffer;
use crate::config::{LayerType, TextConfig};
use crate::fast_slice::{FastBf16Slice, bf16_to_f32_into};
use crate::tensor::Bf16Slice;

pub mod activations;
pub mod dequant;
pub mod dequant_block_matrix;
pub mod matmul;

// Per-kernel CPU oracles live beside their kernels (…/<kernel>/cpu.rs);
// these aliases preserve the historical `cpu::<name>` namespace.
pub use crate::shaders::attn::cpu as attention;
pub use crate::shaders::gemm::gemm_linear_f32::cpu as gemm_linear_f32;
pub use crate::shaders::gemm::gemm_linear_grouped::cpu as gemm_linear_grouped;
pub use crate::shaders::moe::moe_grouped::cpu as moe_grouped;
pub use crate::shaders::moe::moe_router::cpu as moe_router;
pub use crate::shaders::moe::moe_scatter_weighted::cpu as moe_scatter_weighted;
pub use crate::shaders::sample::cpu as sampler;

const GELU_TANH_COEF: f32 = 0.797_884_6; // sqrt(2/pi)

/// RMSNorm without learnable scale (`router` gate input, `v_norm`).
pub fn rms_norm_no_scale(out: &mut [f32], x: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    let hidden = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    for i in 0..hidden {
        out[i] = x[i] * rms_inv;
    }
}

/// Gemma RMSNorm: `out = x / rms(x) * weight`, `rms = sqrt(mean(x^2) + eps)` —
/// the unscaled normalize (shared with `rms_norm_no_scale`) then per-lane weight.
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(weight.len(), x.len());
    rms_norm_no_scale(out, x, eps);
    for i in 0..out.len() {
        out[i] *= weight[i];
    }
}

/// Per-row RMSNorm over `[seq_len, hidden]` row-major buffers.
pub fn rms_norm_rows(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) {
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(x.len(), seq_len * hidden);
    assert_eq!(weight.len(), hidden);
    for s in 0..seq_len {
        let off = s * hidden;
        rms_norm(
            &mut out[off..off + hidden],
            &x[off..off + hidden],
            weight,
            eps,
        );
    }
}

/// Per-row RMSNorm without learnable scale over `[seq_len, hidden]`.
pub fn rms_norm_rows_no_scale(out: &mut [f32], x: &[f32], seq_len: usize, hidden: usize, eps: f32) {
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(x.len(), seq_len * hidden);
    for s in 0..seq_len {
        let off = s * hidden;
        rms_norm_no_scale(&mut out[off..off + hidden], &x[off..off + hidden], eps);
    }
}

/// `y[seq, out] = x[seq, in] @ W[out, in]^T + b` (PyTorch linear weight layout).
pub fn linear(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) {
    assert_eq!(x.len(), seq_len * in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(y.len(), seq_len * out_dim);
    if let Some(bias) = b {
        assert_eq!(bias.len(), out_dim);
    }

    matmul::matmul_b_transpose(y, x, w, seq_len, in_dim, out_dim, 1.0, 0.0);
    if let Some(bias) = b {
        for s in 0..seq_len {
            let row = &mut y[s * out_dim..(s + 1) * out_dim];
            for (o, bv) in row.iter_mut().zip(bias.iter()) {
                *o += *bv;
            }
        }
    }
}

/// Linear with bf16 weights (dequantized on the fly, no weight copy).
pub fn linear_bf16(
    y: &mut [f32],
    x: &[f32],
    w: Bf16Slice<'_>,
    b: Option<&[f32]>,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
    w_scratch: &mut Buffer<f32>,
) {
    assert_eq!(w.len(), out_dim * in_dim);
    w_scratch.ensure_len(out_dim * in_dim);
    bf16_to_f32_into(FastBf16Slice::from_bf16(w), w_scratch.as_fast_slice_mut());
    linear(y, x, w_scratch.as_slice(), b, seq_len, in_dim, out_dim);
}

/// `y = x @ W[out, in]^T` reading bf16 weights starting at `w_elem_offset`.
pub fn linear_bf16_slice(
    y: &mut [f32],
    x: &[f32],
    w: Bf16Slice<'_>,
    w_elem_offset: usize,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) {
    assert_eq!(x.len(), seq_len * in_dim);
    assert_eq!(y.len(), seq_len * out_dim);
    let wfast = FastBf16Slice::from_bf16(w);
    for s in 0..seq_len {
        let x_row = &x[s * in_dim..(s + 1) * in_dim];
        let y_row = &mut y[s * out_dim..(s + 1) * out_dim];
        for (o, y_o) in y_row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let row_off = w_elem_offset + o * in_dim;
            for (i, &xi) in x_row.iter().enumerate() {
                sum += xi * unsafe { wfast.to_f32_unchecked(row_off + i) };
            }
            *y_o = sum;
        }
    }
}

pub fn silu(x: &mut [f32]) {
    for v in x {
        *v = silu_f32(*v);
    }
}

pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `gelu_pytorch_tanh` used by Gemma / DiffusionGemma.
pub fn gelu_pytorch_tanh(x: &mut [f32]) {
    for v in x {
        *v = gelu_pytorch_tanh_f32(*v);
    }
}

pub fn gelu_pytorch_tanh_f32(x: f32) -> f32 {
    let x3 = x * x * x;
    0.5 * x * (1.0 + (GELU_TANH_COEF * (x + 0.044_715 * x3)).tanh())
}

/// Stable softmax over row-major `[rows, cols]`.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(x.len(), rows * cols);
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RopeKind {
    /// Sliding-window layers: rotate full `head_dim`, standard inverse frequencies.
    Default { head_dim: usize, theta: f32 },
    /// Full-attention layers: proportional partial RoPE on `rotary_dim` of `full_head_dim`.
    Proportional {
        rotary_dim: usize,
        full_head_dim: usize,
        theta: f32,
    },
}

/// Write interleaved `[cos, sin]` pairs per frequency into `freqs` (length `rotary_dim`).
pub fn compute_rope_freqs(freqs: &mut [f32], positions: &[i64], kind: RopeKind) {
    let (rotary_dim, full_dim, theta) = match kind {
        RopeKind::Default { head_dim, theta } => (head_dim, head_dim, theta),
        RopeKind::Proportional {
            rotary_dim,
            full_head_dim,
            theta,
        } => (rotary_dim, full_head_dim, theta),
    };
    assert_eq!(freqs.len(), positions.len() * rotary_dim);
    let half = rotary_dim / 2;
    for (si, &pos) in positions.iter().enumerate() {
        let p = pos as f32;
        let base = si * rotary_dim;
        for d in 0..half {
            let exponent = (2 * d) as f32 / full_dim as f32;
            let freq = 1.0 / theta.powf(exponent);
            let angle = p * freq;
            freqs[base + 2 * d] = angle.cos();
            freqs[base + 2 * d + 1] = angle.sin();
        }
    }
}

/// Apply RoPE in-place to `vec` (length `head_dim`) using interleaved cos/sin freqs.
/// When `rotary_dim < head_dim` (proportional partial RoPE), pairs `vec[d]` with
/// `vec[head_dim/2 + d]` for `d < rotary_dim/2`; dims outside those pairs are unchanged.
pub fn apply_rope(vec: &mut [f32], freqs: &[f32], rotary_dim: usize) {
    assert!(rotary_dim <= vec.len());
    assert_eq!(freqs.len(), rotary_dim);
    let head_dim = vec.len();
    let half = rotary_dim / 2;
    let half_head = head_dim / 2;
    let proportional = rotary_dim < head_dim;
    for d in 0..half {
        let cos = freqs[2 * d];
        let sin = freqs[2 * d + 1];
        let i1 = if proportional {
            half_head + d
        } else {
            d + half
        };
        let x0 = vec[d];
        let x1 = vec[i1];
        vec[d] = x0 * cos - x1 * sin;
        vec[i1] = x0 * sin + x1 * cos;
    }
}

/// Apply RoPE to `[seq, num_heads, head_dim]` tensor (row-major, seq outermost).
pub fn apply_rope_tensor(
    x: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    freqs: &[f32],
    rotary_dim: usize,
) {
    assert_eq!(x.len(), seq_len * num_heads * head_dim);
    assert_eq!(freqs.len(), seq_len * rotary_dim);
    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = (s * num_heads + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut x[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
    }
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn rope_kind_for_layer(cfg: &TextConfig, layer: usize) -> Option<RopeKind> {
    let layer_type = cfg.layer_types.get(layer)?;
    let rotary_dim = cfg.rotary_dim_for_layer(layer)?;
    let theta = cfg.rope_theta_for_layer(layer)?;
    Some(match layer_type {
        LayerType::SlidingAttention => RopeKind::Default {
            head_dim: rotary_dim,
            theta,
        },
        LayerType::FullAttention => RopeKind::Proportional {
            rotary_dim,
            full_head_dim: cfg.full_head_dim_for_layer(layer)?,
            theta,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_golden() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rms_norm(&mut out, &x, &w, 1e-6);
        let expected = [0.365_148_35, 0.730_296_7, 1.095_445_, 1.460_593_4];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn default_rope_golden() {
        let mut freqs = [0.0f32; 4];
        compute_rope_freqs(
            &mut freqs,
            &[1],
            RopeKind::Default {
                head_dim: 4,
                theta: 10_000.0,
            },
        );
        assert!((freqs[0] - 0.540_302_3).abs() < 1e-5);
        assert!((freqs[1] - 0.841_470_96).abs() < 1e-5);
        assert!((freqs[2] - 0.999_95).abs() < 1e-5);
        assert!((freqs[3] - 0.01).abs() < 1e-5);

        let mut vec = [1.0f32, 0.0, 0.0, 1.0];
        apply_rope(&mut vec, &freqs, 4);
        assert!((vec[0] - 0.540_302_3).abs() < 1e-5);
        assert!((vec[1] - (-0.01)).abs() < 1e-4);
        assert!((vec[2] - 0.841_470_96).abs() < 1e-5);
        assert!((vec[3] - 0.999_95).abs() < 1e-5);
    }

    #[test]
    fn proportional_rope_freqs_differ_from_default() {
        let mut default = [0.0f32; 4];
        let mut prop = [0.0f32; 4];
        compute_rope_freqs(
            &mut default,
            &[1],
            RopeKind::Default {
                head_dim: 4,
                theta: 1_000_000.0,
            },
        );
        compute_rope_freqs(
            &mut prop,
            &[1],
            RopeKind::Proportional {
                rotary_dim: 4,
                full_head_dim: 8,
                theta: 1_000_000.0,
            },
        );
        assert!((default[2] - prop[2]).abs() > 1e-4);
    }

    #[test]
    fn proportional_partial_rope_leaves_non_rotated_dims() {
        let mut freqs = [0.0f32; 4];
        compute_rope_freqs(
            &mut freqs,
            &[1],
            RopeKind::Default {
                head_dim: 4,
                theta: 10_000.0,
            },
        );
        let mut vec = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        apply_rope(&mut vec, &freqs, 4);
        // rotary_dim=4 on head_dim=6: pairs (0,3) and (1,4); dims 2 and 5 untouched.
        assert!((vec[2] - 3.0).abs() < 1e-6);
        assert!((vec[5] - 6.0).abs() < 1e-6);
        assert!((vec[0] - 1.0).abs() > 1e-6 || (vec[3] - 4.0).abs() > 1e-6);
        assert!((vec[1] - 2.0).abs() > 1e-6 || (vec[4] - 5.0).abs() > 1e-6);
    }

    #[test]
    fn silu_and_gelu() {
        assert!((silu_f32(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu_pytorch_tanh_f32(0.0) - 0.0).abs() < 1e-6);
        assert!(silu_f32(1.0) > 0.0 && silu_f32(1.0) < 1.0);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
        softmax_rows(&mut x, 2, 3);
        let s0: f32 = x[0..3].iter().sum();
        let s1: f32 = x[3..6].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn diffusiongemma_layer0_sliding_rope_params() {
        let path = std::path::Path::new("model/transformer/config.json");
        if !path.exists() {
            return;
        }
        let cfg_json = std::fs::read_to_string(path).unwrap();
        let cfg: crate::config::ModelConfig = serde_json::from_str(&cfg_json).unwrap();
        let kind = rope_kind_for_layer(&cfg.text_config, 0).unwrap();
        match kind {
            RopeKind::Default { head_dim, theta } => {
                assert_eq!(head_dim, 256);
                assert!((theta - 10_000.0).abs() < 1e-3);
            }
            _ => panic!("layer 0 should be sliding/default rope"),
        }
        let kind5 = rope_kind_for_layer(&cfg.text_config, 5).unwrap();
        match kind5 {
            RopeKind::Proportional {
                rotary_dim,
                full_head_dim,
                theta,
            } => {
                assert_eq!(rotary_dim, 128);
                assert_eq!(full_head_dim, 512);
                assert!((theta - 1_000_000.0).abs() < 1e-3);
            }
            _ => panic!("layer 5 should be proportional rope"),
        }
    }

    #[test]
    fn linear_small() {
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut y = [0.0f32; 3];
        linear(&mut y, &x, &w, None, 1, 2, 3);
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }
}
