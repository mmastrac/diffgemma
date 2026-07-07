//! Viability probe for the M3 rotated-weights memory lever (ROADMAP task #45):
//! does a block-diagonal Hadamard rotation make low-bit quantization of the
//! REAL non-expert weights (attn q/k/v/o, dense gate/up/down) near-lossless?
//!
//! Measures the OUTPUT error of `Y = W·x` for random x — the metric that
//! actually matters (not weight rel-RMS) — comparing:
//! - plain group-32 affine q4/q6, vs
//! - rotated: `Q(W·Rᵀ)·(R·x)` with `R = I ⊗ H_blk` (block-diagonal Hadamard
//!   over the contraction dim, since hidden 2816 / FFN 2112 aren't pow2).
//! Since R is orthogonal, `W·Rᵀ·R·x = W·x`, so the only error is quantizing
//! the rotated weight — if rotation cuts it toward bf16, the lever is real.
//!
//! Run: `cargo test --release --features metal,blas weight_rot -- --ignored --nocapture`
//! (reads model/diffusiongemma-q4emb; ignored so normal CI skips the model).

#![allow(dead_code)]

use crate::kernels::sub::hadamard::{block_fwht, fwht_normalized, largest_pow2_divisor};
use crate::kernels::sub::kv_quant::{affine_roundtrip_levels, rel_rms};

/// Quantize every row of a row-major `[out, in]` weight group-32 along `in`,
/// optionally block-Hadamard-rotating each row first (rows = W·Rᵀ). Returns
/// the dequantized weight (same shape).
fn quant_weight(w: &[f32], out_dim: usize, in_dim: usize, levels: f32, blk: Option<usize>) -> Vec<f32> {
    let mut q = vec![0.0f32; w.len()];
    let mut row = vec![0.0f32; in_dim];
    for o in 0..out_dim {
        row.copy_from_slice(&w[o * in_dim..(o + 1) * in_dim]);
        if let Some(b) = blk {
            block_fwht(&mut row, b);
        }
        affine_roundtrip_levels(&row, &mut q[o * in_dim..(o + 1) * in_dim], levels);
    }
    q
}

fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let r = &w[o * in_dim..(o + 1) * in_dim];
            r.iter().zip(x).map(|(&a, &b)| a * b).sum()
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Rot {
    None,
    /// Block-diagonal H over the largest pow2 divisor (within-block mixing).
    Block,
    /// Full-dim mixing: pad `in` up to the next pow2 and Hadamard the whole
    /// padded vector (zeros dilute but every real channel is globally mixed).
    FullPadded,
}

fn quant_weight_pad(
    w: &[f32],
    out_dim: usize,
    in_dim: usize,
    padded: usize,
    levels: f32,
) -> Vec<f32> {
    // W·Rᵀ with R = full H over the padded dim: rotate each zero-padded row.
    let mut q = vec![0.0f32; out_dim * padded];
    let mut row = vec![0.0f32; padded];
    for o in 0..out_dim {
        row[..in_dim].copy_from_slice(&w[o * in_dim..(o + 1) * in_dim]);
        for r in row.iter_mut().skip(in_dim) {
            *r = 0.0;
        }
        fwht_normalized(&mut row);
        affine_roundtrip_levels(&row, &mut q[o * padded..(o + 1) * padded], levels);
    }
    q
}

/// Output rel-RMS of quantized `W` vs bf16 `W` over several random inputs.
fn out_err(w: &[f32], out_dim: usize, in_dim: usize, levels: f32, rot: Rot) -> f32 {
    let padded = in_dim.next_power_of_two();
    let (wq, blk) = match rot {
        Rot::None => (quant_weight(w, out_dim, in_dim, levels, None), None),
        Rot::Block => {
            let b = largest_pow2_divisor(in_dim);
            (quant_weight(w, out_dim, in_dim, levels, Some(b)), Some(b))
        }
        Rot::FullPadded => (quant_weight_pad(w, out_dim, in_dim, padded, levels), None),
    };
    let mut acc = 0.0f32;
    let trials = 4;
    for t in 0..trials {
        let x: Vec<f32> = (0..in_dim)
            .map(|i| (((i * 2654435761).wrapping_add(t * 40503) as u32 >> 9) as f32 / 8_388_608.0 - 1.0))
            .collect();
        let y_true = matvec(w, &x, out_dim, in_dim);
        let y_q = match rot {
            Rot::None => matvec(&wq, &x, out_dim, in_dim),
            Rot::Block => {
                let mut xr = x.clone();
                block_fwht(&mut xr, blk.unwrap());
                matvec(&wq, &xr, out_dim, in_dim)
            }
            Rot::FullPadded => {
                let mut xr = vec![0.0f32; padded];
                xr[..in_dim].copy_from_slice(&x);
                fwht_normalized(&mut xr);
                matvec(&wq, &xr, out_dim, padded)
            }
        };
        acc += rel_rms(&y_true, &y_q);
    }
    acc / trials as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgq::DgqStore;

    #[test]
    #[ignore]
    fn weight_rot_viability() {
        let dir = "model/diffusiongemma-q4emb";
        let store = match DgqStore::open(dir) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: no model at {dir}: {e:?}");
                return;
            }
        };
        // Representative non-expert weights (layer 0 sliding; a full layer).
        let names = [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.self_attn.k_proj.weight",
            "model.decoder.layers.0.self_attn.o_proj.weight",
            "model.decoder.layers.0.mlp.gate_proj.weight",
            "model.decoder.layers.0.mlp.down_proj.weight",
            "model.decoder.layers.5.self_attn.q_proj.weight",
        ];
        println!("\n{:<52} {:>6} {:>6}  q4:plain->rot   q6:plain->rot   blk", "tensor", "out", "in");
        for name in names {
            let Some(entry) = store.get_entry(name) else {
                println!("{name:<52}  (missing)");
                continue;
            };
            let shape = &entry.meta.shape;
            if shape.len() != 2 {
                continue;
            }
            let (out_dim, in_dim) = (shape[0] as usize, shape[1] as usize);
            let w = match store.tensor_f32(name) {
                Ok(w) => w,
                Err(e) => {
                    println!("{name:<52}  (read err {e:?})");
                    continue;
                }
            };
            if w.len() != out_dim * in_dim {
                continue;
            }
            let blk = largest_pow2_divisor(in_dim);
            let q4p = out_err(&w, out_dim, in_dim, 15.0, Rot::None);
            let q4b = out_err(&w, out_dim, in_dim, 15.0, Rot::Block);
            let q4f = out_err(&w, out_dim, in_dim, 15.0, Rot::FullPadded);
            let q6p = out_err(&w, out_dim, in_dim, 63.0, Rot::None);
            let q6f = out_err(&w, out_dim, in_dim, 63.0, Rot::FullPadded);
            let short = name.rsplit("layers.").next().unwrap_or(name);
            println!(
                "{short:<40} {out_dim:>5}x{in_dim:<5}  q4 none/blk/full {q4p:.4}/{q4b:.4}/{q4f:.4}   q6 none/full {q6p:.4}/{q6f:.4}  blk{blk}"
            );
        }
        println!();
    }
}
