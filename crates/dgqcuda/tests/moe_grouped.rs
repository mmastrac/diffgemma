//! Tier-1 test for the grouped MoE expert GEMM: the CUDA kernel against a CPU
//! oracle over the same q4 weights, with buckets that overflow one row block
//! so the padding path is exercised too.
#![cfg(feature = "cuda")]

use dgqcuda::moe_grouped::{BM, GroupedGemm, GroupedPlan, grouped_gemm_cpu, q4_row_bytes};
use gpukit::cuda::{DeviceBuffer, cached_context};

/// Quantize a row to q4, then decode it back — the exact bytes the kernel reads
/// and the f32 values the oracle multiplies.
fn q4_roundtrip(row: &[f32], k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = vec![0u8; q4_row_bytes(k)];
    dgqcuda_quantize_row_q4(row, k, &mut bytes);
    let mut back = vec![0.0f32; k];
    dgqcuda_dequant_row_q4(&bytes, k, &mut back);
    (bytes, back)
}

fn dgqcuda_quantize_row_q4(row: &[f32], k: usize, dst: &mut [u8]) {
    // Mirrors src/dgq/block.rs::quantize_row_q4 (affine group of 32, bf16
    // scale/min, low nibble = even index).
    let mut off = 0usize;
    let mut gi = 0usize;
    while gi < k {
        let g_end = (gi + 32).min(k);
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in &row[gi..g_end] {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        if mn == f32::INFINITY {
            mn = 0.0;
            mx = 0.0;
        }
        let delta = if mx - mn < 1e-8 {
            1.0
        } else {
            (mx - mn) / 15.0
        };
        let d = (delta.to_bits() >> 16) as u16;
        let m = (mn.to_bits() >> 16) as u16;
        dst[off] = d as u8;
        dst[off + 1] = (d >> 8) as u8;
        dst[off + 2] = m as u8;
        dst[off + 3] = (m >> 8) as u8;
        off += 4;
        let mut nib = [0u8; 16];
        for j in 0..(g_end - gi) {
            let q = if delta <= 0.0 {
                0u8
            } else {
                ((row[gi + j] - mn) / delta).round().clamp(0.0, 15.0) as u8
            };
            if j % 2 == 0 {
                nib[j / 2] = q;
            } else {
                nib[j / 2] |= q << 4;
            }
        }
        dst[off..off + 16].copy_from_slice(&nib);
        off += 16;
        gi += 32;
    }
}

fn dgqcuda_dequant_row_q4(src: &[u8], k: usize, dst: &mut [f32]) {
    let mut si = 0usize;
    let mut gi = 0usize;
    while gi < k {
        let g_end = (gi + 32).min(k);
        let delta = f32::from_bits((u16::from_le_bytes([src[si], src[si + 1]]) as u32) << 16);
        let mn = f32::from_bits((u16::from_le_bytes([src[si + 2], src[si + 3]]) as u32) << 16);
        si += 4;
        for j in 0..(g_end - gi) {
            let byte = src[si + j / 2];
            let q = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
            dst[gi + j] = delta * q + mn;
        }
        si += 16;
        gi += 32;
    }
}

fn case(seq: usize, top_k: usize, n_experts: usize, k_dim: usize, n_dim: usize) -> f64 {
    let a: Vec<f32> = (0..seq * k_dim)
        .map(|i| ((i as f32 * 0.0017).sin() * 1.5) + 0.25)
        .collect();
    // Routing: expert = (7*s + 3*k) % n_experts keeps the buckets uneven.
    let mut idx = vec![0u32; seq * top_k];
    let mut wts = vec![0.0f32; seq * top_k];
    for s in 0..seq {
        for k in 0..top_k {
            idx[s * top_k + k] = ((7 * s + 3 * k) % n_experts) as u32;
            wts[s * top_k + k] = 0.5 + 0.05 * (s as f32 + k as f32);
        }
    }
    let plan = GroupedPlan::new(&idx, &wts, seq, top_k, n_experts);
    assert!(plan.rows() == seq * top_k);

    // Expert weights: q4-encode, then decode for the oracle.
    let mut blob = Vec::new();
    let mut w_f32 = vec![0.0f32; n_experts * n_dim * k_dim];
    for e in 0..n_experts {
        for o in 0..n_dim {
            let row: Vec<f32> = (0..k_dim)
                .map(|k| ((e * 31 + o * 7 + k) as f32 * 0.0021).cos() * 0.8)
                .collect();
            let (bytes, back) = q4_roundtrip(&row, k_dim);
            blob.extend_from_slice(&bytes);
            w_f32[e * n_dim * k_dim + o * k_dim..e * n_dim * k_dim + (o + 1) * k_dim]
                .copy_from_slice(&back);
        }
    }

    let want = grouped_gemm_cpu(&a, &w_f32, &plan, k_dim, n_dim);

    let ctx = cached_context().expect("ctx");
    let ba = DeviceBuffer::alloc(&ctx, a.len() * 4).unwrap();
    ba.write_f32(&a).unwrap();
    let bw = DeviceBuffer::alloc(&ctx, blob.len()).unwrap();
    bw.write_bytes(&blob).unwrap();
    let rows = plan.rows();
    let bc = DeviceBuffer::alloc(&ctx, rows * n_dim * 4).unwrap();
    let starts = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.starts).unwrap();
    let tok_idx = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.tok_idx).unwrap();
    let gemm = GroupedGemm::new(k_dim, n_dim, "dgq_moe_gate_up");
    gemm.run(
        &ctx,
        &ba,
        &bw,
        &starts,
        &tok_idx,
        &bc,
        rows,
        plan.num_jobs(),
    )
    .expect("launch");
    ctx.synchronize().unwrap();
    let mut got = vec![0.0f32; rows * n_dim];
    bc.read_f32(&mut got).unwrap();

    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..want.len() {
        dot += want[i] as f64 * got[i] as f64;
        na += want[i] as f64 * want[i] as f64;
        nb += got[i] as f64 * got[i] as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    println!("seq {seq} k {k_dim} n {n_dim}: cos {cos:.9}");
    cos
}

#[test]
fn grouped_gemm_matches_the_cpu_oracle() {
    // A bucket larger than BM exercises the row-block padding, and n_dim not a
    // multiple of BN exercises the column guard.
    let cos = case(70, 2, 8, 128, 192);
    assert!(cos > 0.99999, "cos {cos}");
}

#[test]
fn grouped_gemm_handles_one_token_per_bucket() {
    let cos = case(5, 1, 8, 64, 64);
    assert!(cos > 0.99999, "cos {cos}");
}

#[test]
fn plan_buckets_are_expert_major_and_weighted() {
    // Two tokens, three experts, top-2: token 0 -> {0, 2}, token 1 -> {1, 2}.
    let idx = [0u32, 2, 1, 2];
    let wts = [0.1f32, 0.2, 0.3, 0.4];
    let plan = GroupedPlan::new(&idx, &wts, 2, 2, 3);
    assert_eq!(plan.starts, vec![0, 1, 2, 4]);
    assert_eq!(plan.tok_idx, vec![0, 1, 0, 1]);
    assert_eq!(plan.row_w, vec![0.1, 0.3, 0.2, 0.4]);
    assert_eq!(plan.rows(), 4);
    assert_eq!(plan.num_jobs(), 3);
}

#[test]
fn bm_is_the_kernel_tile() {
    assert_eq!(BM, 32);
}
