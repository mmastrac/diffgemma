//! Tier-1 test for the grouped MoE expert GEMM: the CUDA kernel against a CPU
//! oracle over the same q4 weights, with buckets that overflow one row block
//! so the padding path is exercised too.
//!
//! The A matrix is the bucketed row space (what `dgq_moe_gather` produces and
//! what the down projection reads from the gate/up output), so the test applies
//! the same gather the device path does.
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

    // The kernel reads bucketed rows; gather the token rows in plan order
    // exactly as dgq_moe_gather does, and give the oracle the same matrix.
    let a_rows: Vec<f32> = (0..plan.rows())
        .flat_map(|r| {
            let tok = plan.tok_idx[r] as usize;
            a[tok * k_dim..(tok + 1) * k_dim].to_vec()
        })
        .collect();
    let want = grouped_gemm_cpu(&a_rows, &w_f32, &plan, k_dim, n_dim);

    let ctx = cached_context().expect("ctx");
    let ba = DeviceBuffer::alloc(&ctx, a_rows.len() * 4).unwrap();
    ba.write_f32(&a_rows).unwrap();
    let bw = DeviceBuffer::alloc(&ctx, blob.len()).unwrap();
    bw.write_bytes(&blob).unwrap();
    let rows = plan.rows();
    let bc = DeviceBuffer::alloc(&ctx, rows * n_dim * 4).unwrap();
    let starts = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.starts).unwrap();
    let tok_idx = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.tok_idx).unwrap();
    let experts = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.experts).unwrap();
    let gemm = GroupedGemm::new(k_dim, n_dim, "dgq_moe_gate_up");
    gemm.run(
        &ctx,
        &ba,
        &bw,
        &starts,
        &tok_idx,
        &experts,
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
    if std::env::var("DGQCUDA_DBG").is_ok() {
        println!("  want[0..4] {:?}", &want[..4]);
        println!("  got[0..4]  {:?}", &got[..4]);
        println!("  want[64..68] {:?}", &want[64..68]);
        println!("  got[64..68]  {:?}", &got[64..68]);
        println!("  starts {:?}", &plan.starts);
        println!("  tok_idx {:?}", &plan.tok_idx);
    }
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

/// The gather kernel writes one element per thread, so its grid must count
/// `rows * width` elements, not rows. Sizing it by rows fills only the leading
/// `blocks * blockDim / width` rows; the grouped GEMMs then read zeros for
/// every bucket past that and the whole expert path is silently wrong while the
/// router, the attention path, and every per-kernel test still pass.
#[test]
fn gather_fills_every_bucketed_row() {
    let (seq, top_k, n_experts, hidden) = (20usize, 8usize, 12usize, 2816usize);
    let idx: Vec<u32> = (0..seq * top_k).map(|i| (i % n_experts) as u32).collect();
    let wts: Vec<f32> = (0..seq * top_k).map(|i| 0.01 + 0.001 * i as f32).collect();
    let plan = GroupedPlan::new(&idx, &wts, seq, top_k, n_experts);
    let n_rows = plan.rows();
    assert_eq!(n_rows, seq * top_k, "every entry is non-empty");
    // The discriminator: more than one element per grid block.
    assert!(
        n_rows * hidden > n_rows * 256,
        "need several elements per block"
    );

    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32 * 0.0013).sin() * 0.9) + 0.1)
        .collect();
    let ctx = cached_context().expect("ctx");
    let bx = DeviceBuffer::alloc(&ctx, x.len() * 4).unwrap();
    bx.write_f32(&x).unwrap();
    let dst = DeviceBuffer::alloc(&ctx, n_rows * hidden * 4).unwrap();
    let tok_idx = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.tok_idx).unwrap();
    let kernel =
        gpukit::cuda::cached_source_kernel(dgqcuda::moe_grouped::MOE_KERNELS, "dgq_moe_gather")
            .expect("kernel");
    let mut args = gpukit::cuda::KernelArgs::new();
    args.device_ptr(bx.device_ptr())
        .device_ptr(dst.device_ptr())
        .device_ptr(tok_idx.device_ptr())
        .u32(n_rows as u32)
        .u32(hidden as u32);
    ctx.launch(
        &kernel,
        ((n_rows * hidden).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        0,
        &mut args,
    )
    .expect("gather launch");
    ctx.synchronize().unwrap();

    let mut got = vec![0.0f32; n_rows * hidden];
    dst.read_f32(&mut got).unwrap();
    for r in 0..n_rows {
        let src = plan.tok_idx[r] as usize * hidden;
        assert_eq!(
            got[r * hidden..(r + 1) * hidden],
            x[src..src + hidden],
            "bucketed row {r} was not gathered"
        );
    }
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
