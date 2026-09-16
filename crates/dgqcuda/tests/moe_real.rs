//! Tier-2 test for the grouped MoE expert GEMM against the real pack: the CUDA
//! kernel and the CPU oracle must agree on the pack's actual q4 expert bytes.
//! The tier-1 cases use synthetic weights, so they cannot catch a wrong expert
//! stride or blob base in the real [n_experts, n_dim, k_dim] layout.
//!
//! Set DGQCUDA_MODEL to the pack directory to run it (it reads two expert
//! tensors, not the whole model).
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::moe_grouped::{GroupedGemm, GroupedPlan, grouped_gemm_cpu, q4_row_bytes};
use dgqcuda::weights::Weights;
use gpukit::cuda::{DeviceBuffer, cached_context};

fn pack_dir() -> Option<std::path::PathBuf> {
    std::env::var("DGQCUDA_MODEL").ok().map(Into::into)
}

/// Decode the real q4 expert blob to f32 the way the oracle multiplies it.
fn dequant_blob(blob: &[u8], rows: usize, k_dim: usize) -> Vec<f32> {
    let row_bytes = q4_row_bytes(k_dim);
    let mut out = vec![0.0f32; rows * k_dim];
    for r in 0..rows {
        let row = &blob[r * row_bytes..(r + 1) * row_bytes];
        let mut si = 0usize;
        let mut gi = 0usize;
        while gi < k_dim {
            let g_end = (gi + 32).min(k_dim);
            let delta = f32::from_bits((u16::from_le_bytes([row[si], row[si + 1]]) as u32) << 16);
            let mn = f32::from_bits((u16::from_le_bytes([row[si + 2], row[si + 3]]) as u32) << 16);
            si += 4;
            for j in 0..(g_end - gi) {
                let byte = row[si + j / 2];
                let q = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                out[r * k_dim + gi + j] = delta * q + mn;
            }
            si += 16;
            gi += 32;
        }
    }
    out
}

/// The pack reader and the raw q4 path must agree on the expert stack: the
/// kernel decodes the raw blob while the CPU oracle multiplies tensor_f32, and
/// a mismatch here silently routes every expert to the wrong weights.
#[test]
fn real_pack_tensor_f32_matches_the_raw_q4_blob() {
    let Some(dir) = pack_dir() else {
        eprintln!("DGQCUDA_MODEL unset; skipping");
        return;
    };
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let t = &cfg.text_config;
    let hidden = t.hidden_size;
    let inter = t.moe_intermediate_size;
    let gu_n = inter * 2;

    let raw = w
        .raw_bf16_bytes("model.decoder.layers.0.experts.gate_up_proj")
        .expect("gate_up bytes");
    let decoded = dequant_blob(&raw, t.num_experts * gu_n, hidden);
    let via_tensor = w
        .tensor_f32("model.decoder.layers.0.experts.gate_up_proj")
        .expect("tensor_f32");
    assert_eq!(via_tensor.len(), decoded.len(), "length");

    // Expert 79 is the one layer 0 routes its first token to.
    for e in [0usize, 1, 79] {
        let mut worst = 0.0f32;
        for i in 0..gu_n * hidden {
            let a = decoded[e * gu_n * hidden + i];
            let b = via_tensor[e * gu_n * hidden + i];
            worst = worst.max((a - b).abs());
        }
        eprintln!("expert {e}: max |raw - tensor_f32| = {worst:.6}");
        assert!(worst < 1e-3, "expert {e} weight mismatch {worst}");
    }

    // The f32 view must decode the same way for every expert in the tensor, not
    // just the ones a single layer happens to route to.
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for i in 0..decoded.len() {
        let d = (decoded[i] - via_tensor[i]).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    eprintln!("whole gate_up tensor: max |raw - tensor_f32| = {worst:.6} at {worst_at}");
    assert!(worst < 1e-3, "expert stack mismatch {worst} at {worst_at}");
}

#[test]
fn real_pack_experts_match_the_cpu_oracle() {
    let Some(dir) = pack_dir() else {
        eprintln!("DGQCUDA_MODEL unset; skipping");
        return;
    };
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let t = &cfg.text_config;
    let hidden = t.hidden_size;
    let inter = t.moe_intermediate_size;

    // Routing chosen so two buckets hold two tokens each, and expert 0 and
    // expert 2 get no tokens at all — an empty bucket still occupies a job
    // slot, so the kernel must take the expert index from the plan rather
    // than use the job index.
    let seq = 5usize;
    let top_k = 2usize;
    let idx: Vec<u32> = vec![1, 3, 1, 3, 1, 4, 3, 4, 1, 4];
    let wts: Vec<f32> = (0..seq * top_k).map(|i| 0.1 + 0.05 * i as f32).collect();
    let plan = GroupedPlan::new(&idx, &wts, seq, top_k, t.num_experts);
    let rows = plan.rows();
    // Only the experts that were actually routed to become buckets, and the
    // kernel takes each bucket's expert from the plan rather than assuming
    // job j is expert j.
    assert_eq!(plan.experts, vec![1, 3, 4]);
    for j in 0..plan.num_jobs() {
        assert!(plan.starts[j] < plan.starts[j + 1], "bucket {j} is empty");
    }
    let a_tok: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32 * 0.0011).sin() * 0.7) + 0.05)
        .collect();
    // A is the bucketed row space: row r holds token tok_idx[r].
    let a: Vec<f32> = (0..plan.rows())
        .flat_map(|r| {
            let tok = plan.tok_idx[r] as usize;
            a_tok[tok * hidden..(tok + 1) * hidden].to_vec()
        })
        .collect();

    let ctx = cached_context().expect("ctx");
    let ba = DeviceBuffer::alloc(&ctx, a.len() * 4).unwrap();
    ba.write_f32(&a).unwrap();
    let starts = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.starts).unwrap();
    let tok_idx = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.tok_idx).unwrap();
    let experts = dgqcuda::moe_grouped::upload_u32(&ctx, &plan.experts).unwrap();

    // ---- gate/up ---------------------------------------------------------
    let gu_blob = w
        .raw_bf16_bytes("model.decoder.layers.0.experts.gate_up_proj")
        .expect("gate_up bytes");
    let gu_n = inter * 2;
    assert_eq!(gu_blob.len(), t.num_experts * gu_n * q4_row_bytes(hidden));
    let gu_f32 = dequant_blob(&gu_blob, t.num_experts * gu_n, hidden);
    let want_gu = grouped_gemm_cpu(&a, &gu_f32, &plan, hidden, gu_n);
    let bw = DeviceBuffer::alloc(&ctx, gu_blob.len()).unwrap();
    bw.write_bytes(&gu_blob).unwrap();
    let bc = DeviceBuffer::alloc(&ctx, rows * gu_n * 4).unwrap();
    GroupedGemm::new(hidden, gu_n, "dgq_moe_gate_up")
        .run(
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
        .expect("gate_up launch");
    ctx.synchronize().unwrap();
    let mut got_gu = vec![0.0f32; rows * gu_n];
    bc.read_f32(&mut got_gu).unwrap();
    let cos = cosine(&want_gu, &got_gu);
    eprintln!("real gate_up cos {cos:.7}");
    assert!(cos > 0.9999, "gate_up cos {cos}");

    // ---- down ------------------------------------------------------------
    let dn_blob = w
        .raw_bf16_bytes("model.decoder.layers.0.experts.down_proj")
        .expect("down bytes");
    let dn_k = inter;
    assert_eq!(dn_blob.len(), t.num_experts * hidden * q4_row_bytes(dn_k));
    let dn_f32 = dequant_blob(&dn_blob, t.num_experts * hidden, dn_k);
    let act_tok: Vec<f32> = (0..seq * inter)
        .map(|i| ((i as f32 * 0.0021).cos() * 0.4) - 0.1)
        .collect();
    // The down projection's A is the gate/up output, which the grouped GEMM
    // already wrote in bucketed (expert-major) order — the kernel reads row r
    // directly and does not gather by tok_idx.
    let act: Vec<f32> = (0..rows)
        .flat_map(|r| {
            let tok = plan.tok_idx[r] as usize;
            act_tok[tok * inter..(tok + 1) * inter].to_vec()
        })
        .collect();
    let want_dn = grouped_gemm_cpu(&act, &dn_f32, &plan, dn_k, hidden);
    let ba2 = DeviceBuffer::alloc(&ctx, act.len() * 4).unwrap();
    ba2.write_f32(&act).unwrap();
    let bw2 = DeviceBuffer::alloc(&ctx, dn_blob.len()).unwrap();
    bw2.write_bytes(&dn_blob).unwrap();
    let bc2 = DeviceBuffer::alloc(&ctx, rows * hidden * 4).unwrap();
    GroupedGemm::new(dn_k, hidden, "dgq_moe_down")
        .run(
            &ctx,
            &ba2,
            &bw2,
            &starts,
            &tok_idx,
            &experts,
            &bc2,
            rows,
            plan.num_jobs(),
        )
        .expect("down launch");
    ctx.synchronize().unwrap();
    let mut got_dn = vec![0.0f32; rows * hidden];
    bc2.read_f32(&mut got_dn).unwrap();
    let cos = cosine(&want_dn, &got_dn);
    eprintln!("real down cos {cos:.7}");
    assert!(cos > 0.9999, "down cos {cos}");
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}
