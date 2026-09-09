//! dgq_attention against an explicit CPU reference over the KV layout.
#![cfg(feature = "cuda")]

use gpukit::cuda::{DeviceBuffer, KernelArgs, cached_context, cached_source_kernel};

/// Build [seq, n_kv, 2*hd] = per position all K heads then all V heads, run the
/// kernel, and compare against a reference that reads the same buffer with the
/// kernel's indexing:
///   k = pos_base + kvh*hd,  v = pos_base + n_kv*hd + kvh*hd,
///   pos_base = t * 2 * n_kv * hd
fn run_case(seq: usize, n_heads: usize, n_kv: usize, hd: usize) -> f64 {
    let q: Vec<f32> = (0..seq * n_heads * hd)
        .map(|i| (i as f32 * 0.0007).sin())
        .collect();
    let k: Vec<f32> = (0..seq * n_kv * hd)
        .map(|i| (i as f32 * 0.0011).cos())
        .collect();
    let v: Vec<f32> = (0..seq * n_kv * hd)
        .map(|i| (i as f32 * 0.0013).sin() + 0.3)
        .collect();

    let row = n_kv * hd;
    let mut kv = vec![0.0f32; seq * 2 * row];
    for t in 0..seq {
        kv[t * 2 * row..t * 2 * row + row].copy_from_slice(&k[t * row..(t + 1) * row]);
        kv[t * 2 * row + row..(t + 1) * 2 * row].copy_from_slice(&v[t * row..(t + 1) * row]);
    }

    let ctx = cached_context().expect("ctx");
    let bq = DeviceBuffer::alloc(&ctx, q.len() * 4).unwrap();
    bq.write_f32(&q).unwrap();
    let bkv = DeviceBuffer::alloc(&ctx, kv.len() * 4).unwrap();
    bkv.write_f32(&kv).unwrap();
    let bo = DeviceBuffer::alloc(&ctx, q.len() * 4).unwrap();
    let kk = cached_source_kernel(dgqcuda::KERNELS, "dgq_attention_v2").expect("kernel");
    let mut args = KernelArgs::new();
    args.device_ptr(bq.device_ptr())
        .device_ptr(bkv.device_ptr())
        .device_ptr(bo.device_ptr())
        .u32(seq as u32)
        .u32(n_heads as u32)
        .u32(n_kv as u32)
        .u32(hd as u32)
        .u32(seq as u32)
        .u32(1024);
    ctx.launch(
        &kk,
        ((seq * n_heads) as u32, 1, 1),
        (128, 1, 1),
        0,
        &mut args,
    )
    .expect("launch");
    ctx.synchronize().unwrap();
    let mut got = vec![0.0f32; q.len()];
    bo.read_f32(&mut got).unwrap();

    let mut want = vec![0.0f32; q.len()];
    for tok in 0..seq {
        for qh in 0..n_heads {
            let kvh = qh / (n_heads / n_kv);
            let qv = &q[(tok * n_heads + qh) * hd..(tok * n_heads + qh + 1) * hd];
            let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
            let mut acc = vec![0.0f32; hd];
            for t in 0..=tok {
                let pos_base = t * 2 * row;
                let koff = pos_base + kvh * hd;
                let dot: f32 = qv
                    .iter()
                    .zip(&kv[koff..koff + hd])
                    .map(|(a, b)| a * b)
                    .sum();
                let mn = m.max(dot);
                let corr = (m - mn).exp();
                let p = (dot - mn).exp();
                for a in acc.iter_mut() {
                    *a *= corr;
                }
                l = l * corr + p;
                m = mn;
                let voff = pos_base + n_kv * hd + kvh * hd;
                for d in 0..hd {
                    acc[d] += p * kv[voff + d];
                }
            }
            let o = &mut want[(tok * n_heads + qh) * hd..(tok * n_heads + qh + 1) * hd];
            for d in 0..hd {
                o[d] = acc[d] / l;
            }
        }
    }
    let (mut d, mut a, mut b) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..want.len() {
        d += want[i] as f64 * got[i] as f64;
        a += want[i] as f64 * want[i] as f64;
        b += got[i] as f64 * got[i] as f64;
    }
    d / (a.sqrt() * b.sqrt())
}

#[test]
fn attention_matches_reference() {
    let cos = run_case(4, 16, 8, 256);
    println!("attention cos {cos}");
    assert!(cos > 0.9999, "cos {cos}");
}
