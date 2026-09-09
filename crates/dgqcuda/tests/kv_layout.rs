//! Local (no GPU) check: the kernel's KV indexing against the oracle buffers.
//!
//! dgq_attention reads, for position t and head h:
//!   k = kv + t*2*n_kv*hd + h*hd
//!   v = kv + t*2*n_kv*hd + n_kv*hd + h*hd
//! This pins that against the oracle's separate K/V buffers, which the GPU
//! interleave is supposed to reproduce.
#![cfg(not(feature = "cuda"))]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward::{self, Scratch};
use dgqcuda::weights::Weights;

#[test]
fn kv_layout_matches_oracle() {
    let dir = match std::env::var_os("DGQ_MODEL_DIR") {
        Some(d) => std::path::PathBuf::from(d),
        None => std::path::PathBuf::from("../../model/diffgemma-26b-a4b-it-q4"),
    };
    if !dir.join("model.dgq.bin").is_file() {
        eprintln!("skip: no local pack");
        return;
    }
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let ids: Vec<u32> = vec![2, 105, 2364, 107];
    let mut sc = Scratch::new(ids.len(), &cfg);
    let bufs = forward::attn_buffers(&w, &cfg, &ids, &mut sc).expect("attn");

    let t = &cfg.text_config;
    let seq = ids.len();
    let (n_kv, hd, _, _, _) = t.attn_geometry(0);
    let row = n_kv * hd;

    // GPU interleave: [t][2*row] = all K heads then all V heads.
    let mut kv = vec![0.0f32; seq * 2 * row];
    for pos in 0..seq {
        kv[pos * 2 * row..pos * 2 * row + row].copy_from_slice(&bufs.k[pos * row..(pos + 1) * row]);
        kv[pos * 2 * row + row..(pos + 1) * 2 * row]
            .copy_from_slice(&bufs.v[pos * row..(pos + 1) * row]);
    }

    let mut worst_k = 0.0f32;
    let mut worst_v = 0.0f32;
    for tpos in 0..seq {
        for kvh in 0..n_kv {
            let off = tpos * row + kvh * hd;
            let koff = tpos * 2 * row + kvh * hd;
            let voff = koff + row;
            for d in 0..hd {
                worst_k = worst_k.max((bufs.k[off + d] - kv[koff + d]).abs());
                worst_v = worst_v.max((bufs.v[off + d] - kv[voff + d]).abs());
            }
        }
    }
    println!("K layout max diff {worst_k:.3e}, V layout max diff {worst_v:.3e}");
    assert!(worst_k < 1e-6, "K layout mismatch: {worst_k}");
    assert!(worst_v < 1e-6, "V layout mismatch: {worst_v}");
}
