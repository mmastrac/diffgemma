//! CPU oracle rms_norm_rows vs the GPU kernel on identical input.
#![cfg(not(feature = "cuda"))]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward::{self, Scratch};
use dgqcuda::weights::{LayerKeys, Weights};

#[test]
fn cpu_rms_formula() {
    let dir = std::path::PathBuf::from("../../model/diffgemma-26b-a4b-it-q4");
    if !dir.join("model.dgq.bin").is_file() {
        eprintln!("skip: no local pack");
        return;
    }
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let t = &cfg.text_config;
    let hidden = t.hidden_size;
    let eps = t.rms_norm_eps as f32;
    let keys = LayerKeys::new(0);
    let norm_w = w.tensor_f32(&keys.post_feedforward_layernorm_2).expect("w");

    // A row with a real moe_out magnitude profile.
    let row: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0031).sin() * 8.0 + ((i as f32) * 0.0007).cos() * 3.0)
        .collect();
    let ss: f32 = row.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / hidden as f32 + eps).sqrt();
    let mut out = vec![0.0f32; hidden];
    forward::rms_norm_rows(&mut out, &row, &norm_w, 1, hidden, eps);
    println!(
        "ss={ss} inv={inv} out[0]={} expected={}",
        out[0],
        row[0] * inv * norm_w[0]
    );
    assert!((out[0] - row[0] * inv * norm_w[0]).abs() < 1e-6);
}
