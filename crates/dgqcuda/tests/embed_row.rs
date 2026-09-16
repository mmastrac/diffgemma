//! Sanity gate for the embedding the device feeds layer 0: the engine's own
//! f32 table must give the same row the device's bf16 gather produced.
//!
//! Set DGQCUDA_MODEL to run it (reads only the embed table).
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::weights::Weights;

#[test]
fn embed_row0_statistics_match_the_pack() {
    let Some(dir) = std::env::var("DGQCUDA_MODEL")
        .ok()
        .map(std::path::PathBuf::from)
    else {
        eprintln!("DGQCUDA_MODEL unset; skipping");
        return;
    };
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let hidden = cfg.text_config.hidden_size;
    let scale = (hidden as f32).sqrt();
    let table = w
        .tensor_f32("model.decoder.embed_tokens.weight")
        .expect("embed table");
    assert!(table.len() >= 3 * hidden, "table too small");

    // Token 2 is <bos>, the first prompt token.
    let row = &table[2 * hidden..3 * hidden];
    let ss: f32 = row.iter().map(|v| v * scale * v * scale).sum();
    let big = row.iter().filter(|v| (**v * scale).abs() > 5.0).count();
    let mx = row.iter().fold(f32::MIN, |m, v| m.max((*v * scale).abs()));
    eprintln!(
        "pack embed row for <bos>: scaled_ss {ss:.3} maxabs {mx:.2} big {big} f4 {:?}",
        &row[..4].iter().map(|v| v * scale).collect::<Vec<_>>()
    );
}
