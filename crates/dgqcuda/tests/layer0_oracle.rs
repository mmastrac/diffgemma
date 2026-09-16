//! The port's CPU decoder layer against the engine's, on the same synthetic
//! input and the same weights.
//!
//! `fixtures/layer0_row0_engine.f32` is row 0 of
//! `DGQ_LAYER0_DUMP=... diffgemma layer0 -m model/diffgemma-26b-a4b-it-q4`,
//! little-endian f32. Both sides build the input from the same formula, so
//! nothing but the answer has to cross.
//!
//! This is the comparison that found the dense branch folding into the
//! pre-feedforward norm instead of replacing it. With that bug the cosine here
//! is 0.9972; without it, 0.99999.
#![cfg(not(feature = "cuda"))]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward;
use dgqcuda::weights::Weights;

fn golden() -> Vec<f32> {
    let bytes = include_bytes!("fixtures/layer0_row0_engine.f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn layer0_matches_the_engine_oracle() {
    let dir = std::path::PathBuf::from("../../model/diffgemma-26b-a4b-it-q4");
    if !dir.join("model.dgq.bin").is_file() {
        eprintln!("skip: no local pack");
        return;
    }
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let hidden = cfg.text_config.hidden_size;

    let out = forward::layer0_synthetic_cpu(&w, &cfg).expect("layer 0");
    let row = &out[..hidden];
    let want = golden();
    assert_eq!(row.len(), want.len());

    let dot: f64 = row
        .iter()
        .zip(&want)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = row.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb);
    let err: f64 = row
        .iter()
        .zip(&want)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / nb;
    eprintln!("layer0 vs engine: cos={cos:.9} rel_l2={err:.6}");

    // The floor is the engine oracle reading the q4 experts widened to bf16
    // while this path dequantizes them to f32. The bug this guards against
    // costs cos 0.0028 and rel_l2 0.083, two orders above that floor.
    assert!(cos > 0.9999, "cosine {cos:.9} against the engine oracle");
    assert!(err < 0.005, "rel_l2 {err:.6} against the engine oracle");
}
