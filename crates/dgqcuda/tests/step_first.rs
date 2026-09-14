//! Tier-2 test for the denoise pass's first-step contract.
//!
//! The self-conditioning MLP runs on EVERY step, the first included. Only its
//! input changes: from step 2 on it is the soft embedding of the previous
//! step's logits, and step 1 -- which has no previous prediction -- seeds it
//! with the canvas embeddings themselves, reading the initial canvas as the
//! step-0 prediction. That is the `first_step == 1` branch of the engine's
//! `interpret_step` (src/metal/step_kernel/enc.rs), which generation runs.
//!
//! Skipping the MLP is not a neutral simplification: with no signal the MLP's
//! bias-free linears emit zero, so the canvas row collapses to a bare
//! `rms_norm_no_scale` of its embedding, and the engine's own comment on that
//! branch records the model as degenerate under it (a cold-start empty reply).
//! On the port it saturated the final softcap -- every canvas logit pinned at
//! the 30.0 cap, so the post-cap distribution was flat, per-row entropy sat at
//! ~11 nats of a 12.48 ceiling, and the sampler accepted one token per step.
//!
//! The expected row below is the ENGINE'S PRODUCTION value, read from
//! `diffgemma step-layer-probe --seed 7 --layer-position 0` after that probe
//! was moved onto the production preamble. Do not re-derive it from
//! `encode_step_preamble`: that is a different function, which skips
//! self-conditioning whenever `first_step` is non-zero.
//!
//! Set DGQCUDA_MODEL to the pack directory to run it.
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward;
use dgqcuda::weights::Weights;

/// The first canvas token for seed 7.
const CANVAS_TOKEN: u32 = 13;

/// `after_preamble`, canvas position 0, seed 7, through a bf16 readback.
const ENGINE_AFTER_PREAMBLE: [f32; 4] = [0.7266, -0.3281, -1.0078, 0.2373];
/// bf16 keeps 8 mantissa bits, so the readback carries ~2^-9 relative error.
const BF16_TOL: f32 = 0.01;

struct Sc {
    pre_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

fn load_sc(w: &Weights) -> Sc {
    let get = |n: &str| w.tensor_f32(n).unwrap_or_else(|e| panic!("{n}: {e}"));
    Sc {
        pre_norm: get("model.decoder.self_conditioning.pre_norm.weight"),
        gate: get("model.decoder.self_conditioning.gate_proj.weight"),
        up: get("model.decoder.self_conditioning.up_proj.weight"),
        down: get("model.decoder.self_conditioning.down_proj.weight"),
    }
}

/// One canvas row through the SC MLP with `signal`, the shape `forward_sc`
/// and the device `Session::self_condition` both run.
fn sc_row(embed_row: &[f32], signal: &[f32], cfg: &ModelConfig, sc: &Sc) -> Vec<f32> {
    let mut out = vec![0.0f32; cfg.text_config.hidden_size];
    forward::apply_self_conditioning(
        &mut out,
        embed_row,
        signal,
        0,
        1,
        cfg,
        &sc.pre_norm,
        &sc.gate,
        &sc.up,
        &sc.down,
    );
    out
}

#[test]
fn first_step_canvas_row_self_conditions_on_its_own_embedding() {
    let Some(dir) = std::env::var("DGQCUDA_MODEL")
        .ok()
        .map(std::path::PathBuf::from)
    else {
        eprintln!("DGQCUDA_MODEL unset; skipping");
        return;
    };
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let h = cfg.text_config.hidden_size;
    let scale = (h as f32).sqrt();

    let embed = w
        .tensor_f32("model.decoder.embed_tokens.weight")
        .expect("embed");
    let row: Vec<f32> = embed[CANVAS_TOKEN as usize * h..(CANVAS_TOKEN as usize + 1) * h]
        .iter()
        .map(|v| v * scale)
        .collect();
    let sc = load_sc(&w);

    // Step 1: the signal IS the row's own embedding.
    let got = sc_row(&row, &row, &cfg, &sc);
    for (i, (got, want)) in got.iter().zip(ENGINE_AFTER_PREAMBLE.iter()).enumerate() {
        assert!(
            (got - want).abs() < BF16_TOL,
            "slot {i}: port {got} != engine {want}"
        );
    }

    // The preamble ends in a scale-free norm, so the row leaves at unit RMS
    // whatever the embedding's scale was.
    let rms = (got.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-4, "row rms {rms}");

    // The zero signal -- the no-SC branch the port used to take -- must NOT
    // reproduce it, or the contract would be untestable. It collapses to the
    // bare norm of the embedding, which is the degenerate case.
    let zero = sc_row(&row, &vec![0.0f32; h], &cfg, &sc);
    let mut bare = vec![0.0f32; h];
    forward::rms_norm_no_scale_row(&mut bare, &row, cfg.text_config.rms_norm_eps as f32);
    let to_bare = zero
        .iter()
        .zip(bare.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        to_bare < 1e-6,
        "zero signal is not the bare norm ({to_bare})"
    );
    let to_engine = zero
        .iter()
        .zip(ENGINE_AFTER_PREAMBLE.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        to_engine > BF16_TOL,
        "zero signal matched the engine ({to_engine}); the test cannot fail"
    );
}
