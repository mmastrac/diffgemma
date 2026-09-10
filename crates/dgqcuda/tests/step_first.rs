//! Tier-2 test for the denoise pass's first-step contract. Step 1 has no
//! previous prediction, and the engine's production preamble takes that as a
//! signal to skip the self-conditioning MLP outright: `encode_step_preamble`
//! leaves `dense_off` zero, so the canvas rows are nothing but a scale-free RMS
//! norm of their own embeddings. The device used to run those embeddings
//! through the MLP instead, which put the canvas rows into layer 0 at the wrong
//! scale and sent the whole reply off.
//!
//! Two things are pinned here, both on CPU with the real pack:
//!   * what the row must be, computed independently of the code path that
//!     produces it, and against the engine's own printed value;
//!   * that the older `apply_self_conditioning` helper, fed the zero signal the
//!     production preamble implies, agrees with it exactly -- the two engine
//!     paths that had appeared to disagree.
//!
//! The device's own first-step branch needs no GPU to check: it calls the same
//! `rms_norm_no_scale_row` on the same row.
//!
//! Set DGQCUDA_MODEL to the pack directory to run it.
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward;
use dgqcuda::weights::Weights;

/// The first canvas token for seed 7.
const CANVAS_TOKEN: u32 = 13;

#[test]
fn first_step_canvas_row_is_the_bare_norm_of_its_embedding() {
    let Some(dir) = std::env::var("DGQCUDA_MODEL")
        .ok()
        .map(std::path::PathBuf::from)
    else {
        eprintln!("DGQCUDA_MODEL unset; skipping");
        return;
    };
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let eps = t.rms_norm_eps as f32;
    let scale = (h as f32).sqrt();

    let embed = w
        .tensor_f32("model.decoder.embed_tokens.weight")
        .expect("embed");
    let row: Vec<f32> = embed[CANVAS_TOKEN as usize * h..(CANVAS_TOKEN as usize + 1) * h]
        .iter()
        .map(|v| v * scale)
        .collect();
    let mut want = vec![0.0f32; h];
    forward::rms_norm_no_scale_row(&mut want, &row, eps);

    // The engine's own value for this row: `diffgemma step-layer-probe
    // --position 0` prints after_preamble as bf16, and the same computation in
    // f32 through the SC weights gives the value below. They agree to bf16
    // rounding.
    let eng = [-0.73053414f32, -0.22829193, -1.1427641, 0.3574399];
    for (i, (got, want)) in want.iter().zip(eng.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "slot {i}: bare norm {got} != engine {want}"
        );
    }
    // Scale-free norm: the row comes out at unit RMS whatever the embedding's
    // scale is, which is exactly why the extra SC MLP changed the layer-0 input
    // instead of just rescaling it.
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-4, "row rms {rms}");

    // ---- the older engine path, fed the signal production implies ---------
    let pre_norm = w
        .tensor_f32("model.decoder.self_conditioning.pre_norm.weight")
        .expect("sc pre_norm");
    let gate = w
        .tensor_f32("model.decoder.self_conditioning.gate_proj.weight")
        .expect("sc gate");
    let up = w
        .tensor_f32("model.decoder.self_conditioning.up_proj.weight")
        .expect("sc up");
    let down = w
        .tensor_f32("model.decoder.self_conditioning.down_proj.weight")
        .expect("sc down");
    let mut out = vec![0.0f32; h];
    forward::apply_self_conditioning(
        &mut out,
        &row,
        &vec![0.0f32; h],
        0,
        1,
        &cfg,
        &pre_norm,
        &gate,
        &up,
        &down,
    );
    for (i, (got, want)) in out.iter().zip(want.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "apply_self_conditioning with a zero signal (slot {i}): {got} != {want}"
        );
    }
    // And with a real signal it must NOT reproduce the bare norm, or the
    // first-step distinction would be untestable.
    let mut out2 = vec![0.0f32; h];
    forward::apply_self_conditioning(
        &mut out2, &row, &want, 0, 1, &cfg, &pre_norm, &gate, &up, &down,
    );
    let d = out2
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(d > 1e-3, "a real signal changed nothing (max|d| {d})");
}
