//! Tier-2 test for the denoise step's prompt contract.
//!
//! The step runs one sequence, [prompt][canvas], through the whole layer stack,
//! because the port has no resident KV cache and must rebuild the prompt's K/V
//! on every step. Those prompt rows are causal (`causal_split` is the prompt
//! length), so a prompt row attends only prompt keys up to itself and cannot
//! see the canvas. Two consequences are pinned here:
//!
//!   * the prompt rows are identical for two different canvases -- the step is
//!     not letting canvas rows influence causal prompt rows;
//!   * the layer stack's output on the prompt rows equals a standalone causal
//!     prefill over the same tokens, which is what the engine's KV cache holds
//!     when its denoise step runs the canvas alone.
//!
//! Handing the layer stack a post-layer prompt hidden instead of the embedding
//! (the bug this file exists to catch) applies all 30 layers to the prompt rows
//! a second time. Nothing crashed, the standalone `prompt_hidden` probe still
//! matched, and the only symptom was a canvas attending a wrong K/V that would
//! not sharpen.
//!
//! Set DGQCUDA_MODEL to the pack directory to run it.
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward::{self, LogitRows, Scratch};
use dgqcuda::weights::Weights;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

/// Run one step sequence and return the POST-final-norm value of the prompt
/// rows, which is what `forward_sc` leaves in `hidden_b`. Both step runs below
/// pass through that same final norm, and the causal comparison applies it
/// explicitly, so all three sides are the same quantity.
fn step_prompt_rows(
    w: &Weights,
    cfg: &ModelConfig,
    prompt: &[u32],
    canvas_seed: u32,
    canvas: usize,
) -> Vec<f32> {
    let hidden = cfg.text_config.hidden_size;
    let mut ids = prompt.to_vec();
    ids.extend(std::iter::repeat_n(canvas_seed, canvas));
    let mut sc = Scratch::new(ids.len(), cfg);
    forward::forward_sc(
        w,
        cfg,
        &ids,
        None,
        LogitRows::All,
        &mut sc,
        prompt.len(),
        canvas,
        None,
    )
    .expect("step oracle");
    sc.hidden_b[..prompt.len() * hidden].to_vec()
}

#[test]
fn step_prompt_rows_are_causal_and_canvas_independent() {
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
    let hidden = t.hidden_size;
    let prompt: Vec<u32> = vec![2, 105, 2364, 107];

    // Two step runs differing only in the canvas content.
    let a = step_prompt_rows(&w, &cfg, &prompt, 11, 3);
    let b = step_prompt_rows(&w, &cfg, &prompt, 12, 3);
    assert_eq!(a.len(), prompt.len() * hidden);
    let cross = cos(&a, &b);
    assert!(
        cross > 0.9999,
        "the prompt rows moved when only the canvas changed (cos {cross}); \
         canvas rows are influencing causal prompt rows"
    );

    // The prompt rows of a step must equal a standalone causal pass over the
    // same tokens: that is the KV the engine's prefill would have cached. Both
    // sides are post-final-norm here -- the causal side gets its norm applied
    // explicitly, the same way `forward_sc` applies it to the step.
    let mut csc = Scratch::new(prompt.len(), &cfg);
    let residual = forward::causal_hidden_after(&w, &cfg, &prompt, t.num_hidden_layers, &mut csc)
        .expect("causal prefill");
    let norm_w = w
        .tensor_f32("model.decoder.norm.weight")
        .expect("final norm");
    let mut causal = vec![0.0f32; residual.len()];
    forward::rms_norm_rows(
        &mut causal,
        &residual,
        &norm_w,
        prompt.len(),
        hidden,
        t.rms_norm_eps as f32,
    );

    let c = cos(&a, &causal);
    assert!(
        c > 0.999,
        "the step's prompt rows are not the causal prefill (cos {c}); the \
         layer stack is being applied to the prompt rows more than once"
    );
}
