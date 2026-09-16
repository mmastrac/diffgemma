//! Tier-2 test for the resident KV cache: a canvas step that runs only the
//! canvas rows against cached prompt K/V must produce the same logits as the
//! CPU oracle's `[prompt][canvas]` pass, on the first step (self-conditioned
//! on the canvas embedding) and on a second (self-conditioned on a previous
//! prediction). Same oracle `denoise --parity` uses, small enough to run in a
//! test.
//!
//! Set DGQCUDA_MODEL to the pack directory to run it.
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::forward::{self, LogitRows, Scratch};
use dgqcuda::gpu::session::Session;
use dgqcuda::weights::Weights;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
        .0
}

fn softcap(v: &[f32], cap: f32) -> Vec<f32> {
    v.iter().map(|&x| (x / cap).tanh() * cap).collect()
}

/// Canvas rows of the oracle's logits for `[prompt][canvas]`, softcapped the
/// way the denoise loop feeds them back.
fn oracle(
    w: &Weights,
    cfg: &ModelConfig,
    prompt: &[u32],
    canvas: &[u32],
    prev: Option<&[f32]>,
) -> Vec<f32> {
    let t = &cfg.text_config;
    let mut ids = prompt.to_vec();
    ids.extend_from_slice(canvas);
    let mut sc = Scratch::new(ids.len(), cfg);
    let out = forward::forward_sc(
        w,
        cfg,
        &ids,
        None,
        LogitRows::All,
        &mut sc,
        prompt.len(),
        canvas.len(),
        prev,
    )
    .expect("oracle step");
    softcap(
        &out.logits[prompt.len() * t.vocab_size..],
        t.final_logit_softcapping as f32,
    )
}

#[test]
fn canvas_step_against_the_cache_matches_the_whole_sequence_oracle() {
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
    let vocab = t.vocab_size;
    let cap = t.final_logit_softcapping as f32;
    let prompt: Vec<u32> = vec![2, 105, 2364, 107];
    let canvas: Vec<u32> = vec![11, 11, 11];

    let mut sess = Session::open(&w, &cfg, prompt.len(), canvas.len()).expect("session");
    assert_eq!(sess.cache_len(), 0);
    sess.prefill(&prompt).expect("prefill");
    assert_eq!(sess.cache_len(), prompt.len(), "prefill appends every row");

    // Step 1: self-conditioned on the canvas embedding.
    let dev1 = softcap(&sess.step(&canvas).expect("step 1"), cap);
    assert_eq!(sess.cache_len(), prompt.len(), "a step must not extend the cache");
    let cpu1 = oracle(&w, &cfg, &prompt, &canvas, None);
    for row in 0..canvas.len() {
        let d = &dev1[row * vocab..(row + 1) * vocab];
        let c = &cpu1[row * vocab..(row + 1) * vocab];
        let cs = cos(d, c);
        assert!(cs > 0.9999, "step 1 row {row}: cos {cs} against the oracle");
        assert_eq!(argmax(d), argmax(c), "step 1 row {row}: argmax");
    }

    // Step 2: self-conditioned on a previous prediction, the same one on both
    // sides. A sharp synthetic one rather than step 1's own logits: this
    // prompt is the bare chat-template prefix, so step 1 is high-entropy, and
    // there the device soft embed (16 candidates per thread) and the CPU's
    // (every token within 10 logits) legitimately differ. That is a
    // soft-embed property, not a cache one, and this test pins the cache.
    let mut prev = vec![0.0f32; canvas.len() * vocab];
    for (row, p) in prev.chunks_mut(vocab).enumerate() {
        p[1000 + row] = 25.0;
    }
    sess.set_prev_logits(&prev).expect("prev logits");
    let dev2 = softcap(&sess.step(&canvas).expect("step 2"), cap);
    let cpu2 = oracle(&w, &cfg, &prompt, &canvas, Some(&prev));
    for row in 0..canvas.len() {
        let d = &dev2[row * vocab..(row + 1) * vocab];
        let c = &cpu2[row * vocab..(row + 1) * vocab];
        let cs = cos(d, c);
        assert!(cs > 0.9999, "step 2 row {row}: cos {cs} against the oracle");
        assert_eq!(argmax(d), argmax(c), "step 2 row {row}: argmax");
    }

    // The two steps must differ: if the soft embedding were not reaching the
    // canvas rows, step 2 would repeat step 1 and the check above would be
    // vacuous for it.
    let same = cos(&dev1, &dev2);
    assert!(same < 0.99999, "step 2 repeats step 1 (cos {same})");
}
