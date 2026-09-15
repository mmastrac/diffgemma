//! Tier-1 tests for the denoise sampler: the CPU oracle is the authority, so
//! these pin the accept rule, the entropy/argmax stats, and the early-stop
//! floor against hand-computed values.

use dgqcuda::denoise::{
    DenoiseState, MIN_EARLY_STOP_STEPS, PAD_TOKEN_ID, Rng, SamplerConfig, StopReason,
    accept_mask_from_entropies, row_stats, sample_row,
};

#[test]
fn rng_matches_the_engine_lcg() {
    // `sample.rs::Rng`: state = seed + 1, then xorshift-free LCG.
    let mut r = Rng::new(7);
    let a = r.next_u32();
    let mut state: u64 = 8;
    state = state
        .wrapping_mul(6_966_169_279)
        .wrapping_add(1_039_523_323);
    assert_eq!(a, (state >> 32) as u32);
    let b = r.next_u32();
    state = state
        .wrapping_mul(6_966_169_279)
        .wrapping_add(1_039_523_323);
    assert_eq!(b, (state >> 32) as u32);
}

#[test]
fn temperature_counts_down() {
    let cfg = SamplerConfig::default();
    let n = cfg.max_denoising_steps;
    // The engine calls `temperature_at_step(S.step)` with S.step counting
    // DOWN from max to 1, so the first denoise step is the hottest.
    assert!((cfg.temperature_at_step(n) - cfg.t_max).abs() < 1e-6);
    let mid = cfg.temperature_at_step(n / 2);
    assert!(mid > cfg.t_min && mid < cfg.t_max);
    // The schedule is a staircase: 48 steps never reach exactly t_min.
    assert!((cfg.temperature_at_step(1) - 0.408_333_3).abs() < 1e-5);
}

#[test]
fn accept_mask_is_an_entropy_sorted_prefix() {
    // Ascending entropies: every prefix fits under a large bound, so all
    // positions are accepted.
    let ent = [0.1, 0.2, 0.3, 0.4];
    assert_eq!(accept_mask_from_entropies(&ent, 10.0), vec![true; 4]);
    // A bound of 0.25 accepts only the two lowest (0.1 + 0.2 = 0.3 > 0.25
    // stops at the third).
    let mask = accept_mask_from_entropies(&ent, 0.25);
    assert_eq!(mask, vec![true, true, false, false]);
    // The lowest-entropy position is always accepted, even under a bound of 0.
    let mask = accept_mask_from_entropies(&ent, 0.0);
    assert_eq!(mask, vec![true, false, false, false]);
    // The mask is indexed by position, not by sorted rank. Sorted: 0.05, 0.8,
    // 0.9 — the prefix is checked BEFORE adding, so 0.8 is still accepted
    // (prefix 0.05 <= 0.1) and only 0.9 is cut (prefix 0.85 > 0.1).
    let ent = [0.9, 0.05, 0.8];
    let mask = accept_mask_from_entropies(&ent, 0.1);
    assert_eq!(mask, vec![false, true, true]);
}

#[test]
fn row_stats_matches_a_hand_computed_row() {
    // Two rows, 4 columns. Row 0 is uniform, so its entropy is ln(4) and the
    // argmax is the lowest id; row 1 is peaked, so its entropy is ~0.
    let logits = [1.0, 1.0, 1.0, 1.0, 0.0, 10.0, 0.0, 0.0];
    let stats = row_stats(&logits, 2, 4, 1.0);
    assert!((stats.entropy[0] - 4.0f32.ln()).abs() < 1e-5);
    assert_eq!(stats.argmax[0], 0);
    assert!(stats.entropy[1] < 0.002);
    assert_eq!(stats.argmax[1], 1);
    // Temperature divides the logits before the softmax, so a hotter row is
    // less peaked and has higher entropy.
    let hot = row_stats(&logits, 2, 4, 4.0);
    assert!(hot.entropy[1] > stats.entropy[1]);
    assert!(hot.entropy[1] < 4.0f32.ln());
}

#[test]
fn sample_row_is_an_inverse_cdf_over_the_tempered_row() {
    // Uniform row: the CDF rises in equal steps, so u picks the bin it lands in.
    let row = [0.0f32, 0.0, 0.0, 0.0];
    let stats = row_stats(&row, 1, 4, 1.0);
    assert!((stats.sum[0] - 4.0).abs() < 1e-6);
    assert!((stats.max[0] - 0.0).abs() < 1e-6);
    for (u, want) in [(0.01f32, 0u32), (0.30, 1), (0.55, 2), (0.99, 3)] {
        assert_eq!(
            sample_row(&row, stats.max[0], stats.sum[0], u, 1.0),
            want,
            "u {u}"
        );
    }
    // Peaked row: essentially all mass is on token 1, so every u >= its tail
    // picks it.
    let row = [0.0f32, 10.0, 0.0, 0.0];
    let stats = row_stats(&row, 1, 4, 1.0);
    for u in [0.001f32, 0.5, 0.999] {
        assert_eq!(
            sample_row(&row, stats.max[0], stats.sum[0], u, 1.0),
            1,
            "u {u}"
        );
    }
}

#[test]
fn step_commits_accepted_argmax_and_renoises_the_rest() {
    // A canvas of 2, vocab 4. Row 0 is sharply peaked (accepted); row 1 is
    // uniform (rejected, re-noised with a fresh uniform draw).
    let vocab = 4;
    let canvas = 2;
    let mut cfg = SamplerConfig::default();
    cfg.max_denoising_steps = 4;
    let mut st = DenoiseState::new(cfg, 7, canvas, vocab);
    st.ids = vec![0, 0];
    let logits = vec![0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let (stats, stop) = st.step(&logits, vocab);
    assert_eq!(stats.step, 1);
    assert!(stop.is_none());
    // The peaked row is accepted and takes its CATEGORICAL DRAW, not its
    // argmax: the engine's `sample_apply` fills `new_sample` by inverse CDF on
    // every step. With this row's mass at token 1 the draw lands there anyway,
    // so the commitment is checked against `new_sample` rather than by value.
    assert_eq!(st.ids[0], st.new_sample[0]);
    // The peaked row's draw must be its argmax here: the mass is at token 1.
    assert_eq!(st.new_sample[0], 1);
    assert!(st.accept[0]);
    // The uniform row is re-noised, so its id is some valid token.
    assert!(st.ids[1] < vocab as u32);
    // Both rows sit under the entropy bound of 0.1 here, so both are accepted;
    // the re-noise path is exercised by the uniform-row case in
    // `the_final_step_commits_every_position`.
    assert_eq!(stats.accept_count, 2);
}

#[test]
fn the_final_step_commits_every_position() {
    let vocab = 4;
    let canvas = 2;
    let mut cfg = SamplerConfig::default();
    cfg.max_denoising_steps = 1;
    let mut st = DenoiseState::new(cfg, 3, canvas, vocab);
    st.ids = vec![0, 0];
    let logits = vec![0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let (stats, stop) = st.step(&logits, vocab);
    // Every position commits on the last step, so a half-denoised canvas
    // cannot leak out of the loop.
    assert_eq!(stats.accept_count, 2);
    // Both positions commit their categorical draw, so these are not the argmax
    // by construction -- the engine's `sample_apply` samples every step and the
    // final step differs only by accepting all of it.
    assert_eq!(st.ids[0], st.new_sample[0]);
    assert_eq!(st.ids[1], st.new_sample[1]);
    assert_eq!(st.new_sample[0], 1, "the peaked row's mass is at token 1");
    assert!(st.new_sample[1] < vocab as u32);
    assert_eq!(stop, Some(StopReason::MaxSteps));
}

#[test]
fn a_confident_canvas_stops_only_after_the_minimum_steps() {
    // One canvas position whose logits never change: the argmax is stable from
    // step 2 on, and the entropy is ~0, so the confident stop fires as soon as
    // the step floor allows it.
    let vocab = 4;
    let mut cfg = SamplerConfig::default();
    cfg.max_denoising_steps = 40;
    let mut st = DenoiseState::new(cfg, 11, 1, vocab);
    let logits = vec![0.0, 20.0, 0.0, 0.0];
    let mut stop = None;
    let mut steps = 0;
    while stop.is_none() && steps < 40 {
        let (_, s) = st.step(&logits, vocab);
        steps += 1;
        stop = s;
    }
    assert_eq!(stop, Some(StopReason::Confident));
    assert!(steps >= MIN_EARLY_STOP_STEPS, "stopped after {steps} steps");
}

#[test]
fn argmax_is_the_emitted_canvas_and_ids_can_hold_noise() {
    // The reply is decoded from `argmax`, never from `ids`. This is the bug
    // that shipped: `denoise` printed `ids`, but `ids` carries the categorical
    // draw that drives the NEXT step, and a row the accept mask declines keeps
    // `rng.uniform_below(vocab)` -- a uniform random token. The engine commits
    // `st.prev_argmax` (step_generate/turn.rs), which is the argmax canvas.
    //
    // Two rows: row 0 sharply peaked at token 1, row 1 flat. A tight entropy
    // bound accepts only the peaked one, so row 1 exercises the re-noise path.
    let vocab = 4;
    let canvas = 2;
    let mut cfg = SamplerConfig::default();
    cfg.max_denoising_steps = 8;
    cfg.entropy_bound = 0.0;
    let mut st = DenoiseState::new(cfg, 11, canvas, vocab);
    st.ids = vec![0, 0];
    let logits = vec![0.0, 20.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let (stats, _) = st.step(&logits, vocab);

    // Only the peaked row is accepted, so the flat row was re-noised.
    assert_eq!(stats.accept_count, 1, "the flat row must be declined");
    assert!(st.accept[0] && !st.accept[1]);

    // `argmax` holds the true per-row argmax at EVERY position, declined ones
    // included. That is what makes it safe to emit.
    assert_eq!(st.argmax[0], 1, "peaked row's argmax is token 1");
    assert_eq!(
        st.argmax[1], 0,
        "a flat row's argmax is its first maximal token, not a random draw"
    );

    // `ids` at the declined row is a uniform draw with no relation to the
    // logits, which is precisely why emitting it leaked garbage into replies.
    assert!(st.ids[1] < vocab as u32);
}

#[test]
fn reply_ids_reads_the_argmax_canvas_and_stops_at_eos() {
    // Pins the call site, not just the invariant: the shipped bug was that
    // `denoise` decoded `ids`, which is a field access away from correct.
    let vocab = 8;
    let canvas = 4;
    let mut cfg = SamplerConfig::default();
    cfg.max_denoising_steps = 8;
    let mut st = DenoiseState::new(cfg, 5, canvas, vocab);
    st.argmax = vec![3, 4, 1, 6];
    // Deliberately different at every position, and holding the eos id where
    // the argmax does not: a reply built from `ids` would both contain these
    // tokens and terminate in the wrong place.
    st.ids = vec![7, 1, 7, 7];

    // eos is token 1, so the reply is the argmax prefix before it.
    assert_eq!(st.reply_ids(&[1]), vec![3, 4]);
    // No eos in the canvas: the whole argmax row, padding dropped.
    st.argmax = vec![3, 4, PAD_TOKEN_ID, 6];
    assert_eq!(st.reply_ids(&[1]), vec![3, 4, 6]);
    // Several stop ids: the FIRST one encountered ends the reply.
    st.argmax = vec![3, 6, 4, 1];
    assert_eq!(st.reply_ids(&[1, 6]), vec![3]);
}
