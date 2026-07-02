//! CPU reference for monolithic sampler kernels (tempered rowstats, commit, apply, write).

use crate::sample::{
    accept_mask_from_entropies, accept_mask_sig, diffusion_append_argmax_hist,
    diffusion_argmax_canvas_stable, ARGMAX_HIST_MAX,
};
use crate::sample::{STOP_FLAG_CONFIDENT, STOP_FLAG_MAX_STEPS, STOP_FLAG_PLATEAU};

/// Temperature after `steps_done` denoise iterations (matches `temp_at` in Metal).
pub fn temp_at(steps_done: u32, max_steps: u32, t_min: f32, t_max: f32) -> f32 {
    let n = max_steps.max(1) as f32;
    let cur = (max_steps - steps_done) as f32;
    t_min + (t_max - t_min) * (cur / n)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperedRowStats {
    pub mx: f32,
    pub sum: f32,
    pub entropy: f32,
    pub argmax: u32,
}

/// Tempered row max/sumexp/entropy/argmax (matches `sample_rowstats` per row).
pub fn tempered_row_stats(row: &[f32], temperature: f32) -> TemperedRowStats {
    let t = temperature.max(1e-6);
    let mut mx = f32::NEG_INFINITY;
    let mut am = 0u32;
    let mut amv = f32::NEG_INFINITY;
    for (v, &lv) in row.iter().enumerate() {
        let x = lv / t;
        if x > amv {
            amv = x;
            am = v as u32;
        }
        mx = mx.max(x);
    }
    let mut sum = 0.0f32;
    let mut ent_acc = 0.0f32;
    for &lv in row {
        let x = lv / t;
        let e = (x - mx).exp();
        sum += e;
        ent_acc += e * (x - mx);
    }
    let entropy = if sum > 0.0 {
        sum.ln() - ent_acc / sum
    } else {
        0.0
    };
    TemperedRowStats {
        mx,
        sum,
        entropy,
        argmax: am,
    }
}

/// Inverse-CDF sample from tempered logits (matches `sample_apply` per row).
pub fn tempered_sample_row(row: &[f32], mx: f32, z: f32, u: f32, temperature: f32) -> u32 {
    let t = temperature.max(1e-6);
    let target = u * z;
    let mut cum = 0.0f32;
    for (v, &lv) in row.iter().enumerate() {
        cum += (lv / t - mx).exp();
        if cum >= target {
            return v as u32;
        }
    }
    row.len().saturating_sub(1) as u32
}

pub struct CommitParams<'a> {
    pub max_steps: u32,
    pub entropy_bound: f32,
    /// Per-row accept cap (<=0 disables): see `accept_mask_from_entropies_capped`.
    pub accept_row_ent_max: f32,
    pub conf_threshold: f32,
    pub stability_threshold: u32,
    pub min_early_stop_steps: u32,
    pub accept_plateau_threshold: u32,
    pub plateau_prefix_mean_max: f32,
    pub canvas_size: usize,
    pub pad_token: u32,
    pub filler_token: u32,
    pub eos_token_id: u32,
    pub entropy: &'a [f32],
    pub prev_argmax: &'a [u32],
    pub accept_plateau: u32,
    pub prev_accept_sig: u32,
    pub argmax_hist_len: u32,
    pub argmax_hist_base: u32,
    pub argmax_hist: &'a mut [u32],
}

pub struct CommitOut {
    pub u_cat: Vec<f32>,
    pub accept: Vec<u32>,
    pub sorted_idx: Vec<u32>,
    pub mean_entropy: f32,
    pub canvas_stable: u32,
    pub argmax_hist_len: u32,
    pub argmax_hist_base: u32,
    pub step: u32,
    pub stop_flag: u32,
    pub rng_state: u64,
    pub accept_plateau: u32,
    pub prev_accept_sig: u32,
}

/// Matches `sample_commit`.
pub fn sample_commit_cpu(
    step: u32,
    rng_state: u64,
    p: CommitParams<'_>,
) -> CommitOut {
    let canvas = p.canvas_size;
    let mut u_cat = vec![0.0f32; canvas];
    let mut st = rng_state;
    for i in 0..canvas {
        st = st.wrapping_mul(6_966_169_279).wrapping_add(1_039_523_323);
        u_cat[i] = (st >> 32) as f32 / 4_294_967_296.0;
    }

    let mut sorted_idx: Vec<u32> = (0..canvas as u32).collect();
    for i in 1..canvas {
        let id = sorted_idx[i];
        let e = p.entropy[id as usize];
        let mut j = i as i32 - 1;
        while j >= 0 && p.entropy[sorted_idx[j as usize] as usize] > e {
            sorted_idx[(j + 1) as usize] = sorted_idx[j as usize];
            j -= 1;
        }
        sorted_idx[(j + 1) as usize] = id;
    }

    let final_step = step + 1 >= p.max_steps;
    let mut accept = vec![0u32; canvas];
    if !final_step {
        let mask = crate::sample::accept_mask_from_entropies_capped(
            p.entropy,
            p.entropy_bound,
            p.accept_row_ent_max,
        );
        for (i, &m) in mask.iter().enumerate() {
            if m {
                accept[i] = 1;
            }
        }
    }

    let mean_entropy = if canvas > 0 {
        p.entropy.iter().sum::<f32>() / canvas as f32
    } else {
        0.0
    };

    let sig = accept_mask_sig(&accept);
    let accept_plateau = if sig == p.prev_accept_sig {
        p.accept_plateau.saturating_add(1)
    } else {
        0
    };

    let canvas_stable = diffusion_argmax_canvas_stable(
        p.prev_argmax,
        p.argmax_hist_len,
        p.argmax_hist_base,
        p.argmax_hist,
        canvas,
        p.stability_threshold,
    );
    let (hist_len, hist_base) = diffusion_append_argmax_hist(
        p.argmax_hist,
        p.argmax_hist_len,
        p.argmax_hist_base,
        p.prev_argmax,
        canvas,
        p.stability_threshold,
    );
    let new_step = step + 1;

    let confident_stable = canvas_stable && mean_entropy < p.conf_threshold;
    let plateau_stop = accept_plateau >= p.accept_plateau_threshold
        && mean_entropy < p.plateau_prefix_mean_max;
    let mut stop_flag = 0u32;
    if confident_stable {
        stop_flag = STOP_FLAG_CONFIDENT;
    } else if plateau_stop {
        stop_flag = STOP_FLAG_PLATEAU;
    } else if new_step >= p.max_steps {
        stop_flag = STOP_FLAG_MAX_STEPS;
    }

    CommitOut {
        u_cat,
        accept,
        sorted_idx,
        mean_entropy,
        canvas_stable: u32::from(canvas_stable),
        argmax_hist_len: hist_len,
        argmax_hist_base: hist_base,
        step: new_step,
        stop_flag,
        rng_state: st,
        accept_plateau,
        prev_accept_sig: sig,
    }
}

/// Matches `sample_write`.
pub fn sample_write_cpu(
    ids: &mut [u32],
    accept: &[u32],
    new_sample: &[u32],
    canvas_size: usize,
    vocab_size: u32,
    mut rng_state: u64,
) -> u64 {
    for i in 0..canvas_size {
        if accept[i] != 0 {
            ids[i] = new_sample[i];
        } else {
            rng_state = rng_state
                .wrapping_mul(6_966_169_279)
                .wrapping_add(1_039_523_323);
            ids[i] = ((rng_state >> 32) as u32) % vocab_size.max(1);
        }
    }
    rng_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{FILLER_TOKEN_ID, PAD_TOKEN_ID};

    #[test]
    fn temp_at_matches_sample_rs() {
        let cfg = crate::sample::SamplerConfig::default();
        for step in 1..=8 {
            let cpu = cfg.temperature_at_step(step);
            let metal = temp_at(
                (cfg.max_denoising_steps - step) as u32,
                cfg.max_denoising_steps as u32,
                cfg.t_min,
                cfg.t_max,
            );
            assert!((cpu - metal).abs() < 1e-6, "step={step} cpu={cpu} metal={metal}");
        }
    }

    #[test]
    fn commit_respects_pad_filler() {
        let entropy = vec![0.1, 0.2, 0.3, 0.4];
        let prev = vec![PAD_TOKEN_ID, FILLER_TOKEN_ID, 42, 43];
        let mut hist = vec![0u32; 4 * ARGMAX_HIST_MAX];
        let out = sample_commit_cpu(
            0,
            12345,
            CommitParams {
                max_steps: 4,
                entropy_bound: 0.25,
                accept_row_ent_max: 0.0,
                conf_threshold: f32::MAX,
                stability_threshold: 99,
                min_early_stop_steps: 12,
                accept_plateau_threshold: 2,
                plateau_prefix_mean_max: 0.05,
                canvas_size: 4,
                pad_token: PAD_TOKEN_ID,
                filler_token: FILLER_TOKEN_ID,
                eos_token_id: 1,
                entropy: &entropy,
                prev_argmax: &prev,
                accept_plateau: 0,
                prev_accept_sig: 0,
                argmax_hist_len: 0,
                argmax_hist_base: 0,
                argmax_hist: &mut hist,
            },
        );
        assert_eq!(out.accept.iter().filter(|&&a| a != 0).count(), 2);
    }
}
