//! CPU reference for monolithic sampler kernels (tempered rowstats, commit, apply, write).

use crate::sample::{accept_mask_from_entropies, early_stop_allowed, FILLER_TOKEN_ID, PAD_TOKEN_ID};

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
    pub conf_threshold: f32,
    pub stability_threshold: u32,
    pub min_early_stop_steps: u32,
    pub canvas_size: usize,
    pub pad_token: u32,
    pub filler_token: u32,
    pub entropy: &'a [f32],
    pub prev_argmax: &'a [u32],
}

pub struct CommitOut {
    pub u_cat: Vec<f32>,
    pub accept: Vec<u32>,
    pub sorted_idx: Vec<u32>,
    pub mean_entropy: f32,
    pub argmax_stable: u32,
    pub step: u32,
    pub stop_flag: u32,
    pub rng_state: u64,
}

/// Matches `sample_commit`.
pub fn sample_commit_cpu(
    step: u32,
    argmax_stable: u32,
    argmax_changed: u32,
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
        let mask = accept_mask_from_entropies(p.entropy, p.entropy_bound);
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

    let new_stable = if argmax_changed != 0 {
        0
    } else {
        argmax_stable + 1
    };
    let new_step = step + 1;

    let argmax: Vec<u32> = p.prev_argmax.to_vec();
    let degenerate = argmax
        .iter()
        .take(canvas)
        .all(|&t| t == p.pad_token || t == p.filler_token);
    let confident_stable =
        mean_entropy < p.conf_threshold && new_stable >= p.stability_threshold;
    let allowed = early_stop_allowed(new_step, &argmax[..canvas]);
    let mut stop_flag = 0u32;
    if confident_stable && allowed && !degenerate {
        stop_flag = 1;
    }
    if new_step >= p.max_steps {
        stop_flag = 1;
    }

    CommitOut {
        u_cat,
        accept,
        sorted_idx,
        mean_entropy,
        argmax_stable: new_stable,
        step: new_step,
        stop_flag,
        rng_state: st,
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
        let out = sample_commit_cpu(
            0,
            0,
            0,
            12345,
            CommitParams {
                max_steps: 4,
                entropy_bound: 0.25,
                conf_threshold: f32::MAX,
                stability_threshold: 99,
                min_early_stop_steps: 12,
                canvas_size: 4,
                pad_token: PAD_TOKEN_ID,
                filler_token: FILLER_TOKEN_ID,
                entropy: &entropy,
                prev_argmax: &prev,
            },
        );
        assert_eq!(out.accept.iter().filter(|&&a| a != 0).count(), 2);
    }
}
