//! `fit-token-probe` — fit an output-token classification probe for THIS model.
//!
//! The probe in `crate::token_class` is a direction in hidden space, and hidden
//! space is checkpoint-specific: a direction fitted on one model is meaningless
//! on another. Rather than ship a baked artifact, this command refits it from a
//! labelled prompt spec in a few minutes, so an end user can regenerate it
//! locally after swapping or requantizing a model.
//!
//! Method is difference-of-class-means over L2-normalised hidden rows,
//! deliberately NOT logistic regression: at hidden 2816 with a few dozen
//! samples any fitted separator reaches 100% on its own training set and
//! reports nothing. Difference-of-means is low variance, and the leave-one-out
//! score printed here is a real held-out number.
//!
//! The run also prints an EFFECT SIZE (between-class gap over within-class
//! scatter). Accuracy saturates — several layers hit 1.00 — while effect size
//! keeps ranking them, so it is the number to compare layers on.

use super::*;
use crate::token_class::TokenClassProbe;

#[derive(serde::Deserialize)]
struct ProbeSpec {
    class_names: Vec<String>,
    #[serde(default = "default_layer")]
    layer: usize,
    #[serde(default = "default_position")]
    position: usize,
    samples: Vec<ProbeSample>,
}
fn default_layer() -> usize {
    29
}
fn default_position() -> usize {
    129
}

#[derive(serde::Deserialize)]
struct ProbeSample {
    class: String,
    prompt: String,
}

/// Between-class gap over within-class scatter. Layer-comparable, unlike
/// accuracy, which saturates.
fn effect_size(rows: &[Vec<f32>], y: &[u8]) -> f32 {
    let dim = rows[0].len();
    let mean = |want: u8| {
        let sel: Vec<&Vec<f32>> = rows
            .iter()
            .zip(y)
            .filter(|(_, c)| **c == want)
            .map(|(r, _)| r)
            .collect();
        let mut m = vec![0.0f32; dim];
        for r in &sel {
            for (a, b) in m.iter_mut().zip(r.iter()) {
                *a += b;
            }
        }
        for a in &mut m {
            *a /= sel.len().max(1) as f32;
        }
        m
    };
    let (m1, m0) = (mean(1), mean(0));
    let gap = m1
        .iter()
        .zip(&m0)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let mut scat = 0.0f32;
    for (r, c) in rows.iter().zip(y) {
        let m = if *c == 1 { &m1 } else { &m0 };
        scat += r.iter().zip(m).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
    }
    let scat = (scat / rows.len().max(1) as f32).sqrt();
    if scat <= 0.0 { 0.0 } else { gap / scat }
}

/// Leave-one-out accuracy of the difference-of-means rule. Refuses to guess
/// when the direction is degenerate — a zero-signal layer would otherwise leak
/// its label through floating-point rounding of the class means and report a
/// perfect score (this is not hypothetical; it happened during development).
fn loo_accuracy(rows: &[Vec<f32>], y: &[u8]) -> f32 {
    let dim = rows[0].len();
    let mut ok = 0usize;
    for held in 0..rows.len() {
        let (mut m1, mut m0) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        let (mut n1, mut n0) = (0usize, 0usize);
        for (i, (r, c)) in rows.iter().zip(y).enumerate() {
            if i == held {
                continue;
            }
            let (m, n) = if *c == 1 {
                (&mut m1, &mut n1)
            } else {
                (&mut m0, &mut n0)
            };
            for (a, b) in m.iter_mut().zip(r.iter()) {
                *a += b;
            }
            *n += 1;
        }
        if n1 == 0 || n0 == 0 {
            continue;
        }
        for a in &mut m1 {
            *a /= n1 as f32;
        }
        for a in &mut m0 {
            *a /= n0 as f32;
        }
        let wnorm: f32 = m1
            .iter()
            .zip(&m0)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        if wnorm < 1e-8 {
            continue; // degenerate: no signal, do not manufacture one
        }
        let s: f32 = rows[held]
            .iter()
            .zip(&m1)
            .zip(&m0)
            .map(|((h, a), b)| (h - (a + b) * 0.5) * (a - b))
            .sum();
        if (s > 0.0) == (y[held] == 1) {
            ok += 1;
        }
    }
    ok as f32 / rows.len().max(1) as f32
}

fn l2_normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn run_fit_token_probe_cmd(
    model_dir: &std::path::Path,
    spec_path: &std::path::Path,
    output: &std::path::Path,
    layer_override: Option<usize>,
    seed: u64,
    max_seq: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_layer_hidden_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: fit-token-probe requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let spec: ProbeSpec = match std::fs::read_to_string(spec_path)
        .map_err(|e| e.to_string())
        .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: parse {}: {e}", spec_path.display());
            return ExitCode::FAILURE;
        }
    };
    if spec.class_names.len() != 2 {
        eprintln!("error: spec must name exactly 2 classes (sign-thresholded direction)");
        return ExitCode::FAILURE;
    }
    let layers = match resolve_model_layers(model_dir, None) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let layer = layer_override.unwrap_or(spec.layer).min(layers);
    // The probe emits `after_layer_{i}` per layer plus a final-norm checkpoint;
    // index `layers` selects the latter.
    let want_label = if layer >= layers {
        "after_final_norm".to_string()
    } else {
        format!("after_layer_{layer}")
    };

    eprintln!(
        "fit-token-probe: {} samples, layer {} ({}), position {}",
        spec.samples.len(),
        layer,
        want_label,
        spec.position
    );

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(spec.samples.len());
    let mut y: Vec<u8> = Vec::with_capacity(spec.samples.len());
    for (i, s) in spec.samples.iter().enumerate() {
        let Some(class_idx) = spec.class_names.iter().position(|c| *c == s.class) else {
            eprintln!("error: sample {i} has unknown class {:?}", s.class);
            return ExitCode::FAILURE;
        };
        let mut cfg = StepSmokeConfig {
            layers,
            steps: 2,
            kv_len: 0,
            seed,
            max_seq,
            finish: metal::StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: false,
        };
        if let Err(e) = attach_step_prefill(&mut cfg, model_dir, 0, Some(&s.prompt), false) {
            eprintln!("error: sample {i}: {e}");
            return ExitCode::FAILURE;
        }
        let probe = match run_step_layer_hidden_dump(model_dir, &cfg, &s.prompt, spec.position, 0) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: sample {i}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let Some(cp) = probe.checkpoints.iter().find(|c| c.label == want_label) else {
            eprintln!("error: layer {layer} not captured (label {want_label})");
            return ExitCode::FAILURE;
        };
        let mut h = cp.hidden.clone();
        l2_normalise(&mut h);
        rows.push(h);
        y.push(class_idx as u8);
        if (i + 1) % 8 == 0 || i + 1 == spec.samples.len() {
            eprintln!("fit-token-probe: {}/{}", i + 1, spec.samples.len());
        }
    }

    if !y.contains(&0) || !y.contains(&1) {
        eprintln!("error: spec must contain samples of BOTH classes");
        return ExitCode::FAILURE;
    }

    let dim = rows[0].len();
    let mut m1 = vec![0.0f32; dim];
    let mut m0 = vec![0.0f32; dim];
    let (mut n1, mut n0) = (0f32, 0f32);
    for (r, c) in rows.iter().zip(&y) {
        let (m, n) = if *c == 1 {
            (&mut m1, &mut n1)
        } else {
            (&mut m0, &mut n0)
        };
        for (a, b) in m.iter_mut().zip(r.iter()) {
            *a += b;
        }
        *n += 1.0;
    }
    for a in &mut m1 {
        *a /= n1;
    }
    for a in &mut m0 {
        *a /= n0;
    }
    let w: Vec<f32> = m1.iter().zip(&m0).map(|(a, b)| a - b).collect();
    let mid: Vec<f32> = m1.iter().zip(&m0).map(|(a, b)| (a + b) * 0.5).collect();

    let acc = loo_accuracy(&rows, &y);
    let eff = effect_size(&rows, &y);
    println!(
        "fit-token-probe: layer {layer}  leave-one-out accuracy {acc:.3}  effect size {eff:.2}"
    );
    if acc < 0.9 {
        eprintln!(
            "fit-token-probe: WARNING held-out accuracy {acc:.3} is low — this probe will \
             mislabel in use. Try a different --layer, or add samples."
        );
    }

    let out = TokenClassProbe {
        layer,
        class_names: spec.class_names.clone(),
        w,
        mid,
        provenance: format!(
            "fit-token-probe: {} samples from {}, layer {layer}, position {}, seed {seed}; \
             leave-one-out acc {acc:.3}, effect {eff:.2}",
            spec.samples.len(),
            spec_path.display(),
            spec.position,
        ),
    };
    let text = match serde_json::to_string_pretty(&out) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: serialize probe: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(parent) = output.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!("error: create {}: {e}", parent.display());
        return ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::write(output, text) {
        eprintln!("error: write {}: {e}", output.display());
        return ExitCode::FAILURE;
    }
    println!("fit-token-probe: wrote {}", output.display());
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Separable classes score perfectly; the estimator is sane.
    #[test]
    fn loo_and_effect_on_separable_data() {
        let rows = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.95, -0.1],
            vec![-1.0, 0.0],
            vec![-0.9, 0.1],
            vec![-0.95, -0.1],
        ];
        let y = vec![1, 1, 1, 0, 0, 0];
        assert!(loo_accuracy(&rows, &y) > 0.99);
        assert!(effect_size(&rows, &y) > 1.0);
    }

    /// THE regression test for the bug that nearly shipped a fake result:
    /// identical rows carry no signal, so class means differ only by
    /// floating-point summation order. Leave-one-out changes which class loses
    /// a member, making that rounding correlate with the held-out label. The
    /// degenerate guard must refuse rather than report a perfect score.
    #[test]
    fn zero_signal_layer_does_not_leak_the_label() {
        let rows = vec![vec![0.5, 0.5]; 8];
        let y = vec![1, 1, 1, 1, 0, 0, 0, 0];
        assert_eq!(loo_accuracy(&rows, &y), 0.0, "must refuse, not guess");
        assert!(effect_size(&rows, &y) < 1e-6);
    }
}
