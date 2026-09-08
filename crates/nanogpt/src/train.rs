//! AdamW training loop plus a finite-difference check of the analytic grads.

use crate::Args;
use crate::backward;
use crate::data::Corpus;
use crate::model::{self, GptConfig, Weights};
use crate::tensor as t;
use dgops::Error;

pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.95;
pub const ADAM_EPS: f32 = 1e-8;
pub const WEIGHT_DECAY: f32 = 0.1;

/// AdamW moments, one pair per parameter tensor in param_list order.
pub struct OptState {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl OptState {
    pub fn new(w: &mut Weights) -> Self {
        let sizes: Vec<usize> = backward::param_list(w).iter().map(|p| p.len()).collect();
        Self {
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
        }
    }

    /// One AdamW step over every parameter tensor.
    pub fn step(
        &mut self,
        w: &mut Weights,
        grads: &mut backward::Grads,
        step: u32,
        lr: f32,
    ) -> Result<(), Error> {
        let params = backward::param_list(w);
        let gs = backward::grad_list(grads);
        assert_eq!(params.len(), gs.len(), "param/grad list mismatch");
        assert_eq!(params.len(), self.m.len(), "optimizer state mismatch");
        for (i, (p, g)) in params.into_iter().zip(gs).enumerate() {
            assert_eq!(p.len(), g.len(), "param/grad length mismatch at {i}");
            let out = t::adamw(
                p,
                g,
                &self.m[i],
                &self.v[i],
                step,
                lr,
                BETA1,
                BETA2,
                ADAM_EPS,
                WEIGHT_DECAY,
            )?;
            let n = p.len();
            p.copy_from_slice(&out[..n]);
            self.m[i].copy_from_slice(&out[n..2 * n]);
            self.v[i].copy_from_slice(&out[2 * n..]);
        }
        Ok(())
    }
}

/// Linear warmup then cosine decay to 10% of the base rate.
fn lr_at(base: f32, step: usize, total: usize) -> f32 {
    let warmup = 50.min(total / 10).max(1);
    if step < warmup {
        return base * step as f32 / warmup as f32;
    }
    let t = (step - warmup) as f32 / (total - warmup).max(1) as f32;
    base * (0.1 + 0.45 * (1.0 + (std::f32::consts::PI * t).cos()))
}

pub fn train(w: &mut Weights, cfg: &GptConfig, corpus: &Corpus, args: &Args) -> Result<(), String> {
    let mut opt = OptState::new(w);
    let mut start = 0usize;
    let mut window_loss = 0.0f32;
    let span = corpus.len() - cfg.block - 1;

    for step in 1..=args.steps {
        let (x, y) = corpus.batch(start, args.batch, cfg.block);
        start = (start + args.batch * cfg.block) % span;
        let cache = model::forward(w, cfg, &x, cfg.block).map_err(|e| e.to_string())?;
        let (loss, mut grads) =
            backward::backward(w, cfg, &cache, &y).map_err(|e| e.to_string())?;
        let lr = lr_at(args.lr, step, args.steps);
        opt.step(w, &mut grads, step as u32, lr)
            .map_err(|e| e.to_string())?;
        window_loss += loss;
        if step % 100 == 0 || step == 1 {
            println!(
                "step {step:5}/{} loss {:.4} (mean over last 100: {:.4}) lr {:.2e}",
                args.steps,
                loss,
                window_loss / 100.0f32.min(step as f32),
                lr
            );
            window_loss = 0.0;
        }
    }
    Ok(())
}

/// Central-difference check of the analytic gradient, as a directional
/// derivative over whole tensors: for a unit direction d, the analytic
/// g.d must equal (L(w + eps d) - L(w - eps d)) / 2 eps. Aggregating over the
/// tensor keeps the signal far above f32 loss resolution, where a per-entry
/// probe of a small gradient is pure rounding noise (and tests the gradient's
/// direction, not just its scale).
pub fn gradcheck(w: &mut Weights, cfg: &GptConfig, corpus: &Corpus) -> Result<(), String> {
    let (x, y) = corpus.batch(0, 1, cfg.block);
    let cache = model::forward(w, cfg, &x, cfg.block).map_err(|e| e.to_string())?;
    let (_, mut grads) = backward::backward(w, cfg, &cache, &y).map_err(|e| e.to_string())?;

    let mut rng = crate::model::Rng::new(99);
    // The error-vs-eps curve separates the two failure modes: truncation grows
    // with eps^2, f32 loss rounding shrinks as 1/eps. The minimum is the real
    // agreement.
    let eps_values = [1e-2f32, 3e-3, 1e-3];
    let mut worst = 0.0f32;

    // 0 = tok_emb, 3 = layer 0 wqkv, last = lm_head.
    let last = {
        let gs = backward::grad_list(&mut grads);
        gs.len() - 1
    };
    for k in [0usize, 3, last] {
        let n = {
            let ps = backward::param_list(w);
            ps[k].len()
        };
        let raw: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        let dir: Vec<f32> = raw.iter().map(|v| v / norm).collect();

        let analytic: f32 = {
            let gs = backward::grad_list(&mut grads);
            gs[k].iter().zip(dir.iter()).map(|(g, d)| g * d).sum()
        };

        for eps in eps_values {
            {
                let mut ps = backward::param_list(w);
                for (p, d) in ps[k].iter_mut().zip(dir.iter()) {
                    *p += eps * d;
                }
            }
            let lp = backward::loss(w, cfg, &x, &y).map_err(|e| e.to_string())?;
            {
                let mut ps = backward::param_list(w);
                for (p, d) in ps[k].iter_mut().zip(dir.iter()) {
                    *p -= 2.0 * eps * d;
                }
            }
            let lm = backward::loss(w, cfg, &x, &y).map_err(|e| e.to_string())?;
            {
                let mut ps = backward::param_list(w);
                for (p, d) in ps[k].iter_mut().zip(dir.iter()) {
                    *p += eps * d;
                }
            }
            let numeric = (lp - lm) / (2.0 * eps);
            let rel = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-8);
            // Gate on the largest step: it is the least noise-dominated, and
            // truncation is negligible for a smooth loss at this scale.
            if eps == 1e-2 {
                worst = worst.max(rel);
            }
            println!(
                "gradcheck: tensor {k:2} ({n:6} entries) eps {eps:.0e} analytic {analytic:+.4e} numeric {numeric:+.4e} rel {rel:.2e}"
            );
        }
    }
    println!("gradcheck: worst relative error {worst:.2e}");
    if worst > 2e-2 {
        return Err(format!(
            "gradient check failed: worst relative error {worst:.2e}"
        ));
    }
    println!("gradcheck: OK");
    Ok(())
}
