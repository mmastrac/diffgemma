//! Entropy-bound block diffusion sampler (CPU).
//!
//! Matches HuggingFace `EntropyBoundSampler` and `LinearTemperatureScheduleLogitsProcessor`.

use crate::kernels::cpu::softmax_rows;

/// Simple deterministic PRNG (LCG); same family as `KvCache::dummy`.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Resume LCG from a stored state (matches `CanvasState.rng_state` after canvas init).
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_966_169_279)
            .wrapping_add(1_039_523_323);
        (self.state >> 32) as u32
    }

    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        const INV: f32 = 1.0 / 4_294_967_296.0;
        self.next_u32() as f32 * INV
    }

    pub fn uniform_below(&mut self, high: u32) -> u32 {
        if high == 0 {
            0
        } else {
            self.next_u32() % high
        }
    }

    pub fn state(&self) -> u64 {
        self.state
    }
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub entropy_bound: f32,
    pub max_denoising_steps: usize,
    pub t_min: f32,
    pub t_max: f32,
    pub stability_threshold: usize,
    pub confidence_threshold: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            entropy_bound: 0.1,
            max_denoising_steps: 48,
            t_min: 0.4,
            t_max: 0.8,
            stability_threshold: 1,
            confidence_threshold: 0.005,
        }
    }
}

/// Build sampler config for a run; `no_early_stop` forces all `steps` denoise iterations.
pub fn sampler_for_steps(steps: usize, no_early_stop: bool) -> SamplerConfig {
    let mut cfg = SamplerConfig::default();
    cfg.max_denoising_steps = steps.max(1);
    if no_early_stop {
        cfg.confidence_threshold = f32::MAX;
        cfg.stability_threshold = usize::MAX;
    }
    cfg
}

impl SamplerConfig {
    /// Temperature at denoising step `cur_step` (counts down from `max_denoising_steps` to 1).
    pub fn temperature_at_step(&self, cur_step: usize) -> f32 {
        let n = self.max_denoising_steps.max(1) as f32;
        let step = cur_step as f32;
        self.t_min + (self.t_max - self.t_min) * (step / n)
    }
}

pub fn initialize_canvas(canvas_len: usize, vocab_size: usize, rng: &mut Rng) -> Vec<u32> {
    let vocab = vocab_size.max(1) as u32;
    (0..canvas_len)
        .map(|_| rng.uniform_below(vocab))
        .collect()
}

pub fn apply_temperature(logits: &mut [f32], cur_step: usize, cfg: &SamplerConfig) {
    let t = cfg.temperature_at_step(cur_step).max(1e-6);
    for v in logits.iter_mut() {
        *v /= t;
    }
}

/// Per-position categorical entropy from raw logits (before temperature).
pub fn token_entropy(logits: &[f32], canvas_len: usize, vocab_size: usize) -> Vec<f32> {
    let mut probs = logits.to_vec();
    softmax_rows(&mut probs, canvas_len, vocab_size);
    let mut out = vec![0.0f32; canvas_len];
    for pos in 0..canvas_len {
        let row = &probs[pos * vocab_size..(pos + 1) * vocab_size];
        let mut h = 0.0f32;
        for &p in row {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        out[pos] = h;
    }
    out
}

pub fn mean_entropy(entropies: &[f32]) -> f32 {
    if entropies.is_empty() {
        0.0
    } else {
        entropies.iter().sum::<f32>() / entropies.len() as f32
    }
}

/// Accept lowest-entropy positions until the entropy-bound constraint is met.
pub fn accept_canvas(
    current: &[u32],
    denoiser: &[u32],
    processed_logits: &[f32],
    canvas_len: usize,
    vocab_size: usize,
    entropy_bound: f32,
) -> (Vec<u32>, Vec<bool>) {
    assert_eq!(current.len(), canvas_len);
    assert_eq!(denoiser.len(), canvas_len);

    let ent = token_entropy(processed_logits, canvas_len, vocab_size);
    let mut order: Vec<usize> = (0..canvas_len).collect();
    order.sort_by(|&a, &b| ent[a].partial_cmp(&ent[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut accepted_mask = vec![false; canvas_len];
    let mut prefix_sum = 0.0f32;
    for &idx in &order {
        if prefix_sum <= entropy_bound {
            accepted_mask[idx] = true;
            prefix_sum += ent[idx];
        } else {
            break;
        }
    }

    let mut accepted = current.to_vec();
    for i in 0..canvas_len {
        if accepted_mask[i] {
            accepted[i] = denoiser[i];
        }
    }
    (accepted, accepted_mask)
}

/// Accept lowest-entropy positions using precomputed per-position entropies.
pub fn accept_canvas_from_entropies(
    current: &[u32],
    denoiser: &[u32],
    entropies: &[f32],
    canvas_len: usize,
    entropy_bound: f32,
) -> (Vec<u32>, Vec<bool>) {
    assert_eq!(current.len(), canvas_len);
    assert_eq!(denoiser.len(), canvas_len);
    assert_eq!(entropies.len(), canvas_len);

    let mut order: Vec<usize> = (0..canvas_len).collect();
    order.sort_by(|&a, &b| {
        entropies[a]
            .partial_cmp(&entropies[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut accepted_mask = vec![false; canvas_len];
    let mut prefix_sum = 0.0f32;
    for &idx in &order {
        if prefix_sum <= entropy_bound {
            accepted_mask[idx] = true;
            prefix_sum += entropies[idx];
        } else {
            break;
        }
    }

    let mut accepted = current.to_vec();
    for i in 0..canvas_len {
        if accepted_mask[i] {
            accepted[i] = denoiser[i];
        }
    }
    (accepted, accepted_mask)
}

pub fn renoise_canvas(
    accepted: &[u32],
    accepted_mask: &[bool],
    vocab_size: usize,
    rng: &mut Rng,
) -> Vec<u32> {
    let mut out = accepted.to_vec();
    for (i, slot) in out.iter_mut().enumerate() {
        if !accepted_mask[i] {
            *slot = rng.uniform_below(vocab_size.max(1) as u32);
        }
    }
    out
}

pub fn argmax_canvas(logits: &[f32], canvas_len: usize, vocab_size: usize) -> Vec<u32> {
    let mut out = vec![0u32; canvas_len];
    for pos in 0..canvas_len {
        let row = &logits[pos * vocab_size..(pos + 1) * vocab_size];
        let (best_idx, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        out[pos] = best_idx as u32;
    }
    out
}

pub fn sample_canvas(
    logits: &mut [f32],
    canvas_len: usize,
    vocab_size: usize,
    rng: &mut Rng,
) -> Vec<u32> {
    softmax_rows(logits, canvas_len, vocab_size);
    let mut out = vec![0u32; canvas_len];
    for pos in 0..canvas_len {
        let row = &logits[pos * vocab_size..(pos + 1) * vocab_size];
        let r = rng.next_f32();
        let mut cum = 0.0f32;
        let mut chosen = 0u32;
        for (i, &p) in row.iter().enumerate() {
            cum += p;
            if r < cum {
                chosen = i as u32;
                break;
            }
        }
        out[pos] = chosen;
    }
    out
}

/// Adaptive stopping: mean entropy below threshold and argmax stable across prior steps.
#[derive(Debug)]
pub struct StableConfidentStopper {
    stability_threshold: usize,
    confidence_threshold: f32,
    argmax_history: Vec<Vec<u32>>,
}

impl StableConfidentStopper {
    pub fn new(stability_threshold: usize, confidence_threshold: f32) -> Self {
        Self {
            stability_threshold,
            confidence_threshold,
            argmax_history: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.argmax_history.clear();
    }

    pub fn should_stop(&mut self, argmax: &[u32], processed_logits: &[f32], canvas_len: usize, vocab_size: usize) -> bool {
        let ent = token_entropy(processed_logits, canvas_len, vocab_size);
        self.should_stop_with_entropies(argmax, &ent)
    }

    /// Early stop using precomputed per-position entropies (GPU path).
    pub fn should_stop_with_entropies(&mut self, argmax: &[u32], entropies: &[f32]) -> bool {
        let confident = mean_entropy(entropies) < self.confidence_threshold;

        let stable = if self.stability_threshold == 0 {
            true
        } else if self.argmax_history.len() < self.stability_threshold {
            self.argmax_history.push(argmax.to_vec());
            false
        } else {
            let all_match = self.argmax_history.iter().all(|prev| prev == argmax);
            self.argmax_history.remove(0);
            self.argmax_history.push(argmax.to_vec());
            all_match
        };

        stable && confident
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_logits(canvas_len: usize, vocab: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; canvas_len * vocab];
        for pos in 0..canvas_len {
            logits[pos * vocab + (pos % vocab)] = 3.0;
            logits[pos * vocab + ((pos + 1) % vocab)] = 1.0;
        }
        logits
    }

    #[test]
    fn temperature_schedule_endpoints() {
        let cfg = SamplerConfig::default();
        assert!((cfg.temperature_at_step(48) - 0.8).abs() < 1e-5);
        assert!((cfg.temperature_at_step(1) - 0.4083333).abs() < 1e-4);
    }

    #[test]
    fn accept_canvas_respects_entropy_bound() {
        let canvas_len = 4;
        let vocab = 8;
        let logits = toy_logits(canvas_len, vocab);
        let mut processed = logits.clone();
        apply_temperature(&mut processed, 48, &SamplerConfig::default());
        let current = vec![0, 1, 2, 3];
        let denoiser = vec![4, 5, 6, 7];
        let (accepted, mask) = accept_canvas(
            &current,
            &denoiser,
            &processed,
            canvas_len,
            vocab,
            0.5,
        );
        assert!(mask.iter().filter(|&&m| m).count() >= 1);
        for i in 0..canvas_len {
            if mask[i] {
                assert_eq!(accepted[i], denoiser[i]);
            } else {
                assert_eq!(accepted[i], current[i]);
            }
        }
    }

    #[test]
    fn argmax_and_sample_deterministic() {
        let canvas_len = 2;
        let vocab = 4;
        let logits = toy_logits(canvas_len, vocab);
        let mut processed = logits.clone();
        apply_temperature(&mut processed, 10, &SamplerConfig::default());
        let argmax = argmax_canvas(&processed, canvas_len, vocab);
        assert_eq!(argmax, vec![0, 1]);

        let mut rng = Rng::new(42);
        let mut probs = processed.clone();
        let sample1 = sample_canvas(&mut probs, canvas_len, vocab, &mut rng);
        let mut rng2 = Rng::new(42);
        let mut probs2 = processed.clone();
        let sample2 = sample_canvas(&mut probs2, canvas_len, vocab, &mut rng2);
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn stopper_requires_confidence_and_stability() {
        let canvas_len = 2;
        let vocab = 4;
        let mut stopper = StableConfidentStopper::new(1, 0.5);
        let logits = toy_logits(canvas_len, vocab);
        let mut processed = logits.clone();
        apply_temperature(&mut processed, 48, &SamplerConfig::default());
        let argmax = argmax_canvas(&processed, canvas_len, vocab);

        assert!(!stopper.should_stop(&argmax, &processed, canvas_len, vocab));
        assert!(stopper.should_stop(&argmax, &processed, canvas_len, vocab));
    }
}
