//! Entropy-bound block diffusion sampler (CPU).
//!
//! Matches HuggingFace `EntropyBoundSampler` and `LinearTemperatureScheduleLogitsProcessor`.

use crate::kernels::cpu::softmax_rows;

/// `<pad>` token id (Gemma tokenizer).
pub const PAD_TOKEN_ID: u32 = 0;
/// Sentinel id used when logits are invalid (vocab - 1).
pub const FILLER_TOKEN_ID: u32 = 262_143;

/// Minimum denoise steps before pad-aware early stop may fire.
pub const MIN_EARLY_STOP_STEPS: u32 = 12;
/// Minimum non-degenerate argmax positions required (unless `steps` exceeds min steps).
pub const MIN_REAL_ARGMAX_POSITIONS: u32 = 8;

/// True when every canvas position argmax is pad or filler.
pub fn argmax_is_degenerate(argmax: &[u32]) -> bool {
    !argmax.is_empty()
        && argmax
            .iter()
            .all(|&t| t == PAD_TOKEN_ID || t == FILLER_TOKEN_ID)
}

pub fn count_real_argmax_positions(argmax: &[u32]) -> usize {
    argmax
        .iter()
        .filter(|&&t| t != PAD_TOKEN_ID && t != FILLER_TOKEN_ID)
        .count()
}

/// Pad-aware gate for early stop (shared by CPU, engine GPU, monolithic GPU).
pub fn early_stop_allowed(steps_done: u32, argmax: &[u32]) -> bool {
    if argmax_is_degenerate(argmax) {
        return false;
    }
    let real = count_real_argmax_positions(argmax);
    steps_done >= MIN_EARLY_STOP_STEPS || real >= MIN_REAL_ARGMAX_POSITIONS as usize
}

/// Strip pad/filler ids for display-only decode (KV commit still uses full argmax).
pub fn strip_degenerate_token_ids(ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .filter(|&id| id != PAD_TOKEN_ID && id != FILLER_TOKEN_ID)
        .collect()
}

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

/// Denoise steps completed when the loop variable counts down `max..=1`.
pub fn denoise_steps_completed(max_steps: usize, cur_step: usize) -> u32 {
    (max_steps.saturating_sub(cur_step) + 1) as u32
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

/// Per-step sampler diagnostics (monolithic readback of `CanvasState.entropy`).
#[derive(Debug, Clone, Copy)]
pub struct StepEntropyStats {
    pub accept_count: u32,
    /// Positions with H < 0.1 nats (same scale as `entropy_bound`).
    pub low_entropy_positions: u32,
    pub min_entropy: f32,
    pub mean_entropy: f32,
    pub max_entropy: f32,
}

pub fn count_low_entropy_positions(entropies: &[f32], threshold: f32) -> u32 {
    entropies
        .iter()
        .filter(|&&e| e < threshold)
        .count() as u32
}

pub fn step_entropy_stats(entropies: &[f32], accept: &[u32]) -> StepEntropyStats {
    let accept_count = accept.iter().filter(|&&a| a != 0).count() as u32;
    let (min_entropy, max_entropy) = entropies.iter().fold((f32::INFINITY, 0.0f32), |(mn, mx), &e| {
        (mn.min(e), mx.max(e))
    });
    StepEntropyStats {
        accept_count,
        low_entropy_positions: count_low_entropy_positions(entropies, 0.1),
        min_entropy: if entropies.is_empty() {
            0.0
        } else {
            min_entropy
        },
        mean_entropy: mean_entropy(entropies),
        max_entropy,
    }
}

/// HuggingFace `EntropyBoundSampler.accept_canvas` selection mask:
/// accept sorted position `i` when `sum(entropy[0..i-1]) <= bound`.
pub fn accept_mask_from_entropies(entropies: &[f32], entropy_bound: f32) -> Vec<bool> {
    let n = entropies.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        entropies[a]
            .partial_cmp(&entropies[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = vec![false; n];
    let mut prefix = 0.0f32;
    for &idx in &order {
        if prefix <= entropy_bound {
            mask[idx] = true;
            prefix += entropies[idx];
        } else {
            break;
        }
    }
    mask
}

pub fn accept_count_from_entropies(entropies: &[f32], entropy_bound: f32) -> usize {
    accept_mask_from_entropies(entropies, entropy_bound)
        .iter()
        .filter(|&&m| m)
        .count()
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
    let accepted_mask = accept_mask_from_entropies(&ent, entropy_bound);

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

    let accepted_mask = accept_mask_from_entropies(entropies, entropy_bound);

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

    pub fn should_stop(
        &mut self,
        argmax: &[u32],
        processed_logits: &[f32],
        canvas_len: usize,
        vocab_size: usize,
        steps_done: u32,
    ) -> bool {
        let ent = token_entropy(processed_logits, canvas_len, vocab_size);
        self.should_stop_with_entropies(argmax, &ent, steps_done)
    }

    /// Early stop using precomputed per-position entropies (GPU path).
    pub fn should_stop_with_entropies(
        &mut self,
        argmax: &[u32],
        entropies: &[f32],
        steps_done: u32,
    ) -> bool {
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

        stable && confident && early_stop_allowed(steps_done, argmax)
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
    fn accept_hf_bound_accepts_many_low_entropy_positions() {
        let ent = vec![0.005f32; 256];
        assert_eq!(accept_count_from_entropies(&ent, 0.1), 20);
        let ent_high = vec![0.15f32; 256];
        assert_eq!(accept_count_from_entropies(&ent_high, 0.1), 1);
    }

    #[test]
    fn step_entropy_stats_counts_accept_mask() {
        let ent = vec![0.05f32; 4];
        let accept = vec![1u32, 0, 1, 0];
        let s = step_entropy_stats(&ent, &accept);
        assert_eq!(s.accept_count, 2);
        assert!((s.min_entropy - 0.05).abs() < 1e-6);
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

        assert!(!stopper.should_stop(&argmax, &processed, canvas_len, vocab, 1));
        assert!(stopper.should_stop(&argmax, &processed, canvas_len, vocab, MIN_EARLY_STOP_STEPS));
    }

    #[test]
    fn stopper_blocks_degenerate_all_pad_argmax() {
        let canvas_len = 4;
        let mut stopper = StableConfidentStopper::new(0, f32::MAX);
        let argmax = vec![PAD_TOKEN_ID; canvas_len];
        let entropies = vec![0.0f32; canvas_len];
        assert!(!stopper.should_stop_with_entropies(
            &argmax,
            &entropies,
            MIN_EARLY_STOP_STEPS
        ));
    }

    #[test]
    fn stopper_matches_mlx_calgary_late_step() {
        // MLX calgary/seed42 stops at step 29 when mean_H=0.0019 and argmax stable.
        let mut stopper = StableConfidentStopper::new(1, 0.005);
        let argmax = vec![42u32; 256];
        // Rust nvfp4 late-window (~73/256 below 0.1): mean stays ~0.07, must not stop.
        let ent: Vec<f32> = vec![0.05f32; 73]
            .into_iter()
            .chain(std::iter::repeat(0.12f32).take(256 - 73))
            .collect();
        assert!(!stopper.should_stop_with_entropies(&argmax, &ent, MIN_EARLY_STOP_STEPS));
        assert!(!stopper.should_stop_with_entropies(&argmax, &ent, MIN_EARLY_STOP_STEPS + 1));

        stopper.reset();
        let ent = vec![0.001f32; 256];
        assert!(!stopper.should_stop_with_entropies(&argmax, &ent, 12));
        assert!(stopper.should_stop_with_entropies(&argmax, &ent, 13));
    }

    #[test]
    fn early_stop_allowed_requires_steps_or_real_tokens() {
        let degenerate = vec![PAD_TOKEN_ID; 256];
        assert!(!early_stop_allowed(2, &degenerate));

        let mut sparse = vec![PAD_TOKEN_ID; 256];
        for i in 0..4 {
            sparse[i] = 42;
        }
        assert!(!early_stop_allowed(2, &sparse));
        assert!(early_stop_allowed(MIN_EARLY_STOP_STEPS, &sparse));
        assert!(early_stop_allowed(
            2,
            &vec![42u32; MIN_REAL_ARGMAX_POSITIONS as usize]
        ));
    }
}
