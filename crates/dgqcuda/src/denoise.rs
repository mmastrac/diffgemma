//! The denoise loop: the entropy-bound block-diffusion sampler from
//! `src/sample.rs` / the engine's GPU sampler, with the model forward left to
//! the caller. Every decision here is a host-side port of the CPU oracle — the
//! GPU forward only produces the canvas logits.

/// `<pad>` token id (Gemma tokenizer).
pub const PAD_TOKEN_ID: u32 = 0;
/// Sentinel id used when logits are invalid (vocab - 1).
pub const FILLER_TOKEN_ID: u32 = 262_143;
/// Fixed canvas slots (matches `DGQ_SAMPLER_MAX_CANVAS`).
pub const SAMPLER_CANVAS: usize = 256;
/// Ring slots for the argmax canvas history.
pub const ARGMAX_HIST_MAX: usize = 8;
/// Consecutive steps with an identical accept mask before the plateau stop.
pub const ACCEPT_PLATEAU_THRESHOLD: usize = 8;
/// Minimum denoise steps before a confident/plateau early stop may fire.
pub const MIN_EARLY_STOP_STEPS: usize = 12;
/// Plateau backstop also needs the mean entropy below this.
pub const PLATEAU_MAX_PREFIX_MEAN: f32 = 0.05;

/// The LCG the engine's sampler uses (`src/sample.rs::Rng`).
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_966_169_279)
            .wrapping_add(1_039_523_323);
        (self.state >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        const INV: f32 = 1.0 / 4_294_967_296.0;
        self.next_u32() as f32 * INV
    }

    pub fn uniform_below(&mut self, high: u32) -> u32 {
        if high == 0 { 0 } else { self.next_u32() % high }
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
    pub accept_plateau_threshold: usize,
    pub plateau_prefix_mean_max: f32,
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
            accept_plateau_threshold: ACCEPT_PLATEAU_THRESHOLD,
            plateau_prefix_mean_max: PLATEAU_MAX_PREFIX_MEAN,
        }
    }
}

impl SamplerConfig {
    /// Temperature at denoising step `cur_step` (counts down to 1).
    pub fn temperature_at_step(&self, cur_step: usize) -> f32 {
        let n = self.max_denoising_steps.max(1) as f32;
        let step = cur_step as f32;
        self.t_min + (self.t_max - self.t_min) * (step / n)
    }
}

/// Per-step diagnostics, mirroring the engine's `StepEntropyStats`.
#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    pub step: usize,
    pub temperature: f32,
    pub accept_count: usize,
    pub mean_entropy: f32,
    pub min_entropy: f32,
    pub max_entropy: f32,
    pub changed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Argmax stable for `stability_threshold` steps and mean entropy under
    /// `confidence_threshold`.
    Confident,
    /// The accept mask repeated for `accept_plateau_threshold` steps.
    Plateau,
    /// `max_denoising_steps` reached.
    MaxSteps,
}

pub fn initialize_canvas(canvas_len: usize, vocab_size: usize, rng: &mut Rng) -> Vec<u32> {
    let vocab = vocab_size.max(1) as u32;
    (0..canvas_len).map(|_| rng.uniform_below(vocab)).collect()
}

/// Per-position natural-log entropy of the tempered logits, the argmax, and
/// the row softmax max/sum. Mirrors `sample_rowstats.metal`.
pub struct RowStats {
    pub entropy: Vec<f32>,
    pub argmax: Vec<u32>,
}

pub fn row_stats(logits: &[f32], rows: usize, cols: usize, t: f32) -> RowStats {
    let mut entropy = vec![0.0f32; rows];
    let mut argmax = vec![0u32; rows];
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let mut mx = f32::NEG_INFINITY;
        let mut am = 0usize;
        let mut amv = f32::NEG_INFINITY;
        for (v, &lg) in row.iter().enumerate() {
            let x = lg / t;
            if x > amv {
                amv = x;
                am = v;
            }
            if x > mx {
                mx = x;
            }
        }
        let mut z = 0.0f32;
        let mut acc = 0.0f32;
        for &lg in row {
            let x = lg / t;
            let e = (x - mx).exp();
            z += e;
            acc += e * (x - mx);
        }
        entropy[r] = z.ln() - acc / z;
        argmax[r] = am as u32;
    }
    RowStats { entropy, argmax }
}

/// HuggingFace/MLX `EntropyBoundSampler`: sort by entropy ascending, accept
/// while the running prefix sum is `<= entropy_bound` (the first is always
/// accepted).
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
        if prefix > entropy_bound {
            break;
        }
        mask[idx] = true;
        prefix += entropies[idx];
    }
    mask
}

pub fn accept_mask_sig(accept: &[bool]) -> u32 {
    accept
        .iter()
        .fold(0u32, |h, &a| h.wrapping_mul(31).wrapping_add(u32::from(a)))
}

fn is_active_token(id: u32) -> bool {
    id != PAD_TOKEN_ID && id != FILLER_TOKEN_ID
}

fn argmax_is_degenerate(argmax: &[u32]) -> bool {
    argmax.iter().all(|&t| !is_active_token(t))
}

/// The loop's state: canvas ids, the argmax history ring, and the counters the
/// early-stop rules read.
pub struct DenoiseState {
    pub cfg: SamplerConfig,
    pub rng: Rng,
    pub ids: Vec<u32>,
    pub argmax: Vec<u32>,
    pub entropy: Vec<f32>,
    pub accept: Vec<bool>,
    pub u_cat: Vec<f32>,
    pub step: usize,
    argmax_hist: Vec<u32>,
    argmax_hist_len: u32,
    argmax_hist_base: u32,
    prev_accept_sig: Option<u32>,
    accept_plateau: u32,
}

impl DenoiseState {
    pub fn new(cfg: SamplerConfig, seed: u64, canvas: usize, vocab: usize) -> Self {
        let mut rng = Rng::new(seed);
        let ids = initialize_canvas(canvas, vocab, &mut rng);
        Self {
            cfg,
            rng,
            ids,
            argmax: vec![0; canvas],
            entropy: vec![0.0; canvas],
            accept: vec![false; canvas],
            u_cat: vec![0.0; canvas],
            step: 0,
            argmax_hist: vec![0; ARGMAX_HIST_MAX * SAMPLER_CANVAS],
            argmax_hist_len: 0,
            argmax_hist_base: 0,
            prev_accept_sig: None,
            accept_plateau: 0,
        }
    }

    /// Consume one step's logits (`[canvas, vocab]`, pre-softcap): compute the
    /// row stats, draw the categorical uniforms, decide the accept mask, write
    /// accepted argmaxes and re-noise the rest, and update the early-stop
    /// state. Returns the step's diagnostics and the stop reason, if any.
    pub fn step(&mut self, logits: &[f32], vocab: usize) -> (StepStats, Option<StopReason>) {
        let canvas = self.ids.len();
        // 1-based step index, like the engine's `S.step` after the increment.
        let cur_step = self.step + 1;
        // The schedule counts down: step 1 is the hottest (`t_max`), the last
        // step is `t_min` (engine `sample_rowstats` reads `S.step` before the
        // increment).
        let t = self
            .cfg
            .temperature_at_step(self.cfg.max_denoising_steps + 1 - cur_step);
        let stats = row_stats(logits, canvas, vocab, t);
        self.entropy.copy_from_slice(&stats.entropy);
        let prev_ids = self.ids.clone();
        self.argmax.copy_from_slice(&stats.argmax);

        // Categorical uniforms, drawn before the accept decision (the engine
        // draws them in `sample_commit` before using them).
        for u in self.u_cat.iter_mut() {
            *u = self.rng.next_f32();
        }

        // Accept mask. On the final step every position commits, so the loop
        // cannot end on a half-denoised canvas.
        let final_step = cur_step >= self.cfg.max_denoising_steps;
        if final_step {
            self.accept.iter_mut().for_each(|a| *a = true);
        } else {
            let mask = accept_mask_from_entropies(&self.entropy, self.cfg.entropy_bound);
            self.accept.copy_from_slice(&mask);
        }

        // Plateau: consecutive identical accept masks.
        let sig = accept_mask_sig(&self.accept);
        if Some(sig) == self.prev_accept_sig {
            self.accept_plateau += 1;
        } else {
            self.accept_plateau = 0;
        }
        self.prev_accept_sig = Some(sig);

        let mean_entropy = self.entropy.iter().sum::<f32>() / canvas.max(1) as f32;
        // Canvas stability: the argmax is unchanged over the history ring.
        let canvas_stable = self.canvas_stable(canvas);
        self.push_argmax_hist(canvas);

        // Commit accepted rows, re-noise the rest.
        let mut changed = 0usize;
        for i in 0..canvas {
            let next = if self.accept[i] {
                self.argmax[i]
            } else {
                self.rng.uniform_below(vocab.max(1) as u32)
            };
            if next != prev_ids[i] {
                changed += 1;
            }
            self.ids[i] = next;
        }
        self.step = cur_step;

        // Early stop needs a floor on the steps taken, so a degenerate or
        // barely-denoised canvas cannot stop the loop early.
        let degenerate = argmax_is_degenerate(&self.argmax);
        let early_ok = cur_step >= MIN_EARLY_STOP_STEPS && !degenerate;
        let confident = early_ok && canvas_stable && mean_entropy < self.cfg.confidence_threshold;
        let plateau = early_ok
            && self.accept_plateau >= self.cfg.accept_plateau_threshold as u32
            && mean_entropy < self.cfg.plateau_prefix_mean_max;
        let stop = if confident {
            Some(StopReason::Confident)
        } else if plateau {
            Some(StopReason::Plateau)
        } else if cur_step >= self.cfg.max_denoising_steps {
            Some(StopReason::MaxSteps)
        } else {
            None
        };
        let (mut mn, mut mx) = (f32::INFINITY, 0.0f32);
        for &e in &self.entropy {
            mn = mn.min(e);
            mx = mx.max(e);
        }
        (
            StepStats {
                step: cur_step,
                temperature: t,
                accept_count: self.accept.iter().filter(|&&a| a).count(),
                mean_entropy,
                min_entropy: mn,
                max_entropy: mx,
                changed,
            },
            stop,
        )
    }

    fn canvas_stable(&self, canvas: usize) -> bool {
        let thresh = self.cfg.stability_threshold;
        if thresh == 0 {
            return true;
        }
        if self.argmax_hist_len as usize != thresh {
            return false;
        }
        let ring_cap = thresh.min(ARGMAX_HIST_MAX);
        for s in 0..thresh {
            let slot = ((self.argmax_hist_base as usize + s) % ring_cap) * SAMPLER_CANVAS;
            for i in 0..canvas {
                if self.argmax[i] != self.argmax_hist[slot + i] {
                    return false;
                }
            }
        }
        true
    }

    fn push_argmax_hist(&mut self, canvas: usize) {
        let thresh = self.cfg.stability_threshold;
        if thresh == 0 {
            return;
        }
        let ring_cap = thresh.min(ARGMAX_HIST_MAX);
        if (self.argmax_hist_len as usize) < thresh {
            let slot = ((self.argmax_hist_base as usize + self.argmax_hist_len as usize)
                % ring_cap)
                * SAMPLER_CANVAS;
            self.argmax_hist[slot..slot + canvas].copy_from_slice(&self.argmax[..canvas]);
            self.argmax_hist_len += 1;
        } else {
            let slot = (self.argmax_hist_base as usize % ring_cap) * SAMPLER_CANVAS;
            self.argmax_hist[slot..slot + canvas].copy_from_slice(&self.argmax[..canvas]);
            self.argmax_hist_base = ((self.argmax_hist_base as usize + 1) % ring_cap) as u32;
        }
    }
}
