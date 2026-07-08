//! End-to-end block diffusion generation (CPU decoder; optional Metal GPU decoder).

use crate::config::ModelConfig;
use crate::model::decoder::{DecoderForwardInput, DecoderScratch};
use crate::model::encoder::extend_prefill;
use crate::model::encoder::{EncoderPrefillInput, EncoderScratch, prefill};
use crate::model::kv_cache::KvCache;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;
use crate::sample::{
    Rng, SamplerConfig, StableConfidentStopper, accept_canvas, apply_temperature, argmax_canvas,
    denoise_steps_completed, initialize_canvas, renoise_canvas, sample_canvas,
};
use crate::weights::WeightStore;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub sampler: SamplerConfig,
    pub max_new_tokens: usize,
    pub seed: u64,
    /// Limit decoder layers (None = full stack). For smoke tests only.
    pub max_layers: Option<usize>,
    /// When true, run every denoise step (disable stable/confident early stop).
    pub no_early_stop: bool,
    /// Parity / golden tests: native Q4 kernels + CPU sampler (deterministic, slower).
    pub deterministic: bool,
    /// Optional label stored in denoise trace JSON.
    pub trace_prompt: Option<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerConfig::default(),
            max_new_tokens: 256,
            seed: 42,
            max_layers: None,
            no_early_stop: false,
            deterministic: false,
            trace_prompt: None,
        }
    }
}

pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub denoise_steps_run: usize,
    pub blocks_committed: usize,
    /// True when generation ended because a committed block emitted an
    /// end-of-turn / EOS token (full-message mode) rather than exhausting the
    /// `max_new_tokens` budget.
    pub stopped_on_eot: bool,
    /// Effective denoise steps per committed block (monolithic path).
    pub block_steps_eff: Vec<u32>,
    /// Accepted positions per step in the last committed block.
    pub last_block_accept_hist: Vec<u32>,
    /// Min per-position entropy (nats) each denoise step in the last block.
    pub last_block_min_entropy_hist: Vec<f32>,
    pub prefill_elapsed: std::time::Duration,
    pub denoise_elapsed: std::time::Duration,
    pub extend_elapsed: std::time::Duration,
    #[cfg(target_os = "macos")]
    pub session_telemetry: crate::metal::SessionTelemetry,
    #[cfg(target_os = "macos")]
    pub denoise_trace: Option<crate::denoise_trace::DenoiseTrace>,
}

#[cfg(target_os = "macos")]
pub fn generate_monolithic_gpu(
    model_dir: &std::path::Path,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    max_seq: usize,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    use crate::metal::{StepGenerateConfig, generate_monolithic, validate_step_model};
    let validated = validate_step_model(model_dir)?;
    let layers = gen_cfg
        .max_layers
        .unwrap_or(validated.num_layers)
        .min(validated.num_layers);
    let mut cfg = StepGenerateConfig::from_generate(
        gen_cfg.seed,
        gen_cfg.max_new_tokens,
        max_seq,
        layers,
        gen_cfg.sampler.clone(),
        gen_cfg.no_early_stop,
    );
    // E6 empty/degenerate-reply canvas re-roll (only when enabled). Detects an
    // empty user-facing reply from the decoded+sanitized committed block.
    cfg.degenerate_reply_check = crate::chat_template::empty_reply_check(model_dir, Vec::new());
    generate_monolithic(model_dir, prompt_token_ids, &cfg, prompt_label)
}

#[cfg(all(test, target_os = "macos"))]
mod gpu_determinism {
    use super::*;
    use crate::metal::{
        GpuDecoderEngine, GpuDecoderScratch, decoder_forward, load_weight_cache, prefill_gpu,
    };
    use crate::model::decoder::DecoderForwardInput;
    use crate::model::encoder::{EncoderPrefillInput, EncoderScratch};
    use crate::model::mask::DecoderAttnMask;
    use crate::sample::{Rng, argmax_canvas, initialize_canvas};
    use crate::weights::WeightStore;

    const DGQ_HELLO_PROMPT: [u32; 1] = [9259];
    const DGQ_MAX_LAYERS: usize = 3;

    fn dgq_fixture_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new("/tmp/quantized-weights");
        if dir.join("model.dgq.json").exists() {
            Some(dir.to_path_buf())
        } else {
            None
        }
    }

    fn hello_gen_cfg() -> GenerateConfig {
        GenerateConfig {
            sampler: crate::sample::sampler_for_steps(1, false),
            max_new_tokens: 256,
            seed: 42,
            max_layers: Some(DGQ_MAX_LAYERS),
            no_early_stop: false,
            deterministic: true,
            trace_prompt: None,
        }
    }

    fn forward_logits_once(
        store: &WeightStore,
        cfg: &ModelConfig,
        weights: &crate::metal::GpuDecoderWeightCache,
        engine: &mut GpuDecoderEngine,
        dec: &mut GpuDecoderScratch,
        canvas_tokens: &[u32],
        kv_cache: &KvCache,
        max_layers: usize,
    ) -> Result<Vec<f32>, Error> {
        dec.use_gpu_sampler = false;
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let mut logits = vec![0.0f32; canvas * vocab];
        let mask = DecoderAttnMask::all_valid(canvas, kv_cache.kv_len);
        let mut input = DecoderForwardInput {
            token_ids: canvas_tokens,
            kv_cache,
            self_conditioning_logits: None,
            mask: Some(&mask),
            logits_out: Some(&mut logits),
            compute_logits: true,
            return_hidden: false,
        };
        decoder_forward(
            store,
            cfg,
            &mut input,
            dec,
            weights,
            engine,
            Some(max_layers),
        )?;
        Ok(logits)
    }

    fn prefill_and_canvas(
        store: &WeightStore,
        cfg: &ModelConfig,
        weights: &mut crate::metal::GpuDecoderWeightCache,
        engine: &mut GpuDecoderEngine,
        enc: &mut EncoderScratch,
        dec: &mut GpuDecoderScratch,
        max_kv: usize,
    ) -> Result<(KvCache, Vec<u32>), Error> {
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let input = EncoderPrefillInput {
            token_ids: &DGQ_HELLO_PROMPT,
            position_offset: 0,
        };
        dec.use_gpu_sampler = false;
        let kv = prefill_gpu(
            store,
            cfg,
            &input,
            enc,
            dec,
            weights,
            engine,
            max_kv,
            canvas,
            Some(DGQ_MAX_LAYERS),
        )?;
        let mut rng = Rng::new(42);
        let canvas_tokens = initialize_canvas(canvas, vocab, &mut rng);
        Ok((kv, canvas_tokens))
    }

    fn first_logit_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .find_map(|(i, (&x, &y))| {
                if x.to_bits() != y.to_bits() {
                    Some((i, x, y))
                } else {
                    None
                }
            })
    }

    /// Same KV + canvas: two decoder forwards back-to-back must match bit-for-bit.
    #[test]
    fn dgq_forward_logits_same_inputs_twice() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights =
            load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
        let mut engine = GpuDecoderEngine::new().expect("engine");
        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) = prefill_and_canvas(
            &store,
            &cfg,
            &mut weights,
            &mut engine,
            &mut enc,
            &mut dec,
            max_kv,
        )
        .expect("prefill");

        let a = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward a");
        let b = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward b");
        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift on repeat: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    /// Isolate engine pool: second forward uses a fresh engine (same dec/kv/canvas).
    #[test]
    fn dgq_forward_logits_fresh_engine_second_pass() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights =
            load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
        let mut engine_a = GpuDecoderEngine::new().expect("engine a");
        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) = prefill_and_canvas(
            &store,
            &cfg,
            &mut weights,
            &mut engine_a,
            &mut enc,
            &mut dec,
            max_kv,
        )
        .expect("prefill");

        let a = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine_a,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward a");

        let mut engine_b = GpuDecoderEngine::new().expect("engine b");
        let b = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine_b,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward b");

        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift with fresh engine: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    /// Second pass repeats prefill (fresh gpu_kv) — if this passes, gpu_kv canvas corruption is the bug.
    #[test]
    fn dgq_forward_logits_fresh_prefill_second_pass() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights =
            load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");

        let mut engine = GpuDecoderEngine::new().expect("engine");
        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) = prefill_and_canvas(
            &store,
            &cfg,
            &mut weights,
            &mut engine,
            &mut enc,
            &mut dec,
            max_kv,
        )
        .expect("prefill a");
        let a = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward a");

        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) = prefill_and_canvas(
            &store,
            &cfg,
            &mut weights,
            &mut engine,
            &mut enc,
            &mut dec,
            max_kv,
        )
        .expect("prefill b");
        let b = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward b");

        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift with fresh prefill: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    /// Fully isolated second chain (fresh weights + engine + scratch).
    #[test]
    fn dgq_forward_logits_fully_isolated_chains() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let run_chain = || -> Vec<f32> {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            let (kv, canvas_tokens) = prefill_and_canvas(
                &store,
                &cfg,
                &mut weights,
                &mut engine,
                &mut enc,
                &mut dec,
                max_kv,
            )
            .expect("prefill");
            forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                DGQ_MAX_LAYERS,
            )
            .expect("forward")
        };

        let a = run_chain();
        let b = run_chain();
        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift isolated chains: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    fn drift_survey_layers(layers: usize, trials: usize) -> std::collections::HashSet<u32> {
        let dgq_dir = dgq_fixture_dir().expect("dgq dir");
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;
        let mut argmax_samples = std::collections::HashSet::new();
        for trial in 0..trials {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            dec.use_gpu_sampler = false;
            let input = EncoderPrefillInput {
                token_ids: &DGQ_HELLO_PROMPT,
                position_offset: 0,
            };
            let kv = prefill_gpu(
                &store,
                &cfg,
                &input,
                &mut enc,
                &mut dec,
                &mut weights,
                &mut engine,
                max_kv,
                canvas,
                Some(layers),
            )
            .expect("prefill");
            let mut rng = Rng::new(42);
            let canvas_tokens = initialize_canvas(canvas, vocab, &mut rng);
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                layers,
            )
            .expect("forward");
            let nan_n = logits.iter().filter(|v| v.is_nan()).count();
            let argmax = argmax_canvas(&logits, canvas, vocab);
            eprintln!(
                "{layers}-layer trial {trial}: nan={nan_n} pos1_argmax={} logit0={}",
                argmax[1], logits[0]
            );
            argmax_samples.insert(argmax[1]);
        }
        argmax_samples
    }

    #[test]
    fn dgq_drift_prefill_vs_decoder_layers() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let dgq_dir = dgq_fixture_dir().unwrap();
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let run = |prefill_layers: usize, decoder_layers: usize| -> u32 {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            dec.use_gpu_sampler = false;
            let input = EncoderPrefillInput {
                token_ids: &DGQ_HELLO_PROMPT,
                position_offset: 0,
            };
            let kv = prefill_gpu(
                &store,
                &cfg,
                &input,
                &mut enc,
                &mut dec,
                &mut weights,
                &mut engine,
                max_kv,
                canvas,
                Some(prefill_layers),
            )
            .expect("prefill");
            let mut rng = Rng::new(42);
            let canvas_tokens = initialize_canvas(canvas, vocab, &mut rng);
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                decoder_layers,
            )
            .expect("forward");
            argmax_canvas(&logits, canvas, vocab)[1]
        };

        let mut a = std::collections::HashSet::new();
        let mut b = std::collections::HashSet::new();
        let mut c = std::collections::HashSet::new();
        let mut d = std::collections::HashSet::new();
        let mut e = std::collections::HashSet::new();
        let mut f = std::collections::HashSet::new();
        for _ in 0..8 {
            a.insert(run(1, 1));
            b.insert(run(2, 1));
            c.insert(run(2, 2));
            d.insert(run(3, 1));
            e.insert(run(3, 2));
            f.insert(run(3, 3));
        }
        eprintln!(
            "prefill1/dec1={} prefill2/dec1={} prefill2/dec2={} prefill3/dec1={} prefill3/dec2={} prefill3/dec3={}",
            a.len(),
            b.len(),
            c.len(),
            d.len(),
            e.len(),
            f.len()
        );
        assert_eq!(a.len(), 1, "1/1 drift");
        assert_eq!(b.len(), 1, "2/1 drift");
        assert_eq!(c.len(), 1, "2/2 drift");
        assert_eq!(d.len(), 1, "3/1 drift");
        assert_eq!(e.len(), 1, "3/2 drift");
        assert_eq!(f.len(), 1, "3/3 drift");
    }

    #[test]
    fn dgq_drift_survey_two_layers() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let unique = drift_survey_layers(2, 8);
        eprintln!("2-layer unique pos1 argmax: {}", unique.len());
        assert_eq!(unique.len(), 1, "2-layer drift: {unique:?}");
    }

    #[test]
    fn dgq_drift_survey_one_layer() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let unique = drift_survey_layers(1, 8);
        eprintln!("1-layer unique pos1 argmax: {}", unique.len());
        assert_eq!(unique.len(), 1, "1-layer drift: {unique:?}");
    }

    #[test]
    fn dgq_drift_survey_three_layers() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let unique = drift_survey_layers(3, 8);
        eprintln!("3-layer unique pos1 argmax: {}", unique.len());
        assert_eq!(unique.len(), 1, "3-layer drift: {unique:?}");
    }

    /// Full prefill + forward repeated in-process (reused engine pool).
    #[test]
    fn dgq_forward_logits_repeatable_with_reused_engine() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights =
            load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
        let mut engine = GpuDecoderEngine::new().expect("engine");

        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) = prefill_and_canvas(
            &store,
            &cfg,
            &mut weights,
            &mut engine,
            &mut enc,
            &mut dec,
            max_kv,
        )
        .expect("prefill baseline");
        let baseline = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("baseline forward");
        let argmax1 = argmax_canvas(&baseline, canvas, vocab)[1];

        for trial in 1..=20 {
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            let (kv_t, canvas_t) = prefill_and_canvas(
                &store,
                &cfg,
                &mut weights,
                &mut engine,
                &mut enc,
                &mut dec,
                max_kv,
            )
            .expect("prefill");
            assert_eq!(
                canvas_t, canvas_tokens,
                "canvas init drift at trial {trial}"
            );
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_t,
                &kv_t,
                DGQ_MAX_LAYERS,
            )
            .expect("forward");
            if logits != baseline {
                let argmax_t = argmax_canvas(&logits, canvas, vocab)[1];
                let detail = first_logit_diff(&baseline, &logits)
                    .map(|(i, x, y)| format!("flat idx {i}: {x} vs {y}"))
                    .unwrap_or_default();
                panic!(
                    "forward drift trial {trial}: pos1 argmax {argmax1} vs {argmax_t}; {detail}"
                );
            }
        }
    }

    /// Monolithic generate should be repeatable in deterministic mode.
    #[test]
    fn dgq_generate_stable_deterministic_mode() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let prompt = DGQ_HELLO_PROMPT.to_vec();
        let gen_cfg = hello_gen_cfg();
        let max_seq = prompt.len() + gen_cfg.max_new_tokens;
        let out_a = generate_monolithic_gpu(&dgq_dir, &prompt, &gen_cfg, max_seq, "hello")
            .expect("generate a");
        let out_b = generate_monolithic_gpu(&dgq_dir, &prompt, &gen_cfg, max_seq, "hello")
            .expect("generate b");
        if out_a.token_ids != out_b.token_ids {
            let idx = out_a
                .token_ids
                .iter()
                .zip(out_b.token_ids.iter())
                .position(|(a, b)| a != b);
            panic!(
                "token drift at index {idx:?}: a={:?} b={:?}",
                out_a.token_ids.get(1),
                out_b.token_ids.get(1)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_config_defaults_match_model_card() {
        let cfg = GenerateConfig::default();
        assert_eq!(cfg.sampler.entropy_bound, 0.1);
        assert_eq!(cfg.sampler.max_denoising_steps, 48);
        assert!((cfg.sampler.t_max - 0.8).abs() < 1e-6);
        assert!((cfg.sampler.t_min - 0.4).abs() < 1e-6);
        assert_eq!(cfg.sampler.confidence_threshold, 0.005);
    }
}
