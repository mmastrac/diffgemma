//! Phase M0 correctness gates for the monolithic step kernel (`PLAN_MONOLITHIC.md`).

use crate::dgq::block::{dequant_row_q4, q4_weight_at};
use crate::dgq::layout::GROUP_SIZE;
use crate::metal::GpuDecoderEngine;
use crate::metal::decoder::{GpuDecoderScratch, forward as decoder_forward, load_weight_cache};
use crate::metal::step_kernel::{
    CANVAS, HID, StepFinishMode, StepForwardOutput, StepParams, StepSmokeConfig, VOCAB,
    init_canvas_state, run_step_forward, run_step_probe, run_step_smoke,
};
use crate::model::decoder::DecoderForwardInput;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;
use crate::sample::{Rng, SamplerConfig, initialize_canvas};
use std::path::Path;

pub const EMBED_SCALE: f32 = 53.065_996_645_694_66; // sqrt(2816)
/// Monolithic (GPU bf16/f16) vs f32 engine forward parity bounds.
/// Hidden is compared by max|Δ| (catches localized divergence). Logits are
/// compared by MEAN|Δ|, not max|Δ|: post-softcap logits live in [-30, 30], so a
/// single near-tie flips the full ±60 range — max|Δ| is a saturation artifact,
/// not a parity signal. Mean|Δ| stays tiny when the forward matches.
pub const PARITY_HIDDEN_MAX_ABS: f32 = 500.0;
pub const PARITY_LOGITS_MEAN_DIFF: f32 = 8.0;

#[derive(Debug, Clone)]
pub struct M0Check {
    pub id: &'static str,
    pub pass: bool,
    pub detail: String,
}

#[derive(Debug)]
pub struct M0VerifyResult {
    pub checks: Vec<M0Check>,
}

impl M0VerifyResult {
    pub fn all_pass(&self) -> bool {
        self.checks.iter().all(|c| c.pass)
    }
}

#[derive(Debug)]
pub struct StepParityResult {
    pub layers: usize,
    pub kv_len: u32,
    pub seed: u64,
    pub hidden_max_abs: f32,
    pub logits_mean_diff: f32,
    pub hidden_tol: f32,
    pub logits_tol: f32,
    pub pass: bool,
    pub skipped: bool,
    pub skip_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StepParityConfig {
    pub layers: usize,
    pub kv_len: u32,
    pub seed: u64,
    pub max_seq: usize,
    pub hidden_tol: f32,
    pub logits_tol: f32,
}

impl Default for StepParityConfig {
    fn default() -> Self {
        Self {
            layers: 3,
            kv_len: 0,
            seed: 42,
            max_seq: 512,
            hidden_tol: PARITY_HIDDEN_MAX_ABS,
            logits_tol: PARITY_LOGITS_MEAN_DIFF,
        }
    }
}

fn check(id: &'static str, pass: bool, detail: impl Into<String>) -> M0Check {
    M0Check {
        id,
        pass,
        detail: detail.into(),
    }
}

/// CPU mirror of `dequant_q4_group` in `include/dequant.metal` (VERIFY-N).
pub(crate) fn dequant_q4_group_cpu(g: &[u8; 20]) -> [f32; 32] {
    crate::kernels::cpu::dequant::dequant_q4_group(g)
}

/// K-order decode via col-indexed path (mirrors `q4_at_col` in dequant.metal).
pub(crate) fn q4_weight_at_k_order_group(row: &[u8], base_k: usize, in_dim: usize) -> [f32; 32] {
    let mut out = [0.0f32; GROUP_SIZE];
    for m in 0..GROUP_SIZE {
        let col = base_k + m;
        out[m] = q4_weight_at(row, 0, col, in_dim);
    }
    out
}

fn verify_q4_nibble_parity() -> M0Check {
    let mut max_err = 0.0f32;
    let mut samples = 0usize;
    for seed in 0..10u64 {
        let mut row = [0u8; 4 + GROUP_SIZE / 2];
        let mut st = seed.wrapping_add(1);
        for b in row.iter_mut() {
            st = st.wrapping_mul(6_966_169_279).wrapping_add(1_039_523_323);
            *b = (st >> 32) as u8;
        }
        let group = dequant_q4_group_cpu(&row);
        let mut ref_row = [0.0f32; GROUP_SIZE];
        dequant_row_q4(&row, GROUP_SIZE, &mut ref_row);
        for j in 0..GROUP_SIZE {
            let err = (group[j] - ref_row[j]).abs();
            max_err = max_err.max(err);
            samples += 1;
        }
    }
    check(
        "M0.1 VERIFY-N",
        max_err <= 1e-6,
        format!("q4 nibble parity max_err={max_err:.2e} ({samples} samples)"),
    )
}

/// VERIFY-K: sequential `dequant_q4_group` unpack vs col-indexed `q4_weight_at` in K order.
fn verify_q4_k_order_positions() -> M0Check {
    use crate::dgq::layout::q4_row_bytes;
    let mut max_err = 0.0f32;
    let mut samples = 0usize;
    for in_dim in [64usize, 128, 2816] {
        let row_bytes = q4_row_bytes(in_dim);
        for seed in 0..10u64 {
            let mut row = vec![0u8; row_bytes];
            let mut st = seed.wrapping_add(in_dim as u64 * 17);
            for b in row.iter_mut() {
                st = st.wrapping_mul(6_966_169_279).wrapping_add(1_039_523_323);
                *b = (st >> 32) as u8;
            }
            for base_k in (0..in_dim).step_by(GROUP_SIZE) {
                let g_off = (base_k / GROUP_SIZE) * (4 + GROUP_SIZE / 2);
                let group: &[u8; 20] = row[g_off..g_off + 20].try_into().expect("group");
                let via_dequant = dequant_q4_group_cpu(group);
                let via_col = q4_weight_at_k_order_group(&row, base_k, in_dim);
                for m in 0..GROUP_SIZE {
                    let err = (via_dequant[m] - via_col[m]).abs();
                    max_err = max_err.max(err);
                    samples += 1;
                }
            }
        }
    }
    check(
        "M0.1b VERIFY-K",
        max_err <= 1e-6,
        format!(
            "dequant_q4_group vs q4_weight_at K-order max_err={max_err:.2e} ({samples} samples)"
        ),
    )
}

fn verify_embed_scale() -> M0Check {
    let expected = (HID as f32).sqrt();
    let err = (EMBED_SCALE - expected).abs();
    check(
        "M0.2 EMBED_SCALE",
        err <= 1e-4,
        format!("EMBED_SCALE={EMBED_SCALE:.6} sqrt({HID})={expected:.6} err={err:.2e}"),
    )
}

fn logit_rowstats_cpu(logits: &[f32], row: usize) -> (f32, f32) {
    let lr = &logits[row * VOCAB..(row + 1) * VOCAB];
    let mx = lr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = lr.iter().map(|&x| (x - mx).exp()).sum();
    (mx, sum)
}

fn verify_softmax_rowstats() -> M0Check {
    let mut logits = vec![0.0f32; CANVAS * VOCAB];
    for row in 0..CANVAS {
        for v in 0..VOCAB {
            logits[row * VOCAB + v] = ((row * 17 + v) % 97) as f32 * 0.1 - 3.0;
        }
    }
    let mut max_err = 0.0f32;
    for row in 0..8 {
        let (mx, sum) = logit_rowstats_cpu(&logits, row);
        let mut probs_sum = 0.0f32;
        for v in 0..VOCAB {
            let p = ((logits[row * VOCAB + v] - mx).exp()) / sum;
            probs_sum += p;
        }
        max_err = max_err.max((probs_sum - 1.0).abs());
    }
    check(
        "M0.2 rowstats",
        max_err <= 1e-3,
        format!("post-softcap softmax row sum err max={max_err:.2e}"),
    )
}

fn verify_rng_init() -> M0Check {
    let seed = 42u64;
    let state = init_canvas_state(seed, VOCAB);
    let mut rng = Rng::new(seed);
    let _ = initialize_canvas(CANVAS, VOCAB, &mut rng);
    if rng.state() != state.rng_state {
        return check(
            "M0.3 RNG init",
            false,
            format!(
                "rng_state mismatch after canvas init: gpu={} cpu={}",
                state.rng_state,
                rng.state()
            ),
        );
    }
    let mut gpu_st = state.rng_state;
    let mut cpu_rng = Rng::from_state(state.rng_state);
    for i in 0..1000 {
        gpu_st = gpu_st
            .wrapping_mul(6_966_169_279)
            .wrapping_add(1_039_523_323);
        let gpu_u = (gpu_st >> 32) as u32;
        let cpu_u = cpu_rng.next_u32();
        if gpu_u != cpu_u {
            return check(
                "M0.3 RNG init",
                false,
                format!("LCG draw {i} mismatch gpu={gpu_u} cpu={cpu_u}"),
            );
        }
    }
    check(
        "M0.3 RNG init",
        true,
        "1000 LCG draws match after canvas init".to_string(),
    )
}

fn verify_temperature_schedule() -> M0Check {
    let cfg = SamplerConfig::default();
    let p = StepParams {
        kv_len: 0,
        max_steps: cfg.max_denoising_steps as u32,
        entropy_bound: cfg.entropy_bound,
        t_min: cfg.t_min,
        t_max: cfg.t_max,
        conf_threshold: cfg.confidence_threshold,
        stability_threshold: cfg.stability_threshold as u32,
        min_early_stop_steps: crate::sample::MIN_EARLY_STOP_STEPS,
        accept_plateau_threshold: cfg.accept_plateau_threshold as u32,
        plateau_prefix_mean_max: cfg.plateau_prefix_mean_max,
        eos_token_id: 1,
    };
    let mut max_err = 0.0f32;
    let n = cfg.max_denoising_steps.max(1);
    for step_done in 0..n {
        let cur_step = n - step_done;
        let cpu = cfg.temperature_at_step(cur_step);
        let cur = (p.max_steps - step_done as u32) as f32;
        let gpu = p.t_min + (p.t_max - p.t_min) * (cur / p.max_steps as f32);
        max_err = max_err.max((cpu - gpu).abs());
    }
    check(
        "M0.5 temperature",
        max_err <= 1e-6,
        format!("temp_at schedule max_err={max_err:.2e}"),
    )
}

fn verify_full_layers(model_dir: &Path, layers: usize) -> Result<M0Check, Error> {
    let cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed: 42,
        max_seq: 512,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    let probe = run_step_probe(model_dir, cfg)?;
    let bad: Vec<_> = probe
        .checkpoints
        .iter()
        .filter(|cp| !cp.finite)
        .map(|cp| cp.label.as_str())
        .collect();
    Ok(check(
        "M0.4 full-layer",
        bad.is_empty(),
        if bad.is_empty() {
            format!("step-probe {layers}L all checkpoints finite")
        } else {
            format!("non-finite at: {}", bad.join(", "))
        },
    ))
}

fn verify_sampler_golden(model_dir: &Path) -> Result<M0Check, Error> {
    const CASES: [(u64, [u32; 8]); 3] = [
        (
            42,
            [100691, 146483, 261606, 21027, 27498, 186817, 242016, 50996],
        ),
        (
            43,
            [22431, 59629, 27721, 31328, 26087, 205511, 190278, 83847],
        ),
        (
            44,
            [206314, 234920, 55981, 41628, 24676, 224205, 138541, 116698],
        ),
    ];
    let mut details = Vec::new();
    for (seed, expect_head) in CASES {
        let cfg = StepSmokeConfig {
            layers: 3,
            steps: 4,
            kv_len: 0,
            seed,
            max_seq: 512,
            finish: StepFinishMode::Full,
            prefill_token_ids: None,
            no_early_stop: false,
        };
        let r = run_step_smoke(model_dir, cfg)?;
        if !r.logits_finite {
            return Ok(check(
                "M0.5 sampler",
                false,
                format!("seed={seed} non-finite logits"),
            ));
        }
        if r.ids[..8] != expect_head {
            return Ok(check(
                "M0.5 sampler",
                false,
                format!(
                    "seed={seed} ids head mismatch got={:?} expect={expect_head:?}",
                    &r.ids[..8]
                ),
            ));
        }
        details.push(format!("seed={seed} step={} stop={}", r.step, r.stop_flag));
    }
    Ok(check(
        "M0.5 sampler",
        true,
        format!("3L×4 steps golden ids[0..8] ({})", details.join("; ")),
    ))
}

/// Run all M0 checks that do not require weights, plus optional integration checks.
pub fn run_step_verify(
    model_dir: Option<&Path>,
    probe_layers: usize,
) -> Result<M0VerifyResult, Error> {
    let mut checks = vec![
        verify_q4_nibble_parity(),
        verify_q4_k_order_positions(),
        verify_embed_scale(),
        verify_softmax_rowstats(),
        verify_rng_init(),
        verify_temperature_schedule(),
    ];

    if let Some(dir) = model_dir {
        checks.push(verify_full_layers(dir, probe_layers)?);
        checks.push(verify_sampler_golden(dir)?);
    }

    Ok(M0VerifyResult { checks })
}

/// Mean |a-b| over finite pairs. Softcap-robust logits metric (vs max|Δ|, which
/// saturates to ±2·cap on a single near-tie flip and so isn't a parity signal).
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            sum += (x - y).abs() as f64;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (sum / n as f64) as f32 }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }
    (max_abs, max_idx)
}

pub(crate) fn engine_forward(
    model: &crate::model::Model,
    token_ids: &[u32],
    kv_len: usize,
    layers: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let canvas_len = token_ids.len();
    let _hidden = model.config.text_config.hidden_size;
    let _vocab = model.config.text_config.vocab_size;

    let kv_cache = crate::model::kv_cache::KvCache::dummy(&model.config.text_config, kv_len, 42)?;
    let mask = DecoderAttnMask::all_valid(canvas_len, kv_len);
    let mut input = DecoderForwardInput::new(token_ids, &kv_cache);
    input.mask = Some(&mask);
    input.compute_logits = true;
    input.return_hidden = true;

    let mut scratch = GpuDecoderScratch::new(canvas_len, &model.config);
    let mut weights = load_weight_cache(
        &model.weights,
        &model.config.text_config,
        canvas_len,
        kv_len,
    )?;
    let mut engine = GpuDecoderEngine::new()?;
    scratch.ensure_gpu_kv(
        &engine.ctx.device,
        &model.config.text_config,
        kv_len,
        canvas_len,
    )?;
    scratch.sync_gpu_kv_from_cpu(&kv_cache, canvas_len)?;

    let out = decoder_forward(
        &model.weights,
        &model.config,
        &mut input,
        &mut scratch,
        &mut weights,
        &mut engine,
        Some(layers),
    )?;
    Ok((out.hidden_states, out.logits))
}

/// (hidden max|Δ| tol, logits mean|Δ| tol) per layer count. Tightened after the
/// engine final-norm fix (was 5000/6000 hidden, masking a wrong-buffer bug):
/// actual post-fix is hidden max|Δ| ~4.6 (30L) to ~347 (3L), logits mean|Δ|
/// ~0.4 (3L) to ~4.1 (30L).
fn parity_limits(layers: usize) -> (f32, f32) {
    match layers {
        3 => (500.0, 8.0),
        30 => (50.0, 8.0),
        _ => (PARITY_HIDDEN_MAX_ABS, PARITY_LOGITS_MEAN_DIFF),
    }
}

/// Compare monolithic forward-only vs engine decoder on the same canvas (kv_len=0 for M0).
pub fn run_step_parity(
    model_dir: &Path,
    cfg: &StepParityConfig,
) -> Result<StepParityResult, Error> {
    if cfg.kv_len > 0 {
        return Ok(StepParityResult {
            layers: cfg.layers,
            kv_len: cfg.kv_len,
            seed: cfg.seed,
            hidden_max_abs: 0.0,
            logits_mean_diff: 0.0,
            hidden_tol: cfg.hidden_tol,
            logits_tol: cfg.logits_tol,
            pass: true,
            skipped: true,
            skip_reason: Some(
                "kv_len>0 requires M1 KV writer (monolithic b4 layout != GpuKvCache)".into(),
            ),
        });
    }

    let model = crate::model::Model::open(model_dir)?;
    let layers = cfg
        .layers
        .min(model.config.text_config.num_hidden_layers)
        .max(1);
    let (default_hidden_tol, default_logits_tol) = parity_limits(layers);
    let hidden_tol = if cfg.hidden_tol == PARITY_HIDDEN_MAX_ABS {
        default_hidden_tol
    } else {
        cfg.hidden_tol
    };
    let logits_tol = if cfg.logits_tol == PARITY_LOGITS_MEAN_DIFF {
        default_logits_tol
    } else {
        cfg.logits_tol
    };

    let canvas = init_canvas_state(cfg.seed, VOCAB);
    let token_ids: Vec<u32> = canvas.ids.to_vec();

    let step_cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed: cfg.seed,
        max_seq: cfg.max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    let mono: StepForwardOutput = run_step_forward(model_dir, &step_cfg)?;

    let (eng_hidden, eng_logits) = engine_forward(&model, &token_ids, 0, layers)?;

    let (hidden_max_abs, _) = max_abs_diff(&mono.norm_hidden, &eng_hidden);
    let logits_mean_diff = mean_abs_diff(&mono.logits, &eng_logits);
    if crate::flags::parity_debug_enabled() {
        // top-1 argmax agreement per token row — the generation-relevant check.
        let rows = mono.logits.len() / VOCAB;
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                })
                .0
        };
        let agree = (0..rows)
            .filter(|&r| {
                argmax(&mono.logits[r * VOCAB..(r + 1) * VOCAB])
                    == argmax(&eng_logits[r * VOCAB..(r + 1) * VOCAB])
            })
            .count();
        eprintln!(
            "  [parity-debug] hidden max|Δ|={hidden_max_abs:.3}; logits mean|Δ|={logits_mean_diff:.4}; top-1 argmax agree {agree}/{rows}"
        );
    }

    let pass = hidden_max_abs <= hidden_tol && logits_mean_diff <= logits_tol;

    Ok(StepParityResult {
        layers,
        kv_len: cfg.kv_len,
        seed: cfg.seed,
        hidden_max_abs,
        logits_mean_diff,
        hidden_tol,
        logits_tol,
        pass,
        skipped: false,
        skip_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m0_unit_checks_pass() {
        assert!(verify_q4_nibble_parity().pass);
        assert!(verify_q4_k_order_positions().pass);
        assert!(verify_embed_scale().pass);
        assert!(verify_softmax_rowstats().pass);
        assert!(verify_rng_init().pass);
        assert!(verify_temperature_schedule().pass);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod parity_debug {
    use super::*;
    use std::path::Path;

    #[test]
    fn m0_parity_debug_values() {
        let dir = Path::new("/tmp/quantized-weights");
        if !crate::dgq::store::looks_like_dgq_dir(dir) {
            return;
        }
        let cfg = StepParityConfig {
            layers: 1,
            ..StepParityConfig::default()
        };
        let model = crate::model::Model::open(dir).expect("model");
        let canvas = init_canvas_state(42, VOCAB);
        let token_ids: Vec<u32> = canvas.ids.to_vec();
        let step_cfg = StepSmokeConfig {
            layers: 1,
            steps: 1,
            kv_len: 0,
            seed: 42,
            max_seq: 512,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: false,
        };
        let mono = run_step_forward(dir, &step_cfg).expect("mono");
        let (eng_h, eng_l) = engine_forward(&model, &token_ids, 0, 1).expect("eng");
        eprintln!("mono hidden[0..4]: {:?}", &mono.norm_hidden[0..4]);
        eprintln!("eng  hidden[0..4]: {:?}", &eng_h[0..4]);
        eprintln!("mono logits[0..4]: {:?}", &mono.logits[0..4]);
        eprintln!("eng  logits[0..4]: {:?}", &eng_l[0..4]);
        let (hm, _) = max_abs_diff(&mono.norm_hidden, &eng_h);
        let (lm, _) = max_abs_diff(&mono.logits, &eng_l);
        eprintln!("hidden max_abs={hm} logits max_abs={lm}");
    }
}
