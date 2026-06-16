//! Step-1 logit row dumps for MLX parity (post-softcap, pre/post temperature).

use crate::metal::step_kernel::{
    run_step_forward, run_step_layer_hidden_probe, scheduled_temperature, LayerHiddenProbeResult,
    StepFinishMode, StepForwardOutput, StepSmokeConfig, CANVAS, HID, VOCAB,
};
use crate::safetensors::Error;
use crate::sample::{self, SamplerConfig};
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct TopKEntry {
    pub token: u32,
    pub logit_raw: f32,
    pub logit_tempered: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenLogit {
    pub token: u32,
    pub logit_raw: f32,
    pub logit_tempered: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogitsRowDump {
    pub position: usize,
    pub canvas_token: u32,
    /// Final RMSNorm output before tied lm_head (hidden_dim=2816).
    pub hidden: Vec<f32>,
    pub hidden_l2: f32,
    pub hidden_max_abs: f32,
    pub argmax_raw: u32,
    pub argmax_tempered: u32,
    pub logit_raw_at_argmax_tempered: f32,
    pub logit_tempered_at_argmax_tempered: f32,
    pub entropy_raw_nats: f32,
    pub entropy_tempered_nats: f32,
    pub top_k_tempered: Vec<TopKEntry>,
    pub token_logits: Vec<TokenLogit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepLogitsDump {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub initial_canvas_ids: Vec<u32>,
    pub seed: u64,
    pub max_denoise_steps: usize,
    pub layers: usize,
    pub temperature: f32,
    pub entropy_bound: f32,
    pub rows: Vec<LogitsRowDump>,
}

fn row_slice<'a>(logits: &'a [f32], row: usize) -> &'a [f32] {
    let base = row * VOCAB;
    &logits[base..base + VOCAB]
}

fn argmax_row(row: &[f32]) -> (u32, f32) {
    let (idx, &val) = row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    (idx as u32, val)
}

fn row_entropy_nats(row: &[f32]) -> f32 {
    sample::token_entropy(row, 1, VOCAB)[0]
}

fn top_k_tempered(row: &[f32], temperature: f32, k: usize) -> Vec<TopKEntry> {
    let t = temperature.max(1e-6);
    let mut scored: Vec<(u32, f32, f32)> = row
        .iter()
        .enumerate()
        .map(|(i, &raw)| (i as u32, raw, raw / t))
        .collect();
    scored.sort_by(|a, b| {
        b.2
            .partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored
        .into_iter()
        .take(k)
        .map(|(token, raw, tempered)| TopKEntry {
            token,
            logit_raw: raw,
            logit_tempered: tempered,
        })
        .collect()
}

fn hidden_row(out: &StepForwardOutput, position: usize) -> Vec<f32> {
    let base = position * HID;
    out.norm_hidden[base..base + HID].to_vec()
}

fn vec_l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn vec_max_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

fn dump_row(
    out: &StepForwardOutput,
    position: usize,
    temperature: f32,
    top_k: usize,
    watch_tokens: &[u32],
) -> Result<LogitsRowDump, Error> {
    if position >= CANVAS {
        return Err(Error::Format("logits dump position out of range"));
    }
    let row = row_slice(&out.logits, position);
    let t = temperature.max(1e-6);
    let (argmax_raw, _) = argmax_row(row);
    let tempered: Vec<f32> = row.iter().map(|v| v / t).collect();
    let (argmax_tempered, logit_tempered_at_argmax_tempered) = argmax_row(&tempered);
    let logit_raw_at_argmax_tempered = row[argmax_tempered as usize];
    let hidden = hidden_row(out, position);
    let hidden_l2 = vec_l2(&hidden);
    let hidden_max_abs = vec_max_abs(&hidden);
    let mut token_set = std::collections::BTreeSet::new();
    token_set.insert(1);
    token_set.insert(argmax_raw);
    token_set.insert(argmax_tempered);
    for e in top_k_tempered(row, temperature, top_k) {
        token_set.insert(e.token);
    }
    for &tok in watch_tokens {
        token_set.insert(tok);
    }
    let token_logits = token_set
        .into_iter()
        .map(|token| TokenLogit {
            token,
            logit_raw: row[token as usize],
            logit_tempered: row[token as usize] / t,
        })
        .collect();
    Ok(LogitsRowDump {
        position,
        canvas_token: out.token_ids[position],
        hidden,
        hidden_l2,
        hidden_max_abs,
        argmax_raw,
        argmax_tempered,
        logit_raw_at_argmax_tempered,
        logit_tempered_at_argmax_tempered,
        entropy_raw_nats: row_entropy_nats(row),
        entropy_tempered_nats: row_entropy_nats(&tempered),
        top_k_tempered: top_k_tempered(row, temperature, top_k),
        token_logits,
    })
}

pub fn run_step_logits_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    positions: &[usize],
    top_k: usize,
) -> Result<StepLogitsDump, Error> {
    let mut fwd_cfg = cfg.clone();
    fwd_cfg.finish = StepFinishMode::ForwardOnly;
    let out = run_step_forward(model_dir, &fwd_cfg)?;
    build_logits_dump_from_forward(cfg, prompt_label, positions, top_k, out, "rust-monolithic")
}

/// CPU decoder forward with bf16 safetensors (oracle for Q4 monolithic logic).
pub fn run_step_bf16_oracle_logits_dump(
    bf16_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    positions: &[usize],
    top_k: usize,
) -> Result<StepLogitsDump, Error> {
    use crate::model::decoder::{forward, DecoderForwardInput, DecoderScratch};
    use crate::model::encoder::{prefill, EncoderPrefillInput, EncoderScratch};
    use crate::model::mask::DecoderAttnMask;
    use crate::model::Model;
    use crate::metal::step_kernel::init_canvas_state;

    let model = Model::open(bf16_dir)?;
    if model.weights.is_quantized() {
        return Err(Error::Format(
            "bf16 oracle dump requires bf16 safetensors, not .dgq",
        ));
    }
    let prompt = cfg
        .prefill_token_ids
        .as_ref()
        .ok_or_else(|| Error::Format("bf16 oracle dump requires prefill token ids"))?;
    let layers = cfg.layers.min(model.config.text_config.num_hidden_layers).max(1);
    let kv_len = prompt.len();
    let mut enc_scratch = EncoderScratch::new(kv_len.max(CANVAS), &model.config);
    let prefill_out = prefill(
        &model.weights,
        &model.config,
        &EncoderPrefillInput {
            token_ids: prompt,
            position_offset: 0,
        },
        &mut enc_scratch,
    )?;
    let canvas = init_canvas_state(cfg.seed, VOCAB);
    let token_ids: Vec<u32> = canvas.ids.to_vec();
    let mask = DecoderAttnMask::all_valid(CANVAS, kv_len);
    let mut dec_scratch = DecoderScratch::new(CANVAS, &model.config);
    let mut logits = vec![0.0f32; CANVAS * VOCAB];
    let mut input = DecoderForwardInput {
        mask: Some(&mask),
        logits_out: Some(&mut logits),
        self_conditioning_logits: None,
        ..DecoderForwardInput::new(&token_ids, &prefill_out.kv_cache)
    };
    let out = forward(
        &model.weights,
        &model.config,
        &mut input,
        &mut dec_scratch,
        Some(layers),
    )?;
    build_logits_dump_from_forward(
        cfg,
        prompt_label,
        positions,
        top_k,
        StepForwardOutput {
            norm_hidden: out.hidden_states,
            logits,
            token_ids,
        },
        "rust-bf16-oracle",
    )
}

/// CPU bf16 decoder with KV from GPU monolithic encoder (.dgq) — isolates decoder vs KV drift.
pub fn run_step_bf16_oracle_logits_dump_gpu_kv(
    dgq_dir: &Path,
    bf16_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    positions: &[usize],
    top_k: usize,
) -> Result<StepLogitsDump, Error> {
    use crate::model::decoder::{forward, DecoderForwardInput, DecoderScratch};
    use crate::model::mask::DecoderAttnMask;
    use crate::model::Model;
    use crate::metal::step_kernel::{
        build_layout, build_offsets_from_store, init_canvas_state,
    };
    use crate::metal::step_kv::{
        prefill_monolithic_kv_with_cache, read_monolithic_kv_prefix_to_cpu_cache,
        MonolithicEncoderCache, kv_cache_total_bytes,
    };
    use crate::dgq::DgqStore;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLDevice, MTLResourceOptions};

    let model = Model::open(bf16_dir)?;
    if model.weights.is_quantized() {
        return Err(Error::Format(
            "bf16 gpu-kv oracle requires bf16 safetensors, not .dgq",
        ));
    }
    let prompt = cfg
        .prefill_token_ids
        .as_ref()
        .ok_or_else(|| Error::Format("bf16 gpu-kv oracle requires prefill token ids"))?;
    let layers = cfg.layers.min(model.config.text_config.num_hidden_layers).max(1);
    let max_seq = cfg.max_seq.max(prompt.len()).max(CANVAS);
    let dgq_store = DgqStore::open(dgq_dir)?;
    let layout = build_layout(&build_offsets_from_store(&dgq_store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
    let gpu_buf = ctx
        .device
        .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(Error::Format("gpu-kv oracle buffer alloc failed"))?;
    let mut enc_cache =
        MonolithicEncoderCache::open_opt(dgq_dir, CANVAS, max_seq, None)?;
    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut enc_cache,
        prompt,
        &gpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    let kv_cache = read_monolithic_kv_prefix_to_cpu_cache(
        &gpu_buf,
        &layout,
        &model.config,
        kv_len,
    )?;
    let canvas = init_canvas_state(cfg.seed, VOCAB);
    let token_ids: Vec<u32> = canvas.ids.to_vec();
    let mask = DecoderAttnMask::all_valid(CANVAS, kv_len);
    let mut dec_scratch = DecoderScratch::new(CANVAS, &model.config);
    let mut logits = vec![0.0f32; CANVAS * VOCAB];
    let mut input = DecoderForwardInput {
        mask: Some(&mask),
        logits_out: Some(&mut logits),
        self_conditioning_logits: None,
        ..DecoderForwardInput::new(&token_ids, &kv_cache)
    };
    let out = forward(
        &model.weights,
        &model.config,
        &mut input,
        &mut dec_scratch,
        Some(layers),
    )?;
    build_logits_dump_from_forward(
        cfg,
        prompt_label,
        positions,
        top_k,
        StepForwardOutput {
            norm_hidden: out.hidden_states,
            logits,
            token_ids,
        },
        "rust-bf16-oracle-gpu-kv",
    )
}

fn build_logits_dump_from_forward(
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    positions: &[usize],
    top_k: usize,
    out: StepForwardOutput,
    source: &str,
) -> Result<StepLogitsDump, Error> {
    let params = crate::metal::step_kernel::step_params_from_sampler(
        &SamplerConfig {
            max_denoising_steps: cfg.steps.max(1),
            ..SamplerConfig::default()
        },
        cfg.prefill_token_ids
            .as_ref()
            .map(|t| t.len() as u32)
            .unwrap_or(cfg.kv_len),
        cfg.no_early_stop,
        1,
    );
    let temperature = scheduled_temperature(0, &params);
    let rows = positions
        .iter()
        .map(|&pos| dump_row(&out, pos, temperature, top_k, &[]))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StepLogitsDump {
        schema_version: SCHEMA_VERSION,
        source: source.into(),
        prompt: prompt_label.to_string(),
        prompt_token_ids: cfg.prefill_token_ids.clone().unwrap_or_default(),
        initial_canvas_ids: out.token_ids,
        seed: cfg.seed,
        max_denoise_steps: cfg.steps.max(1),
        layers: cfg.layers,
        temperature,
        entropy_bound: params.entropy_bound,
        rows,
    })
}

pub fn write_step_logits_dump(path: &Path, dump: &StepLogitsDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}

pub const LAYER_HIDDEN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct LayerHiddenCheckpoint {
    pub label: String,
    pub layer: Option<usize>,
    pub hidden: Vec<f32>,
    pub hidden_l2: f32,
    pub hidden_max_abs: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepLayerHiddenDump {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub initial_canvas_ids: Vec<u32>,
    pub seed: u64,
    pub layers: usize,
    pub position: usize,
    pub canvas_token: u32,
    pub checkpoints: Vec<LayerHiddenCheckpoint>,
}

fn layer_hidden_dump_from_probe(
    probe: &LayerHiddenProbeResult,
    prompt: &str,
    prompt_token_ids: &[u32],
    seed: u64,
    layers: usize,
) -> StepLayerHiddenDump {
    StepLayerHiddenDump {
        schema_version: LAYER_HIDDEN_SCHEMA_VERSION,
        source: "rust-monolithic".into(),
        prompt: prompt.to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        initial_canvas_ids: probe.token_ids.clone(),
        seed,
        layers,
        position: probe.position,
        canvas_token: probe.canvas_token,
        checkpoints: probe
            .checkpoints
            .iter()
            .map(|cp| LayerHiddenCheckpoint {
                label: cp.label.clone(),
                layer: cp.layer,
                hidden: cp.hidden.clone(),
                hidden_l2: cp.hidden_l2,
                hidden_max_abs: cp.hidden_max_abs,
            })
            .collect(),
    }
}

pub fn run_step_layer_hidden_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    position: usize,
) -> Result<StepLayerHiddenDump, Error> {
    let probe = run_step_layer_hidden_probe(model_dir, cfg, position)?;
    Ok(layer_hidden_dump_from_probe(
        &probe,
        prompt_label,
        cfg.prefill_token_ids.as_deref().unwrap_or(&[]),
        cfg.seed,
        cfg.layers,
    ))
}

pub fn write_step_layer_hidden_dump(path: &Path, dump: &StepLayerHiddenDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}

pub fn parse_positions(spec: &str) -> Result<Vec<usize>, Error> {
    if spec.trim().is_empty() {
        return Ok(vec![0, 43, 58]);
    }
    spec.split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| Error::Format("invalid --positions"))
        })
        .collect()
}
