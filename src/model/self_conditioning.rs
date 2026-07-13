use crate::Error;
use crate::config::TextConfig;
use crate::shaders::cpu::{gelu_pytorch_tanh, linear, rms_norm_no_scale, rms_norm_rows};
use crate::tensor::TensorView;
use crate::weights::WeightStore;

const SC_PRE_NORM: &str = "model.decoder.self_conditioning.pre_norm.weight";
const SC_GATE: &str = "model.decoder.self_conditioning.gate_proj.weight";
const SC_UP: &str = "model.decoder.self_conditioning.up_proj.weight";
const SC_DOWN: &str = "model.decoder.self_conditioning.down_proj.weight";

pub struct SelfConditioningWeights<'a> {
    pub pre_norm: TensorView<'a>,
    pub gate_proj: TensorView<'a>,
    pub up_proj: TensorView<'a>,
    pub down_proj: TensorView<'a>,
}

impl<'a> SelfConditioningWeights<'a> {
    pub fn load(store: &'a WeightStore, cfg: &TextConfig) -> Result<Self, Error> {
        let hidden = cfg.hidden_size as i64;
        let inter = cfg.intermediate_size as i64;
        let pre_norm = store.tensor(SC_PRE_NORM)?;
        pre_norm.expect_shape(&[hidden])?;
        let gate_proj = store.tensor(SC_GATE)?;
        gate_proj.expect_shape(&[inter, hidden])?;
        let up_proj = store.tensor(SC_UP)?;
        up_proj.expect_shape(&[inter, hidden])?;
        let down_proj = store.tensor(SC_DOWN)?;
        down_proj.expect_shape(&[hidden, inter])?;
        Ok(Self {
            pre_norm,
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

pub struct SelfConditioningScratch {
    pub normed: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub signal: Vec<f32>,
    pub pre_norm_w: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub up_w: Vec<f32>,
    pub down_w: Vec<f32>,
}

impl SelfConditioningScratch {
    pub fn new(seq_len: usize, cfg: &TextConfig) -> Self {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Self {
            normed: vec![0.0; seq_len * hidden],
            gate: vec![0.0; seq_len * inter],
            up: vec![0.0; seq_len * inter],
            signal: vec![0.0; seq_len * hidden],
            pre_norm_w: Vec::new(),
            gate_w: Vec::new(),
            up_w: Vec::new(),
            down_w: Vec::new(),
        }
    }

    fn load_weights(&mut self, weights: &SelfConditioningWeights<'_>) -> Result<(), Error> {
        self.pre_norm_w = weights.pre_norm.bf16()?.to_f32_vec();
        self.gate_w = weights.gate_proj.bf16()?.to_f32_vec();
        self.up_w = weights.up_proj.bf16()?.to_f32_vec();
        self.down_w = weights.down_proj.bf16()?.to_f32_vec();
        Ok(())
    }

    fn ensure_loaded_from_store(
        &mut self,
        store: &WeightStore,
        cfg: &TextConfig,
    ) -> Result<(), Error> {
        if !self.gate_w.is_empty() {
            return Ok(());
        }
        match store {
            WeightStore::Dgq(dgq) => {
                self.pre_norm_w = dgq.tensor_f32(SC_PRE_NORM)?;
                self.gate_w = dgq.tensor_f32(SC_GATE)?;
                self.up_w = dgq.tensor_f32(SC_UP)?;
                self.down_w = dgq.tensor_f32(SC_DOWN)?;
            }
            _ => {
                let weights = SelfConditioningWeights::load(store, cfg)?;
                self.load_weights(&weights)?;
            }
        }
        Ok(())
    }
}

/// `inputs_embeds += MLP(pre_norm(soft_signal))`, then RMSNorm without scale on the sum.
pub fn apply_from_store(
    out: &mut [f32],
    inputs_embeds: &[f32],
    soft_signal: &[f32],
    store: &WeightStore,
    cfg: &TextConfig,
    seq_len: usize,
    scratch: &mut SelfConditioningScratch,
) -> Result<(), Error> {
    scratch.ensure_loaded_from_store(store, cfg)?;
    apply_loaded(out, inputs_embeds, soft_signal, cfg, seq_len, scratch)
}

fn apply_loaded(
    out: &mut [f32],
    inputs_embeds: &[f32],
    soft_signal: &[f32],
    cfg: &TextConfig,
    seq_len: usize,
    scratch: &mut SelfConditioningScratch,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps as f32;

    assert_eq!(inputs_embeds.len(), seq_len * hidden);
    assert_eq!(soft_signal.len(), seq_len * hidden);
    assert_eq!(out.len(), seq_len * hidden);

    rms_norm_rows(
        &mut scratch.normed,
        soft_signal,
        &scratch.pre_norm_w,
        seq_len,
        hidden,
        eps,
    );
    linear(
        &mut scratch.gate,
        &scratch.normed,
        &scratch.gate_w,
        None,
        seq_len,
        hidden,
        inter,
    );
    linear(
        &mut scratch.up,
        &scratch.normed,
        &scratch.up_w,
        None,
        seq_len,
        hidden,
        inter,
    );
    gelu_pytorch_tanh(&mut scratch.gate);
    for i in 0..scratch.up.len() {
        scratch.up[i] *= scratch.gate[i];
    }
    linear(
        &mut scratch.signal,
        &scratch.up,
        &scratch.down_w,
        None,
        seq_len,
        inter,
        hidden,
    );

    for i in 0..out.len() {
        out[i] = inputs_embeds[i] + scratch.signal[i];
    }
    for s in 0..seq_len {
        let off = s * hidden;
        scratch.normed[off..off + hidden].copy_from_slice(&out[off..off + hidden]);
        rms_norm_no_scale(
            &mut out[off..off + hidden],
            &scratch.normed[off..off + hidden],
            eps,
        );
    }
    Ok(())
}
