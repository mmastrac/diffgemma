use crate::config::ModelConfig;
use crate::kernels::cpu::rms_norm_rows;
use crate::model::decoder_layer::{forward_encoder as layer_forward_encoder, DecoderLayerScratch};
use crate::model::embed::embed_tokens;
use crate::model::kv_cache::KvCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::weights::WeightStore;

pub struct EncoderPrefillInput<'a> {
    pub token_ids: &'a [u32],
    /// Absolute position offset for RoPE (0 on first prefill, existing `kv_len` on extend).
    pub position_offset: i64,
}

pub struct EncoderPrefillOutput {
    pub kv_cache: KvCache,
    pub hidden_states: Vec<f32>,
}

pub struct EncoderScratch {
    pub hidden_a: Vec<f32>,
    pub hidden_b: Vec<f32>,
    pub embed_buf: Vec<f32>,
    pub norm_w: Vec<f32>,
}

impl EncoderScratch {
    pub fn new(seq_len: usize, cfg: &ModelConfig) -> Self {
        let hidden = cfg.text_config.hidden_size;
        Self {
            hidden_a: vec![0.0; seq_len * hidden],
            hidden_b: vec![0.0; seq_len * hidden],
            embed_buf: vec![0.0; seq_len * hidden],
            norm_w: Vec::new(),
        }
    }
}

/// Causal encoder prefill using tied `model.decoder.layers.*` weights.
pub fn prefill(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &EncoderPrefillInput<'_>,
    scratch: &mut EncoderScratch,
) -> Result<EncoderPrefillOutput, Error> {
    let text = &cfg.text_config;
    let seq_len = input.token_ids.len();
    let hidden = text.hidden_size;
    let embed_scale = (hidden as f32).sqrt();

    let embed = store.tensor("model.decoder.embed_tokens.weight")?;
    embed.expect_shape(&[text.vocab_size as i64, hidden as i64])?;
    embed_tokens(
        &mut scratch.embed_buf,
        input.token_ids,
        embed.bf16()?,
        hidden,
        embed_scale,
    )?;

    let mut kv_cache = KvCache::empty(text)?;
    let positions: Vec<i64> = (0..seq_len as i64)
        .map(|i| input.position_offset + i)
        .collect();

    let mut in_buf = &mut scratch.hidden_a;
    let mut out_buf = &mut scratch.hidden_b;
    in_buf.copy_from_slice(&scratch.embed_buf);

    for layer in 0..text.num_hidden_layers {
        let weights = DecoderLayerWeights::load(store, layer, text)?;
        let mut layer_scratch = DecoderLayerScratch::new(seq_len, text, layer)?;
        let kv_layer = kv_cache
            .layer_mut(layer)
            .ok_or(Error::Format("missing kv layer"))?;
        layer_forward_encoder(
            out_buf,
            in_buf,
            &weights,
            text,
            layer,
            seq_len,
            &positions,
            kv_layer,
            &mut layer_scratch,
        )?;
        std::mem::swap(&mut in_buf, &mut out_buf);
    }

    kv_cache.kv_len = seq_len;
    kv_cache.validate_dims(text)?;

    scratch.norm_w = store.tensor("model.decoder.norm.weight")?.bf16()?.to_f32_vec();
    rms_norm_rows(
        out_buf,
        in_buf,
        &scratch.norm_w,
        seq_len,
        hidden,
        text.rms_norm_eps as f32,
    );

    let mut hidden_out = vec![0.0f32; seq_len * hidden];
    hidden_out.copy_from_slice(out_buf);

    Ok(EncoderPrefillOutput {
        kv_cache,
        hidden_states: hidden_out,
    })
}
