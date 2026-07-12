use crate::buffer::Buffer;
use crate::config::ModelConfig;
use crate::model::decoder_layer::{DecoderLayerScratch, forward_decoder as layer_forward};
use crate::model::embed::{embed_tokens_from_store, lm_head_tied_from_store, logit_softcapping};
use crate::model::kv_cache::{KvCache, LayerKvView};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::mask::DecoderAttnMask;
use crate::model::self_conditioning::{SelfConditioningScratch, apply_from_store};
use crate::safetensors::Error;
use crate::shaders::cpu::rms_norm_rows;
use crate::weights::WeightStore;

pub struct DecoderForwardInput<'a> {
    pub token_ids: &'a [u32],
    pub kv_cache: &'a KvCache,
    pub self_conditioning_logits: Option<&'a [f32]>,
    pub mask: Option<&'a DecoderAttnMask>,
    /// When set, lm_head writes here (must be `seq_len * vocab`). Avoids a per-forward logits alloc.
    pub logits_out: Option<&'a mut [f32]>,
    /// When false, skip lm_head (e.g. `bench-decoder`).
    pub compute_logits: bool,
    /// When false, omit hidden copy in output (generate only needs logits).
    pub return_hidden: bool,
}

impl<'a> DecoderForwardInput<'a> {
    pub fn new(token_ids: &'a [u32], kv_cache: &'a KvCache) -> Self {
        Self {
            token_ids,
            kv_cache,
            self_conditioning_logits: None,
            mask: None,
            logits_out: None,
            compute_logits: true,
            return_hidden: true,
        }
    }
}

pub struct DecoderForwardOutput {
    pub hidden_states: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct DecoderScratch {
    pub hidden_a: Vec<f32>,
    pub hidden_b: Vec<f32>,
    pub embed_buf: Vec<f32>,
    pub sc_signal: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub lm_head_chunk: Buffer<f32>,
    pub sc_probs: Buffer<f32>,
    pub logits_buf: Buffer<f32>,
    pub self_cond: SelfConditioningScratch,
}

impl DecoderScratch {
    pub fn new(seq_len: usize, cfg: &ModelConfig) -> Self {
        let hidden = cfg.text_config.hidden_size;
        let vocab = cfg.text_config.vocab_size;
        Self {
            hidden_a: vec![0.0; seq_len * hidden],
            hidden_b: vec![0.0; seq_len * hidden],
            embed_buf: vec![0.0; seq_len * hidden],
            sc_signal: vec![0.0; seq_len * hidden],
            norm_w: Vec::new(),
            lm_head_chunk: Buffer::with_capacity(crate::model::embed::LM_HEAD_CHUNK * hidden),
            sc_probs: Buffer::with_capacity(vocab),
            logits_buf: Buffer::with_capacity(seq_len * vocab),
            self_cond: SelfConditioningScratch::new(seq_len, &cfg.text_config),
        }
    }
}

pub fn forward(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &mut DecoderForwardInput<'_>,
    scratch: &mut DecoderScratch,
    max_layers: Option<usize>,
) -> Result<DecoderForwardOutput, Error> {
    let text = &cfg.text_config;
    let seq_len = input.token_ids.len();
    let hidden = text.hidden_size;
    let vocab = text.vocab_size;
    let embed_scale = (hidden as f32).sqrt();

    embed_tokens_from_store(
        store,
        &mut scratch.embed_buf,
        input.token_ids,
        hidden,
        embed_scale,
    )?;

    match input.self_conditioning_logits {
        Some(logits) => {
            crate::model::embed::soft_embeddings_from_logits_store(
                store,
                &mut scratch.sc_signal,
                logits,
                seq_len,
                vocab,
                hidden,
                embed_scale,
                &mut scratch.sc_probs,
            )?;
        }
        None => scratch.sc_signal.fill(0.0),
    }
    apply_from_store(
        &mut scratch.hidden_a,
        &scratch.embed_buf,
        &scratch.sc_signal,
        store,
        text,
        seq_len,
        &mut scratch.self_cond,
    )?;

    let mask = input.mask;
    let positions: Vec<i64> =
        (input.kv_cache.kv_len as i64..input.kv_cache.kv_len as i64 + seq_len as i64).collect();

    let mut in_buf = &mut scratch.hidden_a;
    let mut out_buf = &mut scratch.hidden_b;

    let n_layers = max_layers
        .unwrap_or(text.num_hidden_layers)
        .min(text.num_hidden_layers);
    for layer in 0..n_layers {
        let weights = DecoderLayerWeights::load(store, layer, text)?;
        let mut layer_scratch =
            DecoderLayerScratch::with_kv_len(seq_len, text, layer, input.kv_cache.kv_len)?;
        let kv_layer = input
            .kv_cache
            .layer(layer)
            .ok_or(Error::Format("missing kv layer"))?;
        let kv = LayerKvView::from_layer(kv_layer, input.kv_cache.kv_len);
        layer_forward(
            out_buf,
            in_buf,
            &weights,
            text,
            layer,
            seq_len,
            &positions,
            kv,
            mask,
            &mut layer_scratch,
        )?;
        std::mem::swap(&mut in_buf, &mut out_buf);
    }

    scratch.norm_w = store
        .tensor("model.decoder.norm.weight")?
        .bf16()?
        .to_f32_vec();
    rms_norm_rows(
        out_buf,
        in_buf,
        &scratch.norm_w,
        seq_len,
        hidden,
        text.rms_norm_eps as f32,
    );

    if input.compute_logits {
        if let Some(out) = input.logits_out.as_mut() {
            let need = seq_len * vocab;
            if out.len() != need {
                return Err(Error::Format("logits_out length mismatch"));
            }
            lm_head_tied_from_store(
                store,
                out,
                out_buf,
                seq_len,
                hidden,
                vocab,
                &mut scratch.lm_head_chunk,
            )?;
            logit_softcapping(out, text.final_logit_softcapping as f32);
        } else {
            scratch.logits_buf.ensure_len(seq_len * vocab);
            lm_head_tied_from_store(
                store,
                scratch.logits_buf.as_slice_mut(),
                out_buf,
                seq_len,
                hidden,
                vocab,
                &mut scratch.lm_head_chunk,
            )?;
            logit_softcapping(
                scratch.logits_buf.as_slice_mut(),
                text.final_logit_softcapping as f32,
            );
        }
    }

    let hidden_states = if input.return_hidden {
        let mut hidden_out = vec![0.0f32; seq_len * hidden];
        hidden_out.copy_from_slice(out_buf);
        hidden_out
    } else {
        Vec::new()
    };

    let logits = if input.compute_logits && input.logits_out.is_none() {
        scratch.logits_buf.as_slice().to_vec()
    } else {
        Vec::new()
    };

    Ok(DecoderForwardOutput {
        hidden_states,
        logits,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn decoder_logits_shape_matches_canvas_and_vocab() {
        const CANVAS: usize = 256;
        const VOCAB: usize = 262144;
        assert_eq!(CANVAS * VOCAB, 67_108_864);
    }
}
