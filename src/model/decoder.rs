use crate::config::ModelConfig;
use crate::kernels::cpu::rms_norm_rows;
use crate::model::decoder_layer::{forward_decoder as layer_forward, DecoderLayerScratch};
use crate::model::embed::{embed_tokens, lm_head_tied_bf16, logit_softcapping};
use crate::model::kv_cache::{KvCache, LayerKvView};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::mask::DecoderAttnMask;
use crate::model::self_conditioning::{apply as apply_self_conditioning, SelfConditioningScratch, SelfConditioningWeights};
use crate::safetensors::Error;
use crate::weights::WeightStore;

pub struct DecoderForwardInput<'a> {
    pub token_ids: &'a [u32],
    pub kv_cache: &'a KvCache,
    pub self_conditioning_logits: Option<&'a [f32]>,
    pub mask: Option<DecoderAttnMask>,
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
    pub self_cond: SelfConditioningScratch,
}

impl DecoderScratch {
    pub fn new(seq_len: usize, cfg: &ModelConfig) -> Self {
        let hidden = cfg.text_config.hidden_size;
        Self {
            hidden_a: vec![0.0; seq_len * hidden],
            hidden_b: vec![0.0; seq_len * hidden],
            embed_buf: vec![0.0; seq_len * hidden],
            sc_signal: vec![0.0; seq_len * hidden],
            norm_w: Vec::new(),
            self_cond: SelfConditioningScratch::new(seq_len, &cfg.text_config),
        }
    }
}

pub fn forward(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &DecoderForwardInput<'_>,
    scratch: &mut DecoderScratch,
) -> Result<DecoderForwardOutput, Error> {
    let text = &cfg.text_config;
    let seq_len = input.token_ids.len();
    let hidden = text.hidden_size;
    let vocab = text.vocab_size;
    let embed_scale = (hidden as f32).sqrt();

    let embed = store.tensor("model.decoder.embed_tokens.weight")?;
    embed.expect_shape(&[vocab as i64, hidden as i64])?;
    let embed_bf16 = embed.bf16()?;

    embed_tokens(
        &mut scratch.embed_buf,
        input.token_ids,
        embed_bf16,
        hidden,
        embed_scale,
    )?;

    let sc_weights = SelfConditioningWeights::load(store, text)?;
    match input.self_conditioning_logits {
        Some(logits) => {
            crate::model::embed::soft_embeddings_from_logits(
                &mut scratch.sc_signal,
                logits,
                embed_bf16,
                seq_len,
                vocab,
                hidden,
                embed_scale,
            );
        }
        None => scratch.sc_signal.fill(0.0),
    }
    apply_self_conditioning(
        &mut scratch.hidden_a,
        &scratch.embed_buf,
        &scratch.sc_signal,
        &sc_weights,
        text,
        seq_len,
        &mut scratch.self_cond,
    )?;

    let mask = input.mask.as_ref();
    let positions: Vec<i64> = (input.kv_cache.kv_len as i64..input.kv_cache.kv_len as i64 + seq_len as i64).collect();

    let mut in_buf = &mut scratch.hidden_a;
    let mut out_buf = &mut scratch.hidden_b;

    for layer in 0..text.num_hidden_layers {
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

    scratch.norm_w = store.tensor("model.decoder.norm.weight")?.bf16()?.to_f32_vec();
    rms_norm_rows(
        out_buf,
        in_buf,
        &scratch.norm_w,
        seq_len,
        hidden,
        text.rms_norm_eps as f32,
    );

    let mut logits = vec![0.0f32; seq_len * vocab];
    lm_head_tied_bf16(&mut logits, out_buf, embed, seq_len, hidden, vocab)?;
    logit_softcapping(&mut logits, text.final_logit_softcapping as f32);

    let mut hidden_out = vec![0.0f32; seq_len * hidden];
    hidden_out.copy_from_slice(out_buf);

    Ok(DecoderForwardOutput {
        hidden_states: hidden_out,
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
