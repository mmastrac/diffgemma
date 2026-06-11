use crate::config::ModelConfig;
use crate::metal::decoder_layer::{forward_decoder as layer_forward, GpuDecoderLayerScratch};
use crate::metal::engine::GpuDecoderEngine;
use crate::model::decoder::{DecoderForwardInput, DecoderForwardOutput, DecoderScratch};
use crate::model::embed::{embed_tokens, lm_head_tied_bf16, logit_softcapping};
use crate::model::kv_cache::LayerKvView;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::self_conditioning::{apply as apply_self_conditioning, SelfConditioningWeights};
use crate::safetensors::Error;
use crate::weights::WeightStore;

pub struct GpuDecoderScratch {
    pub cpu: DecoderScratch,
}

impl GpuDecoderScratch {
    pub fn new(seq_len: usize, cfg: &ModelConfig) -> Self {
        Self {
            cpu: DecoderScratch::new(seq_len, cfg),
        }
    }
}

pub fn forward(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &DecoderForwardInput<'_>,
    scratch: &mut GpuDecoderScratch,
    engine: &mut GpuDecoderEngine,
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
        &mut scratch.cpu.embed_buf,
        input.token_ids,
        embed_bf16,
        hidden,
        embed_scale,
    )?;

    let sc_weights = SelfConditioningWeights::load(store, text)?;
    match input.self_conditioning_logits {
        Some(logits) => {
            crate::model::embed::soft_embeddings_from_logits(
                &mut scratch.cpu.sc_signal,
                logits,
                embed_bf16,
                seq_len,
                vocab,
                hidden,
                embed_scale,
            );
        }
        None => scratch.cpu.sc_signal.fill(0.0),
    }
    apply_self_conditioning(
        &mut scratch.cpu.hidden_a,
        &scratch.cpu.embed_buf,
        &scratch.cpu.sc_signal,
        &sc_weights,
        text,
        seq_len,
        &mut scratch.cpu.self_cond,
    )?;

    let mask = input.mask.as_ref();
    let positions: Vec<i64> =
        (input.kv_cache.kv_len as i64..input.kv_cache.kv_len as i64 + seq_len as i64).collect();

    let mut in_buf = &mut scratch.cpu.hidden_a;
    let mut out_buf = &mut scratch.cpu.hidden_b;

    for layer in 0..text.num_hidden_layers {
        let weights = DecoderLayerWeights::load(store, layer, text)?;
        let mut layer_scratch =
            GpuDecoderLayerScratch::with_kv_len(seq_len, text, layer, input.kv_cache.kv_len)?;
        layer_forward(
            out_buf,
            in_buf,
            &weights,
            text,
            layer,
            seq_len,
            &positions,
            LayerKvView::from_layer(
                input
                    .kv_cache
                    .layer(layer)
                    .ok_or(Error::Format("missing kv layer"))?,
                input.kv_cache.kv_len,
            ),
            mask,
            &mut layer_scratch,
            engine,
        )?;
        std::mem::swap(&mut in_buf, &mut out_buf);
    }

    scratch.cpu.norm_w = store.tensor("model.decoder.norm.weight")?.bf16()?.to_f32_vec();
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        out_buf,
        in_buf,
        &scratch.cpu.norm_w,
        seq_len,
        hidden,
        text.rms_norm_eps as f32,
    )?;

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
