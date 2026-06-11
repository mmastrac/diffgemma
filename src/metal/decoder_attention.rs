use crate::config::TextConfig;
use crate::kernels::cpu::linear;
use crate::metal::engine::GpuDecoderEngine;
use crate::model::attention::{
    concat_kv_for_decoder, prepare_qkv_pre_rope, AttentionParams, AttentionScratch, GqaMask,
};
use crate::model::kv_cache::LayerKvView;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;

/// Decoder self-attention: CPU Q/K/V + norms, GPU RoPE + GQA, CPU `o_proj`.
pub fn forward_decoder_attention(
    out: &mut [f32],
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut AttentionScratch,
    engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);
    assert_eq!(kv.kv_len + seq_len, total_kv);

    prepare_qkv_pre_rope(hidden, weights, cfg, layer, seq_len, positions, scratch)?;

    engine.attention.apply_rope_qk(
        &mut scratch.q,
        &mut scratch.k,
        &scratch.rope_freqs,
        seq_len,
        params.n_heads,
        params.n_kv_heads,
        params.head_dim,
        params.rotary_dim,
    )?;

    concat_kv_for_decoder(
        &mut scratch.k_full,
        &mut scratch.v_full,
        &kv,
        &scratch.k,
        &scratch.v,
        &params,
    );

    let default_mask;
    let gqa_mask = match mask {
        Some(m) => GqaMask::DecoderBitmap(m),
        None => {
            default_mask = DecoderAttnMask::all_valid(seq_len, kv.kv_len);
            GqaMask::DecoderBitmap(&default_mask)
        }
    };

    engine.attention.gqa_attention(
        &mut scratch.attn_out,
        &scratch.q,
        &scratch.k_full,
        &scratch.v_full,
        seq_len,
        total_kv,
        &params,
        gqa_mask,
    )?;

    linear(
        out,
        &scratch.attn_out,
        &scratch.o_w,
        None,
        seq_len,
        q_dim,
        hidden_size,
    );
    Ok(())
}
