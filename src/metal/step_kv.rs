//! Monolithic step-kernel KV cache (buffer b4) layout and CPU prefill writer.
//!
//! # Layout (M1.1)
//!
//! One unified `half` blob (b4). Per decoder layer `L` (0..29), `LayerOffsets.kv_region` is the
//! **byte** offset of that layer's region. Shaders index halves via `kvcache + kv_region/2`.
//!
//! ```text
//! b4 layout (per layer L):
//!   region_bytes = max_seq * n_kv_heads(L) * head_dim(L) * 2 * sizeof(half)
//!   layer L+1 kv_region = layer L kv_region + region_bytes(L)
//!
//! Per absolute token position pos in [0, max_seq):
//!   half_base = kv_region/2 + pos * (n_kv * head_dim * 2)
//!   [K: n_kv * head_dim halves][V: n_kv * head_dim halves]   // K then V, head-major
//!
//! Sliding layers (not in FULL_LAYERS): n_kv=8, head_dim=256  -> 4096 halves/token
//! Full layers (5,11,17,23,29):         n_kv=2, head_dim=512  -> 2048 halves/token
//!
//! Read path (`k_attention`): T = P.kv_len + CANVAS; attends positions 0..T-1.
//! Write path (`k_qk_rope_kv`): canvas tokens write at pos = P.kv_len + tok (post-RoPE K, normed V).
//! ```
//!
//! Engine `GpuKvCache` uses separate f32 K/V buffers with RoPE applied at read time; the monolithic
//! cache stores **post-RoPE K** and **V** in the layout above. M1.2 packs CPU encoder prefill output.

use crate::config::ModelConfig;
use crate::metal::decoder::load_weight_cache;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::encoder_extend::prefill_gpu;
use crate::metal::step_kernel::{ModelLayout, N_LAYERS};
use crate::metal::GpuDecoderScratch;
use crate::model::encoder::{EncoderPrefillInput, EncoderScratch};
use crate::model::kv_cache::KvCache;
use crate::safetensors::Error;
use crate::weights::WeightStore;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::path::Path;

pub fn kv_region_bytes(n_kv_heads: u32, head_dim: u32, max_seq: usize) -> u64 {
    (max_seq as u64) * (n_kv_heads as u64) * (head_dim as u64) * 2 * 2
}

pub fn kv_cache_total_bytes(layout: &ModelLayout, max_seq: usize) -> u64 {
    (0..N_LAYERS)
        .map(|i| {
            let l = &layout.layers[i];
            kv_region_bytes(l.n_kv_heads, l.head_dim, max_seq)
        })
        .sum()
}

fn f32_to_half_bits(v: f32) -> u16 {
    let bits = v.clamp(-65504.0, 65504.0).to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = (bits & 0x7fffff) as u32;
    if exp == 0xff {
        return sign | if mant == 0 { 0 } else { 0x7e00 };
    }
    if exp == 0 {
        return sign;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Pack CPU post-RoPE encoder KV into monolithic b4 layout (half, K then V per token).
pub fn pack_kv_cache_to_monolithic(
    dst: &mut [u8],
    layout: &ModelLayout,
    kv: &KvCache,
    max_seq: usize,
) -> Result<(), Error> {
    if kv.kv_len > max_seq {
        return Err(Error::Format("kv_len exceeds max_seq"));
    }
    for layer in 0..N_LAYERS.min(kv.layers.len()) {
        let l = &layout.layers[layer];
        let kv_layer = kv
            .layer(layer)
            .ok_or(Error::Format("missing kv layer"))?;
        if kv_layer.n_kv_heads as u32 != l.n_kv_heads || kv_layer.head_dim as u32 != l.head_dim {
            return Err(Error::Format("kv layer head dims mismatch"));
        }
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let token_stride_half = nkv * hd * 2;
        let byte_base = l.kv_region as usize;
        if byte_base + max_seq * token_stride_half * 2 > dst.len() {
            return Err(Error::Format("monolithic kv buffer too small"));
        }
        let per_token = nkv * hd;
        for pos in 0..kv.kv_len {
            let half_base = byte_base / 2 + pos * token_stride_half;
            for hh in 0..nkv {
                for d in 0..hd {
                    let src_i = pos * per_token + hh * hd + d;
                    let k_half = f32_to_half_bits(kv_layer.keys[src_i]);
                    let v_half = f32_to_half_bits(kv_layer.values[src_i]);
                    let k_dst = (half_base + hh * hd + d) * 2;
                    let v_dst = (half_base + nkv * hd + hh * hd + d) * 2;
                    dst[k_dst..k_dst + 2].copy_from_slice(&k_half.to_le_bytes());
                    dst[v_dst..v_dst + 2].copy_from_slice(&v_half.to_le_bytes());
                }
            }
        }
    }
    Ok(())
}

pub fn write_monolithic_kv_buffer(
    buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    kv: &KvCache,
    max_seq: usize,
) -> Result<(), Error> {
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if buf.length() < need {
        return Err(Error::Format("monolithic kv buffer too small"));
    }
    let dst = unsafe {
        std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u8, need)
    };
    dst.fill(0);
    pack_kv_cache_to_monolithic(dst, layout, kv, max_seq)
}

/// Pack one layer prefix from f32 K/V (engine GPU layout) into monolithic b4.
fn pack_layer_f32_kv_to_monolithic(
    dst: &mut [u8],
    layout: &ModelLayout,
    layer: usize,
    keys: &[f32],
    values: &[f32],
    kv_len: usize,
    max_seq: usize,
) -> Result<(), Error> {
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let per_token = nkv * hd;
    if keys.len() < kv_len * per_token || values.len() < kv_len * per_token {
        return Err(Error::Format("gpu kv prefix too short"));
    }
    let token_stride_half = nkv * hd * 2;
    let byte_base = l.kv_region as usize;
    if byte_base + max_seq * token_stride_half * 2 > dst.len() {
        return Err(Error::Format("monolithic kv buffer too small"));
    }
    for pos in 0..kv_len {
        let half_base = byte_base / 2 + pos * token_stride_half;
        for hh in 0..nkv {
            for d in 0..hd {
                let src_i = pos * per_token + hh * hd + d;
                let k_half = f32_to_half_bits(keys[src_i]);
                let v_half = f32_to_half_bits(values[src_i]);
                let k_dst = (half_base + hh * hd + d) * 2;
                let v_dst = (half_base + nkv * hd + hh * hd + d) * 2;
                dst[k_dst..k_dst + 2].copy_from_slice(&k_half.to_le_bytes());
                dst[v_dst..v_dst + 2].copy_from_slice(&v_half.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn read_f32_prefix(buf: &ProtocolObject<dyn MTLBuffer>, elems: usize) -> Vec<f32> {
    let ptr = buf.contents().as_ptr() as *const f32;
    (0..elems).map(|i| unsafe { *ptr.add(i) }).collect()
}

/// GPU encoder prefill → read back post-RoPE K/V → monolithic b4 (M1.2, `.dgq` path).
pub fn prefill_monolithic_kv(
    model_dir: &Path,
    token_ids: &[u32],
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
    max_layers: usize,
) -> Result<usize, Error> {
    if token_ids.is_empty() {
        return Err(Error::Format("prefill requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Format("prefill exceeds max_seq"));
    }
    let model = crate::model::Model::open(model_dir)?;
    let text = &model.config.text_config;
    let canvas = crate::metal::step_kernel::CANVAS;
    let layers = max_layers.min(text.num_hidden_layers);

    let mut enc_scratch = EncoderScratch::new(token_ids.len(), &model.config);
    let mut dec_scratch = GpuDecoderScratch::new(canvas, &model.config);
    let mut weights = load_weight_cache(
        &model.weights,
        text,
        canvas,
        token_ids.len(),
    )?;
    let mut engine = GpuDecoderEngine::new()?;
    engine.set_use_mps_q4(false);

    let _cpu_kv = prefill_gpu(
        &model.weights,
        &model.config,
        &EncoderPrefillInput {
            token_ids,
            position_offset: 0,
        },
        &mut enc_scratch,
        &mut dec_scratch,
        &mut weights,
        &mut engine,
        max_seq,
        canvas,
        Some(layers),
    )?;

    let gpu_kv = dec_scratch
        .gpu_kv
        .as_ref()
        .ok_or(Error::Format("gpu kv missing after prefill"))?;
    let kv_len = gpu_kv.kv_len;
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if kv_buf.length() < need {
        return Err(Error::Format("monolithic kv buffer too small"));
    }
    let dst = unsafe {
        std::slice::from_raw_parts_mut(kv_buf.contents().as_ptr() as *mut u8, need)
    };
    dst.fill(0);

    for layer in 0..layers {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let elems = kv_len * nkv * hd;
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        let keys = read_f32_prefix(&k_buf, elems);
        let values = read_f32_prefix(&v_buf, elems);
        pack_layer_f32_kv_to_monolithic(dst, layout, layer, &keys, &values, kv_len, max_seq)?;
    }
    Ok(kv_len)
}

/// CPU encoder prefill (bf16 weights only).
pub fn prefill_monolithic_kv_cpu(
    store: &WeightStore,
    cfg: &ModelConfig,
    token_ids: &[u32],
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
) -> Result<usize, Error> {
    if store.is_quantized() {
        return Err(Error::Format(
            "cpu monolithic prefill unsupported on .dgq; use prefill_monolithic_kv",
        ));
    }
    if token_ids.is_empty() {
        return Err(Error::Format("prefill requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Format("prefill exceeds max_seq"));
    }
    let mut scratch = EncoderScratch::new(token_ids.len(), cfg);
    let out = crate::model::encoder::prefill(
        store,
        cfg,
        &EncoderPrefillInput {
            token_ids,
            position_offset: 0,
        },
        &mut scratch,
    )?;
    write_monolithic_kv_buffer(kv_buf, layout, &out.kv_cache, max_seq)?;
    Ok(out.kv_cache.kv_len)
}
