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
use crate::metal::decoder::load_weight_cache_opt;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::encoder_extend::{extend_prefill_gpu, prefill_gpu};
use crate::metal::step_kernel::{f16_bits_to_f32, ModelLayout, N_LAYERS, CANVAS, StepSmokeConfig, StepFinishMode, run_step_forward, build_layout, build_offsets_from_store};
use crate::metal::device::MetalContext;
use crate::dgq::DgqStore;
use crate::metal::GpuDecoderScratch;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::Model;
use crate::model::encoder::{EncoderPrefillInput, EncoderScratch};
use crate::model::kv_cache::KvCache;
use crate::safetensors::Error;
use crate::weights::WeightStore;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
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

/// Pack one layer prefix from f32 K/V (engine GPU layout) into monolithic b4 at `dst_pos..`.
fn pack_layer_f32_kv_to_monolithic(
    dst: &mut [u8],
    layout: &ModelLayout,
    layer: usize,
    keys: &[f32],
    values: &[f32],
    dst_pos: usize,
    token_count: usize,
    max_seq: usize,
) -> Result<(), Error> {
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let per_token = nkv * hd;
    if keys.len() < token_count * per_token || values.len() < token_count * per_token {
        return Err(Error::Format("gpu kv prefix too short"));
    }
    if dst_pos + token_count > max_seq {
        return Err(Error::Format("monolithic kv extend exceeds max_seq"));
    }
    let token_stride_half = nkv * hd * 2;
    let byte_base = l.kv_region as usize;
    if byte_base + max_seq * token_stride_half * 2 > dst.len() {
        return Err(Error::Format("monolithic kv buffer too small"));
    }
    for pos in 0..token_count {
        let half_base = byte_base / 2 + (dst_pos + pos) * token_stride_half;
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

/// Reusable GPU encoder stack for monolithic KV prefill/extend (P1.8).
pub struct MonolithicEncoderCache {
    model: Model,
    weights: GpuDecoderWeightCache,
    engine: GpuDecoderEngine,
    dec_scratch: GpuDecoderScratch,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MonolithicPrefillTiming {
    pub gpu_forward_ms: f64,
    pub kv_pack_ms: f64,
    pub total_ms: f64,
}

impl MonolithicEncoderCache {
    pub fn open(model_dir: &Path, canvas: usize, max_seq_hint: usize) -> Result<Self, Error> {
        Self::open_opt(model_dir, canvas, max_seq_hint, None, None)
    }

    pub fn use_mps_q4(&self) -> bool {
        self.engine.use_mps_q4()
    }

    pub fn open_opt(
        model_dir: &Path,
        canvas: usize,
        max_seq_hint: usize,
        shared_dgq_blob: Option<std::sync::Arc<crate::metal::dgq_gpu::DgqGpuBlob>>,
        use_mps_q4: Option<bool>,
    ) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        let model = Model::open(model_dir)?;
        let text = &model.config.text_config;
        let weights = load_weight_cache_opt(
            &model.weights,
            text,
            canvas,
            max_seq_hint,
            shared_dgq_blob,
        )?;
        let mut engine = GpuDecoderEngine::new()?;
        if let Some(v) = use_mps_q4 {
            engine.set_use_mps_q4(v);
        }
        let dec_scratch = GpuDecoderScratch::new(canvas, &model.config);
        eprintln!(
            "monolithic-encoder: cache open {:.2?} (model + engine weights, use_mps_q4={})",
            open_started.elapsed(),
            engine.use_mps_q4(),
        );
        Ok(Self {
            model,
            weights,
            engine,
            dec_scratch,
        })
    }
}

/// GPU encoder prefill → read back post-RoPE K/V → monolithic b4 (reuses `cache`).
pub fn prefill_monolithic_kv_with_cache(
    cache: &mut MonolithicEncoderCache,
    token_ids: &[u32],
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
    max_layers: usize,
) -> Result<(usize, MonolithicPrefillTiming), Error> {
    prefill_monolithic_kv_with_cache_timed(
        cache,
        token_ids,
        kv_buf,
        layout,
        max_seq,
        max_layers,
    )
}

pub fn prefill_monolithic_kv_with_cache_timed(
    cache: &mut MonolithicEncoderCache,
    token_ids: &[u32],
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
    max_layers: usize,
) -> Result<(usize, MonolithicPrefillTiming), Error> {
    let total_started = std::time::Instant::now();
    if token_ids.is_empty() {
        return Err(Error::Format("prefill requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Format("prefill exceeds max_seq"));
    }
    let text = &cache.model.config.text_config;
    let canvas = CANVAS;
    let layers = max_layers.min(text.num_hidden_layers);
    let encoder_kv_cap = token_ids.len().max(1);

    let mut enc_scratch = EncoderScratch::new(token_ids.len(), &cache.model.config);
    if let Some(gpu_kv) = cache.dec_scratch.gpu_kv.as_mut() {
        gpu_kv.reset_len();
    }

    let gpu_started = std::time::Instant::now();
    let _cpu_kv = prefill_gpu(
        &cache.model.weights,
        &cache.model.config,
        &EncoderPrefillInput {
            token_ids,
            position_offset: 0,
        },
        &mut enc_scratch,
        &mut cache.dec_scratch,
        &mut cache.weights,
        &mut cache.engine,
        encoder_kv_cap,
        canvas,
        Some(layers),
    )?;
    let gpu_forward_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;

    let gpu_kv = cache
        .dec_scratch
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

    let pack_started = std::time::Instant::now();
    for layer in 0..layers {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let elems = kv_len * nkv * hd;
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        let keys = read_f32_prefix(&k_buf, elems);
        let values = read_f32_prefix(&v_buf, elems);
        pack_layer_f32_kv_to_monolithic(dst, layout, layer, &keys, &values, 0, kv_len, max_seq)?;
    }
    let kv_pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;
    let timing = MonolithicPrefillTiming {
        gpu_forward_ms,
        kv_pack_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1000.0,
    };
    eprintln!(
        "monolithic-prefill: kv_len={kv_len} use_mps_q4={} gpu_forward={gpu_forward_ms:.1}ms kv_pack={kv_pack_ms:.1}ms total={:.1}ms",
        cache.engine.use_mps_q4(),
        timing.total_ms
    );
    Ok((kv_len, timing))
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
    let mut cache = MonolithicEncoderCache::open(model_dir, CANVAS, max_seq)?;
    Ok(prefill_monolithic_kv_with_cache(
        &mut cache,
        token_ids,
        kv_buf,
        layout,
        max_seq,
        max_layers,
    )?
    .0)
}

fn read_half_at(kv_buf: &ProtocolObject<dyn MTLBuffer>, byte_off: usize) -> u16 {
    let ptr = unsafe { kv_buf.contents().as_ptr().add(byte_off) as *const u8 };
    u16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }])
}

/// Load monolithic b4 prefix `[0, kv_len)` into engine `GpuKvCache` (for extend after prefill).
fn hydrate_gpu_kv_from_monolithic(
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    kv_len: usize,
    gpu_kv: &mut crate::metal::GpuKvCache,
    layers: usize,
) -> Result<(), Error> {
    if kv_len == 0 {
        gpu_kv.reset_len();
        return Ok(());
    }
    use crate::metal::buffer::BufferPool;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let per_token = nkv * hd;
        let token_stride_half = nkv * hd * 2;
        let byte_base = l.kv_region as usize;
        let mut keys = vec![0f32; kv_len * per_token];
        let mut values = vec![0f32; kv_len * per_token];
        for pos in 0..kv_len {
            let half_base = byte_base / 2 + pos * token_stride_half;
            for hh in 0..nkv {
                for d in 0..hd {
                    let dst_i = pos * per_token + hh * hd + d;
                    let k_byte = (half_base + hh * hd + d) * 2;
                    let v_byte = (half_base + nkv * hd + hh * hd + d) * 2;
                    keys[dst_i] = f16_bits_to_f32(read_half_at(kv_buf, k_byte));
                    values[dst_i] = f16_bits_to_f32(read_half_at(kv_buf, v_byte));
                }
            }
        }
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        BufferPool::write_f32(&k_buf, &keys);
        BufferPool::write_f32(&v_buf, &values);
    }
    gpu_kv.kv_len = kv_len;
    Ok(())
}

/// GPU encoder extend → read back new suffix K/V → append into monolithic b4 (reuses `cache`).
pub fn extend_monolithic_kv_with_cache(
    cache: &mut MonolithicEncoderCache,
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    kv_len_before: usize,
    new_token_ids: &[u32],
    max_seq: usize,
    max_layers: usize,
) -> Result<usize, Error> {
    if new_token_ids.is_empty() {
        return Ok(kv_len_before);
    }
    if kv_len_before + new_token_ids.len() > max_seq {
        return Err(Error::Format("monolithic kv extend exceeds max_seq"));
    }
    let text = &cache.model.config.text_config;
    let canvas = CANVAS;
    let layers = max_layers.min(text.num_hidden_layers);

    let mut enc_scratch = EncoderScratch::new(new_token_ids.len(), &cache.model.config);
    let encoder_kv_cap = (kv_len_before + new_token_ids.len()).min(max_seq);

    cache.dec_scratch.ensure_gpu_kv(
        &cache.engine.ctx.device,
        text,
        encoder_kv_cap,
        canvas,
    )?;
    let mut gpu_kv = cache
        .dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Format("gpu kv cache missing"))?;
    hydrate_gpu_kv_from_monolithic(kv_buf, layout, kv_len_before, &mut gpu_kv, layers)?;
    cache.dec_scratch.gpu_kv = Some(gpu_kv);

    let mut cpu_kv = KvCache::empty(text)?;
    cpu_kv.kv_len = kv_len_before;

    extend_prefill_gpu(
        &cache.model.weights,
        &cache.model.config,
        &mut cpu_kv,
        new_token_ids,
        &mut enc_scratch,
        &mut cache.dec_scratch,
        &mut cache.weights,
        &mut cache.engine,
        Some(layers),
    )?;

    let gpu_kv = cache
        .dec_scratch
        .gpu_kv
        .as_ref()
        .ok_or(Error::Format("gpu kv missing after extend"))?;
    let new_kv_len = gpu_kv.kv_len;
    let append_len = new_token_ids.len();
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if kv_buf.length() < need {
        return Err(Error::Format("monolithic kv buffer too small"));
    }
    let dst = unsafe {
        std::slice::from_raw_parts_mut(kv_buf.contents().as_ptr() as *mut u8, need)
    };

    for layer in 0..layers {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let per_token = nkv * hd;
        let elems = append_len * per_token;
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        let byte_off = kv_len_before * per_token * 4;
        let keys = read_f32_at(&k_buf, byte_off, elems);
        let values = read_f32_at(&v_buf, byte_off, elems);
        pack_layer_f32_kv_to_monolithic(
            dst,
            layout,
            layer,
            &keys,
            &values,
            kv_len_before,
            append_len,
            max_seq,
        )?;
    }
    Ok(new_kv_len)
}

/// GPU encoder extend → read back new suffix K/V → append into monolithic b4 (M1.3).
pub fn extend_monolithic_kv(
    model_dir: &Path,
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    kv_len_before: usize,
    new_token_ids: &[u32],
    max_seq: usize,
    max_layers: usize,
) -> Result<usize, Error> {
    let mut cache = MonolithicEncoderCache::open(model_dir, CANVAS, max_seq)?;
    extend_monolithic_kv_with_cache(
        &mut cache,
        kv_buf,
        layout,
        kv_len_before,
        new_token_ids,
        max_seq,
        max_layers,
    )
}

fn read_f32_at(buf: &ProtocolObject<dyn MTLBuffer>, byte_offset: usize, elems: usize) -> Vec<f32> {
    let ptr = unsafe { buf.contents().as_ptr().add(byte_offset) as *const f32 };
    (0..elems).map(|i| unsafe { *ptr.add(i) }).collect()
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

/// Max abs half value in layer `L` prefix `[0, kv_len)`.
pub fn kvcache_prefix_max_abs(
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    layer: usize,
    kv_len: usize,
) -> f32 {
    if kv_len == 0 || layer >= N_LAYERS {
        return 0.0;
    }
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let token_stride_half = nkv * hd * 2;
    let byte_base = l.kv_region as usize;
    let half_base = byte_base / 2;
    let mut max_abs = 0.0f32;
    for pos in 0..kv_len {
        let start = (half_base + pos * token_stride_half) * 2;
        let end = start + token_stride_half * 2;
        let bytes = unsafe {
            std::slice::from_raw_parts(kv_buf.contents().as_ptr().add(start) as *const u8, end - start)
        };
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            max_abs = max_abs.max(f16_bits_to_f32(bits).abs());
        }
    }
    max_abs
}

#[derive(Debug)]
pub struct StepKvAuditResult {
    pub kv_len: usize,
    pub prefix_max_abs_l0: f32,
    pub hidden_max_abs_vs_zero: f32,
    pub logits_max_abs_vs_zero: f32,
    pub extend_kv_len: Option<usize>,
    pub extend_hidden_diff: Option<f32>,
    pub pass: bool,
}

/// M1.4: verify b4 prefix is populated and forward changes vs kv_len=0.
pub fn run_step_kv_audit(
    model_dir: &Path,
    kv_len: usize,
    layers: usize,
    seed: u64,
    max_seq: usize,
) -> Result<StepKvAuditResult, Error> {
    let vocab = crate::metal::step_kernel::VOCAB;
    let mut prompt = vec![0u32; kv_len];
    for (i, id) in prompt.iter_mut().enumerate() {
        *id = ((i * 131 + 7) % vocab.max(1)) as u32;
    }
    let base_cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        use_mps_q4: Some(false),
        prefill_token_ids: None,
    };
    let zero = run_step_forward(model_dir, &base_cfg)?;
    let mut kv_cfg = base_cfg.clone();
    kv_cfg.prefill_token_ids = Some(prompt.clone());
    let with_kv = run_step_forward(model_dir, &kv_cfg)?;

    let store = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
    let kv_buf = ctx
        .device
        .newBufferWithLength_options(
            kv_bytes,
            objc2_metal::MTLResourceOptions::StorageModeShared,
        )
        .ok_or(Error::Format("kv audit buffer alloc failed"))?;
    let actual_kv = prefill_monolithic_kv(
        model_dir,
        &prompt,
        &kv_buf,
        &layout,
        max_seq,
        layers,
    )?;
    let prefix_max = kvcache_prefix_max_abs(&kv_buf, &layout, 0, actual_kv);

    let hidden_diff = zero
        .norm_hidden
        .iter()
        .zip(with_kv.norm_hidden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let logits_diff = zero
        .logits
        .iter()
        .zip(with_kv.logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mut extend_kv_len = None;
    let mut extend_hidden_diff = None;
    let mut pass = prefix_max > 1e-4 && hidden_diff > 1e-3;

    if pass && kv_len >= 32 {
        let prefill_len = kv_len / 2;
        let extend_tokens = prompt[prefill_len..].to_vec();
        let prefill_only = &prompt[..prefill_len];
        let mut extend_buf = ctx
            .device
            .newBufferWithLength_options(
                kv_bytes,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .ok_or(Error::Format("kv extend audit buffer alloc failed"))?;
        let prefill_actual = prefill_monolithic_kv(
            model_dir,
            prefill_only,
            &extend_buf,
            &layout,
            max_seq,
            layers,
        )?;
        let extended = extend_monolithic_kv(
            model_dir,
            &extend_buf,
            &layout,
            prefill_actual,
            &extend_tokens,
            max_seq,
            layers,
        )?;
        extend_kv_len = Some(extended);
        if extended != kv_len {
            pass = false;
        } else {
            let mut prefill_cfg = base_cfg.clone();
            prefill_cfg.prefill_token_ids = Some(prefill_only.to_vec());
            let half_kv = run_step_forward(model_dir, &prefill_cfg)?;
            let ext_diff = half_kv
                .norm_hidden
                .iter()
                .zip(with_kv.norm_hidden.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            extend_hidden_diff = Some(ext_diff);
            if ext_diff <= 1e-3 {
                pass = false;
            }
        }
    }

    Ok(StepKvAuditResult {
        kv_len,
        prefix_max_abs_l0: prefix_max,
        hidden_max_abs_vs_zero: hidden_diff,
        logits_max_abs_vs_zero: logits_diff,
        extend_kv_len,
        extend_hidden_diff,
        pass,
    })
}
