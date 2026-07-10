//! Monolithic step-kernel KV cache (buffer b4) layout and CPU prefill writer.
//!
//! # Layout (M1.1)
//!
//! One unified bf16 blob (b4). Per decoder layer `L` (0..29), `LayerOffsets.kv_region` is the
//! **byte** offset of that layer's region. Shaders index bf16 slots via `kvcache + kv_region/2`.
//!
//! ```text
//! b4 layout (per layer L):
//!   region_bytes = max_seq * n_kv_heads(L) * head_dim(L) * 2 * sizeof(bf16)
//!   layer L+1 kv_region = layer L kv_region + region_bytes(L)
//!
//! Per absolute token position pos in [0, max_seq):
//!   slot_base = kv_region/2 + pos * (n_kv * head_dim * 2)
//!   [K: n_kv * head_dim bf16 slots][V: n_kv * head_dim bf16 slots]   // K then V, head-major
//!
//! Sliding layers (not in FULL_LAYERS): n_kv=8, head_dim=256  -> 4096 slots/token
//! Full layers (5,11,17,23,29):         n_kv=2, head_dim=512  -> 2048 slots/token
//!
//! Read path (`k_attention`): T = P.kv_len + CANVAS; attends positions 0..T-1.
//! Write path (`k_qk_rope_kv`): canvas tokens write at pos = P.kv_len + tok (post-RoPE K, normed V).
//! ```
//!
//! Engine `GpuKvCache` uses separate f32 K/V buffers with RoPE applied at read time; the monolithic
//! cache stores **post-RoPE K** and **V** in the layout above. M1.2 packs CPU encoder prefill output.

use crate::config::ModelConfig;
use crate::dgq::DgqStore;
use crate::flags::progress_enabled;
use crate::kernels::sub::f16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::metal::GpuDecoderScratch;
use crate::metal::decoder::load_weight_cache_opt;
use crate::metal::device::MetalContext;
use crate::metal::encoder_extend::{extend_prefill_gpu, prefill_gpu};
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::step_kernel::{
    CANVAS, ModelLayout, N_LAYERS, StepFinishMode, StepSmokeConfig, VOCAB, build_layout,
    build_offsets_from_store, build_step_runtime, run_step_forward, step_params_from_sampler,
};
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::Model;
use crate::model::encoder::{EncoderPrefillInput, EncoderScratch};
use crate::model::kv_cache::KvCache;
use crate::safetensors::Error;
use crate::sample::{Rng, SamplerConfig, step_entropy_stats};
use crate::weights::WeightStore;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::path::Path;

use crate::kernels::sub::kv_quant::KvFormat;

/// Bytes of one (slot, head, K-or-V) row for the given format. MUST match
/// attention_device.metal `kv_row_bytes` (delegates to `KvFormat::row_bytes`).
pub fn kv_row_bytes(head_dim: u32, fmt: KvFormat) -> u64 {
    fmt.row_bytes(head_dim)
}

pub fn kv_region_bytes(n_kv_heads: u32, head_dim: u32, slots: usize, fmt: KvFormat) -> u64 {
    (slots as u64) * (n_kv_heads as u64) * 2 * kv_row_bytes(head_dim, fmt)
}

/// Per-layer KV slots — MUST agree with `build_layout`'s region offsets: full
/// layers are linear (max_seq slots), sliding layers a power-of-two ring
/// (only the last window-1 + canvas positions are ever live).
fn layer_slots(l: &crate::metal::step_kernel::LayerOffsets, max_seq: usize) -> usize {
    crate::metal::step_kernel::layer_kv_slots(l.is_full != 0, max_seq)
}

/// Absolute position -> slot within the layer's KV region.
#[inline]
fn kv_slot(l: &crate::metal::step_kernel::LayerOffsets, pos: usize) -> usize {
    if l.kv_ring_mask != 0 {
        pos & (l.kv_ring_mask as usize)
    } else {
        pos
    }
}

pub fn kv_cache_total_bytes(layout: &ModelLayout, max_seq: usize) -> u64 {
    let fmt = crate::flags::kv_format(max_seq);
    (0..N_LAYERS)
        .map(|i| {
            let l = &layout.layers[i];
            kv_region_bytes(l.n_kv_heads, l.head_dim, layer_slots(l, max_seq), fmt)
        })
        .sum()
}

/// Gather the live `[0, kv_len)` KV of every layer into a compact blob (per-layer
/// regions concatenated, no capacity tail) — for saving a conversation out of the
/// hot buffer. Full layers contribute `kv_len` slots; sliding (ring) layers cap
/// at their window (`kv_slot` is the stateless `pos & mask`, so the live ring is
/// exactly the first `min(kv_len, window)` physical slots). Inverse:
/// [`scatter_kv_prefix`], which must be called with the same `max_seq`/`kv_len`.
pub fn gather_kv_prefix(
    buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
    kv_len: usize,
) -> Vec<u8> {
    let fmt = crate::flags::kv_format(max_seq);
    let src =
        unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, buf.length()) };
    let mut out = Vec::new();
    for i in 0..N_LAYERS {
        let l = &layout.layers[i];
        let slots = kv_len.min(layer_slots(l, max_seq));
        let bytes = kv_region_bytes(l.n_kv_heads, l.head_dim, slots, fmt) as usize;
        let base = l.kv_region as usize;
        out.extend_from_slice(&src[base..base + bytes]);
    }
    out
}

/// Restore a blob from [`gather_kv_prefix`] into the KV buffer. Writes only each
/// layer's live prefix; slots past it are left as-is (never read before the next
/// prefill overwrites them — the same invariant a fresh prefill leaves). `max_seq`
/// and `kv_len` MUST match the gather, as they set each layer's slice length.
pub fn scatter_kv_prefix(
    buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
    kv_len: usize,
    blob: &[u8],
) {
    let fmt = crate::flags::kv_format(max_seq);
    let dst =
        unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u8, buf.length()) };
    let mut off = 0;
    for i in 0..N_LAYERS {
        let l = &layout.layers[i];
        let slots = kv_len.min(layer_slots(l, max_seq));
        let bytes = kv_region_bytes(l.n_kv_heads, l.head_dim, slots, fmt) as usize;
        let base = l.kv_region as usize;
        dst[base..base + bytes].copy_from_slice(&blob[off..off + bytes]);
        off += bytes;
    }
}

/// Pack CPU post-RoPE encoder KV into monolithic b4 layout (bf16, K then V per token).
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
        let kv_layer = kv.layer(layer).ok_or(Error::Format("missing kv layer"))?;
        if kv_layer.n_kv_heads as u32 != l.n_kv_heads || kv_layer.head_dim as u32 != l.head_dim {
            return Err(Error::Format("kv layer head dims mismatch"));
        }
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let fmt = crate::flags::kv_format(max_seq);
        if fmt == KvFormat::Q4 {
            return Err(Error::Format("q4 KV pack not implemented"));
        }
        let row_b = kv_row_bytes(l.head_dim, fmt) as usize;
        let slot_stride_b = 2 * nkv * row_b;
        let byte_base = l.kv_region as usize;
        if byte_base + layer_slots(l, max_seq) * slot_stride_b > dst.len() {
            return Err(Error::Format("monolithic kv buffer too small"));
        }
        let per_token = nkv * hd;
        for pos in 0..kv.kv_len {
            let slot_b = byte_base + kv_slot(l, pos) * slot_stride_b;
            for hh in 0..nkv {
                let ksrc = pos * per_token + hh * hd;
                if fmt == KvFormat::Q8 {
                    kv_q8_pack_row(
                        &mut dst[slot_b + hh * row_b..],
                        &kv_layer.keys[ksrc..ksrc + hd],
                    );
                    kv_q8_pack_row(
                        &mut dst[slot_b + (nkv + hh) * row_b..],
                        &kv_layer.values[ksrc..ksrc + hd],
                    );
                } else {
                    for d in 0..hd {
                        let k_bits = f32_to_f16_bits(kv_layer.keys[ksrc + d]);
                        let v_bits = f32_to_f16_bits(kv_layer.values[ksrc + d]);
                        let k_dst = slot_b + hh * row_b + d * 2;
                        let v_dst = slot_b + (nkv + hh) * row_b + d * 2;
                        dst[k_dst..k_dst + 2].copy_from_slice(&k_bits.to_le_bytes());
                        dst[v_dst..v_dst + 2].copy_from_slice(&v_bits.to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Quantize one head-vector into a q8 row (group-32 symmetric; mirrors the
/// Metal `kv_q8_store_group` bit-for-bit: f16-rounded scale, round-half-even).
fn kv_q8_pack_row(row: &mut [u8], src: &[f32]) {
    let hd = src.len();
    for g in 0..hd / 32 {
        let grp = &src[g * 32..g * 32 + 32];
        let mx = grp.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale_bits = f32_to_f16_bits((mx / 127.0).max(1e-8));
        let sf = f16_bits_to_f32(scale_bits);
        row[hd + g * 2..hd + g * 2 + 2].copy_from_slice(&scale_bits.to_le_bytes());
        for j in 0..32 {
            let q = (grp[j] / sf).round_ties_even().clamp(-127.0, 127.0);
            row[g * 32 + j] = (q as i8) as u8;
        }
    }
}

/// Dequantize one element of a q8 row (inverse of `kv_q8_pack_row`).
fn kv_q8_read(row: &[u8], d: usize, hd: usize) -> f32 {
    let s = f16_bits_to_f32(u16::from_le_bytes([
        row[hd + (d / 32) * 2],
        row[hd + (d / 32) * 2 + 1],
    ]));
    (row[d] as i8) as f32 * s
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
    let dst = unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u8, need) };
    dst.fill(0);
    pack_kv_cache_to_monolithic(dst, layout, kv, max_seq)
}

/// Read monolithic b4 KV prefix into CPU `KvCache` (inverse of `pack_kv_cache_to_monolithic`).
/// f16 sessions only (parity tooling; q8 sessions would need group dequant — all
/// current callers are small-max_seq harnesses).
pub fn read_monolithic_kv_prefix_to_cpu_cache(
    buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    cfg: &ModelConfig,
    kv_len: usize,
) -> Result<KvCache, Error> {
    let mut kv = KvCache::empty(&cfg.text_config)?;
    kv.kv_len = kv_len;
    for layer in 0..N_LAYERS.min(kv.layers.len()) {
        let l = &layout.layers[layer];
        let kv_layer = kv
            .layer_mut(layer)
            .ok_or(Error::Format("missing kv layer"))?;
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let per_token = nkv * hd;
        kv_layer.keys.resize(kv_len * per_token, 0.0);
        kv_layer.values.resize(kv_len * per_token, 0.0);
        let token_stride = nkv * hd * 2;
        let slot_base_region = l.kv_region as usize / 2;
        for pos in 0..kv_len {
            let slot_base = slot_base_region + kv_slot(l, pos) * token_stride;
            for hh in 0..nkv {
                for d in 0..hd {
                    let src_i = pos * per_token + hh * hd + d;
                    let k_dst = (slot_base + hh * hd + d) * 2;
                    let v_dst = (slot_base + nkv * hd + hh * hd + d) * 2;
                    kv_layer.keys[src_i] = f16_bits_to_f32(read_half_at(buf, k_dst));
                    kv_layer.values[src_i] = f16_bits_to_f32(read_half_at(buf, v_dst));
                }
            }
        }
    }
    Ok(kv)
}

#[derive(Debug)]
pub struct StepKvBf16CrossResult {
    pub kv_len: usize,
    pub layers: usize,
    pub max_kv_diff: f32,
    pub max_kv_diff_layer: usize,
    pub max_kv_diff_pos: usize,
    pub gpu_prefix_max_l0: f32,
    pub cpu_prefix_max_l0: f32,
}

/// Compare GPU monolithic encoder KV (.dgq) vs CPU bf16 encoder KV (safetensors).
pub fn run_step_kv_bf16_cross_parity(
    dgq_dir: &Path,
    bf16_dir: &Path,
    token_ids: &[u32],
    layers: usize,
    max_seq: usize,
) -> Result<StepKvBf16CrossResult, Error> {
    if token_ids.is_empty() {
        return Err(Error::Format("kv cross parity requires at least one token"));
    }
    let layers = layers.max(1).min(N_LAYERS);
    let dgq_store = DgqStore::open(dgq_dir)?;
    let layout = build_layout(&build_offsets_from_store(&dgq_store), max_seq);
    let bf16_model = Model::open(bf16_dir)?;
    if bf16_model.weights.is_quantized() {
        return Err(Error::Format(
            "bf16 cross parity requires bf16 safetensors dir",
        ));
    }
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
    let alloc_kv = || -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        ctx.device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Format("kv cross buffer alloc failed"))
    };
    let gpu_buf = alloc_kv()?;
    let cpu_buf = alloc_kv()?;
    let mut gpu_cache = MonolithicEncoderCache::open_opt(dgq_dir, CANVAS, max_seq, None)?;
    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut gpu_cache,
        token_ids,
        &gpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    let cpu_kv_len = prefill_monolithic_kv_cpu(
        &bf16_model.weights,
        &bf16_model.config,
        token_ids,
        &cpu_buf,
        &layout,
        max_seq,
    )?;
    if cpu_kv_len != kv_len {
        return Err(Error::Format("gpu/cpu bf16 cross kv_len mismatch"));
    }
    let (max_kv_diff, max_kv_diff_layer, max_kv_diff_pos) =
        monolithic_kv_prefix_max_diff(&gpu_buf, &cpu_buf, &layout, kv_len, layers);
    Ok(StepKvBf16CrossResult {
        kv_len,
        layers,
        max_kv_diff,
        max_kv_diff_layer,
        max_kv_diff_pos,
        gpu_prefix_max_l0: kvcache_prefix_max_abs(&gpu_buf, &layout, 0, kv_len),
        cpu_prefix_max_l0: kvcache_prefix_max_abs(&cpu_buf, &layout, 0, kv_len),
    })
}

fn pack_gpu_kv_prefix_to_monolithic(
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    token_count: usize,
    layers: usize,
    dst_pos: usize,
    src_pos: usize,
    fmt: KvFormat,
) -> Result<(), Error> {
    use crate::kernels::sub::pack_encoder_kv;
    use crate::metal::batch::begin_engine_batch;

    let pack_pipeline = if fmt == KvFormat::F16 {
        engine.kernels.pack_encoder_kv.pipeline.clone()
    } else {
        // q8 (q4 not yet wired) — the quantized pack pipeline.
        engine.kernels.pack_encoder_kv_q8.pipeline.clone()
    };
    let telemetry = engine.batch_telemetry();
    let batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        let (grid, tg) = pack_encoder_kv::dispatch_shape(token_count, nkv, hd, fmt);
        batch.dispatch_with_grid(&pack_pipeline, grid, tg, |enc| {
            pack_encoder_kv::bind_gpu_buffers(
                enc,
                &k_buf,
                &v_buf,
                kv_buf,
                token_count as u32,
                dst_pos as u32,
                nkv as u32,
                hd as u32,
                l.kv_region,
                src_pos as u32,
                l.kv_ring_mask,
            );
        });
    }
    batch.end()
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
        Self::open_opt(model_dir, canvas, max_seq_hint, None)
    }

    pub fn open_opt(
        model_dir: &Path,
        canvas: usize,
        max_seq_hint: usize,
        shared_dgq_blob: Option<std::sync::Arc<crate::metal::dgq_gpu::DgqGpuBlob>>,
    ) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        let model = Model::open(model_dir)?;
        let text = &model.config.text_config;
        let weights =
            load_weight_cache_opt(&model.weights, text, canvas, max_seq_hint, shared_dgq_blob)?;
        let engine = GpuDecoderEngine::new()?;
        if model.weights.is_quantized() {
            engine.set_encoder_gpu_moe(true);
        }
        let dec_scratch = GpuDecoderScratch::new(canvas, &model.config);
        if progress_enabled() {
            eprintln!(
                "monolithic-encoder: cache open {:.2?} (model + engine weights)",
                open_started.elapsed(),
            );
        }
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
    prefill_monolithic_kv_with_cache_timed(cache, token_ids, kv_buf, layout, max_seq, max_layers)
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
    let telemetry_was_on = cache.engine.telemetry_enabled();
    cache.engine.reset_forward_telemetry();
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
    {
        // Sync/readback profile of the engine prefill (each gpu_sync is a
        // commit+waitUntilCompleted with host round-trips — the suspected
        // dominant cost vs MLX's single-graph prefill).
        let tel = cache.engine.telemetry_handle();
        let t = tel.borrow();
        if progress_enabled() {
            eprintln!(
                "monolithic-prefill: engine telemetry gpu_syncs={} readback={:.2} MiB",
                t.gpu_syncs,
                t.gpu_readback_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
    if !telemetry_was_on {
        let _ = cache.engine.take_forward_telemetry();
    }

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
    let dst =
        unsafe { std::slice::from_raw_parts_mut(kv_buf.contents().as_ptr() as *mut u8, need) };
    dst.fill(0);

    let pack_started = std::time::Instant::now();
    pack_gpu_kv_prefix_to_monolithic(
        &mut cache.engine,
        gpu_kv,
        kv_buf,
        layout,
        kv_len,
        layers,
        0,
        0,
        crate::flags::kv_format(max_seq),
    )?;
    let kv_pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;
    let timing = MonolithicPrefillTiming {
        gpu_forward_ms,
        kv_pack_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1000.0,
    };
    if progress_enabled() {
        eprintln!(
            "monolithic-prefill: kv_len={kv_len} gpu_forward={gpu_forward_ms:.1}ms kv_pack={kv_pack_ms:.1}ms total={:.1}ms",
            timing.total_ms
        );
    }
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
        &mut cache, token_ids, kv_buf, layout, max_seq, max_layers,
    )?
    .0)
}

fn read_half_at(kv_buf: &ProtocolObject<dyn MTLBuffer>, byte_off: usize) -> u16 {
    let ptr = unsafe { kv_buf.contents().as_ptr().add(byte_off) } as *const u8;
    u16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }])
}

/// Read K plane for layer `L` for token positions `[0, total_kv)` from monolithic b4.
pub fn read_layer_k_cache_f32(
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    layer: usize,
    total_kv: usize,
) -> Vec<f32> {
    if layer >= N_LAYERS || total_kv == 0 {
        return Vec::new();
    }
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let per_token = nkv * hd;
    let token_stride = nkv * hd * 2;
    let byte_base = l.kv_region as usize;
    let mut keys = vec![0f32; total_kv * per_token];
    for pos in 0..total_kv {
        let slot_base = byte_base / 2 + pos * token_stride;
        for hh in 0..nkv {
            for d in 0..hd {
                let dst_i = pos * per_token + hh * hd + d;
                let k_byte = (slot_base + hh * hd + d) * 2;
                keys[dst_i] = f16_bits_to_f32(read_half_at(kv_buf, k_byte));
            }
        }
    }
    keys
}

/// Load monolithic b4 prefix `[0, kv_len)` into engine `GpuKvCache` (for extend
/// after prefill). GPU kernel (inverse of `pack_encoder_kv`), one batch for all
/// layers.
///
/// Ring-aware: sliding layers hold only the last `min(kv_len, ring)` positions
/// (`slot = pos & mask`), so only that live range is hydrated — into the SAME
/// absolute engine indices (the extend attention mask uses `pos_k = ki`, buffer
/// index == position). Engine positions below the live range are left as-is:
/// their K scores are sliding-window-masked (score overwritten, so even NaN K
/// is harmless) and their V rows are multiplied by an exactly-0.0 softmax
/// weight — safe because GpuKvCache buffers are zero-filled at allocation and
/// only ever written by forwards/hydrates (always finite; 0.0 * finite = 0.0).
///
/// (The pre-3285ebe-era CPU predecessor of this function read monolithic slots
/// LINEARLY — wrong past the ring wrap — and cost O(kv_len) scalar f16
/// conversions per call, O(n²) over a chunked delta.)
fn hydrate_gpu_kv_from_monolithic(
    engine: &mut GpuDecoderEngine,
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    max_seq: usize,
    kv_len: usize,
    gpu_kv: &mut crate::metal::GpuKvCache,
    layers: usize,
) -> Result<(), Error> {
    if kv_len == 0 {
        gpu_kv.reset_len();
        return Ok(());
    }
    use crate::kernels::sub::unpack_encoder_kv;
    use crate::metal::batch::begin_engine_batch;

    let fmt = crate::flags::kv_format(max_seq);
    let pipeline = if fmt == KvFormat::F16 {
        engine.kernels.unpack_encoder_kv.pipeline.clone()
    } else {
        // q8 (q4 not yet wired) — the quantized unpack pipeline.
        engine.kernels.unpack_encoder_kv_q8.pipeline.clone()
    };
    let telemetry = engine.batch_telemetry();
    let batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let live_from = if l.kv_ring_mask != 0 {
            kv_len.saturating_sub(layer_slots(l, max_seq))
        } else {
            0
        };
        let count = kv_len - live_from;
        if count == 0 {
            continue;
        }
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        let (grid, tg) = unpack_encoder_kv::dispatch_shape(count, nkv, hd);
        batch.dispatch_with_grid(&pipeline, grid, tg, |enc| {
            unpack_encoder_kv::bind_gpu_buffers(
                enc,
                kv_buf,
                &k_buf,
                &v_buf,
                count as u32,
                live_from as u32,
                nkv as u32,
                hd as u32,
                l.kv_region,
                live_from as u32,
                l.kv_ring_mask,
            );
        });
    }
    batch.end()?;
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
    let total_started = std::time::Instant::now();
    let text = &cache.model.config.text_config;
    let canvas = CANVAS;
    let layers = max_layers.min(text.num_hidden_layers);

    let mut enc_scratch = EncoderScratch::new(new_token_ids.len(), &cache.model.config);
    let encoder_kv_cap = (kv_len_before + new_token_ids.len()).min(max_seq);

    cache
        .dec_scratch
        .ensure_gpu_kv(&cache.engine.ctx.device, text, encoder_kv_cap, canvas)?;
    let mut gpu_kv = cache
        .dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Format("gpu kv cache missing"))?;
    let hydrate_started = std::time::Instant::now();
    hydrate_gpu_kv_from_monolithic(
        &mut cache.engine,
        kv_buf,
        layout,
        max_seq,
        kv_len_before,
        &mut gpu_kv,
        layers,
    )?;
    let hydrate_ms = hydrate_started.elapsed().as_secs_f64() * 1000.0;
    cache.dec_scratch.gpu_kv = Some(gpu_kv);

    let mut cpu_kv = KvCache::empty(text)?;
    cpu_kv.kv_len = kv_len_before;

    let forward_started = std::time::Instant::now();
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
    let forward_ms = forward_started.elapsed().as_secs_f64() * 1000.0;

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

    let pack_started = std::time::Instant::now();
    pack_gpu_kv_prefix_to_monolithic(
        &mut cache.engine,
        gpu_kv,
        kv_buf,
        layout,
        append_len,
        layers,
        kv_len_before,
        kv_len_before,
        crate::flags::kv_format(max_seq),
    )?;
    if progress_enabled() {
        eprintln!(
            "monolithic-extend: +{append_len} tok at kv={kv_len_before}: hydrate={hydrate_ms:.1}ms forward={forward_ms:.1}ms pack={:.1}ms total={:.1}ms",
            pack_started.elapsed().as_secs_f64() * 1000.0,
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(new_kv_len)
}

/// Multi-chunk engine extend: hydrate the engine KV from monolithic b4 ONCE,
/// then forward + pack `new_token_ids` in `CANVAS`-sized chunks. Byte-identical
/// KV to chaining [`extend_monolithic_kv_with_cache`] per chunk (same kernels,
/// same chunking, same f16 pack/unpack roundtrips at chunk boundaries) minus
/// that path's per-chunk re-hydration — which is O(prefix) per chunk, O(n²)
/// over a long delta. This is the production path for cross-turn deltas past
/// the fast-prefill trust cap (task #64).
pub fn extend_monolithic_kv_chunked(
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
    let total_started = std::time::Instant::now();
    let text = &cache.model.config.text_config;
    let canvas = CANVAS;
    let layers = max_layers.min(text.num_hidden_layers);
    let fmt = crate::flags::kv_format(max_seq);
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if kv_buf.length() < need {
        return Err(Error::Format("monolithic kv buffer too small"));
    }

    let encoder_kv_cap = (kv_len_before + new_token_ids.len()).min(max_seq);
    cache
        .dec_scratch
        .ensure_gpu_kv(&cache.engine.ctx.device, text, encoder_kv_cap, canvas)?;
    let mut gpu_kv = cache
        .dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Format("gpu kv cache missing"))?;
    let hydrate_started = std::time::Instant::now();
    hydrate_gpu_kv_from_monolithic(
        &mut cache.engine,
        kv_buf,
        layout,
        max_seq,
        kv_len_before,
        &mut gpu_kv,
        layers,
    )?;
    let hydrate_ms = hydrate_started.elapsed().as_secs_f64() * 1000.0;
    cache.dec_scratch.gpu_kv = Some(gpu_kv);

    let mut cpu_kv = KvCache::empty(text)?;
    cpu_kv.kv_len = kv_len_before;
    // Scratch sized once for the largest chunk (forwards slice to chunk len).
    let mut enc_scratch = EncoderScratch::new(canvas.min(new_token_ids.len()), &cache.model.config);

    let mut off = kv_len_before;
    let mut forward_ms = 0f64;
    let mut pack_ms = 0f64;
    for chunk in new_token_ids.chunks(canvas) {
        let forward_started = std::time::Instant::now();
        extend_prefill_gpu(
            &cache.model.weights,
            &cache.model.config,
            &mut cpu_kv,
            chunk,
            &mut enc_scratch,
            &mut cache.dec_scratch,
            &mut cache.weights,
            &mut cache.engine,
            Some(layers),
        )?;
        forward_ms += forward_started.elapsed().as_secs_f64() * 1000.0;

        let gpu_kv = cache
            .dec_scratch
            .gpu_kv
            .as_ref()
            .ok_or(Error::Format("gpu kv missing after extend"))?;
        let pack_started = std::time::Instant::now();
        pack_gpu_kv_prefix_to_monolithic(
            &mut cache.engine,
            gpu_kv,
            kv_buf,
            layout,
            chunk.len(),
            layers,
            off,
            off,
            fmt,
        )?;
        pack_ms += pack_started.elapsed().as_secs_f64() * 1000.0;
        off += chunk.len();
    }
    if progress_enabled() {
        eprintln!(
            "monolithic-extend-chunked: +{} tok at kv={kv_len_before}: hydrate={hydrate_ms:.1}ms forward={forward_ms:.1}ms pack={pack_ms:.1}ms total={:.1}ms",
            new_token_ids.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(off)
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
    let token_stride = nkv * hd * 2;
    let byte_base = l.kv_region as usize;
    let slot_base = byte_base / 2;
    let mut max_abs = 0.0f32;
    for pos in 0..kv_len {
        let start = (slot_base + pos * token_stride) * 2;
        let end = start + token_stride * 2;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                kv_buf.contents().as_ptr().add(start) as *const u8,
                end - start,
            )
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
        prefill_token_ids: None,
        no_early_stop: false,
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
        .newBufferWithLength_options(kv_bytes, objc2_metal::MTLResourceOptions::StorageModeShared)
        .ok_or(Error::Format("kv audit buffer alloc failed"))?;
    let actual_kv = prefill_monolithic_kv(model_dir, &prompt, &kv_buf, &layout, max_seq, layers)?;
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
        let extend_buf = ctx
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

fn copy_metal_buffer(
    dst: &ProtocolObject<dyn MTLBuffer>,
    src: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.contents().as_ptr() as *const u8,
            dst.contents().as_ptr() as *mut u8,
            len,
        );
    }
}

/// Max abs f32 diff between two monolithic b4 prefixes `[0, kv_len)` over `layers`.
pub fn monolithic_kv_prefix_max_diff(
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    kv_len: usize,
    layers: usize,
) -> (f32, usize, usize) {
    let mut max_diff = 0.0f32;
    let mut max_layer = 0usize;
    let mut max_pos = 0usize;
    for layer in 0..layers.min(N_LAYERS) {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let token_stride = nkv * hd * 2;
        let slot_base = l.kv_region as usize / 2;
        for pos in 0..kv_len {
            for hidx in 0..token_stride {
                let byte = (slot_base + pos * token_stride + hidx) * 2;
                let va = f16_bits_to_f32(read_half_at(a, byte));
                let vb = f16_bits_to_f32(read_half_at(b, byte));
                let d = (va - vb).abs();
                if d > max_diff {
                    max_diff = d;
                    max_layer = layer;
                    max_pos = pos;
                }
            }
        }
    }
    (max_diff, max_layer, max_pos)
}

/// One denoise step with an external b4 KV prefix; returns min position entropy (nats).
pub fn step_min_entropy_with_kv(
    model_dir: &Path,
    kv_src: &ProtocolObject<dyn MTLBuffer>,
    kv_len: usize,
    layers: usize,
    max_seq: usize,
    seed: u64,
) -> Result<f32, Error> {
    let cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: kv_len as u32,
        seed,
        max_seq,
        finish: StepFinishMode::Full,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    let (mut rt, _) = build_step_runtime(model_dir, &cfg)?;
    let kv_bytes = kv_cache_total_bytes(rt.layout(), max_seq) as usize;
    copy_metal_buffer(rt.kvcache(), kv_src, kv_bytes);
    rt.set_kv_len(kv_len as u32);
    let params = step_params_from_sampler(&SamplerConfig::default(), kv_len as u32, false, 1);
    let mut rng = Rng::new(seed);
    rt.reset_block(VOCAB, &mut rng, params);
    rt.run_denoise_step()?;
    let st = rt.read_canvas_state();
    Ok(step_entropy_stats(&st.entropy, &st.accept).min_entropy)
}

/// Max abs half in K or V plane only for layer `L` prefix `[0, kv_len)`.
pub fn kvcache_plane_max_abs(
    kv_buf: &ProtocolObject<dyn MTLBuffer>,
    layout: &ModelLayout,
    layer: usize,
    kv_len: usize,
    plane: u8,
) -> f32 {
    if kv_len == 0 || layer >= N_LAYERS || plane > 1 {
        return 0.0;
    }
    let l = &layout.layers[layer];
    let nkv = l.n_kv_heads as usize;
    let hd = l.head_dim as usize;
    let plane_slots = nkv * hd;
    let token_stride = plane_slots * 2;
    let slot_base = l.kv_region as usize / 2;
    let plane_off = if plane == 0 { 0 } else { plane_slots };
    let mut max_abs = 0.0f32;
    for pos in 0..kv_len {
        let start = (slot_base + pos * token_stride + plane_off) * 2;
        let end = start + plane_slots * 2;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                kv_buf.contents().as_ptr().add(start) as *const u8,
                end - start,
            )
        };
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            max_abs = max_abs.max(f16_bits_to_f32(bits).abs());
        }
    }
    max_abs
}

#[derive(Debug)]
pub struct StepAttnProbeResult {
    pub kv_len: usize,
    pub layer: usize,
    pub k_plane_max_l0: f32,
    pub v_plane_max_l0: f32,
    pub q_norm_weight_mean_abs: f32,
    pub q_norm_weight_rms: f32,
    pub k_norm_weight_mean_abs: f32,
    pub k_norm_weight_rms: f32,
    pub cpu_raw_dot_max: f32,
    pub cpu_raw_dot_min: f32,
    pub cpu_mean_softmax_entropy: f32,
    pub cpu_mean_max_prob: f32,
    pub cpu_q_head_rms: f32,
    pub cpu_k_head_rms: f32,
    pub canvas_len: usize,
    pub attn_keys_t: usize,
    pub cpu_keys_per_row: usize,
    pub cpu_mean_weight_sum: f32,
}

/// P1.0: attention magnitude + KV plane probe (monolithic b4 + CPU score model with real QK-norm weights).
pub fn run_step_attn_probe(
    model_dir: &Path,
    token_ids: &[u32],
    layer: usize,
    seed: u64,
    max_seq: usize,
) -> Result<StepAttnProbeResult, Error> {
    use crate::kernels::cpu::rms_norm;
    use crate::model::attention::{
        AttentionParams, attn_score_stats_decoder, qk_norm_weight_stats,
    };
    use crate::model::mask::DecoderAttnMask;
    use crate::sample::Rng;

    if token_ids.is_empty() {
        return Err(Error::Format("step-attn-probe requires prompt tokens"));
    }
    let layer = layer.min(N_LAYERS.saturating_sub(1));
    let canvas_len = CANVAS;
    let cfg = ModelConfig::load(model_dir)?;
    let text = &cfg.text_config;
    let dgq = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&dgq), max_seq);
    let params = AttentionParams::for_layer(text, layer)?;

    let native_buf = MetalContext::new()?
        .device
        .newBufferWithLength_options(
            kv_cache_total_bytes(&layout, max_seq) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or(Error::Format("kv buffer alloc failed"))?;
    let mut native_cache = MonolithicEncoderCache::open_opt(model_dir, canvas_len, max_seq, None)?;
    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut native_cache,
        token_ids,
        &native_buf,
        &layout,
        max_seq,
        layer + 1,
    )?;

    let p = format!("model.decoder.layers.{layer}.self_attn");
    let q_norm_w = dgq.tensor_f32(&format!("{p}.q_norm.weight"))?;
    let k_norm_w = dgq.tensor_f32(&format!("{p}.k_norm.weight"))?;
    let (q_na, q_nr) = qk_norm_weight_stats(&q_norm_w);
    let (k_na, k_nr) = qk_norm_weight_stats(&k_norm_w);

    // Build synthetic post-QK-norm Q/K from monolithic K-cache prefix + canvas-sized query grid.
    let hd = params.head_dim;
    let nkv = params.n_kv_heads;
    let nheads = params.n_heads;
    let total_kv = kv_len + canvas_len;
    let kv_dim = nkv * hd;
    let q_dim = nheads * hd;
    let l = &layout.layers[layer];
    let slot_base = l.kv_region as usize / 2;
    let token_stride = nkv * hd * 2;

    let mut k_full = vec![0.0f32; total_kv * kv_dim];
    let mut v_full = vec![0.0f32; total_kv * kv_dim];
    for pos in 0..kv_len {
        let k_off = pos * kv_dim;
        let byte_k = (slot_base + pos * token_stride) * 2;
        let byte_v = byte_k + nkv * hd * 2;
        for i in 0..nkv * hd {
            k_full[k_off + i] = f16_bits_to_f32(read_half_at(&native_buf, byte_k + i * 2));
            v_full[k_off + i] = f16_bits_to_f32(read_half_at(&native_buf, byte_v + i * 2));
        }
    }

    let mut rng = Rng::new(seed);
    let mut q = vec![0.0f32; canvas_len * q_dim];
    let mut k_canvas = vec![0.0f32; canvas_len * kv_dim];
    let eps = text.rms_norm_eps as f32;
    let mut head = vec![0.0f32; hd];
    for tok in 0..canvas_len {
        for h in 0..nheads {
            let off = (tok * nheads + h) * hd;
            for d in 0..hd {
                head[d] = (rng.next_f32() - 0.5) * 2.0;
            }
            rms_norm(&mut q[off..off + hd], &head, &q_norm_w, eps);
        }
        for h in 0..nkv {
            let off = (tok * nkv + h) * hd;
            for d in 0..hd {
                head[d] = (rng.next_f32() - 0.5) * 2.0;
            }
            rms_norm(&mut k_canvas[off..off + hd], &head, &k_norm_w, eps);
        }
    }
    k_full[kv_len * kv_dim..].copy_from_slice(&k_canvas);

    let mask = DecoderAttnMask::all_valid(canvas_len, kv_len);
    let stats = attn_score_stats_decoder(&q, &k_full, canvas_len, total_kv, &params, &mask);
    let q_rms = head_rms_probe(&q, canvas_len, nheads, hd);
    let k_rms = head_rms_probe(&k_full, total_kv, nkv, hd);

    Ok(StepAttnProbeResult {
        kv_len,
        layer,
        k_plane_max_l0: kvcache_plane_max_abs(&native_buf, &layout, 0, kv_len, 0),
        v_plane_max_l0: kvcache_plane_max_abs(&native_buf, &layout, 0, kv_len, 1),
        q_norm_weight_mean_abs: q_na,
        q_norm_weight_rms: q_nr,
        k_norm_weight_mean_abs: k_na,
        k_norm_weight_rms: k_nr,
        cpu_raw_dot_max: stats.raw_dot_max,
        cpu_raw_dot_min: stats.raw_dot_min,
        cpu_mean_softmax_entropy: stats.mean_row_softmax_entropy,
        cpu_mean_max_prob: stats.mean_row_max_prob,
        cpu_q_head_rms: q_rms,
        cpu_k_head_rms: k_rms,
        canvas_len,
        attn_keys_t: total_kv,
        cpu_keys_per_row: stats.keys_per_row,
        cpu_mean_weight_sum: stats.mean_weight_sum,
    })
}

fn head_rms_probe(buf: &[f32], seq_len: usize, n_heads: usize, head_dim: usize) -> f32 {
    let mut sum = 0.0f32;
    let n = (seq_len * n_heads).max(1) as f32;
    for s in 0..seq_len {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                let v = buf[off + d];
                ss += v * v;
            }
            sum += (ss / head_dim as f32).sqrt();
        }
    }
    sum / n
}

#[derive(Debug)]
pub struct StepKvParityResult {
    pub kv_len: usize,
    pub layers: usize,
    pub prefix_max_l0: f32,
    pub prefix_max_l0_b: f32,
    pub max_kv_diff: f32,
    pub max_kv_diff_layer: usize,
    pub max_kv_diff_pos: usize,
    pub min_ent: f32,
    pub min_ent_b: f32,
    pub min_ent_diff: f32,
    pub entropy_pass: bool,
    pub ln_vocab: f32,
    pub pass: bool,
}

/// Compare monolithic b4 KV from two independent encoder prefills, then one denoise step each.
pub fn run_step_kv_parity(
    model_dir: &Path,
    token_ids: &[u32],
    layers: usize,
    max_seq: usize,
    seed: u64,
) -> Result<StepKvParityResult, Error> {
    if token_ids.is_empty() {
        return Err(Error::Format("step-kv-parity requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Format("token_ids exceed max_seq"));
    }
    let layers = layers.max(1).min(N_LAYERS);
    let store = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;

    let alloc_kv = || -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        ctx.device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Format("kv parity buffer alloc failed"))
    };

    let mut cache_a = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;
    let mut cache_b = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;

    let buf_a = alloc_kv()?;
    let buf_b = alloc_kv()?;

    let (kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut cache_a,
        token_ids,
        &buf_a,
        &layout,
        max_seq,
        layers,
    )?;
    let (kv_len_b, _) = prefill_monolithic_kv_with_cache(
        &mut cache_b,
        token_ids,
        &buf_b,
        &layout,
        max_seq,
        layers,
    )?;
    if kv_len_b != kv_len {
        return Err(Error::Format("encoder prefill kv_len mismatch"));
    }

    let prefix_max_l0 = kvcache_prefix_max_abs(&buf_a, &layout, 0, kv_len);
    let prefix_max_l0_b = kvcache_prefix_max_abs(&buf_b, &layout, 0, kv_len);
    let (max_kv_diff, max_kv_diff_layer, max_kv_diff_pos) =
        monolithic_kv_prefix_max_diff(&buf_a, &buf_b, &layout, kv_len, layers);

    let min_ent = step_min_entropy_with_kv(model_dir, &buf_a, kv_len, layers, max_seq, seed)?;
    let min_ent_b = step_min_entropy_with_kv(model_dir, &buf_b, kv_len, layers, max_seq, seed)?;

    let ln_vocab = (VOCAB as f32).ln();
    let min_ent_diff = (min_ent_b - min_ent).abs();
    let kv_ok = max_kv_diff < 0.5 && prefix_max_l0_b > 1e-4;
    const MAX_MIN_ENT_DIFF: f32 = 0.25;
    let entropy_pass = min_ent_diff < MAX_MIN_ENT_DIFF;
    let pass = kv_ok && entropy_pass;

    Ok(StepKvParityResult {
        kv_len,
        layers,
        prefix_max_l0,
        prefix_max_l0_b,
        max_kv_diff,
        max_kv_diff_layer,
        max_kv_diff_pos,
        min_ent,
        min_ent_b,
        min_ent_diff,
        entropy_pass,
        ln_vocab,
        pass,
    })
}

#[cfg(test)]
/// Compare monolithic b4 KV from encoder prefill with CPU MoE vs grouped GPU MoE (same dense path).
pub fn run_encoder_moe_kv_parity(
    model_dir: &Path,
    token_ids: &[u32],
    layers: usize,
    max_seq: usize,
) -> Result<(f32, usize, usize), Error> {
    if token_ids.is_empty() {
        return Err(Error::Format(
            "encoder moe kv parity requires at least one token",
        ));
    }
    let layers = layers.max(1).min(N_LAYERS);
    let store = DgqStore::open(model_dir)?;
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new()?;
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;

    let alloc_kv = || -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        ctx.device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Format("encoder moe kv buffer alloc failed"))
    };

    let mut cpu_cache = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;
    cpu_cache.engine.set_encoder_gpu_moe(false);
    let mut gpu_cache = MonolithicEncoderCache::open_opt(model_dir, CANVAS, max_seq, None)?;
    gpu_cache.engine.set_encoder_gpu_moe(true);

    let cpu_buf = alloc_kv()?;
    let gpu_buf = alloc_kv()?;

    let (cpu_kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut cpu_cache,
        token_ids,
        &cpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    let (gpu_kv_len, _) = prefill_monolithic_kv_with_cache(
        &mut gpu_cache,
        token_ids,
        &gpu_buf,
        &layout,
        max_seq,
        layers,
    )?;
    if gpu_kv_len != cpu_kv_len {
        return Err(Error::Format("cpu/gpu encoder moe prefill kv_len mismatch"));
    }
    let (max_diff, layer, pos) =
        monolithic_kv_prefix_max_diff(&cpu_buf, &gpu_buf, &layout, cpu_kv_len, layers);
    Ok((max_diff, layer, pos))
}

#[cfg(all(test, target_os = "macos"))]
mod engine_extend_bench_tests {
    use super::*;
    use crate::metal::device::MetalContext;

    fn model_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new("model/diffusiongemma-q4emb");
        if dir.join("model.dgq.json").exists() {
            Some(dir.to_path_buf())
        } else {
            eprintln!("skip: model/diffusiongemma-q4emb missing");
            None
        }
    }

    fn synth_ids(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 131 + 7) % VOCAB) as u32).collect()
    }

    /// Ring-aware max |Δ| between two monolithic KV buffers over each layer's
    /// LIVE slots (full layers: all of [0, kv_len); sliding: the last
    /// min(kv_len, ring) positions). `monolithic_kv_prefix_max_diff` reads
    /// positions linearly and is only valid below the ring wrap.
    fn live_kv_max_diff(
        a: &ProtocolObject<dyn MTLBuffer>,
        b: &ProtocolObject<dyn MTLBuffer>,
        layout: &ModelLayout,
        max_seq: usize,
        kv_len: usize,
        layers: usize,
    ) -> (f32, usize, usize) {
        let mut max_diff = 0.0f32;
        let mut max_layer = 0usize;
        let mut max_pos = 0usize;
        for layer in 0..layers.min(N_LAYERS) {
            let l = &layout.layers[layer];
            let nkv = l.n_kv_heads as usize;
            let hd = l.head_dim as usize;
            let token_stride = nkv * hd * 2;
            let slot_base = l.kv_region as usize / 2;
            let live_from = if l.kv_ring_mask != 0 {
                kv_len.saturating_sub(layer_slots(l, max_seq))
            } else {
                0
            };
            for pos in live_from..kv_len {
                let slot = kv_slot(l, pos);
                for hidx in 0..token_stride {
                    let byte = (slot_base + slot * token_stride + hidx) * 2;
                    let va = f16_bits_to_f32(read_half_at(a, byte));
                    let vb = f16_bits_to_f32(read_half_at(b, byte));
                    let d = (va - vb).abs();
                    if d > max_diff {
                        max_diff = d;
                        max_layer = layer;
                        max_pos = pos;
                    }
                }
            }
        }
        (max_diff, max_layer, max_pos)
    }

    /// E12 baseline: extend-vs-full-prefill KV parity below (1500) and above
    /// (3000) the sliding ring wrap (2048), plus hydrate/forward/pack timing
    /// per 256-token extend (printed by the monolithic-extend instrumentation).
    #[test]
    #[ignore = "model-gated bench: cargo test --release engine_extend_baseline -- --ignored --nocapture"]
    fn engine_extend_baseline() {
        let Some(dir) = model_dir() else { return };
        let max_seq = 4096usize;
        let layers = N_LAYERS;
        let store = DgqStore::open(&dir).expect("dgq");
        let layout = build_layout(&build_offsets_from_store(&store), max_seq);
        let ctx = MetalContext::new().expect("metal");
        let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
        let alloc = || {
            ctx.device
                .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
                .expect("kv buf")
        };
        let mut cache = MonolithicEncoderCache::open(&dir, CANVAS, max_seq).expect("cache");

        for &(total, label) in &[(1500usize, "below-ring"), (3000usize, "above-ring")] {
            let ids = synth_ids(total);
            let split = total - 2 * CANVAS;
            let buf_full = alloc();
            let buf_ext = alloc();

            let t = std::time::Instant::now();
            let (kv_full, _) = prefill_monolithic_kv_with_cache(
                &mut cache, &ids, &buf_full, &layout, max_seq, layers,
            )
            .expect("full prefill");
            let full_s = t.elapsed().as_secs_f64();

            let (kv_pre, _) = prefill_monolithic_kv_with_cache(
                &mut cache,
                &ids[..split],
                &buf_ext,
                &layout,
                max_seq,
                layers,
            )
            .expect("split prefill");
            assert_eq!(kv_pre, split);
            let t = std::time::Instant::now();
            let mut off = split;
            for chunk in ids[split..].chunks(CANVAS) {
                off = extend_monolithic_kv_with_cache(
                    &mut cache, &buf_ext, &layout, off, chunk, max_seq, layers,
                )
                .expect("extend");
            }
            let ext_s = t.elapsed().as_secs_f64();
            assert_eq!(off, total);
            assert_eq!(kv_full, total);

            // Chunked (hydrate-once) extend — must be byte-identical to the
            // per-chunk path (same kernels + chunk boundaries).
            let buf_chunked = alloc();
            let (kv_pre2, _) = prefill_monolithic_kv_with_cache(
                &mut cache,
                &ids[..split],
                &buf_chunked,
                &layout,
                max_seq,
                layers,
            )
            .expect("split prefill 2");
            assert_eq!(kv_pre2, split);
            let t = std::time::Instant::now();
            let off2 = extend_monolithic_kv_chunked(
                &mut cache,
                &buf_chunked,
                &layout,
                split,
                &ids[split..],
                max_seq,
                layers,
            )
            .expect("chunked extend");
            let chunked_s = t.elapsed().as_secs_f64();
            assert_eq!(off2, total);

            let (diff, dl, dp) =
                live_kv_max_diff(&buf_full, &buf_ext, &layout, max_seq, total, layers);
            let (cdiff, cdl, cdp) =
                live_kv_max_diff(&buf_ext, &buf_chunked, &layout, max_seq, total, layers);
            eprintln!(
                "[{label}] total={total} full_prefill={full_s:.2}s ({:.1} ms/tok) extend(2x256 @{split})={ext_s:.2}s ({:.1} ms/tok) chunked={chunked_s:.2}s live_max_diff={diff:.6} @L{dl} pos{dp} chunked_vs_perchunk={cdiff:.6} @L{cdl} pos{cdp}",
                full_s * 1000.0 / total as f64,
                ext_s * 1000.0 / (2.0 * CANVAS as f64),
            );
            assert_eq!(
                cdiff, 0.0,
                "[{label}] chunked extend != per-chunk extend: {cdiff} @ L{cdl} pos {cdp}"
            );
            assert!(
                diff < 0.02,
                "[{label}] extend diverges from full prefill: {diff} @ L{dl} pos {dp}"
            );
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod encoder_moe_kv_tests {
    use super::*;
    use crate::chat_template::{ChatFormatOptions, ChatTurn, format_chat_token_ids};
    use crate::metal::device::MetalContext;
    use crate::tokenizer::Tokenizer;

    fn calgary_prefill(model_dir: &Path) -> Vec<u32> {
        let tok = Tokenizer::load(&model_dir.join("tokenizer.json")).expect("tokenizer");
        format_chat_token_ids(
            &tok,
            &[ChatTurn::user("How can I get from Calgary to Namibia?")],
            &ChatFormatOptions::default(),
        )
        .expect("prefill ids")
    }

    #[test]
    fn nvfp4_encoder_prefill_long_prompt_kv_finite() {
        let dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let ids = calgary_prefill(dir);
        assert!(
            ids.len() >= 20,
            "expected long chat prompt, got {}",
            ids.len()
        );
        let ctx = MetalContext::new().expect("metal");
        let layout = build_layout(
            &build_offsets_from_store(&DgqStore::open(dir).expect("dgq")),
            512,
        );
        let kv_bytes = kv_cache_total_bytes(&layout, 512) as usize;
        let kv_buf = ctx
            .device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .expect("kv buf");
        let mut cache =
            MonolithicEncoderCache::open_opt(dir, CANVAS, 512, None).expect("encoder cache");
        cache.engine.set_encoder_gpu_moe(true);
        let (kv_len, _) =
            prefill_monolithic_kv_with_cache(&mut cache, &ids, &kv_buf, &layout, 512, 2)
                .expect("prefill");
        for layer in 0..2 {
            let k_max = kvcache_plane_max_abs(&kv_buf, &layout, layer, kv_len, 0);
            eprintln!("nvfp4 prefill L{layer}: kv_len={kv_len} k_max={k_max:.4}");
            assert!(
                k_max > 1e-4,
                "layer {layer} K prefix looks unset (max={k_max})"
            );
        }
    }

    #[test]
    fn encoder_moe_kv_matches_cpu_for_calgary_prompt() {
        let dir = std::path::Path::new("/tmp/quantized-weights");
        if !dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let ids = calgary_prefill(dir);
        assert_eq!(ids.len(), 22);
        let (max_diff, layer, pos) = run_encoder_moe_kv_parity(dir, &ids, 30, 512).expect("parity");
        eprintln!(
            "encoder moe kv parity (Calgary, 22 tok): max_diff={max_diff:.6} layer={layer} pos={pos}"
        );
        assert!(
            max_diff < 0.02,
            "cpu vs gpu encoder MoE KV diverged: {max_diff} @ L{layer} pos {pos}"
        );
    }
}
