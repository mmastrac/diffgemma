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
use crate::flags::progress_enabled;
use crate::metal::GpuDecoderScratch;
use crate::metal::decoder::load_weight_cache_opt;
use crate::metal::encoder_extend::{extend_prefill_gpu, prefill_gpu};
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::step_kernel::{CANVAS, ModelLayout, N_LAYERS};
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::Model;
use crate::model::encoder::{EncoderPrefillInput, EncoderScratch};
use crate::model::kv_cache::KvCache;
use crate::safetensors::Error;
use crate::shaders::f16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::weights::WeightStore;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::path::Path;

use crate::shaders::kv_quant::KvFormat;

// Test-only: the `engine_extend_bench_tests` / `encoder_moe_kv_tests` mods reach
// these via `use super::*`. Their production users moved to `step_kv_audits`, so
// gate the imports on test to keep the release build warning-clean.
#[cfg(test)]
use crate::dgq::DgqStore;
#[cfg(test)]
use crate::metal::step_kernel::{
    StepFinishMode, StepSmokeConfig, VOCAB, build_layout, build_offsets_from_store,
    build_step_runtime,
};
#[cfg(test)]
use objc2_metal::{MTLDevice, MTLResourceOptions};

// KV-cache audit / probe harnesses (CLI step-kv-* / step-attn-probe
// subcommands). Split out for size (backlog item 4); a child module, so it sees
// this module's private items via ancestry. Re-exported flat so the existing
// `step_kv::<fn>` paths keep resolving.
#[path = "step_kv_audits.rs"]
mod audits;
pub use audits::*;

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
#[allow(dead_code)]
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
    use crate::metal::batch::begin_engine_batch;
    use crate::shaders::pack_encoder_kv;

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
    use crate::metal::batch::begin_engine_batch;
    use crate::shaders::unpack_encoder_kv;

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

    /// Exactness gate for the GPU hydrate: engine f32 K/V after
    /// `hydrate_gpu_kv_from_monolithic` must equal the f16-widened monolithic
    /// live slots bit-for-bit (f16 -> f32 widening is exact), with the ring
    /// mapping applied on the source side. This is the ground truth for the
    /// ring fix — end-to-end extend-vs-full diffs are chaos-amplified by the
    /// forward and cannot distinguish mapping bugs from f16 boundary rounding.
    fn assert_hydrate_exact(
        cache: &mut MonolithicEncoderCache,
        kv_buf: &ProtocolObject<dyn MTLBuffer>,
        layout: &ModelLayout,
        max_seq: usize,
        kv_len: usize,
        layers: usize,
    ) {
        let text = &cache.model.config.text_config;
        cache
            .dec_scratch
            .ensure_gpu_kv(&cache.engine.ctx.device, text, kv_len, CANVAS)
            .expect("ensure gpu kv");
        let mut gpu_kv = cache.dec_scratch.gpu_kv.take().expect("gpu kv");
        hydrate_gpu_kv_from_monolithic(
            &mut cache.engine,
            kv_buf,
            layout,
            max_seq,
            kv_len,
            &mut gpu_kv,
            layers,
        )
        .expect("hydrate");
        for layer in 0..layers {
            let l = &layout.layers[layer];
            let nkv = l.n_kv_heads as usize;
            let hd = l.head_dim as usize;
            let per_token = nkv * hd;
            let token_stride = nkv * hd * 2;
            let slot_base = l.kv_region as usize / 2;
            let live_from = if l.kv_ring_mask != 0 {
                kv_len.saturating_sub(layer_slots(l, max_seq))
            } else {
                0
            };
            let (k_buf, v_buf) = gpu_kv.layer_buffers(layer).expect("bufs");
            let k_eng = unsafe {
                std::slice::from_raw_parts(
                    k_buf.contents().as_ptr() as *const f32,
                    kv_len * per_token,
                )
            };
            let v_eng = unsafe {
                std::slice::from_raw_parts(
                    v_buf.contents().as_ptr() as *const f32,
                    kv_len * per_token,
                )
            };
            let mut bad = 0usize;
            for pos in live_from..kv_len {
                let slot = kv_slot(l, pos);
                for i in 0..per_token {
                    let k_exp = f16_bits_to_f32(read_half_at(
                        kv_buf,
                        (slot_base + slot * token_stride + i) * 2,
                    ));
                    let v_exp = f16_bits_to_f32(read_half_at(
                        kv_buf,
                        (slot_base + slot * token_stride + per_token + i) * 2,
                    ));
                    if k_eng[pos * per_token + i].to_bits() != k_exp.to_bits()
                        || v_eng[pos * per_token + i].to_bits() != v_exp.to_bits()
                    {
                        bad += 1;
                    }
                }
            }
            assert_eq!(
                bad, 0,
                "hydrate not exact: layer {layer} kv_len={kv_len} live_from={live_from} bad={bad}"
            );
        }
        cache.dec_scratch.gpu_kv = Some(gpu_kv);
        eprintln!("hydrate exact at kv_len={kv_len} (all {layers} layers, live slots bit-equal)");
    }

    /// E12 baseline: extend-vs-full-prefill KV diffs below (1500) and above
    /// (3000) the sliding ring wrap (2048) — DIAGNOSTIC (chunk boundaries
    /// expose the prefix through f16 pack/unpack; the forward chaos-amplifies
    /// that, so nonzero diffs are physics, not defects). The hard gates are
    /// (a) hydrate exactness incl. ring mapping, (b) extend completes and
    /// produces finite KV. Timing per extend printed by the monolithic-extend
    /// instrumentation.
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
            // Hard gate: hydrate exactness (incl. ring mapping above 2048).
            assert_hydrate_exact(&mut cache, &buf_ext, &layout, max_seq, split, layers);
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
            // Diffs above are diagnostics (chaos-amplified f16 chunk-boundary
            // rounding — per-chunk re-hydrates the prior chunk as f16 while
            // hydrate-once keeps it f32, and full prefill never rounds).
            // Gate: finite KV in every live slot the extends wrote.
            for (name, buf) in [("perchunk", &buf_ext), ("chunked", &buf_chunked)] {
                for layer in 0..layers {
                    let l = &layout.layers[layer];
                    let nkv = l.n_kv_heads as usize;
                    let hd = l.head_dim as usize;
                    let token_stride = nkv * hd * 2;
                    let slot_base = l.kv_region as usize / 2;
                    let live_from = if l.kv_ring_mask != 0 {
                        total.saturating_sub(layer_slots(l, max_seq))
                    } else {
                        0
                    };
                    for pos in live_from..total {
                        let slot = kv_slot(l, pos);
                        for i in 0..token_stride {
                            let v = f16_bits_to_f32(read_half_at(
                                buf,
                                (slot_base + slot * token_stride + i) * 2,
                            ));
                            assert!(
                                v.is_finite(),
                                "[{label}] {name} non-finite KV @ L{layer} pos {pos}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Ring-fix gate, fast: prefill past the sliding ring wrap (2488 > 2048),
    /// then assert the GPU hydrate reproduces every LIVE monolithic slot
    /// bit-exactly in the engine cache (ring-mapped source, linear dest).
    /// The old CPU hydrate read slots linearly and failed exactly this.
    #[test]
    #[ignore = "model-gated: cargo test --release engine_hydrate_ring_exactness -- --ignored --nocapture"]
    fn engine_hydrate_ring_exactness() {
        let Some(dir) = model_dir() else { return };
        let max_seq = 4096usize;
        let store = DgqStore::open(&dir).expect("dgq");
        let layout = build_layout(&build_offsets_from_store(&store), max_seq);
        let ctx = MetalContext::new().expect("metal");
        let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
        let kv_buf = ctx
            .device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .expect("kv buf");
        let mut cache = MonolithicEncoderCache::open(&dir, CANVAS, max_seq).expect("cache");
        let ids = synth_ids(2488);
        let (kv_len, _) =
            prefill_monolithic_kv_with_cache(&mut cache, &ids, &kv_buf, &layout, max_seq, N_LAYERS)
                .expect("prefill");
        assert_eq!(kv_len, 2488);
        assert_hydrate_exact(&mut cache, &kv_buf, &layout, max_seq, kv_len, N_LAYERS);
        // And one extend across the wrap completes with finite live KV.
        let ext = synth_ids(2488 + CANVAS);
        let off = extend_monolithic_kv_chunked(
            &mut cache,
            &kv_buf,
            &layout,
            2488,
            &ext[2488..],
            max_seq,
            N_LAYERS,
        )
        .expect("extend past wrap");
        assert_eq!(off, 2488 + CANVAS);
    }

    /// E15: fast-vs-engine per-layer per-position-band KV divergence at 4.2k.
    /// The DGQ_KV_NOISE anchor (engine + 1% noise on every KV value answers
    /// the 4.2k doc probe correctly) makes the criterion hard: rel-RMS > ~3%
    /// in some (layer, band) = the bug's first expression; below = ignorable.
    /// Fixture path via DGQ_E15_PROMPT (defaults to the session scratchpad
    /// probe_4k.txt).
    #[test]
    #[ignore = "model-gated: cargo test --release e15_layer_kv_bisect -- --ignored --nocapture"]
    fn e15_layer_kv_bisect() {
        use crate::metal::step_kernel::{StepFinishMode, StepSmokeConfig, build_step_runtime};
        use crate::tokenizer::Tokenizer;
        let Some(dir) = model_dir() else { return };
        let prompt_path = std::env::var("DGQ_E15_PROMPT").unwrap_or_else(|_| {
            "/private/tmp/claude-501/-Users-matt-Documents-github-diffgemma-mps/2f485cb9-ac37-41aa-bfea-e7eb74c44b4d/scratchpad/probe_4k.txt".into()
        });
        let text = std::fs::read_to_string(&prompt_path).expect("probe fixture");
        let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer");
        let ids = tok.encode(&text, false);
        let n = ids.len();
        eprintln!("e15: {n} tokens from {prompt_path}");
        let max_seq = 8192usize;

        // FAST path: step runtime + prefill_chunks (the production fast prefill).
        let cfg = StepSmokeConfig {
            layers: N_LAYERS,
            steps: 1,
            kv_len: 0,
            seed: 7,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: true,
        };
        let (mut rt, _) = build_step_runtime(&dir, &cfg).expect("runtime");
        let kv_fast = rt.prefill_chunks(&ids).expect("fast prefill");
        assert_eq!(kv_fast, n);

        // ENGINE path: shares the dgq blob; separate KV buffer, same layout.
        let layout = *rt.layout();
        let ctx = MetalContext::new().expect("metal");
        let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
        let eng_buf = ctx
            .device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .expect("engine kv buf");
        let mut cache =
            MonolithicEncoderCache::open_opt(&dir, CANVAS, max_seq, Some(rt.shared_dgq_blob()))
                .expect("encoder cache");
        let (kv_eng, _) = prefill_monolithic_kv_with_cache(
            &mut cache, &ids, &eng_buf, &layout, max_seq, N_LAYERS,
        )
        .expect("engine prefill");
        assert_eq!(kv_eng, n);

        // Per-(layer, band) rel-RMS over LIVE slots. Bands of 512 positions.
        const BAND: usize = 512;
        eprintln!("e15: rel-RMS fast-vs-engine per (layer, position band); * = >3%");
        let mut worst: (f32, usize, usize) = (0.0, 0, 0);
        for layer in 0..N_LAYERS {
            let l = &layout.layers[layer];
            let nkv = l.n_kv_heads as usize;
            let hd = l.head_dim as usize;
            let token_stride = nkv * hd * 2;
            let slot_base = l.kv_region as usize / 2;
            let live_from = if l.kv_ring_mask != 0 {
                n.saturating_sub(layer_slots(l, max_seq))
            } else {
                0
            };
            let mut row = format!("L{layer:02}{} ", if l.is_full != 0 { "F" } else { " " });
            let mut band_start = live_from - live_from % BAND;
            while band_start < n {
                let b0 = band_start.max(live_from);
                let band_end = (band_start + BAND).min(n);
                let mut num = 0f64;
                let mut den = 0f64;
                for pos in b0..band_end {
                    let slot = kv_slot(l, pos);
                    for i in 0..token_stride {
                        let byte = (slot_base + slot * token_stride + i) * 2;
                        let va = f16_bits_to_f32(read_half_at(rt.kvcache(), byte)) as f64;
                        let vb = f16_bits_to_f32(read_half_at(&eng_buf, byte)) as f64;
                        num += (va - vb) * (va - vb);
                        den += vb * vb;
                    }
                }
                let rel = if den > 0.0 {
                    (num / den).sqrt() as f32
                } else {
                    0.0
                };
                if rel > worst.0 {
                    worst = (rel, layer, b0);
                }
                row.push_str(&format!(
                    "{}{:5.3} ",
                    if rel > 0.03 { "*" } else { " " },
                    rel
                ));
                band_start = band_end;
            }
            eprintln!("{row}");
        }
        eprintln!(
            "e15: worst rel-RMS {:.4} at L{} band starting {}",
            worst.0, worst.1, worst.2
        );
    }

    /// E15 causality check (chaos-immune): fast-prefill of tokens[..k] must
    /// leave BYTE-IDENTICAL full-layer KV for positions [0, k) as a fast
    /// prefill of the whole prompt — each position's KV is fixed by its causal
    /// context. Any difference = later tokens corrupting earlier state (a
    /// write-range/aliasing bug), with no routing-chaos excuse (same path,
    /// same kernels, bit comparison). Full layers only (linear storage; the
    /// sliding rings hold different position windows in the two runs).
    #[test]
    #[ignore = "model-gated: cargo test --release e15_causality_check -- --ignored --nocapture"]
    fn e15_causality_check() {
        use crate::metal::step_kernel::{StepFinishMode, StepSmokeConfig, build_step_runtime};
        use crate::tokenizer::Tokenizer;
        let Some(dir) = model_dir() else { return };
        let prompt_path = std::env::var("DGQ_E15_PROMPT").unwrap_or_else(|_| {
            "/private/tmp/claude-501/-Users-matt-Documents-github-diffgemma-mps/2f485cb9-ac37-41aa-bfea-e7eb74c44b4d/scratchpad/probe_4k.txt".into()
        });
        let text = std::fs::read_to_string(&prompt_path).expect("probe fixture");
        let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer");
        let ids = tok.encode(&text, false);
        let n = ids.len();
        let k = 2048usize.min(n / 2 * 2);
        let max_seq = 8192usize;
        eprintln!("e15-causality: prefix {k} of {n} tokens");
        let cfg = StepSmokeConfig {
            layers: N_LAYERS,
            steps: 1,
            kv_len: 0,
            seed: 7,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: true,
        };
        let (mut rt, _) = build_step_runtime(&dir, &cfg).expect("runtime");

        // Run 1: prefix only. Snapshot full-layer KV for [0, k).
        let got = rt.prefill_chunks(&ids[..k]).expect("prefix prefill");
        assert_eq!(got, k);
        let layout = *rt.layout();
        let mut snap: Vec<(usize, Vec<u16>)> = Vec::new();
        for layer in 0..N_LAYERS {
            let l = &layout.layers[layer];
            if l.is_full == 0 {
                continue;
            }
            let token_stride = (l.n_kv_heads * l.head_dim) as usize * 2;
            let slot_base = l.kv_region as usize / 2;
            let mut vals = vec![0u16; k * token_stride];
            for (i, v) in vals.iter_mut().enumerate() {
                *v = read_half_at(rt.kvcache(), (slot_base + i) * 2);
            }
            snap.push((layer, vals));
        }

        // Run 2: full prompt on the same runtime (prefill_chunks rewrites from 0).
        rt.set_kv_len(0);
        let got = rt.prefill_chunks(&ids).expect("full prefill");
        assert_eq!(got, n);

        for (layer, vals) in &snap {
            let l = &layout.layers[*layer];
            let token_stride = (l.n_kv_heads * l.head_dim) as usize * 2;
            let slot_base = l.kv_region as usize / 2;
            let mut bad = 0usize;
            let mut first: Option<usize> = None;
            for (i, v) in vals.iter().enumerate() {
                let now = read_half_at(rt.kvcache(), (slot_base + i) * 2);
                if now != *v {
                    bad += 1;
                    if first.is_none() {
                        first = Some(i / token_stride);
                    }
                }
            }
            eprintln!(
                "e15-causality: L{layer:02}F prefix [0,{k}) mismatched u16s: {bad}/{} first_pos={first:?}",
                vals.len()
            );
        }
    }

    /// E8-M0 (task #71, the "un-RoPE the KV" idea): does PRE-RoPE K quantize
    /// better than the POST-RoPE K we store today? RoPE mixes dim pairs with
    /// position-dependent angles and can widen per-group ranges; if pre-RoPE K
    /// brings affine-q4 error toward q8-class, storing K pre-RoPE (RoPE at
    /// attention-read time) both restores the offline Hadamard fold AND
    /// improves plain quantization. Method: fast-prefill a real prompt at
    /// max_seq=2048 (slot == pos on every layer: sliding rings are 2048 slots,
    /// no wrap), read the stored post-RoPE f16 K rows, invert RoPE exactly on
    /// CPU (orthogonal rotation; the f16 storage noise ~1e-3 is far below the
    /// 1-8% quant errors under study), and compare kv_quant round-trips.
    #[test]
    #[ignore = "model-gated: cargo test --release e8_prerope_k_quant_stats -- --ignored --nocapture"]
    fn e8_prerope_k_quant_stats() {
        use crate::metal::step_kernel::{StepFinishMode, StepSmokeConfig, build_step_runtime};
        use crate::shaders::kv_quant::{
            q4_affine_rotated_roundtrip, q4_affine_roundtrip, q4_affine_row_rotated_roundtrip,
            q4_affine_row_roundtrip, q4_rotated_roundtrip, q4_roundtrip, q8_roundtrip,
            q8_row_rotated_roundtrip, q8_row_roundtrip,
        };
        use crate::tokenizer::Tokenizer;
        let Some(dir) = model_dir() else { return };
        let prompt_path = std::env::var("DGQ_E8_PROMPT")
            .unwrap_or_else(|_| "fixtures/smoketest/longdoc.md".into());
        let text = std::fs::read_to_string(&prompt_path).expect("probe fixture");
        let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer");
        let mut ids = tok.encode(&text, false);
        // Keep slot == pos (sliding ring = 2048 slots at this max_seq) AND
        // leave the CANVAS block the prefill reserves (kv_len + 256 <= max_seq).
        ids.truncate(1750);
        let n = ids.len();
        let max_seq = 2048usize;
        let cfg = StepSmokeConfig {
            layers: N_LAYERS,
            steps: 1,
            kv_len: 0,
            seed: 7,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: true,
        };
        let (mut rt, _) = build_step_runtime(&dir, &cfg).expect("runtime");
        let kv = rt.prefill_chunks(&ids).expect("fast prefill");
        assert_eq!(kv, n);
        let layout = *rt.layout();

        // Exact inverses of qk_rope_kv.metal's two rotations (transpose = -sin).
        fn un_rope(head: &mut [f32], full: bool, pos: usize) {
            let hd = head.len();
            if full {
                // apply_proportional_rope_f32: rot = hd/4, theta 1e6.
                let half_head = hd / 2;
                for d in 0..hd / 8 {
                    let inv_freq = 1.0e6f32.powf(-2.0 * d as f32 / hd as f32);
                    let (s, c) = (pos as f32 * inv_freq).sin_cos();
                    let x0 = head[d];
                    let x1 = head[half_head + d];
                    head[d] = x0 * c + x1 * s;
                    head[half_head + d] = -x0 * s + x1 * c;
                }
            } else {
                // apply_split_half_rope_f32: rot = hd, theta 1e4.
                let half_rot = hd / 2;
                for d in 0..half_rot {
                    let inv_freq = 1.0e4f32.powf(-2.0 * d as f32 / hd as f32);
                    let (s, c) = (pos as f32 * inv_freq).sin_cos();
                    let x0 = head[d];
                    let x1 = head[d + half_rot];
                    head[d] = x0 * c + x1 * s;
                    head[d + half_rot] = -x0 * s + x1 * c;
                }
            }
        }

        let names = [
            "q8",
            "q4_sym",
            "q4_affine",
            "q4_sym_rot",
            "q4_affine_rot",
            "q8_row",
            "q8_row_rot",
            "q4_aff_row",
            "q4_aff_row_rot",
        ];
        let fns: [fn(&[f32], &mut [f32]); 9] = [
            q8_roundtrip,
            q4_roundtrip,
            q4_affine_roundtrip,
            q4_rotated_roundtrip,
            q4_affine_rotated_roundtrip,
            q8_row_roundtrip,
            q8_row_rotated_roundtrip,
            q4_affine_row_roundtrip,
            q4_affine_row_rotated_roundtrip,
        ];

        eprintln!(
            "e8-m0: K quant rel-RMS, post- vs pre-RoPE ({n} tokens, aggregate per layer class)"
        );
        for full_class in [false, true] {
            let mut num = [[0f64; 2]; 9];
            let mut den = [[0f64; 2]; 9];
            for layer in 0..N_LAYERS {
                let l = &layout.layers[layer];
                if (l.is_full != 0) != full_class {
                    continue;
                }
                let nkv = l.n_kv_heads as usize;
                let hd = l.head_dim as usize;
                let token_stride = nkv * hd * 2;
                let slot_base = l.kv_region as usize / 2;
                let mut out = vec![0f32; hd];
                for pos in 0..n {
                    for hh in 0..nkv {
                        let mut post = vec![0f32; hd];
                        for (d, p) in post.iter_mut().enumerate() {
                            let byte = (slot_base + pos * token_stride + hh * hd + d) * 2;
                            *p = f16_bits_to_f32(read_half_at(rt.kvcache(), byte));
                        }
                        let mut pre = post.clone();
                        un_rope(&mut pre, full_class, pos);
                        for (v, f) in fns.iter().enumerate() {
                            for (side, src) in [&post, &pre].into_iter().enumerate() {
                                f(src, &mut out);
                                for d in 0..hd {
                                    let e = (src[d] - out[d]) as f64;
                                    num[v][side] += e * e;
                                    den[v][side] += (src[d] as f64) * (src[d] as f64);
                                }
                            }
                        }
                    }
                }
            }
            let label = if full_class {
                "full  (hd 512, rot 128, theta 1e6, 5 layers, linear KV)"
            } else {
                "sliding (hd 256, rot 256, theta 1e4, 25 layers, ring KV)"
            };
            eprintln!("  [{label}]");
            for (v, name) in names.iter().enumerate() {
                let post = (num[v][0] / den[v][0]).sqrt();
                let pre = (num[v][1] / den[v][1]).sqrt();
                eprintln!(
                    "    {name:<14} post {post:.4}  pre {pre:.4}  post/pre {:.2}x",
                    post / pre
                );
            }
        }
    }

    /// E16-M0b (token fusion): how MERGEABLE are neighboring full-layer KV
    /// rows? Token fusion (CaM/KVMerger/Compressive-Transformer class) would
    /// coalesce aged rows of the 5 FULL layers (the only long-range memory =
    /// all the KV bytes and the O(kv_len) step cost). Merging by averaging is
    /// promising iff neighbors are correlated — and structurally they should
    /// be: proportional RoPE ropes only rot=hd/4 dims, so 75% of every
    /// full-layer K row is position-independent. Measures adjacent-row and
    /// block-of-4 cosine (K whole / K roped-dims / K unroped-dims / V), by
    /// age band, plus the constant-row-norm check (scalar k_norm → rows on a
    /// sphere). Full layers store linearly (slot == pos at any length), so a
    /// longer prompt is fine here.
    #[test]
    #[ignore = "model-gated: cargo test --release e16_fusion_mergeability_stats -- --ignored --nocapture"]
    fn e16_fusion_mergeability_stats() {
        use crate::metal::step_kernel::{StepFinishMode, StepSmokeConfig, build_step_runtime};
        use crate::tokenizer::Tokenizer;
        let Some(dir) = model_dir() else { return };
        // Carrier selection (`DGQ_E16_CARRIER`): a text file path, or "random"
        // for pseudo-random token ids — the null control that separates
        // content redundancy from model-intrinsic KV structure. Default = the
        // English technical markdown fixture. Similarity structure may differ
        // by content class (code / logs / non-English), so the M0b verdict
        // must not rest on one carrier.
        let carrier = std::env::var("DGQ_E16_CARRIER")
            .unwrap_or_else(|_| "fixtures/smoketest/longdoc.md".into());
        let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer");
        let max_seq = 8192usize;
        let mut ids = if carrier == "random" {
            synth_ids(max_seq)
        } else {
            tok.encode(
                &std::fs::read_to_string(&carrier).expect("carrier fixture"),
                false,
            )
        };
        ids.truncate(max_seq - CANVAS - 36);
        let n = ids.len();
        eprintln!("e16-m0b carrier: {carrier}");
        let cfg = StepSmokeConfig {
            layers: N_LAYERS,
            steps: 1,
            kv_len: 0,
            seed: 7,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: true,
        };
        let (mut rt, _) = build_step_runtime(&dir, &cfg).expect("runtime");
        let kv = rt.prefill_chunks(&ids).expect("fast prefill");
        assert_eq!(kv, n);
        let layout = *rt.layout();

        fn cos(a: &[f32], b: &[f32]) -> f64 {
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                d += (x as f64) * (y as f64);
                na += (x as f64) * (x as f64);
                nb += (y as f64) * (y as f64);
            }
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                d / (na * nb).sqrt()
            }
        }

        eprintln!("e16-m0b: full-layer KV mergeability ({n} tokens; aged = all but last 2048)");
        for layer in 0..N_LAYERS {
            let l = &layout.layers[layer];
            if l.is_full == 0 {
                continue;
            }
            let nkv = l.n_kv_heads as usize;
            let hd = l.head_dim as usize;
            let rot = hd / 4;
            let half_head = hd / 2;
            // roped dims = {0..rot/2} ∪ {half_head..half_head+rot/2}
            let roped: Vec<usize> = (0..rot / 2).chain(half_head..half_head + rot / 2).collect();
            let unroped: Vec<usize> = (0..hd).filter(|d| !roped.contains(d)).collect();
            let token_stride = nkv * hd * 2;
            let slot_base = l.kv_region as usize / 2;
            // read all K and V rows for this layer: [pos][head][hd]
            let read_row = |pos: usize, hh: usize, is_v: bool| -> Vec<f32> {
                let base = slot_base + pos * token_stride + if is_v { nkv * hd } else { 0 };
                (0..hd)
                    .map(|d| f16_bits_to_f32(read_half_at(rt.kvcache(), (base + hh * hd + d) * 2)))
                    .collect()
            };
            let sub =
                |v: &[f32], idx: &[usize]| -> Vec<f32> { idx.iter().map(|&i| v[i]).collect() };
            let bands: [(usize, usize, &str); 2] = [
                (0, n.saturating_sub(2048), "aged  "),
                (n.saturating_sub(2048), n, "recent"),
            ];
            for (b0, b1, label) in bands {
                if b1 - b0 < 8 {
                    continue;
                }
                let (mut ck, mut ckr, mut cku, mut cv, mut c4k, mut c4v) =
                    (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
                let (mut n_adj, mut n_blk) = (0usize, 0usize);
                let mut norm_sum = 0f64;
                let mut norm_sq = 0f64;
                let mut n_norm = 0usize;
                // sample every head, stride positions by 4 to bound CPU time
                for hh in 0..nkv {
                    let mut pos = b0;
                    while pos + 4 < b1 {
                        let k: Vec<Vec<f32>> =
                            (0..4).map(|i| read_row(pos + i, hh, false)).collect();
                        let v: Vec<Vec<f32>> =
                            (0..4).map(|i| read_row(pos + i, hh, true)).collect();
                        ck += cos(&k[0], &k[1]);
                        ckr += cos(&sub(&k[0], &roped), &sub(&k[1], &roped));
                        cku += cos(&sub(&k[0], &unroped), &sub(&k[1], &unroped));
                        cv += cos(&v[0], &v[1]);
                        n_adj += 1;
                        // mean pairwise cos within the block of 4
                        let (mut sk, mut sv, mut np) = (0f64, 0f64, 0usize);
                        for i in 0..4 {
                            for j in i + 1..4 {
                                sk += cos(&k[i], &k[j]);
                                sv += cos(&v[i], &v[j]);
                                np += 1;
                            }
                        }
                        c4k += sk / np as f64;
                        c4v += sv / np as f64;
                        n_blk += 1;
                        let nn = k[0]
                            .iter()
                            .map(|&x| (x as f64) * (x as f64))
                            .sum::<f64>()
                            .sqrt();
                        norm_sum += nn;
                        norm_sq += nn * nn;
                        n_norm += 1;
                        pos += 16;
                    }
                }
                let m = |s: f64, c: usize| s / c.max(1) as f64;
                let nm = norm_sum / n_norm.max(1) as f64;
                let nsd = (norm_sq / n_norm.max(1) as f64 - nm * nm).max(0.0).sqrt();
                eprintln!(
                    "  L{layer:02} {label} adjK {:+.3} (roped {:+.3} unroped {:+.3}) adjV {:+.3} | blk4 K {:+.3} V {:+.3} | ‖K‖ {nm:.2}±{nsd:.2}",
                    m(ck, n_adj),
                    m(ckr, n_adj),
                    m(cku, n_adj),
                    m(cv, n_adj),
                    m(c4k, n_blk),
                    m(c4v, n_blk)
                );
            }
        }
    }

    /// E16-M0c (token fusion ORACLE): quality frontier of count-weighted KV
    /// fusion with ZERO kernel changes. Trick: prefill normally, then rewrite
    /// each aged full-layer block of r rows as r DUPLICATES of its mean-K /
    /// mean-V — duplicated keys contribute r·exp(q·k̄) to the softmax, which
    /// is EXACTLY the count-weighted merged-attention semantics a real fused
    /// kernel would implement (and duplicated V̄ gives the right weighted
    /// average). Then re-enter generation on the doctored cache (restore →
    /// mutate → generate re-entry skips prefill since kv_valid == prompt) and
    /// judge the doc_13k ladder question. Faithful to a real M1 including the
    /// merged-RoPE-position effect (we merge stored post-RoPE rows). Sliding
    /// layers untouched (they never see aged tokens anyway).
    #[test]
    #[ignore = "model-gated: cargo test --release e16_fusion_oracle_replay -- --ignored --nocapture"]
    fn e16_fusion_oracle_replay() {
        use crate::chat_template;
        use crate::config;
        use crate::metal::step_generate::{
            StepGenerateConfig, StepGenerateSession, generate_with_session,
        };
        use crate::sample;
        use crate::shaders::f16::f32_to_f16_bits;
        use crate::tokenizer::Tokenizer;
        let Some(dir) = model_dir() else { return };
        let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer");
        let text = std::fs::read_to_string("fixtures/smoketest/longdoc.md").expect("probe fixture");
        let doc_ids = tok.encode(&text, false);
        let excerpt = tok.decode(&doc_ids[..13300]);
        // Facts at doc depths ~979 ("1085") and ~7716 ("20.25") — DEEP inside
        // the fused aged region at every grid W (recent window = last 1-2k).
        // (First run of this oracle asked about facts at ~12.7-13.0k, which sit
        // INSIDE the protected recent window — all cells passed identically and
        // proved only that the harness works. Aim questions at fused depths.)
        let question = "(a) How many seconds did the 105k-token needle prefill take, and (b) \
                        what is the Metal single-buffer allocation cap in GiB?";
        let prompt_text =
            format!("{excerpt}\n\n[end of document]\nAnswer from the document above: {question}");
        let turns = [chat_template::ChatTurn::user(&prompt_text)];
        let prompt = chat_template::format_chat_token_ids(
            &tok,
            &turns,
            &chat_template::ChatFormatOptions::default(),
        )
        .expect("chat prompt");
        let n = prompt.len();
        let max_seq = 16384usize;
        let sampler = sample::sampler_for_steps(48, false);
        let mut cfg = StepGenerateConfig::from_generate(7, 512, max_seq, N_LAYERS, sampler, false);
        cfg.stop_token_ids = config::load_generation_stop_tokens(&dir);
        let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
        session.extend_kv(&prompt).expect("prefill");
        let snap = session.snapshot_kv();
        let layout = *session.layout_for_test();
        eprintln!("e16-oracle: prompt {n} tokens; facts at ~979 / ~7716, aged region = [0, n-W)");

        // Word-run match with digit<->letter splits (mirror of smoke_normalize).
        fn normalize(s: &str) -> String {
            let mut out = String::new();
            let mut prev = 0u8; // 0 space, 1 digit, 2 letter
            for c in s.chars() {
                if c.is_alphanumeric() {
                    let cur = if c.is_ascii_digit() { 1 } else { 2 };
                    if prev != 0 && prev != cur {
                        out.push(' ');
                    }
                    for lc in c.to_lowercase() {
                        out.push(lc);
                    }
                    prev = cur;
                } else if prev != 0 {
                    out.push(' ');
                    prev = 0;
                }
            }
            out.trim().to_string()
        }
        let matches = |reply: &str, k: &str| {
            format!(" {} ", normalize(reply)).contains(&format!(" {} ", normalize(k)))
        };

        fn write_half_at(
            buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            byte_off: usize,
            bits: u16,
        ) {
            use objc2_metal::MTLBuffer as _;
            unsafe {
                *(buf.contents().as_ptr() as *mut u8)
                    .add(byte_off)
                    .cast::<u16>() = bits;
            }
        }

        // Count-weighted fusion via duplication over aged range [a0, a1):
        // each block of r rows -> r copies of the block mean (K and V). With
        // `tau`, a block merges ONLY if the mean pairwise K-cosine of its rows
        // (per head) clears it — sharp singleton rows stay exact (similarity-
        // gated merging). Returns (candidate_rows, effective_rows): what a
        // real M1 layout would store (merged block of m -> 1 row).
        let fuse_range = |session: &StepGenerateSession,
                          a0: usize,
                          a1: usize,
                          r: usize,
                          tau: Option<f64>|
         -> (usize, usize) {
            let buf = session.kv_buffer_for_test();
            let (mut candidates, mut effective) = (0usize, 0usize);
            for layer in 0..N_LAYERS {
                let l = &layout.layers[layer];
                if l.is_full == 0 {
                    continue;
                }
                let nkv = l.n_kv_heads as usize;
                let hd = l.head_dim as usize;
                let token_stride = nkv * hd * 2;
                let half_base = l.kv_region as usize / 2;
                let read_row = |pos: usize, hh: usize, off: usize| -> Vec<f64> {
                    let base = half_base + pos * token_stride + off + hh * hd;
                    (0..hd)
                        .map(|d| f16_bits_to_f32(read_half_at(buf, (base + d) * 2)) as f64)
                        .collect()
                };
                let mut b0 = a0;
                while b0 < a1 {
                    let b1 = (b0 + r).min(a1);
                    let m = b1 - b0;
                    if m >= 2 {
                        for hh in 0..nkv {
                            candidates += m;
                            let ks: Vec<Vec<f64>> = (b0..b1).map(|p| read_row(p, hh, 0)).collect();
                            let merge = match tau {
                                None => true,
                                Some(t) => {
                                    let (mut s, mut c) = (0f64, 0usize);
                                    for i in 0..m {
                                        for j in i + 1..m {
                                            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
                                            for x in 0..hd {
                                                d += ks[i][x] * ks[j][x];
                                                na += ks[i][x] * ks[i][x];
                                                nb += ks[j][x] * ks[j][x];
                                            }
                                            s += d / (na * nb).sqrt().max(1e-12);
                                            c += 1;
                                        }
                                    }
                                    s / c as f64 >= t
                                }
                            };
                            if !merge {
                                effective += m;
                                continue;
                            }
                            effective += 1;
                            for off in [0usize, nkv * hd] {
                                let rows: Vec<Vec<f64>> = if off == 0 {
                                    ks.clone()
                                } else {
                                    (b0..b1).map(|p| read_row(p, hh, off)).collect()
                                };
                                let mut mean = vec![0f64; hd];
                                for row in &rows {
                                    for (d, m) in mean.iter_mut().enumerate() {
                                        *m += row[d];
                                    }
                                }
                                for mm in mean.iter_mut() {
                                    *mm /= m as f64;
                                }
                                for pos in b0..b1 {
                                    let base = half_base + pos * token_stride + off + hh * hd;
                                    for (d, &mm) in mean.iter().enumerate() {
                                        write_half_at(
                                            buf,
                                            (base + d) * 2,
                                            f32_to_f16_bits(mm as f32),
                                        );
                                    }
                                }
                            }
                        }
                    } else {
                        effective += m;
                        candidates += m;
                    }
                    b0 = b1;
                }
            }
            (candidates, effective)
        };

        // Cells: uniform r=2 (known-clean), TIERED (r=4 for distant, r=2 for
        // mid-age — facts sit one per tier), similarity-GATED r=4/r=8 at
        // tau=0.5 (merge only redundant runs; sharp singletons stay exact).
        let w = 2048usize;
        let names = [
            "control",
            "uniform_r2",
            "tiered_r4_r2",
            "gated_t50_r4",
            "gated_t50_r8",
        ];
        for name in names {
            session.restore_kv(&snap);
            let (cand, eff) = match name {
                "control" => (0, 0),
                "uniform_r2" => fuse_range(&session, 0, n - w, 2, None),
                "tiered_r4_r2" => {
                    let far = fuse_range(&session, 0, n.saturating_sub(8192), 4, None);
                    let mid = fuse_range(&session, n.saturating_sub(8192), n - w, 2, None);
                    (far.0 + mid.0, far.1 + mid.1)
                }
                "gated_t50_r4" => fuse_range(&session, 0, n - w, 4, Some(0.50)),
                "gated_t50_r8" => fuse_range(&session, 0, n - w, 8, Some(0.50)),
                _ => unreachable!(),
            };
            let out =
                generate_with_session(&mut session, &prompt, &cfg, "e16-oracle").expect("gen");
            let new_ids = sample::strip_degenerate_token_ids(out.token_ids.get(n..).unwrap_or(&[]));
            let reply = chat_template::sanitize_model_reply(&tok.decode(&new_ids));
            let r1 = matches(&reply, "1085");
            let r2 = matches(&reply, "20.25");
            let keep = if cand > 0 {
                format!("{:.0}%", 100.0 * eff as f64 / cand as f64)
            } else {
                "100%".into()
            };
            let prev = reply
                .chars()
                .take(80)
                .collect::<String>()
                .replace('\n', " ");
            eprintln!(
                "e16-oracle {name:<13} keys-kept {keep:<4} 1085:{} 20.25:{} | {prev}",
                if r1 { "PASS" } else { "FAIL" },
                if r2 { "PASS" } else { "FAIL" },
            );
        }
    }

    /// E16 multi-needle oracle: binomial-strength fusion-quality stats,
    /// carrier-selectable (`DGQ_E16_CARRIER`: text file path; default = the
    /// English markdown fixture — run it on code/log carriers too, the
    /// mergeability census showed raw K-similarity is mostly model-intrinsic
    /// common-mode, so only task quality can validate a carrier class).
    /// Plants 8 synthetic city-code needles at even depths through the FUSED
    /// region of a ~13k-token carrier; one generation asks for the full list
    /// -> k-of-8 per cell instead of a coin flip per cell.
    #[test]
    #[ignore = "model-gated: cargo test --release e16_fusion_multineedle_oracle -- --ignored --nocapture"]
    fn e16_fusion_multineedle_oracle() {
        use crate::chat_template;
        use crate::config;
        use crate::metal::step_generate::{
            StepGenerateConfig, StepGenerateSession, generate_with_session,
        };
        use crate::sample;
        use crate::shaders::f16::f32_to_f16_bits;
        use crate::tokenizer::Tokenizer;
        let Some(dir) = model_dir() else { return };
        let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer");
        let carrier = std::env::var("DGQ_E16_CARRIER")
            .unwrap_or_else(|_| "fixtures/smoketest/longdoc.md".into());
        let mut raw = std::fs::read_to_string(&carrier).expect("carrier fixture");
        while tok.encode(&raw, false).len() < 14000 {
            let again = raw.clone();
            raw.push_str("\n\n");
            raw.push_str(&again);
        }
        const NEEDLES: [(&str, &str); 8] = [
            ("Lisbon", "48291"),
            ("Osaka", "57364"),
            ("Nairobi", "81437"),
            ("Quito", "29586"),
            ("Tromso", "63917"),
            ("Adelaide", "74128"),
            ("Reykjavik", "36852"),
            ("Valparaiso", "95274"),
        ];
        for (city, code) in NEEDLES {
            assert!(!raw.contains(code), "carrier already contains {code}");
            assert!(!raw.contains(city), "carrier already contains {city}");
        }
        // Truncate the carrier to ~13.1k tokens, then plant needles at even
        // char depths within the first 85% (all needles stay in the aged
        // region for W=2048), on line boundaries.
        let ids = tok.encode(&raw, false);
        let base_text = tok.decode(&ids[..13100.min(ids.len())]);
        let mut text = String::new();
        let bytes = base_text.len();
        let mut cursor = 0usize;
        for (i, (city, code)) in NEEDLES.iter().enumerate() {
            let mut target = bytes * 85 * (2 * i + 1) / (100 * 2 * NEEDLES.len());
            target = target.min(bytes - 1);
            while !base_text.is_char_boundary(target) {
                target += 1;
            }
            let cut = base_text[target..]
                .find('\n')
                .map(|o| target + o + 1)
                .unwrap_or(bytes);
            text.push_str(&base_text[cursor..cut]);
            text.push_str(&format!("\nThe secret access code for {city} is {code}.\n"));
            cursor = cut;
        }
        text.push_str(&base_text[cursor..]);
        let prompt_text = format!(
            "{text}\n\n[end of document]\nAnswer from the document above: List every city that \
             has a secret access code in the document, together with its code."
        );
        let turns = [chat_template::ChatTurn::user(&prompt_text)];
        let prompt = chat_template::format_chat_token_ids(
            &tok,
            &turns,
            &chat_template::ChatFormatOptions::default(),
        )
        .expect("chat prompt");
        let n = prompt.len();
        // Report each needle's token depth (verify they sit in the fused region).
        let depths: Vec<usize> = NEEDLES
            .iter()
            .map(|(_, code)| {
                let off = text.find(code).expect("planted");
                tok.encode(&text[..off], false).len()
            })
            .collect();
        let w = 2048usize;
        eprintln!(
            "e16-needles carrier={carrier} prompt {n} tok; needle depths {depths:?}; aged=[0,{})",
            n - w
        );

        let max_seq = 16384usize;
        let sampler = sample::sampler_for_steps(48, false);
        let mut cfg = StepGenerateConfig::from_generate(7, 512, max_seq, N_LAYERS, sampler, false);
        cfg.stop_token_ids = config::load_generation_stop_tokens(&dir);
        let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
        session.extend_kv(&prompt).expect("prefill");
        let snap = session.snapshot_kv();
        let layout = *session.layout_for_test();

        fn write_half_at(
            buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
            byte_off: usize,
            bits: u16,
        ) {
            use objc2_metal::MTLBuffer as _;
            unsafe {
                *(buf.contents().as_ptr() as *mut u8)
                    .add(byte_off)
                    .cast::<u16>() = bits;
            }
        }
        // Count-weighted fusion of full-layer rows in [a0, a1) at ratio r
        // (duplication trick; see e16_fusion_oracle_replay). With `resid_tau`,
        // a block merges only if the mean pairwise cosine of its K RESIDUALS
        // (row minus the 256-position band mean — the model-intrinsic
        // common-mode that dominates raw cosine even on random tokens) clears
        // it: distinctive rows (needles) refuse to merge, boilerplate merges
        // freely. Returns (candidate_rows, effective_rows).
        let fuse_range = |session: &StepGenerateSession,
                          a0: usize,
                          a1: usize,
                          r: usize,
                          resid_tau: Option<f64>|
         -> (usize, usize) {
            let buf = session.kv_buffer_for_test();
            let (mut cand, mut eff) = (0usize, 0usize);
            for layer in 0..N_LAYERS {
                let l = &layout.layers[layer];
                if l.is_full == 0 {
                    continue;
                }
                let nkv = l.n_kv_heads as usize;
                let hd = l.head_dim as usize;
                let token_stride = nkv * hd * 2;
                let half_base = l.kv_region as usize / 2;
                let read_k = |pos: usize, hh: usize| -> Vec<f64> {
                    let base = half_base + pos * token_stride + hh * hd;
                    (0..hd)
                        .map(|d| f16_bits_to_f32(read_half_at(buf, (base + d) * 2)) as f64)
                        .collect()
                };
                // Per-head band means of K over 256-position windows (common-mode).
                const BAND: usize = 256;
                let n_bands = a1.div_ceil(BAND);
                let mut band_mean: Vec<Vec<Vec<f64>>> = vec![vec![vec![0f64; hd]; n_bands]; nkv];
                if resid_tau.is_some() {
                    let mut band_cnt = vec![vec![0usize; n_bands]; nkv];
                    for pos in a0..a1 {
                        let b = pos / BAND;
                        for hh in 0..nkv {
                            let k = read_k(pos, hh);
                            for (d, &v) in k.iter().enumerate() {
                                band_mean[hh][b][d] += v;
                            }
                            band_cnt[hh][b] += 1;
                        }
                    }
                    for hh in 0..nkv {
                        for b in 0..n_bands {
                            let c = band_cnt[hh][b].max(1) as f64;
                            for d in 0..hd {
                                band_mean[hh][b][d] /= c;
                            }
                        }
                    }
                }
                let mut b0 = a0;
                while b0 < a1 {
                    let b1 = (b0 + r).min(a1);
                    let m = b1 - b0;
                    if m >= 2 {
                        for hh in 0..nkv {
                            cand += m;
                            let merge = match resid_tau {
                                None => true,
                                Some(t) => {
                                    let mu = &band_mean[hh][b0 / BAND];
                                    let rs: Vec<Vec<f64>> = (b0..b1)
                                        .map(|p| {
                                            let k = read_k(p, hh);
                                            k.iter().zip(mu).map(|(&a, &b)| a - b).collect()
                                        })
                                        .collect();
                                    let (mut s, mut c) = (0f64, 0usize);
                                    for i in 0..m {
                                        for j in i + 1..m {
                                            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
                                            for x in 0..hd {
                                                d += rs[i][x] * rs[j][x];
                                                na += rs[i][x] * rs[i][x];
                                                nb += rs[j][x] * rs[j][x];
                                            }
                                            s += d / (na * nb).sqrt().max(1e-12);
                                            c += 1;
                                        }
                                    }
                                    s / c as f64 >= t
                                }
                            };
                            if !merge {
                                eff += m;
                                continue;
                            }
                            eff += 1;
                            for off in [0usize, nkv * hd] {
                                let mut mean = vec![0f64; hd];
                                for pos in b0..b1 {
                                    let base = half_base + pos * token_stride + off + hh * hd;
                                    for (d, mm) in mean.iter_mut().enumerate() {
                                        *mm += f16_bits_to_f32(read_half_at(buf, (base + d) * 2))
                                            as f64;
                                    }
                                }
                                for mm in mean.iter_mut() {
                                    *mm /= m as f64;
                                }
                                for pos in b0..b1 {
                                    let base = half_base + pos * token_stride + off + hh * hd;
                                    for (d, &mm) in mean.iter().enumerate() {
                                        write_half_at(
                                            buf,
                                            (base + d) * 2,
                                            f32_to_f16_bits(mm as f32),
                                        );
                                    }
                                }
                            }
                        }
                    } else {
                        cand += m;
                        eff += m;
                    }
                    b0 = b1;
                }
            }
            (cand, eff)
        };

        for name in [
            "control",
            "uniform_r2",
            "tiered_r4_r2",
            "uniform_r4",
            "resid_t20_r2",
            "resid_t20_r4",
        ] {
            session.restore_kv(&snap);
            let (cand, eff) = match name {
                "control" => (0, 0),
                "uniform_r2" => fuse_range(&session, 0, n - w, 2, None),
                "tiered_r4_r2" => {
                    let a = fuse_range(&session, 0, n.saturating_sub(8192), 4, None);
                    let b = fuse_range(&session, n.saturating_sub(8192), n - w, 2, None);
                    (a.0 + b.0, a.1 + b.1)
                }
                "uniform_r4" => fuse_range(&session, 0, n - w, 4, None),
                "resid_t20_r2" => fuse_range(&session, 0, n - w, 2, Some(0.20)),
                "resid_t20_r4" => fuse_range(&session, 0, n - w, 4, Some(0.20)),
                _ => unreachable!(),
            };
            let keep = if cand > 0 {
                format!("{:.0}%", 100.0 * eff as f64 / cand as f64)
            } else {
                "100%".into()
            };
            let out =
                generate_with_session(&mut session, &prompt, &cfg, "e16-needles").expect("gen");
            let new_ids = sample::strip_degenerate_token_ids(out.token_ids.get(n..).unwrap_or(&[]));
            let reply = chat_template::sanitize_model_reply(&tok.decode(&new_ids));
            let hits: Vec<bool> = NEEDLES
                .iter()
                .map(|(_, code)| reply.contains(code))
                .collect();
            let k = hits.iter().filter(|&&h| h).count();
            let missed: Vec<&str> = NEEDLES
                .iter()
                .zip(&hits)
                .filter(|(_, h)| !**h)
                .map(|((city, _), _)| *city)
                .collect();
            eprintln!(
                "e16-needles {name:<13} keys-kept {keep:<4} {k}/8 retrieved; missed {missed:?}"
            );
        }
    }

    /// One resident engine prefill of 1024 synthetic tokens with the per-layer
    /// phase profile on — splits waitA (attention+dense GEMMs+router GPU) from
    /// waitB (MoE GPU) to locate the ms/tok. Diagnostic only.
    #[test]
    #[ignore = "model-gated bench: cargo test --release engine_prefill_profile -- --ignored --nocapture"]
    fn engine_prefill_profile() {
        let Some(dir) = model_dir() else { return };
        let mut cfg = crate::flags::RuntimeConfig::default();
        cfg.debug.prefill_profile = true;
        let _g = crate::flags::install_for_test(cfg);
        let max_seq = 2048usize;
        let store = DgqStore::open(&dir).expect("dgq");
        let layout = build_layout(&build_offsets_from_store(&store), max_seq);
        let ctx = MetalContext::new().expect("metal");
        let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
        let kv_buf = ctx
            .device
            .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
            .expect("kv buf");
        let mut cache = MonolithicEncoderCache::open(&dir, CANVAS, max_seq).expect("cache");
        let ids = synth_ids(1024);
        let t = std::time::Instant::now();
        let (kv_len, _) =
            prefill_monolithic_kv_with_cache(&mut cache, &ids, &kv_buf, &layout, max_seq, N_LAYERS)
                .expect("prefill");
        eprintln!(
            "engine_prefill_profile: kv_len={kv_len} total={:.2}s ({:.1} ms/tok)",
            t.elapsed().as_secs_f64(),
            t.elapsed().as_secs_f64() * 1000.0 / kv_len as f64
        );
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
}
