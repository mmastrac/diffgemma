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

use crate::Error;
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
use crate::metal::step_kernel::{VOCAB, build_layout, build_offsets_from_store};
#[cfg(test)]
use objc2_metal::{MTLDevice, MTLResourceOptions};

// KV-cache audit / probe harnesses (CLI step-kv-* / step-attn-probe
// subcommands). Split out for size (backlog item 4); a child module, so it sees
// this module's private items via ancestry. Re-exported flat so the existing
// `step_kv::<fn>` paths keep resolving.
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
        return Err(Error::Runtime("kv_len exceeds max_seq"));
    }
    for layer in 0..N_LAYERS.min(kv.layers.len()) {
        let l = &layout.layers[layer];
        let kv_layer = kv.layer(layer).ok_or(Error::Runtime("missing kv layer"))?;
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
            return Err(Error::Gpu("monolithic kv buffer too small"));
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
        let scale_bits = f32_to_f16_bits((mx / 127.0).max(crate::shaders::kv_quant::Q8_MIN_SCALE));
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
        return Err(Error::Gpu("monolithic kv buffer too small"));
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
            .ok_or(Error::Runtime("missing kv layer"))?;
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
        return Err(Error::Runtime("prefill requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Runtime("prefill exceeds max_seq"));
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
        .ok_or(Error::Gpu("gpu kv missing after prefill"))?;
    let kv_len = gpu_kv.kv_len;
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if kv_buf.length() < need {
        return Err(Error::Gpu("monolithic kv buffer too small"));
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
/// (The CPU predecessor of this function read monolithic slots LINEARLY —
/// wrong past the ring wrap — and cost O(kv_len) scalar f16 conversions per
/// call, O(n²) over a chunked delta.)
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
        return Err(Error::Runtime("monolithic kv extend exceeds max_seq"));
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
        .ok_or(Error::Gpu("gpu kv cache missing"))?;
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
        .ok_or(Error::Gpu("gpu kv missing after extend"))?;
    let new_kv_len = gpu_kv.kv_len;
    let append_len = new_token_ids.len();
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if kv_buf.length() < need {
        return Err(Error::Gpu("monolithic kv buffer too small"));
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
/// the fast-prefill trust cap.
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
        return Err(Error::Runtime("monolithic kv extend exceeds max_seq"));
    }
    let total_started = std::time::Instant::now();
    let text = &cache.model.config.text_config;
    let canvas = CANVAS;
    let layers = max_layers.min(text.num_hidden_layers);
    let fmt = crate::flags::kv_format(max_seq);
    let need = kv_cache_total_bytes(layout, max_seq) as usize;
    if kv_buf.length() < need {
        return Err(Error::Gpu("monolithic kv buffer too small"));
    }

    let encoder_kv_cap = (kv_len_before + new_token_ids.len()).min(max_seq);
    cache
        .dec_scratch
        .ensure_gpu_kv(&cache.engine.ctx.device, text, encoder_kv_cap, canvas)?;
    let mut gpu_kv = cache
        .dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Gpu("gpu kv cache missing"))?;
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
            .ok_or(Error::Gpu("gpu kv missing after extend"))?;
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
        return Err(Error::Runtime("prefill requires at least one token"));
    }
    if token_ids.len() > max_seq {
        return Err(Error::Runtime("prefill exceeds max_seq"));
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
mod engine_extend_bench_tests;

#[cfg(all(test, target_os = "macos"))]
mod kv_lineage_tests;

#[cfg(all(test, target_os = "macos"))]
mod fusion_tests;

#[cfg(all(test, target_os = "macos"))]
mod encoder_moe_kv_tests;
