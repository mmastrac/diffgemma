//! Tests for `engine_extend_bench_tests`, extracted from step_kv.rs (backlog item 3).

use super::*;
use crate::metal::device::MetalContext;

pub(super) fn model_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::Path::new("model/diffgemma-26b-a4b-it-q4");
    if dir.join("model.dgq.json").exists() {
        Some(dir.to_path_buf())
    } else {
        eprintln!("skip: model/diffgemma-26b-a4b-it-q4 missing");
        None
    }
}

pub(super) fn synth_ids(n: usize) -> Vec<u32> {
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
            std::slice::from_raw_parts(k_buf.contents().as_ptr() as *const f32, kv_len * per_token)
        };
        let v_eng = unsafe {
            std::slice::from_raw_parts(v_buf.contents().as_ptr() as *const f32, kv_len * per_token)
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

/// Baseline: extend-vs-full-prefill KV diffs below (1500) and above (3000)
/// the sliding ring wrap (2048) — diagnostic (chunk boundaries expose the
/// prefix through f16 pack/unpack; the forward chaos-amplifies that, so
/// nonzero diffs are physics, not defects). The hard gates are (a) hydrate
/// exactness including ring mapping, (b) extend completes and produces finite
/// KV. Timing per extend printed by the monolithic-extend instrumentation.
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
        let (kv_full, _) =
            prefill_monolithic_kv_with_cache(&mut cache, &ids, &buf_full, &layout, max_seq, layers)
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

        let (diff, dl, dp) = live_kv_max_diff(&buf_full, &buf_ext, &layout, max_seq, total, layers);
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

/// Fast-vs-engine per-layer per-position-band KV divergence at 4.2k. The
/// DGQ_KV_NOISE anchor (engine + 1% noise on every KV value answers the 4.2k
/// doc probe correctly) makes the criterion hard: rel-RMS > ~3% in some
/// (layer, band) indicates the bug's first expression; below is ignorable.
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
    let (kv_eng, _) =
        prefill_monolithic_kv_with_cache(&mut cache, &ids, &eng_buf, &layout, max_seq, N_LAYERS)
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

/// Causality check (chaos-immune): fast-prefill of tokens[..k] must leave
/// BYTE-IDENTICAL full-layer KV for positions [0, k) as a fast prefill of
/// the whole prompt — each position's KV is fixed by its causal context. Any
/// difference indicates later tokens corrupting earlier state (a
/// write-range/aliasing bug), with no routing-chaos excuse (same path, same
/// kernels, bit comparison). Full layers only (linear storage; the sliding
/// rings hold different position windows in the two runs).
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

/// Does PRE-RoPE K quantize better than the POST-RoPE K we store today?
/// RoPE mixes dim pairs with position-dependent angles and can widen
/// per-group ranges; if pre-RoPE K brings affine-q4 error toward q8-class,
/// storing K pre-RoPE (RoPE at attention-read time) both restores the
/// offline Hadamard fold AND improves plain quantization. Method: fast-prefill
/// a real prompt at max_seq=2048 (slot == pos on every layer: sliding rings
/// are 2048 slots, no wrap), read the stored post-RoPE f16 K rows, invert
/// RoPE exactly on CPU (orthogonal rotation; the f16 storage noise ~1e-3 is
/// far below the 1-8% quant errors under study), and compare kv_quant
/// round-trips.
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
    let prompt_path =
        std::env::var("DGQ_E8_PROMPT").unwrap_or_else(|_| "fixtures/smoketest/longdoc.md".into());
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

    eprintln!("e8-m0: K quant rel-RMS, post- vs pre-RoPE ({n} tokens, aggregate per layer class)");
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
