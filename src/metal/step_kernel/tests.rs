//! Tests for `tests`, extracted from step_kernel.rs (backlog item 3).

use super::*;

#[test]
#[ignore = "model-gated diagnostic: e2e accept-count/entropy tripwire (cargo test --release monolith_one_step_accept_regression -- --ignored)"]
fn monolith_one_step_accept_regression() {
    // MLX @ 30L/Hello/seed42 accepts ~196 positions on denoise step 1 alone.
    // `.dgq` sharpens over the 8-step block; GEMM tg=(128,1,1) bug held ~1 accept/step
    // (~8 total) with mean_entropy~10. Healthy cumulative accept >> 150.
    const MIN_ACCEPT: u32 = 150;
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        eprintln!("skip monolith_one_step_accept_regression: quantized model not present");
        return;
    };
    let dir = dir.as_path();
    let prefill = hello_chat_prefill_token_ids(dir).expect("hello prefill");
    let cfg = StepSmokeConfig {
        layers: 30,
        steps: 8,
        kv_len: 0,
        seed: 42,
        max_seq: 512,
        finish: StepFinishMode::Full,
        prefill_token_ids: Some(prefill),
        no_early_stop: true,
    };
    let steps = run_denoise_steps(dir, &cfg).expect("monolith denoise block");
    assert!(!steps.is_empty(), "denoise block produced no steps");
    let accepts: Vec<u32> = steps.iter().map(|s| s.accept_count).collect();
    let total_accept: u32 = accepts.iter().sum();
    eprintln!(
        "monolith 30L block: steps={} accept/step={accepts:?} total_accept={} step1(mean_H={:.3})",
        steps.len(),
        total_accept,
        steps[0].mean_entropy,
    );
    assert!(
        steps[0].mean_entropy < 6.0,
        "step1 mean_entropy {:.3} (GEMM tg bug ~10)",
        steps[0].mean_entropy
    );
    assert!(
        total_accept > MIN_ACCEPT,
        "total accept {total_accept} accept/step={accepts:?} (GEMM tg bug ~1/step, sum~8)",
    );
}

#[test]
#[ignore = "model-gated diagnostic: expert offset/bytes layout parity vs real store (run with --ignored)"]
fn moe_grouped_layer2_expert_offset_and_bytes_parity() {
    use crate::dgq::layout::q4_matrix_bytes;
    use crate::metal::moe::expert_forward_staged_dgq;
    use crate::model::moe::MoeScratch;
    use std::path::Path;

    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        eprintln!("skip moe_grouped_layer2_expert_offset_and_bytes_parity");
        return;
    };
    let dir = dir.as_path();
    let layer = 2usize;
    let text = crate::config::ModelConfig::load(dir)
        .expect("cfg")
        .text_config;
    let store = DgqStore::open(dir).expect("dgq");
    let ws = WeightStore::open(dir).expect("ws");
    let ctx = MetalContext::new().expect("metal");
    let cache = GpuDecoderWeightCache::load(&ws, &text, 0, &ctx.device).expect("cache");

    let offsets = build_offsets_from_store(&store);
    let layout = build_layout(&offsets, 512);
    let manifest_gu = offsets
        .get(&format!(
            "model.decoder.layers.{layer}.experts.gate_up_proj"
        ))
        .copied()
        .expect("manifest gate_up");
    let manifest_dn = offsets
        .get(&format!("model.decoder.layers.{layer}.experts.down_proj"))
        .copied()
        .expect("manifest down");
    let layout_gu = layout.layers[layer].experts_gate_up;
    let layout_dn = layout.layers[layer].experts_down;
    eprintln!(
        "L{layer} experts_gate_up manifest={manifest_gu} layout={layout_gu} delta={}",
        manifest_gu as i64 - layout_gu as i64
    );
    eprintln!(
        "L{layer} experts_down   manifest={manifest_dn} layout={layout_dn} delta={}",
        manifest_dn as i64 - layout_dn as i64
    );
    assert_eq!(layout_gu, manifest_gu);
    assert_eq!(layout_dn, manifest_dn);

    let per_expert = q4_matrix_bytes(text.moe_intermediate_size * 2, text.hidden_size) as u64;
    let down_per = q4_matrix_bytes(text.hidden_size, text.moe_intermediate_size) as u64;
    eprintln!(
        "per_expert gate_up={per_expert} down={down_per} fused={}",
        per_expert + down_per
    );

    for expert in [0usize, 18] {
        let cache_gu = cache.expert_gate_up_q4(layer, expert);
        let kernel_gu = layout_gu + expert as u64 * per_expert;
        assert_eq!(cache_gu.byte_offset, kernel_gu, "expert {expert} gu offset");
        let cache_bytes = cache_gu.src_slice();
        let blob_ptr = unsafe {
            cache_gu
                .weight_buffer()
                .0
                .contents()
                .as_ptr()
                .add(kernel_gu as usize)
        };
        let kernel_bytes =
            unsafe { std::slice::from_raw_parts(blob_ptr as *const u8, cache_bytes.len().min(64)) };
        assert_eq!(
            &kernel_bytes[..64],
            &cache_bytes[..64],
            "expert {expert} first 64 gu bytes"
        );
    }

    // Mirror act from cache bytes + dump moe_in should match CPU gate_act, not GPU act.
    let dump_path = Path::new("/tmp/rust_moe_single_l2_act_probe.json");
    if !dump_path.exists() {
        eprintln!("skip act compare: run step-moe-single-dump first");
        return;
    }
    let dump: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dump_path).expect("read dump"))
            .expect("json");
    let moe_in: Vec<f32> = dump["moe_in"]
        .as_array()
        .expect("moe_in")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let gpu_act: Vec<f32> = dump["gpu_act_after_barrier"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let expert = 18usize;
    let gate_up = cache.expert_gate_up_q4(layer, expert);
    let down = cache.expert_down_q4(layer, expert);
    let mut scratch = MoeScratch::new(1, &text);
    let staged = expert_forward_staged_dgq(
        &moe_in,
        &gate_up,
        &down,
        text.moe_intermediate_size,
        text.hidden_size,
        &mut scratch,
    );
    let cos_staged_gpu_act = {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (a, b) in staged.gate_act.iter().zip(gpu_act.iter()) {
            dot += *a as f64 * *b as f64;
            na += *a as f64 * *a as f64;
            nb += *b as f64 * *b as f64;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    };
    eprintln!("E18 act: cos(staged_gate_act,gpu_act)={cos_staged_gpu_act:.4} (expect ~0.015)");
}

/// NONDET-SC-1 diagnostic: run the chunked SC softembed twice on identical
/// seeded logits and diff. Isolates the run-to-run nondeterminism to a single
/// softembed invocation (no generation loop). `--ignored` (needs a model dir).
#[test]
#[ignore]
fn sc_chunked_softembed_nondeterminism_probe() {
    use std::path::Path;

    let dir = [
        Path::new("model/diffgemma-26b-a4b-it-q4"),
        Path::new("/tmp/quantized-weights"),
        Path::new("model/diffusiongemma-q4"),
        Path::new("model/q4"),
    ]
    .into_iter()
    .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
    let Some(dir) = dir else {
        eprintln!("skip sc_chunked_softembed_nondeterminism_probe");
        return;
    };
    let cfg = StepSmokeConfig {
        finish: StepFinishMode::Full,
        steps: 1,
        ..StepSmokeConfig::default()
    };
    let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
    let layout = rt.layout;
    rt.run_forward_once(StepFinishMode::Full)
        .expect("seed logits");
    let soft_elems = CANVAS * HID;
    let soft_off = rt.bufs.arena_map.soft_off() as usize;

    // (soft bf16-as-f32, gemm_b f32 accumulator). If gemm_b differs the
    // accumulate/GEMM is the source; if only soft differs the convert is.
    let run_chunked = |rt: &mut StepRuntime| -> (Vec<f32>, Vec<f32>) {
        write_buffer_bytes(&rt.bufs.arena, soft_off, &vec![0u8; soft_elems * 2]);
        rt.dispatch_and_wait(|enc| {
            enc.encode_sc_logit_rowstats();
            enc.encode_sc_softembed_exact(&layout)?;
            Ok(())
        })
        .expect("chunked sc softembed");
        let soft = read_arena_buffer_f32(&rt.bufs.arena, soft_off, soft_elems);
        let gb_ptr = rt.bufs.gemm_b.contents().as_ptr() as *const f32;
        let gb = (0..soft_elems).map(|i| unsafe { *gb_ptr.add(i) }).collect();
        (soft, gb)
    };

    let (run1, gb1) = run_chunked(&mut rt);
    let (run2, gb2) = run_chunked(&mut rt);
    let (run3, _gb3) = run_chunked(&mut rt);

    // Where does the f32 accumulator (gemm_b) first diverge?
    let mut gb_max = 0.0f32;
    let mut gb_first = None;
    for (i, (a, b)) in gb1.iter().zip(gb2.iter()).enumerate() {
        let d = (a - b).abs();
        if d > 0.0 && gb_first.is_none() {
            gb_first = Some((i, i / HID, i % HID, *a, *b));
        }
        gb_max = gb_max.max(d);
    }
    eprintln!(
        "NONDET-SC gemm_b accumulator run1-vs-run2: max_abs={gb_max:.6} first(idx,row,col,a,b)={gb_first:?}"
    );

    let mut max_abs = 0.0f32;
    let mut ndiff = 0usize;
    let mut first = None;
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        let d = (a - b).abs();
        if d > 0.0 {
            ndiff += 1;
            if first.is_none() {
                first = Some((i, *a, *b));
            }
        }
        max_abs = max_abs.max(d);
    }
    let diff13 = run1
        .iter()
        .zip(run3.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "NONDET-SC chunked softembed run1-vs-run2: max_abs={max_abs:.6} ndiff={ndiff}/{soft_elems} first={first:?}; run1-vs-run3 max_abs={diff13:.6}"
    );
    // Regression guard for the memzero-coverage bug (NONDET-SC-1): the chunked
    // SC softembed must be bit-deterministic on identical seeded logits.
    assert_eq!(gb_max, 0.0, "gemm_b accumulator nondeterministic");
    assert_eq!(max_abs, 0.0, "chunked softembed nondeterministic");
    assert_eq!(diff13, 0.0, "chunked softembed nondeterministic (run3)");
}

/// Offset-prefill bit-identity (cross-turn KV reuse, e3f79cd):
/// `prefill_chunks(prefix)` + `prefill_chunks_from(offset, delta)` must
/// produce KV bytes identical to a single `prefill_chunks(all)` over the
/// valid positions [0..n), every layer. The offset is deliberately NOT a
/// chunk multiple so the delta chunk boundary lands mid-canvas. `--ignored`
/// (needs a model dir).
#[test]
#[ignore]
fn offset_prefill_kv_bit_identity() {
    use std::path::Path;

    let dir = [
        Path::new("model/diffgemma-26b-a4b-it-q4"),
        Path::new("/tmp/quantized-weights"),
        Path::new("model/diffusiongemma-q4"),
        Path::new("model/q4"),
    ]
    .into_iter()
    .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
    let Some(dir) = dir else {
        eprintln!("skip offset_prefill_kv_bit_identity");
        return;
    };
    let cfg = StepSmokeConfig {
        finish: StepFinishMode::Full,
        steps: 1,
        // Mid-chunk offset resume writes up to offset+CANVAS: needs headroom.
        max_seq: 1024,
        ..StepSmokeConfig::default()
    };
    let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
    let layout = rt.layout;

    // Spans two chunks in the full run; split point mid-chunk.
    let n = 400usize;
    let offset = 300usize;
    let tokens: Vec<u32> = (0..n)
        .map(|i| ((i.wrapping_mul(2654435761)) % 200_000) as u32 + 5)
        .collect();

    let read_kv_prefix = |rt: &StepRuntime, n: usize| -> Vec<Vec<u8>> {
        let ptr = rt.bufs.kvcache.contents().as_ptr() as *const u8;
        (0..N_LAYERS)
            .map(|layer| {
                let l = &layout.layers[layer];
                let token_stride = (l.n_kv_heads as usize) * (l.head_dim as usize) * 2 * 2;
                let base = l.kv_region as usize;
                unsafe { std::slice::from_raw_parts(ptr.add(base), n * token_stride).to_vec() }
            })
            .collect()
    };

    let kv_len = rt.prefill_chunks(&tokens).expect("full prefill");
    assert_eq!(kv_len, n);
    let full = read_kv_prefix(&rt, n);

    zero_buffer(&rt.bufs.kvcache);
    let k1 = rt
        .prefill_chunks(&tokens[..offset])
        .expect("prefix prefill");
    assert_eq!(k1, offset);
    let k2 = rt
        .prefill_chunks_from(offset, &tokens[offset..])
        .expect("offset prefill");
    assert_eq!(k2, n);
    let split = read_kv_prefix(&rt, n);

    let mut bad = 0usize;
    for layer in 0..N_LAYERS {
        if full[layer] != split[layer] {
            let l = &layout.layers[layer];
            let token_stride = (l.n_kv_heads as usize) * (l.head_dim as usize) * 2 * 2;
            let first = full[layer]
                .iter()
                .zip(split[layer].iter())
                .position(|(a, b)| a != b)
                .unwrap();
            eprintln!(
                "layer {layer}: KV mismatch first at byte {first} (pos {})",
                first / token_stride
            );
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "offset prefill not bit-identical to full prefill");
}

// Full-GPU-state snapshot/restore harness (all step buffers). Currently unused;
// kept as the primitive for KV-rewind / run-to-run determinism debugging.
#[allow(dead_code)]
fn read_buffer_bytes(buf: &ProtocolObject<dyn MTLBuffer>, byte_off: usize, len: usize) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(buf.contents().as_ptr().add(byte_off) as *const u8, len).to_vec()
    }
}

fn write_buffer_bytes(buf: &ProtocolObject<dyn MTLBuffer>, byte_off: usize, data: &[u8]) {
    unsafe {
        std::slice::from_raw_parts_mut(
            buf.contents().as_ptr().add(byte_off) as *mut u8,
            data.len(),
        )
        .copy_from_slice(data);
    }
}

#[allow(dead_code)]
struct StepGpuSnapshot {
    state: CanvasState,
    arena: Vec<u8>,
    logits: Vec<u8>,
    sc_probs: Vec<u8>,
    kvcache: Vec<u8>,
    route: Vec<u8>,
    gemm_a: Vec<u8>,
    gemm_b: Vec<u8>,
}

#[allow(dead_code)]
fn snapshot_step_gpu(rt: &StepRuntime) -> StepGpuSnapshot {
    StepGpuSnapshot {
        state: rt.read_canvas_state(),
        arena: read_buffer_bytes(&rt.bufs.arena, 0, rt.bufs.arena_map.bytes() as usize),
        logits: read_buffer_bytes(&rt.bufs.logits, 0, CANVAS * VOCAB * 2),
        sc_probs: read_buffer_bytes(&rt.bufs.sc_probs, 0, rt.bufs.sc_probs.length()),
        kvcache: read_buffer_bytes(&rt.bufs.kvcache, 0, rt.bufs.kvcache.length()),
        route: read_buffer_bytes(&rt.bufs.route, 0, rt.bufs.route.length()),
        gemm_a: read_buffer_bytes(&rt.bufs.gemm_a, 0, rt.bufs.gemm_a.length()),
        gemm_b: read_buffer_bytes(&rt.bufs.gemm_b, 0, rt.bufs.gemm_b.length()),
    }
}

#[allow(dead_code)]
fn restore_step_gpu(rt: &mut StepRuntime, snap: &StepGpuSnapshot) {
    rt.write_canvas_state(&snap.state);
    write_buffer_bytes(&rt.bufs.arena, 0, &snap.arena);
    write_buffer_bytes(&rt.bufs.logits, 0, &snap.logits);
    write_buffer_bytes(&rt.bufs.sc_probs, 0, &snap.sc_probs);
    write_buffer_bytes(&rt.bufs.kvcache, 0, &snap.kvcache);
    write_buffer_bytes(&rt.bufs.route, 0, &snap.route);
    write_buffer_bytes(&rt.bufs.gemm_a, 0, &snap.gemm_a);
    write_buffer_bytes(&rt.bufs.gemm_b, 0, &snap.gemm_b);
}

#[test]
fn fused_gate_up_gemm_matches_split_in_full_arena() {
    use crate::metal::DgqGpuBlob;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::QuantFormat;
    use crate::shaders::gemm_block_stacked::pipeline_for;
    use crate::shaders::gemm_q4;
    use std::path::Path;

    fn read_plane(
        arena: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        byte_off: u64,
    ) -> Vec<f32> {
        let elems = CANVAS * DENSE_FF as usize;
        let ptr = arena.contents().as_ptr() as *const u16;
        let base = (byte_off / 2) as usize;
        (0..elems)
            .map(|i| crate::shaders::bf16::bf16_bits_to_f32(unsafe { *ptr.add(base + i) }))
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    let dir = [
        Path::new("model/diffgemma-26b-a4b-it-q4"),
        Path::new("/tmp/quantized-weights"),
    ]
    .into_iter()
    .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
    let Some(dir) = dir else {
        eprintln!("skip fused_gate_up_gemm_matches_split_in_full_arena");
        return;
    };
    let store = crate::dgq::store::DgqStore::open(dir).expect("dgq");
    let offsets = build_offsets_from_store(&store);
    let layout = build_layout(&offsets, 512);
    let arena_map = step_arena_layout(&crate::metal::step_config::ModelDims::reference());
    let l = &layout.layers[0];
    let (segs, n_total) = gate_up_stacked_segments(l, &arena_map, DENSE_FF);

    let ctx = MetalContext::new().expect("metal");
    let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
    let mut pool = BufferPool::new();
    let arena = pool
        .allocate(&ctx.device, arena_map.bytes() as usize)
        .expect("arena");

    let m = CANVAS;
    let k = HID;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 + 11.0) * 0.0006).sin() * 0.14)
        .collect();
    let x_bits = crate::shaders::bf16::f32_slice_to_bf16_bits(&x);
    unsafe {
        let dst = arena.contents().as_ptr().add(arena_map.tmp_off() as usize) as *mut u16;
        std::ptr::copy_nonoverlapping(x_bits.as_ptr(), dst, x_bits.len());
    }

    let pipeline =
        pipeline_for(&ctx, n_total, k as u32, QuantFormat::Q4Affine, &segs).expect("pipe");
    let (grid, tg) = crate::shaders::gemm_common::dispatch_shape(m, n_total as usize);
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = cmd.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipeline.pipeline);
    let m_u32 = m as u32;
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&arena), arena_map.tmp_off() as usize, 0);
        enc.setBuffer_offset_atIndex(Some(&arena), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&gpu_blob.buffer), 0, 2);
    }
    crate::shaders::gpu_common::set_bytes(&enc, &m_u32, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let fused_gate = read_plane(&arena, arena_map.ffg_off());
    let fused_up = read_plane(&arena, arena_map.ffu_off());

    let arena2 = pool
        .allocate(&ctx.device, arena_map.bytes() as usize)
        .expect("arena2");
    unsafe {
        let dst = arena2.contents().as_ptr().add(arena_map.tmp_off() as usize) as *mut u16;
        std::ptr::copy_nonoverlapping(x_bits.as_ptr(), dst, x_bits.len());
    }
    for (w_off, y_off) in [
        (l.mlp_gate, arena_map.ffg_off()),
        (l.mlp_up, arena_map.ffu_off()),
    ] {
        let pipeline1 = gemm_q4::pipeline_for(&ctx, DENSE_FF, k as u32).expect("split pipe");
        let cmd2 = ctx.queue.commandBuffer().expect("cmd");
        let enc2 = cmd2.computeCommandEncoder().expect("enc");
        enc2.setComputePipelineState(&pipeline1.pipeline);
        unsafe {
            enc2.setBuffer_offset_atIndex(Some(&arena2), arena_map.tmp_off() as usize, 0);
            enc2.setBuffer_offset_atIndex(Some(&arena2), y_off as usize, 1);
            enc2.setBuffer_offset_atIndex(Some(&gpu_blob.buffer), 0, 2);
        }
        crate::shaders::gpu_common::set_bytes(&enc2, &w_off, 3);
        crate::shaders::gpu_common::set_bytes(&enc2, &m_u32, 4);
        let (g2, t2) = crate::shaders::gemm_common::dispatch_shape(m, DENSE_FF as usize);
        enc2.dispatchThreadgroups_threadsPerThreadgroup(g2, t2);
        enc2.endEncoding();
        cmd2.commit();
        cmd2.waitUntilCompleted();
    }
    let gate_max = max_diff(&fused_gate, &read_plane(&arena2, arena_map.ffg_off()));
    let up_max = max_diff(&fused_up, &read_plane(&arena2, arena_map.ffu_off()));
    eprintln!("arena gate_up fused vs split: gate_max={gate_max:.4e} up_max={up_max:.4e}");
    assert!(gate_max < 0.05, "gate plane max_abs={gate_max}");
    assert!(up_max < 0.05, "up plane max_abs={up_max}");
}

#[test]
fn generate_path_reset_block_matches_step_smoke_ids() {
    use crate::sample::Rng;
    use std::path::Path;

    let dir = [
        Path::new("model/diffgemma-26b-a4b-it-q4"),
        Path::new("/tmp/quantized-weights"),
    ]
    .into_iter()
    .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
    let Some(dir) = dir else {
        eprintln!("skip generate_path_reset_block_matches_step_smoke_ids");
        return;
    };
    let prompt = vec![23391u32]; // raw "hello" (matches generate-monolithic --raw)
    let cfg = StepSmokeConfig {
        layers: 3,
        steps: 1,
        seed: 42,
        prefill_token_ids: Some(prompt),
        ..StepSmokeConfig::default()
    };
    let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
    let eos = rt.read_params().eos_token_id;
    let sampler = crate::sample::sampler_for_steps(4, true);
    let params = step_params_from_sampler(&sampler, rt.read_params().kv_len, true, eos);
    let mut rng = Rng::new(42);
    rt.reset_block(VOCAB, &mut rng, params);
    for _ in 0..4 {
        rt.run_denoise_step().expect("denoise");
    }
    let ids = rt.read_canvas_state().ids;
    let argmax = rt.read_canvas_state().prev_argmax;
    eprintln!("generate-path 4x ids[:8]={:?}", &ids[..8]);
    eprintln!("generate-path 4x argmax[:8]={:?}", &argmax[..8]);
}

#[test]
fn fusion_matches_unfused_forward_logits() {
    use crate::sample::Rng;
    use std::path::Path;

    fn row0_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn forward_row0_logits(dir: &Path, cfg: &StepSmokeConfig, fused: bool) -> Vec<f32> {
        let mut fcfg = crate::flags::RuntimeConfig::default();
        fcfg.perf.fused_algebra = fused;
        let _g = crate::flags::install_for_test(fcfg);
        let mut cfg = cfg.clone();
        cfg.finish = StepFinishMode::ForwardOnly;
        let (mut rt, _) = build_step_runtime(dir, &cfg).expect("runtime");
        let sampler = crate::sample::sampler_for_steps(4, true);
        let params = step_params_from_sampler(
            &sampler,
            rt.read_params().kv_len,
            true,
            rt.read_params().eos_token_id,
        );
        let mut rng = Rng::new(42);
        rt.reset_block(VOCAB, &mut rng, params);
        rt.run_forward_once(StepFinishMode::ForwardOnly)
            .expect("forward");
        read_half_buffer_f32(&rt.bufs.logits, 0, VOCAB)
    }

    let dir = [
        Path::new("model/diffgemma-26b-a4b-it-q4"),
        Path::new("/tmp/quantized-weights"),
    ]
    .into_iter()
    .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
    let Some(dir) = dir else {
        eprintln!("skip fusion_matches_unfused_forward_logits");
        return;
    };
    let cfg = StepSmokeConfig {
        layers: 3,
        steps: 1,
        seed: 42,
        prefill_token_ids: Some(vec![23391u32]),
        ..StepSmokeConfig::default()
    };

    let off = forward_row0_logits(dir, &cfg, false);
    let on = forward_row0_logits(dir, &cfg, true);
    let off2 = forward_row0_logits(dir, &cfg, false);

    let max = row0_max_abs_diff(&off, &on);
    let baseline = row0_max_abs_diff(&off, &off2);
    eprintln!("fusion row0 logits max_abs: fused={max:.4e} unfused_repeat={baseline:.4e}");
    assert!(baseline < 1e-5, "unfused not deterministic: {baseline}");

    let mut smoke = cfg.clone();
    smoke.finish = StepFinishMode::Full;
    smoke.steps = 1;
    smoke.prefill_token_ids = Some(vec![23391u32]);
    let (st_off, logits_off) = {
        let mut c = crate::flags::RuntimeConfig::default();
        c.perf.fused_algebra = false;
        let _g = crate::flags::install_for_test(c);
        let (mut rt_off, _) = build_step_runtime(dir, &smoke).expect("rt off");
        rt_off.run_forward_once(StepFinishMode::Full).expect("step");
        (
            rt_off.read_canvas_state(),
            read_half_buffer_f32(&rt_off.bufs.logits, 0, VOCAB),
        )
    };
    let (st_on, logits_on) = {
        let mut c = crate::flags::RuntimeConfig::default();
        c.perf.fused_algebra = true;
        let _g = crate::flags::install_for_test(c);
        let (mut rt_on, _) = build_step_runtime(dir, &smoke).expect("rt on");
        rt_on.run_forward_once(StepFinishMode::Full).expect("step");
        (
            rt_on.read_canvas_state(),
            read_half_buffer_f32(&rt_on.bufs.logits, 0, VOCAB),
        )
    };
    let full_row0 = row0_max_abs_diff(&logits_off, &logits_on);
    eprintln!(
        "full-step ids[:4] off={:?} on={:?} row0_logits_max_abs={full_row0:.4e}",
        &st_off.ids[..4],
        &st_on.ids[..4],
    );
    assert_eq!(st_off.ids, st_on.ids, "sampled ids diverged");
    assert!(
        full_row0 < 0.05,
        "full step row0 logits max_abs={full_row0}"
    );
}
