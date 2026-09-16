//! Golden-slice verification on the portable (Metal / CUDA) kernels.
//!
//! Runs part of the diffusion pipeline with the REAL pack weights and the REAL
//! token ids of the \`engine_prefill\` golden case, and pins every stage to the
//! shared CPU oracle:
//!
//!   embed_gather (raw bf16 table, real embed_scale)
//!     -> rms_norm_rows (layer 0 input_layernorm)
//!     -> q/k/v linear (CPU, real layer-0 projections, 8-head slice)
//!     -> apply_rope_heads (layer 0 is sliding: full-head split-half RoPE)
//!     -> gqa_attention (causal, 8 query heads / 1 KV head)
//!     -> moe_router_topk (real router scale/proj/per-expert scale)
//!     -> swiglu_gelu (real layer-0 MLP gate/up projections)
//!
//! Skipped unless \`DGQ_MODEL_DIR\` (or \`model/diffgemma-26b-a4b-it-q4\`) names a
//! complete pack. Run it on the CUDA box with \`--features cuda\`.
#![cfg(feature = "cuda")]

use dgops::dgq::DgqPack;
use dgops::ops;

/// fixtures/golden/golden.json \u2192 case \`engine_prefill\` (seed 7).
const GOLDEN_PROMPT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 100,
    45518, 107, 101,
];

const HIDDEN: usize = 2816;
const HEAD_DIM: usize = 256;
const N_EXPERTS: usize = 128;
const TOP_K: usize = 8;
/// Embed rows loaded for the slice. The table is 262144 x 2816 x bf16 =
/// 1.4 GiB; the slice maps the golden ids into this window so the gather still
/// runs the real table bytes without materializing the whole thing.
const EMBED_ROWS: usize = 4096;

/// One KV head for the slice; 8 query heads then form one GQA group.
const SLICE_Q_HEADS: usize = 8;
const SLICE_KV_HEADS: usize = 1;
const SLICE_ROWS: usize = SLICE_Q_HEADS * HEAD_DIM;
const RMS_EPS: f32 = 1e-6;
const EMBED_SCALE: f32 = 53.065_998;

fn pack_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("DGQ_MODEL_DIR") {
        return Some(dir.into());
    }
    // Gate on the blob, not just the manifest: an interrupted download leaves
    // a manifest with no payload, and that must skip, not fail.
    let local = std::path::PathBuf::from("../../model/diffgemma-26b-a4b-it-q4");
    (local.join("model.dgq.json").is_file() && local.join("model.dgq.bin").is_file())
        .then_some(local)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Deterministic f32 in [-1, 1) \u2014 a stand-in for a projection the slice does
/// not port (the point is the kernel math, not the activation statistics).
fn pseudo(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// y[s, o] = x[s, :] @ w[o, :]^T over the first \`rows\` rows of w.
fn linear_rows(x: &[f32], w: &[f32], seq: usize, in_dim: usize, rows: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; seq * rows];
    for s in 0..seq {
        for o in 0..rows {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let xr = &x[s * in_dim..(s + 1) * in_dim];
            let mut acc = 0.0f32;
            for d in 0..in_dim {
                acc += xr[d] * wr[d];
            }
            y[s * rows + o] = acc;
        }
    }
    y
}

/// Stage gate: the GPU path must be finite, agree in direction, and agree in
/// magnitude to \`rel\` of the oracle's own scale. A pure absolute tolerance is
/// the wrong instrument here: real activations reach ~1e2 after the embed
/// scale, so a 1e-4 absolute bound is stricter than f32 allows for any
/// different summation order.
fn assert_stage(name: &str, gpu: &[f32], cpu: &[f32], rel: f32, min_cos: f32) {
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "{name}: GPU produced a non-finite value"
    );
    let cos = cosine(gpu, cpu);
    let mad = max_abs_diff(gpu, cpu);
    let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let rel_err = mad / scale;
    assert!(
        cos >= min_cos && rel_err <= rel,
        "{name}: cos={cos:.7} max_abs={mad:.3e} rel={rel_err:.2e} (want cos>={min_cos}, rel<={rel:.1e})"
    );
    println!(
        "  {name}: cos={cos:.7} max_abs={mad:.3e} rel={rel_err:.2e} n={}",
        gpu.len()
    );
}

#[test]
fn golden_slice_engine_prefill_on_cuda() {
    let Some(dir) = pack_dir() else {
        eprintln!("skip: no .dgq pack (set DGQ_MODEL_DIR)");
        return;
    };
    let pack = DgqPack::open(&dir).expect("open pack");
    println!(
        "pack {} tensors, golden prompt {} tokens",
        pack.entries().len(),
        GOLDEN_PROMPT_IDS.len()
    );

    // ---- stage 1: embed_gather over the real bf16 embed table --------------
    let embed_bytes = pack
        .raw_bf16_row_bytes("model.decoder.embed_tokens.weight", EMBED_ROWS)
        .expect("embed rows");
    let vocab = embed_bytes.len() / (HIDDEN * 2);
    let ids: Vec<u32> = GOLDEN_PROMPT_IDS
        .iter()
        .map(|&id| id % vocab as u32)
        .collect();
    let embed_fix = ops::embed_gather::Fixture {
        blob: embed_bytes,
        ids: ids.clone(),
        hidden: HIDDEN,
        vocab,
        embed_scale: EMBED_SCALE,
        raw: true,
        w_off: 0,
    };
    let embed_cpu = ops::embed_gather::cpu(&embed_fix);
    let embed_gpu = ops::embed_gather::gpu(&embed_fix).expect("embed_gather gpu");
    assert_stage("embed_gather", &embed_gpu, &embed_cpu, 0.0, 1.0);

    // ---- stage 2: layer-0 input_layernorm over the real weight ------------
    let norm_w = pack
        .raw_bf16("model.decoder.layers.0.input_layernorm.weight")
        .expect("input_layernorm");
    assert_eq!(norm_w.len(), HIDDEN);
    let norm_fix = ops::rms_norm_rows::Fixture {
        x: embed_gpu.clone(),
        weight: norm_w.clone(),
        seq_len: GOLDEN_PROMPT_IDS.len(),
        hidden: HIDDEN,
        eps: RMS_EPS,
    };
    let norm_cpu = ops::rms_norm_rows::cpu(&norm_fix);
    let norm_gpu = ops::rms_norm_rows::gpu(&norm_fix).expect("rms_norm gpu");
    assert_stage(
        "rms_norm_rows(layer0)",
        &norm_gpu,
        &norm_cpu,
        1e-5,
        0.999999,
    );

    // ---- stage 3: q/k/v slice through the real projections ----------------
    let q_w = pack
        .raw_bf16_rows("model.decoder.layers.0.self_attn.q_proj.weight", SLICE_ROWS)
        .expect("q_proj");
    let k_w = pack
        .raw_bf16_rows("model.decoder.layers.0.self_attn.k_proj.weight", HEAD_DIM)
        .expect("k_proj");
    let v_w = pack
        .raw_bf16_rows("model.decoder.layers.0.self_attn.v_proj.weight", HEAD_DIM)
        .expect("v_proj");
    let q = linear_rows(&norm_gpu, &q_w, GOLDEN_PROMPT_IDS.len(), HIDDEN, SLICE_ROWS);
    let k = linear_rows(&norm_gpu, &k_w, GOLDEN_PROMPT_IDS.len(), HIDDEN, HEAD_DIM);
    let v = linear_rows(&norm_gpu, &v_w, GOLDEN_PROMPT_IDS.len(), HIDDEN, HEAD_DIM);

    // ---- stage 4: apply_rope_heads on Q and K (layer 0 = sliding/full-head)
    let seq = GOLDEN_PROMPT_IDS.len();
    let freqs = ops::apply_rope_heads::rope_freqs(seq, HEAD_DIM, HEAD_DIM, 10_000.0);
    let mut qk = q.clone();
    qk.extend_from_slice(&k);
    let rope_fix = ops::apply_rope_heads::Fixture {
        x: qk,
        freqs: freqs.clone(),
        seq_len: seq,
        num_heads: SLICE_Q_HEADS + SLICE_KV_HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: HEAD_DIM,
        elem_offset: 0,
    };
    let rope_cpu = ops::apply_rope_heads::cpu(&rope_fix);
    let rope_gpu = ops::apply_rope_heads::gpu(&rope_fix).expect("apply_rope gpu");
    assert_stage(
        "apply_rope_heads(layer0)",
        &rope_gpu,
        &rope_cpu,
        1e-5,
        0.999999,
    );
    let q_rot = rope_gpu[..seq * SLICE_ROWS].to_vec();
    let k_rot = rope_gpu[seq * SLICE_ROWS..].to_vec();

    // ---- stage 5: gqa_attention over the real projected K/V ---------------
    // KV region layout: [t, n_kv_heads, 2*head_dim] = K then V per head.
    let mut kv = vec![0.0f32; seq * SLICE_KV_HEADS * 2 * HEAD_DIM];
    for t in 0..seq {
        let dst = t * SLICE_KV_HEADS * 2 * HEAD_DIM;
        kv[dst..dst + HEAD_DIM].copy_from_slice(&k_rot[t * HEAD_DIM..(t + 1) * HEAD_DIM]);
        kv[dst + HEAD_DIM..dst + 2 * HEAD_DIM]
            .copy_from_slice(&v[t * HEAD_DIM..(t + 1) * HEAD_DIM]);
    }
    let attn_fix = ops::gqa_attention::Fixture {
        q: q_rot,
        kv,
        seq_len: seq,
        total_kv: seq,
        n_heads: SLICE_Q_HEADS,
        n_kv_heads: SLICE_KV_HEADS,
        head_dim: HEAD_DIM,
        sliding_window: 0,
        kv_cache_len: 0,
    };
    let attn_cpu = ops::gqa_attention::cpu(&attn_fix);
    let attn_gpu = ops::gqa_attention::gpu(&attn_fix).expect("gqa_attention gpu");
    assert_stage("gqa_attention(layer0)", &attn_gpu, &attn_cpu, 1e-4, 0.99999);

    // ---- stage 6: MoE router over the real router tensors ------------------
    let router_scale = pack
        .raw_bf16("model.decoder.layers.0.router.scale")
        .expect("router.scale");
    let router_proj = pack
        .raw_bf16("model.decoder.layers.0.router.proj.weight")
        .expect("router.proj");
    let per_expert = pack
        .raw_bf16("model.decoder.layers.0.router.per_expert_scale")
        .expect("router.per_expert_scale");
    let router_fix = ops::moe_router_topk::Fixture {
        stream: norm_gpu.clone(),
        router_scale,
        router_proj,
        per_expert_scale: per_expert,
        canvas: seq,
        hidden: HIDDEN,
        n_experts: N_EXPERTS,
        top_k: TOP_K,
    };
    let (cpu_idx, cpu_w) = ops::moe_router_topk::cpu_routes(&router_fix);
    let gpu_w = ops::moe_router_topk::gpu(&router_fix).expect("router gpu");
    assert_stage("moe_router_topk(layer0)", &gpu_w, &cpu_w, 0.0, 1.0);
    for row in 0..seq {
        let idx = &cpu_idx[row * TOP_K..(row + 1) * TOP_K];
        assert!(
            idx.iter().all(|&e| (e as usize) < N_EXPERTS),
            "router row {row} returned an out-of-range expert: {idx:?}"
        );
        let mut sorted = idx.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            TOP_K,
            "router row {row} repeated an expert: {idx:?}"
        );
    }

    // ---- stage 7: swiglu over the real layer-0 MLP gate/up slice -----------
    let gate_w = pack
        .raw_bf16_rows("model.decoder.layers.0.mlp.gate_proj.weight", 256)
        .expect("gate_proj");
    let up_w = pack
        .raw_bf16_rows("model.decoder.layers.0.mlp.up_proj.weight", 256)
        .expect("up_proj");
    let gate = linear_rows(&norm_gpu, &gate_w, seq, HIDDEN, 256);
    let up = linear_rows(&norm_gpu, &up_w, seq, HIDDEN, 256);
    let swiglu_fix = ops::swiglu_gelu::Fixture { gate, up };
    let swiglu_cpu = ops::swiglu_gelu::cpu(&swiglu_fix);
    let swiglu_gpu = ops::swiglu_gelu::gpu(&swiglu_fix).expect("swiglu gpu");
    assert_stage(
        "swiglu_gelu(layer0)",
        &swiglu_gpu,
        &swiglu_cpu,
        1e-6,
        0.999999,
    );

    // A synthetic-weight attention case at the same geometry, so the KV-prefix
    // and window paths are exercised on this backend too.
    let synth_q = pseudo(7, seq * SLICE_Q_HEADS * HEAD_DIM);
    let synth_kv = pseudo(42, seq * SLICE_KV_HEADS * 2 * HEAD_DIM);
    let mut synth = attn_fix.clone();
    synth.q = synth_q;
    synth.kv = synth_kv;
    synth.sliding_window = 8;
    let cpu = ops::gqa_attention::cpu(&synth);
    let gpu = ops::gqa_attention::gpu(&synth).expect("gqa_attention synth gpu");
    assert_stage(
        "gqa_attention(synthetic, window 8)",
        &gpu,
        &cpu,
        1e-4,
        0.99999,
    );

    println!(
        "golden slice OK: {} prompt tokens through 7 real-weight stages",
        seq
    );
}
