//! KV lineage path-independence (Tier-1 correctness probe).
//!
//! The same token sequence built via (a) one fresh prefill, (b) prefix
//! prefill + delta-extend, or (c) overshoot + truncate (+ re-extend — the
//! finalize thought-strip shape) must yield an identical READABLE KV:
//! `live_kv_fingerprint` (linear layers whole prefix, ring layers
//! window-only, ring-residue-invariant). A live serve turn on reused-delta
//! KV walked a different trajectory than a byte-identical fresh-prefill
//! replay (memory: kv-reuse-delta-divergence) — one of these seams is the
//! suspect; this isolates them generation-free.

use crate::metal::step_generate::{StepGenerateConfig, StepGenerateSession};
use crate::metal::step_kernel::N_LAYERS;
use crate::sample;

use super::engine_extend_bench_tests::{model_dir, synth_ids};

/// Rebuild the session's KV from scratch via `build`, then fingerprint.
fn fingerprint(
    session: &mut StepGenerateSession,
    build: impl FnOnce(&mut StepGenerateSession),
) -> u64 {
    session.reset_kv();
    build(session);
    session.live_kv_fingerprint()
}

#[test]
fn kv_lineage_paths_are_fingerprint_identical() {
    lineage_matrix(false);
}

/// f16 lineage across delta offsets that are NOT 256-aligned — the axis
/// [`lineage_matrix`] structurally cannot reach.
///
/// The sliding ring (4096) is a multiple of the 256-token prefill chunk, so
/// with a 256-ALIGNED resume offset a chunk can never straddle the ring
/// wrap. Every case in the matrix was aligned (offsets 6144 / 5376), while
/// the diverging live turn resumed at 14,509 (residue 173) — so a
/// wrap-straddling chunk, and chunk-phase effects generally, were never
/// under test. Cases here: a plain misaligned resume, one whose delta chunk
/// STRADDLES the ring wrap, and the live turn's exact shape
/// (14,509 + 266). Fresh vs prefix+delta must still agree.
///
/// Buffers are scrubbed between arms: a `reset_kv` alone leaves residue in
/// rows the next build never writes, which is what made the q8 arm look
/// path-dependent (see `q8_ring_wrap_divergence_probe`).
///
/// KNOWN RED (2026-07-19) — and it reproduces the LIVE configuration on the
/// production f16 path, so this is the real "KV-lineage drift", not q8.
/// Verdicts:
///   - misaligned resume, delta fits ONE chunk           -> OK
///   - misaligned resume, delta spans TWO chunks         -> FORK
///   - the live turn's exact shape (14,509 + 266)        -> FORK
/// The fork needs all three of: a WRAPPED ring, a MISALIGNED resume, and a
/// MULTI-CHUNK delta. Any one removed and it passes — below the ring
/// (`kv_lineage_alignment_vs_chunkcount_sweep`, totals < 4096) every
/// alignment x chunk-count combination is clean, and the original
/// 256-ALIGNED matrix passes at every size. Batched super-chunks are
/// EXONERATED (`kv_lineage_unaligned_batch_bisect`: identical verdicts with
/// `prefill_batch` on and off).
///
/// CULPRIT: **E20 top-k sparse attention** (`DGQ_ATTN_TOPK`, default ON).
/// `kv_lineage_fork_attention_bisect` runs the live case under each
/// attention lever; only `attn_topk=0` comes back clean (gemm_attn=0,
/// attn_mma=0, attn_mma_full=0 all still FORK).
///
/// Mechanism: top-k's parameters are derived from the DISPATCH's
/// `t_total = kv_len + canvas`, not from the query's own causal context —
/// so the same absolute position gets a different approximation depending
/// on how the prefill was chunked. Concretely, `attn_topk_k_for` computes
/// `k = (t_total/128).clamp(64,512)`, and in the live case the same query
/// gets k=114 in the fresh arm vs k=115 in the delta arm (also 116 vs 115,
/// 116 vs 117 further in) — a different NUMBER OF ATTENDED KEYS for
/// identical Q and identical KV. Note case 2 forks even where k is EQUAL in
/// both arms, so the t_total-dependent selection/tiling is a second
/// phase-dependent input; that one is not yet isolated.
///
/// This also explains the layer pattern, and corrects an earlier misread:
/// top-k runs on FULL layers (5, 11, 17, 23, 29), and a layer's KV is
/// written BEFORE its attention (QkRopeKv -> Attention). So layer 5's KV is
/// identical while its attention OUTPUT differs, and the first KV to carry
/// the difference is layer 6. **Layer 6 is "first top-k layer + 1", NOT a
/// sub-ULP visibility threshold** — do not go hunting for a tiny seed in
/// layers 0-5; there isn't one.
///
/// Fix direction: make k (and the selection) depend on the query's own
/// causal key count (`kv_len + tok`) rather than the dispatch's `t_total`,
/// which is phase-invariant and should preserve E20's win.
#[test]
#[ignore = "KNOWN RED: wrapped-ring + misaligned + multi-chunk delta forks (PLAN correctness debt)"]
fn kv_lineage_unaligned_delta_offsets_are_fingerprint_identical() {
    unaligned_matrix(None);
}

/// Same matrix with the batched prefill super-chunk forced OFF, to test its
/// "bit-identical to sequential chunks" contract at misaligned offsets.
#[test]
#[ignore = "diagnostic: unaligned lineage matrix with DGQ_PREFILL_BATCH forced on/off"]
fn kv_lineage_unaligned_batch_bisect() {
    eprintln!("--- prefill_batch = true (production default)");
    let on = std::panic::catch_unwind(|| unaligned_matrix(Some(true))).is_ok();
    eprintln!("--- prefill_batch = false (plain 256-chunks)");
    let off = std::panic::catch_unwind(|| unaligned_matrix(Some(false))).is_ok();
    eprintln!("VERDICT: batch_on_clean={on} batch_off_clean={off}");
}

/// Sweep the two candidate axes independently: resume ALIGNMENT and delta
/// CHUNK COUNT, at totals below the sliding ring (4096) so the ring cannot
/// participate. Fast (a few seconds per case).
#[test]
#[ignore = "diagnostic: isolates resume-alignment vs delta-chunk-count, ring excluded"]
fn kv_lineage_alignment_vs_chunkcount_sweep() {
    let Some(dir) = model_dir() else {
        return;
    };
    let mut fl = crate::flags::RuntimeConfig::default();
    fl.kv.q8_override = Some(false);
    let _g = crate::flags::install_for_test(fl);
    let cfg = StepGenerateConfig::from_generate(
        7,
        64,
        16384,
        N_LAYERS,
        sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
    // total 2000 < ring 4096: no wrap anywhere. Resume aligned vs not,
    // delta spanning one chunk vs two.
    let cases = [
        (2000usize, 256usize),  // aligned resume 1744? no: 1744%256=208
        (2048, 256),            // resume 1792 = 7*256 ALIGNED, 1 chunk
        (2048, 512),            // resume 1536 ALIGNED, 2 chunks
        (2000, 200),            // resume 1800 misaligned, 1 chunk
        (2000, 300),            // resume 1700 misaligned, 2 chunks
        (2000, 257),            // resume 1743 misaligned, 2 chunks (just over)
        (2000, 255),            // resume 1745 misaligned, 1 chunk (just under)
    ];
    for (total, delta) in cases {
        let offset = total - delta;
        let ids = synth_ids(total);
        session.scrub_kv_for_test();
        session.extend_kv(&ids).expect("fresh");
        let fresh = session.live_kv_fingerprint();
        session.scrub_kv_for_test();
        session.extend_kv(&ids[..offset]).expect("prefix");
        session.extend_kv(&ids[offset..]).expect("delta");
        let two = session.live_kv_fingerprint();
        eprintln!(
            "sweep total={total:5} delta={delta:4} resume={offset:5} \
             (aligned={:5}, delta_chunks={}): {}",
            offset % 256 == 0,
            delta.div_ceil(256),
            if fresh == two { "OK" } else { "FORK" },
        );
    }
}

/// WHICH positions diverge, for the live-shape case. Compares the readable
/// window position-by-position between the fresh and prefix+delta arms and
/// reports contiguous differing ranges, per layer kind — the fingerprint
/// only says "something differs".
#[test]
#[ignore = "diagnostic: locates the diverging POSITIONS of the wrapped misaligned fork"]
fn kv_lineage_fork_position_map() {
    let Some(dir) = model_dir() else {
        return;
    };
    let mut fl = crate::flags::RuntimeConfig::default();
    fl.kv.q8_override = Some(false);
    let _g = crate::flags::install_for_test(fl);
    let max_seq = 16384usize;
    let cfg = StepGenerateConfig::from_generate(
        7,
        64,
        max_seq,
        N_LAYERS,
        sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
    let (total, delta) = (14775usize, 266usize);
    let offset = total - delta;
    let ids = synth_ids(total);

    // Snapshot each layer's whole region so positions can be compared by slot.
    let grab = |s: &StepGenerateSession| -> Vec<Vec<u8>> {
        use objc2_metal::MTLBuffer as _;
        let buf = s.kv_buffer_for_test();
        let layout = s.layout_for_test();
        let src =
            unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, buf.length()) };
        (0..N_LAYERS)
            .map(|i| {
                let l = &layout.layers[i];
                let slots = if l.kv_ring_mask != 0 {
                    l.kv_ring_mask as usize + 1
                } else {
                    max_seq
                };
                let bytes = crate::metal::step_kv::kv_region_bytes(
                    l.n_kv_heads,
                    l.head_dim,
                    slots,
                    crate::shaders::kv_quant::KvFormat::F16,
                ) as usize;
                let base = l.kv_region as usize;
                src[base..base + bytes].to_vec()
            })
            .collect()
    };

    session.scrub_kv_for_test();
    session.extend_kv(&ids).expect("fresh");
    let fresh = grab(&session);
    session.scrub_kv_for_test();
    session.extend_kv(&ids[..offset]).expect("prefix");
    session.extend_kv(&ids[offset..]).expect("delta");
    let two = grab(&session);

    let layout = *session.layout_for_test();
    let window = 1024usize;
    // Which layers differ at all, and by how much (K vs V, max |delta|).
    for li in 0..N_LAYERS {
        let l = &layout.layers[li];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let row_b = 2 * hd;
        let stride = 2 * nkv * row_b;
        // Ring layers: scan the WHOLE ring (every live position), not just
        // the fingerprint window — a query reads back `window` positions, so
        // divergence below the window still feeds the forward.
        let (lo, hi) = if l.kv_ring_mask != 0 {
            (total.saturating_sub(l.kv_ring_mask as usize + 1), total)
        } else {
            (0, total)
        };
        let (mut k_diff, mut v_diff, mut max_abs) = (0usize, 0usize, 0.0f32);
        let mut first_diff = None;
        for p in lo..hi {
            let slot = if l.kv_ring_mask != 0 {
                p & l.kv_ring_mask as usize
            } else {
                p
            };
            for row in 0..2 * nkv {
                let o = slot * stride + row * row_b;
                if fresh[li][o..o + row_b] != two[li][o..o + row_b] {
                    if row < nkv {
                        k_diff += 1;
                    } else {
                        v_diff += 1;
                    }
                    first_diff.get_or_insert(p);
                    for d in (0..row_b).step_by(2) {
                        let a = crate::shaders::f16::f16_bits_to_f32(u16::from_le_bytes([
                            fresh[li][o + d],
                            fresh[li][o + d + 1],
                        ]));
                        let b = crate::shaders::f16::f16_bits_to_f32(u16::from_le_bytes([
                            two[li][o + d],
                            two[li][o + d + 1],
                        ]));
                        max_abs = max_abs.max((a - b).abs());
                    }
                }
            }
        }
        if k_diff + v_diff > 0 {
            eprintln!(
                "fork-layers layer{li:2} ({}, scanned [{lo},{hi})): K rows differ={k_diff} \
                 V rows differ={v_diff} first_diff_pos={first_diff:?} max|delta|={max_abs:.6}",
                if l.kv_ring_mask != 0 { "ring" } else { "linear" },
            );
        }
    }
    for li in [0usize, 1, 5, 29] {
        let l = &layout.layers[li];
        let stride = 2 * l.n_kv_heads as usize * (2 * l.head_dim as usize);
        // Positions the fingerprint actually reads.
        let (lo, hi) = if l.kv_ring_mask != 0 {
            (total.saturating_sub(window), total)
        } else {
            (0, total)
        };
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for p in lo..hi {
            let slot = if l.kv_ring_mask != 0 {
                p & l.kv_ring_mask as usize
            } else {
                p
            };
            let o = slot * stride;
            let differs = fresh[li][o..o + stride] != two[li][o..o + stride];
            match (differs, runs.last_mut()) {
                (true, Some(r)) if r.1 == p => r.1 = p + 1,
                (true, _) => runs.push((p, p + 1)),
                _ => {}
            }
        }
        eprintln!(
            "fork-map layer{li:2} ({}, window=[{lo},{hi})): {} differing run(s) {:?}",
            if l.kv_ring_mask != 0 { "ring" } else { "linear" },
            runs.len(),
            &runs[..runs.len().min(6)],
        );
    }
    eprintln!(
        "reference points: resume={offset} (chunk1 {offset}..{}), chunk2 {}..{total}, \
         window starts {}, ring slot of resume {}",
        offset + 256,
        offset + 256,
        total - window,
        offset % 4096,
    );
}

/// Which attention path carries the phase dependence? Runs the live-shape
/// case under each attention lever. A config that comes back OK is the
/// culprit's off-switch; all-FORK means the phase dependence is not in
/// attention at all (look at the MoE/GEMM stages next).
#[test]
#[ignore = "diagnostic: bisects the lineage fork across attention levers"]
fn kv_lineage_fork_attention_bisect() {
    let Some(dir) = model_dir() else {
        return;
    };
    // (label, mutator)
    type Tweak = (&'static str, fn(&mut crate::flags::RuntimeConfig));
    let configs: &[Tweak] = &[
        ("baseline (production defaults)", |_| {}),
        ("gemm_attn=0 (E17 off, full layers)", |f| {
            f.perf.gemm_attn = false;
        }),
        ("attn_topk=0 (E20 off)", |f| {
            f.perf.attn_topk = false;
        }),
        ("attn_mma=0 (scalar sliding attn)", |f| {
            f.perf.attn_mma = false;
        }),
        ("attn_mma_full=0", |f| {
            f.perf.attn_mma_full = false;
        }),
        ("all attention levers off", |f| {
            f.perf.gemm_attn = false;
            f.perf.attn_topk = false;
            f.perf.attn_mma = false;
            f.perf.attn_mma_full = false;
        }),
    ];
    let (total, delta) = (14775usize, 266usize);
    let offset = total - delta;
    let ids = synth_ids(total);
    for (label, tweak) in configs {
        let mut fl = crate::flags::RuntimeConfig::default();
        fl.kv.q8_override = Some(false);
        tweak(&mut fl);
        let _g = crate::flags::install_for_test(fl);
        let cfg = StepGenerateConfig::from_generate(
            7,
            64,
            16384,
            N_LAYERS,
            sample::sampler_for_steps(48, false),
            false,
        );
        let (mut s, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
        s.scrub_kv_for_test();
        s.extend_kv(&ids).expect("fresh");
        let fresh = s.live_kv_fingerprint();
        s.scrub_kv_for_test();
        s.extend_kv(&ids[..offset]).expect("prefix");
        s.extend_kv(&ids[offset..]).expect("delta");
        let two = s.live_kv_fingerprint();
        eprintln!(
            "attn-bisect {:38}: {}",
            label,
            if fresh == two { "OK   <-- suspect" } else { "FORK" }
        );
    }
}

fn unaligned_matrix(batch: Option<bool>) {
    let Some(dir) = model_dir() else {
        return;
    };
    let mut fl = crate::flags::RuntimeConfig::default();
    fl.kv.q8_override = Some(false); // f16: what production actually runs
    if let Some(b) = batch {
        fl.perf.prefill_batch = b;
    }
    let _g = crate::flags::install_for_test(fl);
    let max_seq = 16384usize;
    let cfg = StepGenerateConfig::from_generate(
        7,
        64,
        max_seq,
        N_LAYERS,
        sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
    let ring = 4096usize;

    // (total, delta, why)
    let cases = [
        (6400usize, 173usize, "misaligned resume, no wrap crossing"),
        (8300, 300, "delta chunk STRADDLES the ring wrap"),
        (14775, 266, "the live turn's exact shape (resume 14,509)"),
    ];
    // Report the whole matrix, then assert — a per-case panic hides the
    // cells after it, and which cells fail IS the diagnosis.
    let mut failures = Vec::new();
    for (total, delta, why) in cases {
        let offset = total - delta;
        let ids = synth_ids(total);
        session.scrub_kv_for_test();
        session.extend_kv(&ids).expect("fresh prefill");
        let fresh = session.live_kv_fingerprint();
        session.scrub_kv_for_test();
        session.extend_kv(&ids[..offset]).expect("prefix prefill");
        session.extend_kv(&ids[offset..]).expect("delta extend");
        let two_path = session.live_kv_fingerprint();
        let straddles = (offset % ring) + delta.min(256) > ring;
        let verdict = if fresh == two_path { "OK  " } else { "FORK" };
        eprintln!(
            "lineage total={total:6} delta={delta:4} resume={offset:6} \
             (256-residue {:3}, ring {:4}->{:4}, straddles_wrap={straddles:5}): {verdict} — {why}",
            offset % 256,
            offset % ring,
            (offset + delta) % ring,
        );
        if fresh != two_path {
            failures.push(format!("total={total} delta={delta} resume={offset} ({why})"));
        }
    }
    assert!(
        failures.is_empty(),
        "delta-extend KV diverges from fresh prefill: {failures:?}"
    );
}

/// Same matrix on Q8 KV.
///
/// KNOWN RED — but NOT a ring/requant seam (forensics 2026-07-18, see
/// `q8_ring_wrap_divergence_probe` + PLAN "q8 fast-prefill is broken
/// wholesale"): the q8 fast-prefill FORWARD goes NaN after layer 0 —
/// layer-0 Q/K/V land real in cache and arena, yet the attention output
/// plane comes back non-finite (scalar AND mma2), so every later layer's
/// K/V quantizes a NaN hidden stream into all-`-127` rows.
/// Below-wrap "passed" because both arms
/// produced IDENTICAL garbage; at wrap, chunk grouping changes which
/// bands survive ring overwrites, so fingerprints fork. f16 is fully
/// healthy (path- and chunking-independent, every row written). The
/// production auto-q8 threshold is max_seq 178,176 on this 36 GB box
/// (measured: `q8_auto_threshold_crossover`), so live serve (ctx 100k)
/// always ran f16 — the live reuse-vs-fresh fork is NOT explained by
/// this. Un-ignore when the q8 read path is fixed and harness-covered.
#[test]
#[ignore = "KNOWN RED: q8 fast-prefill forward dies after layer 0 (PLAN correctness debt)"]
fn kv_lineage_paths_are_fingerprint_identical_q8() {
    lineage_matrix(true);
}

/// DIAGNOSTIC (run explicitly): the forensics battery that reframed the q8
/// ring-wrap RED (2026-07-18). Findings, in the order the arms establish
/// them:
///   1. Scrubbed builds are deterministic AND path-independent (fresh ==
///      two-path); the original "divergence" needed cross-build residue.
///   2. Chunking (super vs plain) changes q8 stored bytes — but only
///      because different bands of DEAD rows survive ring overwrites.
///   3. Most q8 rows are all-`-127` codes — the q8 signature of a NaN
///      source (`fmax(NaN,lo)=lo`, so `clamp(NaN,-127,127)` = -127), NOT a
///      quantized zero. Layer 0 writes real K/V; every later layer writes
///      NaN-derived garbage, because the forward's hidden stream goes
///      non-finite after layer 0 (plane trace: attnq real 11.9, then
///      attn_out/hidden/ffg/dense/moein all `non_finite=true`), on BOTH
///      the scalar and mma2 attention kernels.
///   4. The f16 control writes every row in every layer — the hole is
///      q8-only.
/// Next step lives in PLAN: an isolated q8 attention harness case
/// (currently ZERO q8 coverage) to pin the read-side defect, then fix and
/// un-ignore the q8 lineage matrix.
#[test]
#[ignore = "diagnostic: q8 fast-prefill forensics battery (see PLAN q8 item)"]
fn q8_ring_wrap_divergence_probe() {
    let Some(dir) = model_dir() else {
        return;
    };
    let max_seq = 16384usize;
    let total = 6400usize; // wraps the 4096-slot sliding ring
    let ids = synth_ids(total);
    let cfg = || {
        StepGenerateConfig::from_generate(
            7,
            64,
            max_seq,
            N_LAYERS,
            sample::sampler_for_steps(48, false),
            false,
        )
    };

    /// Per-layer classification of the first `n` slots' K and V rows.
    /// A q8 row is DEAD when every code is -127 (the NaN signature); it is
    /// UNWRITTEN when the row is still all zero bytes.
    fn classify(
        session: &StepGenerateSession,
        n: usize,
        layers: &[usize],
        label: &str,
        q8: bool,
    ) {
        use objc2_metal::MTLBuffer as _;
        let layout = *session.layout_for_test();
        let buf = session.kv_buffer_for_test();
        let src =
            unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, buf.length()) };
        let fmt = if q8 {
            crate::shaders::kv_quant::KvFormat::Q8
        } else {
            crate::shaders::kv_quant::KvFormat::F16
        };
        for &li in layers {
            let l = &layout.layers[li];
            let hd = l.head_dim as usize;
            let nkv = l.n_kv_heads as usize;
            let row_b = crate::metal::step_kv::kv_row_bytes(l.head_dim, fmt) as usize;
            let stride = 2 * nkv * row_b;
            let base = l.kv_region as usize;
            let (mut live, mut dead, mut unwritten) = (0, 0, 0);
            for slot in 0..n {
                // Row 0 is the first K row; row `nkv` the first V row.
                for row in [0usize, nkv] {
                    let o = base + slot * stride + row * row_b;
                    let body = &src[o..o + if q8 { hd } else { 32 }];
                    if body.iter().all(|&b| b == 0) {
                        unwritten += 1;
                    } else if q8 && body.iter().all(|&b| b as i8 == -127) {
                        dead += 1;
                    } else {
                        live += 1;
                    }
                }
            }
            eprintln!("{label} layer{li:2}: live={live} dead(NaN)={dead} unwritten={unwritten}");
        }
    }

    // ---- ARM 1: the crisp repro. A single 256-token extend on a SCRUBBED
    // q8 session — no ring wrap, no super-chunks, no residue. Layer 0 lands
    // real K/V; every later layer is all -127 = quantized NaN.
    {
        let mut fl = crate::flags::RuntimeConfig::default();
        fl.kv.q8_override = Some(true);
        let _g = crate::flags::install_for_test(fl);
        let (mut s, _) = StepGenerateSession::open(&dir, &cfg(), None).expect("q8 session");
        s.scrub_kv_for_test();
        s.extend_kv(&ids[..256]).expect("minimal 256");
        classify(&s, 256, &[0, 1, 4, 6, 10], "q8 minimal-256", true);

        // ARM 1b: the SAME shape with the scalar attention kernel — the
        // defect is not specific to mma2.
        drop(s);
        let mut fl = crate::flags::RuntimeConfig::default();
        fl.kv.q8_override = Some(true);
        fl.perf.attn_mma = false;
        let _g2 = crate::flags::install_for_test(fl);
        let (mut s2, _) = StepGenerateSession::open(&dir, &cfg(), None).expect("scalar session");
        s2.scrub_kv_for_test();
        s2.extend_kv(&ids[..256]).expect("scalar 256");
        classify(&s2, 256, &[0, 1, 6], "q8 minimal-256 scalar-attn", true);
    }

    // ---- ARM 2: WHERE it goes non-finite. `layers=1` traces the planes
    // after layer 0 only. Q is real (max_abs ~11.9) but attn_out and every
    // downstream plane come back non_finite -> the NaN is born in / just
    // after layer 0's attention, reading a q8 cache.
    for nlayers in [1usize, 2] {
        let mut fl = crate::flags::RuntimeConfig::default();
        fl.kv.q8_override = Some(true);
        fl.debug.trace_ranges = true;
        let _g = crate::flags::install_for_test(fl);
        let c = StepGenerateConfig::from_generate(
            7,
            64,
            max_seq,
            nlayers,
            sample::sampler_for_steps(48, false),
            false,
        );
        let (mut s, _) = StepGenerateSession::open(&dir, &c, None).expect("layers session");
        s.scrub_kv_for_test();
        eprintln!("q8 plane-ranges after layer {}:", nlayers - 1);
        s.extend_kv(&ids[..256]).expect("range extend");
    }

    // ---- ARM 3: the f16 CONTROL. Same build, f16 KV: every row in every
    // layer is written and live. The hole is q8-only.
    {
        let mut fl = crate::flags::RuntimeConfig::default();
        fl.kv.q8_override = Some(false);
        let _g = crate::flags::install_for_test(fl);
        let (mut s, _) = StepGenerateSession::open(&dir, &cfg(), None).expect("f16 session");
        s.scrub_kv_for_test();
        s.extend_kv(&ids[..256]).expect("f16 256");
        classify(&s, 256, &[0, 1, 4, 6, 10], "f16 control minimal-256", false);
    }

    // ---- ARM 4: why the ORIGINAL "ring-wrap path-dependence" RED was an
    // artifact. On a SCRUBBED session, fresh and prefix+delta builds are
    // byte-identical even at a wrapped ring; the RED only appears when the
    // second build inherits the first's residue over rows this build never
    // writes (all the dead ones). Chunking then decides which dead bands
    // survive ring overwrites, which is what made it look chunking- and
    // wrap-dependent. LESSON: scrub before comparing KV across builds.
    {
        let mut fl = crate::flags::RuntimeConfig::default();
        fl.kv.q8_override = Some(true);
        let _g = crate::flags::install_for_test(fl);
        let (mut s, _) = StepGenerateSession::open(&dir, &cfg(), None).expect("q8 session");
        let fp = |s: &mut StepGenerateSession, scrub: bool, build: &dyn Fn(&mut StepGenerateSession)| {
            if scrub {
                s.scrub_kv_for_test();
            } else {
                s.reset_kv();
            }
            build(s);
            s.live_kv_fingerprint()
        };
        let fresh = |s: &mut StepGenerateSession| {
            s.extend_kv(&ids).expect("fresh");
        };
        let two_path = |s: &mut StepGenerateSession| {
            s.extend_kv(&ids[..total - 256]).expect("prefix");
            s.extend_kv(&ids[total - 256..]).expect("delta");
        };
        let scrubbed_fresh = fp(&mut s, true, &fresh);
        let scrubbed_two = fp(&mut s, true, &two_path);
        let dirty_two = fp(&mut s, false, &two_path);
        eprintln!(
            "q8 wrapped-ring @{total}: scrubbed fresh=={:x} two-path=={:x} (equal: {}) | \
             UNSCRUBBED two-path=={:x} (equal to scrubbed: {})",
            scrubbed_fresh,
            scrubbed_two,
            scrubbed_fresh == scrubbed_two,
            dirty_two,
            dirty_two == scrubbed_two,
        );
    }
}

/// The q8 EXPOSURE boundary on this machine: `kv_format` auto-selects q8
/// only when the f16-resident estimate exceeds 85% of the device working-set
/// cap. Prints the cap and the crossover `max_seq`, so the "has production
/// ever run q8?" question is answered by measurement, not arithmetic in a
/// doc comment. (Fast — device only, no model load.)
#[test]
#[ignore = "diagnostic: prints this device's q8 auto-selection crossover"]
fn q8_auto_threshold_crossover() {
    use objc2_metal::MTLDevice;
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    let cap = device.recommendedMaxWorkingSetSize();
    crate::flags::set_gpu_working_set_cap(cap);
    let _g = crate::flags::install_for_test(crate::flags::RuntimeConfig::default());
    let mut crossover = None;
    for n in (1024..1_000_000).step_by(1024) {
        if crate::flags::kv_format(n) == crate::shaders::kv_quant::KvFormat::Q8 {
            crossover = Some(n);
            break;
        }
    }
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    eprintln!(
        "q8 auto-selection: working-set cap {:.2} GiB; q8 engages at max_seq >= {crossover:?} \
         (ctx 100000 -> {:?})",
        cap as f64 / GIB,
        crate::flags::kv_format(100_000),
    );
}

fn lineage_matrix(force_q8: bool) {
    let Some(dir) = model_dir() else {
        return;
    };
    let mut fl = crate::flags::RuntimeConfig::default();
    fl.kv.q8_override = Some(force_q8);
    let _g = crate::flags::install_for_test(fl);
    let max_seq = 16384usize;
    let sampler = sample::sampler_for_steps(48, false);
    let cfg = StepGenerateConfig::from_generate(7, 64, max_seq, N_LAYERS, sampler, false);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");

    // (total, delta): 2048 stays below the 4096-slot sliding ring (the
    // offset-resume-proven regime); 6400 wraps it — the live-divergence
    // regime the golden gate never composes with delta extends.
    for (total, delta) in [(2048usize, 256usize), (6400, 256), (6400, 1024)] {
        let ids = synth_ids(total);
        let fresh = fingerprint(&mut session, |s| {
            s.extend_kv(&ids).expect("fresh prefill");
        });
        let two_path = fingerprint(&mut session, |s| {
            s.extend_kv(&ids[..total - delta]).expect("prefix prefill");
            s.extend_kv(&ids[total - delta..]).expect("delta extend");
        });
        assert_eq!(
            fresh, two_path,
            "delta-extend KV diverges from fresh prefill at total={total} delta={delta}"
        );
    }

    // Truncate shape (finalize thought-strip): overshoot past the target,
    // O(1)-truncate back (512 < the 2817-slot slack), compare to a fresh
    // build of the kept prefix.
    let total = 6400usize;
    let over = 512usize;
    let ids = synth_ids(total + over);
    let fresh = fingerprint(&mut session, |s| {
        s.extend_kv(&ids[..total]).expect("fresh prefill");
    });
    let truncated = fingerprint(&mut session, |s| {
        s.extend_kv(&ids).expect("overshoot prefill");
        s.truncate_kv_to(total).expect("truncate");
    });
    assert_eq!(
        fresh, truncated,
        "truncate KV diverges from fresh prefill of the kept prefix"
    );

    // Full finalize shape: overshoot, truncate, then extend a DIFFERENT
    // tail (the canonical thought-free continuation).
    let tail = synth_ids(total + over + 256);
    let tail = &tail[total + over..];
    let fresh_with_tail = fingerprint(&mut session, |s| {
        s.extend_kv(&ids[..total]).expect("fresh prefill");
        s.extend_kv(tail).expect("tail extend");
    });
    let finalize_shape = fingerprint(&mut session, |s| {
        s.extend_kv(&ids).expect("overshoot prefill");
        s.truncate_kv_to(total).expect("truncate");
        s.extend_kv(tail).expect("tail extend");
    });
    assert_eq!(
        fresh_with_tail, finalize_shape,
        "truncate+re-extend (finalize shape) KV diverges from fresh prefill"
    );
}
