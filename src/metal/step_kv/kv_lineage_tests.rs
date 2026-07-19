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
/// production auto-q8 threshold is ~186k tokens on 36 GB, so live serve
/// (ctx 100k) always ran f16 — the live reuse-vs-fresh fork is NOT
/// explained by this. Un-ignore when the q8 read path is fixed and
/// harness-covered.
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
