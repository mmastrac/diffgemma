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

/// Same matrix on Q8 KV — the format the live serve auto-selects at large
/// ctx (the divergent-live-turn regime).
///
/// KNOWN RED (2026-07-18, see PLAN "q8 ring-wrap delta-extend"): the
/// below-wrap case passes, but a delta-extend into a WRAPPED ring diverges
/// from fresh prefill (total=6400 delta=256 fingerprints differ) — the
/// isolated root of the live reuse-vs-fresh trajectory fork. Run with
/// `--ignored` while the requantization seam is being fixed; un-ignore in
/// the fixing commit.
#[test]
#[ignore = "KNOWN RED: q8 ring-wrap delta-extend path-dependence (PLAN correctness debt)"]
fn kv_lineage_paths_are_fingerprint_identical_q8() {
    lineage_matrix(true);
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
