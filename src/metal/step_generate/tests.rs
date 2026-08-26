//! Tests for `tests`, extracted from step_generate.rs (backlog item 3).

use super::progress::condense_step_text;
use super::session::kv_truncate_needs_ring_rebuild;
use super::*;
use crate::metal::step_kernel::{
    CANVAS, StepRuntime, VOCAB, init_canvas_state_from_rng, logits_finite_check_enabled,
};
use crate::sample::{Rng, initialize_canvas};

/// Byte length of layer 0's live KV region — the leading slice of a
/// `snapshot_kv` blob (`gather_kv_prefix` concatenates layers in order).
fn layer0_live_bytes(session: &StepGenerateSession, kv_len: usize, max_seq: usize) -> usize {
    let l = &session.layout_for_test().layers[0];
    let ring = l.kv_ring_mask as usize + 1;
    assert!(l.kv_ring_mask != 0, "layer 0 must be a sliding/ring layer");
    let slots = kv_len.min(ring);
    crate::metal::step_kv::kv_region_bytes(
        l.n_kv_heads,
        l.head_dim,
        slots,
        crate::flags::kv_format(max_seq),
    ) as usize
}

/// PREMISE CONTROL for `truncate_after_uncommitted_canvas_write_matches_fresh_prefill`.
///
/// Layer 0's K/V is a pure function of `embed(tokens)` (projections + RoPE +
/// norms — no attention dependence), so identical prefills must produce
/// bit-identical layer-0 KV. The oracle compares layer 0 only and this is why
/// that is sound.
///
/// It is ALSO only layer 0 that is sound to compare: layers 1..29 are NOT
/// reproducible across identical `reset_kv` + `extend_kv` cycles at 1200 tokens
/// (measured: ~80% of bytes differ, and differ again on a third run, so it is
/// not a stale-buffer function). Layer 0 clean + layer 1 dirty localizes that to
/// layer 0's attention/MoE OUTPUT, amplified through depth. This is
/// a separate, pre-existing bug (this control touches neither `truncate_kv_to`
/// nor `rollback_to`). Widen this assert to the whole blob once that lands.
#[test]
fn layer0_prefill_kv_is_bit_reproducible() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let sampler = crate::sample::sampler_for_steps(24, false);
    let cfg = StepGenerateConfig::from_generate(7, 64, 4096, layers, sampler, false);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    let ids: Vec<u32> = (0..1200u32).map(|i| 1000 + (i * 7919) % 30000).collect();

    session.reset_kv();
    session.extend_kv(&ids).unwrap();
    let a = session.snapshot_kv();
    session.reset_kv();
    session.extend_kv(&ids).unwrap();
    let b = session.snapshot_kv();
    let l0 = layer0_live_bytes(&session, 1200, 4096);
    assert_eq!(
        a.kv_bytes[..l0],
        b.kv_bytes[..l0],
        "layer-0 prefill KV must be bit-reproducible (it has no attention dependence)"
    );
}

/// DIAGNOSTIC: is prefill reproducible across identical
/// `reset_kv` + `extend_kv` cycles? Reports per-layer A-vs-B and B-vs-C byte
/// diffs. B-vs-C is the discriminator: if A!=B but B==C, prefill is a
/// deterministic function of stale buffer residue; if B!=C it is genuinely
/// nondeterministic.
///
/// Env: `DGQ_PROBE_N` = prompt tokens (default 1200). Bisect by running under
/// `DGQ_FLASH_PREFILL=0`, or at N below/above the M=1024 super-chunk threshold.
///
/// Run: `cargo test --release prefill_nondeterminism_probe -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic: cargo test --release prefill_nondeterminism_probe -- --ignored --nocapture"]
fn prefill_nondeterminism_probe() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    const MAX_SEQ: usize = 4096;
    let n: usize = std::env::var("DGQ_PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1200);
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let sampler = crate::sample::sampler_for_steps(24, false);
    let cfg = StepGenerateConfig::from_generate(7, 64, MAX_SEQ, layers, sampler, false);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    let ids: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 30000).collect();

    let mut snaps = Vec::new();
    for _ in 0..3 {
        session.reset_kv();
        session.extend_kv(&ids).unwrap();
        snaps.push(session.snapshot_kv());
    }
    let (a, b, c) = (&snaps[0], &snaps[1], &snaps[2]);
    let count = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p != q).count();
    eprintln!(
        "\n=== prefill probe: n={n} flash={} (DGQ_FLASH_PREFILL) ===",
        crate::flags::flash_prefill().0
    );
    eprintln!(
        "TOTAL  A-vs-B={:>9}  B-vs-C={:>9}  of {}",
        count(&a.kv_bytes, &b.kv_bytes),
        count(&b.kv_bytes, &c.kv_bytes),
        a.kv_bytes.len()
    );
    let fmt = crate::flags::kv_format(MAX_SEQ);
    let layout = session.layout_for_test();
    let mut off = 0usize;
    for i in 0..crate::metal::step_kernel::N_LAYERS {
        let l = &layout.layers[i];
        let slots = n.min(if l.kv_ring_mask != 0 {
            l.kv_ring_mask as usize + 1
        } else {
            (MAX_SEQ + 8).next_multiple_of(8)
        });
        let bytes =
            crate::metal::step_kv::kv_region_bytes(l.n_kv_heads, l.head_dim, slots, fmt) as usize;
        let per_slot = bytes / slots;
        let (la, lb, lc) = (
            &a.kv_bytes[off..off + bytes],
            &b.kv_bytes[off..off + bytes],
            &c.kv_bytes[off..off + bytes],
        );
        eprintln!(
            "layer {i:>2} {} hd={:<4} A-vs-B={:>9} B-vs-C={:>9} /{bytes:<9} first_diff_slot={:?}",
            if l.kv_ring_mask != 0 {
                "sliding"
            } else {
                "full   "
            },
            l.head_dim,
            count(la, lb),
            count(lb, lc),
            la.iter()
                .zip(lb)
                .position(|(x, y)| x != y)
                .map(|p| p / per_slot),
        );
        off += bytes;
    }
}

/// THE RING-TRUNCATE ORACLE.
///
/// INVARIANT: after `truncate_kv_to(n)`, the live KV must equal what a fresh
/// prefill of the same `n` tokens produces. `snapshot_kv` gathers exactly each
/// layer's live slots (`gather_kv_prefix`: `min(kv_len, layer_slots)` physical
/// slots, and for a ring layer `slot = pos & mask` makes that the live window),
/// so a byte compare of two snapshots at the same length is the whole invariant.
///
/// THE REPRO IS THE ORDINARY PRODUCTION FLOW, which is what made this bug so
/// easy to miss. The **final** answer block is never committed to causal KV, but
/// denoise still writes its canvas at `[kv_len, kv_len+CANVAS)` unconditionally
/// (`kv_write_end = u32::MAX`). So after a short reply at `kv_len=2000`, slots
/// 0..=207 hold positions 2048..=2255 while `kv_valid_tokens` still reads 2000 —
/// under the ring size. The initial predicate tested `old_len > ring`, called
/// that safe, and `finalize`'s truncate then handed the next turn a window with
/// 31 poisoned positions in it.
///
/// SCOPE: asserts on LAYER 0 only. Layer 0 is a sliding/ring layer, so it is
/// poisoned by exactly the same mechanism as every other sliding layer, and its
/// KV is attention-independent hence bit-reproducible — see
/// `layer0_prefill_kv_is_bit_reproducible`. Layers 1..29 cannot be byte-compared
/// today; widen this when that capability is available.
///
/// This test FAILS on the pre-fix predicate (it takes the O(1) clamp path and
/// slots 177..=207 differ) and passes on the corrected one.
#[test]
fn truncate_after_uncommitted_canvas_write_matches_fresh_prefill() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    // max_seq > 2048 so the sliding ring is capped at 2048 slots and can wrap.
    const MAX_SEQ: usize = 4096;
    const PROMPT: usize = 2000; // + CANVAS = 2256 > 2048 ring => canvas wraps
    const KEEP: usize = 1200; // window [177, 1200] reaches the poisoned slots

    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let sampler = crate::sample::sampler_for_steps(24, false);
    // One block only: max_new_tokens <= CANVAS keeps the reply in the final
    // (uncommitted) block, which is the precondition the bug needs.
    let cfg = StepGenerateConfig::from_generate(7, 64, MAX_SEQ, layers, sampler, false);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();

    let ids: Vec<u32> = (0..PROMPT as u32)
        .map(|i| 1000 + (i * 7919) % 30000)
        .collect();

    session.extend_kv(&ids).unwrap();
    assert_eq!(session.kv_valid_tokens().len(), PROMPT);
    let _ = generate_with_session(&mut session, &ids, &cfg, "ring-truncate-oracle").unwrap();
    // Precondition: the final block must NOT have committed, or `old_len` would
    // exceed the ring and even the pre-fix predicate would rebuild. If this ever
    // trips, the repro has drifted — fix the test rather than deleting it.
    assert_eq!(
        session.kv_valid_tokens().len(),
        PROMPT,
        "repro precondition: the final answer block must not commit to causal KV"
    );

    session.truncate_kv_to(KEEP).unwrap();
    let after_truncate = session.snapshot_kv();
    assert_eq!(after_truncate.tokens.len(), KEEP);

    // Oracle: the same KEEP tokens, prefilled into a KV that never saw a wrap.
    session.reset_kv();
    session.extend_kv(&ids[..KEEP]).unwrap();
    let fresh = session.snapshot_kv();

    assert_eq!(
        after_truncate.kv_bytes.len(),
        fresh.kv_bytes.len(),
        "snapshot geometry must match at equal kv_len"
    );

    let l0 = layer0_live_bytes(&session, KEEP, MAX_SEQ);
    let (got, want) = (&after_truncate.kv_bytes[..l0], &fresh.kv_bytes[..l0]);
    if got != want {
        let per_slot = l0 / KEEP;
        let bad: Vec<usize> = (0..KEEP)
            .filter(|s| {
                got[s * per_slot..(s + 1) * per_slot] != want[s * per_slot..(s + 1) * per_slot]
            })
            .collect();
        panic!(
            "truncate_kv_to({KEEP}) after an uncommitted canvas write past the ring left \
             {} of {KEEP} layer-0 ring slots differing from a fresh prefill (slots {:?}..={:?}) \
             — poisoned slots survived into the live window",
            bad.len(),
            bad.first(),
            bad.last()
        );
    }
}

/// THE CROSS-TURN CONTAMINATION ORACLE (the foreign-draft-probe leak).
///
/// An ABANDONED turn (`begin_turn` with no `commit_block` and no
/// `finish_turn`, the shape of every propose-only experiment) leaves the
/// prompt's KV resident. When only `finish_turn` recorded `kv_valid_tokens`,
/// that state read as an empty log with a primed KV, and the next
/// `begin_turn`'s session-open-prefill exemption (empty log and `kv_len >=
/// n_prompt`, no token verification) adopted the resident KV as this
/// prompt's prefill. The second turn then denoised against the FIRST
/// prompt's context: the foreign-draft probe's from-noise arms answered
/// prompt 1 on prompt 2. `DGQ_KV_REUSE=0` did not protect, since the
/// exemption was checked independently of the reuse flag.
///
/// The prompts share no common prefix, so the correct second `begin_turn` is
/// a full re-prefill, and the oracle is a fresh prefill of the second
/// prompt. Layer 0 only, bit-identical by
/// `layer0_prefill_kv_is_bit_reproducible` (and `prefill_chunks` is
/// `prefill_chunks_from(0, ..)`, so both sides use the same chunk geometry).
#[test]
fn begin_turn_after_abandoned_turn_prefills_the_new_prompt() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    const MAX_SEQ: usize = 4096;
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let sampler = crate::sample::sampler_for_steps(24, false);
    let cfg = StepGenerateConfig::from_generate(7, 64, MAX_SEQ, layers, sampler, false);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();

    // p2 shorter than p1 (the buggy exemption also required `kv_len >=
    // n_prompt`), first tokens differ (zero common prefix).
    let p1: Vec<u32> = (0..500u32).map(|i| 1000 + (i * 7919) % 30000).collect();
    let p2: Vec<u32> = (0..300u32).map(|i| 2000 + (i * 104_729) % 30000).collect();
    assert_ne!(p1[0], p2[0]);

    let _abandoned = begin_turn(&mut session, &p1, &cfg, "abandoned").unwrap();
    let _victim = begin_turn(&mut session, &p2, &cfg, "victim").unwrap();
    let got = session.rt.snapshot_kv(p2.len());

    session.reset_kv();
    session.extend_kv(&p2).unwrap();
    let want = session.rt.snapshot_kv(p2.len());

    assert_eq!(got.len(), want.len(), "snapshot geometry must match");
    let l0 = layer0_live_bytes(&session, p2.len(), MAX_SEQ);
    if got[..l0] != want[..l0] {
        let per_slot = l0 / p2.len();
        let bad = (0..p2.len())
            .filter(|s| {
                got[s * per_slot..(s + 1) * per_slot] != want[s * per_slot..(s + 1) * per_slot]
            })
            .count();
        panic!(
            "begin_turn after an abandoned turn reused the previous prompt's KV: \
             {bad} of {} layer-0 slots differ from a fresh prefill of the new prompt",
            p2.len()
        );
    }
}

#[test]
fn p21_denoise_readback_under_1mb() {
    let bytes =
        StepRuntime::denoise_step_host_readback_bytes(false, &crate::metal::ModelDims::reference());
    assert!(
        bytes <= 1024 * 1024,
        "hot-path readback {bytes} B exceeds 1 MiB"
    );
    assert_eq!(bytes, (StepRuntime::CANVAS_STATE_BYTES * 2) as u64);
    if logits_finite_check_enabled() {
        let with_check = StepRuntime::denoise_step_host_readback_bytes(
            true,
            &crate::metal::ModelDims::reference(),
        );
        assert!(with_check <= 1024 * 1024);
    }
}

#[test]
fn block_reset_uses_fresh_canvas() {
    let mut rng = Rng::new(42);
    let a = initialize_canvas(CANVAS, VOCAB, &mut rng);
    let b = initialize_canvas(CANVAS, VOCAB, &mut rng);
    assert_ne!(a, b);
    let mut r = Rng::new(99);
    let st = init_canvas_state_from_rng(VOCAB, &mut r);
    // ids array is PREFILL_M-sized (batched prefill); the canvas uses [0..CANVAS).
    assert_eq!(st.ids.len(), crate::metal::PREFILL_M);
    assert!(st.ids[..CANVAS].iter().any(|&v| v != 0));
}

/// The rebuild predicate, checked against an independent model of when the ring
/// is actually corrupt rather than against a restatement of its own formula.
///
/// `ring_is_corrupt` below is derived from the storage rule only (slot =
/// `p & (ring-1)`, so the ring holds the last `ring` WRITTEN positions; denoise
/// writes CANVAS past `kv_len` whether or not the block commits; a query at `q`
/// reads down to `q - (window-1)`). It shares no code with the predicate, so
/// agreement is evidence and not tautology.
#[test]
fn kv_truncate_needs_ring_rebuild_matches_corruption_model() {
    const RING: usize = 2048;
    const WINDOW: usize = 1024;

    fn ring_is_corrupt(old: usize, new: usize) -> bool {
        if new >= old {
            return false;
        }
        let highest_written = old + crate::metal::CANVAS - 1;
        let oldest_live = (highest_written + 1).saturating_sub(RING);
        let deepest_needed = new.saturating_sub(WINDOW - 1);
        deepest_needed < oldest_live
    }

    for old in 0..6000usize {
        for new in [
            0,
            1,
            100,
            458,
            1023,
            1024,
            1200,
            old / 2,
            old.saturating_sub(1),
            old,
        ] {
            if new > old {
                continue;
            }
            let got = kv_truncate_needs_ring_rebuild(old, new, Some(RING), WINDOW);
            assert_eq!(
                got,
                ring_is_corrupt(old, new),
                "predicate disagrees with the corruption model at old={old} new={new}"
            );
        }
    }
}

#[test]
fn kv_truncate_needs_ring_rebuild_policy() {
    const W: usize = 1024;
    // No ring (all-linear) — O(1) truncate is always safe.
    assert!(!kv_truncate_needs_ring_rebuild(3000, 100, None, W));
    // Never wrapped (even counting the canvas overshoot) — early slots are intact.
    assert!(!kv_truncate_needs_ring_rebuild(1000, 100, Some(2048), W));
    assert!(!kv_truncate_needs_ring_rebuild(1792, 1023, Some(2048), W));
    // REGRESSION (the bug 632aa69 missed): `old_len` is the highest COMMITTED
    // position, but denoise wrote its canvas at [old_len, old_len+CANVAS) even
    // though the block never committed — so the ring HAS wrapped and slots
    // 0..=(old_len+255-2048) hold post-wrap K/V. The old predicate tested
    // `old_len > ring` and called all three of these safe.
    assert!(kv_truncate_needs_ring_rebuild(2048, 100, Some(2048), W));
    assert!(kv_truncate_needs_ring_rebuild(2000, 1200, Some(2048), W));
    assert!(kv_truncate_needs_ring_rebuild(1900, 1000, Some(2048), W));
    // Past wrap, shortening — must rebuild (the serve-finalize alpha-soup case).
    assert!(kv_truncate_needs_ring_rebuild(2049, 100, Some(2048), W));
    assert!(kv_truncate_needs_ring_rebuild(4096, 458, Some(2048), W));
    // Shallow truncation at long context: the kept window is entirely inside the
    // live ring, so this is safe at ANY kv. The old predicate rebuilt here —
    // a full re-prefill of the whole conversation to rewind one token.
    // Exact edge: safe iff old-new <= ring - CANVAS - (window-1) = 769.
    assert!(!kv_truncate_needs_ring_rebuild(30000, 29999, Some(2048), W));
    assert!(!kv_truncate_needs_ring_rebuild(30000, 29231, Some(2048), W));
    // ...one token deeper crosses into overwritten slots.
    assert!(kv_truncate_needs_ring_rebuild(30000, 29230, Some(2048), W));
    // A ring that covers the whole sequence (max_seq <= ring, or
    // DGQ_KV_RING_UNCAPPED) provably never wraps.
    assert!(!kv_truncate_needs_ring_rebuild(4000, 10, Some(8192), W));
    // No shortening — nothing to rebuild.
    assert!(!kv_truncate_needs_ring_rebuild(3000, 3000, Some(2048), W));
    assert!(!kv_truncate_needs_ring_rebuild(100, 200, Some(2048), W));
}

#[test]
fn condense_step_text_transforms() {
    // Whitespace runs (incl. newlines) collapse to one space; edges trimmed.
    assert_eq!(condense_step_text("a   b\n\n  c  ", 80), "a b c");
    // All but the LAST <eos> are dropped — interior churn and the tail run.
    assert_eq!(
        condense_step_text("A<eos>B <eos><eos><eos>", 80),
        "AB <eos>"
    );
    // The harmony ceremony case from real logs: newline + eos run.
    assert_eq!(
        condense_step_text("<|channel>thought\n<channel|>12<turn|><eos><eos>", 80),
        "<|channel>thought <channel|>12<turn|><eos>"
    );
    // Short text passes through untouched (no clip marker).
    assert_eq!(condense_step_text("short", 80), "short");
    // Long text middle-clips: head + marker + tail, tail gets the larger share.
    let long: String = (0..200)
        .map(|i| char::from(b'a' + (i % 26) as u8))
        .collect();
    let out = condense_step_text(&long, 80);
    assert!(
        out.contains("<... [120] chars clipped>"),
        "marker missing: {out}"
    );
    assert!(out.starts_with(&long[..32]), "head missing: {out}");
    assert!(out.ends_with(&long[200 - 48..]), "tail missing: {out}");
}
