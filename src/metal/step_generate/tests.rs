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

#[test]
fn p21_denoise_readback_under_1mb() {
    let bytes = StepRuntime::denoise_step_host_readback_bytes(false);
    assert!(
        bytes <= 1024 * 1024,
        "hot-path readback {bytes} B exceeds 1 MiB"
    );
    assert_eq!(bytes, (StepRuntime::CANVAS_STATE_BYTES * 2) as u64);
    if logits_finite_check_enabled() {
        let with_check = StepRuntime::denoise_step_host_readback_bytes(true);
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

/// EXPERIMENT: does seeding the canvas with a converged answer skip the
/// denoise schedule?
///
/// Per (prompt, seed): arm A runs the full schedule from noise
/// (`no_early_stop`) and its final argmax becomes arm B's initial canvas via
/// `initial_canvas_ids`, against the same uncommitted KV. Arm C re-denoises
/// from fresh noise as the negative control. B and C share one early-stop
/// config, so the initial canvas is the only variable.
///
/// Caveat: arm B's first step has no previous-step self-conditioning probs
/// (ARCHITECTURE.md divergence #2): the seed reaches the model only through
/// the canvas embedding, and non-accepted rows are re-noised every step.
///
/// Env: `DGQ_PROBE_STEPS` = schedule length (default 48).
///
/// Run: `cargo test --release rewound_canvas_probe -- --ignored --nocapture`
#[test]
#[ignore = "experiment: cargo test --release rewound_canvas_probe -- --ignored --nocapture"]
fn rewound_canvas_probe() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let steps: usize = std::env::var("DGQ_PROBE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Write a haiku about the ocean.",
        "Explain why the sky is blue in two sentences.",
    ];
    let seeds = [7u64, 1234];

    let mut cfg = StepGenerateConfig::from_generate(
        seeds[0],
        CANVAS,
        4096,
        layers,
        crate::sample::sampler_for_steps(steps, true),
        true,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();

    let propose = |session: &mut StepGenerateSession,
                   cfg: &StepGenerateConfig,
                   ts: &mut TurnState|
     -> Option<Box<ProposedBlock>> {
        match propose_block(session, cfg, ts).unwrap() {
            BlockOutcome::Proposal(pb) => Some(pb),
            _ => None,
        }
    };

    eprintln!("\n=== rewound-canvas probe: schedule={steps} ===");
    eprintln!(
        "{:<44} {:>4}  {:>7} {:>10}  {:>7} {:>10}  {:>6} {:>5}",
        "prompt/seed", "A", "B_steps", "B_stop", "C_steps", "C_stop", "B==A%", "drift"
    );
    for prompt in prompts {
        let ids = crate::chat_template::format_chat_token_ids(
            &tokenizer,
            &[crate::chat_template::ChatTurn::user(prompt)],
            &crate::chat_template::ChatFormatOptions::default(),
        )
        .unwrap();
        for seed in seeds {
            // Arm A: full schedule from noise; its argmax is the seed canvas.
            cfg.seed = seed;
            cfg.sampler = crate::sample::sampler_for_steps(steps, true);
            cfg.no_early_stop = true;
            cfg.initial_canvas_ids = None;
            let mut ts = begin_turn(&mut session, &ids, &cfg, prompt).unwrap();
            let Some(pa) = propose(&mut session, &cfg, &mut ts) else {
                eprintln!("{prompt:.40}/{seed}: arm A yielded no proposal, skipping");
                continue;
            };
            if pa.token_ids.len() != CANVAS {
                eprintln!(
                    "{prompt:.40}/{seed}: arm A canvas={}, skipping",
                    pa.token_ids.len()
                );
                continue;
            }

            // Arm B: same KV, early stop on, canvas pinned to A's output.
            cfg.sampler = crate::sample::sampler_for_steps(steps, false);
            cfg.no_early_stop = false;
            cfg.initial_canvas_ids = Some(pa.token_ids.clone());
            let pb = propose(&mut session, &cfg, &mut ts);

            // Arm C: identical config from fresh noise (negative control).
            cfg.initial_canvas_ids = None;
            let pc = propose(&mut session, &cfg, &mut ts);

            let (b_steps, b_stop, b_match, b_ment) = pb
                .as_ref()
                .map(|p| {
                    let m = p
                        .token_ids
                        .iter()
                        .zip(&pa.token_ids)
                        .filter(|(x, y)| x == y)
                        .count();
                    (
                        p.stats.steps_eff.to_string(),
                        p.stats.denoise_stop.clone(),
                        format!("{:.1}", 100.0 * m as f64 / CANVAS as f64),
                        p.stats.mean_ent_per_step.clone(),
                    )
                })
                .unwrap_or(("-".into(), "-".into(), "-".into(), Vec::new()));
            let (c_steps, c_stop) = pc
                .as_ref()
                .map(|p| (p.stats.steps_eff.to_string(), p.stats.denoise_stop.clone()))
                .unwrap_or(("-".into(), "-".into()));
            let label = format!("{:.38}/{seed}", prompt);
            eprintln!(
                "{label:<44} {:>4}  {b_steps:>7} {b_stop:>10}  {c_steps:>7} {c_stop:>10}  {b_match:>6} {}",
                pa.stats.steps_eff,
                if b_match == "100.0" { "no" } else { "yes" }
            );
            if !b_ment.is_empty() {
                let head: Vec<String> = b_ment.iter().take(6).map(|e| format!("{e:.3}")).collect();
                eprintln!("    B mean_ent/step: [{}]", head.join(", "));
            }
        }
    }
}

/// EXPERIMENT: how does convergence degrade with seed fidelity?
///
/// Follow-up to `rewound_canvas_probe` (which measured the perfect-seed
/// ceiling: 2 steps, zero drift). Per (prompt, seed), arm A runs the full
/// schedule from noise; then for each fidelity x the canvas is pinned to A's
/// output with a random (1-x) fraction of positions replaced by uniform
/// noise, and the early-stop config re-denoises against the same uncommitted
/// KV. x=0 is the all-noise negative control. This maps the regime an
/// imperfect sidecar drafter would live in.
///
/// Run: `cargo test --release degraded_seed_sweep -- --ignored --nocapture`
#[test]
#[ignore = "experiment: cargo test --release degraded_seed_sweep -- --ignored --nocapture"]
fn degraded_seed_sweep() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let steps: usize = std::env::var("DGQ_PROBE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    // `DGQ_PROBE_SET=codegen` swaps in prompts whose answers fill most of the
    // canvas and whose noise baseline sits well above the 2-step floor.
    let prompts = if std::env::var("DGQ_PROBE_SET").as_deref() == Ok("codegen") {
        [
            "Write a Python function is_prime(n) that returns True if n is prime, using trial division up to sqrt(n). Only code, no explanation.",
            "Write a Python function that reverses a singly linked list given its head node. Only code, no explanation.",
            "Write a Rust function fib(n: u64) -> u64 that computes the nth Fibonacci number iteratively. Only code, no explanation.",
        ]
    } else {
        [
            "What is the capital of France? Answer in one sentence.",
            "Write a haiku about the ocean.",
            "Explain why the sky is blue in two sentences.",
        ]
    };
    let seeds = [7u64, 1234];
    let fidelities = [1.0f32, 0.95, 0.9, 0.75, 0.5, 0.25, 0.0];

    let mut cfg = StepGenerateConfig::from_generate(
        seeds[0],
        CANVAS,
        4096,
        layers,
        crate::sample::sampler_for_steps(steps, true),
        true,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();

    eprintln!("\n=== degraded-seed sweep: schedule={steps} fidelities={fidelities:?} ===");
    eprintln!(
        "{:<44} {:>4}  {}",
        "prompt/seed",
        "A",
        fidelities
            .map(|f| format!("{:>14}", format!("x={:.2}", f)))
            .join(" ")
    );
    // Per-fidelity accumulators for the summary: (steps_sum, match_sum, runs).
    let mut agg = vec![(0u64, 0f64, 0u32); fidelities.len()];
    for prompt in prompts {
        let ids = crate::chat_template::format_chat_token_ids(
            &tokenizer,
            &[crate::chat_template::ChatTurn::user(prompt)],
            &crate::chat_template::ChatFormatOptions::default(),
        )
        .unwrap();
        for seed in seeds {
            cfg.seed = seed;
            cfg.sampler = crate::sample::sampler_for_steps(steps, true);
            cfg.no_early_stop = true;
            cfg.initial_canvas_ids = None;
            let mut ts = begin_turn(&mut session, &ids, &cfg, prompt).unwrap();
            let BlockOutcome::Proposal(pa) = propose_block(&mut session, &cfg, &mut ts).unwrap()
            else {
                eprintln!("{prompt:.40}/{seed}: arm A yielded no proposal, skipping");
                continue;
            };
            if pa.token_ids.len() != CANVAS {
                eprintln!(
                    "{prompt:.40}/{seed}: arm A canvas={}, skipping",
                    pa.token_ids.len()
                );
                continue;
            }

            cfg.sampler = crate::sample::sampler_for_steps(steps, false);
            cfg.no_early_stop = false;
            let mut cells = Vec::new();
            for (fi, &fid) in fidelities.iter().enumerate() {
                // Deterministic corruption: an independent LCG per (seed, fid)
                // picks which positions revert to uniform noise.
                let mut rng = Rng::new(seed ^ ((fi as u64 + 1) << 32));
                let seeded: Vec<u32> = pa
                    .token_ids
                    .iter()
                    .map(|&t| {
                        if rng.next_f32() < fid {
                            t
                        } else {
                            rng.uniform_below(VOCAB as u32)
                        }
                    })
                    .collect();
                cfg.initial_canvas_ids = Some(seeded);
                let BlockOutcome::Proposal(pb) =
                    propose_block(&mut session, &cfg, &mut ts).unwrap()
                else {
                    cells.push(format!("{:>14}", "-"));
                    continue;
                };
                let m = pb
                    .token_ids
                    .iter()
                    .zip(&pa.token_ids)
                    .filter(|(x, y)| x == y)
                    .count();
                let match_pct = 100.0 * m as f64 / CANVAS as f64;
                agg[fi].0 += u64::from(pb.stats.steps_eff);
                agg[fi].1 += match_pct;
                agg[fi].2 += 1;
                cells.push(format!(
                    "{:>14}",
                    format!(
                        "{}/{}/{:.0}%",
                        pb.stats.steps_eff,
                        &pb.stats.denoise_stop[..4.min(pb.stats.denoise_stop.len())],
                        match_pct
                    )
                ));
            }
            eprintln!(
                "{:<44} {:>4}  {}",
                format!("{:.38}/{seed}", prompt),
                pa.stats.steps_eff,
                cells.join(" ")
            );
        }
    }
    eprintln!("\nsummary (mean over runs): cell = steps / output match vs arm A");
    for (fi, &fid) in fidelities.iter().enumerate() {
        let (s, m, n) = agg[fi];
        if n > 0 {
            eprintln!(
                "  x={fid:.2}: steps={:.1} match={:.1}% (n={n})",
                s as f64 / f64::from(n),
                m / f64::from(n)
            );
        }
    }
}

/// EXPERIMENT: does the denoiser ratify a FOREIGN draft or fight it?
///
/// The degraded-seed sweeps corrupted the target's own answer, so every seed
/// was near the target's manifold. A sidecar draft is different: fluent text
/// from another model (different style, names, indentation). This probe seeds
/// the canvas with drafts written by an off-the-shelf small model and
/// measures steps-to-stop, whether the output keeps the draft's content, and
/// whether it drifts back to the target's own from-noise answer.
///
/// Env: `DGQ_DRAFT_FILE` = JSON `[{"prompt": ..., "draft": ...}]` (see
/// scratchpad gen_drafts.py). Two pad variants per draft: eos-fill (the shape
/// of a converged block) and noise-fill.
///
/// Run: `DGQ_DRAFT_FILE=... cargo test --release foreign_draft_probe -- --ignored --nocapture`
#[test]
#[ignore = "experiment: DGQ_DRAFT_FILE=drafts.json cargo test --release foreign_draft_probe -- --ignored --nocapture"]
fn foreign_draft_probe() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let Ok(draft_file) = std::env::var("DGQ_DRAFT_FILE") else {
        eprintln!("foreign_draft_probe: set DGQ_DRAFT_FILE");
        return;
    };
    #[derive(serde::Deserialize)]
    struct Entry {
        prompt: String,
        draft: String,
    }
    let entries: Vec<Entry> =
        serde_json::from_str(&std::fs::read_to_string(&draft_file).unwrap()).unwrap();
    let steps: usize = std::env::var("DGQ_PROBE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let seeds = [7u64, 1234];

    let mut cfg = StepGenerateConfig::from_generate(
        seeds[0],
        CANVAS,
        4096,
        layers,
        crate::sample::sampler_for_steps(steps, true),
        true,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    let eos = session.rt.read_params().eos_token_id;

    let snippet = |ids: &[u32]| -> String {
        let text = tokenizer.decode(ids);
        let t: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        t.chars().take(72).collect()
    };

    eprintln!("\n=== foreign-draft probe: schedule={steps} eos={eos} file={draft_file} ===");
    for entry in &entries {
        let ids = crate::chat_template::format_chat_token_ids(
            &tokenizer,
            &[crate::chat_template::ChatTurn::user(&entry.prompt)],
            &crate::chat_template::ChatFormatOptions::default(),
        )
        .unwrap();
        let mut draft_ids = tokenizer.encode(&entry.draft, false);
        draft_ids.truncate(CANVAS);
        let d_len = draft_ids.len();
        eprintln!("\n--- {:.60} (draft {d_len} tokens)", entry.prompt);
        eprintln!("    draft:  {}", snippet(&draft_ids));
        for seed in seeds {
            cfg.seed = seed;
            cfg.sampler = crate::sample::sampler_for_steps(steps, true);
            cfg.no_early_stop = true;
            cfg.initial_canvas_ids = None;
            let mut ts = begin_turn(&mut session, &ids, &cfg, &entry.prompt).unwrap();
            let BlockOutcome::Proposal(pa) = propose_block(&mut session, &cfg, &mut ts).unwrap()
            else {
                eprintln!("    seed {seed}: no arm-A proposal, skipping");
                continue;
            };
            eprintln!(
                "    A/{seed}: steps={} out: {}",
                pa.stats.steps_eff,
                snippet(&pa.token_ids)
            );

            cfg.sampler = crate::sample::sampler_for_steps(steps, false);
            cfg.no_early_stop = false;
            let mut noise_rng = Rng::new(seed.wrapping_mul(0x9e3779b9));
            let arms: [(&str, Option<Vec<u32>>); 3] = [
                ("D_eos", {
                    let mut c = draft_ids.clone();
                    c.resize(CANVAS, eos);
                    Some(c)
                }),
                ("D_noise", {
                    let mut c = draft_ids.clone();
                    while c.len() < CANVAS {
                        c.push(noise_rng.uniform_below(VOCAB as u32));
                    }
                    Some(c)
                }),
                ("C_noise", None),
            ];
            for (name, canvas) in arms {
                cfg.initial_canvas_ids = canvas;
                let BlockOutcome::Proposal(pb) =
                    propose_block(&mut session, &cfg, &mut ts).unwrap()
                else {
                    eprintln!("    {name}/{seed}: no proposal");
                    continue;
                };
                let vs_a = pb
                    .token_ids
                    .iter()
                    .zip(&pa.token_ids)
                    .filter(|(x, y)| x == y)
                    .count();
                let vs_draft = pb
                    .token_ids
                    .iter()
                    .zip(&draft_ids)
                    .filter(|(x, y)| x == y)
                    .count();
                eprintln!(
                    "    {name}/{seed}: steps={} stop={} vs_A={:.0}% vs_draft={:.0}% out: {}",
                    pb.stats.steps_eff,
                    pb.stats.denoise_stop,
                    100.0 * vs_a as f64 / CANVAS as f64,
                    100.0 * vs_draft as f64 / d_len.max(1) as f64,
                    snippet(&pb.token_ids)
                );
            }
        }
    }
}

/// MTP head hookup probe: the native (CPU f32) Gemma 4 draft head against
/// (A) the HF-captured oracle states, checking implementation parity with the
/// transformers head, and (B) the ENGINE's own states: prefill KV decoded
/// from `snapshot_kv`, layer-29 hiddens from the arena, embed rows from the
/// .dgq. Phase B is the real drafts-from-our-engine measurement.
///
/// Env: `DGQ_MTP_HEAD` = assistant snapshot dir (model.safetensors inside);
/// `DGQ_MTP_ORACLE` = dir with oracle_meta.json / oracle_*.bin /
/// drift_results.json from the python drift test. Phase B needs the model
/// and runs with the default bf16 arena (`DGQ_PREFILL_F16` must be off).
///
/// Run: DGQ_MTP_HEAD=... DGQ_MTP_ORACLE=... cargo test --release mtp_draft_probe -- --ignored --nocapture
#[test]
#[ignore = "experiment: DGQ_MTP_HEAD=<snapshot> DGQ_MTP_ORACLE=<dir> cargo test --release mtp_draft_probe -- --ignored --nocapture"]
fn mtp_draft_probe() {
    use crate::mtp_head::{BackboneKv, MtpHead, draft_tokens};
    let Ok(head_dir) = std::env::var("DGQ_MTP_HEAD") else {
        eprintln!("mtp_draft_probe: set DGQ_MTP_HEAD");
        return;
    };
    let Ok(oracle_dir) = std::env::var("DGQ_MTP_ORACLE") else {
        eprintln!("mtp_draft_probe: set DGQ_MTP_ORACLE");
        return;
    };
    let head = MtpHead::load(&std::path::Path::new(&head_dir).join("model.safetensors")).unwrap();
    eprintln!("head loaded: vocab={}", head.vocab);

    let read_bin = |name: &str| -> Vec<f32> {
        let bytes = std::fs::read(format!("{oracle_dir}/{name}")).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{oracle_dir}/oracle_meta.json")).unwrap(),
    )
    .unwrap();
    let seq: Vec<u32> = meta["seq"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let ans_start = meta["ans_start"].as_u64().unwrap() as usize;
    let s = seq.len();
    let k_draft = 8usize;
    let py: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(format!("{oracle_dir}/drift_results.json")).unwrap(),
    )
    .unwrap();

    let oracle_h29 = read_bin("oracle_h29.bin");
    let oracle_kv = BackboneKv {
        k_swa: read_bin("oracle_k_swa.bin"),
        v_swa: read_bin("oracle_v_swa.bin"),
        k_full: read_bin("oracle_k_full.bin"),
        v_full: read_bin("oracle_v_full.bin"),
        seq: s,
    };
    let emb_rows = read_bin("oracle_emb_rows.bin");
    let row_tokens: Vec<u32> = meta["row_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let oracle_row = |tok: u32| -> Option<Vec<f32>> {
        let i = row_tokens.iter().position(|&t| t == tok)?;
        Some(emb_rows[i * 2816..(i + 1) * 2816].to_vec())
    };

    // Phase A: oracle states in, drafts out; must reproduce the python head.
    let positions: Vec<usize> = (ans_start - 1..s - k_draft - 1).collect();
    let mut first_match = 0usize;
    let mut full_match = 0usize;
    for (pi, &p) in positions.iter().enumerate() {
        let drafted = draft_tokens(
            &head,
            &oracle_kv,
            p,
            &oracle_h29[p * 2816..(p + 1) * 2816],
            seq[p],
            k_draft,
            &mut |tok| oracle_row(tok),
        );
        let py_drafted: Vec<u32> = py[pi]["drafted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if drafted.first() == py_drafted.first() {
            first_match += 1;
        }
        if drafted == py_drafted {
            full_match += 1;
        }
    }
    eprintln!(
        "phase A (HF-state parity): first-token match {}/{}  full-draft match {}/{}",
        first_match,
        positions.len(),
        full_match,
        positions.len()
    );

    // Phase B: the same drafting from the engine's own states.
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        eprintln!("phase B skipped: no model dir");
        return;
    };
    const MAX_SEQ: usize = 4096;
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let sampler = crate::sample::sampler_for_steps(2, false);
    let cfg = StepGenerateConfig::from_generate(7, 64, MAX_SEQ, layers, sampler, false);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    session.reset_kv();
    assert!(
        s <= 1024,
        "hidden rows only survive for the last prefill chunk"
    );
    session.rt.set_prefill_hidden_capture(s);
    session.extend_kv(&seq).unwrap();
    let h29_all = session
        .rt
        .take_prefill_hidden()
        .expect("prefill hidden capture");
    assert_eq!(h29_all.len(), s * 2816);

    let mut h29_eng = vec![0.0f32; s * 2816];
    let mut cos_sum = 0.0f64;
    let mut cos_min = f64::MAX;
    let mut bad_rows = 0usize;
    for p in 0..s {
        let row = &h29_all[p * 2816..(p + 1) * 2816];
        let nonfinite = row.iter().filter(|v| !v.is_finite()).count();
        let o = &oracle_h29[p * 2816..(p + 1) * 2816];
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..2816 {
            d += row[i] as f64 * o[i] as f64;
            na += (row[i] as f64).powi(2);
            nb += (o[i] as f64).powi(2);
        }
        let cos = d / (na.sqrt() * nb.sqrt());
        if p % 50 == 0 || !cos.is_finite() && bad_rows < 3 {
            eprintln!(
                "  row {p}: cos={cos:.4} nonfinite={nonfinite} |row|={:.1} |oracle|={:.1} row[:4]={:?}",
                na.sqrt(),
                nb.sqrt(),
                &row[..4]
            );
        }
        if cos.is_finite() {
            cos_sum += cos;
            cos_min = cos_min.min(cos);
        } else {
            bad_rows += 1;
        }
        h29_eng[p * 2816..(p + 1) * 2816].copy_from_slice(row);
    }
    eprintln!(
        "phase B hidden tap: cosine(engine, HF bf16) mean={:.4} min={:.4} bad_rows={bad_rows}",
        cos_sum / (s - bad_rows).max(1) as f64,
        cos_min
    );

    // Decode layers 28/29 K/V out of the snapshot blob into [n_kv, seq, hd].
    let snap = session.snapshot_kv();
    let layout = session.layout_for_test();
    let fmt = crate::flags::kv_format(MAX_SEQ);
    let decode_layer = |target: usize| -> (Vec<f32>, Vec<f32>) {
        let mut off = 0usize;
        for i in 0..crate::metal::step_kernel::N_LAYERS {
            let l = &layout.layers[i];
            let cap = if l.kv_ring_mask != 0 {
                l.kv_ring_mask as usize + 1
            } else {
                (MAX_SEQ + 8).next_multiple_of(8)
            };
            let slots = s.min(cap);
            let bytes = crate::metal::step_kv::kv_region_bytes(l.n_kv_heads, l.head_dim, slots, fmt)
                as usize;
            if i == target {
                let (n_kv, hd) = (l.n_kv_heads as usize, l.head_dim as usize);
                let row_bytes = hd * 2;
                let slot_stride = 2 * n_kv * row_bytes;
                let read_row = |slot: usize, r: usize| -> Vec<f32> {
                    let base = off + slot * slot_stride + r * row_bytes;
                    snap.kv_bytes[base..base + row_bytes]
                        .chunks_exact(2)
                        .map(|c| {
                            crate::shaders::f16::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))
                        })
                        .collect()
                };
                let mut k = vec![0.0f32; n_kv * s * hd];
                let mut v = vec![0.0f32; n_kv * s * hd];
                for pos in 0..s {
                    let slot = if l.kv_ring_mask != 0 {
                        pos & l.kv_ring_mask as usize
                    } else {
                        pos
                    };
                    for hh in 0..n_kv {
                        let kr = read_row(slot, hh);
                        let vr = read_row(slot, n_kv + hh);
                        k[hh * s * hd + pos * hd..][..hd].copy_from_slice(&kr);
                        v[hh * s * hd + pos * hd..][..hd].copy_from_slice(&vr);
                    }
                }
                return (k, v);
            }
            off += bytes;
        }
        unreachable!("layer {target} not reached");
    };
    let (k_swa, v_swa) = decode_layer(28);
    let (k_full, v_full) = decode_layer(29);
    let engine_kv = BackboneKv {
        k_swa,
        v_swa,
        k_full,
        v_full,
        seq: s,
    };

    // Raw bf16 embed rows straight out of the .dgq blob.
    let store = crate::dgq::DgqStore::open(&dir).unwrap();
    let emb_bytes = store
        .tensor_bytes("model.decoder.embed_tokens.weight")
        .unwrap();
    assert_eq!(
        emb_bytes.len(),
        262144 * 2816 * 2,
        "expected raw bf16 embed"
    );
    let dgq_row = |tok: u32| -> Option<Vec<f32>> {
        let base = tok as usize * 2816 * 2;
        Some(
            emb_bytes[base..base + 2816 * 2]
                .chunks_exact(2)
                .map(|c| crate::shaders::cpu::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        )
    };

    let accepted = |drafted: &[u32], p: usize| -> usize {
        let truth = &seq[p + 1..p + 1 + k_draft];
        drafted
            .iter()
            .zip(truth)
            .take_while(|(d, t)| d == t)
            .count()
    };
    let mut acc_sum = 0usize;
    let mut acc_first = 0usize;
    let mut acc_ge4 = 0usize;
    let mut py_first = 0usize;
    let mut cpu_drafts = Vec::with_capacity(positions.len());
    let mut cpu_tokens = 0usize;
    let cpu_started = std::time::Instant::now();
    for (pi, &p) in positions.iter().enumerate() {
        let drafted = draft_tokens(
            &head,
            &engine_kv,
            p,
            &h29_eng[p * 2816..(p + 1) * 2816],
            seq[p],
            k_draft,
            &mut |tok| dgq_row(tok),
        );
        cpu_tokens += drafted.len();
        let a = accepted(&drafted, p);
        acc_sum += a;
        acc_first += usize::from(a >= 1);
        acc_ge4 += usize::from(a >= 4);
        let py_tok = py[pi]["drafted"]
            .as_array()
            .unwrap()
            .first()
            .map(|v| v.as_u64().unwrap() as u32);
        if drafted.first().copied() == py_tok {
            py_first += 1;
        }
        cpu_drafts.push(drafted);
    }
    let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0 / cpu_tokens.max(1) as f64;
    let n = positions.len();
    eprintln!(
        "phase B (engine states): mean accepted {:.2}/8  first-token {:.1}%  >=4 {:.1}%  first-token-agrees-with-python {:.1}%  cpu {:.1}ms/token",
        acc_sum as f64 / n as f64,
        100.0 * acc_first as f64 / n as f64,
        100.0 * acc_ge4 as f64 / n as f64,
        100.0 * py_first as f64 / n as f64,
        cpu_ms,
    );

    // Phase C: the GPU head on the same engine states, checked against the
    // CPU head (bf16-weight accumulation order may drift deep drafts).
    let mut gpu = crate::mtp_head_gpu::MtpHeadGpu::load(
        &std::path::Path::new(&head_dir).join("model.safetensors"),
    )
    .unwrap();
    gpu.upload_kv(&engine_kv).unwrap();
    let mut first_eq_cpu = 0usize;
    let mut full_eq_cpu = 0usize;
    let mut gpu_acc_sum = 0usize;
    let mut gpu_tokens = 0usize;
    let gpu_started = std::time::Instant::now();
    for (pi, &p) in positions.iter().enumerate() {
        let drafted = gpu
            .draft_tokens(
                p,
                &h29_eng[p * 2816..(p + 1) * 2816],
                seq[p],
                k_draft,
                None,
                &mut |tok| dgq_row(tok),
            )
            .unwrap();
        gpu_tokens += drafted.len();
        gpu_acc_sum += accepted(&drafted, p);
        first_eq_cpu += usize::from(drafted.first() == cpu_drafts[pi].first());
        full_eq_cpu += usize::from(drafted == cpu_drafts[pi]);
    }
    let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0 / gpu_tokens.max(1) as f64;
    eprintln!(
        "phase C (gpu head): mean accepted {:.2}/8  first-token-eq-cpu {}/{}  full-draft-eq-cpu {}/{}  gpu {:.2}ms/token",
        gpu_acc_sum as f64 / n as f64,
        first_eq_cpu,
        n,
        full_eq_cpu,
        n,
        gpu_ms,
    );
}

/// MTP training-data dry run: generate a reply per prompt with the 26B, then
/// re-prefill [prompt + reply] with the hidden capture armed and write one
/// record per sequence (tokens, all-position layer-29 hiddens, layers-28/29
/// K/V decoded to f32). Output feeds the python overfit trainer; the full
/// corpus dump graduates to a subcommand once this loop is proven.
///
/// Env: `DGQ_MTP_DUMP_DIR` = output dir; `DGQ_MTP_DUMP_N` caps prompts.
///
/// Run: DGQ_MTP_DUMP_DIR=... cargo test --release mtp_dump_dry_run -- --ignored --nocapture
#[test]
#[ignore = "experiment: DGQ_MTP_DUMP_DIR=<dir> cargo test --release mtp_dump_dry_run -- --ignored --nocapture"]
fn mtp_dump_dry_run() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let Ok(out_dir) = std::env::var("DGQ_MTP_DUMP_DIR") else {
        eprintln!("mtp_dump_dry_run: set DGQ_MTP_DUMP_DIR");
        return;
    };
    std::fs::create_dir_all(&out_dir).unwrap();
    let langs = ["Python", "Rust", "JavaScript"];
    let tasks = [
        "checks if a string is a palindrome",
        "computes the factorial of n iteratively",
        "returns the maximum value in a list without using builtins",
        "counts vowels in a string",
        "merges two sorted lists into one sorted list",
        "computes the greatest common divisor of two integers",
        "reverses the words in a sentence",
        "returns the n-th triangular number",
        "removes duplicate values from a list preserving order",
        "converts a temperature between celsius and fahrenheit",
    ];
    let mut prompts: Vec<String> = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let lang = langs[i % langs.len()];
        prompts.push(format!(
            "Write a {lang} function that {t}. Only code, no explanation."
        ));
    }
    for topic in [
        "why binary search needs a sorted input",
        "the difference between a stack and a queue",
        "what a hash collision is",
        "why floating point addition is not associative",
        "what tail recursion is",
        "the two's complement representation of negative integers",
        "what a race condition is",
        "why caching improves latency",
        "what big-O notation measures",
        "the difference between TCP and UDP",
    ] {
        prompts.push(format!("Explain {topic} in two or three sentences."));
    }
    for q in [
        "What is the capital of Japan?",
        "How many bits are in a byte?",
        "What year did the first moon landing happen?",
        "What does CPU stand for?",
        "Name the largest planet in the solar system.",
    ] {
        prompts.push(format!("{q} Answer in one sentence."));
    }
    let cap: usize = std::env::var("DGQ_MTP_DUMP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(prompts.len());
    prompts.truncate(cap);

    const MAX_SEQ: usize = 4096;
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let mut cfg = StepGenerateConfig::from_generate(
        7,
        CANVAS,
        MAX_SEQ,
        layers,
        crate::sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    let eos = session.rt.read_params().eos_token_id;

    let fmt = crate::flags::kv_format(MAX_SEQ);
    let write_f32 = |path: &str, data: &[f32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    };

    let mut manifest = Vec::new();
    for (si, prompt) in prompts.iter().enumerate() {
        let prompt_ids = crate::chat_template::format_chat_token_ids(
            &tokenizer,
            &[crate::chat_template::ChatTurn::user(prompt)],
            &crate::chat_template::ChatFormatOptions::default(),
        )
        .unwrap();
        cfg.seed = 7 + si as u64;
        let out = generate_with_session(&mut session, &prompt_ids, &cfg, prompt).unwrap();
        let reply = &out.token_ids[prompt_ids.len()..];
        let reply_end = reply.iter().position(|&t| t == eos).unwrap_or(reply.len());
        let seq: Vec<u32> = prompt_ids
            .iter()
            .chain(&reply[..reply_end])
            .copied()
            .collect();
        let s = seq.len();
        if reply_end < 8 || s > 1024 {
            eprintln!("seq {si}: skipped (reply_end={reply_end}, s={s})");
            continue;
        }

        session.reset_kv();
        session.rt.set_prefill_hidden_capture(s);
        session.extend_kv(&seq).unwrap();
        let hidden = session.rt.take_prefill_hidden().expect("capture");
        assert_eq!(hidden.len(), s * 2816);

        let snap = session.snapshot_kv();
        let layout = session.layout_for_test();
        let decode_layer = |target: usize| -> (Vec<f32>, Vec<f32>) {
            let mut off = 0usize;
            for i in 0..crate::metal::step_kernel::N_LAYERS {
                let l = &layout.layers[i];
                let cap = if l.kv_ring_mask != 0 {
                    l.kv_ring_mask as usize + 1
                } else {
                    (MAX_SEQ + 8).next_multiple_of(8)
                };
                let slots = s.min(cap);
                let bytes =
                    crate::metal::step_kv::kv_region_bytes(l.n_kv_heads, l.head_dim, slots, fmt)
                        as usize;
                if i == target {
                    let (n_kv, hd) = (l.n_kv_heads as usize, l.head_dim as usize);
                    let row_bytes = hd * 2;
                    let slot_stride = 2 * n_kv * row_bytes;
                    let read_row = |slot: usize, r: usize| -> Vec<f32> {
                        let base = off + slot * slot_stride + r * row_bytes;
                        snap.kv_bytes[base..base + row_bytes]
                            .chunks_exact(2)
                            .map(|c| {
                                crate::shaders::f16::f16_bits_to_f32(u16::from_le_bytes([
                                    c[0], c[1],
                                ]))
                            })
                            .collect()
                    };
                    let mut k = vec![0.0f32; n_kv * s * hd];
                    let mut v = vec![0.0f32; n_kv * s * hd];
                    for pos in 0..s {
                        let slot = if l.kv_ring_mask != 0 {
                            pos & l.kv_ring_mask as usize
                        } else {
                            pos
                        };
                        for hh in 0..n_kv {
                            k[hh * s * hd + pos * hd..][..hd].copy_from_slice(&read_row(slot, hh));
                            v[hh * s * hd + pos * hd..][..hd]
                                .copy_from_slice(&read_row(slot, n_kv + hh));
                        }
                    }
                    return (k, v);
                }
                off += bytes;
            }
            unreachable!("layer {target} not reached");
        };
        let (k_swa, v_swa) = decode_layer(28);
        let (k_full, v_full) = decode_layer(29);

        write_f32(&format!("{out_dir}/seq{si}_hidden.bin"), &hidden);
        write_f32(&format!("{out_dir}/seq{si}_k_swa.bin"), &k_swa);
        write_f32(&format!("{out_dir}/seq{si}_v_swa.bin"), &v_swa);
        write_f32(&format!("{out_dir}/seq{si}_k_full.bin"), &k_full);
        write_f32(&format!("{out_dir}/seq{si}_v_full.bin"), &v_full);
        manifest.push(serde_json::json!({
            "id": si,
            "prompt": prompt,
            "seq": seq,
            "ans_start": prompt_ids.len(),
        }));
        eprintln!(
            "seq {si}/{}: s={s} reply={} \"{:.50}\"",
            prompts.len(),
            reply_end,
            prompt
        );
    }
    std::fs::write(
        format!("{out_dir}/manifest.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    eprintln!("wrote {} sequences to {out_dir}", manifest.len());
}

/// GPU-vs-CPU head parity on oracle states only (no engine session): fast
/// iteration for debugging checkpoint-specific GPU divergence.
///
/// Run: DGQ_MTP_HEAD=<dir> DGQ_MTP_ORACLE=<dir> cargo test --release mtp_gpu_parity_debug -- --ignored --nocapture
#[test]
#[ignore = "debug: DGQ_MTP_HEAD=<dir> DGQ_MTP_ORACLE=<dir> cargo test --release mtp_gpu_parity_debug -- --ignored --nocapture"]
fn mtp_gpu_parity_debug() {
    use crate::mtp_head::{BackboneKv, MtpHead, draft_tokens};
    let head_dir = std::env::var("DGQ_MTP_HEAD").expect("DGQ_MTP_HEAD");
    let oracle_dir = std::env::var("DGQ_MTP_ORACLE").expect("DGQ_MTP_ORACLE");
    let head_path = std::path::Path::new(&head_dir).join("model.safetensors");
    let read_bin = |name: &str| -> Vec<f32> {
        std::fs::read(format!("{oracle_dir}/{name}"))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{oracle_dir}/oracle_meta.json")).unwrap(),
    )
    .unwrap();
    let seq: Vec<u32> = meta["seq"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let s = seq.len();
    let kv = BackboneKv {
        k_swa: read_bin("oracle_k_swa.bin"),
        v_swa: read_bin("oracle_v_swa.bin"),
        k_full: read_bin("oracle_k_full.bin"),
        v_full: read_bin("oracle_v_full.bin"),
        seq: s,
    };
    let h29 = read_bin("oracle_h29.bin");
    let emb_rows = read_bin("oracle_emb_rows.bin");
    let row_tokens: Vec<u32> = meta["row_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let row = |tok: u32| -> Option<Vec<f32>> {
        let i = row_tokens.iter().position(|&t| t == tok)?;
        Some(emb_rows[i * 2816..(i + 1) * 2816].to_vec())
    };

    let cpu = MtpHead::load(&head_path).unwrap();
    let mut gpu = crate::mtp_head_gpu::MtpHeadGpu::load(&head_path).unwrap();
    gpu.upload_kv(&kv).unwrap();
    if std::env::var("DGQ_MTP_TIMING").is_ok() {
        for n in [1usize, 5, 10] {
            let ms = gpu.debug_time_chains(n).unwrap();
            eprintln!("chain x{n}: {ms:.2}ms total, {:.2}ms/chain", ms / n as f64);
        }
        for op in ["lm", "gate", "down", "qproj", "preproj", "postproj", "attend"] {
            let ms = gpu.debug_time_op(op, 20).unwrap();
            eprintln!("op {op}: {:.3}ms each", ms / 20.0);
        }
    }
    let ans_start = meta["ans_start"].as_u64().unwrap() as usize;
    for p in [ans_start - 1, ans_start + 20, ans_start + 50] {
        let c = draft_tokens(&cpu, &kv, p, &h29[p * 2816..(p + 1) * 2816], seq[p], 5, &mut |t| row(t));
        let g = gpu
            .draft_tokens(p, &h29[p * 2816..(p + 1) * 2816], seq[p], 5, None, &mut |t| {
                row(t)
            })
            .unwrap();
        eprintln!("pos {p}: cpu={c:?}\n         gpu={g:?}  match={}", c == g);
    }
}

/// HEAD-TO-HEAD: tok/s with and without MTP drafting, per prompt, one
/// session. Baseline denoises the block from noise; the drafted arm has the
/// GPU head free-run a whole answer from prompt hiddens (stopping at eos),
/// seeds the canvas with draft + eos-fill, and denoises. Decode-side wall
/// clock only (both arms share the prefill); outputs printed for the
/// quality check.
///
/// Run: DGQ_MTP_HEAD=<dir> cargo test --release mtp_head_to_head -- --ignored --nocapture
#[test]
#[ignore = "demo: DGQ_MTP_HEAD=<dir> cargo test --release mtp_head_to_head -- --ignored --nocapture"]
fn mtp_head_to_head() {
    use std::time::Instant;
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let Ok(head_dir) = std::env::var("DGQ_MTP_HEAD") else {
        eprintln!("set DGQ_MTP_HEAD");
        return;
    };
    let prompts = [
        "Write a Python function is_prime(n) that returns True if n is prime, using trial division up to sqrt(n). Only code, no explanation.",
        "Write a Python function that returns the sum of squares of the first n integers. Only code, no explanation.",
    ];
    const MAX_SEQ: usize = 4096;
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let mut cfg = StepGenerateConfig::from_generate(
        7,
        CANVAS,
        MAX_SEQ,
        layers,
        crate::sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    let eos = session.eos_token_id();
    let mut gpu = crate::mtp_head_gpu::MtpHeadGpu::load(
        &std::path::Path::new(&head_dir).join("model.safetensors"),
    )
    .unwrap();
    let store = crate::dgq::DgqStore::open(&dir).unwrap();
    let emb_bytes = store
        .tensor_bytes("model.decoder.embed_tokens.weight")
        .unwrap();
    let mut dgq_row = |tok: u32| -> Option<Vec<f32>> {
        let base = tok as usize * 2816 * 2;
        Some(
            emb_bytes[base..base + 2816 * 2]
                .chunks_exact(2)
                .map(|c| crate::shaders::cpu::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        )
    };
    let snippet = |ids: &[u32]| -> String {
        let end = ids.iter().position(|&t| t == eos).unwrap_or(ids.len());
        let text = tokenizer.decode(&ids[..end]);
        text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(160).collect()
    };

    for (pi, prompt) in prompts.iter().enumerate() {
        let prompt_ids = crate::chat_template::format_chat_token_ids(
            &tokenizer,
            &[crate::chat_template::ChatTurn::user(*prompt)],
            &crate::chat_template::ChatFormatOptions::default(),
        )
        .unwrap();
        let pl = prompt_ids.len();

        // Draft whole answer from prompt states (also leaves prompt KV warm).
        let draft_started = Instant::now();
        let (hidden, kv) = session.capture_mtp_states(&prompt_ids).unwrap();
        let capture_ms = draft_started.elapsed().as_secs_f64() * 1e3;
        gpu.upload_kv(&kv).unwrap();
        let t0 = Instant::now();
        let mut confs = Vec::new();
        let draft = gpu
            .draft_tokens_conf(
                pl - 1,
                &hidden[(pl - 1) * 2816..pl * 2816],
                prompt_ids[pl - 1],
                96,
                Some(eos),
                &mut dgq_row,
                Some(&mut confs),
            )
            .unwrap();
        let draft_ms = t0.elapsed().as_secs_f64() * 1e3;
        // Trim on the head's own confidence: cut at the first drafted token
        // whose softmax probability drops below the threshold.
        let conf_tau = 0.15f32;
        let cut = confs.iter().position(|&c| c < conf_tau).unwrap_or(draft.len());
        let profile: Vec<String> = confs.iter().take(40).map(|c| format!("{c:.2}")).collect();
        eprintln!("  conf: [{}]", profile.join(" "));
        // Fallback policy: a draft this short means the head was not
        // confident; denoise from noise instead of seeding a stub.
        let fallback = cut < 6;
        let mut canvas = draft[..cut].to_vec();
        canvas.truncate(CANVAS);
        canvas.resize(CANVAS, eos);
        // Optional alternating-noise arm (`DGQ_MTP_ALT_NOISE=1`): every second
        // position re-noised, so the denoiser re-derives half the canvas with
        // draft anchors between. Measured quality-safe but perf-neutral.
        if std::env::var("DGQ_MTP_ALT_NOISE").is_ok_and(|v| v == "1") {
            let mut rng = Rng::new(1234 + pi as u64);
            for slot in canvas.iter_mut().skip(1).step_by(2) {
                *slot = rng.uniform_below(VOCAB as u32);
            }
        }

        cfg.seed = 7 + pi as u64;
        let mut ts = begin_turn(&mut session, &prompt_ids, &cfg, prompt).unwrap();

        // Baseline: from noise.
        cfg.initial_canvas_ids = None;
        let t0 = Instant::now();
        let BlockOutcome::Proposal(pb) = propose_block(&mut session, &cfg, &mut ts).unwrap()
        else {
            continue;
        };
        let base_ms = t0.elapsed().as_secs_f64() * 1e3;
        let base_kept = pb.token_ids.iter().position(|&t| t == eos).unwrap_or(pb.token_ids.len());

        // Drafted: seeded canvas (or the fallback path from noise).
        cfg.initial_canvas_ids = if fallback { None } else { Some(canvas) };
        let t0 = Instant::now();
        let BlockOutcome::Proposal(ps) = propose_block(&mut session, &cfg, &mut ts).unwrap()
        else {
            continue;
        };
        let seed_ms = t0.elapsed().as_secs_f64() * 1e3;
        let seed_kept = ps.token_ids.iter().position(|&t| t == eos).unwrap_or(ps.token_ids.len());
        let seed_total_ms = draft_ms + seed_ms;

        eprintln!("\n=== {prompt:.60}");
        eprintln!(
            "  baseline: {:>4} steps  {base_ms:7.0}ms            {:5.1} tok/s  | {}",
            pb.stats.steps_eff,
            base_kept as f64 / (base_ms / 1e3),
            snippet(&pb.token_ids)
        );
        eprintln!(
            "  drafted:  {:>4} steps  {seed_ms:7.0}ms +{draft_ms:5.0}ms draft {:5.1} tok/s  | {}",
            ps.stats.steps_eff,
            seed_kept as f64 / (seed_total_ms / 1e3),
            snippet(&ps.token_ids)
        );
        eprintln!(
            "  draft len {} trimmed to {cut}{} (capture {capture_ms:.0}ms, shared with prefill)  speedup x{:.2}",
            draft.len(),
            if fallback { " FALLBACK" } else { "" },
            base_ms / seed_total_ms
        );
    }
}

/// Step-simulator training-data dump: per prompt, generate one block with
/// per-step canvas capture armed and write (step hiddens, step argmax) pairs
/// plus prefill hiddens and the final committed tokens. Labels for all three
/// head targets (step t -> t+1, step t -> final, prefill -> step 1) come out
/// of the same records.
///
/// Env: `DGQ_MTP_STEP_DUMP_DIR` = out dir; `DGQ_MTP_STEP_DUMP_N` caps
/// prompts (default 30, from corpus/inputs/corpus_self.jsonl).
///
/// Run: DGQ_MTP_STEP_DUMP_DIR=<dir> cargo test --release mtp_step_dump_dry -- --ignored --nocapture
#[test]
#[ignore = "experiment: DGQ_MTP_STEP_DUMP_DIR=<dir> cargo test --release mtp_step_dump_dry -- --ignored --nocapture"]
fn mtp_step_dump_dry() {
    use std::io::BufRead;
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let Ok(out_dir) = std::env::var("DGQ_MTP_STEP_DUMP_DIR") else {
        eprintln!("set DGQ_MTP_STEP_DUMP_DIR");
        return;
    };
    std::fs::create_dir_all(&out_dir).unwrap();
    let n: usize = std::env::var("DGQ_MTP_STEP_DUMP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let corpus = std::path::Path::new("corpus/inputs/corpus_self.jsonl");
    let prompts: Vec<String> = std::io::BufReader::new(std::fs::File::open(corpus).unwrap())
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| {
            serde_json::from_str::<serde_json::Value>(&l)
                .ok()
                .and_then(|v| v["prompt"].as_str().map(String::from))
        })
        .take(n)
        .collect();

    const MAX_SEQ: usize = 4096;
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    let mut cfg = StepGenerateConfig::from_generate(
        7,
        CANVAS,
        MAX_SEQ,
        layers,
        crate::sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();
    let write_f32 = |path: String, data: &[f32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    };

    let mut manifest = Vec::new();
    for (si, prompt) in prompts.iter().enumerate() {
        let final_done = std::path::Path::new(&format!("{out_dir}/blk{si}_final.json")).exists();
        let prefill_done =
            std::path::Path::new(&format!("{out_dir}/blk{si}_prefill_hidden.bin")).exists();
        if final_done && prefill_done {
            continue;
        }
        let prompt_ids = crate::chat_template::format_chat_token_ids(
            &tokenizer,
            &[crate::chat_template::ChatTurn::user(prompt)],
            &crate::chat_template::ChatFormatOptions::default(),
        )
        .unwrap();
        // Fresh capturing prefill: begin_turn's own prefill skips capture on
        // KV-reuse deltas (offset > 0) and short prompts (f32 engine path).
        // This also leaves the prompt KV resident, so begin_turn fully reuses it.
        if !prefill_done {
            let (ph, _) = session.capture_mtp_states(&prompt_ids).unwrap();
            write_f32(format!("{out_dir}/blk{si}_prefill_hidden.bin"), &ph);
        }
        if final_done {
            eprintln!("blk{si}: backfilled prefill hiddens");
            continue;
        }
        cfg.seed = 7 + si as u64;
        let mut ts = begin_turn(&mut session, &prompt_ids, &cfg, prompt).unwrap();
        session.rt.set_step_capture(true);
        let outcome = propose_block(&mut session, &cfg, &mut ts).unwrap();
        let captures = session.rt.take_step_captures();
        session.rt.set_step_capture(false);
        let BlockOutcome::Proposal(pb) = outcome else {
            eprintln!("blk{si}: no proposal, skipped");
            continue;
        };
        for (t, (hidden, argmax)) in captures.iter().enumerate() {
            write_f32(format!("{out_dir}/blk{si}_step{t}_hidden.bin"), hidden);
            let bytes: Vec<u8> = argmax.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{out_dir}/blk{si}_step{t}_argmax.bin"), bytes).unwrap();
        }
        std::fs::write(
            format!("{out_dir}/blk{si}_final.json"),
            serde_json::to_string(&serde_json::json!({
                "id": si,
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "final": pb.token_ids,
                "steps": captures.len(),
            }))
            .unwrap(),
        )
        .unwrap();
        manifest.push(si);
        eprintln!(
            "blk{si}/{}: steps={} prompt=\"{prompt:.50}\"",
            prompts.len(),
            captures.len()
        );
    }
    eprintln!("wrote {} blocks to {out_dir}", manifest.len());
}

/// Step-1 seeding probe: for each held block with a head-predicted step-1
/// canvas, run three arms with the dump's exact seed/config — noise baseline
/// (recorded in the dump's final.json), head-prediction seed, and the
/// majority-prior seed (negative control) — and compare real steps to
/// convergence plus fidelity to the baseline's committed tokens.
///
/// Env: `DGQ_STEP1_PRED_DIR` = dir with blk{si}_pred.bin + prior.bin;
/// `DGQ_MTP_STEP_DUMP_DIR` = the steps_v0 dump (prompts + baselines).
#[test]
#[ignore = "experiment: DGQ_STEP1_PRED_DIR=<dir> DGQ_MTP_STEP_DUMP_DIR=<dir> cargo test --release mtp_step1_seed_probe -- --ignored --nocapture"]
fn mtp_step1_seed_probe() {
    let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
        return;
    };
    let Ok(pred_dir) = std::env::var("DGQ_STEP1_PRED_DIR") else {
        eprintln!("set DGQ_STEP1_PRED_DIR");
        return;
    };
    let dump_dir = std::env::var("DGQ_MTP_STEP_DUMP_DIR").expect("DGQ_MTP_STEP_DUMP_DIR");
    let read_u32 = |path: &str| -> Vec<u32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let prior = read_u32(&format!("{pred_dir}/prior.bin"));
    let mut block_ids: Vec<usize> = std::fs::read_dir(&pred_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_prefix("blk")?.strip_suffix("_pred.bin")?.parse().ok()
        })
        .collect();
    block_ids.sort();

    const MAX_SEQ: usize = 4096;
    let layers = crate::commands::resolve_model_layers(&dir, None).unwrap();
    let mut cfg = StepGenerateConfig::from_generate(
        7,
        CANVAS,
        MAX_SEQ,
        layers,
        crate::sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).unwrap();

    let mut rows = Vec::new();
    for &si in &block_ids {
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{dump_dir}/blk{si}_final.json")).unwrap(),
        )
        .unwrap();
        let prompt = meta["prompt"].as_str().unwrap().to_string();
        let prompt_ids: Vec<u32> = meta["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let base_steps = meta["steps"].as_u64().unwrap() as usize;
        let base_final: Vec<u32> = meta["final"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let pred = read_u32(&format!("{pred_dir}/blk{si}_pred.bin"));

        let run_arm = |session: &mut StepGenerateSession,
                           cfg: &mut StepGenerateConfig,
                           seed_ids: Option<&[u32]>|
         -> (usize, f32) {
            cfg.seed = 7 + si as u64;
            cfg.initial_canvas_ids = seed_ids.map(|s| s.to_vec());
            let mut ts = begin_turn(session, &prompt_ids, cfg, &prompt).unwrap();
            session.rt.set_step_capture(true);
            let outcome = propose_block(session, cfg, &mut ts).unwrap();
            let steps = session.rt.take_step_captures().len();
            session.rt.set_step_capture(false);
            let BlockOutcome::Proposal(pb) = outcome else {
                return (steps, -1.0);
            };
            let n = pb.token_ids.len().min(base_final.len());
            let agree = pb
                .token_ids
                .iter()
                .zip(&base_final)
                .filter(|(a, b)| a == b)
                .count();
            (steps, agree as f32 / n.max(1) as f32)
        };
        // Oracle arm: the REAL step-1 argmax. If even this saves nothing,
        // the step-1 target is dead independent of head quality.
        let oracle = read_u32(&format!("{dump_dir}/blk{si}_step0_argmax.bin"));
        let (pred_steps, pred_agree) = run_arm(&mut session, &mut cfg, Some(&pred));
        let (prior_steps, prior_agree) = run_arm(&mut session, &mut cfg, Some(&prior));
        let (oracle_steps, oracle_agree) = run_arm(&mut session, &mut cfg, Some(&oracle));
        cfg.initial_canvas_ids = None;
        eprintln!(
            "blk{si}: baseline={base_steps} pred={pred_steps} (agree {:.1}%) prior={prior_steps} (agree {:.1}%) oracle={oracle_steps} (agree {:.1}%)",
            100.0 * pred_agree,
            100.0 * prior_agree,
            100.0 * oracle_agree
        );
        rows.push((base_steps, pred_steps, prior_steps, oracle_steps));
    }
    let sum =
        |f: fn(&(usize, usize, usize, usize)) -> usize| rows.iter().map(f).sum::<usize>() as f32;
    eprintln!(
        "mean steps: baseline {:.2}  pred-seeded {:.2}  prior-seeded {:.2}  oracle-seeded {:.2} ({} blocks)",
        sum(|r| r.0) / rows.len() as f32,
        sum(|r| r.1) / rows.len() as f32,
        sum(|r| r.2) / rows.len() as f32,
        sum(|r| r.3) / rows.len() as f32,
        rows.len()
    );
}
