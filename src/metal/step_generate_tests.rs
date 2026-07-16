//! Tests for `tests`, extracted from step_generate.rs (backlog item 3).

use super::*;
use crate::metal::step_kernel::{
    StepRuntime, init_canvas_state_from_rng, logits_finite_check_enabled,
};
use crate::sample::initialize_canvas;

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

#[test]
fn kv_truncate_needs_ring_rebuild_policy() {
    // No ring (all-linear) — O(1) truncate is always safe.
    assert!(!kv_truncate_needs_ring_rebuild(3000, 100, None));
    // Below wrap — early slots were never overwritten.
    assert!(!kv_truncate_needs_ring_rebuild(1000, 100, Some(2048)));
    assert!(!kv_truncate_needs_ring_rebuild(2048, 100, Some(2048)));
    // Past wrap, shortening — must rebuild (the serve-finalize alpha-soup case).
    assert!(kv_truncate_needs_ring_rebuild(2049, 100, Some(2048)));
    assert!(kv_truncate_needs_ring_rebuild(4096, 458, Some(2048)));
    // No shortening — nothing to rebuild.
    assert!(!kv_truncate_needs_ring_rebuild(3000, 3000, Some(2048)));
    assert!(!kv_truncate_needs_ring_rebuild(100, 200, Some(2048)));
}
