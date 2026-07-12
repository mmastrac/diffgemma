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
