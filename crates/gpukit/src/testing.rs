//! Oracle comparison helpers shared by every crate's tier-1 tests.
//!
//! One implementation, three consumers (the engine's `test_util`, `dgops` and
//! `dgemm` testing modules re-export these): the copies had drifted, and the
//! engine's lacked the two guards below.

/// Largest absolute element-wise difference. Panics if the lengths differ.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Cosine similarity of two equal-length vectors; 1.0 when both are all-zero.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    // f32 rounding in sqrt(na)*sqrt(na) can put an exact match a hair off 1.0.
    (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
}

/// Assert `a` matches `b` within `max_tol` and `min_cos`, and is finite.
///
/// A bit-identical pair is the strongest possible pass and skips the cosine
/// round-trip: fixtures that expect `min_cos = 1.0` must not fail because
/// `dot / (|a||b|)` rounded to 0.99999994.
pub fn assert_oracle(a: &[f32], b: &[f32], max_tol: f32, min_cos: f32) {
    let max = max_abs_diff(a, b);
    let cos = if max == 0.0 { 1.0 } else { cosine_f32(a, b) };
    assert!(
        max <= max_tol && cos >= min_cos,
        "oracle mismatch max_abs={max:.6} cos={cos:.6} (tol={max_tol}, min_cos={min_cos})"
    );
    assert!(
        a.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );
}
