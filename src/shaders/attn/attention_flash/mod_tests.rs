//! Pure-CPU validation of the flash block-streaming numerics (no GPU): the
//! block-streaming online softmax must match the `cpu_causal` reference (which
//! itself matches the shipped GPU decomposition within eps) for every block
//! size — this pins the kernel's inner-loop rescale before any Metal exists.

use super::cpu_flash_blocked;
use crate::shaders::attention::{
    full_grp8_hd512_fixture, full_hd512_fixture, sliding_hd256_fixture,
};
use crate::shaders::attn::attention_gemm::cpu_causal;
use crate::shaders::test_util::ElemFormat;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// Block-streaming online softmax is mathematically identical to the single-pass
// reference; only f32 rounding differs, so a tight eps holds for ALL block
// sizes, including bk=1 (degenerate), bk that splits the causal cutoff, and
// bk >= t_total (single block == batch softmax).
fn assert_flash_matches(f: &crate::shaders::attention::Fixture, round_kv_f16: bool) {
    let reference = cpu_causal(f, round_kv_f16);
    for bk in [1usize, 3, 8, 16, 64, 4096] {
        let flash = cpu_flash_blocked(f, bk, round_kv_f16);
        let d = max_abs_diff(&reference, &flash);
        assert!(
            d < 1e-3,
            "flash bk={bk} round_kv_f16={round_kv_f16} diverged from cpu_causal: max|Δ|={d}"
        );
    }
}

// Full-attention layer (hd=512, group 2) — target shape for fused flash — f16 + f32-side.
#[test]
fn flash_blocked_matches_cpu_causal_full_hd512_f16() {
    assert_flash_matches(&full_hd512_fixture(ElemFormat::F32), true);
}

#[test]
fn flash_blocked_matches_cpu_causal_full_hd512_f32() {
    assert_flash_matches(&full_hd512_fixture(ElemFormat::F32), false);
}

// GQA group 8 (16 Q / 2 KV) — exercises the kvh = qh/group indexing.
#[test]
fn flash_blocked_matches_cpu_causal_grp8_f16() {
    assert_flash_matches(&full_grp8_hd512_fixture(ElemFormat::F32), true);
}

// Sliding-layer shape (hd=256).
#[test]
fn flash_blocked_matches_cpu_causal_sliding_hd256_f16() {
    assert_flash_matches(&sliding_hd256_fixture(ElemFormat::F32), true);
}

// ---- GPU oracle: the fused flash kernel vs the causal CPU reference. ----
#[cfg(target_os = "macos")]
#[test]
fn flash_gpu_full_grp8_causal_vs_cpu() {
    use crate::shaders::attn::attention_flash::gpu_flash;
    use crate::shaders::test_util::assert_oracle;
    let f = full_grp8_hd512_fixture(ElemFormat::F32);
    let got = gpu_flash(&f, true).expect("gpu flash causal");
    let oracle = cpu_causal(&f, true);
    assert_oracle(&got, &oracle, 2e-2, 0.9999);
}

#[cfg(target_os = "macos")]
#[test]
fn flash_gpu_full_grp2_causal_vs_cpu() {
    use crate::shaders::attn::attention_flash::gpu_flash;
    use crate::shaders::test_util::assert_oracle;
    let f = full_hd512_fixture(ElemFormat::F32);
    let got = gpu_flash(&f, true).expect("gpu flash causal");
    let oracle = cpu_causal(&f, true);
    assert_oracle(&got, &oracle, 2e-2, 0.9999);
}

/// Isolated attention-kernel bench: fused flash vs GEMM-decomposition vs
/// attention_mma_full at real full-layer shape (canvas=256, 16 Q / 2 KV,
/// hd=512). Ignored (timing). Note: isolated attention kernel measurements
/// over-weight attention vs real prefill — a win here is necessary but not
/// sufficient; confirm on real prefill benchmarks after integration.
/// Run: `cargo test --release flash_bench -- --ignored --nocapture`
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn flash_bench() {
    use crate::shaders::attn::attention_flash::bench_flash;
    use crate::shaders::attn::attention_gemm::{bench_gpu, model_full_fixture};
    let iters = 10usize;
    println!("  kv      mma_full   e17(hc16)  e18-flash   flash/e17");
    for kv_len in [8192u32, 30000, 60000, 100000] {
        let f = model_full_fixture(kv_len);
        let mma_full = crate::shaders::attention::bench_path(&f, iters, 3).unwrap();
        let e17 = bench_gpu(&f, iters, 16, true).unwrap();
        let flash = bench_flash(&f, iters, 16, true).unwrap();
        println!(
            "  {:>6}  {:>8.3}  {:>9.3}  {:>9.3}   {:>5.2}x",
            kv_len,
            mma_full,
            e17,
            flash,
            e17 / flash
        );
    }
}

/// Hypothesis: at hd=256 the O register cap relaxes, so flash can run larger BQ
/// (16/32/64) → bigger MMA tiles. Compares flash at each BQ vs the
/// GEMM-decomposition at the same hd=256 full-attention shape. Ignored (timing).
/// Note: hd=256 applies to sliding layers, which in production are
/// window-bounded (1024) — measurements here test tile physics on a non-bottleneck.
/// Run: `cargo test --release flash_bench_hd256 -- --ignored --nocapture`
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn flash_bench_hd256() {
    use crate::shaders::attention::model_bench_fixture;
    use crate::shaders::attn::attention_flash::bench_flash;
    use crate::shaders::attn::attention_gemm::bench_gpu;
    let iters = 10usize;
    println!("  kv       e17(hd256)  flash-bq16  flash-bq32   best/e17");
    for kv_len in [8192u32, 30000, 60000] {
        let f = model_bench_fixture(256, kv_len, true);
        let e17 = bench_gpu(&f, iters, 16, true).unwrap();
        let b16 = bench_flash(&f, iters, 16, true).unwrap();
        let b32 = bench_flash(&f, iters, 32, true).unwrap();
        println!(
            "  {:>6}  {:>9.3}  {:>9.3}  {:>9.3}   {:>5.2}x",
            kv_len,
            e17,
            b16,
            b32,
            e17 / b16.min(b32)
        );
    }
}
