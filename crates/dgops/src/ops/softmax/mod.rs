//! Numerically stable row softmax, in place over [rows, cols].

use crate::Error;

pub const ENTRY: &str = "softmax_rows";
pub const METAL: &str = include_str!("softmax.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/ops/softmax/softmax.cubin"));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        logits: vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5],
        rows: 2,
        cols: 3,
    }
}

/// Columns wider than one thread block, plus a large-magnitude row.
pub fn wide_fixture() -> Fixture {
    let rows = 4;
    let cols = 1024;
    let mut logits: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.03).sin() * 2.0)
        .collect();
    logits[0] = 500.0;
    logits[1] = -500.0;
    Fixture { logits, rows, cols }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.logits.clone();
    for r in 0..fix.rows {
        let row = &mut out[r * fix.cols..(r + 1) * fix.cols];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    out
}

pub fn gpu(fix: &Fixture) -> Result<Vec<f32>, Error> {
    #[cfg(target_os = "macos")]
    {
        metal::gpu(fix)
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        cuda::gpu(fix)
    }
    #[cfg(not(any(target_os = "macos", all(feature = "cuda", not(target_os = "macos")))))]
    {
        let _ = fix;
        Err(Error::Gpu(
            "no GPU backend enabled (build with --features cuda on a CUDA host)",
        ))
    }
}

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(test)]
mod tests {
    crate::op_oracle_matrix! {
        mod tiny,
        cpu = crate::ops::softmax::cpu,
        gpu = crate::ops::softmax::gpu,
        fixture = crate::ops::softmax::tiny_fixture,
        out_len = crate::ops::softmax::fixture_len,
        max_tol = 1e-6,
        min_cos = 0.999999,
    }

    crate::op_oracle_matrix! {
        mod wide,
        cpu = crate::ops::softmax::cpu,
        gpu = crate::ops::softmax::gpu,
        fixture = crate::ops::softmax::wide_fixture,
        out_len = crate::ops::softmax::fixture_len,
        max_tol = 1e-6,
        min_cos = 0.999999,
    }

    /// The invariant that must hold regardless of backend: rows sum to 1.
    #[test]
    fn rows_sum_to_one() {
        if !crate::testing::gpu_available() {
            return;
        }
        let fix = crate::ops::softmax::wide_fixture();
        let out = crate::ops::softmax::gpu(&fix).expect("gpu dispatch");
        for r in 0..fix.rows {
            let sum: f32 = out[r * fix.cols..(r + 1) * fix.cols].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {r} sums to {sum}");
        }
    }
}
