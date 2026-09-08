//! Backward pass of row softmax: ds = probs * (dp - sum(probs*dp)).

use crate::Error;

pub const ENTRY: &str = "softmax_backward";
pub const METAL: &str = include_str!("softmax_backward.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/cuda/ops/softmax_backward/softmax_backward.cubin"
));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    /// Already-softmaxed rows.
    pub probs: Vec<f32>,
    pub dp: Vec<f32>,
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

/// Row softmax, used to build a fixture's probs from logits.
fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = logits.to_vec();
    for r in 0..rows {
        let row = &mut out[r * cols..(r + 1) * cols];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    out
}

pub fn tiny_fixture() -> Fixture {
    let (rows, cols) = (2, 3);
    let logits = [1.0, 2.0, 3.0, 0.0, -1.0, 0.5];
    Fixture {
        probs: softmax_rows(&logits, rows, cols),
        dp: vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        rows,
        cols,
    }
}

/// Columns wider than one thread block.
pub fn wide_fixture() -> Fixture {
    let rows = 4;
    let cols = 1024;
    let logits: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.03).sin() * 2.0)
        .collect();
    Fixture {
        probs: softmax_rows(&logits, rows, cols),
        dp: (0..rows * cols)
            .map(|i| ((i as f32) * 0.05).cos() * 0.5)
            .collect(),
        rows,
        cols,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for r in 0..fix.rows {
        let off = r * fix.cols;
        let mut s = 0.0f32;
        for j in 0..fix.cols {
            s += fix.probs[off + j] * fix.dp[off + j];
        }
        for i in 0..fix.cols {
            out[off + i] = fix.probs[off + i] * (fix.dp[off + i] - s);
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
        cpu = crate::ops::softmax_backward::cpu,
        gpu = crate::ops::softmax_backward::gpu,
        fixture = crate::ops::softmax_backward::tiny_fixture,
        out_len = crate::ops::softmax_backward::fixture_len,
        max_tol = 1e-6,
        min_cos = 0.999999,
    }

    crate::op_oracle_matrix! {
        mod wide,
        cpu = crate::ops::softmax_backward::cpu,
        gpu = crate::ops::softmax_backward::gpu,
        fixture = crate::ops::softmax_backward::wide_fixture,
        out_len = crate::ops::softmax_backward::fixture_len,
        max_tol = 1e-6,
        min_cos = 0.999999,
    }

    /// The defining invariant: every row's ds sums to zero.
    #[test]
    fn rows_sum_to_zero() {
        if !crate::testing::gpu_available() {
            return;
        }
        let fix = crate::ops::softmax_backward::wide_fixture();
        let out = crate::ops::softmax_backward::gpu(&fix).expect("gpu dispatch");
        for r in 0..fix.rows {
            let sum: f32 = out[r * fix.cols..(r + 1) * fix.cols].iter().sum();
            assert!(sum.abs() < 1e-5, "row {r} sums to {sum}");
        }
    }
}
