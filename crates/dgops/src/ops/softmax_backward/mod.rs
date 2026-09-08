//! Backward pass of row softmax: ds = probs * (dp - sum(probs*dp)).

crate::op_kernel! {
    name = "softmax_backward",
    metal = "softmax_backward.metal",
    cuda = "softmax_backward.cu",
    fixture = Fixture => fix,
    abi = [
        in(buf_probs = fix.probs),
        in(buf_dp = fix.dp),
        out(buf_out = fix.len()),
        u32x2(fix.rows, fix.cols),
    ],
    launch = rows(fix.rows),
    result = (buf_out, fix.len()),
    tests = [
        tiny => tiny_fixture => (1e-6, 0.999999),
        wide => wide_fixture => (1e-6, 0.999999),
    ],
}

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
#[cfg(test)]
mod extra_tests {
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
