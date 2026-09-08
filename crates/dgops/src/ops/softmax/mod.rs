//! Numerically stable row softmax, in place over [rows, cols].

crate::op_kernel! {
    name = "softmax_rows",
    metal = "softmax.metal",
    cuda = "ops/softmax/softmax",
    fixture = Fixture,
    tests = [
        tiny => tiny_fixture => (1e-6, 0.999999),
        wide => wide_fixture => (1e-6, 0.999999),
    ],
}

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
#[cfg(test)]
mod extra_tests {
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
