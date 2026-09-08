//! Per-row RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight.

crate::op_kernel! {
    name = "rms_norm_rows",
    metal = "rms_norm.metal",
    cuda = "rms_norm.cu",
    fixture = Fixture,
    tests = [
        tiny => tiny_fixture => (1e-5, 0.99999),
        wide => wide_fixture => (1e-4, 0.99999),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub weight: Vec<f32>,
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.rows * self.hidden
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: vec![1.0, 0.5, 2.0, 1.5],
        rows: 2,
        hidden: 4,
        eps: 1e-6,
    }
}

/// Hidden wider than one thread block (256), several rows.
pub fn wide_fixture() -> Fixture {
    let rows = 5;
    let hidden = 1024;
    Fixture {
        x: (0..rows * hidden)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect(),
        weight: (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect(),
        rows,
        hidden,
        eps: 1e-6,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for r in 0..fix.rows {
        let off = r * fix.hidden;
        let row = &fix.x[off..off + fix.hidden];
        let mut sum_sq = 0.0f32;
        for v in row {
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq / fix.hidden as f32 + fix.eps).sqrt();
        for i in 0..fix.hidden {
            out[off + i] = row[i] * inv * fix.weight[i];
        }
    }
    out
}
