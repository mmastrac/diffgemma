//! CPU oracle for the SC softmax-from-logits path. The GPU kernel was retired
//! (production materializes probs via `sc_prob_cols`); this `cpu` reference is
//! kept as the oracle that `sc_prob_cols`'s tests validate against.

use crate::shaders::f16;
use crate::shaders::test_util::ElemFormat;

/// Prob up-scale before the fp16 GEMM tiles; must match `SC_PROB_GEMM_SCALE` in
/// shaders/include/sc_prob_scale.metal and src/metal/step_kernel.rs.
const SC_PROB_GEMM_SCALE: f32 = 4096.0;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Fixture {
    pub fn rowstat(&self) -> Vec<f32> {
        crate::shaders::logit_rowstats::cpu(&crate::shaders::logit_rowstats::Fixture {
            logits: self.logits.clone(),
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn out_len(&self) -> usize {
        self.rows * self.cols
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        logits: vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5],
        rows: 2,
        cols: 3,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let rows = 2usize;
    let cols = 512usize;
    let logits: Vec<f32> = (0..rows * cols).map(|i| (i as f32 % 17.0) - 8.0).collect();
    Fixture { logits, rows, cols }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let rowstat = f.rowstat();
    let mut out = vec![0.0f32; f.out_len()];
    for row in 0..f.rows {
        let mx = rowstat[row * 2];
        let sum = rowstat[row * 2 + 1];
        for v in 0..f.cols {
            let x = f.logits[row * f.cols + v];
            // The (retired) kernel stored fp16(prob * SCALE); recover the prob so
            // the oracle compares probs at the kernel's precision.
            let prob = ((x - mx).exp()) / sum;
            out[row * f.cols + v] = f16::round_half(prob * SC_PROB_GEMM_SCALE) / SC_PROB_GEMM_SCALE;
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_rows_sum_to_one() {
        for fix in [tiny_fixture(ElemFormat::F32), wide_fixture(ElemFormat::F32)] {
            let out = cpu(&fix);
            for row in 0..fix.rows {
                let s: f32 = out[row * fix.cols..(row + 1) * fix.cols].iter().sum();
                assert!((s - 1.0).abs() < 0.01, "row {row} sum={s}");
            }
        }
    }
}
