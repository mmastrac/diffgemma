//! The CPU oracle: plain loops over flat arrays, no shared helper with the
//! GPU path.

use crate::problem::Call;

pub fn cpu(call: &Call) -> Vec<f32> {
    let p = call.problem;
    let mut out = call.c.clone();
    for i in 0..p.m {
        for j in 0..p.n {
            let mut acc = 0.0f32;
            for kk in 0..p.k {
                let av = if p.trans_a {
                    call.a[kk * p.lda + i]
                } else {
                    call.a[i * p.lda + kk]
                };
                let bv = if p.trans_b {
                    call.b[j * p.ldb + kk]
                } else {
                    call.b[kk * p.ldb + j]
                };
                acc += av * bv;
            }
            let idx = i * p.ldc + j;
            out[idx] = p.alpha * acc + p.beta * call.c[idx];
        }
    }
    out
}
