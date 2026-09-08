//! Thin wrappers over dgops that keep the model code readable.

use dgops::Error;
use dgops::ops::gemm::{Fixture, GemmParams, gpu as gemm_gpu};

/// C = alpha * op(A) @ op(B) + beta * C, with explicit leading dimensions.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    trans_a: bool,
    trans_b: bool,
    beta: f32,
) -> Result<Vec<f32>, Error> {
    let fix = Fixture {
        a: a.to_vec(),
        b: b.to_vec(),
        c,
        params: GemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: ldc as u32,
            alpha: 1.0,
            beta,
            trans_a: u32::from(trans_a),
            trans_b: u32::from(trans_b),
        },
    };
    gemm_gpu(&fix)
}

/// Y = X @ W^T with X (m,k), W (n,k) (PyTorch linear layout).
pub fn linear(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, Error> {
    gemm(
        x,
        w,
        vec![0.0; m * n],
        m,
        n,
        k,
        k,
        k,
        n,
        false,
        true,
        0.0,
    )
}

pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>, Error> {
    dgops::ops::rms_norm::gpu(&dgops::ops::rms_norm::Fixture {
        x: x.to_vec(),
        weight: weight.to_vec(),
        rows,
        hidden,
        eps,
    })
}

pub fn gelu(x: &[f32]) -> Result<Vec<f32>, Error> {
    dgops::ops::gelu::gpu(&dgops::ops::gelu::Fixture { x: x.to_vec() })
}

pub fn vec_add(x: &[f32], addend: &[f32]) -> Result<Vec<f32>, Error> {
    dgops::ops::vec_add::gpu(&dgops::ops::vec_add::Fixture {
        x: x.to_vec(),
        addend: addend.to_vec(),
    })
}

pub fn softmax(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, Error> {
    dgops::ops::softmax::gpu(&dgops::ops::softmax::Fixture {
        logits: x.to_vec(),
        rows,
        cols,
    })
}

pub fn gather_rows(src: &[f32], indices: &[u32], hidden: usize) -> Result<Vec<f32>, Error> {
    dgops::ops::gather_rows::gpu(&dgops::ops::gather_rows::Fixture {
        src: src.to_vec(),
        indices: indices.to_vec(),
        hidden,
    })
}
