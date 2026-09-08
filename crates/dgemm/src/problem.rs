//! The GEMM problem and the buffers it runs on.

/// C = alpha * op(A) @ op(B) + beta * C, row-major with explicit leading
/// dimensions.
///
/// A is (m,k) when trans_a is false, else (k,m); B is (k,n) when trans_b is
/// false, else (n,k). Every leading dimension must be at least the width of a
/// stored row. One kernel body covers every transpose combination, so a
/// forward linear (W is (n,k)) and the backward forms (dW = dY^T @ X,
/// dX = dY @ W^T) are the same code with different flags.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Problem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

impl Problem {
    /// C = A @ B^T with A (m,k) and B (n,k): the PyTorch linear weight layout.
    pub fn linear(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            lda: k,
            ldb: k,
            ldc: n,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: true,
        }
    }

    /// C = A @ B with A (m,k) and B (k,n).
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            lda: k,
            ldb: n,
            ldc: n,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
        }
    }

    /// Elements A occupies under this layout.
    pub fn a_len(&self) -> usize {
        if self.trans_a {
            self.k * self.lda
        } else {
            self.m * self.lda
        }
    }

    /// Elements B occupies under this layout.
    pub fn b_len(&self) -> usize {
        if self.trans_b {
            self.n * self.ldb
        } else {
            self.k * self.ldb
        }
    }

    /// Elements C occupies under this layout.
    pub fn c_len(&self) -> usize {
        self.m * self.ldc
    }

    /// Elements the result occupies (m * n, ignoring the leading dimension).
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }
}

/// Kernel argument block; its layout must match the Metal struct and the CUDA
/// struct in gemm.metal / gemm.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AbiParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: u32,
    pub trans_b: u32,
}

impl From<Problem> for AbiParams {
    fn from(p: Problem) -> Self {
        Self {
            m: p.m as u32,
            n: p.n as u32,
            k: p.k as u32,
            lda: p.lda as u32,
            ldb: p.ldb as u32,
            ldc: p.ldc as u32,
            alpha: p.alpha,
            beta: p.beta,
            trans_a: u32::from(p.trans_a),
            trans_b: u32::from(p.trans_b),
        }
    }
}

/// One GEMM call: the operand buffers plus the problem describing them.
#[derive(Debug, Clone)]
pub struct Call {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub problem: Problem,
}

impl Call {
    /// Build a call, sizing nothing: the buffers must already match the
    /// problem's lengths (see Problem::a_len / b_len / c_len).
    pub fn new(a: Vec<f32>, b: Vec<f32>, c: Vec<f32>, problem: Problem) -> Self {
        Self { a, b, c, problem }
    }

    pub fn out_len(&self) -> usize {
        self.problem.out_len()
    }
}

pub fn out_len(call: &Call) -> usize {
    call.out_len()
}
