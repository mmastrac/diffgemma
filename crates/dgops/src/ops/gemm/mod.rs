//! Tiled f32 GEMM: C = alpha * op(A) @ op(B) + beta * C.
//!
//! A is (M,K) row-major when trans_a is false, else (K,M); B is (K,N) when
//! trans_b is false, else (N,K); leading dimensions are explicit. One kernel
//! body covers every transpose combination, so forward (W is (N,K)) and
//! backward (dW = dY^T @ X, dX = dY @ W^T) use the same code.

use crate::Error;

pub const ENTRY: &str = "gemm_f32";
pub const METAL: &str = include_str!("gemm.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/ops/gemm/gemm.cubin"));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

/// Block tile. Must match the #defines in gemm.metal / gemm.cu.
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;
pub const THREADS: u32 = 256;

/// Kernel parameters; layout must match the Metal struct and the CUDA struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmParams {
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

#[derive(Debug, Clone)]
pub struct Fixture {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub params: GemmParams,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.params.m as usize * self.params.n as usize
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

/// Build a fixture. A/B/C are sized from the transpose flags and leading dims.
fn build(m: usize, n: usize, k: usize, trans_a: bool, trans_b: bool, beta: f32) -> Fixture {
    let lda = if trans_a { m } else { k };
    let ldb = if trans_b { k } else { n };
    let ldc = n;
    let a_len = if trans_a { k * lda } else { m * lda };
    let b_len = if trans_b { n * ldb } else { k * ldb };
    let a: Vec<f32> = (0..a_len).map(|i| ((i as f32) * 0.013).sin() * 0.25).collect();
    let b: Vec<f32> = (0..b_len).map(|i| ((i as f32) * 0.007).cos() * 0.02).collect();
    let c: Vec<f32> = (0..m * ldc)
        .map(|i| ((i as f32) * 0.021).sin() * 0.1)
        .collect();
    Fixture {
        a,
        b,
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
    }
}

/// Small square case, no transposes, beta = 0.
pub fn tiny_fixture() -> Fixture {
    build(3, 4, 5, false, false, 0.0)
}

/// Linear layer: W is (N,K), so trans_b.
pub fn linear_fixture() -> Fixture {
    build(8, 16, 32, false, true, 0.0)
}

/// Backward: A is K x M (trans_a), B is K x N.
pub fn backward_a_fixture() -> Fixture {
    build(6, 5, 7, true, false, 0.0)
}

/// Both transposes, with a nonzero C accumulated at beta.
pub fn both_trans_fixture() -> Fixture {
    build(9, 7, 11, true, true, 0.5)
}

/// Tile-boundary case: M, N and K all cross the 64/16 tiles with a tail.
pub fn tile_fixture() -> Fixture {
    build(BM + 13, BN + 7, 3 * BK + 5, false, true, 0.0)
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let p = fix.params;
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let (lda, ldb, ldc) = (p.lda as usize, p.ldb as usize, p.ldc as usize);
    let mut out = fix.c.clone();
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = if p.trans_a != 0 {
                    fix.a[kk * lda + i]
                } else {
                    fix.a[i * lda + kk]
                };
                let bv = if p.trans_b != 0 {
                    fix.b[j * ldb + kk]
                } else {
                    fix.b[kk * ldb + j]
                };
                acc += av * bv;
            }
            let idx = i * ldc + j;
            out[idx] = p.alpha * acc + p.beta * fix.c[idx];
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

#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod tests {
    macro_rules! case {
        ($name:ident, $fix:path) => {
            crate::op_oracle_matrix! {
                mod $name,
                cpu = crate::ops::gemm::cpu,
                gpu = crate::ops::gemm::gpu,
                fixture = $fix,
                out_len = crate::ops::gemm::fixture_len,
                max_tol = 1e-4,
                min_cos = 0.9999,
            }
        };
    }

    case!(tiny, crate::ops::gemm::tiny_fixture);
    case!(linear, crate::ops::gemm::linear_fixture);
    case!(backward_a, crate::ops::gemm::backward_a_fixture);
    case!(both_trans, crate::ops::gemm::both_trans_fixture);
    case!(tile, crate::ops::gemm::tile_fixture);
}
