/// C[M, N] = alpha * op(A)[M, K] @ op(B)[K, N] + beta * C
/// Row-major: A[M, K], B[K, N], C[M, N].
pub fn matmul(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    matmul_scaled(c, a, b, m, k, n, 1.0, 0.0);
}

pub fn matmul_scaled(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    #[cfg(feature = "blas")]
    {
        if try_blas_sgemm(c, a, b, m, k, n, alpha, beta, false, false) {
            return;
        }
    }

    matmul_naive(c, a, b, m, k, n, alpha, beta, false, false);
}

/// C[M, N] = alpha * A[M, K] @ B[N, K]^T + beta * C  (row-major B stored as [N, K])
pub fn matmul_b_transpose(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    #[cfg(feature = "blas")]
    {
        if try_blas_sgemm(c, a, b, m, k, n, alpha, beta, false, true) {
            return;
        }
    }

    matmul_naive(c, a, b, m, k, n, alpha, beta, false, true);
}

#[cfg(feature = "blas")]
fn try_blas_sgemm(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> bool {
    #[cfg(not(target_os = "macos"))]
    let _ = (c, a, b, m, k, n, alpha, beta, trans_a, trans_b);
    #[cfg(target_os = "macos")]
    {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                if trans_a { CBLAS_TRANS } else { CBLAS_NO_TRANS },
                if trans_b { CBLAS_TRANS } else { CBLAS_NO_TRANS },
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                if trans_a { m as i32 } else { k as i32 },
                b.as_ptr(),
                if trans_b { k as i32 } else { n as i32 },
                beta,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        return true;
    }
    #[cfg(not(target_os = "macos"))]
    false
}

fn matmul_naive(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let av = if trans_a {
                    a[p * m + i]
                } else {
                    a[i * k + p]
                };
                let bv = if trans_b {
                    b[j * k + p]
                } else {
                    b[p * n + j]
                };
                sum += av * bv;
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

#[cfg(all(feature = "blas", target_os = "macos"))]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(all(feature = "blas", target_os = "macos"))]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(all(feature = "blas", target_os = "macos"))]
const CBLAS_TRANS: i32 = 112;

#[cfg(all(feature = "blas", target_os = "macos"))]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        matmul(&mut c, &a, &b, 2, 2, 2);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_b_transpose_matches_linear_weight_layout() {
        // x[1,2] @ W[3,2]^T -> [1,3], W rows are out_features
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2]
        let mut y = [0.0f32; 3];
        matmul_b_transpose(&mut y, &x, &w, 1, 2, 3, 1.0, 0.0);
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }
}
