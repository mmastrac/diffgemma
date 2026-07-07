//! CPU oracle for scalar f32 block GEMM (`gemm_linear_f32`).

use crate::dgq::block::q4_gemm_cpu;
use crate::dgq::layout::{NVFP4_HEADER_BYTES, nvfp4_matrix_bytes, q4_matrix_bytes};
use crate::dgq::nvfp4::nvfp4_gemm_cpu;
use crate::kernels::sub::QuantFormat;

pub fn gemm_linear_f32_cpu(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[u8],
    n: usize,
    format: QuantFormat,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    match format {
        QuantFormat::NvFp4 => {
            assert!(w.len() >= NVFP4_HEADER_BYTES);
            assert_eq!(w.len(), nvfp4_matrix_bytes(n, k));
            let gscale = f32::from_le_bytes(w[..4].try_into().expect("nvfp4 header"));
            nvfp4_gemm_cpu(a, m, k, &w[NVFP4_HEADER_BYTES..], n, gscale, &mut out);
        }
        _ => {
            assert_eq!(w.len(), q4_matrix_bytes(n, k));
            q4_gemm_cpu(a, m, k, w, n, &mut out);
        }
    }
    out
}
