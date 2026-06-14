//! CPU oracle for block matrix dequant (`dequant_block_matrix`).

use crate::dgq::block::q4_weight_at;
use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes, NVFP4_HEADER_BYTES};
use crate::dgq::nvfp4::nvfp4_weight_at;
use crate::kernels::sub::QuantFormat;

/// `[N,K]` f32 row-major from Q4 or NVFP4 block matrix bytes.
pub fn dequant_block_matrix_cpu(w: &[u8], n: usize, k: usize, format: QuantFormat) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    match format {
        QuantFormat::NvFp4 => {
            assert!(w.len() >= NVFP4_HEADER_BYTES);
            assert_eq!(w.len(), nvfp4_matrix_bytes(n, k));
            let gscale = f32::from_le_bytes(w[..4].try_into().expect("nvfp4 header"));
            let body = &w[NVFP4_HEADER_BYTES..];
            for row in 0..n {
                for col in 0..k {
                    out[row * k + col] = nvfp4_weight_at(body, row, col, k, gscale);
                }
            }
        }
        _ => {
            assert_eq!(w.len(), q4_matrix_bytes(n, k));
            for row in 0..n {
                for col in 0..k {
                    out[row * k + col] = q4_weight_at(w, row, col, k);
                }
            }
        }
    }
    out
}
