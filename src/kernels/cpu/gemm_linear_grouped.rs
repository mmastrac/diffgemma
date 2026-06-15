//! CPU oracle for grouped MoE block GEMM (`gemm_linear_grouped`).

use crate::dgq::block::q4_gemm_cpu;
use crate::dgq::layout::{
    blob_offset_usize, nvfp4_matrix_bytes, q4_matrix_bytes, NVFP4_HEADER_BYTES,
};
use crate::dgq::nvfp4::nvfp4_gemm_cpu;
use crate::kernels::sub::QuantFormat;
use crate::metal::BlockGroupedJob;

fn resolve_job(row_starts: &[u32], global_row: usize) -> usize {
    let num_jobs = row_starts.len().saturating_sub(1);
    if num_jobs == 0 {
        return 0;
    }
    let mut lo = 0usize;
    let mut hi = num_jobs;
    while lo + 1 < hi {
        let mid = (lo + hi) >> 1;
        if row_starts[mid] as usize <= global_row {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `C[total_m,N] = A[total_m,K] @ W[job]^T` with per-row job metadata.
pub fn gemm_linear_grouped_cpu(
    a: &[f32],
    total_m: usize,
    k: usize,
    n: usize,
    w_blob: &[u8],
    jobs: &[BlockGroupedJob],
    row_starts: &[u32],
    format: QuantFormat,
) -> Vec<f32> {
    assert_eq!(a.len(), total_m * k);
    let mut out = vec![0.0f32; total_m * n];
    for global_row in 0..total_m {
        let job_id = resolve_job(row_starts, global_row);
        if global_row >= row_starts.get(job_id + 1).copied().unwrap_or(0) as usize {
            continue;
        }
        let job = jobs[job_id];
        let w_off = blob_offset_usize(job.w_byte_off).expect("grouped job w_byte_off");
        let w_len = match format {
            QuantFormat::NvFp4 => nvfp4_matrix_bytes(n, k),
            _ => q4_matrix_bytes(n, k),
        };
        assert!(w_off + w_len <= w_blob.len(), "grouped job weight slice OOB");
        let w = &w_blob[w_off..w_off + w_len];
        let row_out = &mut out[global_row * n..(global_row + 1) * n];
        let row_a = &a[global_row * k..(global_row + 1) * k];
        match format {
            QuantFormat::NvFp4 => {
                assert!(w.len() >= NVFP4_HEADER_BYTES);
                let gscale = f32::from_le_bytes(w[..4].try_into().expect("nvfp4 header"));
                nvfp4_gemm_cpu(row_a, 1, k, &w[NVFP4_HEADER_BYTES..], n, gscale, row_out);
            }
            _ => {
                q4_gemm_cpu(row_a, 1, k, w, n, row_out);
            }
        }
    }
    out
}
