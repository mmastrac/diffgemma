//! Backward pass of per-row RMSNorm.

use crate::Error;

pub const ENTRY: &str = "rms_norm_backward";
pub const METAL: &str = include_str!("rms_norm_backward.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/cuda/ops/rms_norm_backward/rms_norm_backward.cubin"
));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub weight: Vec<f32>,
    pub dy: Vec<f32>,
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.rows * self.hidden
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: vec![1.0, 0.5, 2.0, 1.5],
        dy: vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
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
        dy: (0..rows * hidden)
            .map(|i| ((i as f32) * 0.011).cos() * 0.1)
            .collect(),
        rows,
        hidden,
        eps: 1e-6,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for r in 0..fix.rows {
        let off = r * fix.hidden;
        let mut sum_sq = 0.0f32;
        let mut dot = 0.0f32;
        for j in 0..fix.hidden {
            let xv = fix.x[off + j];
            sum_sq += xv * xv;
            dot += fix.dy[off + j] * fix.weight[j] * xv;
        }
        let inv = 1.0 / (sum_sq / fix.hidden as f32 + fix.eps).sqrt();
        let inv3 = inv * inv * inv;
        for i in 0..fix.hidden {
            out[off + i] = inv * fix.dy[off + i] * fix.weight[i]
                - inv3 * fix.x[off + i] * dot / fix.hidden as f32;
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

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(test)]
mod tests {
    crate::op_oracle_matrix! {
        mod tiny,
        cpu = crate::ops::rms_norm_backward::cpu,
        gpu = crate::ops::rms_norm_backward::gpu,
        fixture = crate::ops::rms_norm_backward::tiny_fixture,
        out_len = crate::ops::rms_norm_backward::fixture_len,
        max_tol = 1e-5,
        min_cos = 0.99999,
    }

    crate::op_oracle_matrix! {
        mod wide,
        cpu = crate::ops::rms_norm_backward::cpu,
        gpu = crate::ops::rms_norm_backward::gpu,
        fixture = crate::ops::rms_norm_backward::wide_fixture,
        out_len = crate::ops::rms_norm_backward::fixture_len,
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
