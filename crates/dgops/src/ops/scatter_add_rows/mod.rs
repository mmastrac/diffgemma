//! Scatter-add whole rows into a row-major destination with atomic adds.

use crate::Error;

pub const ENTRY: &str = "scatter_add_rows";
pub const METAL: &str = include_str!("scatter_add_rows.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/cuda/ops/scatter_add_rows/scatter_add_rows.cubin"
));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    /// Destination rows, [dst.len()/hidden, hidden].
    pub dst: Vec<f32>,
    pub indices: Vec<u32>,
    pub src: Vec<f32>,
    pub hidden: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.dst.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        dst: vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        indices: vec![0, 2, 0],
        src: vec![
            0.5, 0.25, 0.125, 0.0625, //
            1.0, 2.0, 3.0, 4.0, //
            0.5, 0.5, 0.5, 0.5,
        ],
        hidden: 4,
    }
}

/// Repeated and out-of-order indices over a hidden wider than one block.
pub fn repeated_fixture() -> Fixture {
    let dst_rows = 8;
    let hidden = 512;
    let indices: Vec<u32> = vec![0, 1, 1, 2, 3, 3, 3, 0];
    Fixture {
        dst: (0..dst_rows * hidden)
            .map(|i| ((i as f32) * 0.0013).cos() * 0.25)
            .collect(),
        src: (0..indices.len() * hidden)
            .map(|i| ((i as f32) * 0.0023).sin() * 0.5)
            .collect(),
        indices,
        hidden,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.dst.clone();
    for (t, &row) in fix.indices.iter().enumerate() {
        let dst_off = row as usize * fix.hidden;
        let src_off = t * fix.hidden;
        for d in 0..fix.hidden {
            out[dst_off + d] += fix.src[src_off + d];
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
        cpu = crate::ops::scatter_add_rows::cpu,
        gpu = crate::ops::scatter_add_rows::gpu,
        fixture = crate::ops::scatter_add_rows::tiny_fixture,
        out_len = crate::ops::scatter_add_rows::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }

    // Repeated indices are accumulated by device atomics, so the summation
    // order differs from the CPU reference's row-by-row order and the result
    // is not bit-identical: the measured worst case on the repeated fixture is
    // 1.19e-7 absolute (2 ULP at ~0.99, 376/4096 elements off by 1-2 ULP).
    // The tiny fixture's small exact-binary values still match bit for bit.
    crate::op_oracle_matrix! {
        mod repeated,
        cpu = crate::ops::scatter_add_rows::cpu,
        gpu = crate::ops::scatter_add_rows::gpu,
        fixture = crate::ops::scatter_add_rows::repeated_fixture,
        out_len = crate::ops::scatter_add_rows::fixture_len,
        max_tol = 1e-6,
        min_cos = 1.0,
    }
}
