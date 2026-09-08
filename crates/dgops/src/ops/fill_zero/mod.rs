//! Zero a range of an f32 buffer.

use crate::Error;

pub const ENTRY: &str = "vec_fill_zero";
pub const METAL: &str = include_str!("fill_zero.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/cuda/ops/fill_zero/fill_zero.cubin"
));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub base: usize,
    pub count: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        base: 2,
        count: 3,
    }
}

/// Offset range spanning several thread blocks.
pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.003).sin() + 2.0).collect(),
        base: 1000,
        count: 6000,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.x.clone();
    for v in out[fix.base..fix.base + fix.count].iter_mut() {
        *v = 0.0;
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
        cpu = crate::ops::fill_zero::cpu,
        gpu = crate::ops::fill_zero::gpu,
        fixture = crate::ops::fill_zero::tiny_fixture,
        out_len = crate::ops::fill_zero::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }

    crate::op_oracle_matrix! {
        mod long,
        cpu = crate::ops::fill_zero::cpu,
        gpu = crate::ops::fill_zero::gpu,
        fixture = crate::ops::fill_zero::long_fixture,
        out_len = crate::ops::fill_zero::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
