//! x *= scale, elementwise.

use crate::Error;

pub const ENTRY: &str = "vec_scale_inplace";
pub const METAL: &str = include_str!("vec_scale.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/ops/vec_scale/vec_scale.cubin"));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub scale: f32,
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
        x: vec![1.0, -2.0, 3.0, 0.0],
        scale: 0.5,
    }
}

pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.003).sin()).collect(),
        scale: -1.25,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.x.iter().map(|v| v * fix.scale).collect()
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
    crate::op_oracle_matrix! {
        mod tiny,
        cpu = crate::ops::vec_scale::cpu,
        gpu = crate::ops::vec_scale::gpu,
        fixture = crate::ops::vec_scale::tiny_fixture,
        out_len = crate::ops::vec_scale::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }

    crate::op_oracle_matrix! {
        mod long,
        cpu = crate::ops::vec_scale::cpu,
        gpu = crate::ops::vec_scale::gpu,
        fixture = crate::ops::vec_scale::long_fixture,
        out_len = crate::ops::vec_scale::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
