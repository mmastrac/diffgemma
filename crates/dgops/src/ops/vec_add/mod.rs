//! out += addend, elementwise.

use crate::Error;

pub const ENTRY: &str = "vec_add_inplace";
pub const METAL: &str = include_str!("vec_add.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/ops/vec_add/vec_add.cubin"));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub addend: Vec<f32>,
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
        x: vec![1.0, 2.0, 3.0, 4.0],
        addend: vec![0.5, -1.0, 2.0, 0.0],
    }
}

/// Long enough to cross several thread blocks.
pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.003).sin()).collect(),
        addend: (0..len).map(|i| ((i as f32) * 0.007).cos() * 0.25).collect(),
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.x
        .iter()
        .zip(fix.addend.iter())
        .map(|(x, a)| x + a)
        .collect()
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
        cpu = crate::ops::vec_add::cpu,
        gpu = crate::ops::vec_add::gpu,
        fixture = crate::ops::vec_add::tiny_fixture,
        out_len = crate::ops::vec_add::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }

    crate::op_oracle_matrix! {
        mod long,
        cpu = crate::ops::vec_add::cpu,
        gpu = crate::ops::vec_add::gpu,
        fixture = crate::ops::vec_add::long_fixture,
        out_len = crate::ops::vec_add::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
