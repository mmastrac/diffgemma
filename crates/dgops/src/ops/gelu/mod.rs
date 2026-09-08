//! PyTorch tanh-approximation GELU, in place.

use crate::Error;

pub const ENTRY: &str = "gelu";
pub const METAL: &str = include_str!("gelu.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/ops/gelu/gelu.cubin"));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

/// Matches the engine's gelu_tanh in include/activations.metal.
const GELU_TANH_COEF: f32 = 0.7978845608028654;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
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
        x: vec![-2.0, -1.0, 0.0, 0.5, 1.5, 3.0, 10.229641],
    }
}

/// Exercises the MLP intermediate width, well past one thread block.
pub fn mlp_shape_fixture() -> Fixture {
    let len = 16 * 2112;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.001).sin()).collect(),
    }
}

pub fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    let u = GELU_TANH_COEF * (x + 0.044_715 * x3);
    let t = if u > 8.0 {
        1.0
    } else if u < -8.0 {
        -1.0
    } else {
        u.tanh()
    };
    0.5 * x * (1.0 + t)
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.x.iter().copied().map(gelu_tanh).collect()
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
        cpu = crate::ops::gelu::cpu,
        gpu = crate::ops::gelu::gpu,
        fixture = crate::ops::gelu::tiny_fixture,
        out_len = crate::ops::gelu::fixture_len,
        max_tol = 1e-5,
        min_cos = 0.99999,
    }

    crate::op_oracle_matrix! {
        mod mlp_shape,
        cpu = crate::ops::gelu::cpu,
        gpu = crate::ops::gelu::gpu,
        fixture = crate::ops::gelu::mlp_shape_fixture,
        out_len = crate::ops::gelu::fixture_len,
        max_tol = 1e-5,
        min_cos = 0.99999,
    }
}
