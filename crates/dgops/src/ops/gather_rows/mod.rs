//! Gather rows by index from a row-major [tokens, hidden] f32 source.

use crate::Error;

pub const ENTRY: &str = "gather_rows";
pub const METAL: &str = include_str!("gather_rows.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/cuda/ops/gather_rows/gather_rows.cubin"
));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub src: Vec<f32>,
    pub indices: Vec<u32>,
    pub hidden: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.indices.len() * self.hidden
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        src: vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        indices: vec![2, 0, 1],
        hidden: 4,
    }
}

/// Repeated and out-of-order indices over a hidden wider than one block.
pub fn moe_fixture() -> Fixture {
    let tokens = 64;
    let hidden = 512;
    Fixture {
        src: (0..tokens * hidden)
            .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
            .collect(),
        indices: (0..32).map(|i| ((i * 7 + 3) % tokens) as u32).collect(),
        hidden,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for (t, &row) in fix.indices.iter().enumerate() {
        let src_off = row as usize * fix.hidden;
        let dst_off = t * fix.hidden;
        out[dst_off..dst_off + fix.hidden].copy_from_slice(&fix.src[src_off..src_off + fix.hidden]);
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
        cpu = crate::ops::gather_rows::cpu,
        gpu = crate::ops::gather_rows::gpu,
        fixture = crate::ops::gather_rows::tiny_fixture,
        out_len = crate::ops::gather_rows::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }

    crate::op_oracle_matrix! {
        mod moe,
        cpu = crate::ops::gather_rows::cpu,
        gpu = crate::ops::gather_rows::gpu,
        fixture = crate::ops::gather_rows::moe_fixture,
        out_len = crate::ops::gather_rows::fixture_len,
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
