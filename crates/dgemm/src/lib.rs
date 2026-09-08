//! The GEMM family: one tiled body per backend, one CPU oracle, one parity
//! test per shape.
//!
//! C = alpha * op(A) @ op(B) + beta * C. A GEMM here is the operation plus its
//! tiling; anything that changes what the kernel computes (weight format,
//! stacked segment tables, grouped MoE index maps, epilogue fusion) is a
//! compile-time axis value on the same body, never a forked kernel.
//!
//! This crate currently covers f32 operands. The quantized bodies the engine
//! runs (q4/q6/q8/nvfp4 decode fused into the tile load, stacked and grouped
//! structures, arena/gather epilogues) join as axis values, and the decode
//! they need arrives as the crate's Format axis rather than as a dependency on
//! the engine.

mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_rt;
pub mod fixtures;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
mod metal_rt;
pub mod problem;
pub mod testing;

pub use gpukit::Error;
pub use problem::{Call, Problem, out_len};

/// Block tile. Must match the #defines in gemm.metal / gemm.cu.
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;
pub const THREADS: u32 = 256;

pub const ENTRY: &str = "gemm_f32";
pub const METAL: &str = include_str!("gemm.metal");

#[cfg(all(feature = "cuda", dgemm_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/gemm.cubin"));
#[cfg(all(feature = "cuda", not(dgemm_cuda_kernels)))]
const CUBIN: &[u8] = &[];

/// The CPU reference: the same function in plain loops. This is the oracle.
pub fn cpu(call: &Call) -> Vec<f32> {
    cpu::cpu(call)
}

/// Run on the backend this build has: Metal on macOS, CUDA elsewhere when the
/// cuda feature is on.
pub fn gpu(call: &Call) -> Result<Vec<f32>, Error> {
    #[cfg(target_os = "macos")]
    {
        metal::gpu(call)
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        cuda::gpu(call)
    }
    #[cfg(not(any(target_os = "macos", all(feature = "cuda", not(target_os = "macos")))))]
    {
        let _ = call;
        Err(Error::Gpu(
            "no GPU backend enabled (build with --features cuda on a CUDA host)",
        ))
    }
}

#[cfg(test)]
mod tests {
    macro_rules! case {
        ($name:ident, $fixture:path) => {
            crate::gemm_case! {
                mod $name,
                fixture = $fixture,
                max_tol = 1e-4,
                min_cos = 0.9999,
            }
        };
    }

    case!(tiny, crate::fixtures::tiny);
    case!(linear, crate::fixtures::linear);
    case!(backward_a, crate::fixtures::backward_a);
    case!(both_trans, crate::fixtures::both_trans);
    case!(tile, crate::fixtures::tile);
}
