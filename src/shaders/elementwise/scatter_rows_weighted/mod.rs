//! Weighted scatter of grouped MoE arena rows back to token rows (engine f32
//! prefill). GPU analog of `scatter_weighted_expert_outputs`: per-token
//! (arena_row, weight) lists are built in expert-job order so the f32 sum
//! order — and thus rounding — is bit-identical to the CPU scatter.

use crate::shaders::gpu_common;

crate::shader_kernel! {
    name = "scatter_rows_weighted",
    metal = "scatter_rows_weighted.metal",
    pipeline = |ctx, variant| ctx.compile_subkernel(SHADER, ENTRY, variant),
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}

/// CPU reference: same accumulation order as the kernel (k ascending per token).
pub fn cpu_reference(
    arena: &[f32],
    rows: &[u32],
    weights: &[f32],
    seq_len: usize,
    hidden: usize,
    top_k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * hidden];
    for tok in 0..seq_len {
        for d in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..top_k {
                let slot = tok * top_k + k;
                acc += weights[slot] * arena[rows[slot] as usize * hidden + d];
            }
            out[tok * hidden + d] = acc;
        }
    }
    out
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    arena: &ProtocolObject<dyn MTLBuffer>,
    rows: &ProtocolObject<dyn MTLBuffer>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 3],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(arena), 0, 0);
        enc.setBuffer_offset_atIndex(Some(rows), 0, 1);
        enc.setBuffer_offset_atIndex(Some(weights), 0, 2);
        enc.setBuffer_offset_atIndex(Some(out), 0, 3);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    gpu_common::set_bytes(enc, dims, 4);
}
