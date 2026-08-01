//! Weighted scatter of grouped MoE arena rows back to token rows (engine f32
//! prefill). GPU analog of `scatter_weighted_expert_outputs`: per-token
//! (arena_row, weight) lists are built in expert-job order so the f32 sum
//! order — and thus rounding — is bit-identical to the CPU scatter.

use crate::Error;
use crate::shaders::gpu_common;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "scatter_rows_weighted";

pub const SHADER: &str = include_str!("scatter_rows_weighted.metal");

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
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    #[test]
    fn gpu_matches_cpu_reference() {
        let seq_len = 5usize;
        let hidden = 32usize;
        let top_k = 3usize;
        let arena_rows = seq_len * top_k;
        let arena: Vec<f32> = (0..arena_rows * hidden)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        // Shuffle-ish rows: token t's k-th route hits arena row (t + k*seq_len).
        let mut rows = Vec::with_capacity(seq_len * top_k);
        let mut weights = Vec::with_capacity(seq_len * top_k);
        for t in 0..seq_len {
            for k in 0..top_k {
                rows.push((t + k * seq_len) as u32);
                weights.push(0.1 + 0.31 * (t as f32) + 0.07 * (k as f32));
            }
        }
        let cpu = cpu_reference(&arena, &rows, &weights, seq_len, hidden, top_k);

        let ctx = MetalContext::new().expect("metal");
        let pipeline = pipeline_for(&ctx, KernelVariant::PRODUCTION).expect("pipeline");
        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf_arena = batch.alloc_f32(&arena).expect("arena");
        let rows_bytes =
            unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), rows.len() * 4) };
        let buf_rows = batch.alloc_bytes(rows_bytes).expect("rows");
        let buf_w = batch.alloc_f32(&weights).expect("weights");
        let buf_out = batch.alloc_f32_out(seq_len * hidden).expect("out");
        let buf_dump = batch.alloc_f32_out(1).expect("dump");
        let dims = [seq_len as u32, hidden as u32, top_k as u32];
        batch.dispatch_1d(&pipeline.pipeline, seq_len * hidden, |enc| {
            bind_gpu_buffers(
                enc, &buf_arena, &buf_rows, &buf_w, &buf_out, &buf_dump, &dims,
            );
        });
        let mut gpu = vec![0.0f32; seq_len * hidden];
        batch.register_read(buf_out, &mut gpu);
        batch.end().expect("end");

        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "bit mismatch at {i}: cpu={a} gpu={b}"
            );
        }
    }
}
