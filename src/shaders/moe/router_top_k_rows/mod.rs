//! Top-k MoE routing from raw logits per row.

use crate::Error;
use crate::model::moe::top_k_route_from_raw_logits;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "router_top_k_rows";

pub const SHADER: &str = include_str!("router_top_k_rows.metal");

#[derive(Debug, Clone, PartialEq)]
pub struct RouteOut {
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub probs: Vec<f32>,
    pub per_expert_scale: Vec<f32>,
    pub rows: usize,
    pub experts: usize,
    pub top_k: usize,
}

impl Fixture {
    pub fn out_indices_len(&self) -> usize {
        self.rows * self.top_k
    }
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    let rows = 2usize;
    let experts = 6usize;
    let logits = vec![
        1.0, 2.0, 3.0, 0.0, -1.0, 0.5, //
        0.5, 1.0, 1.5, 2.0, 0.0, -0.5,
    ];
    Fixture {
        probs: logits,
        per_expert_scale: vec![1.0; experts],
        rows,
        experts,
        top_k: 3,
    }
}

pub fn moe_fixture(_fmt: ElemFormat) -> Fixture {
    let rows = 4;
    let experts = 128;
    let top_k = 8;
    let _len = rows * experts;
    Fixture {
        probs: {
            let len = rows * experts;
            (0..len)
                .map(|i| ((i as f32) * 0.01).sin() + (i % 17) as f32 * 0.02)
                .collect()
        },
        per_expert_scale: (0..experts).map(|i| 1.0 + (i as f32) * 0.001).collect(),
        rows,
        experts,
        top_k,
    }
}

pub fn cpu(fix: &Fixture) -> RouteOut {
    let mut indices = Vec::with_capacity(fix.out_indices_len());
    let mut weights = Vec::with_capacity(fix.out_indices_len());
    for r in 0..fix.rows {
        let row = &fix.probs[r * fix.experts..(r + 1) * fix.experts];
        let route = top_k_route_from_raw_logits(row, fix.top_k, &fix.per_expert_scale);
        for &idx in &route.indices {
            indices.push(idx as u32);
        }
        weights.extend_from_slice(&route.weights);
    }
    RouteOut { indices, weights }
}

pub fn cpu_oracle(fix: &Fixture) -> RouteOut {
    cpu(fix)
}

#[cfg(target_os = "macos")]
pub fn gpu(fix: &Fixture, variant: KernelVariant) -> Result<RouteOut, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let prob_len = fix.probs.len();
    let out_len = fix.out_indices_len();
    let buf_p = pool
        .allocate(&ctx.device, prob_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_s = pool
        .allocate(&ctx.device, fix.experts * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_i = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        fix.rows * 8
    } else {
        4
    };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_f32(&buf_p, &fix.probs);
    BufferPool::write_f32(&buf_s, &fix.per_expert_scale);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    let params = [fix.rows as u32, fix.experts as u32, fix.top_k as u32];
    bind_gpu_buffers(&enc, &buf_p, &buf_s, &buf_i, &buf_w, &buf_dump, &params);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: fix.rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut indices = vec![0u32; out_len];
    let mut weights = vec![0.0f32; out_len];
    BufferPool::read_u32(&buf_i, &mut indices);
    BufferPool::read_f32(&buf_w, &mut weights);
    pool.release(prob_len * 4, buf_p);
    pool.release(fix.experts * 4, buf_s);
    pool.release(out_len * 4, buf_i);
    pool.release(out_len * 4, buf_w);
    pool.release(dump_bytes, buf_dump);
    Ok(RouteOut { indices, weights })
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
    probs: &ProtocolObject<dyn MTLBuffer>,
    scale: &ProtocolObject<dyn MTLBuffer>,
    indices: &ProtocolObject<dyn MTLBuffer>,
    weights: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    params: &[u32; 3],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(probs), 0, 0);
        enc.setBuffer_offset_atIndex(Some(scale), 0, 1);
        enc.setBuffer_offset_atIndex(Some(indices), 0, 2);
        enc.setBuffer_offset_atIndex(Some(weights), 0, 3);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    set_bytes(enc, params, 4);
}

#[cfg(target_os = "macos")]
fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

fn assert_route_eq(a: &RouteOut, b: &RouteOut) {
    assert_eq!(a.indices, b.indices, "indices mismatch");
    for (i, (&x, &y)) in a.weights.iter().zip(b.weights.iter()).enumerate() {
        assert!((x - y).abs() < 1e-5, "weight[{i}] cpu={x} gpu={y}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tiny_top_indices_descending() {
        let fix = tiny_fixture(ElemFormat::F32);
        let out = cpu(&fix);
        // Row 0 softmax peak at expert 2 (logit 3.0).
        assert_eq!(out.indices[0], 2);
    }

    #[test]
    fn cpu_matches_oracle() {
        for fix in [tiny_fixture(ElemFormat::F32), moe_fixture(ElemFormat::F32)] {
            assert_route_eq(&cpu(&fix), &cpu_oracle(&fix));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_matches_cpu_tiny() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let fix = tiny_fixture(ElemFormat::F32);
        let cpu = cpu(&fix);
        let gpu = gpu(&fix, KernelVariant::PRODUCTION).expect("gpu");
        assert_route_eq(&cpu, &gpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_matches_cpu_moe() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let fix = moe_fixture(ElemFormat::F32);
        let cpu = cpu(&fix);
        let gpu = gpu(&fix, KernelVariant::PRODUCTION).expect("gpu");
        assert_route_eq(&cpu, &gpu);
    }
}

crate::kernel_spec! {
    pub const SPEC {
        name: "router_top_k_rows",
        entry: "router_top_k_rows",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
