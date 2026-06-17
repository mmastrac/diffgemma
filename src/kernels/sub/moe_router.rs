//! Monolithic MoE router: RMSNorm → scale → linear → top-k → softmax(top-k).

use super::bf16;
use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::moe_router::{moe_router_rows, pack_route_rows, RouterDims as CpuRouterDims};
use crate::metal::{LayerOffsets, RouteScratch, TOP_K};
use crate::safetensors::Error;

pub const ENTRY: &str = "moe_router";
pub const THREADGROUP_WIDTH: usize = 128;

const SHADER: &str = shader_include::include_metal!("kernels/moe_router.metal");

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RouterDims {
    pub canvas: u32,
    pub hidden: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub router_hscale: f32,
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub stream: Vec<f32>,
    pub router_scale: Vec<f32>,
    pub router_proj: Vec<f32>,
    pub per_expert_scale: Vec<f32>,
    pub canvas: usize,
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

impl Fixture {
    pub fn dims(&self) -> RouterDims {
        RouterDims {
            canvas: self.canvas as u32,
            hidden: self.hidden as u32,
            n_experts: self.n_experts as u32,
            top_k: self.top_k as u32,
            router_hscale: (self.hidden as f32).powf(-0.5),
        }
    }

    pub fn out_len(&self) -> usize {
        self.canvas * self.top_k * 2
    }

    pub fn blob(&self) -> Vec<u8> {
        let mut blob = bf16::pack_bf16_slice(&self.router_scale);
        for e in 0..self.n_experts {
            let row = &self.router_proj[e * self.hidden..(e + 1) * self.hidden];
            blob.extend_from_slice(&bf16::pack_bf16_slice(row));
        }
        blob.extend_from_slice(&bf16::pack_bf16_slice(&self.per_expert_scale));
        blob
    }

    pub fn layer_offsets(&self) -> LayerOffsets {
        let rs = 0u64;
        let rp = (self.hidden * 2) as u64;
        let pes = rp + (self.n_experts * self.hidden * 2) as u64;
        LayerOffsets {
            input_ln: 0,
            q_proj: 0,
            q_norm: 0,
            k_proj: 0,
            k_norm: 0,
            v_proj: 0,
            o_proj: 0,
            post_attn_ln: 0,
            pre_ff_ln: 0,
            mlp_gate: 0,
            mlp_up: 0,
            mlp_down: 0,
            post_ff_ln_1: 0,
            router_scale: rs,
            router_proj: rp,
            per_expert_scale: pes,
            pre_ff_ln_2: 0,
            experts_gate_up: 0,
            experts_down: 0,
            post_ff_ln_2: 0,
            post_ff_ln: 0,
            layer_scalar: 0,
            kv_region: 0,
            head_dim: 0,
            n_kv_heads: 0,
            is_full: 0,
            _pad: 0,
        }
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let canvas = 2usize;
    let hidden = 64usize;
    let n_experts = 8usize;
    let top_k = 3usize;
    Fixture {
        stream: (0..canvas * hidden)
            .map(|i| (i as f32 * 0.09).sin() * 0.7)
            .collect(),
        router_scale: vec![1.0; hidden],
        router_proj: (0..n_experts * hidden)
            .map(|i| (i as f32 * 0.01).cos() * 0.05)
            .collect(),
        per_expert_scale: (0..n_experts).map(|i| 1.0 + i as f32 * 0.02).collect(),
        canvas,
        hidden,
        n_experts,
        top_k,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let canvas = 4usize;
    let hidden = 128usize;
    let n_experts = 32usize;
    let top_k = 8usize;
    Fixture {
        stream: (0..canvas * hidden)
            .map(|i| (i as f32 * 0.03).sin() * 0.5)
            .collect(),
        router_scale: (0..hidden).map(|i| 1.0 + (i as f32 * 0.001).sin() * 0.1).collect(),
        router_proj: (0..n_experts * hidden)
            .map(|i| (i as f32 * 0.005).cos() * 0.04)
            .collect(),
        per_expert_scale: (0..n_experts).map(|i| 1.0 + (i as f32) * 0.001).collect(),
        canvas,
        hidden,
        n_experts,
        top_k,
    }
}

fn route_from_scratch(route: &RouteScratch, f: &Fixture) -> Vec<f32> {
    let mut out = Vec::with_capacity(f.out_len());
    for tok in 0..f.canvas {
        for kk in 0..f.top_k {
            out.push(route.expert[tok][kk] as f32);
        }
        for kk in 0..f.top_k {
            out.push(bf16::bf16_bits_to_f32(route.weight[tok][kk]));
        }
    }
    out
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let rows = moe_router_rows(
        &bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.stream)),
        &f.router_scale,
        &f.router_proj,
        &f.per_expert_scale,
        CpuRouterDims {
            canvas: f.canvas as u32,
            hidden: f.hidden as u32,
            n_experts: f.n_experts as u32,
            top_k: f.top_k as u32,
            router_hscale: f.dims().router_hscale,
        },
    );
    pack_route_rows(&rows)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_stream = pool
        .allocate(&ctx.device, f.stream.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let blob = f.blob();
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_stream, &bf16::f32_slice_to_bf16_bits(&f.stream));
    BufferPool::write_bytes(&buf_blob, &blob);
    let layer = f.layer_offsets();
    BufferPool::write_bytes(
        &buf_layer,
        unsafe {
            std::slice::from_raw_parts(
                &layer as *const LayerOffsets as *const u8,
                std::mem::size_of::<LayerOffsets>(),
            )
        },
    );
    let zero = RouteScratch {
        weight: [[0; TOP_K]; crate::metal::CANVAS],
        expert: [[0; TOP_K]; crate::metal::CANVAS],
        count: [0; crate::metal::N_EXPERTS],
        row_start: [0; crate::metal::N_EXPERTS + 1],
        num_slots: 0,
        pad_route: 0,
        token_list: [0; crate::metal::CANVAS * TOP_K],
        slot_list: [0; crate::metal::CANVAS * TOP_K],
        token_slot: [[0; TOP_K]; crate::metal::CANVAS],
    };
    BufferPool::write_bytes(
        &buf_route,
        unsafe {
            std::slice::from_raw_parts(
                &zero as *const RouteScratch as *const u8,
                std::mem::size_of::<RouteScratch>(),
            )
        },
    );

    let dims = f.dims();
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_stream), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_blob), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_route), 0, 3);
    }
    gpu_common::set_bytes(&enc, &dims, 4);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let route: RouteScratch = unsafe {
        std::ptr::read(buf_route.contents().as_ptr() as *const RouteScratch)
    };
    Ok(route_from_scratch(&route, f))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::moe_router::cpu,
        cpu_oracle = crate::kernels::sub::moe_router::cpu_oracle,
        gpu = crate::kernels::sub::moe_router::gpu,
        fixture = crate::kernels::sub::moe_router::tiny_fixture,
        out_len = crate::kernels::sub::moe_router::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::moe_router::cpu,
        cpu_oracle = crate::kernels::sub::moe_router::cpu_oracle,
        gpu = crate::kernels::sub::moe_router::gpu,
        fixture = crate::kernels::sub::moe_router::wide_fixture,
        out_len = crate::kernels::sub::moe_router::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }
}
