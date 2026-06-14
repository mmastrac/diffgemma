//! Grouped MoE Q4: per-expert gate||up → GELU×up → down, weighted scatter.

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::block::quantize_row_q4;
use crate::dgq::layout::{q4_matrix_bytes, q4_row_bytes};
use crate::kernels::cpu::moe_grouped::{moe_grouped_q4, GroupedRoute, MoeGroupedDims};
use crate::kernels::cpu::moe_router::moe_bucket_phases;
use crate::metal::{LayerOffsets, RouteScratch, TOP_K};
use crate::safetensors::Error;

pub const ENTRY: &str = "moe_grouped";
pub const THREADGROUP_WIDTH: usize = 128;

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/include/common.metal"),
    include_str!("../../../shaders/include/dequant.metal"),
    include_str!("../../../shaders/include/activations.metal"),
    include_str!("../../../shaders/include/attention.metal"),
    include_str!("../../../shaders/include/moe_router.metal"),
    include_str!("../../../shaders/include/moe_grouped.metal"),
    include_str!("../../../shaders/kernels/moe_grouped.metal"),
);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GroupedDims {
    pub canvas: u32,
    pub hidden: u32,
    pub moe_ff: u32,
    pub n_experts: u32,
}

impl From<GroupedDims> for MoeGroupedDims {
    fn from(d: GroupedDims) -> Self {
        MoeGroupedDims {
            canvas: d.canvas,
            hidden: d.hidden,
            moe_ff: d.moe_ff,
            n_experts: d.n_experts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub moe_in: Vec<f32>,
    pub gate_up_f32: Vec<f32>,
    pub down_f32: Vec<f32>,
    pub expert_ids: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f32>>,
    pub canvas: usize,
    pub hidden: usize,
    pub moe_ff: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

impl Fixture {
    pub fn dims(&self) -> GroupedDims {
        GroupedDims {
            canvas: self.canvas as u32,
            hidden: self.hidden as u32,
            moe_ff: self.moe_ff as u32,
            n_experts: self.n_experts as u32,
        }
    }

    pub fn out_len(&self) -> usize {
        self.canvas * self.hidden
    }

    fn gate_up_q4(&self) -> Vec<u8> {
        quantize_stack(&self.gate_up_f32, self.n_experts, self.moe_ff * 2, self.hidden)
    }

    fn down_q4(&self) -> Vec<u8> {
        quantize_stack(&self.down_f32, self.n_experts, self.hidden, self.moe_ff)
    }

    pub fn blob(&self) -> Vec<u8> {
        let mut blob = self.gate_up_q4();
        blob.extend_from_slice(&self.down_q4());
        blob
    }

    pub fn layer_offsets(&self) -> LayerOffsets {
        let gu = 0u64;
        let dn = self.gate_up_q4().len() as u64;
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
            router_scale: 0,
            router_proj: 0,
            per_expert_scale: 0,
            pre_ff_ln_2: 0,
            experts_gate_up: gu,
            experts_down: dn,
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

    pub(crate) fn grouped_route(&self) -> GroupedRoute {
        let bucket = moe_bucket_phases(
            &self.expert_ids,
            self.n_experts as u32,
            self.top_k as u32,
        );
        GroupedRoute {
            offset: bucket.offset,
            num_slots: bucket.num_slots,
            token_list: bucket.token_list,
            slot_list: bucket.slot_list,
            weight: self.weights.clone(),
        }
    }

    pub(crate) fn route_scratch(&self) -> RouteScratch {
        let bucket = self.grouped_route();
        let mut route = RouteScratch {
            weight: [[0; TOP_K]; crate::metal::CANVAS],
            expert: [[0; TOP_K]; crate::metal::CANVAS],
            count: [0; crate::metal::N_EXPERTS],
            offset: [0; crate::metal::N_EXPERTS],
            num_slots: 0,
            pad_route: 0,
            token_list: [0; crate::metal::CANVAS * TOP_K],
            slot_list: [0; crate::metal::CANVAS * TOP_K],
        };
        for (tok, row) in self.expert_ids.iter().enumerate() {
            for (kk, &e) in row.iter().enumerate() {
                route.expert[tok][kk] = e;
                route.weight[tok][kk] = f16::f32_to_f16_bits(self.weights[tok][kk]);
            }
        }
        for (e, &off) in bucket.offset.iter().enumerate() {
            route.offset[e] = off;
        }
        route.num_slots = bucket.num_slots;
        let slots = bucket.num_slots as usize;
        route.token_list[..slots].copy_from_slice(&bucket.token_list[..slots]);
        route.slot_list[..slots].copy_from_slice(&bucket.slot_list[..slots]);
        route
    }
}

fn quantize_stack(rows: &[f32], experts: usize, out_dim: usize, in_dim: usize) -> Vec<u8> {
    let per = q4_matrix_bytes(out_dim, in_dim);
    let mut dst = vec![0u8; experts * per];
    let row_bytes = q4_row_bytes(in_dim);
    for e in 0..experts {
        for row in 0..out_dim {
            let src_off = e * out_dim * in_dim + row * in_dim;
            let dst_off = e * per + row * row_bytes;
            quantize_row_q4(
                &rows[src_off..src_off + in_dim],
                in_dim,
                &mut dst[dst_off..dst_off + row_bytes],
            );
        }
    }
    dst
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let canvas = 1usize;
    let hidden = 64usize;
    let moe_ff = 32usize;
    let n_experts = 4usize;
    let top_k = 1usize;
    Fixture {
        moe_in: (0..canvas * hidden)
            .map(|i| ((i as f32) * 0.07).sin() * 0.4)
            .collect(),
        gate_up_f32: (0..n_experts * moe_ff * 2 * hidden)
            .map(|i| ((i as f32) * 0.004).cos() * 0.03)
            .collect(),
        down_f32: (0..n_experts * hidden * moe_ff)
            .map(|i| ((i as f32) * 0.006).sin() * 0.025)
            .collect(),
        expert_ids: vec![vec![0]],
        weights: vec![vec![1.0]],
        canvas,
        hidden,
        moe_ff,
        n_experts,
        top_k,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let canvas = 2usize;
    let hidden = 128usize;
    let moe_ff = 64usize;
    let n_experts = 8usize;
    let top_k = 1usize;
    Fixture {
        moe_in: (0..canvas * hidden)
            .map(|i| ((i as f32) * 0.04).sin() * 0.35)
            .collect(),
        gate_up_f32: (0..n_experts * moe_ff * 2 * hidden)
            .map(|i| ((i as f32) * 0.003).cos() * 0.02)
            .collect(),
        down_f32: (0..n_experts * hidden * moe_ff)
            .map(|i| ((i as f32) * 0.005).sin() * 0.018)
            .collect(),
        expert_ids: vec![vec![1], vec![3]],
        weights: vec![vec![0.75], vec![0.5]],
        canvas,
        hidden,
        moe_ff,
        n_experts,
        top_k,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let moe_in: Vec<f32> = f16::f16_slice_to_f32(&f16::f32_slice_to_f16(&f.moe_in));
    let mut out = vec![0.0f32; f.out_len()];
    moe_grouped_q4(
        &mut out,
        &moe_in,
        &f.gate_up_q4(),
        &f.down_q4(),
        &f.grouped_route(),
        f.dims().into(),
    );
    out
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
    let moe_in_f16 = f16::f32_slice_to_f16(&f.moe_in);
    let buf_in = pool
        .allocate(&ctx.device, moe_in_f16.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 4)
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

    BufferPool::write_bf16(&buf_in, &moe_in_f16);
    BufferPool::write_f32(&buf_out, &vec![0.0f32; f.out_len()]);
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
    let route = f.route_scratch();
    BufferPool::write_bytes(
        &buf_route,
        unsafe {
            std::slice::from_raw_parts(
                &route as *const RouteScratch as *const u8,
                std::mem::size_of::<RouteScratch>(),
            )
        },
    );

    let dims = f.dims();
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_in), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_blob), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&buf_route), 0, 4);
    }
    gpu_common::set_bytes(&enc, &dims, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas,
            height: f.n_experts,
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

    let mut out = vec![0.0f32; f.out_len()];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
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
        cpu = crate::kernels::sub::moe_grouped::cpu,
        cpu_oracle = crate::kernels::sub::moe_grouped::cpu_oracle,
        gpu = crate::kernels::sub::moe_grouped::gpu,
        fixture = crate::kernels::sub::moe_grouped::tiny_fixture,
        out_len = crate::kernels::sub::moe_grouped::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::moe_grouped::cpu,
        cpu_oracle = crate::kernels::sub::moe_grouped::cpu_oracle,
        gpu = crate::kernels::sub::moe_grouped::gpu,
        fixture = crate::kernels::sub::moe_grouped::wide_fixture,
        out_len = crate::kernels::sub::moe_grouped::fixture_len,
        formats: [F32],
        max_tol = 3e-2,
        min_cos = 0.9999,
    }
}
