//! Grouped MoE Q4: per-expert gate||up → GELU×up → down, weighted scatter.

use crate::Error;
use crate::dgq::block::quantize_row_q4;
use crate::dgq::layout::{q4_matrix_bytes, q4_row_bytes};
use crate::metal::{LayerOffsets, RouteScratch, TOP_K};
use crate::shaders::bf16;
use crate::shaders::cpu::moe_grouped::{GroupedRoute, MoeGroupedDims, moe_grouped_q4};
use crate::shaders::cpu::moe_router::moe_bucket_phases;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "moe_grouped";
pub const THREADGROUP_WIDTH: usize = 128;

pub const SHADER: &str = include_str!("moe_grouped.metal");

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
        quantize_stack(
            &self.gate_up_f32,
            self.n_experts,
            self.moe_ff * 2,
            self.hidden,
        )
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
            kv_ring_mask: 0,
        }
    }

    pub(crate) fn grouped_route(&self) -> GroupedRoute {
        let bucket = moe_bucket_phases(&self.expert_ids, self.n_experts as u32, self.top_k as u32);
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
            weight: [[0; TOP_K]; crate::metal::PREFILL_M],
            expert: [[0; TOP_K]; crate::metal::PREFILL_M],
            count: [0; crate::metal::N_EXPERTS],
            row_start: [0; crate::metal::N_EXPERTS + 1],
            num_slots: 0,
            num_active_experts: 0,
            active_expert: [0; crate::metal::N_EXPERTS],
            token_list: [0; crate::metal::PREFILL_M * crate::metal::TOP_K],
            slot_list: [0; crate::metal::PREFILL_M * crate::metal::TOP_K],
            token_slot: [[0; crate::metal::TOP_K]; crate::metal::PREFILL_M],
            block_expert: [0; crate::metal::MOE_MAX_BLOCKS],
            block_row0: [0; crate::metal::MOE_MAX_BLOCKS],
            num_blocks: 0,
        };
        for (tok, row) in self.expert_ids.iter().enumerate() {
            for (kk, &e) in row.iter().enumerate() {
                route.expert[tok][kk] = e;
                route.weight[tok][kk] = bf16::f32_to_bf16_bits(self.weights[tok][kk]);
            }
        }
        for (e, &off) in bucket.offset.iter().enumerate() {
            route.row_start[e] = off;
        }
        let n = self.n_experts;
        route.row_start[n] = bucket.num_slots;
        for e in (n + 1)..=crate::metal::N_EXPERTS {
            route.row_start[e] = bucket.num_slots;
        }
        route.num_slots = bucket.num_slots;
        let slots = bucket.num_slots as usize;
        route.token_list[..slots].copy_from_slice(&bucket.token_list[..slots]);
        route.slot_list[..slots].copy_from_slice(&bucket.slot_list[..slots]);
        crate::metal::fill_token_slot(&mut route);
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
    let moe_in: Vec<f32> = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.moe_in));
    // Production chain: kernel writes UNWEIGHTED per-slot rows, then
    // moe_scatter_weighted applies router weights into canvas rows. Mirror both.
    let mut slot_rows = vec![0.0f32; f.out_len()];
    moe_grouped_q4(
        &mut slot_rows,
        &moe_in,
        &f.gate_up_q4(),
        &f.down_q4(),
        &f.grouped_route(),
        f.dims().into(),
    );
    let scattered = crate::shaders::cpu::moe_scatter_weighted::moe_scatter_weighted(
        &slot_rows,
        &f.route_scratch(),
        f.hidden,
    );
    scattered[..f.out_len()].to_vec()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let moe_in_f16 = bf16::f32_slice_to_bf16_bits(&f.moe_in);
    let mut pool = BufferPool::new();
    let buf_in = pool
        .allocate(&ctx.device, moe_in_f16.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let route = f.route_scratch();
    let slots = route.num_slots as usize;
    let slot_elems = slots * f.hidden;
    let buf_slots = pool
        .allocate(&ctx.device, slot_elems * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let blob = f.blob();
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        (f.moe_ff * 2 + 36) * 4
    } else {
        4
    };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_bf16(&buf_in, &moe_in_f16);
    BufferPool::write_f32(&buf_slots, &vec![0.0f32; slot_elems]);
    BufferPool::write_f32(&buf_out, &vec![0.0f32; f.out_len()]);
    BufferPool::write_bytes(&buf_blob, &blob);
    let layer = f.layer_offsets();
    BufferPool::write_bytes(&buf_layer, unsafe {
        std::slice::from_raw_parts(
            &layer as *const LayerOffsets as *const u8,
            std::mem::size_of::<LayerOffsets>(),
        )
    });
    BufferPool::write_bytes(&buf_route, unsafe {
        std::slice::from_raw_parts(
            &route as *const RouteScratch as *const u8,
            std::mem::size_of::<RouteScratch>(),
        )
    });

    let dims = f.dims();
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_in), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_slots), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_blob), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&buf_route), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&buf_dump), 0, 6);
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

    let scatter_ps = crate::shaders::moe_scatter_weighted::pipeline_for(&ctx, variant)?;
    // Production layout: one TG per (256-wide d-tile, token), 256 threads
    // (d = tgid.x * 256 + tid) — matches encode_moe_batched_scatter.
    gpu_common::dispatch_grid(
        &ctx.queue,
        &scatter_ps.pipeline,
        f.hidden.div_ceil(256),
        f.canvas,
        256,
        |enc| {
            crate::shaders::moe_scatter_weighted::bind_gpu_buffers(
                enc,
                &buf_slots,
                &buf_out,
                &buf_route,
                f.hidden as u32,
                f.canvas as u32,
            );
        },
    )?;

    let mut out = vec![0.0f32; f.out_len()];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::moe_grouped::cpu,
        cpu_oracle = crate::shaders::moe_grouped::cpu_oracle,
        gpu = crate::shaders::moe_grouped::gpu,
        fixture = crate::shaders::moe_grouped::tiny_fixture,
        out_len = crate::shaders::moe_grouped::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::shaders::moe_grouped::cpu,
        cpu_oracle = crate::shaders::moe_grouped::cpu_oracle,
        gpu = crate::shaders::moe_grouped::gpu,
        fixture = crate::shaders::moe_grouped::wide_fixture,
        out_len = crate::shaders::moe_grouped::fixture_len,
        formats: [F32],
        max_tol = 3e-2,
        min_cos = 0.9999,
    }
}

pub mod cpu;
pub mod nvfp4;

crate::kernel_spec! {
    pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
        name: "moe_grouped",
        entry: "moe_grouped",
        quant_formats: &[
            crate::shaders::variant::QuantFormat::Q4Affine,
            crate::shaders::variant::QuantFormat::NvFp4,
        ],
        fc: &[],
        variants: crate::shaders::manifest::KernelVariants::Elementwise,
    };
}
