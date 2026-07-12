//! Expert bucketing phases 0/1/2 for grouped MoE dispatch.

use super::moe_router::RouterDims;
use crate::metal::{RouteScratch, TOP_K};
use crate::safetensors::Error;
use crate::shaders::cpu::moe_router::{moe_bucket_phases, pack_bucket_state};
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "moe_bucket_fill";

const SHADER: &str = shader_include::include_metal!("moe/moe_bucket_fill/moe_bucket_fill.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub experts: Vec<Vec<u32>>,
    pub n_experts: u32,
    pub top_k: u32,
}

impl Fixture {
    pub fn canvas(&self) -> usize {
        self.experts.len()
    }

    pub fn dims(&self) -> RouterDims {
        RouterDims {
            canvas: self.canvas() as u32,
            hidden: 1,
            n_experts: self.n_experts,
            top_k: self.top_k,
            router_hscale: 1.0,
            block_m: 32,
        }
    }

    pub fn out_len(&self) -> usize {
        let n = self.n_experts as usize;
        let slots = self.canvas() * self.top_k as usize;
        n + 1 + slots * 2
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    unique_fixture(ElemFormat::F32)
}

pub fn unique_fixture(_: ElemFormat) -> Fixture {
    // Each slot maps to a distinct expert → deterministic bucket order.
    let canvas = 4usize;
    let top_k = 3usize;
    let experts = (0..canvas)
        .map(|tok| (0..top_k).map(|kk| (tok * top_k + kk) as u32).collect())
        .collect();
    Fixture {
        experts,
        n_experts: 12,
        top_k: top_k as u32,
    }
}

fn scratch_from_fixture(f: &Fixture) -> RouteScratch {
    let mut route = RouteScratch {
        weight: [[0; TOP_K]; crate::metal::PREFILL_M],
        expert: [[0; TOP_K]; crate::metal::PREFILL_M],
        count: [0; crate::metal::N_EXPERTS],
        row_start: [0; crate::metal::N_EXPERTS + 1],
        num_slots: 0,
        num_active_experts: 0,
        active_expert: [0; crate::metal::N_EXPERTS],
        token_list: [0; crate::metal::PREFILL_M * TOP_K],
        slot_list: [0; crate::metal::PREFILL_M * TOP_K],
        token_slot: [[0; TOP_K]; crate::metal::PREFILL_M],
        block_expert: [0; crate::metal::MOE_MAX_BLOCKS],
        block_row0: [0; crate::metal::MOE_MAX_BLOCKS],
        num_blocks: 0,
    };
    for (tok, row) in f.experts.iter().enumerate() {
        for (kk, &e) in row.iter().enumerate() {
            route.expert[tok][kk] = e;
        }
    }
    route
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let state = moe_bucket_phases(&f.experts, f.n_experts, f.top_k);
    pack_bucket_state(&state, f.n_experts as usize)
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
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[repr(C)]
struct MoeGroupedGridInfo {
    gate_n: u32,
    hid: u32,
    n_tile: u32,
    tpg: u32,
    tunable_n_tile: u32,
    tunable_wide_n_tile: u32,
}

#[cfg(target_os = "macos")]
fn run_phases(
    enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &crate::metal::device::ComputePipeline,
    buf_route: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_expert_unique: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    buf_indirect: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    f: &Fixture,
) {
    let dims = f.dims();
    let layer_idx = 0u32;
    let grid_info = MoeGroupedGridInfo {
        gate_n: 1408,
        hid: 2816,
        n_tile: 128,
        tpg: 128,
        tunable_n_tile: 64,
        tunable_wide_n_tile: 64,
    };
    for phase in 0u32..3 {
        enc.setComputePipelineState(&pipeline.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf_route), 0, 0);
            enc.setBuffer_offset_atIndex(Some(buf_expert_unique), 0, 3);
            enc.setBuffer_offset_atIndex(Some(buf_indirect), 0, 6);
        }
        gpu_common::set_bytes(enc, &phase, 1);
        gpu_common::set_bytes(enc, &dims, 2);
        gpu_common::set_bytes(enc, &layer_idx, 4);
        gpu_common::set_bytes(enc, &grid_info, 7);
        let count = if phase == 1 {
            1
        } else {
            f.canvas() * f.top_k as usize
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc"))?;
    let buf_expert_unique = pool
        .allocate(&ctx.device, std::mem::size_of::<u32>())
        .ok_or(Error::Format("alloc"))?;
    // 6 grids × 3 u32 (grouped 0/1, block-sparse 2/3, tunable sparse 4/5).
    let buf_indirect = pool
        .allocate(&ctx.device, 6 * 3 * std::mem::size_of::<u32>())
        .ok_or(Error::Format("alloc"))?;
    let scratch = scratch_from_fixture(f);
    BufferPool::write_bytes(&buf_route, unsafe {
        std::slice::from_raw_parts(
            &scratch as *const RouteScratch as *const u8,
            std::mem::size_of::<RouteScratch>(),
        )
    });

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    run_phases(
        &enc,
        &pipeline,
        &buf_route,
        &buf_expert_unique,
        &buf_indirect,
        f,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let out_scratch: RouteScratch =
        unsafe { std::ptr::read(buf_route.contents().as_ptr() as *const RouteScratch) };
    let slots = out_scratch.num_slots as usize;
    let n = f.n_experts as usize;
    let mut out: Vec<f32> = out_scratch.row_start[..n]
        .iter()
        .map(|&v| v as f32)
        .collect();
    out.push(out_scratch.num_slots as f32);
    out.extend(out_scratch.token_list[..slots].iter().map(|&v| v as f32));
    out.extend(out_scratch.slot_list[..slots].iter().map(|&v| v as f32));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::moe_bucket_fill::cpu,
        cpu_oracle = crate::shaders::moe_bucket_fill::cpu_oracle,
        gpu = crate::shaders::moe_bucket_fill::gpu,
        fixture = crate::shaders::moe_bucket_fill::tiny_fixture,
        out_len = crate::shaders::moe_bucket_fill::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }

    kernel_oracle_matrix! {
        mod unique,
        cpu = crate::shaders::moe_bucket_fill::cpu,
        cpu_oracle = crate::shaders::moe_bucket_fill::cpu_oracle,
        gpu = crate::shaders::moe_bucket_fill::gpu,
        fixture = crate::shaders::moe_bucket_fill::unique_fixture,
        out_len = crate::shaders::moe_bucket_fill::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
