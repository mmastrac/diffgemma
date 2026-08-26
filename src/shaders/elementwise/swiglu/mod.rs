//! SwiGLU family: gate×up with layout/dtype/gelu/in-place compile-time axes.

use crate::Error;
use crate::shaders::bf16;
use crate::shaders::cpu::{gelu_pytorch_tanh, gelu_pytorch_tanh_f32};
use crate::shaders::gpu_common;
use crate::shaders::manifest::{self, SwigluSplitVariant};
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "swiglu";
pub const MOE_ENTRY: &str = "swiglu_moe_gate_up";

pub const SHADER: &str = include_str!("swiglu.metal");

pub const MOE_SHADER: &str = include_str!("swiglu_moe_gate_up.metal");

// --- split f32 in-place (decoder swiglu_mul) ---

#[derive(Debug, Clone)]
pub struct SplitFixture {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
}

impl SplitFixture {
    pub fn len(&self) -> usize {
        self.gate.len()
    }
}

pub fn split_tiny_fixture(_: ElemFormat) -> SplitFixture {
    SplitFixture {
        gate: vec![1.0, -0.5, 2.0, 0.0],
        up: vec![2.0, 4.0, -1.0, 3.0],
    }
}

pub fn split_moe_inter_fixture(_: ElemFormat) -> SplitFixture {
    let len = 8 * 704;
    SplitFixture {
        gate: (0..len).map(|i| ((i as f32) * 0.01).sin()).collect(),
        up: (0..len).map(|i| ((i as f32) * 0.013).cos()).collect(),
    }
}

pub fn cpu_split_mul(f: &SplitFixture) -> Vec<f32> {
    let mut out = f.gate.clone();
    for (g, &u) in out.iter_mut().zip(f.up.iter()) {
        *g *= u;
    }
    out
}

// --- split half out (monolith glu) ---

#[derive(Debug, Clone)]
pub struct HalfFixture {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
}

impl HalfFixture {
    pub fn len(&self) -> usize {
        self.gate.len()
    }
}

pub fn half_tiny_fixture(_: ElemFormat) -> HalfFixture {
    HalfFixture {
        gate: vec![-1.0, 0.0, 1.0, 2.0],
        up: vec![1.0, -0.5, 0.25, 3.0],
    }
}

pub fn cpu_half_glu(f: &HalfFixture) -> Vec<f32> {
    f.gate
        .iter()
        .zip(f.up.iter())
        .map(|(&g, &u)| bf16::store_bf16_round_half(gelu_pytorch_tanh_f32(g) * u))
        .collect()
}

pub fn cpu_oracle_half_glu(f: &HalfFixture) -> Vec<f32> {
    let mut gate = f.gate.clone();
    gelu_pytorch_tanh(&mut gate);
    gate.iter()
        .zip(f.up.iter())
        .map(|(&g, &u)| bf16::store_bf16_round_half(g * u))
        .collect()
}

// --- interleaved MoE gate_up ---

#[derive(Debug, Clone)]
pub struct InterleavedFixture {
    pub gate_up: Vec<f32>,
    pub batch_size: usize,
    pub moe_inter: usize,
}

impl InterleavedFixture {
    pub fn out_len(&self) -> usize {
        self.batch_size * self.moe_inter
    }
}

pub fn interleaved_tiny_fixture(_: ElemFormat) -> InterleavedFixture {
    InterleavedFixture {
        gate_up: vec![1.0, -1.0, 2.0, 0.5, 0.0, 1.0, -0.5, 2.0],
        batch_size: 2,
        moe_inter: 2,
    }
}

pub fn interleaved_moe_fixture(_: ElemFormat) -> InterleavedFixture {
    let batch_size = 8;
    let moe_inter = 704;
    let len = batch_size * moe_inter * 2;
    InterleavedFixture {
        gate_up: (0..len).map(|i| ((i as f32) * 0.002).sin()).collect(),
        batch_size,
        moe_inter,
    }
}

pub fn cpu_interleaved(f: &InterleavedFixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    let mi = f.moe_inter;
    for b in 0..f.batch_size {
        for j in 0..mi {
            let off = b * (2 * mi) + j;
            let g = gelu_pytorch_tanh_f32(f.gate_up[off]);
            let u = f.gate_up[off + mi];
            out[b * mi + j] = bf16::round_bf16_f32(g * u);
        }
    }
    out
}

pub fn cpu_oracle_interleaved(f: &InterleavedFixture) -> Vec<f32> {
    let gate: Vec<f32> = f
        .gate_up
        .chunks(f.moe_inter * 2)
        .flat_map(|row| {
            let mi = f.moe_inter;
            let mut g: Vec<f32> = row[..mi].to_vec();
            gelu_pytorch_tanh(&mut g);
            g.into_iter()
                .zip(row[mi..].iter())
                .map(|(gv, &u)| bf16::round_bf16_f32(gv * u))
                .collect::<Vec<_>>()
        })
        .collect();
    debug_assert_eq!(gate.len(), f.out_len());
    gate
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    split: SwigluSplitVariant,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    manifest::validate_shared(ENTRY, variant)?;
    let local = manifest::swiglu_split_variant(split)?;
    manifest::assert_no_fc_collisions(ENTRY, &[4, 5, 6])?;
    let (uints, bools) = local.local_fcs();
    ctx.compile_subkernel_ex(
        SHADER,
        ENTRY,
        variant,
        &local.cache_suffix(),
        &bools,
        &uints,
    )
}

#[cfg(target_os = "macos")]
pub fn pipeline_for_moe(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    manifest::validate_shared(MOE_ENTRY, variant)?;
    ctx.compile_subkernel(MOE_SHADER, MOE_ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_split_in_place_f32(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    len: u32,
) {
    let dims = [len, 0u32];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
        enc.setBuffer_offset_atIndex(Some(gate), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 4);
    }
    gpu_common::set_bytes(enc, &dims, 3);
}

#[cfg(target_os = "macos")]
pub fn bind_split_half(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    len: u32,
) {
    let dims = [len, 0u32];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 4);
    }
    gpu_common::set_bytes(enc, &dims, 3);
}

#[cfg(target_os = "macos")]
pub fn bind_moe_gate_up(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate_up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate_up), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    gpu_common::set_bytes(enc, dims, 2);
}

#[cfg(target_os = "macos")]
fn gpu_split(
    f: &SplitFixture,
    variant: KernelVariant,
    split: SwigluSplitVariant,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLSize};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, split, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let buf_g = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_u = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_g, &f.gate);
    BufferPool::write_f32(&buf_u, &f.up);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_split_in_place_f32(&enc, &buf_g, &buf_u, &buf_dump, len as u32);
    let tg = 256usize.min(len);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: div_up(len, tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_g, &mut out);
    Ok(out)
}

#[cfg(target_os = "macos")]
pub fn gpu_split_mul(f: &SplitFixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu_split(f, variant, SwigluSplitVariant::DECODER_MUL)
}

#[cfg(target_os = "macos")]
pub fn gpu_half_glu(f: &HalfFixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, SwigluSplitVariant::MONOLITH_GLU, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let buf_g = pool
        .allocate(&ctx.device, len * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_u = pool
        .allocate(&ctx.device, len * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, len * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_g, &bf16::f32_slice_to_bf16_bits(&f.gate));
    BufferPool::write_bf16(&buf_u, &bf16::f32_slice_to_bf16_bits(&f.up));
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_split_half(enc, &buf_g, &buf_u, &buf_y, &buf_d, len as u32);
    })?;
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..len)
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(target_os = "macos")]
pub fn gpu_interleaved(f: &InterleavedFixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLSize};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for_moe(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let in_len = f.gate_up.len();
    let out_len = f.out_len();
    let buf_in = pool
        .allocate(&ctx.device, in_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        out_len * 4
    } else {
        4
    };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_in, &f.gate_up);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    let dims = [f.batch_size as u32, f.moe_inter as u32];
    bind_moe_gate_up(&enc, &buf_in, &buf_out, &buf_dump, &dims);
    let tg = 256usize.min(out_len);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: div_up(out_len, tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}

#[cfg(target_os = "macos")]
use crate::shaders::gpu_common::div_up;

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "swiglu",
    entry: "swiglu",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[(4, "K_IO_DTYPE"), (5, "K_GELU_GATE"), (6, "K_IN_PLACE")],
    variants: crate::shaders::manifest::KernelVariants::SwigluSplit {
        rows: &[
            crate::shaders::manifest::SwigluSplitVariant::DECODER_MUL,
            crate::shaders::manifest::SwigluSplitVariant::MONOLITH_GLU,
        ],
    },
};

/// Colocated subkernel registration (swiglu_moe_gate_up.metal).
pub const SPEC_MOE_GATE_UP: crate::shaders::manifest::KernelSpec =
    crate::shaders::manifest::KernelSpec {
        name: "swiglu_moe_gate_up",
        entry: "swiglu_moe_gate_up",
        quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
        fc: &[],
        variants: crate::shaders::manifest::KernelVariants::SwigluMoeGateUp,
    };

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod split_tiny,
        cpu = crate::shaders::swiglu::cpu_split_mul,
        cpu_oracle = crate::shaders::swiglu::cpu_split_mul,
        gpu = crate::shaders::swiglu::gpu_split_mul,
        fixture = crate::shaders::swiglu::split_tiny_fixture,
        out_len = crate::shaders::swiglu::SplitFixture::len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod split_moe_inter,
        cpu = crate::shaders::swiglu::cpu_split_mul,
        cpu_oracle = crate::shaders::swiglu::cpu_split_mul,
        gpu = crate::shaders::swiglu::gpu_split_mul,
        fixture = crate::shaders::swiglu::split_moe_inter_fixture,
        out_len = crate::shaders::swiglu::SplitFixture::len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod half_tiny,
        cpu = crate::shaders::swiglu::cpu_half_glu,
        cpu_oracle = crate::shaders::swiglu::cpu_oracle_half_glu,
        gpu = crate::shaders::swiglu::gpu_half_glu,
        fixture = crate::shaders::swiglu::half_tiny_fixture,
        out_len = crate::shaders::swiglu::HalfFixture::len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod interleaved_tiny,
        cpu = crate::shaders::swiglu::cpu_interleaved,
        cpu_oracle = crate::shaders::swiglu::cpu_oracle_interleaved,
        gpu = crate::shaders::swiglu::gpu_interleaved,
        fixture = crate::shaders::swiglu::interleaved_tiny_fixture,
        out_len = crate::shaders::swiglu::InterleavedFixture::out_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod interleaved_moe,
        cpu = crate::shaders::swiglu::cpu_interleaved,
        cpu_oracle = crate::shaders::swiglu::cpu_oracle_interleaved,
        gpu = crate::shaders::swiglu::gpu_interleaved,
        fixture = crate::shaders::swiglu::interleaved_moe_fixture,
        out_len = crate::shaders::swiglu::InterleavedFixture::out_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}

crate::register_kernel_specs!(SPEC, SPEC_MOE_GATE_UP);
