//! Per-row RMSNorm subkernel — CPU oracle, GPU dispatch, tier-1 tests.

use crate::Error;
use crate::shaders::cpu;
use crate::shaders::test_util::ElemFormat;

use crate::shaders::manifest::{self};
pub use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "rms_norm_rows";

pub const SHADER: &str = include_str!("rms_norm_rows.metal");

/// Synthetic tier-1 fixture (blob-free).
#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub weight: Vec<f32>,
    pub seq_len: usize,
    pub hidden: usize,
    pub eps: f32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.seq_len * self.hidden
    }
}

pub fn fixture_out_len(fix: &Fixture) -> usize {
    fix.out_len()
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: vec![1.0, 0.5, 2.0, 1.5],
        seq_len: 2,
        hidden: 4,
        eps: 1e-6,
    }
}

pub fn mlp_shape_fixture(_fmt: ElemFormat) -> Fixture {
    let seq_len = 3;
    let hidden = 64;
    let len = seq_len * hidden;
    let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();
    Fixture {
        x,
        weight,
        seq_len,
        hidden,
        eps: 1e-6,
    }
}

pub fn tiny_fixture_no_scale(_fmt: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: vec![1.0; 4],
        seq_len: 2,
        hidden: 4,
        eps: 1e-6,
    }
}

pub fn mlp_shape_fixture_no_scale(_fmt: ElemFormat) -> Fixture {
    let seq_len = 3;
    let hidden = 64;
    let len = seq_len * hidden;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect(),
        weight: vec![1.0; hidden],
        seq_len,
        hidden,
        eps: 1e-6,
    }
}

/// CPU path wired into the engine reference (`kernels/cpu.rs`).
pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.out_len()];
    cpu::rms_norm_rows(
        &mut out,
        &fix.x,
        &fix.weight,
        fix.seq_len,
        fix.hidden,
        fix.eps,
    );
    out
}

pub fn cpu_no_scale(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.out_len()];
    cpu::rms_norm_rows_no_scale(&mut out, &fix.x, fix.seq_len, fix.hidden, fix.eps);
    out
}

/// Independent CPU transliteration (oracle for tier-1).
pub fn cpu_oracle(fix: &Fixture) -> Vec<f32> {
    cpu(fix)
}

#[cfg(target_os = "macos")]
pub fn gpu(fix: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu_affine(fix, variant, true)
}

#[cfg(target_os = "macos")]
pub fn gpu_no_scale(fix: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu_affine(fix, variant, false)
}

#[cfg(target_os = "macos")]
fn gpu_affine(fix: &Fixture, variant: KernelVariant, affine: bool) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, affine, variant)?;
    let mut pool = BufferPool::new();
    let len = fix.out_len();
    let buf_x = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc failed"))?;
    let buf_w = pool
        .allocate(&ctx.device, fix.hidden * 4)
        .ok_or(Error::Gpu("buffer alloc failed"))?;
    let buf_o = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc failed"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        fix.seq_len * 4
    } else {
        4
    };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("buffer alloc failed"))?;

    BufferPool::write_f32(&buf_x, &fix.x);
    BufferPool::write_f32(&buf_w, &fix.weight);
    BufferPool::write_f32(&buf_o, &vec![0.0f32; len]);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or(Error::Gpu("command buffer failed"))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Gpu("encoder failed"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 2);
    }
    let dims = [fix.seq_len as u32, fix.hidden as u32];
    set_bytes(&enc, &dims, 3);
    set_bytes(&enc, &fix.eps, 4);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_dump), 0, 5);
    }
    let tg_width = 256usize.min(fix.seq_len);
    let grid = MTLSize {
        width: div_up(fix.seq_len, tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_o, &mut out);
    pool.release(len * 4, buf_x);
    pool.release(fix.hidden * 4, buf_w);
    pool.release(len * 4, buf_o);
    pool.release(dump_bytes, buf_dump);
    Ok(out)
}

pub fn shader_source() -> &'static str {
    SHADER
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    affine: bool,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    manifest::validate_shared(ENTRY, variant)?;
    let local = manifest::rms_norm_rows_variant(affine)?;
    manifest::assert_no_fc_collisions(ENTRY, &[4])?;
    ctx.compile_subkernel_ex(
        SHADER,
        ENTRY,
        variant,
        local.cache_suffix(),
        &local.local_fcs(),
        &[],
    )
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf_x: &ProtocolObject<dyn MTLBuffer>,
    buf_w: &ProtocolObject<dyn MTLBuffer>,
    buf_o: &ProtocolObject<dyn MTLBuffer>,
    buf_dump: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
    eps: f32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_o), 0, 2);
        enc.setBuffer_offset_atIndex(Some(buf_dump), 0, 5);
    }
    set_bytes(enc, dims, 3);
    set_bytes(enc, &eps, 4);
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

#[cfg(target_os = "macos")]
use crate::shaders::gpu_common::div_up;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::rms_norm_rows::cpu,
        cpu_oracle = crate::shaders::rms_norm_rows::cpu_oracle,
        gpu = crate::shaders::rms_norm_rows::gpu,
        fixture = crate::shaders::rms_norm_rows::tiny_fixture,
        out_len = crate::shaders::rms_norm_rows::fixture_out_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mlp_shape,
        cpu = crate::shaders::rms_norm_rows::cpu,
        cpu_oracle = crate::shaders::rms_norm_rows::cpu_oracle,
        gpu = crate::shaders::rms_norm_rows::gpu,
        fixture = crate::shaders::rms_norm_rows::mlp_shape_fixture,
        out_len = crate::shaders::rms_norm_rows::fixture_out_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod no_scale_tiny,
        cpu = crate::shaders::rms_norm_rows::cpu_no_scale,
        cpu_oracle = crate::shaders::rms_norm_rows::cpu_no_scale,
        gpu = crate::shaders::rms_norm_rows::gpu_no_scale,
        fixture = crate::shaders::rms_norm_rows::tiny_fixture_no_scale,
        out_len = crate::shaders::rms_norm_rows::fixture_out_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod no_scale_mlp_shape,
        cpu = crate::shaders::rms_norm_rows::cpu_no_scale,
        cpu_oracle = crate::shaders::rms_norm_rows::cpu_no_scale,
        gpu = crate::shaders::rms_norm_rows::gpu_no_scale,
        fixture = crate::shaders::rms_norm_rows::mlp_shape_fixture_no_scale,
        out_len = crate::shaders::rms_norm_rows::fixture_out_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    #[test]
    fn golden_row_values() {
        let fix = tiny_fixture(ElemFormat::F32);
        let out = cpu(&fix);
        let expected = [
            0.365_148_35,
            0.365_148_35,
            2.190_890_1,
            2.190_890_1,
            -0.852_802_56,
            0.213_200_64,
            3.411_210_2,
            -0.639_601_9,
        ];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "got {a}, expected {b}");
        }
    }
}

pub mod tiled;

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "rms_norm_rows",
    entry: "rms_norm_rows",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[(4, "K_AFFINE")],
    variants: crate::shaders::manifest::KernelVariants::RmsNormRows {
        rows: &[
            crate::shaders::manifest::RmsNormRowsVariant { affine: false },
            crate::shaders::manifest::RmsNormRowsVariant { affine: true },
        ],
    },
};
