//! Gather rows by index from a row-major `[tokens, hidden]` source.

use crate::Error;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "gather_rows";

pub const SHADER: &str = include_str!("gather_rows.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub src: Vec<f32>,
    pub indices: Vec<u32>,
    pub hidden: usize,
    pub num_tokens: usize,
}

impl Fixture {
    pub fn batch_size(&self) -> usize {
        self.indices.len()
    }

    pub fn out_len(&self) -> usize {
        self.batch_size() * self.hidden
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        src: vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        indices: vec![2, 0],
        hidden: 4,
        num_tokens: 3,
    }
}

pub fn moe_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64;
    let num_tokens = 16;
    let src: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let indices: Vec<u32> = (0..8).map(|i| (i * 2) as u32).collect();
    Fixture {
        src,
        indices,
        hidden,
        num_tokens,
    }
}

/// 32-slot gather order mimicking batched MoE bucket fill (non-sequential token indices).
pub fn moe_routing_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64;
    let num_tokens = 256;
    let src: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
        .collect();
    let indices: Vec<u32> = vec![
        103, 87, 44, 12, 201, 155, 3, 98, //
        17, 240, 66, 129, 8, 190, 51, 222, //
        74, 11, 183, 145, 28, 99, 167, 4, //
        131, 58, 210, 37, 172, 95, 21, 248,
    ];
    Fixture {
        src,
        indices,
        hidden,
        num_tokens,
    }
}

/// 64-slot gather order from Calgary L0 batched MoE capture (seed 42).
pub fn moe_batched_pin_l0_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64;
    let num_tokens = crate::metal::CANVAS;
    let token_list = crate::shaders::moe_batched_pin::calgary_l0_token_list();
    let src: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
        .collect();
    Fixture {
        src,
        indices: token_list.to_vec(),
        hidden,
        num_tokens,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    for (bi, &tok) in f.indices.iter().enumerate() {
        let src_off = tok as usize * f.hidden;
        let dst_off = bi * f.hidden;
        out[dst_off..dst_off + f.hidden].copy_from_slice(&f.src[src_off..src_off + f.hidden]);
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

/// Format-specialized pipeline. `src_f32`/`dst_f32` pick f32 (4-byte) vs
/// activation-arena (2-byte bf16/fp16) for the source and destination buffers
/// (function constants 4/5). The three legacy entry points map to:
///   (true,  true)  — the old `gather_rows`             (f32 -> f32)
///   (false, false) — the old `gather_rows_bf16`        (arena -> arena)
///   (false, true)  — the old `gather_rows_bf16_to_f32` (arena -> f32)
#[cfg(target_os = "macos")]
pub fn pipeline_for_fmt(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    src_f32: bool,
    dst_f32: bool,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    use crate::shaders::variant::FcBool;
    let bools = [
        FcBool {
            index: 4,
            value: src_f32,
        },
        FcBool {
            index: 5,
            value: dst_f32,
        },
    ];
    let label = match (src_f32, dst_f32) {
        (true, true) => "f32",
        (false, true) => "arena2f32",
        (false, false) => "arena",
        (true, false) => "f322arena",
    };
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, label, &bools, &[])
}

/// f32 -> f32 gather (the oracle/batched-prefill default).
#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    pipeline_for_fmt(ctx, variant, true, true)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src: &ProtocolObject<dyn MTLBuffer>,
    indices: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    hidden: u32,
    batch_size: u32,
    elem_base: u32,
) {
    let dims = [0u32, hidden];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(indices), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    gpu_common::set_bytes(enc, &dims, 3);
    gpu_common::set_bytes(enc, &batch_size, 4);
    gpu_common::set_bytes(enc, &elem_base, 6);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let out_len = f.out_len();
    let grid = out_len;
    let buf_src = pool
        .allocate(&ctx.device, f.src.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_idx = pool
        .allocate(&ctx.device, f.indices.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_dst = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        out_len * 4
    } else {
        4
    };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_src, &f.src);
    let idx_bytes =
        unsafe { std::slice::from_raw_parts(f.indices.as_ptr().cast::<u8>(), f.indices.len() * 4) };
    BufferPool::write_bytes(&buf_idx, idx_bytes);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, grid, |enc| {
        bind_gpu_buffers(
            enc,
            &buf_src,
            &buf_idx,
            &buf_dst,
            &buf_d,
            f.hidden as u32,
            f.batch_size() as u32,
            0,
        );
    })?;
    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_dst, &mut out);
    Ok(out)
}

/// Run the merged kernel for a given (src_f32, dst_f32) specialization on a
/// fixture whose values are bf16-exact, returning the gathered output widened
/// to f32. Used by the format-agreement test to cover the arena specializations
/// (the old gather_rows_bf16 / gather_rows_bf16_to_f32) the f32 oracle can't.
#[cfg(all(test, target_os = "macos"))]
fn run_fmt(f: &Fixture, src_f32: bool, dst_f32: bool) -> Vec<f32> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new().expect("metal");
    let pipeline =
        pipeline_for_fmt(&ctx, KernelVariant::PRODUCTION, src_f32, dst_f32).expect("pipe");
    let mut pool = BufferPool::new();
    let out_len = f.out_len();

    // bf16 bits = high 16 of the f32 (exact for bf16-representable values).
    let src_bytes: Vec<u8> = if src_f32 {
        f.src
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect()
    } else {
        f.src
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let buf_src = pool
        .allocate(&ctx.device, src_bytes.len())
        .expect("src alloc");
    BufferPool::write_bytes(&buf_src, &src_bytes);

    let idx_bytes =
        unsafe { std::slice::from_raw_parts(f.indices.as_ptr().cast::<u8>(), f.indices.len() * 4) };
    let buf_idx = pool.allocate(&ctx.device, idx_bytes.len()).expect("idx");
    BufferPool::write_bytes(&buf_idx, idx_bytes);

    let dst_elem = if dst_f32 { 4 } else { 2 };
    let buf_dst = pool
        .allocate(&ctx.device, out_len * dst_elem)
        .expect("dst alloc");
    let buf_dump = pool.allocate(&ctx.device, 4).expect("dump");

    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, out_len, |enc| {
        bind_gpu_buffers(
            enc,
            &buf_src,
            &buf_idx,
            &buf_dst,
            &buf_dump,
            f.hidden as u32,
            f.batch_size() as u32,
            0,
        );
    })
    .expect("dispatch");

    if dst_f32 {
        let mut out = vec![0.0f32; out_len];
        BufferPool::read_f32(&buf_dst, &mut out);
        out
    } else {
        let ptr = buf_dst.contents().as_ptr() as *const u16;
        (0..out_len)
            .map(|i| f32::from_bits((unsafe { *ptr.add(i) } as u32) << 16))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    /// All three production specializations of the merged kernel — f32->f32,
    /// arena->arena (the ex-`gather_rows_bf16`, previously untested), and
    /// arena->f32 (ex-`gather_rows_bf16_to_f32`) — must agree bit-for-bit with
    /// the CPU gather on bf16-exact input.
    #[cfg(target_os = "macos")]
    #[test]
    fn format_specializations_agree() {
        let f = tiny_fixture(ElemFormat::F32);
        let expect = cpu(&f);
        for (sf, df, label) in [
            (true, true, "f32->f32"),
            (false, false, "arena->arena"),
            (false, true, "arena->f32"),
        ] {
            let got = run_fmt(&f, sf, df);
            for (i, (a, e)) in got.iter().zip(&expect).enumerate() {
                assert!(
                    a.to_bits() == e.to_bits(),
                    "{label}: mismatch at {i}: got {a} want {e}"
                );
            }
        }
    }

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::gather_rows::cpu,
        cpu_oracle = crate::shaders::gather_rows::cpu_oracle,
        gpu = crate::shaders::gather_rows::gpu,
        fixture = crate::shaders::gather_rows::tiny_fixture,
        out_len = crate::shaders::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe,
        cpu = crate::shaders::gather_rows::cpu,
        cpu_oracle = crate::shaders::gather_rows::cpu_oracle,
        gpu = crate::shaders::gather_rows::gpu,
        fixture = crate::shaders::gather_rows::moe_fixture,
        out_len = crate::shaders::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe_routing,
        cpu = crate::shaders::gather_rows::cpu,
        cpu_oracle = crate::shaders::gather_rows::cpu_oracle,
        gpu = crate::shaders::gather_rows::gpu,
        fixture = crate::shaders::gather_rows::moe_routing_fixture,
        out_len = crate::shaders::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe_batched_pin_l0,
        cpu = crate::shaders::gather_rows::cpu,
        cpu_oracle = crate::shaders::gather_rows::cpu_oracle,
        gpu = crate::shaders::gather_rows::gpu,
        fixture = crate::shaders::gather_rows::moe_batched_pin_l0_fixture,
        out_len = crate::shaders::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }
}
