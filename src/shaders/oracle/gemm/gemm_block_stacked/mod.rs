//! Stacked-segment Q4 GEMM: one input tile load, multiple weight/output segments (QKV, gate+up).
//!
//! The GPU kernel (`oracle/gemm/gemm_block_stacked/gemm_block_stacked.metal`) is a validation
//! ORACLE — production stacked GEMM runs `gemm_tunable`. The segment types/helpers
//! here (`GemmStackedSeg`, `StackedSegFc`, `StackedPipelineKey`) are PRODUCTION:
//! `gemm_tunable::stacked_pipeline_for` builds on them.

use crate::Error;
use crate::dgq::block::{q4_gemm_cpu, quantize_row_q4};
use crate::dgq::layout::{q4_matrix_bytes, q4_row_bytes};
use crate::shaders::QuantFormat;
use crate::shaders::bf16;
use crate::shaders::gemm_common;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;

pub const ENTRY: &str = "gemm_block_stacked";

pub const SHADER: &str = include_str!("gemm_block_stacked.metal");

/// Must match `shaders/include/gemm_stacked.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct GemmStackedSeg {
    pub n_cols: u32,
    pub y_col0: u32,
    pub y_row_cols: u32,
    pub _pad: u32,
    pub w_off: u64,
    pub y_byte_off: u64,
}

/// Function-constant segment table (FC12–27) for `gemm_block_stacked`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct StackedSegFc {
    pub n_segs: u32,
    pub end0: u32,
    pub end1: u32,
    pub end2: u32,
    pub w_off: [u64; 3],
    pub y_byte_off: [u64; 3],
    pub y_col0: [u32; 3],
    pub y_row_cols: [u32; 3],
}

impl StackedSegFc {
    pub fn from_segments(segs: &[GemmStackedSeg]) -> Result<Self, Error> {
        if segs.is_empty() || segs.len() > 3 {
            return Err(Error::Runtime("stacked GEMM supports 1..=3 segments"));
        }
        let mut w_off = [0u64; 3];
        let mut y_byte_off = [0u64; 3];
        let mut y_col0 = [0u32; 3];
        let mut y_row_cols = [0u32; 3];
        let mut end0 = 0u32;
        let mut end1 = 0u32;
        let mut end2 = 0u32;
        for (i, s) in segs.iter().enumerate() {
            w_off[i] = s.w_off;
            y_byte_off[i] = s.y_byte_off;
            y_col0[i] = s.y_col0;
            y_row_cols[i] = s.y_row_cols;
        }
        if segs.len() >= 1 {
            end0 = segs[0].n_cols;
        }
        if segs.len() >= 2 {
            end1 = end0 + segs[1].n_cols;
        }
        if segs.len() >= 3 {
            end2 = end1 + segs[2].n_cols;
        }
        Ok(Self {
            n_segs: segs.len() as u32,
            end0,
            end1,
            end2,
            w_off,
            y_byte_off,
            y_col0,
            y_row_cols,
        })
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct StackedPipelineKey {
    n: u32,
    k: u32,
    format: u32,
    stacked: StackedSegFc,
    /// fp16 activation arena (FC9) — the compiled body differs (E11).
    arena_f16: bool,
}

impl StackedPipelineKey {
    pub(crate) fn new(n: u32, k: u32, format: QuantFormat, stacked: StackedSegFc) -> Self {
        Self {
            n,
            k,
            format: format as u32,
            arena_f16: crate::shaders::variant::arena_f16_compile_enabled(),
            stacked,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SegFixture {
    pub w_f32: Vec<f32>,
    pub n: usize,
    pub y_col0: usize,
    pub y_row_cols: usize,
    pub y_byte_off: usize,
    /// Optional padding after this segment's weight blob (simulates sparse `.dgq` layout).
    pub w_blob_pad_after: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct StackedFixture {
    pub x: Vec<f32>,
    pub segments: Vec<SegFixture>,
    pub m: usize,
    pub k: usize,
}

impl StackedFixture {
    pub fn n_total(&self) -> usize {
        self.segments.iter().map(|s| s.n).sum()
    }

    pub fn y_bytes(&self) -> usize {
        let mut max_end = 0usize;
        for s in &self.segments {
            let end = s.y_byte_off + s.y_row_cols * self.m * 2;
            max_end = max_end.max(end);
        }
        max_end
    }

    pub fn w_blob(&self) -> Vec<u8> {
        let mut blob = Vec::new();
        for s in &self.segments {
            let bytes = quantize_f32_matrix_q4(&s.w_f32, s.n, self.k);
            blob.extend_from_slice(&bytes);
            if let Some(pad) = s.w_blob_pad_after {
                blob.extend(std::iter::repeat_n(0u8, pad));
            }
        }
        blob
    }

    pub fn gpu_segments(&self) -> Vec<GemmStackedSeg> {
        let mut w_off = 0u64;
        self.segments
            .iter()
            .map(|s| {
                let seg = GemmStackedSeg {
                    n_cols: s.n as u32,
                    y_col0: s.y_col0 as u32,
                    y_row_cols: s.y_row_cols as u32,
                    _pad: 0,
                    w_off,
                    y_byte_off: s.y_byte_off as u64,
                };
                w_off += q4_matrix_bytes(s.n, self.k) as u64;
                if let Some(pad) = s.w_blob_pad_after {
                    w_off += pad as u64;
                }
                seg
            })
            .collect()
    }
}

fn quantize_f32_matrix_q4(rows: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let mut dst = vec![0u8; q4_matrix_bytes(out_dim, in_dim)];
    let row_bytes = q4_row_bytes(in_dim);
    for row in 0..out_dim {
        let off = row * in_dim;
        quantize_row_q4(
            &rows[off..off + in_dim],
            in_dim,
            &mut dst[row * row_bytes..],
        );
    }
    dst
}

pub fn gate_up_tiny_fixture(_: ElemFormat) -> StackedFixture {
    let m = 4usize;
    let k = 64usize;
    let n = 32usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let w_gate: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.02)
        .collect();
    let w_up: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.009).sin() * 0.03)
        .collect();
    let plane_bytes = m * n * 2;
    StackedFixture {
        x,
        segments: vec![
            SegFixture {
                w_f32: w_gate,
                n,
                y_col0: 0,
                y_row_cols: n,
                y_byte_off: 0,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: w_up,
                n,
                y_col0: 0,
                y_row_cols: n,
                y_byte_off: plane_bytes,
                w_blob_pad_after: None,
            },
        ],
        m,
        k,
    }
}

/// Gate‖up with non-contiguous weight blobs (mirrors `.dgq` manifest offsets).
pub fn gate_up_padded_blob_fixture(_: ElemFormat) -> StackedFixture {
    let mut f = gate_up_tiny_fixture(ElemFormat::F32);
    f.segments[0].w_blob_pad_after = Some(8192);
    f
}

pub fn qkv_tiny_fixture(_: ElemFormat) -> StackedFixture {
    let m = 4usize;
    let k = 64usize;
    let nq = 48usize;
    let nk = 32usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.011).sin() * 0.2)
        .collect();
    let mk = |seed: f32| {
        (0..nk * k)
            .map(|i| ((i as f32) * 0.005 + seed).cos() * 0.02)
            .collect()
    };
    let mq = (0..nq * k)
        .map(|i| ((i as f32) * 0.006).sin() * 0.02)
        .collect();
    StackedFixture {
        x,
        segments: vec![
            SegFixture {
                w_f32: mq,
                n: nq,
                y_col0: 0,
                y_row_cols: nq,
                y_byte_off: 0,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: mk(0.1),
                n: nk,
                y_col0: 0,
                y_row_cols: nk,
                y_byte_off: nq * m * 2,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: mk(0.2),
                n: nk,
                y_col0: 0,
                y_row_cols: nk,
                y_byte_off: nq * m * 2 + nk * m * 2,
                w_blob_pad_after: None,
            },
        ],
        m,
        k,
    }
}

pub fn qkv_qk_tiny_fixture(_: ElemFormat) -> StackedFixture {
    let m = 4usize;
    let k = 64usize;
    let nq = 48usize;
    let nk = 16usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.011).sin() * 0.2)
        .collect();
    let mq = (0..nq * k)
        .map(|i| ((i as f32) * 0.006).sin() * 0.02)
        .collect();
    let mk = (0..nk * k)
        .map(|i| ((i as f32) * 0.005 + 0.1).cos() * 0.02)
        .collect();
    StackedFixture {
        x,
        segments: vec![
            SegFixture {
                w_f32: mq,
                n: nq,
                y_col0: 0,
                y_row_cols: nq,
                y_byte_off: 0,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: mk,
                n: nk,
                y_col0: 0,
                y_row_cols: nk,
                y_byte_off: nq * m * 2,
                w_blob_pad_after: None,
            },
        ],
        m,
        k,
    }
}

pub fn fixture_len(f: &StackedFixture) -> usize {
    f.y_bytes() / 2
}

pub fn cpu(f: &StackedFixture) -> Vec<f32> {
    let mut y = vec![0.0f32; f.y_bytes() / 2];
    let blob = f.w_blob();
    let mut w_off = 0usize;
    for s in &f.segments {
        let w_q4 = &blob[w_off..w_off + q4_matrix_bytes(s.n, f.k)];
        let mut seg_out = vec![0.0f32; f.m * s.n];
        q4_gemm_cpu(&f.x, f.m, f.k, w_q4, s.n, &mut seg_out);
        let base_half = s.y_byte_off / 2;
        for row in 0..f.m {
            for col in 0..s.n {
                let dst = base_half + row * s.y_row_cols + s.y_col0 + col;
                y[dst] = bf16::store_bf16_round_half(seg_out[row * s.n + col]);
            }
        }
        w_off += q4_matrix_bytes(s.n, f.k);
        if let Some(pad) = s.w_blob_pad_after {
            w_off += pad;
        }
    }
    y
}

pub fn cpu_oracle(f: &StackedFixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
    segs: &[GemmStackedSeg],
) -> Result<crate::metal::device::ComputePipeline, Error> {
    Ok((*stacked_pipeline_for(ctx, n, k, format, segs)?).clone())
}

#[cfg(target_os = "macos")]
pub fn stacked_pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
    segs: &[GemmStackedSeg],
) -> Result<std::sync::Arc<crate::metal::device::ComputePipeline>, Error> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<
        Mutex<HashMap<StackedPipelineKey, std::sync::Arc<crate::metal::device::ComputePipeline>>>,
    > = OnceLock::new();
    let stacked = StackedSegFc::from_segments(segs)?;
    let key = StackedPipelineKey::new(n, k, format, stacked);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| Error::Gpu("stacked pipeline cache poisoned"))?;
    if let Some(pipe) = guard.get(&key) {
        return Ok(std::sync::Arc::clone(pipe));
    }
    let pipe = std::sync::Arc::new(ctx.compile_gemm_stacked_subkernel(
        SHADER,
        ENTRY,
        n,
        k,
        format as u32,
        &stacked,
    )?);
    guard.insert(key, std::sync::Arc::clone(&pipe));
    Ok(pipe)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    blob: &ProtocolObject<dyn MTLBuffer>,
    m: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(y), 0, 1);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 2);
    }
    gpu_common::set_bytes(enc, &m, 3);
}

#[cfg(target_os = "macos")]
pub fn gpu_q4(
    f: &StackedFixture,
    _variant: crate::shaders::variant::KernelVariant,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let n = f.n_total() as u32;
    let segs = f.gpu_segments();
    let pipeline = pipeline_for(&ctx, n, f.k as u32, QuantFormat::Q4Affine, &segs)?;
    let mut pool = BufferPool::new();
    let w_q4 = f.w_blob();
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.y_bytes())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_q4.len())
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bytes(&buf_w, &w_q4);
    let (grid, tg) = gemm_common::dispatch_shape(f.m, f.n_total());
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, f.m as u32);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..f.y_bytes() / 2)
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(target_os = "macos")]
pub fn gpu_q4_split(f: &StackedFixture) -> Result<Vec<f32>, Error> {
    use super::gemm_q4;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let blob = f.w_blob();
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.y_bytes())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bytes(&buf_w, &blob);
    let segs = f.gpu_segments();
    for (s, seg) in f.segments.iter().zip(segs.iter()) {
        let pipeline = gemm_q4::pipeline_for(&ctx, s.n as u32, f.k as u32)?;
        let (grid, tg) = gemm_common::dispatch_shape(f.m, s.n);
        let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
        enc.setComputePipelineState(&pipeline.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_y), seg.y_byte_off as usize, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_w), 0, 2);
        }
        gpu_common::set_bytes(&enc, &seg.w_off, 3);
        let m = f.m as u32;
        gpu_common::set_bytes(&enc, &m, 4);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..f.y_bytes() / 2)
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

/// Production gate‖up shape at canvas width (separate `ffg`/`ffu` planes).
pub fn gate_up_canvas_fixture(_: ElemFormat) -> StackedFixture {
    let m = 256usize;
    let k = 2816usize;
    let n = 2112usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.0007).sin() * 0.15)
        .collect();
    let w_gate: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.0003).cos() * 0.02)
        .collect();
    let w_up: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.0004).sin() * 0.025)
        .collect();
    let plane_bytes = m * n * 2;
    StackedFixture {
        x,
        segments: vec![
            SegFixture {
                w_f32: w_gate,
                n,
                y_col0: 0,
                y_row_cols: n,
                y_byte_off: 0,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: w_up,
                n,
                y_col0: 0,
                y_row_cols: n,
                y_byte_off: plane_bytes,
                w_blob_pad_after: None,
            },
        ],
        m,
        k,
    }
}

/// Production gate‖up shape (separate `ffg`/`ffu` planes).
pub fn gate_up_prod_fixture(_: ElemFormat) -> StackedFixture {
    let m = 4usize;
    let k = 2816usize;
    let n = 2112usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.0007).sin() * 0.15)
        .collect();
    let w_gate: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.0003).cos() * 0.02)
        .collect();
    let w_up: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.0004).sin() * 0.025)
        .collect();
    let plane_bytes = m * n * 2;
    StackedFixture {
        x,
        segments: vec![
            SegFixture {
                w_f32: w_gate,
                n,
                y_col0: 0,
                y_row_cols: n,
                y_byte_off: 0,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: w_up,
                n,
                y_col0: 0,
                y_row_cols: n,
                y_byte_off: plane_bytes,
                w_blob_pad_after: None,
            },
        ],
        m,
        k,
    }
}

/// Sliding-layer Q‖K‖V shape (separate output planes).
pub fn qkv_sliding_prod_fixture(_: ElemFormat) -> StackedFixture {
    let m = 4usize;
    let k = 2816usize;
    let nq = 4096usize;
    let nk = 2048usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.0006).sin() * 0.12)
        .collect();
    let mk = |seed: f32| {
        (0..nk * k)
            .map(|i| ((i as f32) * 0.0002 + seed).cos() * 0.018)
            .collect()
    };
    let mq = (0..nq * k)
        .map(|i| ((i as f32) * 0.00025).sin() * 0.02)
        .collect();
    StackedFixture {
        x,
        segments: vec![
            SegFixture {
                w_f32: mq,
                n: nq,
                y_col0: 0,
                y_row_cols: nq,
                y_byte_off: 0,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: mk(0.1),
                n: nk,
                y_col0: 0,
                y_row_cols: nk,
                y_byte_off: nq * m * 2,
                w_blob_pad_after: None,
            },
            SegFixture {
                w_f32: mk(0.2),
                n: nk,
                y_col0: 0,
                y_row_cols: nk,
                y_byte_off: nq * m * 2 + nk * m * 2,
                w_blob_pad_after: None,
            },
        ],
        m,
        k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod gate_up_tiny,
        cpu = crate::shaders::gemm_block_stacked::cpu,
        cpu_oracle = crate::shaders::gemm_block_stacked::cpu_oracle,
        gpu = crate::shaders::gemm_block_stacked::gpu_q4,
        fixture = crate::shaders::gemm_block_stacked::gate_up_tiny_fixture,
        out_len = crate::shaders::gemm_block_stacked::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod qkv_tiny,
        cpu = crate::shaders::gemm_block_stacked::cpu,
        cpu_oracle = crate::shaders::gemm_block_stacked::cpu_oracle,
        gpu = crate::shaders::gemm_block_stacked::gpu_q4,
        fixture = crate::shaders::gemm_block_stacked::qkv_tiny_fixture,
        out_len = crate::shaders::gemm_block_stacked::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod qkv_qk_tiny,
        cpu = crate::shaders::gemm_block_stacked::cpu,
        cpu_oracle = crate::shaders::gemm_block_stacked::cpu_oracle,
        gpu = crate::shaders::gemm_block_stacked::gpu_q4,
        fixture = crate::shaders::gemm_block_stacked::qkv_qk_tiny_fixture,
        out_len = crate::shaders::gemm_block_stacked::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod gate_up_padded_blob,
        cpu = crate::shaders::gemm_block_stacked::cpu,
        cpu_oracle = crate::shaders::gemm_block_stacked::cpu_oracle,
        gpu = crate::shaders::gemm_block_stacked::gpu_q4,
        fixture = crate::shaders::gemm_block_stacked::gate_up_padded_blob_fixture,
        out_len = crate::shaders::gemm_block_stacked::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod gate_up_prod,
        cpu = crate::shaders::gemm_block_stacked::cpu,
        cpu_oracle = crate::shaders::gemm_block_stacked::cpu_oracle,
        gpu = crate::shaders::gemm_block_stacked::gpu_q4,
        fixture = crate::shaders::gemm_block_stacked::gate_up_prod_fixture,
        out_len = crate::shaders::gemm_block_stacked::fixture_len,
        formats: [F32],
        // Largest output element is ~8; one bf16 ULP (value/128) is ~0.0625, just over a
        // 0.05 abs tol. cos stays the correctness guard (stacked-vs-split parity is exact 0.0).
        max_tol = 0.1,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod qkv_sliding_prod,
        cpu = crate::shaders::gemm_block_stacked::cpu,
        cpu_oracle = crate::shaders::gemm_block_stacked::cpu_oracle,
        gpu = crate::shaders::gemm_block_stacked::gpu_q4,
        fixture = crate::shaders::gemm_block_stacked::qkv_sliding_prod_fixture,
        out_len = crate::shaders::gemm_block_stacked::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stacked_gpu_matches_split_gate_up_canvas() {
        use crate::shaders::gemm_q4;
        use crate::shaders::variant::KernelVariant;
        let f = gate_up_canvas_fixture(ElemFormat::F32);
        let stacked = gpu_q4(&f, KernelVariant::PRODUCTION).expect("stacked");
        let split = gpu_q4_split(&f).expect("split");
        let max = max_abs_diff(&stacked, &split);
        assert!(max < 0.05, "canvas stacked vs split max_abs={max}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stacked_gpu_matches_split_gate_up_tiny() {
        use crate::shaders::gemm_q4;
        use crate::shaders::variant::KernelVariant;
        let f = gate_up_tiny_fixture(ElemFormat::F32);
        let stacked = gpu_q4(&f, KernelVariant::PRODUCTION).expect("stacked");
        let split = gpu_q4_split(&f).expect("split");
        let max = max_abs_diff(&stacked, &split);
        assert!(max < 0.05, "stacked vs split max_abs={max}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stacked_gpu_matches_split_qkv_sliding_prod() {
        use crate::shaders::gemm_q4;
        use crate::shaders::variant::KernelVariant;
        let f = qkv_sliding_prod_fixture(ElemFormat::F32);
        let stacked = gpu_q4(&f, KernelVariant::PRODUCTION).expect("stacked");
        let split = gpu_q4_split(&f).expect("split");
        let max = max_abs_diff(&stacked, &split);
        assert!(max < 0.05, "stacked vs split max_abs={max}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stacked_gpu_matches_split_on_dgq_gate_up() {
        use crate::metal::DgqGpuBlob;
        use crate::metal::build_offsets_from_store;
        use crate::shaders::gemm_q4;
        use objc2::runtime::ProtocolObject;
        use std::path::Path;

        let dir = ["model/diffusiongemma-q4emb", "/tmp/quantized-weights"]
            .into_iter()
            .map(Path::new)
            .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip stacked_gpu_matches_split_on_dgq_gate_up: no .dgq weights");
            return;
        };
        let store = crate::dgq::store::DgqStore::open(dir).expect("dgq open");
        let offsets = build_offsets_from_store(&store);
        let layer = 0usize;
        let gate_off = *offsets
            .get(&format!(
                "model.decoder.layers.{layer}.mlp.gate_proj.weight"
            ))
            .expect("gate");
        let up_off = *offsets
            .get(&format!("model.decoder.layers.{layer}.mlp.up_proj.weight"))
            .expect("up");

        let m = 4usize;
        let k = 2816usize;
        let n = 2112usize;
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 + 17.0) * 0.0009).sin() * 0.11)
            .collect();
        let plane_bytes = m * n * 2;
        let segs = [
            GemmStackedSeg {
                n_cols: n as u32,
                y_col0: 0,
                y_row_cols: n as u32,
                _pad: 0,
                w_off: gate_off,
                y_byte_off: 0,
            },
            GemmStackedSeg {
                n_cols: n as u32,
                y_col0: 0,
                y_row_cols: n as u32,
                _pad: 0,
                w_off: up_off,
                y_byte_off: plane_bytes as u64,
            },
        ];

        use crate::metal::buffer::BufferPool;
        use crate::metal::device::MetalContext;
        let ctx = MetalContext::new().expect("metal");
        let mut pool = BufferPool::new();
        let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let buf_x = pool.allocate(&ctx.device, m * k * 2).expect("x");
        let buf_y_stacked = pool
            .allocate(&ctx.device, plane_bytes * 2)
            .expect("y stacked");
        let buf_y_split = pool
            .allocate(&ctx.device, plane_bytes * 2)
            .expect("y split");
        BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&x));

        let n_total = (n * 2) as u32;
        let pipeline =
            pipeline_for(&ctx, n_total, k as u32, QuantFormat::Q4Affine, &segs).expect("pipe");
        let (grid, tg) = gemm_common::dispatch_shape(m, n_total as usize);
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = cmd.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline.pipeline);
        bind_gpu_buffers(&enc, &buf_x, &buf_y_stacked, &gpu_blob.buffer, m as u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        for (seg, plane) in segs.iter().zip([0usize, plane_bytes]) {
            let pipeline1 = gemm_q4::pipeline_for(&ctx, n as u32, k as u32).expect("p");
            let cmd2 = ctx.queue.commandBuffer().expect("cmd");
            let enc2 = cmd2.computeCommandEncoder().expect("enc");
            enc2.setComputePipelineState(&pipeline1.pipeline);
            unsafe {
                enc2.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                enc2.setBuffer_offset_atIndex(Some(&buf_y_split), plane, 1);
                enc2.setBuffer_offset_atIndex(Some(&gpu_blob.buffer), 0, 2);
            }
            gpu_common::set_bytes(&enc2, &seg.w_off, 3);
            let m_u32 = m as u32;
            gpu_common::set_bytes(&enc2, &m_u32, 4);
            let (g2, t2) = gemm_common::dispatch_shape(m, n);
            enc2.dispatchThreadgroups_threadsPerThreadgroup(g2, t2);
            enc2.endEncoding();
            cmd2.commit();
            cmd2.waitUntilCompleted();
        }

        let read_y = |buf: &ProtocolObject<dyn MTLBuffer>| -> Vec<f32> {
            let ptr = buf.contents().as_ptr() as *const u16;
            (0..plane_bytes * 2 / 2)
                .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
                .collect()
        };
        let stacked = read_y(&buf_y_stacked);
        let split = read_y(&buf_y_split);
        let max = max_abs_diff(&stacked, &split);
        assert!(max < 0.05, "dgq gate_up stacked vs split max_abs={max}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn stacked_gpu_matches_split_on_dgq_qkv_sliding() {
        use crate::metal::DgqGpuBlob;
        use crate::metal::build_offsets_from_store;
        use crate::shaders::gemm_q4;
        use objc2::runtime::ProtocolObject;
        use std::path::Path;

        let dir = ["model/diffusiongemma-q4emb", "/tmp/quantized-weights"]
            .into_iter()
            .map(Path::new)
            .find(|p| crate::dgq::store::looks_like_dgq_dir(p));
        let Some(dir) = dir else {
            eprintln!("skip stacked_gpu_matches_split_on_dgq_qkv_sliding: no .dgq weights");
            return;
        };
        let store = crate::dgq::store::DgqStore::open(dir).expect("dgq open");
        let offsets = build_offsets_from_store(&store);
        let layer = 0usize;
        let q_off = *offsets
            .get(&format!(
                "model.decoder.layers.{layer}.self_attn.q_proj.weight"
            ))
            .expect("q");
        let k_off = *offsets
            .get(&format!(
                "model.decoder.layers.{layer}.self_attn.k_proj.weight"
            ))
            .expect("k");
        let v_off = *offsets
            .get(&format!(
                "model.decoder.layers.{layer}.self_attn.v_proj.weight"
            ))
            .expect("v");

        let m = 4usize;
        let k = 2816usize;
        let nq = 4096usize;
        let nk = 2048usize;
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 + 3.0) * 0.0008).sin() * 0.13)
            .collect();
        let q_plane = m * nq * 2;
        let k_plane = m * nk * 2;
        let segs = [
            GemmStackedSeg {
                n_cols: nq as u32,
                y_col0: 0,
                y_row_cols: nq as u32,
                _pad: 0,
                w_off: q_off,
                y_byte_off: 0,
            },
            GemmStackedSeg {
                n_cols: nk as u32,
                y_col0: 0,
                y_row_cols: nk as u32,
                _pad: 0,
                w_off: k_off,
                y_byte_off: q_plane as u64,
            },
            GemmStackedSeg {
                n_cols: nk as u32,
                y_col0: 0,
                y_row_cols: nk as u32,
                _pad: 0,
                w_off: v_off,
                y_byte_off: (q_plane + k_plane) as u64,
            },
        ];

        use crate::metal::buffer::BufferPool;
        use crate::metal::device::MetalContext;
        let ctx = MetalContext::new().expect("metal");
        let mut pool = BufferPool::new();
        let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let y_bytes = q_plane + k_plane + k_plane;
        let buf_x = pool.allocate(&ctx.device, m * k * 2).expect("x");
        let buf_y_stacked = pool.allocate(&ctx.device, y_bytes).expect("y stacked");
        let buf_y_split = pool.allocate(&ctx.device, y_bytes).expect("y split");
        BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&x));

        let n_total = (nq + nk + nk) as u32;
        let pipeline =
            pipeline_for(&ctx, n_total, k as u32, QuantFormat::Q4Affine, &segs).expect("pipe");
        let (grid, tg) = gemm_common::dispatch_shape(m, n_total as usize);
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = cmd.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline.pipeline);
        bind_gpu_buffers(&enc, &buf_x, &buf_y_stacked, &gpu_blob.buffer, m as u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        for (seg, plane) in segs.iter().zip([0usize, q_plane, q_plane + k_plane]) {
            let n = seg.n_cols as usize;
            let pipeline1 = gemm_q4::pipeline_for(&ctx, n as u32, k as u32).expect("p");
            let cmd2 = ctx.queue.commandBuffer().expect("cmd");
            let enc2 = cmd2.computeCommandEncoder().expect("enc");
            enc2.setComputePipelineState(&pipeline1.pipeline);
            unsafe {
                enc2.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                enc2.setBuffer_offset_atIndex(Some(&buf_y_split), plane, 1);
                enc2.setBuffer_offset_atIndex(Some(&gpu_blob.buffer), 0, 2);
            }
            gpu_common::set_bytes(&enc2, &seg.w_off, 3);
            let m_u32 = m as u32;
            gpu_common::set_bytes(&enc2, &m_u32, 4);
            let (g2, t2) = gemm_common::dispatch_shape(m, n);
            enc2.dispatchThreadgroups_threadsPerThreadgroup(g2, t2);
            enc2.endEncoding();
            cmd2.commit();
            cmd2.waitUntilCompleted();
        }

        let read_y = |buf: &ProtocolObject<dyn MTLBuffer>| -> Vec<f32> {
            let ptr = buf.contents().as_ptr() as *const u16;
            (0..y_bytes / 2)
                .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
                .collect()
        };
        let max = max_abs_diff(&read_y(&buf_y_stacked), &read_y(&buf_y_split));
        assert!(max < 0.05, "dgq qkv stacked vs split max_abs={max}");
    }
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "gemm_block_stacked",
    entry: "gemm_block_stacked",
    quant_formats: &[
        crate::shaders::variant::QuantFormat::Q4Affine,
        crate::shaders::variant::QuantFormat::NvFp4,
    ],
    fc: &[(4, "IS_FULL_LAYER"), (5, "GEMM_N"), (6, "GEMM_K")],
    variants: crate::shaders::manifest::KernelVariants::GemmBlock,
};
