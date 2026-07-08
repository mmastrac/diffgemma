//! Encoder-path RoPE on `[seq, heads, head_dim]` (f32, one thread per head×seq).

use super::engine_gqa_common::{self, GqaParams};
use super::test_util::ElemFormat;
use crate::kernels::cpu::{self, RopeKind};
use crate::safetensors::Error;

pub const ENTRY: &str = "apply_rope_heads";

const SHADER: &str = shader_include::include_metal!("kernels/apply_rope_heads.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub freqs: Vec<f32>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub elem_offset: u32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

fn fill_tensor(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.37 + ((i as f32) * seed * 0.7).cos() * 0.21)
        .collect()
}

fn rope_freqs(seq_len: usize, rotary_dim: usize, kind: RopeKind) -> Vec<f32> {
    let positions: Vec<i64> = (0..seq_len as i64).collect();
    let mut freqs = vec![0.0f32; seq_len * rotary_dim];
    cpu::compute_rope_freqs(&mut freqs, &positions, kind);
    freqs
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 4usize;
    let num_heads = 2usize;
    let head_dim = 8usize;
    let rotary_dim = head_dim;
    Fixture {
        x: fill_tensor(seq_len * num_heads * head_dim, 0.11),
        freqs: rope_freqs(
            seq_len,
            rotary_dim,
            RopeKind::Default {
                head_dim,
                theta: 10_000.0,
            },
        ),
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        elem_offset: 0,
    }
}

/// Sliding-layer prefill: 16 heads × 128 seq × hd 256 (2048 GPU threads).
pub fn sliding_prefill_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 128usize;
    let num_heads = 16usize;
    let head_dim = 256usize;
    let rotary_dim = head_dim;
    Fixture {
        x: fill_tensor(seq_len * num_heads * head_dim, 0.017),
        freqs: rope_freqs(
            seq_len,
            rotary_dim,
            RopeKind::Default {
                head_dim,
                theta: 10_000.0,
            },
        ),
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        elem_offset: 0,
    }
}

/// Full-attention layer: proportional partial RoPE (rotary_dim=128 on hd=512).
pub fn full_partial_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 64usize;
    let num_heads = 16usize;
    let head_dim = 512usize;
    let rotary_dim = 128usize;
    Fixture {
        x: fill_tensor(seq_len * num_heads * head_dim, 0.013),
        freqs: rope_freqs(
            seq_len,
            rotary_dim,
            RopeKind::Proportional {
                rotary_dim,
                full_head_dim: head_dim,
                theta: 1_000_000.0,
            },
        ),
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        elem_offset: 0,
    }
}

/// RoPE only on K slice inside a larger Q+K buffer (elem_offset past Q region).
pub fn k_offset_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 32usize;
    let n_q_heads = 16usize;
    let n_kv_heads = 8usize;
    let head_dim = 256usize;
    let rotary_dim = head_dim;
    let q_len = seq_len * n_q_heads * head_dim;
    let k_len = seq_len * n_kv_heads * head_dim;
    let x = fill_tensor(q_len + k_len, 0.019);
    let freqs = rope_freqs(
        seq_len,
        rotary_dim,
        RopeKind::Default {
            head_dim,
            theta: 10_000.0,
        },
    );
    Fixture {
        x,
        freqs,
        seq_len,
        num_heads: n_kv_heads,
        head_dim,
        rotary_dim,
        elem_offset: q_len as u32,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = f.x.clone();
    let k_off = f.elem_offset as usize;
    cpu::apply_rope_tensor(
        &mut out[k_off..],
        f.seq_len,
        f.num_heads,
        f.head_dim,
        &f.freqs,
        f.rotary_dim,
    );
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let mut pipelines = ctx.compile_kernels(SHADER, &[ENTRY])?;
    pipelines
        .pop()
        .ok_or(Error::Format("Metal pipeline missing"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx)?;
    let mut pool = BufferPool::new();

    let x_bytes = f.x.len() * 4;
    let f_bytes = f.freqs.len() * 4;
    let buf_x = pool
        .allocate(&ctx.device, x_bytes)
        .ok_or(Error::Format("alloc"))?;
    let buf_f = pool
        .allocate(&ctx.device, f_bytes)
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_f32(&buf_x, &f.x);
    BufferPool::write_f32(&buf_f, &f.freqs);

    let params = GqaParams::for_rope(
        f.seq_len,
        f.num_heads,
        f.head_dim,
        f.rotary_dim,
        f.elem_offset,
    );

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_f), 0, 1);
    }
    engine_gqa_common::set_params(&enc, &params, 2);
    engine_gqa_common::dispatch_head_query_2d(&enc, f.num_heads, f.seq_len);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; f.x.len()];
    BufferPool::read_f32(&buf_x, &mut out);
    pool.release(x_bytes, buf_x);
    pool.release(f_bytes, buf_f);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sub::test_util::{ElemFormat, assert_oracle};

    fn run_matrix(fixture_fn: fn(ElemFormat) -> Fixture, max_tol: f32) {
        let fix = fixture_fn(ElemFormat::F32);
        let cpu = cpu(&fix);
        assert!(cpu.iter().all(|v| v.is_finite()));
        assert_eq!(cpu.len(), fixture_len(&fix));

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let gpu = gpu(&fix).expect("gpu");
            assert_oracle(&gpu, &cpu, max_tol, 0.9999);
        }
    }

    #[test]
    fn cpu_tiny() {
        run_matrix(tiny_fixture, 1e-5);
    }

    #[test]
    fn cpu_sliding_prefill() {
        run_matrix(sliding_prefill_fixture, 1e-5);
    }

    #[test]
    fn cpu_full_partial() {
        run_matrix(full_partial_fixture, 1e-5);
    }

    #[test]
    fn cpu_k_offset() {
        run_matrix(k_offset_fixture, 1e-5);
    }

    // (GPU twins removed — `run_matrix` already exercises the GPU path under
    // #[cfg(metal)] from the cpu_* tests above; the gpu_* copies were byte-
    // identical duplicates.)
}
