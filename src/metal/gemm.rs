use crate::kernels::cpu::bf16_to_f32;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::safetensors::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

const GEMM_SHADER: &str = include_str!("../../shaders/gemm.metal");
const GEMM_ENTRY: &str = "bf16_gemm";
const THREADGROUP: usize = 16;

pub struct Bf16Gemm {
    ctx: MetalContext,
    pipeline: ComputePipeline,
    pool: BufferPool,
}

impl Bf16Gemm {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let pipeline = ctx.compile_kernel(GEMM_SHADER, GEMM_ENTRY)?;
        Ok(Self {
            ctx,
            pipeline,
            pool: BufferPool::new(),
        })
    }

    /// Row-major bf16 GEMM: C[M,N] = A[M,K] @ B[K,N], f32 output.
    pub fn matmul(
        &mut self,
        a: &[u16],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), Error> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(Error::Format("bf16 gemm shape mismatch"));
        }

        let a_bytes = a.len() * 2;
        let b_bytes = b.len() * 2;
        let c_bytes = c.len() * 4;

        let buf_a = self
            .pool
            .allocate(&self.ctx.device, a_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_b = self
            .pool
            .allocate(&self.ctx.device, b_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_c = self
            .pool
            .allocate(&self.ctx.device, c_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_bf16(&buf_a, a);
        BufferPool::write_bf16(&buf_b, b);
        BufferPool::write_f32(&buf_c, c);

        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Format("Metal command buffer alloc failed"))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Format("Metal compute encoder alloc failed"))?;

        encode_gemm(
            &encoder,
            &self.pipeline.pipeline,
            &buf_a,
            &buf_b,
            &buf_c,
            m,
            n,
            k,
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        BufferPool::read_f32(&buf_c, c);

        self.pool.release(a_bytes, buf_a);
        self.pool.release(b_bytes, buf_b);
        self.pool.release(c_bytes, buf_c);
        Ok(())
    }
}

fn encode_gemm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    c: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    n: usize,
    k: usize,
) {
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c), 0, 2);
    }

    let dims = [m as u32, n as u32, k as u32];
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(&dims).cast(),
            std::mem::size_of_val(&dims),
            3,
        );
    }

    let tg = MTLSize {
        width: THREADGROUP,
        height: THREADGROUP,
        depth: 1,
    };
    let grid = MTLSize {
        width: div_up(n, THREADGROUP),
        height: div_up(m, THREADGROUP),
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

pub fn bf16_matmul_cpu(c: &mut [f32], a: &[u16], b: &[u16], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let av = bf16_to_f32(a[row * k + p]);
                let bv = bf16_to_f32(b[p * n + col]);
                sum += av * bv;
            }
            c[row * n + col] = sum;
        }
    }
}

pub fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_bf16_roundtrip_zero() {
        assert_eq!(bf16_to_f32(f32_to_bf16(0.0)), 0.0);
    }
}
