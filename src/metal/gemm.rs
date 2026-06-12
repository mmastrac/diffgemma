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
const F32_BF16_GEMM_ENTRY: &str = "f32_bf16_gemm";
const F32_BF16_LINEAR_ENTRY: &str = "f32_bf16_linear";
const THREADGROUP: usize = 16;

pub struct Bf16Gemm {
    ctx: MetalContext,
    pipeline: ComputePipeline,
    f32_bf16_pipeline: ComputePipeline,
    f32_bf16_linear_pipeline: ComputePipeline,
    pool: BufferPool,
}

impl Bf16Gemm {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let pipeline = ctx.compile_kernel(GEMM_SHADER, GEMM_ENTRY)?;
        let f32_bf16_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_BF16_GEMM_ENTRY)?;
        let f32_bf16_linear_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_BF16_LINEAR_ENTRY)?;
        Ok(Self {
            ctx,
            pipeline,
            f32_bf16_pipeline,
            f32_bf16_linear_pipeline,
            pool: BufferPool::new(),
        })
    }

    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }

    pub fn pool_mut(&mut self) -> &mut BufferPool {
        &mut self.pool
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

    /// Row-major GEMM with f32 activations and bf16 weights: C[M,N] = A[M,K] @ B[K,N].
    /// `C = A @ W^T` with PyTorch `W[n,k]`.
    pub fn matmul_f32_bf16_linear(
        &mut self,
        a: &[f32],
        w: &[u16],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), Error> {
        if a.len() != m * k || w.len() != n * k || c.len() != m * n {
            return Err(Error::Format("f32_bf16 linear shape mismatch"));
        }

        let a_bytes = a.len() * 4;
        let w_bytes = w.len() * 2;
        let c_bytes = c.len() * 4;

        let buf_a = self
            .pool
            .allocate(&self.ctx.device, a_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_w = self
            .pool
            .allocate(&self.ctx.device, w_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_c = self
            .pool
            .allocate(&self.ctx.device, c_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_f32(&buf_a, a);
        BufferPool::write_bf16(&buf_w, w);
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
            &self.f32_bf16_linear_pipeline.pipeline,
            &buf_a,
            &buf_w,
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
        self.pool.release(w_bytes, buf_w);
        self.pool.release(c_bytes, buf_c);
        Ok(())
    }

    /// Dispatch only — buffers must already hold A/W/C on device (no upload/readback/sync wrapper).
    pub fn dispatch_f32_bf16_linear(
        &self,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), Error> {
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
            &self.f32_bf16_linear_pipeline.pipeline,
            buf_a,
            buf_w,
            buf_c,
            m,
            n,
            k,
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    /// Multiple dispatches in one command buffer, single sync (closer to fused decoder step).
    pub fn dispatch_f32_bf16_linear_batched(
        &self,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
        count: usize,
    ) -> Result<(), Error> {
        let cmd_buf = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Format("Metal command buffer alloc failed"))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
        for _ in 0..count {
            encode_gemm(
                &encoder,
                &self.f32_bf16_linear_pipeline.pipeline,
                buf_a,
                buf_w,
                buf_c,
                m,
                n,
                k,
            );
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Ok(())
    }

    pub fn f32_bf16_linear_pipeline(&self) -> &ComputePipeline {
        &self.f32_bf16_linear_pipeline
    }

    pub fn matmul_f32_bf16(
        &mut self,
        a: &[f32],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), Error> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(Error::Format("f32_bf16 gemm shape mismatch"));
        }

        let a_bytes = a.len() * 4;
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

        BufferPool::write_f32(&buf_a, a);
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
            &self.f32_bf16_pipeline.pipeline,
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

    pub fn matmul_f32_bf16_shared(
        ctx: &MetalContext,
        pool: &mut BufferPool,
        pipeline: &ComputePipeline,
        a: &[f32],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), Error> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(Error::Format("f32_bf16 gemm shape mismatch"));
        }

        let a_bytes = a.len() * 4;
        let b_bytes = b.len() * 2;
        let c_bytes = c.len() * 4;

        let buf_a = pool
            .allocate(&ctx.device, a_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_b = pool
            .allocate(&ctx.device, b_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_c = pool
            .allocate(&ctx.device, c_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_f32(&buf_a, a);
        BufferPool::write_bf16(&buf_b, b);
        BufferPool::write_f32(&buf_c, c);

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Format("Metal command buffer alloc failed"))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
        encode_gemm(
            &encoder,
            &pipeline.pipeline,
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
        pool.release(a_bytes, buf_a);
        pool.release(b_bytes, buf_b);
        pool.release(c_bytes, buf_c);
        Ok(())
    }

    pub fn matmul_shared(
        ctx: &MetalContext,
        pool: &mut BufferPool,
        pipeline: &ComputePipeline,
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

        let buf_a = pool
            .allocate(&ctx.device, a_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_b = pool
            .allocate(&ctx.device, b_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_c = pool
            .allocate(&ctx.device, c_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_bf16(&buf_a, a);
        BufferPool::write_bf16(&buf_b, b);
        BufferPool::write_f32(&buf_c, c);

        let cmd_buf = ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Format("Metal command buffer alloc failed"))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
        encode_gemm(
            &encoder,
            &pipeline.pipeline,
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
        pool.release(a_bytes, buf_a);
        pool.release(b_bytes, buf_b);
        pool.release(c_bytes, buf_c);
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

pub fn f32_bf16_linear_cpu(c: &mut [f32], a: &[f32], w: &[u16], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let av = a[row * k + p];
                let wv = bf16_to_f32(w[col * k + p]);
                sum += av * wv;
            }
            c[row * n + col] = sum;
        }
    }
}

pub fn f32_bf16_matmul_cpu(c: &mut [f32], a: &[f32], b: &[u16], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let av = a[row * k + p];
                let bv = bf16_to_f32(b[p * n + col]);
                sum += av * bv;
            }
            c[row * n + col] = sum;
        }
    }
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_f32_bf16_linear_matches_cpu_decoder_mlp_shape() {
        let m = 256usize;
        let k = 2816usize;
        let n = 2112usize;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let w: Vec<u16> = (0..n * k)
            .map(|i| f32_to_bf16(((i as f32) * 0.0007).cos() * 0.01))
            .collect();
        let mut cpu = vec![0.0f32; m * n];
        f32_bf16_linear_cpu(&mut cpu, &a, &w, m, k, n);

        let mut gpu = vec![0.0f32; m * n];
        let mut gemm = Bf16Gemm::new().expect("gemm");
        gemm.matmul_f32_bf16_linear(&a, &w, &mut gpu, m, k, n)
            .expect("gpu linear");

        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max_diff={max_diff}");
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_gemm_matches_cpu_decoder_mlp_shape() {
        let m = 256usize;
        let k = 2816usize;
        let n = 2112usize;
        let a: Vec<u16> = (0..m * k).map(|i| f32_to_bf16((i as f32 * 0.001).sin())).collect();
        let b: Vec<u16> = (0..k * n)
            .map(|i| f32_to_bf16(((i as f32) * 0.0007).cos() * 0.01))
            .collect();
        let mut cpu = vec![0.0f32; m * n];
        bf16_matmul_cpu(&mut cpu, &a, &b, m, k, n);

        let mut gpu = vec![0.0f32; m * n];
        let mut gemm = Bf16Gemm::new().expect("gemm");
        gemm.matmul(&a, &b, &mut gpu, m, k, n).expect("gpu gemm");

        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max_diff={max_diff}");
    }
}
