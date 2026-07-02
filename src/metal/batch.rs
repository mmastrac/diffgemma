//! Batched Metal dispatches: many kernels, one `commit` + `waitUntilCompleted`.

use crate::metal::buffer::BufferPool;
use crate::metal::telemetry::ForwardTelemetry;
use crate::safetensors::Error;
use std::cell::RefCell;
use std::rc::Rc;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLSize,
};

struct PendingRead {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dst: *mut u8,
    len_bytes: usize,
    src_byte_offset: usize,
    read_u32: bool,
}

pub struct GpuBatch<'a> {
    pub(crate) pool: &'a mut BufferPool,
    pub(crate) device: &'a ProtocolObject<dyn MTLDevice>,
    cmd: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    reads: Vec<PendingRead>,
    releases: Vec<(usize, Retained<ProtocolObject<dyn MTLBuffer>>)>,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
}

impl<'a> GpuBatch<'a> {
    pub fn begin_with_telemetry(
        queue: &'a ProtocolObject<dyn MTLCommandQueue>,
        pool: &'a mut BufferPool,
        device: &'a ProtocolObject<dyn MTLDevice>,
        telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
    ) -> Result<GpuBatch<'a>, Error> {
        let cmd = queue
            .commandBuffer()
            .ok_or(Error::Format("Metal command buffer alloc failed"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
        Ok(Self {
            pool,
            device,
            cmd: Some(cmd),
            enc: Some(enc),
            reads: Vec::new(),
            releases: Vec::new(),
            telemetry,
        })
    }

    pub fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        self.enc.as_ref().expect("batch not begun")
    }

    fn track_release(&mut self, bytes: usize, buf: Retained<ProtocolObject<dyn MTLBuffer>>) {
        self.releases.push((bytes, buf));
    }

    fn track_read(
        &mut self,
        buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        out: &mut [f32],
    ) {
        self.reads.push(PendingRead {
            buf,
            dst: out.as_mut_ptr() as *mut u8,
            len_bytes: out.len() * 4,
            src_byte_offset: 0,
            read_u32: false,
        });
    }

    pub fn register_read(
        &mut self,
        buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        out: &mut [f32],
    ) {
        self.track_read(buf, out);
    }

    pub fn register_read_u32(
        &mut self,
        buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        out: &mut [u32],
    ) {
        self.reads.push(PendingRead {
            buf,
            dst: out.as_mut_ptr() as *mut u8,
            len_bytes: out.len() * 4,
            src_byte_offset: 0,
            read_u32: true,
        });
    }

    pub fn register_read_offset(
        &mut self,
        buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        src_offset_elems: usize,
        out: &mut [f32],
    ) {
        self.reads.push(PendingRead {
            buf,
            dst: out.as_mut_ptr() as *mut u8,
            len_bytes: out.len() * 4,
            src_byte_offset: src_offset_elems * 4,
            read_u32: false,
        });
    }

    fn run_pending_read(read: PendingRead) {
        if read.read_u32 {
            let out = unsafe {
                std::slice::from_raw_parts_mut(read.dst as *mut u32, read.len_bytes / 4)
            };
            if read.src_byte_offset == 0 {
                BufferPool::read_u32(&read.buf, out);
            } else {
                BufferPool::read_u32_at_offset(&read.buf, read.src_byte_offset, out);
            }
        } else {
            let out = unsafe {
                std::slice::from_raw_parts_mut(read.dst as *mut f32, read.len_bytes / 4)
            };
            if read.src_byte_offset == 0 {
                BufferPool::read_f32(&read.buf, out);
            } else {
                BufferPool::read_f32_at_offset(&read.buf, read.src_byte_offset, out);
            }
        }
    }

    pub fn alloc_f32(
        &mut self,
        data: &[f32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        let bytes = data.len() * 4;
        let buf = self
            .pool
            .allocate(self.device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf, data);
        self.track_release(bytes, buf.clone());
        Ok(buf)
    }

    pub fn alloc_f32_out(&mut self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        let bytes = len * 4;
        let buf = self
            .pool
            .allocate(self.device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        // Pooled buffers may hold stale bytes; zero so partial kernel coverage cannot leak.
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, bytes);
        }
        self.track_release(bytes, buf.clone());
        Ok(buf)
    }

    pub fn alloc_u32_out(&mut self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        self.alloc_f32_out(len)
    }

    pub fn alloc_bytes(
        &mut self,
        data: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        let bytes = data.len();
        let buf = self
            .pool
            .allocate(self.device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_bytes(&buf, data);
        self.track_release(bytes, buf.clone());
        Ok(buf)
    }

    pub fn alloc_i64(
        &mut self,
        data: &[i64],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        let bytes = data.len() * 8;
        let buf = self
            .pool
            .allocate(self.device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_i64(&buf, data);
        self.track_release(bytes, buf.clone());
        Ok(buf)
    }

    pub fn dispatch_1d(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        count: usize,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) {
        const THREADS_PER_TG: usize = 256;
        self.encoder().setComputePipelineState(pipeline);
        encode(self.encoder());
        let tg_width = THREADS_PER_TG.min(count);
        let grid_width = div_up(count, tg_width);
        let grid = MTLSize {
            width: grid_width,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        self.encoder()
            .dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Like [`dispatch_1d`] but splits work when `count` exceeds Metal's 65535 threadgroup grid limit.
    pub fn dispatch_1d_ranged(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        count: usize,
        mut encode: impl FnMut(&ProtocolObject<dyn MTLComputeCommandEncoder>, u32, u32),
    ) {
        const THREADS_PER_TG: usize = 256;
        const MAX_GROUPS: usize = 65535;
        const CHUNK: usize = MAX_GROUPS * THREADS_PER_TG;
        let mut base = 0usize;
        while base < count {
            let chunk = (count - base).min(CHUNK);
            self.encoder().setComputePipelineState(pipeline);
            encode(self.encoder(), base as u32, chunk as u32);
            let tg_width = THREADS_PER_TG.min(chunk);
            let grid_width = div_up(chunk, tg_width);
            let grid = MTLSize {
                width: grid_width,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            };
            self.encoder()
                .dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            base += chunk;
        }
    }

    pub fn dispatch_with_grid(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        grid: MTLSize,
        tg: MTLSize,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) {
        self.encoder().setComputePipelineState(pipeline);
        encode(self.encoder());
        self.encoder()
            .dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// `C[M,N] = A[M,K] @ W[N,K]^T` with PyTorch row-major `W`.
    pub fn dispatch_linear(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::dispatch_matmul(self.encoder(), pipeline, buf_a, buf_w, buf_c, m, n, k);
    }

    /// `dispatch_linear` with the weight bound at a byte offset (bf16 weights
    /// read straight from the resident `.dgq` blob — no upload).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_linear_w_off(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        w_byte_offset: u64,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        const THREADGROUP: usize = 16;
        let encoder = self.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(
                Some(buf_w),
                crate::dgq::layout::blob_offset_for_mtl(w_byte_offset),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(buf_c), 0, 2);
        }
        let dims = [m as u32, n as u32, k as u32];
        set_bytes(encoder, &dims, 3);
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

    /// `C[M,N] = A[M,K] @ W[N,K]^T` with Q4 weights at `buf_w` + byte offset.
    pub fn dispatch_q4_linear(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        w_byte_offset: u64,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
        groups_per_row: u32,
    ) {
        const THREADGROUP: usize = 16;
        let encoder = self.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(
                Some(buf_w),
                crate::dgq::layout::blob_offset_for_mtl(w_byte_offset),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(buf_c), 0, 2);
        }
        let dims = [m as u32, n as u32, k as u32, groups_per_row];
        crate::metal::batch::set_bytes(encoder, &dims, 3);
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

    pub fn dispatch_nvfp4_linear(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        w_byte_offset: u64,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        const THREADGROUP: usize = 16;
        let encoder = self.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(
                Some(buf_w),
                crate::dgq::layout::blob_offset_for_mtl(w_byte_offset),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(buf_c), 0, 2);
        }
        let dims = [m as u32, n as u32, k as u32];
        crate::metal::batch::set_bytes(encoder, &dims, 3);
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

    /// Grouped MoE Q4 GEMM: `total_m` flattened rows, one weight row per job via metadata.
    pub fn dispatch_q4_linear_grouped(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w_blob: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        buf_jobs: &ProtocolObject<dyn MTLBuffer>,
        buf_row_starts: &ProtocolObject<dyn MTLBuffer>,
        total_m: usize,
        k: usize,
        n: usize,
        num_jobs: usize,
    ) {
        const SIMD_WIDTH: usize = 32;
        let encoder = self.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(buf_w_blob), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(buf_c), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(buf_jobs), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(buf_row_starts), 0, 4);
        }
        let dims = [k as u32, n as u32, num_jobs as u32];
        crate::metal::batch::set_bytes(encoder, &dims, 5);
        let tg = MTLSize {
            width: SIMD_WIDTH,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: n,
            height: total_m,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// `C[M,N] = A[M,K] @ W[N,K]^T` with Q8 row weights at `buf_w` + byte offset.
    pub fn dispatch_q8_linear(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        w_byte_offset: u64,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        const THREADGROUP: usize = 16;
        let encoder = self.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(
                Some(buf_w),
                crate::dgq::layout::blob_offset_for_mtl(w_byte_offset),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(buf_c), 0, 2);
        }
        let dims = [m as u32, n as u32, k as u32];
        crate::metal::batch::set_bytes(encoder, &dims, 3);
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

    fn dispatch_matmul(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_b: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        const THREADGROUP: usize = 16;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(buf_b), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(buf_c), 0, 2);
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

    pub(crate) fn record_dense_upload(&mut self, bytes: u64) {
        if let Some(cell) = self.telemetry.as_ref() {
            cell.borrow_mut().dense_gpu_upload_bytes += bytes;
        }
    }

    pub fn end(mut self) -> Result<(), Error> {
        let readback_bytes: u64 = self.reads.iter().map(|r| r.len_bytes as u64).sum();
        let enc = self.enc.take().expect("batch encoder missing");
        enc.endEncoding();
        let cmd = self.cmd.take().expect("batch command buffer missing");
        cmd.commit();
        cmd.waitUntilCompleted();

        if let Some(cell) = self.telemetry.as_ref() {
            let mut tel = cell.borrow_mut();
            tel.gpu_syncs += 1;
            tel.gpu_readback_bytes += readback_bytes;
        }

        for read in self.reads {
            Self::run_pending_read(read);
        }
        for (bytes, buf) in self.releases {
            self.pool.release(bytes, buf);
        }
        Ok(())
    }
}

pub(crate) fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

/// Start a batch with optional step telemetry (only `pool` is mutably borrowed).
pub fn begin_engine_batch<'a>(
    queue: &'a ProtocolObject<dyn MTLCommandQueue>,
    pool: &'a mut BufferPool,
    device: &'a ProtocolObject<dyn MTLDevice>,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<GpuBatch<'a>, Error> {
    GpuBatch::begin_with_telemetry(queue, pool, device, telemetry)
}

pub fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}
