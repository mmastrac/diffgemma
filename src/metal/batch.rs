//! Batched Metal dispatches: many kernels, one `commit` + `waitUntilCompleted`.

use crate::fast_slice::FastSlice;
use crate::metal::buffer::BufferPool;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLSize,
};

struct PendingRead {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dst: *mut f32,
    len: usize,
}

pub struct GpuBatch<'a> {
    queue: &'a ProtocolObject<dyn MTLCommandQueue>,
    pool: &'a mut BufferPool,
    device: &'a ProtocolObject<dyn MTLDevice>,
    cmd: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    enc: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    reads: Vec<PendingRead>,
    releases: Vec<(usize, Retained<ProtocolObject<dyn MTLBuffer>>)>,
}

impl<'a> GpuBatch<'a> {
    pub fn begin(
        queue: &'a ProtocolObject<dyn MTLCommandQueue>,
        pool: &'a mut BufferPool,
        device: &'a ProtocolObject<dyn MTLDevice>,
    ) -> Result<Self, Error> {
        let cmd = queue
            .commandBuffer()
            .ok_or(Error::Format("Metal command buffer alloc failed"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
        Ok(Self {
            queue,
            pool,
            device,
            cmd: Some(cmd),
            enc: Some(enc),
            reads: Vec::new(),
            releases: Vec::new(),
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
            dst: out.as_mut_ptr(),
            len: out.len(),
        });
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

    pub fn alloc_bf16(
        &mut self,
        data: &[u16],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        unsafe { self.alloc_bf16_fast(FastSlice::from_ptr(data.as_ptr(), data.len())) }
    }

    pub fn alloc_bf16_fast(
        &mut self,
        data: FastSlice<'_, u16>,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        let bytes = data.len() * 2;
        let buf = self
            .pool
            .allocate(self.device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_bf16_ptr(&buf, data.ptr, data.len());
        self.track_release(bytes, buf.clone());
        Ok(buf)
    }

    pub fn alloc_f32_out(&mut self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
        let bytes = len * 4;
        let buf = self
            .pool
            .allocate(self.device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
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

    pub fn dispatch_gemm(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_b: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        n: usize,
        k: usize,
    ) {
        const THREADGROUP: usize = 16;
        self.encoder().setComputePipelineState(pipeline);
        unsafe {
            self.encoder().setBuffer_offset_atIndex(Some(buf_a), 0, 0);
            self.encoder().setBuffer_offset_atIndex(Some(buf_b), 0, 1);
            self.encoder().setBuffer_offset_atIndex(Some(buf_c), 0, 2);
        }
        let dims = [m as u32, n as u32, k as u32];
        unsafe {
            self.encoder().setBytes_length_atIndex(
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
        self.encoder()
            .dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    pub fn register_read(
        &mut self,
        buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        out: &mut [f32],
    ) {
        self.track_read(buf, out);
    }

    pub fn end(mut self) -> Result<(), Error> {
        let enc = self.enc.take().expect("batch encoder missing");
        enc.endEncoding();
        let cmd = self.cmd.take().expect("batch command buffer missing");
        cmd.commit();
        cmd.waitUntilCompleted();

        for read in self.reads {
            let slice = unsafe { std::slice::from_raw_parts_mut(read.dst, read.len) };
            BufferPool::read_f32(&read.buf, slice);
        }
        for (bytes, buf) in self.releases {
            self.pool.release(bytes, buf);
        }
        Ok(())
    }
}

fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
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
