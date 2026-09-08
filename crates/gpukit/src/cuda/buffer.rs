//! Device memory and a size-bucketed pool.

use super::context::Context;
use super::driver::CUdeviceptr;
use crate::Error;

/// A device allocation owned by the pool (or the caller) and freed on drop.
pub struct DeviceBuffer {
    ctx: Context,
    ptr: CUdeviceptr,
    size: usize,
}

impl DeviceBuffer {
    pub fn size(&self) -> usize {
        self.size
    }

    /// Raw device address, for binding as a kernel argument.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    fn assert_fits(&self, len: usize) {
        assert!(
            len <= self.size,
            "device buffer copy of {len} bytes exceeds allocation of {}",
            self.size
        );
    }

    /// Host to device. Synchronous.
    pub fn write_bytes(&self, bytes: &[u8]) -> Result<(), Error> {
        self.assert_fits(bytes.len());
        self.ctx.set_current()?;
        self.ctx.driver().check(
            unsafe { (self.ctx.driver().cu_memcpy_htod)(self.ptr, bytes.as_ptr().cast(), bytes.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// Device to host. Synchronous.
    pub fn read_bytes(&self, bytes: &mut [u8]) -> Result<(), Error> {
        self.assert_fits(bytes.len());
        self.ctx.set_current()?;
        self.ctx.driver().check(
            unsafe {
                (self.ctx.driver().cu_memcpy_dtoh)(bytes.as_mut_ptr().cast(), self.ptr, bytes.len())
            },
            "cuMemcpyDtoH",
        )
    }

    pub fn write_f32(&self, data: &[f32]) -> Result<(), Error> {
        // f32 is plain data; viewing it as bytes is exact.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        self.write_bytes(bytes)
    }

    pub fn read_f32(&self, out: &mut [f32]) -> Result<(), Error> {
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        self.read_bytes(bytes)
    }

    /// Zero the whole allocation.
    pub fn zero(&self) -> Result<(), Error> {
        self.ctx.set_current()?;
        self.ctx.driver().check(
            unsafe { (self.ctx.driver().cu_memset_d8)(self.ptr, 0, self.size) },
            "cuMemsetD8",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // Best effort: a context already destroyed by process teardown makes
        // the free a no-op, and there is nothing useful to do about it.
        if self.ctx.set_current().is_ok() {
            unsafe {
                (self.ctx.driver().cu_mem_free)(self.ptr);
            }
        }
    }
}

/// Size-bucketed device-memory pool. Allocations are zeroed on hand-out so a
/// partial write can never leak stale bytes into a kernel.
#[derive(Default)]
pub struct BufferPool {
    free: Vec<DeviceBuffer>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate(&mut self, ctx: &Context, size: usize) -> Result<DeviceBuffer, Error> {
        let buf = if let Some(idx) = self.free.iter().position(|b| b.size >= size) {
            self.free.swap_remove(idx)
        } else {
            ctx.set_current()?;
            let mut ptr: CUdeviceptr = 0;
            ctx.driver().check(
                unsafe { (ctx.driver().cu_mem_alloc)(&mut ptr, size) },
                "cuMemAlloc",
            )?;
            DeviceBuffer {
                ctx: ctx.clone(),
                ptr,
                size,
            }
        };
        buf.zero()?;
        Ok(buf)
    }

    pub fn release(&mut self, buffer: DeviceBuffer) {
        self.free.push(buffer);
    }

    pub fn clear(&mut self) {
        self.free.clear();
    }

    pub fn trim(&mut self, max_buffers: usize) {
        self.free.truncate(max_buffers);
    }
}
