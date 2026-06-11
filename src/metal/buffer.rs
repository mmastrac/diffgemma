use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Simple size-bucketed `MTLBuffer` pool (`StorageModeShared` for unified memory).
pub struct BufferPool {
    free: Vec<(usize, Retained<ProtocolObject<dyn MTLBuffer>>)>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self { free: Vec::new() }
    }

    pub fn allocate(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        if let Some(idx) = self.free.iter().position(|(cap, _)| *cap >= size) {
            let (_, buf) = self.free.swap_remove(idx);
            return Some(buf);
        }
        device.newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
    }

    pub fn release(&mut self, capacity: usize, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        self.free.push((capacity, buffer));
    }

    /// Drop pooled buffers so unified memory can be reclaimed (buffers are not returned to the OS
    /// until the `Retained` handles are released).
    pub fn trim(&mut self, max_buffers: usize) {
        if self.free.len() > max_buffers {
            self.free.truncate(max_buffers);
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.free.iter().map(|(cap, _)| *cap).sum()
    }

    pub fn write_f32(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[f32]) {
        let ptr = buffer.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn write_f32_at_offset(
        buffer: &ProtocolObject<dyn MTLBuffer>,
        byte_offset: usize,
        data: &[f32],
    ) {
        let ptr = buffer.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(byte_offset / 4), data.len());
        }
    }

    pub fn write_bf16(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[u16]) {
        Self::write_bf16_ptr(buffer, data.as_ptr(), data.len());
    }

    pub fn write_bf16_ptr(
        buffer: &ProtocolObject<dyn MTLBuffer>,
        src: *const u16,
        len: usize,
    ) {
        let ptr = buffer.contents().as_ptr() as *mut u16;
        unsafe {
            std::ptr::copy_nonoverlapping(src, ptr, len);
        }
    }

    pub fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, out: &mut [f32]) {
        let ptr = buffer.contents().as_ptr() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
        }
    }

    pub fn write_bytes(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[u8]) {
        let ptr = buffer.contents().as_ptr() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn write_i64(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[i64]) {
        let ptr = buffer.contents().as_ptr() as *mut i64;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }
}
