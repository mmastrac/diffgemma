use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Size-bucketed `MTLBuffer` pool (`StorageModeShared` for unified memory).
pub struct BufferPool {
    free: Vec<(usize, Retained<ProtocolObject<dyn MTLBuffer>>)>,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
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
        let buf = if let Some(idx) = self.free.iter().position(|(cap, _)| *cap >= size) {
            let (_, buf) = self.free.swap_remove(idx);
            buf
        } else {
            device.newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)?
        };
        // Fresh and pooled buffers may hold stale bytes; zero so partial writes cannot leak.
        let len = buf.length();
        unsafe {
            std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, len);
        }
        Some(buf)
    }

    /// Drop all pooled buffers (e.g. at generate start for deterministic reuse).
    pub fn clear(&mut self) {
        self.free.clear();
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
        let ptr = buffer.contents().as_ptr() as *mut u16;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, out: &mut [f32]) {
        Self::read_f32_at_offset(buffer, 0, out);
    }

    pub fn read_f32_at_offset(
        buffer: &ProtocolObject<dyn MTLBuffer>,
        byte_offset: usize,
        out: &mut [f32],
    ) {
        let ptr = buffer.contents().as_ptr() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(byte_offset / 4), out.as_mut_ptr(), out.len());
        }
    }

    pub fn read_u32(buffer: &ProtocolObject<dyn MTLBuffer>, out: &mut [u32]) {
        Self::read_u32_at_offset(buffer, 0, out);
    }

    pub fn read_u32_at_offset(
        buffer: &ProtocolObject<dyn MTLBuffer>,
        byte_offset: usize,
        out: &mut [u32],
    ) {
        let ptr = buffer.contents().as_ptr() as *const u32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(byte_offset / 4), out.as_mut_ptr(), out.len());
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
