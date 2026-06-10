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

    pub fn write_f32(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[f32]) {
        let ptr = buffer.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn write_bf16(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[u16]) {
        let ptr = buffer.contents().as_ptr() as *mut u16;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, out: &mut [f32]) {
        let ptr = buffer.contents().as_ptr() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
        }
    }
}
