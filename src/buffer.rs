//! Owned buffers for hot paths. `Vec` is used only for growth; inner loops use
//! `FastSlice` / `FastSliceMut` (pointer + length).

use crate::fast_slice::{FastSlice, FastSliceMut};

/// Heap buffer with explicit length; no slice types in inner loops.
pub struct Buffer<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> Buffer<T> {
    pub fn new(len: usize) -> Self
    where
        T: Copy,
    {
        let mut buf = Self::with_capacity(len);
        buf.len = len;
        if len > 0 {
            unsafe {
                std::ptr::write_bytes(buf.ptr, 0, len);
            }
        }
        buf
    }

    pub fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: 0,
            };
        }
        let layout = std::alloc::Layout::array::<T>(cap).expect("buffer layout");
        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, len: 0, cap }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn ensure_len(&mut self, len: usize)
    where
        T: Copy,
    {
        if len > self.cap {
            self.grow(len);
        }
        if len > self.len {
            unsafe {
                std::ptr::write_bytes(self.ptr.add(self.len), 0, len - self.len);
            }
        }
        self.len = len;
    }

    fn grow(&mut self, min_cap: usize) {
        let new_cap = min_cap.max(self.cap.saturating_mul(2).max(4));
        let new_layout = std::alloc::Layout::array::<T>(new_cap).expect("buffer layout");
        let new_ptr = if self.cap == 0 {
            unsafe { std::alloc::alloc(new_layout) as *mut T }
        } else {
            let old_layout = std::alloc::Layout::array::<T>(self.cap).expect("buffer layout");
            unsafe {
                std::alloc::realloc(self.ptr as *mut u8, old_layout, new_layout.size()) as *mut T
            }
        };
        if new_ptr.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
    }

    pub fn as_fast_slice(&self) -> FastSlice<'_, T> {
        unsafe { FastSlice::from_ptr(self.ptr, self.len) }
    }

    pub fn as_fast_slice_mut(&mut self) -> FastSliceMut<'_, T> {
        unsafe { FastSliceMut::from_ptr(self.ptr, self.len) }
    }

    /// Boundary helper for APIs that still expect a Rust slice (e.g. Metal upload).
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> Buffer<T> {
    pub fn from_fast_slice_mut(data: FastSliceMut<'_, T>) -> Self {
        let len = data.len();
        let mut buf = Self::with_capacity(len);
        buf.len = len;
        if len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(data.ptr, buf.ptr, len);
            }
        }
        buf
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = std::alloc::Layout::array::<T>(self.cap).expect("buffer layout");
            unsafe {
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_fast_slice_roundtrip() {
        let mut buf = Buffer::new(4);
        {
            let mut s = buf.as_fast_slice_mut();
            unsafe {
                s.write_unchecked(0, 1);
                s.write_unchecked(3, 9);
            }
        }
        let s = buf.as_fast_slice();
        assert_eq!(unsafe { s.read_unchecked(0) }, 1);
        assert_eq!(unsafe { s.read_unchecked(3) }, 9);
    }
}
