//! Pointer+length views for hot loops (no `&[T]` in inner iteration).

use crate::tensor::Bf16Slice;
use std::marker::PhantomData;

/// Immutable view: raw pointer + element count.
#[derive(Copy, Clone)]
pub struct FastSlice<'a, T> {
    pub(crate) ptr: *const T,
    pub(crate) len: usize,
    _marker: PhantomData<&'a T>,
}

/// Mutable view: raw pointer + element count.
#[derive(Copy, Clone)]
pub struct FastSliceMut<'a, T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> FastSlice<'a, T> {
    /// # Safety
    /// `ptr` must be valid for `len` reads for `'a`.
    pub unsafe fn from_ptr(ptr: *const T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn read_unchecked(&self, index: usize) -> T
    where
        T: Copy,
    {
        debug_assert!(index < self.len);
        unsafe { *self.ptr.add(index) }
    }
}

impl<'a, T> FastSliceMut<'a, T> {
    /// # Safety
    /// `ptr` must be valid for `len` writes for `'a`.
    pub unsafe fn from_ptr(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub unsafe fn read_unchecked(&self, index: usize) -> T
    where
        T: Copy,
    {
        debug_assert!(index < self.len);
        unsafe { *self.ptr.add(index) }
    }

    #[inline(always)]
    pub unsafe fn write_unchecked(&mut self, index: usize, value: T) {
        debug_assert!(index < self.len);
        unsafe {
            *self.ptr.add(index) = value;
        }
    }
}

/// BF16 elements stored as little-endian byte pairs (mmap-safe, may be unaligned).
#[derive(Copy, Clone)]
pub struct FastBf16Slice<'a> {
    ptr: *const u8,
    len: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> FastBf16Slice<'a> {
    pub fn from_bf16(slice: Bf16Slice<'a>) -> Self {
        let bytes = slice.as_bytes();
        debug_assert!(bytes.len() % 2 == 0);
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len() / 2,
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub unsafe fn get_u16_unchecked(&self, index: usize) -> u16 {
        debug_assert!(index < self.len);
        let off = index * 2;
        unsafe { u16::from_le_bytes([*self.ptr.add(off), *self.ptr.add(off + 1)]) }
    }

    #[inline(always)]
    pub unsafe fn to_f32_unchecked(&self, index: usize) -> f32 {
        unsafe { f32::from_bits((self.get_u16_unchecked(index) as u32) << 16) }
    }
}

/// Dequantize bf16 weights into `dst` using unchecked reads.
pub fn bf16_to_f32_into(src: FastBf16Slice<'_>, mut dst: FastSliceMut<'_, f32>) {
    assert_eq!(src.len(), dst.len());
    let n = src.len();
    for i in 0..n {
        unsafe {
            dst.write_unchecked(i, src.to_f32_unchecked(i));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_bf16_matches_slice_get() {
        let bytes: [u8; 4] = [0x00, 0x3f, 0x00, 0x40];
        let slice = Bf16Slice::from_bytes(&bytes);
        let fast = FastBf16Slice::from_bf16(slice);
        assert_eq!(unsafe { fast.get_u16_unchecked(0) }, slice.get(0));
        assert_eq!(unsafe { fast.get_u16_unchecked(1) }, slice.get(1));
    }
}
