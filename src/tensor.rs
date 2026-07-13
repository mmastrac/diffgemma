#![allow(dead_code)]

use crate::Error;
use crate::safetensors::{DType, TensorInfo};

/// Zero-copy view into mmap'd safetensors payload.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: &'a [i64],
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn from_info(name: &'a str, info: &'a TensorInfo, data: &'a [u8]) -> Self {
        Self {
            name,
            dtype: info.dtype,
            shape: &info.shape,
            data,
        }
    }

    pub fn from_parts(name: &'a str, dtype: DType, shape: &'a [i64], data: &'a [u8]) -> Self {
        Self {
            name,
            dtype,
            shape,
            data,
        }
    }

    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    pub fn shape_usize(&self) -> Result<Vec<usize>, Error> {
        self.shape
            .iter()
            .map(|&d| {
                if d < 0 {
                    Err(Error::Runtime("negative dimension"))
                } else {
                    Ok(d as usize)
                }
            })
            .collect()
    }

    /// BF16 elements via unaligned LE reads (mmap offset may be 2-byte aligned only).
    pub fn bf16(&self) -> Result<Bf16Slice<'a>, Error> {
        if self.dtype != DType::BF16 {
            return Err(Error::DType {
                name: self.name.to_string(),
                expected: DType::BF16,
                got: self.dtype,
            });
        }
        let expected_bytes = (self.numel() as usize) * 2;
        if self.data.len() != expected_bytes {
            return Err(Error::Format("bf16 byte length mismatch"));
        }
        Ok(Bf16Slice { data: self.data })
    }

    pub fn expect_shape(&self, expected: &[i64]) -> Result<(), Error> {
        if self.shape != expected {
            return Err(Error::ShapeMismatch {
                name: self.name.to_string(),
                expected: expected.to_vec(),
                got: self.shape.to_vec(),
            });
        }
        Ok(())
    }

    pub fn print_info(&self) {
        println!("tensor: {}", self.name);
        println!("  dtype:  {}", self.dtype.as_str());
        println!("  shape:  {:?}", self.shape);
        println!("  numel:  {}", self.numel());
        println!("  bytes:  {}", self.byte_len());
        if let Ok(slice) = self.bf16() {
            println!("  bf16[0..4]: [{}]", slice.preview_hex(4));
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bf16Slice<'a> {
    data: &'a [u8],
}

impl<'a> Bf16Slice<'a> {
    pub fn from_bytes(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> u16 {
        let off = index * 2;
        u16::from_le_bytes([self.data[off], self.data[off + 1]])
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    pub fn preview_hex(&self, n: usize) -> String {
        (0..n.min(self.len()))
            .map(|i| format!("0x{:04x}", self.get(i)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        (0..self.len())
            .map(|i| f32::from_bits((self.get(i) as u32) << 16))
            .collect()
    }
}

impl<'a> TensorView<'a> {
    pub fn bf16_f32(&self) -> Result<Vec<f32>, Error> {
        Ok(self.bf16()?.to_f32_vec())
    }

    pub fn bf16_scalar(&self) -> Result<f32, Error> {
        let slice = self.bf16()?;
        if slice.len() != 1 {
            return Err(Error::Format("expected scalar bf16 tensor"));
        }
        Ok(f32::from_bits((slice.get(0) as u32) << 16))
    }
}
