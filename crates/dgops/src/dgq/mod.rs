//! The CUDA side's view of a `.dgq` pack.
//!
//! The format, the manifest types and every load-time gate live in
//! `dgqpack`, which the engine opens packs through as well, so both backends
//! agree on what a valid pack is and a truncated or unfinished one is
//! refused here for the same reason it is refused there. What stays local is
//! the handful of accessors the CUDA verification slice needs: tensor
//! lookup, raw bytes, and f32 materialization for the raw bf16 tensors the
//! diffusion slice uses.

use dgqpack::PackFile;

pub use dgqpack::{DgqTensorEntry, DgqTensorMeta};
use std::collections::HashMap;
use std::path::Path;

pub use dgqpack::MANIFEST_FILE;

pub struct DgqPack {
    blob: memmap2::Mmap,
    entries: Vec<DgqTensorEntry>,
    index: HashMap<String, usize>,
}

impl DgqPack {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let pack = PackFile::open(model_dir)?;
        let blob = pack.map()?;
        let entries = pack.into_manifest().tensors;
        let mut index = HashMap::with_capacity(entries.len());
        for (i, t) in entries.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }
        Ok(Self {
            blob,
            entries,
            index,
        })
    }

    pub fn entries(&self) -> &[DgqTensorEntry] {
        &self.entries
    }

    pub fn get(&self, name: &str) -> Option<&DgqTensorEntry> {
        self.index.get(name).map(|&i| &self.entries[i])
    }

    /// The tensor's bytes in the pack's own blob. Rejects an out-of-range
    /// entry loudly instead of returning a short slice.
    pub fn bytes(&self, name: &str) -> Result<&[u8], Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let end = e
            .meta
            .offset
            .checked_add(e.meta.byte_len)
            .ok_or(Error::Format("tensor range overflow"))?;
        if end > self.blob.len() as u64 {
            return Err(Error::Format("tensor range past end of blob"));
        }
        Ok(&self.blob[e.meta.offset as usize..end as usize])
    }

    /// A raw bf16 tensor as f32.
    pub fn raw_bf16(&self, name: &str) -> Result<Vec<f32>, Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if !e.is_raw() || e.meta.dtype != "BF16" {
            return Err(Error::Format("tensor is not raw bf16"));
        }
        let src = self.bytes(name)?;
        let mut out = vec![0.0f32; e.numel()];
        for (i, o) in out.iter_mut().enumerate() {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            *o = f32::from_bits((bits as u32) << 16);
        }
        Ok(out)
    }

    /// The first `rows` rows of a raw bf16 `[out, in]` matrix, as f32.
    /// Lets a verification slice run real weights without materializing a
    /// whole 1.4 GiB table.
    pub fn raw_bf16_rows(&self, name: &str, rows: usize) -> Result<Vec<f32>, Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if !e.is_raw() || e.meta.dtype != "BF16" || e.meta.shape.len() != 2 {
            return Err(Error::Format("tensor is not a raw bf16 matrix"));
        }
        let in_dim = e.meta.shape[1] as usize;
        let rows = rows.min(e.meta.shape[0] as usize);
        let src = self.bytes(name)?;
        let mut out = vec![0.0f32; rows * in_dim];
        for (i, o) in out.iter_mut().enumerate() {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            *o = f32::from_bits((bits as u32) << 16);
        }
        Ok(out)
    }

    /// Raw bf16 bytes for the first `rows` rows of a `[out, in]` matrix, the
    /// embed-gather table slice.
    pub fn raw_bf16_row_bytes(&self, name: &str, rows: usize) -> Result<Vec<u8>, Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if !e.is_raw() || e.meta.dtype != "BF16" || e.meta.shape.len() != 2 {
            return Err(Error::Format("tensor is not a raw bf16 matrix"));
        }
        let in_dim = e.meta.shape[1] as usize;
        let rows = rows.min(e.meta.shape[0] as usize);
        Ok(self.bytes(name)?[..rows * in_dim * 2].to_vec())
    }
}

#[derive(Debug)]
pub enum Error {
    /// Opening or validating the pack, carrying `dgqpack`'s message.
    Pack(dgqpack::Error),
    NotFound(String),
    Format(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Pack(e) => write!(f, "{e}"),
            Error::NotFound(n) => write!(f, "tensor not found: {n}"),
            Error::Format(m) => write!(f, "bad dgq pack: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<dgqpack::Error> for Error {
    fn from(e: dgqpack::Error) -> Self {
        Error::Pack(e)
    }
}
