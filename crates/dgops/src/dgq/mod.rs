//! Minimal portable reader for a \`.dgq\` weight pack.
//!
//! The pack is a JSON manifest plus one mmapped blob. This is the subset the
//! CUDA-side verification needs: tensor lookup, raw bytes, and f32
//! materialization for the raw (bf16) tensors the diffusion slice uses. The
//! engine's \`src/dgq\` remains authoritative for everything else (layered packs,
//! external refs, quantized classes); this reader deliberately rejects what it
//! does not implement rather than guessing.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "model.dgq.json";

#[derive(Debug, Deserialize)]
pub struct DgqManifest {
    pub version: u32,
    #[serde(default)]
    pub profile: Option<String>,
    pub blob_file: String,
    pub tensors: Vec<DgqTensorEntry>,
}

#[derive(Debug, Deserialize)]
pub struct DgqTensorEntry {
    pub name: String,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub offset: u64,
    pub byte_len: u64,
}

impl DgqTensorEntry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }

    pub fn is_raw(&self) -> bool {
        self.kind == "raw"
    }
}

pub struct DgqPack {
    blob: memmap2::Mmap,
    entries: Vec<DgqTensorEntry>,
    index: HashMap<String, usize>,
}

impl DgqPack {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let manifest_path = model_dir.join(MANIFEST_FILE);
        let manifest: DgqManifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
        let blob_path: PathBuf = model_dir.join(&manifest.blob_file);
        let file = File::open(&blob_path)?;
        let blob = unsafe { memmap2::Mmap::map(&file)? };
        let mut index = HashMap::with_capacity(manifest.tensors.len());
        for (i, t) in manifest.tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }
        Ok(Self {
            blob,
            entries: manifest.tensors,
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
            .offset
            .checked_add(e.byte_len)
            .ok_or(Error::Format("tensor range overflow"))?;
        if end > self.blob.len() as u64 {
            return Err(Error::Format("tensor range past end of blob"));
        }
        Ok(&self.blob[e.offset as usize..end as usize])
    }

    /// A raw bf16 tensor as f32.
    pub fn raw_bf16(&self, name: &str) -> Result<Vec<f32>, Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if !e.is_raw() || e.dtype != "BF16" {
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

    /// The first \`rows\` rows of a raw bf16 \`[out, in]\` matrix, as f32.
    /// Lets a verification slice run real weights without materializing a
    /// whole 1.4 GiB table.
    pub fn raw_bf16_rows(&self, name: &str, rows: usize) -> Result<Vec<f32>, Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if !e.is_raw() || e.dtype != "BF16" || e.shape.len() != 2 {
            return Err(Error::Format("tensor is not a raw bf16 matrix"));
        }
        let in_dim = e.shape[1] as usize;
        let rows = rows.min(e.shape[0] as usize);
        let src = self.bytes(name)?;
        let mut out = vec![0.0f32; rows * in_dim];
        for (i, o) in out.iter_mut().enumerate() {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            *o = f32::from_bits((bits as u32) << 16);
        }
        Ok(out)
    }

    /// Raw bf16 bytes for the first \`rows\` rows of a \`[out, in]\` matrix — the
    /// embed-gather table slice.
    pub fn raw_bf16_row_bytes(&self, name: &str, rows: usize) -> Result<Vec<u8>, Error> {
        let e = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        if !e.is_raw() || e.dtype != "BF16" || e.shape.len() != 2 {
            return Err(Error::Format("tensor is not a raw bf16 matrix"));
        }
        let in_dim = e.shape[1] as usize;
        let rows = rows.min(e.shape[0] as usize);
        Ok(self.bytes(name)?[..rows * in_dim * 2].to_vec())
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    NotFound(String),
    Format(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Json(e) => write!(f, "manifest json: {e}"),
            Error::NotFound(n) => write!(f, "tensor not found: {n}"),
            Error::Format(m) => write!(f, "bad dgq pack: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}
