//! Mmap-backed `.dgq` weight store.

use crate::dgq::dequant::dequant_to_f32;
use crate::dgq::layout::{dgq_version_supported, parse_quant_kind, DgqManifest, QuantKind, BLOB_FILE, MANIFEST_FILE};
use crate::safetensors::{DType, Error};
use crate::tensor::TensorView;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct DgqStore {
    pub model_dir: PathBuf,
    manifest: DgqManifest,
    blob: Mmap,
    index: HashMap<String, usize>,
}

impl DgqStore {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let manifest_path = model_dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest: DgqManifest = serde_json::from_str(&manifest_json)?;
        if !dgq_version_supported(manifest.version) {
            return Err(Error::Format("unsupported .dgq version"));
        }
        let blob_path = model_dir.join(&manifest.blob_file);
        let file = File::open(&blob_path)?;
        let blob = unsafe { Mmap::map(&file)? };
        let mut index = HashMap::with_capacity(manifest.tensors.len());
        for (i, t) in manifest.tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }
        Ok(Self {
            model_dir,
            manifest,
            blob,
            index,
        })
    }

    pub fn profile(&self) -> crate::dgq::layout::QuantProfile {
        self.manifest.profile
    }

    pub fn blob_bytes(&self) -> u64 {
        self.blob.len() as u64
    }

    pub fn tensor_count(&self) -> usize {
        self.manifest.tensors.len()
    }

    pub fn tensor_entries(&self) -> &[crate::dgq::layout::DgqTensorEntry] {
        &self.manifest.tensors
    }

    pub fn get_entry(&self, name: &str) -> Option<&crate::dgq::layout::DgqTensorEntry> {
        let &idx = self.index.get(name)?;
        Some(&self.manifest.tensors[idx])
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], Error> {
        let entry = self
            .get_entry(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let start = entry.meta.offset as usize;
        let end = start + entry.meta.byte_len as usize;
        if end > self.blob.len() {
            return Err(Error::Format("dgq tensor extends past blob"));
        }
        Ok(&self.blob[start..end])
    }

    /// Materialize tensor as f32 (CPU oracle path).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let entry = self
            .get_entry(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let kind = parse_quant_kind(&entry.meta.kind)?;
        let src = self.tensor_bytes(name)?;
        let numel: usize = entry.meta.shape.iter().product::<i64>() as usize;
        let mut out = vec![0.0f32; numel];
        dequant_to_f32(kind, src, &entry.meta.shape, &mut out)?;
        Ok(out)
    }

    /// Raw tensors only (norms, router). Quantized tensors use `tensor_f32`.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        let entry = self
            .get_entry(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let kind = parse_quant_kind(&entry.meta.kind)?;
        if kind != QuantKind::Raw {
            return Err(Error::Format("dgq tensor is quantized; use tensor_f32"));
        }
        let src = self.tensor_bytes(name)?;
        let dtype = crate::safetensors::DType::parse(&entry.meta.dtype);
        Ok(TensorView::from_parts(
            &entry.name,
            dtype,
            &entry.meta.shape,
            src,
        ))
    }

    pub fn is_quantized(&self) -> bool {
        true
    }

    pub fn summarize(&self) -> crate::weights::Summary {
        use crate::weights::{format_shape, top_prefix};
        use std::collections::BTreeMap;

        let mut dtypes = BTreeMap::new();
        let mut prefix_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_elements = 0i64;
        let mut largest = Vec::new();

        for t in &self.manifest.tensors {
            let key = format!("{} ({})", t.meta.dtype, t.meta.kind);
            *dtypes.entry(key).or_default() += 1;
            *prefix_counts.entry(top_prefix(&t.name)).or_default() += 1;
            let numel: i64 = t.meta.shape.iter().product();
            total_elements += numel;
            largest.push((t.name.clone(), numel, format_shape(&t.meta.shape)));
        }
        largest.sort_by(|a, b| b.1.cmp(&a.1));
        largest.truncate(12);
        let mut top_prefixes: Vec<_> = prefix_counts.into_iter().collect();
        top_prefixes.sort_by(|a, b| b.1.cmp(&a.1));

        crate::weights::Summary {
            shard_count: 1,
            tensor_count_index: self.manifest.tensors.len(),
            tensor_count_headers: self.manifest.tensors.len(),
            total_file_bytes: self.blob.len() as u64,
            total_data_bytes: self.blob.len() as u64,
            total_elements,
            dtypes,
            top_prefixes,
            largest,
        }
    }
}
pub fn looks_like_dgq_dir(path: &Path) -> bool {
    path.join(MANIFEST_FILE).is_file()
}
