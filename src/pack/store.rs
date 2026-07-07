//! Mmap-backed weight store for `iris.pack` models.

use crate::pack::layout::{MANIFEST_FILE, PackManifest};
use crate::safetensors::{DType, Error};
use crate::tensor::TensorView;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct PackedStore {
    pub model_dir: PathBuf,
    manifest: PackManifest,
    mmap: Mmap,
    index: HashMap<String, usize>,
}

impl PackedStore {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let manifest_path = model_dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest: PackManifest = serde_json::from_str(&manifest_json)?;
        if manifest.version != crate::pack::layout::PACK_VERSION {
            return Err(Error::Format("unsupported iris.pack version"));
        }
        let blob_path = model_dir.join(&manifest.blob_file);
        let file = File::open(&blob_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut index = HashMap::with_capacity(manifest.tensors.len());
        for (i, entry) in manifest.tensors.iter().enumerate() {
            index.insert(entry.name.clone(), i);
        }
        Ok(Self {
            model_dir,
            manifest,
            mmap,
            index,
        })
    }

    pub fn is_packed(&self) -> bool {
        true
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        let &idx = self
            .index
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let entry = &self.manifest.tensors[idx];
        let meta = &entry.meta;
        let start = meta.offset as usize;
        let end = start + meta.byte_len as usize;
        if end > self.mmap.len() {
            return Err(Error::Format("packed tensor extends past blob"));
        }
        let dtype = DType::parse(&meta.dtype);
        if dtype == DType::Unknown {
            return Err(Error::Format("unknown packed dtype"));
        }
        Ok(TensorView::from_parts(
            &entry.name,
            dtype,
            &meta.shape,
            &self.mmap[start..end],
        ))
    }

    pub fn summarize(&self) -> crate::weights::Summary {
        let mut dtypes = std::collections::BTreeMap::new();
        let mut prefix_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut total_data_bytes = 0u64;
        let mut total_elements = 0i64;
        let mut largest = Vec::new();

        for entry in &self.manifest.tensors {
            let meta = &entry.meta;
            *dtypes.entry(meta.dtype.clone()).or_default() += 1;
            total_data_bytes += meta.byte_len;
            let numel: i64 = meta.shape.iter().product();
            total_elements += numel;
            *prefix_counts
                .entry(crate::weights::top_prefix(&entry.name))
                .or_default() += 1;
            largest.push((
                entry.name.clone(),
                numel,
                crate::weights::format_shape(&meta.shape),
            ));
        }
        largest.sort_by(|a, b| b.1.cmp(&a.1));
        largest.truncate(12);
        let mut top_prefixes: Vec<_> = prefix_counts.into_iter().collect();
        top_prefixes.sort_by(|a, b| b.1.cmp(&a.1));

        crate::weights::Summary {
            shard_count: 1,
            tensor_count_index: self.manifest.tensors.len(),
            tensor_count_headers: self.manifest.tensors.len(),
            total_file_bytes: self.mmap.len() as u64,
            total_data_bytes,
            total_elements,
            dtypes,
            top_prefixes,
            largest,
        }
    }
}
