//! Mmap-backed `.dgq` weight store.

use crate::Error;
use crate::dgq::dequant::dequant_to_f32;
use crate::dgq::layout::{
    DgqManifest, MANIFEST_FILE, QuantKind, TensorSource, blob_slice_range, dgq_version_supported,
    parse_quant_kind,
};
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
    /// Layered packs only: one resolved mmap per `external_files` key,
    /// opened eagerly at `open()` so a missing/mismatched HF cache fails
    /// loud at load time, not on first read of some tensor deep into a run.
    external: HashMap<String, Mmap>,
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
        let mut external = HashMap::with_capacity(manifest.external_files.len());
        for (key, ext_file) in &manifest.external_files {
            let path = crate::dgq::hf_resolve::resolve_external_file(
                &model_dir,
                manifest.base_model.as_ref(),
                key,
                ext_file,
            )?;
            let f = File::open(&path)?;
            let mm = unsafe { Mmap::map(&f)? };
            external.insert(key.clone(), mm);
        }
        Ok(Self {
            model_dir,
            manifest,
            blob,
            index,
            external,
        })
    }

    pub fn is_layered(&self) -> bool {
        self.manifest.is_layered()
    }

    pub fn profile(&self) -> crate::dgq::layout::QuantProfile {
        self.manifest.profile
    }

    /// Total canonical/unified blob length: for a self-contained pack this is
    /// the local bin's file size (identity); for a layered pack it is the
    /// address-space size the loader materializes on the GPU (`offset` /
    /// `w_off` math is expressed against this, not the local bin's — much
    /// smaller — on-disk size).
    pub fn blob_bytes(&self) -> u64 {
        if self.manifest.is_layered() {
            self.manifest
                .tensors
                .iter()
                .map(|t| t.meta.offset + t.meta.byte_len)
                .max()
                .unwrap_or(0)
        } else {
            self.blob.len() as u64
        }
    }

    pub fn tensor_entries(&self) -> &[crate::dgq::layout::DgqTensorEntry] {
        &self.manifest.tensors
    }

    pub fn get_entry(&self, name: &str) -> Option<&crate::dgq::layout::DgqTensorEntry> {
        let &idx = self.index.get(name)?;
        Some(&self.manifest.tensors[idx])
    }

    /// Resolve a tensor's bytes from wherever they actually live: this pack's
    /// own blob at `offset` (`source: None`, every pre-layering pack), this
    /// pack's own blob at a different (compact) `local_offset`
    /// (`TensorSource::Local`), or a resolved external file
    /// (`TensorSource::External`).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], Error> {
        let entry = self
            .get_entry(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        match &entry.meta.source {
            None => {
                let (start, end) = blob_slice_range(
                    entry.meta.offset,
                    entry.meta.byte_len,
                    self.blob.len() as u64,
                )?;
                Ok(&self.blob[start..end])
            }
            Some(TensorSource::Local { local_offset }) => {
                let (start, end) =
                    blob_slice_range(*local_offset, entry.meta.byte_len, self.blob.len() as u64)?;
                Ok(&self.blob[start..end])
            }
            Some(TensorSource::External { file, offset }) => {
                let mm = self.external.get(file).ok_or_else(|| {
                    Error::Layered(format!(
                        "tensor {name} references external file '{file}' which was not \
                         resolved at open()"
                    ))
                })?;
                let (start, end) = blob_slice_range(*offset, entry.meta.byte_len, mm.len() as u64)?;
                Ok(&mm[start..end])
            }
        }
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
        largest.sort_by_key(|b| std::cmp::Reverse(b.1));
        largest.truncate(12);
        let mut top_prefixes: Vec<_> = prefix_counts.into_iter().collect();
        top_prefixes.sort_by_key(|b| std::cmp::Reverse(b.1));

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
