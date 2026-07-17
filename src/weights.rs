use crate::Error;
use crate::dgq::DgqStore;
use crate::dgq::store::looks_like_dgq_dir;
use crate::safetensors::{SafetensorsFile, TensorInfo};
use crate::tensor::TensorView;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct IndexFile {
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
    weight_map: BTreeMap<String, String>,
}

/// Original HuggingFace sharded safetensors layout.
pub struct SafetensorStore {
    pub model_dir: PathBuf,
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub shards: Vec<SafetensorsFile>,
    pub weight_map: BTreeMap<String, String>,
    index: HashMap<String, (usize, usize)>,
}

impl SafetensorStore {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_json = std::fs::read_to_string(&index_path)?;
        let index: IndexFile = serde_json::from_str(&index_json)?;

        let shard_names: BTreeSet<String> = index.weight_map.values().cloned().collect();
        let mut shards = Vec::with_capacity(shard_names.len());
        for name in &shard_names {
            shards.push(SafetensorsFile::open(model_dir.join(name))?);
        }

        let mut global_index = HashMap::with_capacity(index.weight_map.len());
        for (si, shard) in shards.iter().enumerate() {
            for (ti, tensor) in shard.tensors.iter().enumerate() {
                global_index.insert(tensor.name.clone(), (si, ti));
            }
        }

        for key in index.weight_map.keys() {
            if !global_index.contains_key(key) {
                eprintln!("warning: weight_map key missing from shard header: {key}");
            }
        }

        Ok(Self {
            model_dir,
            metadata: index.metadata,
            shards,
            weight_map: index.weight_map,
            index: global_index,
        })
    }

    pub fn get(&self, name: &str) -> Option<(&SafetensorsFile, &TensorInfo)> {
        let &(shard_idx, tensor_idx) = self.index.get(name)?;
        let shard = &self.shards[shard_idx];
        Some((shard, &shard.tensors[tensor_idx]))
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        let (shard, info) = self
            .get(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let data = shard.data(info);
        Ok(TensorView::from_info(&info.name, info, data))
    }

    pub fn summarize(&self) -> Summary {
        let mut dtypes = BTreeMap::new();
        let mut prefix_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_data_bytes = 0u64;
        let mut total_elements = 0i64;
        let mut largest = Vec::new();

        for (name, &(shard_idx, tensor_idx)) in &self.index {
            let t = &self.shards[shard_idx].tensors[tensor_idx];
            *dtypes.entry(t.dtype.as_str().to_string()).or_default() += 1;
            total_data_bytes += t.data_size as u64;
            total_elements += t.numel();
            *prefix_counts.entry(top_prefix(name)).or_default() += 1;
            largest.push((name.clone(), t.numel(), format_shape(&t.shape)));
        }

        largest.sort_by_key(|b| std::cmp::Reverse(b.1));
        largest.truncate(12);

        let mut top_prefixes: Vec<_> = prefix_counts.into_iter().collect();
        top_prefixes.sort_by_key(|b| std::cmp::Reverse(b.1));

        Summary {
            shard_count: self.shards.len(),
            tensor_count_index: self.weight_map.len(),
            tensor_count_headers: self.index.len(),
            total_file_bytes: self.shards.iter().map(|s| s.file_size() as u64).sum(),
            total_data_bytes,
            total_elements,
            dtypes,
            top_prefixes,
            largest,
        }
    }
}

/// Inference weights: safetensors or `.dgq` quantized.
pub enum WeightStore {
    Safetensors(SafetensorStore),
    Dgq(DgqStore),
}

impl WeightStore {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        if looks_like_dgq_dir(model_dir) {
            Ok(Self::Dgq(DgqStore::open(model_dir)?))
        } else {
            Ok(Self::Safetensors(SafetensorStore::open(model_dir)?))
        }
    }

    pub fn model_dir(&self) -> &Path {
        match self {
            Self::Safetensors(s) => &s.model_dir,
            Self::Dgq(s) => &s.model_dir,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Dgq(_))
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        match self {
            Self::Safetensors(s) => s.tensor(name),
            Self::Dgq(s) => s.tensor(name),
        }
    }

    pub fn summarize(&self) -> Summary {
        match self {
            Self::Safetensors(s) => s.summarize(),
            Self::Dgq(s) => s.summarize(),
        }
    }

    pub fn safetensor_metadata(&self) -> Option<&BTreeMap<String, serde_json::Value>> {
        match self {
            Self::Safetensors(s) => Some(&s.metadata),
            Self::Dgq(_) => None,
        }
    }
}

pub struct Summary {
    pub shard_count: usize,
    pub tensor_count_index: usize,
    pub tensor_count_headers: usize,
    pub total_file_bytes: u64,
    pub total_data_bytes: u64,
    pub total_elements: i64,
    pub dtypes: BTreeMap<String, usize>,
    pub top_prefixes: Vec<(String, usize)>,
    pub largest: Vec<(String, i64, String)>,
}

pub(crate) fn top_prefix(name: &str) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    parts[..parts.len().min(3)].join(".")
}

pub(crate) fn format_shape(shape: &[i64]) -> String {
    if shape.is_empty() {
        "[]".into()
    } else {
        format!(
            "[{}]",
            shape
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
