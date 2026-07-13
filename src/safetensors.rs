use memmap2::Mmap;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    Bool,
    Unknown,
}

impl DType {
    pub fn parse(s: &str) -> Self {
        match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "BOOL" => Self::Bool,
            _ => Self::Unknown,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "BOOL",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub data_offset: usize,
    pub data_size: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }
}

#[derive(Debug, Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<i64>,
    data_offsets: [i64; 2],
}

pub struct SafetensorsFile {
    pub path: PathBuf,
    mmap: Mmap,
    header_size: usize,
    pub tensors: Vec<TensorInfo>,
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// Model-file parse / layout / dtype problems (safetensors, `.dgq`).
    Format(&'static str),
    /// GPU / Metal failure: buffer allocation, encoder / command-buffer
    /// creation, pipeline compile, dispatch.
    Gpu(&'static str),
    /// Runtime / logic error: invalid argument, budget or limit exceeded,
    /// unsupported configuration, missing state.
    Runtime(&'static str),
    NotFound(String),
    DType {
        name: String,
        expected: DType,
        got: DType,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<i64>,
        got: Vec<i64>,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Gpu(msg) => write!(f, "gpu error: {msg}"),
            Self::Runtime(msg) => write!(f, "runtime error: {msg}"),
            Self::NotFound(name) => write!(f, "tensor not found: {name}"),
            Self::DType {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dtype mismatch for {name}: expected {}, got {}",
                    expected.as_str(),
                    got.as_str()
                )
            }
            Self::ShapeMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "shape mismatch for {name}: expected {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(Error::Format("file too small"));
        }

        let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if header_size > mmap.len() - 8 {
            return Err(Error::Format("invalid header size"));
        }

        let header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..8 + header_size])?;

        let mut tensors = Vec::with_capacity(header.len());
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let entry: HeaderEntry = serde_json::from_value(value)?;
            let data_offset = entry.data_offsets[0] as usize;
            let data_size = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
            tensors.push(TensorInfo {
                name,
                dtype: DType::parse(&entry.dtype),
                shape: entry.shape,
                data_offset,
                data_size,
            });
        }

        let data_region = mmap.len() - 8 - header_size;
        for t in &tensors {
            if t.data_offset + t.data_size > data_region {
                return Err(Error::Format("truncated tensor data"));
            }
        }

        Ok(Self {
            path,
            mmap,
            header_size,
            tensors,
        })
    }

    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Valid while this `SafetensorsFile` is alive.
    pub fn data(&self, tensor: &TensorInfo) -> &[u8] {
        let base = 8 + self.header_size + tensor.data_offset;
        &self.mmap[base..base + tensor.data_size]
    }
}
