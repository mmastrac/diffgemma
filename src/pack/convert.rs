//! Offline conversion: safetensors → `iris.pack` (pre-transposed for Metal GEMM).

use crate::fast_slice::{transpose_bf16_weight_into, FastBf16Slice, FastSliceMut};
use crate::pack::layout::{
    align_offset, classify_tensor, PackManifest, PackTensorEntry, PackTensorMeta, TensorLayout,
    BLOB_FILE, MANIFEST_FILE, PACK_VERSION,
};
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use crate::weights::SafetensorStore;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct ConvertOptions {
    pub source_dir: PathBuf,
    pub output_dir: PathBuf,
}

pub struct ConvertSummary {
    pub tensor_count: usize,
    pub blob_bytes: u64,
    pub transposed_gemm: usize,
    pub transposed_experts: usize,
    pub raw_copied: usize,
}

pub fn convert_model(opts: ConvertOptions) -> Result<ConvertSummary, Error> {
    fs::create_dir_all(&opts.output_dir)?;
    copy_sidecar_files(&opts.source_dir, &opts.output_dir)?;

    let store = SafetensorStore::open(&opts.source_dir)?;
    let blob_path = opts.output_dir.join(BLOB_FILE);
    let mut blob = BufWriter::new(File::create(&blob_path)?);

    let mut offset = 0u64;
    let mut entries = Vec::with_capacity(store.weight_map.len());
    let mut transposed_gemm = 0usize;
    let mut transposed_experts = 0usize;
    let mut raw_copied = 0usize;

    let names: Vec<String> = store.weight_map.keys().cloned().collect();
    for (i, name) in names.iter().enumerate() {
        let (shard, info) = store
            .get(name)
            .ok_or_else(|| Error::NotFound(name.clone()))?;
        let src = shard.data(info);
        let layout = classify_tensor(name, &info.shape, info.dtype);

        offset = align_offset(offset);
        let start = offset;
        let byte_len = match layout {
            TensorLayout::Raw => {
                raw_copied += 1;
                blob.write_all(src)?;
                src.len() as u64
            }
            TensorLayout::GemmBf16T => {
                transposed_gemm += 1;
                let out_dim = info.shape[0] as usize;
                let in_dim = info.shape[1] as usize;
                write_gemm_transpose(&mut blob, src, out_dim, in_dim)?
            }
            TensorLayout::ExpertGateUpT => {
                transposed_experts += 1;
                let experts = info.shape[0] as usize;
                let out_dim = info.shape[1] as usize;
                let in_dim = info.shape[2] as usize;
                write_expert_gate_up(&mut blob, src, experts, out_dim, in_dim)?
            }
            TensorLayout::ExpertDownT => {
                transposed_experts += 1;
                let experts = info.shape[0] as usize;
                let out_dim = info.shape[1] as usize;
                let in_dim = info.shape[2] as usize;
                write_expert_down(&mut blob, src, experts, out_dim, in_dim)?
            }
        };
        offset += byte_len;

        entries.push(PackTensorEntry {
            name: name.clone(),
            meta: PackTensorMeta {
                layout: layout.as_str().to_string(),
                dtype: info.dtype.as_str().to_string(),
                shape: info.shape.clone(),
                offset: start,
                byte_len,
            },
        });

        if (i + 1) % 50 == 0 || i + 1 == names.len() {
            eprintln!(
                "  packed {}/{} tensors ({:.1} GiB blob)...",
                i + 1,
                names.len(),
                offset as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
    }
    blob.flush()?;

    let manifest = PackManifest {
        version: PACK_VERSION,
        source_model: opts.source_dir.display().to_string(),
        blob_file: BLOB_FILE.to_string(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(opts.output_dir.join(MANIFEST_FILE), manifest_json)?;

    Ok(ConvertSummary {
        tensor_count: names.len(),
        blob_bytes: offset,
        transposed_gemm,
        transposed_experts,
        raw_copied,
    })
}

fn write_gemm_transpose(
    out: &mut impl Write,
    src: &[u8],
    out_dim: usize,
    in_dim: usize,
) -> Result<u64, Error> {
    let slice = Bf16Slice::from_bytes(src);
    let fast = FastBf16Slice::from_bf16(slice);
    let mut tmp = vec![0u16; in_dim * out_dim];
    transpose_bf16_weight_into(fast, FastSliceMut::from_slice_mut(&mut tmp), out_dim, in_dim);
    write_u16_le(out, &tmp)?;
    Ok((tmp.len() * 2) as u64)
}

fn write_expert_gate_up(
    out: &mut impl Write,
    src: &[u8],
    experts: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<u64, Error> {
    let slice = Bf16Slice::from_bytes(src);
    let fast = FastBf16Slice::from_bf16(slice);
    let stride = out_dim * in_dim;
    let mut tmp = vec![0u16; stride];
    let mut total = 0u64;
    for e in 0..experts {
        let expert = unsafe { fast.slice_unchecked(e * stride, stride) };
        transpose_bf16_weight_into(
            expert,
            FastSliceMut::from_slice_mut(&mut tmp),
            out_dim,
            in_dim,
        );
        write_u16_le(out, &tmp)?;
        total += (tmp.len() * 2) as u64;
    }
    Ok(total)
}

fn write_expert_down(
    out: &mut impl Write,
    src: &[u8],
    experts: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<u64, Error> {
    let slice = Bf16Slice::from_bytes(src);
    let fast = FastBf16Slice::from_bf16(slice);
    let stride = out_dim * in_dim;
    let mut tmp = vec![0u16; in_dim * out_dim];
    let mut total = 0u64;
    for e in 0..experts {
        let expert = unsafe { fast.slice_unchecked(e * stride, stride) };
        transpose_bf16_weight_into(
            expert,
            FastSliceMut::from_slice_mut(&mut tmp),
            out_dim,
            in_dim,
        );
        write_u16_le(out, &tmp)?;
        total += (tmp.len() * 2) as u64;
    }
    Ok(total)
}

fn write_u16_le(out: &mut impl Write, data: &[u16]) -> Result<(), Error> {
    for &v in data {
        out.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn copy_sidecar_files(source: &Path, dest: &Path) -> Result<(), Error> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("model.safetensors")
            || name == MANIFEST_FILE
            || name == BLOB_FILE
        {
            continue;
        }
        fs::copy(&path, dest.join(name.as_ref()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_gemm_and_experts() {
        use crate::safetensors::DType;
        assert_eq!(
            classify_tensor(
                "model.decoder.layers.0.self_attn.q_proj.weight",
                &[4096, 2816],
                DType::BF16
            ),
            TensorLayout::GemmBf16T
        );
        assert_eq!(
            classify_tensor(
                "model.decoder.layers.0.experts.gate_up_proj",
                &[128, 352, 2816],
                DType::BF16
            ),
            TensorLayout::ExpertGateUpT
        );
        assert_eq!(
            classify_tensor(
                "model.decoder.layers.0.router.proj.weight",
                &[128, 2816],
                DType::BF16
            ),
            TensorLayout::Raw
        );
        assert_eq!(
            classify_tensor(
                "model.decoder.self_conditioning.gate_proj.weight",
                &[10240, 2816],
                DType::BF16
            ),
            TensorLayout::Raw
        );
        assert_eq!(
            classify_tensor("model.decoder.embed_tokens.weight", &[262144, 2816], DType::BF16),
            TensorLayout::Raw
        );
    }
}
