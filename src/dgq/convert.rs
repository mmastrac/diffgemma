//! Safetensors → `.dgq` offline quantizer.

use crate::dgq::block::{quantize_bf16_matrix_q4, quantize_bf16_matrix_q8, quantize_expert_stack_q4};
use crate::dgq::layout::{
    align_offset, classify_tensor, DgqManifest, DgqTensorEntry, DgqTensorMeta, QuantKind,
    QuantProfile, BLOB_FILE, DGQ_VERSION, MANIFEST_FILE,
};
use crate::safetensors::Error;
use crate::weights::SafetensorStore;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub struct QuantizeOptions {
    pub source_dir: PathBuf,
    pub output_prefix: PathBuf,
    pub profile: QuantProfile,
}

pub struct QuantizeSummary {
    pub tensor_count: usize,
    pub blob_bytes: u64,
    pub q4_tensors: usize,
    pub q8_tensors: usize,
    pub raw_tensors: usize,
}

pub fn quantize_model(opts: QuantizeOptions) -> Result<QuantizeSummary, Error> {
    let out_dir = opts.output_prefix;
    fs::create_dir_all(&out_dir)?;
    copy_sidecar_files(&opts.source_dir, &out_dir)?;

    let store = SafetensorStore::open(&opts.source_dir)?;
    let blob_path = out_dir.join(BLOB_FILE);
    let mut blob = BufWriter::new(File::create(&blob_path)?);

    let mut offset = 0u64;
    let mut bytes_written = 0u64;
    let mut entries = Vec::with_capacity(store.weight_map.len());
    let mut q4_tensors = 0usize;
    let mut q8_tensors = 0usize;
    let mut raw_tensors = 0usize;

    let names: Vec<String> = store.weight_map.keys().cloned().collect();
    for (i, name) in names.iter().enumerate() {
        let (shard, info) = store
            .get(name)
            .ok_or_else(|| Error::NotFound(name.clone()))?;
        let src = shard.data(info);
        let kind = classify_tensor(name, &info.shape, opts.profile);

        offset = align_offset(offset);
        let start = offset;
        if start > bytes_written {
            let pad = (start - bytes_written) as usize;
            blob.write_all(&vec![0u8; pad])?;
            bytes_written += pad as u64;
        }
        let byte_len = match kind {
            QuantKind::Raw => {
                raw_tensors += 1;
                blob.write_all(src)?;
                src.len() as u64
            }
            QuantKind::Q4Block => {
                q4_tensors += 1;
                write_q4_tensor(&mut blob, src, &info.shape)?
            }
            QuantKind::Q8Row => {
                q8_tensors += 1;
                write_q8_tensor(&mut blob, src, &info.shape)?
            }
        };
        offset += byte_len;
        bytes_written += byte_len;

        entries.push(DgqTensorEntry {
            name: name.clone(),
            meta: DgqTensorMeta {
                kind: kind.as_str().to_string(),
                dtype: info.dtype.as_str().to_string(),
                shape: info.shape.clone(),
                offset: start,
                byte_len,
            },
        });

        if (i + 1) % 50 == 0 || i + 1 == names.len() {
            eprintln!(
                "  quantized {}/{} tensors ({:.2} GiB blob)...",
                i + 1,
                names.len(),
                offset as f64 / (1024.0_f64.powi(3))
            );
        }
    }
    blob.flush()?;

    let manifest = DgqManifest {
        version: DGQ_VERSION,
        profile: opts.profile,
        source_model: opts.source_dir.display().to_string(),
        blob_file: BLOB_FILE.to_string(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(out_dir.join(MANIFEST_FILE), manifest_json)?;

    Ok(QuantizeSummary {
        tensor_count: names.len(),
        blob_bytes: offset,
        q4_tensors,
        q8_tensors,
        raw_tensors,
    })
}

fn write_q4_tensor(out: &mut impl Write, src: &[u8], shape: &[i64]) -> Result<u64, Error> {
    match shape.len() {
        2 => {
            let out_dim = shape[0] as usize;
            let in_dim = shape[1] as usize;
            let need = crate::dgq::layout::q4_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_bf16_matrix_q4(src, out_dim, in_dim, &mut buf);
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        3 => {
            let experts = shape[0] as usize;
            let out_dim = shape[1] as usize;
            let in_dim = shape[2] as usize;
            let need = experts * crate::dgq::layout::q4_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_expert_stack_q4(src, experts, out_dim, in_dim, &mut buf)?;
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        _ => Err(Error::Format("q4 unsupported rank")),
    }
}

fn write_q8_tensor(out: &mut impl Write, src: &[u8], shape: &[i64]) -> Result<u64, Error> {
    if shape.len() != 2 {
        return Err(Error::Format("q8 expects rank 2"));
    }
    let out_dim = shape[0] as usize;
    let in_dim = shape[1] as usize;
    let need = crate::dgq::layout::q8_matrix_bytes(out_dim, in_dim);
    let mut buf = vec![0u8; need];
    quantize_bf16_matrix_q8(src, out_dim, in_dim, &mut buf);
    out.write_all(&buf)?;
    Ok(need as u64)
}

fn copy_sidecar_files(source: &std::path::Path, dest: &std::path::Path) -> Result<(), Error> {
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        let src = source.join(name);
        if src.is_file() {
            fs::copy(&src, dest.join(name))?;
        }
    }
    Ok(())
}
