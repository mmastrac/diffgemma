//! Single embed-table row dequant (q8_row) for parity vs MLX / GPU gather.

use crate::Error;
use crate::dgq::block::dequant_row_q8;
use crate::dgq::layout::q8_row_bytes;
use crate::dgq::store::DgqStore;
use crate::model::embed::EMBED_TENSOR;
use crate::shaders::cpu::bf16_to_f32;
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;
pub const EMBED_SCALE: f32 = 53.065_998;

#[derive(Debug, Clone, Serialize)]
pub struct EmbedRowDump {
    pub schema_version: u32,
    pub source: String,
    pub token: u32,
    pub hidden_dim: usize,
    pub embed_scale: f32,
    pub scale_bf16_bits: u16,
    pub scale_f32: f32,
    pub dequant_raw: Vec<f32>,
    pub dequant_scaled: Vec<f32>,
    pub dequant_raw_l2: f32,
    pub dequant_scaled_l2: f32,
    pub gpu_scaled: Option<Vec<f32>>,
    pub bf16_ref_raw: Option<Vec<f32>>,
    pub bf16_ref_scaled: Option<Vec<f32>>,
}

fn vec_l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn dequant_embed_row_cpu(
    dgq: &DgqStore,
    token: u32,
    hidden: usize,
) -> Result<(Vec<f32>, u16, f32), Error> {
    let src = dgq.tensor_bytes(EMBED_TENSOR)?;
    let row_bytes = q8_row_bytes(hidden);
    let row = token as usize;
    let end = (row + 1)
        .checked_mul(row_bytes)
        .ok_or(Error::Runtime("embed row overflow"))?;
    if end > src.len() {
        return Err(Error::Runtime("embed token out of range"));
    }
    let row_src = &src[row * row_bytes..end];
    let scale_bits = u16::from_le_bytes([row_src[0], row_src[1]]);
    let scale_f32 = bf16_to_f32(scale_bits);
    let mut raw = vec![0.0f32; hidden];
    dequant_row_q8(row_src, hidden, &mut raw);
    Ok((raw, scale_bits, scale_f32))
}

pub fn load_bf16_embed_row(
    safetensors_dir: &Path,
    token: u32,
    hidden: usize,
) -> Result<Vec<f32>, Error> {
    use crate::weights::SafetensorStore;
    let store = SafetensorStore::open(safetensors_dir)?;
    let t = store.tensor(EMBED_TENSOR)?;
    let bf16 = t.bf16()?;
    let row = token as usize;
    let base = row * hidden;
    if base + hidden > bf16.len() {
        return Err(Error::Runtime("embed token out of range in bf16 ref"));
    }
    let bytes = bf16.as_bytes();
    let mut out = vec![0.0f32; hidden];
    for (i, o) in out.iter_mut().enumerate() {
        let off = (base + i) * 2;
        let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        *o = bf16_to_f32(bits);
    }
    Ok(out)
}

pub fn run_embed_row_dump(
    model_dir: &Path,
    token: u32,
    hidden: usize,
    bf16_ref_dir: Option<&Path>,
    gpu_scaled: Option<Vec<f32>>,
) -> Result<EmbedRowDump, Error> {
    let dgq = DgqStore::open(model_dir)?;
    let (raw, scale_bits, scale_f32) = dequant_embed_row_cpu(&dgq, token, hidden)?;
    let scaled: Vec<f32> = raw.iter().map(|v| v * EMBED_SCALE).collect();
    let (bf16_ref_raw, bf16_ref_scaled) = if let Some(dir) = bf16_ref_dir {
        match load_bf16_embed_row(dir, token, hidden) {
            Ok(ref_row) => {
                let ref_scaled: Vec<f32> = ref_row.iter().map(|v| v * EMBED_SCALE).collect();
                (Some(ref_row), Some(ref_scaled))
            }
            Err(err) => {
                eprintln!("warning: bf16 ref unavailable: {err}");
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let source = if gpu_scaled.is_some() {
        "rust-q8+gpu".into()
    } else {
        "rust-cpu-q8".into()
    };
    Ok(EmbedRowDump {
        schema_version: SCHEMA_VERSION,
        source,
        token,
        hidden_dim: hidden,
        embed_scale: EMBED_SCALE,
        scale_bf16_bits: scale_bits,
        scale_f32,
        dequant_raw_l2: vec_l2(&raw),
        dequant_scaled_l2: vec_l2(&scaled),
        dequant_raw: raw,
        dequant_scaled: scaled,
        gpu_scaled,
        bf16_ref_raw,
        bf16_ref_scaled,
    })
}

pub fn write_embed_row_dump(path: &Path, dump: &EmbedRowDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}
