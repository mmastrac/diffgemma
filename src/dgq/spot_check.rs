//! Spot-check: compare `.dgq` dequant vs original bf16 safetensors.

use crate::dgq::DgqStore;
use crate::dgq::layout::QuantKind;
use crate::kernels::cpu::bf16_to_f32;
use crate::safetensors::Error;
use crate::weights::SafetensorStore;

pub struct SpotCheckResult {
    pub name: String,
    pub kind: QuantKind,
    pub max_abs_err: f32,
    pub mean_abs_err: f32,
    pub samples: usize,
}

pub fn spot_check(
    safetensors_dir: &std::path::Path,
    dgq_dir: &std::path::Path,
    tensor_names: &[&str],
) -> Result<Vec<SpotCheckResult>, Error> {
    let src = SafetensorStore::open(safetensors_dir)?;
    let dgq = DgqStore::open(dgq_dir)?;
    let mut results = Vec::new();

    for &name in tensor_names {
        let entry = dgq
            .get_entry(name)
            .ok_or_else(|| Error::NotFound(name.to_string()))?;
        let kind = match entry.meta.kind.as_str() {
            "q4_block" => QuantKind::Q4Block,
            "q8_row" => QuantKind::Q8Row,
            "raw" => QuantKind::Raw,
            _ => return Err(Error::Format("unknown kind")),
        };

        let orig = src.tensor(name)?;
        let orig_bf16 = orig.bf16()?;
        let dec = dgq.tensor_f32(name)?;

        if orig_bf16.len() != dec.len() {
            return Err(Error::Format("spot-check numel mismatch"));
        }

        let (max_abs, sum_abs) = orig_bf16
            .to_f32_vec()
            .iter()
            .zip(dec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold((0.0f32, 0.0f32), |(mx, sum), e| (mx.max(e), sum + e));

        results.push(SpotCheckResult {
            name: name.to_string(),
            kind,
            max_abs_err: max_abs,
            mean_abs_err: sum_abs / dec.len() as f32,
            samples: dec.len(),
        });
    }
    Ok(results)
}

/// Sample first row of a large matrix (e.g. embed) without full dequant compare cost.
pub fn spot_check_embed_row(
    safetensors_dir: &std::path::Path,
    dgq_dir: &std::path::Path,
    name: &str,
    hidden: usize,
) -> Result<SpotCheckResult, Error> {
    let src = SafetensorStore::open(safetensors_dir)?;
    let dgq = DgqStore::open(dgq_dir)?;
    let orig = src.tensor(name)?;
    let orig_bf16 = orig.bf16()?;
    let row_src = &orig_bf16.as_bytes()[..hidden * 2];
    let mut orig_row = vec![0.0f32; hidden];
    for i in 0..hidden {
        let bits = u16::from_le_bytes([row_src[i * 2], row_src[i * 2 + 1]]);
        orig_row[i] = bf16_to_f32(bits);
    }

    let qbytes = dgq.tensor_bytes(name)?;
    let mut dec_row = vec![0.0f32; hidden];
    crate::dgq::block::dequant_row_q8(&qbytes[..2 + hidden], hidden, &mut dec_row);

    let (max_abs, sum_abs) = orig_row
        .iter()
        .zip(dec_row.iter())
        .map(|(a, b)| (a - b).abs())
        .fold((0.0f32, 0.0f32), |(mx, sum), e| (mx.max(e), sum + e));

    Ok(SpotCheckResult {
        name: format!("{name}[0]"),
        kind: QuantKind::Q8Row,
        max_abs_err: max_abs,
        mean_abs_err: sum_abs / hidden as f32,
        samples: hidden,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_check_q_proj_roundtrip_from_safetensors() {
        let src = std::path::Path::new("model/transformer");
        if !src.exists() {
            return;
        }
        let store = SafetensorStore::open(src).expect("open");
        let t = store
            .tensor("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("tensor");
        let shape = t.shape_usize().unwrap();
        let err = crate::dgq::dequant::q4_max_abs_error_vs_bf16(
            t.bf16().unwrap().as_bytes(),
            shape[0],
            shape[1],
        );
        eprintln!("q_proj in-memory q4 roundtrip max_err={err:.6}");
        assert!(err < 0.2, "q4 roundtrip err {err}");
    }

    #[test]
    fn spot_check_tmp_quantized_weights() {
        let src = std::path::Path::new("model/transformer");
        let dgq = std::path::Path::new("/tmp/quantized-weights");
        if !dgq.join("model.dgq.json").exists() || !src.exists() {
            eprintln!("skip: model or /tmp/quantized-weights missing");
            return;
        }

        let tensors = [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.mlp.gate_proj.weight",
            "model.decoder.layers.0.experts.gate_up_proj",
            "model.decoder.layers.0.router.proj.weight",
            "model.decoder.layers.0.input_layernorm.weight",
        ];
        let results = spot_check(src, dgq, &tensors).expect("spot-check");
        for r in &results {
            eprintln!(
                "  {} {:?}: max={:.6} mean={:.6} (n={})",
                r.name, r.kind, r.max_abs_err, r.mean_abs_err, r.samples
            );
            match r.kind {
                QuantKind::Q4Block => assert!(r.max_abs_err < 0.2, "q4 {}", r.name),
                QuantKind::Q8Row => assert!(r.max_abs_err < 0.05, "q8 {}", r.name),
                QuantKind::Raw => assert!(r.max_abs_err < 1e-5, "raw {}", r.name),
            }
        }

        let embed = spot_check_embed_row(src, dgq, "model.decoder.embed_tokens.weight", 2816)
            .expect("embed row");
        eprintln!(
            "  {} {:?}: max={:.6} mean={:.6}",
            embed.name, embed.kind, embed.max_abs_err, embed.mean_abs_err
        );
        assert!(embed.max_abs_err < 0.05);
    }
}
