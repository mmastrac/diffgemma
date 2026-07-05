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
    pub cosine: f32,
    pub rel_l2: f32,
    pub samples: usize,
}

fn weight_stats(orig: &[f32], dec: &[f32]) -> (f32, f32, f32, f32) {
    assert_eq!(orig.len(), dec.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&a, &b) in orig.iter().zip(dec.iter()) {
        let e = (a - b).abs();
        max_abs = max_abs.max(e);
        sum_abs += e;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let n = orig.len() as f32;
    let cos = if na > 1e-12 && nb > 1e-12 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    let l2_diff = orig
        .iter()
        .zip(dec.iter())
        .map(|(a, b)| a - b)
        .map(|d| d * d)
        .sum::<f32>()
        .sqrt();
    let rel_l2 = if na > 1e-12 { l2_diff / na.sqrt() } else { 0.0 };
    (max_abs, sum_abs / n, cos, rel_l2)
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
            "nvfp4_block" => QuantKind::Nvfp4Block,
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

        let orig_f32 = orig_bf16.to_f32_vec();
        let (max_abs, mean_abs, cosine, rel_l2) = weight_stats(&orig_f32, &dec);

        results.push(SpotCheckResult {
            name: name.to_string(),
            kind,
            max_abs_err: max_abs,
            mean_abs_err: mean_abs,
            cosine,
            rel_l2,
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
        cosine: {
            let dot: f32 = orig_row.iter().zip(dec_row.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = orig_row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = dec_row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na > 1e-12 && nb > 1e-12 {
                dot / (na * nb)
            } else {
                1.0
            }
        },
        rel_l2: {
            let l2: f32 = orig_row
                .iter()
                .zip(dec_row.iter())
                .map(|(a, b)| a - b)
                .map(|d| d * d)
                .sum::<f32>()
                .sqrt();
            let na: f32 = orig_row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na > 1e-12 { l2 / na } else { 0.0 }
        },
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
                "  {} {:?}: max={:.6} mean={:.6} cos={:.6} rel_l2={:.6} (n={})",
                r.name, r.kind, r.max_abs_err, r.mean_abs_err, r.cosine, r.rel_l2, r.samples
            );
            match r.kind {
                QuantKind::Q4Block => assert!(r.max_abs_err < 0.2, "q4 {}", r.name),
                QuantKind::Q6Block => assert!(r.max_abs_err < 0.05, "q6 {}", r.name),
                QuantKind::Nvfp4Block => assert!(r.max_abs_err < 0.35, "nvfp4 {}", r.name),
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

    #[test]
    fn roundtrip_q_proj_metrics() {
        let src = std::path::Path::new("model/transformer");
        if !src.exists() {
            eprintln!("skip: model/transformer missing");
            return;
        }
        let name = "model.decoder.layers.0.self_attn.q_proj.weight";
        let store = SafetensorStore::open(src).expect("open");
        let t = store.tensor(name).expect("tensor");
        let shape = t.shape_usize().unwrap();
        let out_dim = shape[0];
        let in_dim = shape[1];
        let bf16 = t.bf16().expect("bf16").as_bytes();
        let orig: Vec<f32> = bf16
            .chunks_exact(2)
            .map(|c| {
                crate::kernels::cpu::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))
            })
            .collect();

        let q4_path = std::path::Path::new("/tmp/quantized-weights");
        let nv_path = std::path::Path::new("/tmp/nvfp4-weights");
        for (label, dgq_path) in [("q4", q4_path), ("nvfp4", nv_path)] {
            if !dgq_path.join("model.dgq.json").exists() {
                eprintln!("skip {label}: missing {}", dgq_path.display());
                continue;
            }
            let dgq = DgqStore::open(dgq_path).expect("dgq");
            let on_disk = dgq.tensor_f32(name).expect("dequant");
            let (max_abs, mean_abs, cos, rel_l2) = super::weight_stats(&orig, &on_disk);
            eprintln!(
                "{name} on-disk {label}: max={max_abs:.6} mean={mean_abs:.6} cos={cos:.6} rel_l2={rel_l2:.6}"
            );
        }

        use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes};
        use crate::dgq::nvfp4::{dequant_matrix_nvfp4_payload, quantize_bf16_matrix_nvfp4};
        use crate::dgq::block::{dequant_matrix_q4, quantize_bf16_matrix_q4};

        let mut q4 = vec![0u8; q4_matrix_bytes(out_dim, in_dim)];
        quantize_bf16_matrix_q4(bf16, out_dim, in_dim, &mut q4);
        let mut q4_dec = vec![0.0f32; orig.len()];
        dequant_matrix_q4(&q4, out_dim, in_dim, &mut q4_dec);
        let q4_stats = super::weight_stats(&orig, &q4_dec);
        eprintln!(
            "{name} in-memory q4: max={:.6} mean={:.6} cos={:.6} rel_l2={:.6}",
            q4_stats.0, q4_stats.1, q4_stats.2, q4_stats.3
        );

        let mut nv = vec![0u8; nvfp4_matrix_bytes(out_dim, in_dim)];
        quantize_bf16_matrix_nvfp4(bf16, out_dim, in_dim, &mut nv);
        let mut nv_dec = vec![0.0f32; orig.len()];
        dequant_matrix_nvfp4_payload(&nv, out_dim, in_dim, &mut nv_dec).unwrap();
        let nv_stats = super::weight_stats(&orig, &nv_dec);
        eprintln!(
            "{name} in-memory nvfp4: max={:.6} mean={:.6} cos={:.6} rel_l2={:.6}",
            nv_stats.0, nv_stats.1, nv_stats.2, nv_stats.3
        );

        if nv_path.join("model.dgq.json").exists() {
            let dgq = DgqStore::open(nv_path).expect("dgq");
            let on_disk = dgq.tensor_f32(name).expect("dequant");
            let (max_abs, _, _, _) = super::weight_stats(&nv_dec, &on_disk);
            eprintln!("{name} on-disk vs in-memory nvfp4 max={max_abs:.9}");
        }
    }

    #[test]
    fn spot_check_nvfp4_weights() {
        let src = std::path::Path::new("model/transformer");
        let dgq = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq.join("model.dgq.json").exists() || !src.exists() {
            eprintln!("skip: model or /tmp/nvfp4-weights missing");
            return;
        }

        let tensors = [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.mlp.gate_proj.weight",
            "model.decoder.layers.0.experts.gate_up_proj",
        ];
        let results = spot_check(src, dgq, &tensors).expect("spot-check");
        for r in &results {
            eprintln!(
                "  {} {:?}: max={:.6} mean={:.6} cos={:.6} rel_l2={:.6} (n={})",
                r.name, r.kind, r.max_abs_err, r.mean_abs_err, r.cosine, r.rel_l2, r.samples
            );
            match r.kind {
                QuantKind::Nvfp4Block => assert!(r.max_abs_err < 0.35, "nvfp4 {}", r.name),
                _ => panic!("expected nvfp4"),
            }
        }
    }
}
