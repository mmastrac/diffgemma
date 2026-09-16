//! Weight loading from a `.dgq` pack into f32.
//!
//! Every tensor the forward pass touches is materialized as f32 up front. That
//! is ~52 GiB for the full 26B model, which is why the CUDA box needs its other
//! GPU tenants stopped; it buys a forward pass with one dtype and no per-op
//! decode, which is the simplest thing that can be verified against the CPU
//! oracle.

use crate::config::{Error, ModelConfig};
use dgemm::format::block::{dequant_matrix_q4, dequant_matrix_q8};
use dgops::dgq::{DgqPack, DgqTensorEntry};

pub struct Weights {
    pub pack: DgqPack,
    pub hidden: usize,
    pub vocab: usize,
    pub n_layers: usize,
}

impl Weights {
    pub fn open(dir: impl AsRef<std::path::Path>, cfg: &ModelConfig) -> Result<Self, Error> {
        let pack = DgqPack::open(dir)?;
        Ok(Self {
            pack,
            hidden: cfg.text_config.hidden_size,
            vocab: cfg.text_config.vocab_size,
            n_layers: cfg.text_config.num_hidden_layers,
        })
    }

    pub fn has(&self, name: &str) -> bool {
        self.pack.get(name).is_some()
    }

    fn entry(&self, name: &str) -> Result<&DgqTensorEntry, Error> {
        self.pack
            .get(name)
            .ok_or_else(|| Error::Msg(format!("missing tensor {name}")))
    }

    /// A tensor as f32, whatever its stored kind.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let e = self.entry(name)?;
        let bytes = self.pack.bytes(name)?;
        let numel = e.numel();
        let mut out = vec![0.0f32; numel];
        match e.kind.as_str() {
            "raw" => {
                if e.dtype != "BF16" {
                    return Err(Error::Msg(format!(
                        "{name}: raw dtype {} unsupported",
                        e.dtype
                    )));
                }
                for (i, o) in out.iter_mut().enumerate() {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    *o = f32::from_bits((bits as u32) << 16);
                }
            }
            "q4_block" => {
                let shape: Vec<usize> = e.shape.iter().map(|&d| d as usize).collect();
                match shape.len() {
                    2 => dequant_matrix_q4(bytes, shape[0], shape[1], &mut out),
                    // Stacked experts: [n_experts, out, in], one q4 matrix each.
                    3 => {
                        let (ne, o, i) = (shape[0], shape[1], shape[2]);
                        let per = dgemm::format::layout::q4_matrix_bytes(o, i);
                        for x in 0..ne {
                            dequant_matrix_q4(
                                &bytes[x * per..(x + 1) * per],
                                o,
                                i,
                                &mut out[x * o * i..(x + 1) * o * i],
                            );
                        }
                    }
                    n => return Err(Error::Msg(format!("{name}: q4 rank {n}"))),
                }
            }
            "q8_row" => {
                let shape: Vec<usize> = e.shape.iter().map(|&d| d as usize).collect();
                if shape.len() != 2 {
                    return Err(Error::Msg(format!("{name}: q8 rank {}", shape.len())));
                }
                dequant_matrix_q8(bytes, shape[0], shape[1], &mut out);
            }
            "nvfp4_block" => {
                let shape: Vec<usize> = e.shape.iter().map(|&d| d as usize).collect();
                if shape.len() != 2 {
                    return Err(Error::Msg(format!("{name}: nvfp4 rank {}", shape.len())));
                }
                let global = dgemm::format::nvfp4::dequant_matrix_nvfp4_payload(
                    bytes, shape[0], shape[1], &mut out,
                )?;
                let _ = global;
            }
            other => return Err(Error::Msg(format!("{name}: unsupported kind {other}"))),
        }
        Ok(out)
    }

    /// A tensor's raw stored bytes (no decode).
    pub fn raw_bf16_bytes(&self, name: &str) -> Result<Vec<u8>, Error> {
        Ok(self.pack.bytes(name)?.to_vec())
    }

    /// A single row of a 2-D tensor, without materializing the whole tensor.
    pub fn row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, Error> {
        let e = self.entry(name)?;
        if e.shape.len() != 2 {
            return Err(Error::Msg(format!("{name}: not a matrix")));
        }
        let in_dim = e.shape[1] as usize;
        let bytes = self.pack.bytes(name)?;
        let mut out = vec![0.0f32; in_dim];
        match e.kind.as_str() {
            "raw" => {
                let off = row * in_dim * 2;
                for (i, o) in out.iter_mut().enumerate() {
                    let bits = u16::from_le_bytes([bytes[off + i * 2], bytes[off + i * 2 + 1]]);
                    *o = f32::from_bits((bits as u32) << 16);
                }
            }
            "q4_block" => {
                let row_bytes = dgemm::format::layout::q4_row_bytes(in_dim);
                dgemm::format::block::dequant_row_q4(
                    &bytes[row * row_bytes..(row + 1) * row_bytes],
                    in_dim,
                    &mut out,
                );
            }
            "q8_row" => {
                let row_bytes = dgemm::format::layout::q8_row_bytes(in_dim);
                dgemm::format::block::dequant_row_q8(
                    &bytes[row * row_bytes..(row + 1) * row_bytes],
                    in_dim,
                    &mut out,
                );
            }
            other => return Err(Error::Msg(format!("{name}: row kind {other} unsupported"))),
        }
        Ok(out)
    }

    pub fn scalar(&self, name: &str) -> Result<f32, Error> {
        Ok(self.tensor_f32(name)?[0])
    }
}

/// The per-layer tensor names, matching the pack manifest.
pub struct LayerKeys {
    pub input_layernorm: String,
    pub q_norm: String,
    pub k_norm: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub post_attention_layernorm: String,
    pub pre_feedforward_layernorm: String,
    pub post_feedforward_layernorm: String,
    pub post_feedforward_layernorm_1: String,
    pub post_feedforward_layernorm_2: String,
    pub pre_feedforward_layernorm_2: String,
    pub mlp_gate: String,
    pub mlp_up: String,
    pub mlp_down: String,
    pub router_proj: String,
    pub router_scale: String,
    pub router_per_expert_scale: String,
    pub experts_gate_up: String,
    pub experts_down: String,
    pub layer_scalar: String,
}

impl LayerKeys {
    pub fn new(layer: usize) -> Self {
        let p = format!("model.decoder.layers.{layer}");
        Self {
            input_layernorm: format!("{p}.input_layernorm.weight"),
            q_norm: format!("{p}.self_attn.q_norm.weight"),
            k_norm: format!("{p}.self_attn.k_norm.weight"),
            q_proj: format!("{p}.self_attn.q_proj.weight"),
            k_proj: format!("{p}.self_attn.k_proj.weight"),
            v_proj: format!("{p}.self_attn.v_proj.weight"),
            o_proj: format!("{p}.self_attn.o_proj.weight"),
            post_attention_layernorm: format!("{p}.post_attention_layernorm.weight"),
            pre_feedforward_layernorm: format!("{p}.pre_feedforward_layernorm.weight"),
            post_feedforward_layernorm: format!("{p}.post_feedforward_layernorm.weight"),
            post_feedforward_layernorm_1: format!("{p}.post_feedforward_layernorm_1.weight"),
            post_feedforward_layernorm_2: format!("{p}.post_feedforward_layernorm_2.weight"),
            pre_feedforward_layernorm_2: format!("{p}.pre_feedforward_layernorm_2.weight"),
            mlp_gate: format!("{p}.mlp.gate_proj.weight"),
            mlp_up: format!("{p}.mlp.up_proj.weight"),
            mlp_down: format!("{p}.mlp.down_proj.weight"),
            router_proj: format!("{p}.router.proj.weight"),
            router_scale: format!("{p}.router.scale"),
            router_per_expert_scale: format!("{p}.router.per_expert_scale"),
            experts_gate_up: format!("{p}.experts.gate_up_proj"),
            experts_down: format!("{p}.experts.down_proj"),
            layer_scalar: format!("{p}.layer_scalar"),
        }
    }
}
