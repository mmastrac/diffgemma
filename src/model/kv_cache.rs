use crate::config::TextConfig;
use crate::model::attention::AttentionParams;
use crate::safetensors::Error;

/// Per-layer encoder KV buffers (post-RoPE layout: `[kv_len, n_kv_heads, head_dim]` row-major).
#[derive(Debug, Clone)]
pub struct LayerKv {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKv {
    pub fn new_empty(n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            n_kv_heads,
            head_dim,
        }
    }

    pub fn kv_len(&self) -> usize {
        if self.n_kv_heads == 0 || self.head_dim == 0 {
            0
        } else {
            self.keys.len() / (self.n_kv_heads * self.head_dim)
        }
    }

    pub fn per_token_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn set_kv(&mut self, keys: &[f32], values: &[f32], seq_len: usize) -> Result<(), Error> {
        let dim = self.per_token_dim();
        if keys.len() != seq_len * dim || values.len() != seq_len * dim {
            return Err(Error::Format("kv buffer length mismatch"));
        }
        self.keys.resize(seq_len * dim, 0.0);
        self.values.resize(seq_len * dim, 0.0);
        self.keys.copy_from_slice(keys);
        self.values.copy_from_slice(values);
        Ok(())
    }

    pub fn append_kv(&mut self, keys: &[f32], values: &[f32], append_len: usize) -> Result<(), Error> {
        let dim = self.per_token_dim();
        if keys.len() != append_len * dim || values.len() != append_len * dim {
            return Err(Error::Format("kv append length mismatch"));
        }
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub kv_len: usize,
    pub layers: Vec<LayerKv>,
}

impl KvCache {
    pub fn empty(cfg: &TextConfig) -> Result<Self, Error> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            let params = AttentionParams::for_layer(cfg, layer)?;
            layers.push(LayerKv::new_empty(params.n_kv_heads, params.head_dim));
        }
        Ok(Self { kv_len: 0, layers })
    }

    pub fn dummy(cfg: &TextConfig, kv_len: usize, seed: u64) -> Result<Self, Error> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            let params = AttentionParams::for_layer(cfg, layer)?;
            let per_token = params.n_kv_heads * params.head_dim;
            let mut keys = vec![0.0f32; kv_len * per_token];
            let mut values = vec![0.0f32; kv_len * per_token];
            let mut state = seed.wrapping_add(layer as u64).wrapping_mul(1_314_159_265);
            for v in keys.iter_mut().chain(values.iter_mut()) {
                state = state
                    .wrapping_mul(6_966_169_279)
                    .wrapping_add(1_039_523_323);
                *v = ((state >> 16) as f32) / 65536.0 - 0.5;
            }
            layers.push(LayerKv {
                keys,
                values,
                n_kv_heads: params.n_kv_heads,
                head_dim: params.head_dim,
            });
        }
        Ok(Self { kv_len, layers })
    }

    pub fn layer(&self, layer: usize) -> Option<&LayerKv> {
        self.layers.get(layer)
    }

    pub fn layer_mut(&mut self, layer: usize) -> Option<&mut LayerKv> {
        self.layers.get_mut(layer)
    }

    pub fn validate_dims(&self, cfg: &TextConfig) -> Result<(), Error> {
        if self.layers.len() != cfg.num_hidden_layers {
            return Err(Error::Format("kv cache layer count mismatch"));
        }
        for (layer, kv) in self.layers.iter().enumerate() {
            let params = AttentionParams::for_layer(cfg, layer)?;
            if kv.kv_len() != self.kv_len {
                return Err(Error::Format("kv cache seq length mismatch"));
            }
            if kv.n_kv_heads != params.n_kv_heads || kv.head_dim != params.head_dim {
                return Err(Error::Format("kv cache head dims mismatch"));
            }
        }
        Ok(())
    }

    pub fn describe_layer(&self, layer: usize) -> String {
        if let Some(kv) = self.layer(layer) {
            format!(
                "layer {layer}: kv_len={} n_kv_heads={} head_dim={} keys={} values={}",
                kv.kv_len(),
                kv.n_kv_heads,
                kv.head_dim,
                kv.keys.len(),
                kv.values.len()
            )
        } else {
            format!("layer {layer}: missing")
        }
    }
}

#[derive(Clone, Copy)]
pub struct LayerKvView<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub kv_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<'a> LayerKvView<'a> {
    pub fn from_layer(layer: &'a LayerKv, kv_len: usize) -> Self {
        Self {
            keys: &layer.keys,
            values: &layer.values,
            kv_len,
            n_kv_heads: layer.n_kv_heads,
            head_dim: layer.head_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_kv_set_and_len() {
        let mut kv = LayerKv::new_empty(8, 256);
        let seq = 4usize;
        let dim = 8 * 256;
        let keys = vec![1.0f32; seq * dim];
        let values = vec![2.0f32; seq * dim];
        kv.set_kv(&keys, &values, seq).unwrap();
        assert_eq!(kv.kv_len(), 4);
        assert_eq!(kv.keys.len(), seq * dim);
    }
}
