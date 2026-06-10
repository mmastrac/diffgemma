use crate::config::TextConfig;
use crate::model::attention::AttentionParams;
use crate::safetensors::Error;

/// Per-layer encoder KV buffers (post-RoPE layout: `[kv_len, n_kv_heads, head_dim]` row-major).
pub struct LayerKv {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKv {
    pub fn kv_len(&self) -> usize {
        if self.n_kv_heads == 0 || self.head_dim == 0 {
            0
        } else {
            self.keys.len() / (self.n_kv_heads * self.head_dim)
        }
    }
}

pub struct KvCache {
    pub kv_len: usize,
    pub layers: Vec<LayerKv>,
}

impl KvCache {
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
}

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
