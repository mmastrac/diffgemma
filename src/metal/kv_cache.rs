//! GPU-resident encoder KV for decoder attention (prefix on GPU; canvas suffix patched per forward).

use crate::config::TextConfig;
use crate::metal::buffer::BufferPool;
use crate::model::attention::AttentionParams;
use crate::model::kv_cache::{KvCache, LayerKvView};
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

pub struct GpuKvLayer {
    keys: Retained<ProtocolObject<dyn MTLBuffer>>,
    values: Retained<ProtocolObject<dyn MTLBuffer>>,
    kv_dim: usize,
    capacity_tokens: usize,
}

pub struct GpuKvCache {
    layers: Vec<GpuKvLayer>,
    pub kv_len: usize,
}

impl GpuKvCache {
    /// `max_encoder_kv` upper bound on prompt+generated prefix; `max_canvas` per denoise forward.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        cfg: &TextConfig,
        max_encoder_kv: usize,
        max_canvas: usize,
    ) -> Result<Self, Error> {
        let capacity_tokens = max_encoder_kv + max_canvas;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer in 0..cfg.num_hidden_layers {
            let params = AttentionParams::for_layer(cfg, layer)?;
            let kv_dim = params.n_kv_heads * params.head_dim;
            let bytes = capacity_tokens * kv_dim * 4;
            let keys = device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or(Error::Format("GPU kv keys alloc failed"))?;
            let values = device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or(Error::Format("GPU kv values alloc failed"))?;
            layers.push(GpuKvLayer {
                keys,
                values,
                kv_dim,
                capacity_tokens,
            });
        }
        Ok(Self { layers, kv_len: 0 })
    }

    pub fn sync_all_from_cpu(
        &mut self,
        cpu: &KvCache,
        canvas_len: usize,
    ) -> Result<(), Error> {
        if cpu.kv_len + canvas_len > self.layers.first().map(|l| l.capacity_tokens).unwrap_or(0) {
            return Err(Error::Format("GPU kv cache capacity exceeded"));
        }
        self.kv_len = cpu.kv_len;
        for layer in 0..cpu.layers.len() {
            let cpu_layer = cpu
                .layer(layer)
                .ok_or(Error::Format("missing cpu kv layer"))?;
            let gpu_layer = self
                .layers
                .get_mut(layer)
                .ok_or(Error::Format("missing gpu kv layer"))?;
            let kv = LayerKvView::from_layer(cpu_layer, cpu.kv_len);
            sync_prefix(gpu_layer, kv)?;
        }
        Ok(())
    }

    pub fn canvas_k_elem_offset(&self, layer: usize) -> Result<usize, Error> {
        let gpu = self
            .layers
            .get(layer)
            .ok_or(Error::Format("missing gpu kv layer"))?;
        Ok(self.kv_len * gpu.kv_dim)
    }

    /// Upload pre-RoPE canvas K and raw V into the suffix of the GPU KV buffers.
    pub fn write_canvas_kv_pre_rope(
        &self,
        layer: usize,
        canvas_len: usize,
        k_canvas: &[f32],
        v_canvas: &[f32],
    ) -> Result<(), Error> {
        let gpu = self
            .layers
            .get(layer)
            .ok_or(Error::Format("missing gpu kv layer"))?;
        let kv_dim = gpu.kv_dim;
        let need = canvas_len * kv_dim;
        if k_canvas.len() != need || v_canvas.len() != need {
            return Err(Error::Format("canvas kv shape mismatch"));
        }
        if self.kv_len + canvas_len > gpu.capacity_tokens {
            return Err(Error::Format("GPU kv cache capacity exceeded"));
        }
        let byte_off = self.kv_len * kv_dim * 4;
        BufferPool::write_f32_at_offset(&gpu.keys, byte_off, k_canvas);
        BufferPool::write_f32_at_offset(&gpu.values, byte_off, v_canvas);
        Ok(())
    }

    pub fn advance_kv_len(&mut self, append_len: usize) -> Result<(), Error> {
        let cap = self
            .layers
            .first()
            .map(|l| l.capacity_tokens)
            .unwrap_or(0);
        if self.kv_len + append_len > cap {
            return Err(Error::Format("GPU kv cache capacity exceeded"));
        }
        self.kv_len += append_len;
        Ok(())
    }

    pub fn layer_buffers(
        &self,
        layer: usize,
    ) -> Result<
        (
            Retained<ProtocolObject<dyn MTLBuffer>>,
            Retained<ProtocolObject<dyn MTLBuffer>>,
        ),
        Error,
    > {
        let gpu = self
            .layers
            .get(layer)
            .ok_or(Error::Format("missing gpu kv layer"))?;
        Ok((gpu.keys.clone(), gpu.values.clone()))
    }
}

fn sync_prefix(gpu: &mut GpuKvLayer, kv: LayerKvView<'_>) -> Result<(), Error> {
    let kv_dim = gpu.kv_dim;
    if kv.n_kv_heads * kv.head_dim != kv_dim {
        return Err(Error::Format("gpu kv dim mismatch"));
    }
    if kv.kv_len > gpu.capacity_tokens {
        return Err(Error::Format("GPU kv cache capacity exceeded"));
    }
    let elems = kv.kv_len * kv_dim;
    if kv.keys.len() < elems || kv.values.len() < elems {
        return Err(Error::Format("cpu kv prefix too short"));
    }
    BufferPool::write_f32(&gpu.keys, &kv.keys[..elems]);
    BufferPool::write_f32(&gpu.values, &kv.values[..elems]);
    Ok(())
}
