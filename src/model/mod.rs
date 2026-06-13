pub mod attention;
pub mod decoder;
pub mod decoder_layer;
pub mod embed;
pub mod encoder;
pub mod kv_cache;
pub mod layer_weights;
pub mod mask;
pub mod moe;
pub mod self_conditioning;

use crate::config::ModelConfig;
use crate::safetensors::Error;
use crate::weights::WeightStore;
use std::path::Path;

pub struct Model {
    pub config: ModelConfig,
    pub weights: WeightStore,
}

impl Model {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        let model_dir = model_dir.as_ref();
        let weights = WeightStore::open(model_dir)?;
        if weights.is_quantized() {
            eprintln!(
                "loaded .dgq quantized weights from {} ({:.2?})",
                model_dir.display(),
                open_started.elapsed()
            );
        } else if weights.is_packed() {
            eprintln!(
                "loaded iris.pack weights from {} ({:.2?})",
                model_dir.display(),
                open_started.elapsed()
            );
        }
        Ok(Self {
            config: ModelConfig::load(model_dir)?,
            weights,
        })
    }
}
