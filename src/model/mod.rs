pub mod layer_weights;

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
        let model_dir = model_dir.as_ref();
        Ok(Self {
            config: ModelConfig::load(model_dir)?,
            weights: WeightStore::open(model_dir)?,
        })
    }
}
