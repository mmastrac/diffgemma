//! Checked-in generate regression fixtures (token ids + run config).

use crate::generate::{GenerateConfig, GenerateOutput};
use crate::safetensors::Error;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateGolden {
    pub name: String,
    pub prompt: String,
    pub seed: u64,
    pub steps: usize,
    pub max_new_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_layers: Option<usize>,
    pub denoise_steps_run: usize,
    pub blocks_committed: usize,
    pub token_ids: Vec<u32>,
}

impl GenerateGolden {
    pub fn from_run(
        name: impl Into<String>,
        prompt: impl Into<String>,
        gen_cfg: &GenerateConfig,
        steps: usize,
        out: &GenerateOutput,
    ) -> Self {
        Self {
            name: name.into(),
            prompt: prompt.into(),
            seed: gen_cfg.seed,
            steps,
            max_new_tokens: gen_cfg.max_new_tokens,
            max_layers: gen_cfg.max_layers,
            denoise_steps_run: out.denoise_steps_run,
            blocks_committed: out.blocks_committed,
            token_ids: out.token_ids.clone(),
        }
    }

    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(Error::Io)?;
        serde_json::from_str(&text).map_err(Error::Json)
    }

    pub fn write(&self, path: &Path) -> Result<(), Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(Error::Io)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(Error::Json)?;
        std::fs::write(path, text).map_err(Error::Io)
    }

    pub fn matches_config(&self, prompt: &str, gen_cfg: &GenerateConfig, steps: usize) -> bool {
        self.prompt == prompt
            && self.seed == gen_cfg.seed
            && self.steps == steps
            && self.max_new_tokens == gen_cfg.max_new_tokens
            && self.max_layers == gen_cfg.max_layers
    }

    pub fn compare(&self, out: &GenerateOutput) -> Result<(), String> {
        if out.denoise_steps_run != self.denoise_steps_run {
            return Err(format!(
                "denoise_steps_run: expected {}, got {}",
                self.denoise_steps_run, out.denoise_steps_run
            ));
        }
        if out.blocks_committed != self.blocks_committed {
            return Err(format!(
                "blocks_committed: expected {}, got {}",
                self.blocks_committed, out.blocks_committed
            ));
        }
        if out.token_ids != self.token_ids {
            let first = out
                .token_ids
                .iter()
                .zip(self.token_ids.iter())
                .position(|(a, b)| a != b);
            return Err(format!(
                "token_ids mismatch at index {first:?} (got len={}, expected len={})",
                out.token_ids.len(),
                self.token_ids.len()
            ));
        }
        Ok(())
    }
}

pub fn default_fixture_dir() -> &'static Path {
    Path::new("fixtures/generate")
}

pub fn resolve_fixture(name: &str) -> std::path::PathBuf {
    let path = Path::new(name);
    if path.extension().is_some() {
        path.to_path_buf()
    } else {
        default_fixture_dir().join(format!("{name}.json"))
    }
}
