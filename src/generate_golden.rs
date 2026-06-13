//! Checked-in generate regression fixtures (token ids + run config).

use crate::generate::{GenerateConfig, GenerateOutput};
use crate::safetensors::Error;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Weight format tag for fixture selection (`safetensors` vs `dgq_q4` vs `monolithic`).
pub fn weights_profile_name(quantized: bool) -> &'static str {
    if quantized {
        "dgq_q4"
    } else {
        "safetensors"
    }
}

pub fn monolithic_weights_profile() -> &'static str {
    "monolithic"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateGolden {
    pub name: String,
    pub prompt: String,
    pub seed: u64,
    pub steps: usize,
    pub max_new_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_layers: Option<usize>,
    /// `safetensors` (bf16) or `dgq_q4`; omitted in legacy bf16 fixtures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights_profile: Option<String>,
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
        weights_profile: &str,
        out: &GenerateOutput,
    ) -> Self {
        Self {
            name: name.into(),
            prompt: prompt.into(),
            seed: gen_cfg.seed,
            steps,
            max_new_tokens: gen_cfg.max_new_tokens,
            max_layers: gen_cfg.max_layers,
            weights_profile: Some(weights_profile.to_string()),
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

    pub fn matches_config(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        steps: usize,
        weights_profile: &str,
    ) -> bool {
        self.prompt == prompt
            && self.seed == gen_cfg.seed
            && self.steps == steps
            && self.max_new_tokens == gen_cfg.max_new_tokens
            && self.max_layers == gen_cfg.max_layers
            && self.expected_weights_profile() == weights_profile
    }

    /// Legacy bf16 fixtures omit `weights_profile`; treat as safetensors.
    pub fn expected_weights_profile(&self) -> &str {
        self.weights_profile
            .as_deref()
            .unwrap_or(weights_profile_name(false))
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

pub fn infer_monolithic_fixture_name(
    prompt_text: Option<&str>,
    steps: usize,
    max_layers: Option<usize>,
) -> Option<String> {
    if prompt_text != Some("hello") {
        return None;
    }
    Some(match (steps, max_layers) {
        (4, Some(3)) => "monolithic_hello_steps4_layers3".into(),
        _ => return None,
    })
}

pub fn infer_fixture_name(
    prompt_text: Option<&str>,
    steps: usize,
    max_layers: Option<usize>,
    quantized: bool,
) -> Option<String> {
    if prompt_text != Some("Hello") {
        return None;
    }
    let prefix = if quantized { "dgq_" } else { "" };
    Some(match (steps, max_layers) {
        (1, None) => format!("{prefix}hello_steps1_full"),
        (1, Some(3)) => format!("{prefix}hello_steps1_layers3"),
        (2, Some(3)) => format!("{prefix}hello_steps2_layers3"),
        (2, None) => format!("{prefix}hello_steps2_full"),
        _ => return None,
    })
}

/// Thresholds for templated-chat quality (P1.5); not a token-id golden.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatQualityFixture {
    pub name: String,
    pub prompt: String,
    pub seed: u64,
    pub steps: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_layers: Option<usize>,
    pub min_real_new_tokens: usize,
    pub max_degenerate_ratio: f32,
    /// Fail if the first block stops before this many steps with a pad-heavy canvas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_block_steps_eff: Option<u32>,
}

impl ChatQualityFixture {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(Error::Io)?;
        serde_json::from_str(&text).map_err(Error::Json)
    }
}

pub fn default_chat_quality_fixture() -> ChatQualityFixture {
    ChatQualityFixture {
        name: "chat_quality_hello_layers3".into(),
        prompt: "Hello".into(),
        seed: 42,
        steps: 48,
        max_layers: Some(3),
        min_real_new_tokens: 8,
        max_degenerate_ratio: 0.9,
        min_block_steps_eff: Some(crate::sample::MIN_EARLY_STOP_STEPS),
    }
}

/// Assess new tokens after prompt; returns (total_new, real_non_degenerate).
pub fn count_new_tokens(out: &GenerateOutput, prompt_len: usize) -> (usize, usize) {
    let new = out.token_ids.get(prompt_len..).unwrap_or(&[]);
    let real = crate::sample::strip_degenerate_token_ids(new).len();
    (new.len(), real)
}

pub fn check_chat_quality(
    out: &GenerateOutput,
    prompt_len: usize,
    gate: &ChatQualityFixture,
) -> Result<(), String> {
    let (total_new, real_new) = count_new_tokens(out, prompt_len);
    if total_new == 0 {
        return Err("no new tokens emitted".into());
    }
    if real_new < gate.min_real_new_tokens {
        return Err(format!(
            "only {real_new} real new tokens (need >= {})",
            gate.min_real_new_tokens
        ));
    }
    let degenerate = total_new - real_new;
    let ratio = degenerate as f32 / total_new as f32;
    if ratio > gate.max_degenerate_ratio {
        return Err(format!(
            "degenerate ratio {ratio:.2} exceeds max {:.2} ({degenerate}/{total_new} pad/filler)",
            gate.max_degenerate_ratio
        ));
    }
    if let Some(min_steps) = gate.min_block_steps_eff {
        if let Some(&steps_eff) = out.block_steps_eff.first() {
            if steps_eff < min_steps && ratio > 0.5 {
                return Err(format!(
                    "block committed after {steps_eff} steps with {ratio:.0}% degenerate tokens (pad-aware stop regression)"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SamplerConfig;

    #[test]
    fn legacy_golden_defaults_to_safetensors_profile() {
        let g = GenerateGolden {
            name: "x".into(),
            prompt: "Hello".into(),
            seed: 42,
            steps: 1,
            max_new_tokens: 256,
            max_layers: None,
            weights_profile: None,
            denoise_steps_run: 1,
            blocks_committed: 1,
            token_ids: vec![1],
        };
        assert_eq!(g.expected_weights_profile(), "safetensors");
    }

    #[test]
    fn infer_monolithic_fixture_name_hello() {
        assert_eq!(
            infer_monolithic_fixture_name(Some("hello"), 4, Some(3)).as_deref(),
            Some("monolithic_hello_steps4_layers3")
        );
    }

    #[test]
    fn infer_fixture_name_dgq_prefix() {
        assert_eq!(
            infer_fixture_name(Some("Hello"), 1, Some(3), true).as_deref(),
            Some("dgq_hello_steps1_layers3")
        );
        assert_eq!(
            infer_fixture_name(Some("Hello"), 1, None, false).as_deref(),
            Some("hello_steps1_full")
        );
    }

    #[test]
    fn matches_config_checks_weights_profile() {
        let g = GenerateGolden {
            name: "x".into(),
            prompt: "Hello".into(),
            seed: 42,
            steps: 1,
            max_new_tokens: 256,
            max_layers: Some(3),
            weights_profile: Some("dgq_q4".into()),
            denoise_steps_run: 1,
            blocks_committed: 1,
            token_ids: vec![1],
        };
        let cfg = GenerateConfig {
            sampler: SamplerConfig::default(),
            max_new_tokens: 256,
            seed: 42,
            max_layers: Some(3),
            no_early_stop: false,
            deterministic: false,
        };
        assert!(g.matches_config("Hello", &cfg, 1, "dgq_q4"));
        assert!(!g.matches_config("Hello", &cfg, 1, "safetensors"));
    }

    #[test]
    fn chat_quality_rejects_pad_heavy_block() {
        use crate::sample::{FILLER_TOKEN_ID, PAD_TOKEN_ID};

        fn fixture_out(token_ids: Vec<u32>, block_steps: Vec<u32>) -> GenerateOutput {
            GenerateOutput {
                token_ids,
                denoise_steps_run: 2,
                blocks_committed: 1,
                block_steps_eff: block_steps,
                last_block_accept_hist: vec![16, 16],
                prefill_elapsed: std::time::Duration::ZERO,
                denoise_elapsed: std::time::Duration::ZERO,
                extend_elapsed: std::time::Duration::ZERO,
                #[cfg(all(feature = "metal", target_os = "macos"))]
                session_telemetry: crate::metal::SessionTelemetry::default(),
            }
        }

        let gate = default_chat_quality_fixture();
        let prompt_len = 10;
        let mut token_ids = vec![1u32; prompt_len];
        token_ids.extend(std::iter::repeat(PAD_TOKEN_ID).take(256));
        let out = fixture_out(token_ids, vec![2]);
        assert!(check_chat_quality(&out, prompt_len, &gate).is_err());

        let mut good_ids = vec![1u32; prompt_len];
        good_ids.extend((0..32).map(|i| 100 + i));
        good_ids.extend(std::iter::repeat(FILLER_TOKEN_ID).take(224));
        assert!(check_chat_quality(&fixture_out(good_ids, vec![20]), prompt_len, &gate).is_ok());
    }
}
