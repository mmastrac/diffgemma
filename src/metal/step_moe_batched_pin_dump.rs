//! Tier-1 pin: per-stage GPU vs CPU oracle inside batched MoE (gather → GEMM×2 → swiglu → scatter).

use crate::kernels::sub::moe_batched_pin::MoeBatchedPinDump;
use crate::metal::step_kernel::{StepSmokeConfig, run_step_moe_batched_pin_capture};
use crate::safetensors::Error;
use std::path::Path;

pub use crate::kernels::sub::moe_batched_pin::print_pin_summary;

pub fn run_step_moe_batched_pin_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt: &str,
    layer: usize,
) -> Result<MoeBatchedPinDump, Error> {
    let mut dump = run_step_moe_batched_pin_capture(model_dir, cfg, layer)?;
    dump.prompt = prompt.to_string();
    Ok(dump)
}

pub fn write_step_moe_batched_pin_dump(path: &Path, dump: &MoeBatchedPinDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::step_kernel::{
        StepFinishMode, StepSmokeConfig, hello_chat_prefill_token_ids,
    };

    fn calgary_cfg(prefill: Vec<u32>) -> StepSmokeConfig {
        StepSmokeConfig {
            layers: 30,
            steps: 1,
            kv_len: 0,
            seed: 42,
            max_seq: 512,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: Some(prefill),
            no_early_stop: false,
        }
    }

    fn run_pin_layer(layer: usize, prompt: &str) -> Option<MoeBatchedPinDump> {
        let dir_buf = match crate::kernels::sub::test_util::dgq_model_dir() {
            Some(d) => d,
            None => std::path::PathBuf::from("/tmp/quantized-weights"),
        };
        let dir = dir_buf.as_path();
        if !dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return None;
        }
        let prefill = hello_chat_prefill_token_ids(dir).ok()?;
        let mut cfg = calgary_cfg(prefill);
        if prompt.contains("calgary") {
            // templated Calgary prompt uses same hello prefill ids in prior captures
        }
        let dump = run_step_moe_batched_pin_dump(dir, &cfg, prompt, layer).ok()?;
        Some(dump)
    }

    #[test]
    #[ignore = "tier-1 pin on demand: batched MoE L0 Calgary (model-gated, run with --ignored)"]
    fn batched_pin_l0_calgary_stages() {
        let dump = run_pin_layer(0, "What's the best way to get from calgary to namibia?")
            .expect("pin capture");
        print_pin_summary(&dump);
        assert!(
            dump.stages.gather >= 0.99,
            "gather cos={}",
            dump.stages.gather
        );
        assert!(
            dump.stages.gate_up_gemm >= 0.99,
            "gate_up_gemm cos={}",
            dump.stages.gate_up_gemm
        );
        assert!(
            dump.stages.swiglu_post >= 0.99,
            "swiglu_post cos={}",
            dump.stages.swiglu_post
        );
        assert!(
            dump.stages.swiglu_isolated >= 0.99,
            "swiglu_isolated cos={}",
            dump.stages.swiglu_isolated
        );
        assert!(dump.stages.down >= 0.99, "down cos={}", dump.stages.down);
        assert!(
            dump.stages.scatter >= 0.99,
            "scatter cos={}",
            dump.stages.scatter
        );
    }
}
