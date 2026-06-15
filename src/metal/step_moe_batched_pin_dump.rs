//! Tier-1 pin: per-stage GPU vs CPU oracle inside batched MoE (gather → GEMM×2 → swiglu → scatter).

use crate::kernels::sub::moe_batched_pin::MoeBatchedPinDump;
use crate::metal::step_kernel::{run_step_moe_batched_pin_capture, StepSmokeConfig};
use crate::safetensors::Error;
use std::path::Path;

pub use crate::kernels::sub::moe_batched_pin::print_pin_summary;

pub const SCHEMA_VERSION: u32 = 1;

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
    use crate::metal::step_kernel::{hello_chat_prefill_token_ids, StepFinishMode, StepSmokeConfig};

    fn calgary_cfg(prefill: Vec<u32>) -> StepSmokeConfig {
        StepSmokeConfig {
            layers: 30,
            steps: 1,
            kv_len: 0,
            seed: 42,
            max_seq: 512,
            finish: StepFinishMode::ForwardOnly,
            use_mps_q4: Some(false),
            prefill_token_ids: Some(prefill),
            no_early_stop: false,
            encoder_use_mps_q4: Some(false),
        }
    }

    fn run_pin_layer(layer: usize, prompt: &str) -> Option<MoeBatchedPinDump> {
        let dir = std::path::Path::new("/tmp/quantized-weights");
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
    #[ignore = "tier-1 pin on demand: batched MoE L0 Calgary (requires /tmp/quantized-weights)"]
    fn batched_pin_l0_calgary_stages() {
        let dump = run_pin_layer(
            0,
            "What's the best way to get from calgary to namibia?",
        )
        .expect("pin capture");
        print_pin_summary(&dump);
        assert!(dump.stages.gather >= 0.99, "gather cos={}", dump.stages.gather);
        assert!(dump.stages.gate_up >= 0.99, "gate_up cos={}", dump.stages.gate_up);
        assert!(dump.stages.swiglu >= 0.99, "swiglu cos={}", dump.stages.swiglu);
        assert!(dump.stages.down >= 0.99, "down cos={}", dump.stages.down);
        assert!(dump.stages.scatter >= 0.99, "scatter cos={}", dump.stages.scatter);
    }

    #[test]
    #[ignore = "tier-1 pin on demand: batched MoE L1 Calgary (requires /tmp/quantized-weights)"]
    fn batched_pin_l1_calgary_stages() {
        let dump = run_pin_layer(
            1,
            "What's the best way to get from calgary to namibia?",
        )
        .expect("pin capture");
        print_pin_summary(&dump);
        assert!(dump.stages.gather >= 0.99, "gather cos={}", dump.stages.gather);
        assert!(dump.stages.gate_up >= 0.99, "gate_up cos={}", dump.stages.gate_up);
        assert!(dump.stages.swiglu >= 0.99, "swiglu cos={}", dump.stages.swiglu);
        assert!(dump.stages.down >= 0.99, "down cos={}", dump.stages.down);
        assert!(dump.stages.scatter >= 0.99, "scatter cos={}", dump.stages.scatter);
    }
}
