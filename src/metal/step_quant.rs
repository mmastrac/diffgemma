//! Step-kernel quantization profile: one algorithm, format selects kernels (STRATEGY.md §4).

use crate::dgq::layout::QuantProfile;
use crate::kernels::sub::QuantFormat;

/// Block-quantized weight format active for this step runtime (dense + MoE experts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBlockProfile {
    pub format: QuantFormat,
}

impl StepBlockProfile {
    pub fn from_store_profile(p: QuantProfile) -> Self {
        Self {
            format: quant_format_from_profile(p),
        }
    }

    pub fn is_nvfp4(self) -> bool {
        self.format == QuantFormat::NvFp4
    }

    pub fn moe_style(self) -> MoeExecutionStyle {
        moe_execution_style(self.format)
    }
}

pub fn quant_format_from_profile(p: QuantProfile) -> QuantFormat {
    match p {
        QuantProfile::Nvfp4 => QuantFormat::NvFp4,
        QuantProfile::Q4 | QuantProfile::Q5 => QuantFormat::Q4Affine,
    }
}

/// MoE expert-forward control flow (independent of router/bucket/scatter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExecutionStyle {
    /// gather → grouped block GEMM (gate/up, down) → swiglu → weighted scatter
    BatchedGrouped,
    /// Legacy per-(expert, token) fused kernel (`moe_grouped` / `moe_grouped_nvfp4`)
    ScalarPerExpert,
}

/// Batched MoE grouped GEMM uses the fast path when enabled for supported formats.
pub fn batched_moe_enabled() -> bool {
    match std::env::var("DGQ_STEP_MOE_GROUPED") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

pub fn moe_execution_style(format: QuantFormat) -> MoeExecutionStyle {
    if batched_moe_enabled() {
        match format {
            QuantFormat::Q4Affine | QuantFormat::NvFp4 => MoeExecutionStyle::BatchedGrouped,
            _ => MoeExecutionStyle::ScalarPerExpert,
        }
    } else {
        MoeExecutionStyle::ScalarPerExpert
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockGroupedJob {
    pub w_byte_off: u64,
    pub groups_per_row: u32,
    pub _pad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_maps_q4_and_nvfp4() {
        assert_eq!(
            quant_format_from_profile(QuantProfile::Q4),
            QuantFormat::Q4Affine
        );
        assert_eq!(
            quant_format_from_profile(QuantProfile::Nvfp4),
            QuantFormat::NvFp4
        );
    }

    #[test]
    fn moe_batched_for_q4_and_nvfp4_when_enabled() {
        assert_eq!(
            moe_execution_style(QuantFormat::Q4Affine),
            MoeExecutionStyle::BatchedGrouped
        );
        assert_eq!(
            moe_execution_style(QuantFormat::NvFp4),
            MoeExecutionStyle::BatchedGrouped
        );
    }
}
