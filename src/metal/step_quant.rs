//! Step-kernel quantization profile: one algorithm, format selects kernels (AGENTS.md §4).
//!
//! Every field here is derived from the ACTUAL per-tensor `kind` stored in the
//! manifest (`DgqStore::get_entry(...).meta.kind`), never from the coarse
//! `QuantProfile` enum. A custom-class pack (`quantize --set class=format`)
//! can freely mix formats across experts / attention / dense-FFN, so the
//! profile enum alone is insufficient to know what format any given tensor
//! is actually stored in — see AGENTS.md §1 (fused/accelerated paths that
//! compute a different function than the reference).

use crate::Error;
use crate::dgq::layout::QuantKind;
use crate::shaders::QuantFormat;

/// Block-quantized weight format active for the MoE experts this step
/// runtime forward (gate_up and down must share one format — the batched
/// grouped-GEMM dispatch and blob-byte math key on a single value; see
/// `build_step_runtime`'s gate_up/down format-match assertion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBlockProfile {
    pub format: QuantFormat,
}

impl StepBlockProfile {
    /// Derive from an expert stack's actual manifest `kind`.
    pub fn from_kind(kind: QuantKind) -> Result<Self, Error> {
        Ok(Self {
            format: quant_format_from_kind(kind)?,
        })
    }

    pub fn is_nvfp4(self) -> bool {
        self.format == QuantFormat::NvFp4
    }

    pub fn moe_style(self) -> MoeExecutionStyle {
        moe_execution_style(self.format)
    }
}

/// Dense (per-token) linear weight format: attention q/k/v/o_proj and dense
/// FFN gate/up/down each resolve one of these independently (custom classes
/// let them diverge — e.g. `--set attn=nvfp4` with dense-FFN staying bf16).
/// `Q6Block` has no dense-linear kernel (q6 is experts-only, block-sparse) —
/// `from_kind` rejects it rather than silently mis-dispatching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseWeightFormat {
    Bf16,
    Q8,
    Block(QuantFormat),
}

impl DenseWeightFormat {
    pub fn from_kind(kind: QuantKind) -> Result<Self, Error> {
        match kind {
            QuantKind::Raw => Ok(Self::Bf16),
            QuantKind::Q8Row => Ok(Self::Q8),
            QuantKind::Q4Block => Ok(Self::Block(QuantFormat::Q4Affine)),
            QuantKind::Nvfp4Block => Ok(Self::Block(QuantFormat::NvFp4)),
            QuantKind::Q6Block => Err(Error::Format(
                "q6 has no dense-linear kernel (block-sparse/experts-only); \
                 supported dense/attn formats are raw, q8, q4, nvfp4",
            )),
        }
    }

    pub fn is_bf16(self) -> bool {
        matches!(self, Self::Bf16)
    }

    pub fn is_q8(self) -> bool {
        matches!(self, Self::Q8)
    }

    /// `Some(fmt)` iff this is a block-quantized (dequant-on-read) format.
    pub fn block_format(self) -> Option<QuantFormat> {
        match self {
            Self::Block(fmt) => Some(fmt),
            _ => None,
        }
    }
}

pub fn quant_format_from_kind(kind: QuantKind) -> Result<QuantFormat, Error> {
    match kind {
        QuantKind::Q4Block => Ok(QuantFormat::Q4Affine),
        QuantKind::Q6Block => Ok(QuantFormat::Q6),
        QuantKind::Nvfp4Block => Ok(QuantFormat::NvFp4),
        QuantKind::Q8Row | QuantKind::Raw => Err(Error::Format(
            "expected a block-quantized expert kind (q4_block/q6_block/nvfp4_block)",
        )),
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

pub fn moe_execution_style(format: QuantFormat) -> MoeExecutionStyle {
    match format {
        QuantFormat::Q4Affine | QuantFormat::NvFp4 | QuantFormat::Q6 => {
            MoeExecutionStyle::BatchedGrouped
        }
        // Unreachable for real expert formats; the scalar kernel survives as
        // the route-dump probe + oracle-test surface.
        _ => MoeExecutionStyle::ScalarPerExpert,
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
    fn expert_format_from_kind() {
        assert_eq!(
            quant_format_from_kind(QuantKind::Q4Block).unwrap(),
            QuantFormat::Q4Affine
        );
        assert_eq!(
            quant_format_from_kind(QuantKind::Nvfp4Block).unwrap(),
            QuantFormat::NvFp4
        );
        assert_eq!(
            quant_format_from_kind(QuantKind::Q6Block).unwrap(),
            QuantFormat::Q6
        );
        assert!(quant_format_from_kind(QuantKind::Raw).is_err());
        assert!(quant_format_from_kind(QuantKind::Q8Row).is_err());
    }

    #[test]
    fn dense_format_from_kind() {
        assert_eq!(
            DenseWeightFormat::from_kind(QuantKind::Raw).unwrap(),
            DenseWeightFormat::Bf16
        );
        assert_eq!(
            DenseWeightFormat::from_kind(QuantKind::Q8Row).unwrap(),
            DenseWeightFormat::Q8
        );
        assert_eq!(
            DenseWeightFormat::from_kind(QuantKind::Q4Block).unwrap(),
            DenseWeightFormat::Block(QuantFormat::Q4Affine)
        );
        assert_eq!(
            DenseWeightFormat::from_kind(QuantKind::Nvfp4Block).unwrap(),
            DenseWeightFormat::Block(QuantFormat::NvFp4)
        );
        assert!(DenseWeightFormat::from_kind(QuantKind::Q6Block).is_err());
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
