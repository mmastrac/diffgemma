//! Declarative denoise-step schedule (P3.6).
//!
//! `build_step_schedule` is the canonical stage list; `StepRuntime` interprets it via `StepEnc`.

use crate::metal::step_quant::{MoeExecutionStyle, StepBlockProfile};

/// One logical GPU stage in a monolithic denoise step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStage {
    // --- preamble (order matters) ---
    ScLogitRowstats,
    ScSoftembed,
    ScPreNorm,
    ScGateGemm,
    ScUpGemm,
    ScGlu,
    ScDownGemm,
    EmbedGather,
    /// `residual(hidden, dense, scal=0) -> hidden` after token embed (metal schedule step 3).
    EmbedScResidual,
    RmsNormHidden,

    // --- per decoder layer ---
    LayerInputNormQkv,
    LayerQkRopeKv,
    LayerAttention,
    LayerOProjPostAttn,
    LayerDenseFfn,
    LayerRouter,
    MoeBatchedGather,
    MoeBatchedGateUp,
    MoeBatchedSwiglu,
    MoeBatchedDown,
    MoeBatchedScatter,
    MoeGroupedScalar,
    LayerMoePostNorm,
    LayerMoePostCombine,

    // --- finish ---
    FinalNorm,
    LmHeadGemm,
    Softcap,
    SampleRowstats,
    SampleCommit,
    SampleApply,
    SampleWrite,
}

#[derive(Debug, Clone)]
pub struct StepSchedule {
    pub per_layer: Vec<StepStage>,
    pub finish: Vec<StepStage>,
}

/// Stages run only when `first_step == 0` (self-conditioning path).
pub fn sc_preamble_stages() -> &'static [StepStage] {
    const STAGES: &[StepStage] = &[
        StepStage::ScLogitRowstats,
        StepStage::ScSoftembed,
        StepStage::ScPreNorm,
        StepStage::ScGateGemm,
        StepStage::ScUpGemm,
        StepStage::ScGlu,
        StepStage::ScDownGemm,
    ];
    STAGES
}

pub fn core_preamble_stages() -> &'static [StepStage] {
    const STAGES: &[StepStage] = &[
        StepStage::EmbedGather,
        StepStage::EmbedScResidual,
        StepStage::RmsNormHidden,
    ];
    STAGES
}

pub fn moe_stages(profile: &StepBlockProfile) -> Vec<StepStage> {
    match profile.moe_style() {
        MoeExecutionStyle::BatchedGrouped => vec![
            StepStage::MoeBatchedGather,
            StepStage::MoeBatchedGateUp,
            StepStage::MoeBatchedSwiglu,
            StepStage::MoeBatchedDown,
            StepStage::MoeBatchedScatter,
        ],
        MoeExecutionStyle::ScalarPerExpert => vec![StepStage::MoeGroupedScalar],
    }
}

pub fn per_layer_stages(profile: &StepBlockProfile) -> Vec<StepStage> {
    let mut stages = vec![
        StepStage::LayerInputNormQkv,
        StepStage::LayerQkRopeKv,
        StepStage::LayerAttention,
        StepStage::LayerOProjPostAttn,
        StepStage::LayerDenseFfn,
        StepStage::LayerRouter,
    ];
    stages.extend(moe_stages(profile));
    stages.extend([StepStage::LayerMoePostNorm, StepStage::LayerMoePostCombine]);
    stages
}

pub fn finish_stages(include_sampler: bool) -> Vec<StepStage> {
    let mut stages = vec![
        StepStage::FinalNorm,
        StepStage::LmHeadGemm,
        StepStage::Softcap,
    ];
    if include_sampler {
        stages.extend([
            StepStage::SampleRowstats,
            StepStage::SampleCommit,
            StepStage::SampleApply,
            StepStage::SampleWrite,
        ]);
    }
    stages
}

pub fn build_step_schedule(profile: &StepBlockProfile, include_sampler: bool) -> StepSchedule {
    StepSchedule {
        per_layer: per_layer_stages(profile),
        finish: finish_stages(include_sampler),
    }
}

pub fn build_preamble(first_step: u32) -> Vec<StepStage> {
    let mut out = Vec::new();
    if first_step == 0 {
        out.extend_from_slice(sc_preamble_stages());
    }
    out.extend_from_slice(core_preamble_stages());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sub::QuantFormat;
    use crate::metal::step_quant::StepBlockProfile;

    #[test]
    fn batched_moe_schedule_has_five_moe_stages() {
        let profile = StepBlockProfile {
            format: QuantFormat::Q4Affine,
        };
        let layers = per_layer_stages(&profile);
        let moe_count = layers
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    StepStage::MoeBatchedGather
                        | StepStage::MoeBatchedGateUp
                        | StepStage::MoeBatchedSwiglu
                        | StepStage::MoeBatchedDown
                        | StepStage::MoeBatchedScatter
                        | StepStage::MoeGroupedScalar
                )
            })
            .count();
        assert_eq!(moe_count, 5);
    }

    #[test]
    fn core_preamble_includes_embed_residual() {
        let stages = core_preamble_stages();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[1], StepStage::EmbedScResidual);
    }

    #[test]
    fn finish_forward_only_omits_sampler() {
        let fin = finish_stages(false);
        assert!(!fin.iter().any(|s| matches!(s, StepStage::SampleRowstats)));
        let full = finish_stages(true);
        assert!(full.iter().any(|s| matches!(s, StepStage::SampleWrite)));
    }
}
