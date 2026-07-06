//! Arena scratch liveness for the declarative step interpreter (P3.6).
//!
//! Before each GPU stage, verifies every arena plane read is initialized for the
//! current step. Layer scratch planes are invalidated at each decoder-layer boundary.

use super::{LayerOffsets, ModelLayout, StepFinishMode, N_LAYERS};
use super::step_schedule::{build_preamble, build_step_schedule, StepStage};
use crate::metal::step_quant::StepBlockProfile;

/// Scratch plane in the monolithic arena (matches `ArenaLayout` fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArenaPlane {
    Hidden = 0,
    Resid,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    FfG,
    FfU,
    MoeIn,
    Dense,
    MoeOut,
    Soft,
    Stream,
    Tmp,
    RsSc,
    RsSamp,
}

/// Planes cleared at each decoder-layer boundary (hidden carries across layers).
const LAYER_SCRATCH: [ArenaPlane; 11] = [
    ArenaPlane::AttnQ,
    ArenaPlane::AttnK,
    ArenaPlane::AttnV,
    ArenaPlane::AttnO,
    ArenaPlane::FfG,
    ArenaPlane::FfU,
    ArenaPlane::MoeIn,
    ArenaPlane::Dense,
    ArenaPlane::MoeOut,
    ArenaPlane::Stream,
    ArenaPlane::Tmp,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageArenaAccess {
    pub reads: &'static [ArenaPlane],
    pub writes: &'static [ArenaPlane],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaLivenessError {
    UninitializedRead {
        stage: StepStage,
        layer: usize,
        plane: ArenaPlane,
    },
}

impl std::fmt::Display for ArenaLivenessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UninitializedRead { stage, layer, plane } => {
                write!(
                    f,
                    "arena liveness: {:?} layer {layer} read uninitialized plane {:?}",
                    stage, plane
                )
            }
        }
    }
}

impl std::error::Error for ArenaLivenessError {}

#[derive(Clone, Copy)]
struct LivenessCtx<'a> {
    first_step: u32,
    layer: usize,
    layer_info: Option<&'a LayerOffsets>,
    finish: StepFinishMode,
}

#[derive(Clone, Debug, Default)]
struct ArenaLiveness {
    valid: u16,
}

impl ArenaLiveness {
    fn new() -> Self {
        Self::default()
    }

    fn bit(p: ArenaPlane) -> u16 {
        1u16 << (p as u8)
    }

    fn is_valid(&self, p: ArenaPlane) -> bool {
        self.valid & Self::bit(p) != 0
    }

    fn mark_valid(&mut self, p: ArenaPlane) {
        self.valid |= Self::bit(p);
    }

    fn invalidate(&mut self, p: ArenaPlane) {
        self.valid &= !Self::bit(p);
    }

    fn invalidate_layer_scratch(&mut self) {
        for &p in &LAYER_SCRATCH {
            self.invalidate(p);
        }
    }

    fn apply_access(
        &mut self,
        stage: StepStage,
        layer: usize,
        access: StageArenaAccess,
    ) -> Result<(), ArenaLivenessError> {
        for &p in access.reads {
            if !self.is_valid(p) {
                return Err(ArenaLivenessError::UninitializedRead { stage, layer, plane: p });
            }
        }
        for &p in access.writes {
            self.mark_valid(p);
        }
        Ok(())
    }

    fn apply_stage(
        &mut self,
        stage: StepStage,
        ctx: LivenessCtx<'_>,
    ) -> Result<(), ArenaLivenessError> {
        let access = stage_arena_access(stage, ctx);
        if access.reads.is_empty() && access.writes.is_empty() {
            return Ok(());
        }
        self.apply_access(stage, ctx.layer, access)?;
        apply_stage_aliases(stage, ctx, self);
        Ok(())
    }
}

fn apply_stage_aliases(stage: StepStage, ctx: LivenessCtx<'_>, live: &mut ArenaLiveness) {
    let _ = stage;
    if let Some(l) = ctx.layer_info {
        if l.v_proj == 0 && matches!(stage, StepStage::LayerInputNormQkv) {
            if live.is_valid(ArenaPlane::AttnK) {
                live.mark_valid(ArenaPlane::AttnV);
            }
        }
    }
}

pub fn stage_arena_access(stage: StepStage, ctx: LivenessCtx<'_>) -> StageArenaAccess {
    use StepStage::*;
    match stage {
        ScLogitRowstats => StageArenaAccess {
            reads: &[],
            writes: &[ArenaPlane::RsSc],
        },
        ScSoftembed => StageArenaAccess {
            reads: &[ArenaPlane::RsSc],
            writes: &[ArenaPlane::Soft],
        },
        ScPreNorm => StageArenaAccess {
            reads: &[ArenaPlane::Soft],
            writes: &[ArenaPlane::Tmp],
        },
        ScGateGemm => StageArenaAccess {
            reads: &[ArenaPlane::Tmp],
            writes: &[ArenaPlane::FfG],
        },
        ScUpGemm => StageArenaAccess {
            reads: &[ArenaPlane::Tmp],
            writes: &[ArenaPlane::FfU],
        },
        ScGlu => StageArenaAccess {
            reads: &[ArenaPlane::FfG, ArenaPlane::FfU],
            writes: &[ArenaPlane::FfG],
        },
        ScDownGemm => StageArenaAccess {
            reads: &[ArenaPlane::FfG],
            writes: &[ArenaPlane::Dense],
        },
        EmbedGather => StageArenaAccess {
            reads: &[],
            writes: &[ArenaPlane::Hidden],
        },
        EmbedScResidual => StageArenaAccess {
            reads: &[ArenaPlane::Hidden, ArenaPlane::Dense],
            writes: &[ArenaPlane::Hidden],
        },
        RmsNormHidden => StageArenaAccess {
            reads: &[ArenaPlane::Hidden],
            writes: &[ArenaPlane::Hidden],
        },
        LayerInputNormQkv => {
            let mut writes: &'static [ArenaPlane] = &[ArenaPlane::Tmp, ArenaPlane::AttnQ, ArenaPlane::AttnK];
            if ctx.layer_info.is_some_and(|l| l.v_proj != 0) {
                writes = &[ArenaPlane::Tmp, ArenaPlane::AttnQ, ArenaPlane::AttnK, ArenaPlane::AttnV];
            }
            StageArenaAccess {
                reads: &[ArenaPlane::Hidden],
                writes,
            }
        }
        LayerQkRopeKv => StageArenaAccess {
            reads: &[ArenaPlane::AttnQ, ArenaPlane::AttnK, ArenaPlane::AttnV],
            writes: &[ArenaPlane::AttnQ, ArenaPlane::AttnK],
        },
        LayerAttention => StageArenaAccess {
            reads: &[ArenaPlane::AttnQ],
            writes: &[ArenaPlane::AttnO],
        },
        LayerOProjPostAttn => StageArenaAccess {
            reads: &[ArenaPlane::AttnO, ArenaPlane::Hidden],
            writes: &[ArenaPlane::Tmp, ArenaPlane::Stream],
        },
        LayerDenseFfn => StageArenaAccess {
            reads: &[ArenaPlane::Stream],
            writes: &[ArenaPlane::Tmp, ArenaPlane::FfG, ArenaPlane::FfU, ArenaPlane::Dense],
        },
        LayerRouter => StageArenaAccess {
            reads: &[ArenaPlane::Stream],
            writes: &[ArenaPlane::MoeIn, ArenaPlane::MoeOut],
        },
        MoeBatchedGather => StageArenaAccess {
            reads: &[ArenaPlane::MoeIn],
            writes: &[],
        },
        MoeBatchedGateUp | MoeBatchedSwiglu | MoeBatchedDown => StageArenaAccess {
            reads: &[],
            writes: &[],
        },
        MoeBatchedScatter => StageArenaAccess {
            reads: &[],
            writes: &[ArenaPlane::MoeOut],
        },
        MoeGroupedScalar => StageArenaAccess {
            reads: &[ArenaPlane::MoeIn],
            writes: &[ArenaPlane::MoeOut],
        },
        LayerMoePostNorm => StageArenaAccess {
            reads: &[ArenaPlane::MoeOut],
            writes: &[ArenaPlane::MoeIn],
        },
        LayerMoePostCombine => StageArenaAccess {
            reads: &[ArenaPlane::Dense, ArenaPlane::MoeIn, ArenaPlane::Stream],
            writes: &[ArenaPlane::Tmp, ArenaPlane::Hidden],
        },
        FinalNorm => StageArenaAccess {
            reads: &[ArenaPlane::Hidden],
            writes: &[ArenaPlane::Tmp],
        },
        LmHeadGemm => StageArenaAccess {
            reads: &[ArenaPlane::Tmp],
            writes: &[],
        },
        Softcap => StageArenaAccess {
            reads: &[],
            writes: &[],
        },
        SampleRowstats => {
            if ctx.finish == StepFinishMode::Full {
                StageArenaAccess {
                    reads: &[],
                    writes: &[ArenaPlane::RsSamp],
                }
            } else {
                StageArenaAccess {
                    reads: &[],
                    writes: &[],
                }
            }
        }
        SampleCommit => StageArenaAccess {
            reads: &[],
            writes: &[],
        },
        SampleApply => {
            if ctx.finish == StepFinishMode::Full {
                StageArenaAccess {
                    reads: &[ArenaPlane::RsSamp],
                    writes: &[],
                }
            } else {
                StageArenaAccess {
                    reads: &[],
                    writes: &[],
                }
            }
        }
        SampleWrite => StageArenaAccess {
            reads: &[],
            writes: &[],
        },
    }
}

/// Walk the full step schedule and verify arena plane liveness (no GPU).
pub fn check_step_arena_liveness(
    profile: &StepBlockProfile,
    layout: &ModelLayout,
    layers: usize,
    first_step: u32,
    finish: StepFinishMode,
) -> Result<(), ArenaLivenessError> {
    let schedule = build_step_schedule(profile, finish == StepFinishMode::Full);
    let mut live = ArenaLiveness::new();
    // Arena is zeroed each step; when SC MLP is skipped, dense is never written but reads as zero.
    if first_step > 0 {
        live.mark_valid(ArenaPlane::Dense);
    }

    for stage in build_preamble(first_step) {
        let ctx = LivenessCtx {
            first_step,
            layer: 0,
            layer_info: None,
            finish,
        };
        live.apply_stage(stage, ctx)?;
    }

    let layer_count = layers.min(N_LAYERS);
    for layer in 0..layer_count {
        live.invalidate_layer_scratch();
        let layer_info = &layout.layers[layer];
        for &stage in &schedule.per_layer {
            let ctx = LivenessCtx {
                first_step,
                layer,
                layer_info: Some(layer_info),
                finish,
            };
            live.apply_stage(stage, ctx)?;
        }
    }

    let mut sampler_checked = false;
    for &stage in &schedule.finish {
        if matches!(
            stage,
            StepStage::SampleRowstats
                | StepStage::SampleCommit
                | StepStage::SampleApply
                | StepStage::SampleWrite
        ) {
            if !sampler_checked && finish == StepFinishMode::Full {
                let ctx = LivenessCtx {
                    first_step,
                    layer: 0,
                    layer_info: None,
                    finish,
                };
                live.apply_stage(StepStage::SampleRowstats, ctx)?;
                live.apply_stage(StepStage::SampleApply, ctx)?;
                sampler_checked = true;
            }
            continue;
        }
        let ctx = LivenessCtx {
            first_step,
            layer: 0,
            layer_info: None,
            finish,
        };
        live.apply_stage(stage, ctx)?;
    }

    Ok(())
}

pub fn runtime_arena_liveness_enabled() -> bool {
    crate::kernels::sub::variant::runtime_assert_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sub::QuantFormat;
    use crate::metal::step_quant::StepBlockProfile;

    fn test_layout() -> ModelLayout {
        let mut layers = [LayerOffsets {
            input_ln: 0,
            q_proj: 0,
            q_norm: 0,
            k_proj: 0,
            k_norm: 0,
            v_proj: 1,
            o_proj: 0,
            post_attn_ln: 0,
            pre_ff_ln: 0,
            mlp_gate: 0,
            mlp_up: 0,
            mlp_down: 0,
            post_ff_ln_1: 0,
            router_scale: 0,
            router_proj: 0,
            per_expert_scale: 0,
            pre_ff_ln_2: 0,
            experts_gate_up: 0,
            experts_down: 0,
            post_ff_ln_2: 0,
            post_ff_ln: 0,
            layer_scalar: 0,
            kv_region: 0,
            head_dim: 256,
            n_kv_heads: 2,
            is_full: 0,
            kv_ring_mask: 0,
        }; N_LAYERS];
        for l in layers.iter_mut().take(26) {
            l.is_full = 1;
            l.v_proj = 0;
        }
        ModelLayout {
            embed: 0,
            sc_pre_norm: 0,
            sc_gate: 0,
            sc_up: 0,
            sc_down: 0,
            final_norm: 0,
            layers,
        }
    }

    #[test]
    fn production_schedule_passes_liveness_first_step() {
        let profile = StepBlockProfile {
            format: QuantFormat::Q4Affine,
        };
        check_step_arena_liveness(&profile, &test_layout(), 30, 0, StepFinishMode::Full)
            .expect("30-layer first-step schedule should be live");
    }

    #[test]
    fn production_schedule_passes_liveness_later_step() {
        let profile = StepBlockProfile {
            format: QuantFormat::Q4Affine,
        };
        check_step_arena_liveness(&profile, &test_layout(), 30, 1, StepFinishMode::Full)
            .expect("30-layer later-step schedule should be live");
    }

    #[test]
    fn forward_only_finish_passes_liveness() {
        let profile = StepBlockProfile {
            format: QuantFormat::Q4Affine,
        };
        check_step_arena_liveness(&profile, &test_layout(), 30, 0, StepFinishMode::ForwardOnly)
            .expect("forward-only finish should be live");
    }

    #[test]
    fn uninitialized_read_is_caught() {
        let mut live = ArenaLiveness::new();
        let ctx = LivenessCtx {
            first_step: 1,
            layer: 0,
            layer_info: None,
            finish: StepFinishMode::Full,
        };
        let err = live
            .apply_stage(StepStage::LayerInputNormQkv, ctx)
            .expect_err("hidden must be initialized");
        assert!(matches!(
            err,
            ArenaLivenessError::UninitializedRead {
                plane: ArenaPlane::Hidden,
                ..
            }
        ));
    }
}
