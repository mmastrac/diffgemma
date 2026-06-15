use crate::kernels::sub::variant::KernelVariant;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::safetensors::Error;

pub struct GpuKernels {
    pub(crate) rms_norm: ComputePipeline,
    pub(crate) rms_norm_no_scale: ComputePipeline,
    pub(crate) vec_add: ComputePipeline,
    pub(crate) gather_rows: ComputePipeline,
    pub(crate) vec_scale: ComputePipeline,
    pub(crate) router_scale: ComputePipeline,
    pub(crate) gelu: ComputePipeline,
    pub(crate) swiglu_mul: ComputePipeline,
    pub(crate) gelu_swiglu_gate_up: ComputePipeline,
    pub(crate) softmax_rows: ComputePipeline,
    pub(crate) router_top_k: ComputePipeline,
    pub(crate) gather_prob_cols: ComputePipeline,
    pub(crate) vec_fill_zero: ComputePipeline,
    pub(crate) embed_gather: ComputePipeline,
    pub(crate) half_to_f32: ComputePipeline,
    pub(crate) pack_encoder_kv: ComputePipeline,
}

impl GpuKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, Error> {
        use crate::kernels::sub::{
            embed_gather, gather_prob_cols, gather_rows, gelu, half_to_f32, pack_encoder_kv,
            rms_norm_rows, router_scale_rows, router_top_k_rows, softmax_rows, swiglu,
            vec_add_inplace, vec_fill_zero, vec_scale_inplace,
        };
        let prod = KernelVariant::PRODUCTION;
        Ok(Self {
            vec_add: vec_add_inplace::pipeline_for(ctx, prod)?,
            gather_rows: gather_rows::pipeline_for(ctx, prod)?,
            vec_scale: vec_scale_inplace::pipeline_for(ctx, prod)?,
            router_scale: router_scale_rows::pipeline_for(ctx, prod)?,
            gelu: gelu::pipeline_for(ctx, prod)?,
            swiglu_mul: swiglu::pipeline_for(
                ctx,
                crate::kernels::sub::SwigluSplitVariant::DECODER_MUL,
                prod,
            )?,
            gelu_swiglu_gate_up: swiglu::pipeline_for_moe(ctx, prod)?,
            softmax_rows: softmax_rows::pipeline_for(ctx, prod)?,
            router_top_k: router_top_k_rows::pipeline_for(ctx, prod)?,
            gather_prob_cols: gather_prob_cols::pipeline_for(ctx, prod)?,
            vec_fill_zero: vec_fill_zero::pipeline_for(ctx, prod)?,
            rms_norm_no_scale: rms_norm_rows::pipeline_for(ctx, false, prod)?,
            rms_norm: rms_norm_rows::pipeline_for(ctx, true, prod)?,
            embed_gather: embed_gather::pipeline_for(ctx, prod)?,
            half_to_f32: half_to_f32::pipeline_for(ctx, prod)?,
            pack_encoder_kv: pack_encoder_kv::pipeline_for(ctx, prod)?,
        })
    }
}
