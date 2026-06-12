//! GPU sampler kernels (logit post-process, entropy, argmax, categorical sample).

use crate::metal::device::{ComputePipeline, MetalContext};
use crate::safetensors::Error;

const SAMPLER_SHADER: &str = include_str!("../../shaders/sampler.metal");

pub struct GpuSamplerKernels {
    pub copy_f32: ComputePipeline,
    pub logit_softcapping: ComputePipeline,
    pub scale_logits: ComputePipeline,
    pub scatter_vocab_chunk: ComputePipeline,
    pub argmax_rows: ComputePipeline,
    pub row_entropy: ComputePipeline,
    pub sample_from_probs_rows: ComputePipeline,
}

impl GpuSamplerKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, Error> {
        Ok(Self {
            copy_f32: ctx.compile_kernel(SAMPLER_SHADER, "copy_f32")?,
            logit_softcapping: ctx.compile_kernel(SAMPLER_SHADER, "logit_softcapping")?,
            scale_logits: ctx.compile_kernel(SAMPLER_SHADER, "scale_logits")?,
            scatter_vocab_chunk: ctx.compile_kernel(SAMPLER_SHADER, "scatter_vocab_chunk")?,
            argmax_rows: ctx.compile_kernel(SAMPLER_SHADER, "argmax_rows")?,
            row_entropy: ctx.compile_kernel(SAMPLER_SHADER, "row_entropy")?,
            sample_from_probs_rows: ctx.compile_kernel(SAMPLER_SHADER, "sample_from_probs_rows")?,
        })
    }
}
