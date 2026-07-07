//! GPU sampler kernels (logit post-process, entropy, argmax, categorical sample).

use crate::kernels::sub::KernelVariant;
use crate::kernels::sub::softmax_rows;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::safetensors::Error;

const COPY_F32_SHADER: &str = shader_include::include_metal!("kernels/copy_f32.metal");
const LOGIT_SOFTCAPPING_SHADER: &str =
    shader_include::include_metal!("kernels/logit_softcapping.metal");
const SCALE_LOGITS_SHADER: &str = shader_include::include_metal!("kernels/scale_logits.metal");
const SCATTER_VOCAB_CHUNK_SHADER: &str =
    shader_include::include_metal!("kernels/scatter_vocab_chunk.metal");
const ARGMAX_ROWS_SHADER: &str = shader_include::include_metal!("kernels/argmax_rows.metal");
const ROW_ENTROPY_SHADER: &str = shader_include::include_metal!("kernels/row_entropy.metal");
const SAMPLE_FROM_PROBS_SHADER: &str =
    shader_include::include_metal!("kernels/sample_from_probs_rows.metal");

pub struct GpuSamplerKernels {
    pub copy_f32: ComputePipeline,
    pub logit_softcapping: ComputePipeline,
    pub scale_logits: ComputePipeline,
    pub scatter_vocab_chunk: ComputePipeline,
    pub argmax_rows: ComputePipeline,
    pub row_entropy: ComputePipeline,
    pub softmax_rows: ComputePipeline,
    pub sample_from_probs_rows: ComputePipeline,
}

impl GpuSamplerKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, Error> {
        Ok(Self {
            copy_f32: ctx.compile_kernel(COPY_F32_SHADER, "copy_f32")?,
            logit_softcapping: ctx.compile_kernel(LOGIT_SOFTCAPPING_SHADER, "logit_softcapping")?,
            scale_logits: ctx.compile_kernel(SCALE_LOGITS_SHADER, "scale_logits")?,
            scatter_vocab_chunk: ctx
                .compile_kernel(SCATTER_VOCAB_CHUNK_SHADER, "scatter_vocab_chunk")?,
            argmax_rows: ctx.compile_kernel(ARGMAX_ROWS_SHADER, "argmax_rows")?,
            row_entropy: ctx.compile_kernel(ROW_ENTROPY_SHADER, "row_entropy")?,
            softmax_rows: ctx.compile_subkernel(
                softmax_rows::shader_source(),
                softmax_rows::ENTRY,
                KernelVariant::PRODUCTION,
            )?,
            sample_from_probs_rows: ctx
                .compile_kernel(SAMPLE_FROM_PROBS_SHADER, "sample_from_probs_rows")?,
        })
    }
}
