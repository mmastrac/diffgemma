use crate::metal::attention::GpuAttentionKernels;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::kernels::GpuKernels;
use crate::safetensors::Error;

const GEMM_SHADER: &str = include_str!("../../shaders/gemm.metal");
const GEMM_ENTRY: &str = "bf16_gemm";
const F32_BF16_GEMM_ENTRY: &str = "f32_bf16_gemm";
const F32_BF16_LINEAR_ENTRY: &str = "f32_bf16_linear";

pub struct GpuDecoderEngine {
    pub ctx: MetalContext,
    pub pool: BufferPool,
    pub gemm_pipeline: ComputePipeline,
    /// PyTorch `[out,in]` weights: `y = x @ W^T` without offline transpose.
    pub f32_bf16_linear_pipeline: ComputePipeline,
    pub kernels: GpuKernels,
    pub attention: GpuAttentionKernels,
}

impl GpuDecoderEngine {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let pool = BufferPool::new();
        let gemm_pipeline = ctx.compile_kernel(GEMM_SHADER, GEMM_ENTRY)?;
        let f32_bf16_linear_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_BF16_LINEAR_ENTRY)?;
        let kernels = GpuKernels::new(&ctx)?;
        let attention = GpuAttentionKernels::new(&ctx)?;
        Ok(Self {
            ctx,
            pool,
            gemm_pipeline,
            f32_bf16_linear_pipeline,
            kernels,
            attention,
        })
    }
}
