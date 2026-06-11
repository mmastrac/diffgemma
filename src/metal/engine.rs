use crate::metal::attention::GpuAttention;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::gemm::Bf16Gemm;
use crate::metal::kernels::GpuKernels;
use crate::safetensors::Error;

const GEMM_SHADER: &str = include_str!("../../shaders/gemm.metal");
const GEMM_ENTRY: &str = "bf16_gemm";
const F32_BF16_GEMM_ENTRY: &str = "f32_bf16_gemm";

pub struct GpuDecoderEngine {
    pub ctx: MetalContext,
    pub pool: BufferPool,
    pub gemm: Bf16Gemm,
    pub gemm_pipeline: ComputePipeline,
    pub f32_bf16_gemm_pipeline: ComputePipeline,
    pub kernels: GpuKernels,
    pub attention: GpuAttention,
}

impl GpuDecoderEngine {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let pool = BufferPool::new();
        let gemm_pipeline = ctx.compile_kernel(GEMM_SHADER, GEMM_ENTRY)?;
        let f32_bf16_gemm_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_BF16_GEMM_ENTRY)?;
        let gemm = Bf16Gemm::new()?;
        let kernels = GpuKernels::new(&ctx)?;
        let attention = GpuAttention::new()?;
        Ok(Self {
            ctx,
            pool,
            gemm,
            gemm_pipeline,
            f32_bf16_gemm_pipeline,
            kernels,
            attention,
        })
    }
}
