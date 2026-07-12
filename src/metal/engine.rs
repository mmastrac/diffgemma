use crate::metal::attention::GpuAttentionKernels;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::kernels::GpuKernels;
use crate::metal::sampler_kernels::GpuSamplerKernels;
use crate::metal::telemetry::ForwardTelemetry;
use crate::safetensors::Error;
use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

const GEMM_SHADER: &str = include_str!("../shaders/gemm/gemm.metal");
const F32_BF16_LINEAR_ENTRY: &str = "f32_bf16_linear";
const F32_F32_LINEAR_ENTRY: &str = "f32_f32_linear";

pub struct GpuDecoderEngine {
    pub ctx: MetalContext,
    pub pool: BufferPool,
    /// PyTorch `[out,in]` weights: `y = x @ W^T` without offline transpose.
    pub f32_bf16_linear_pipeline: ComputePipeline,
    pub f32_f32_linear_pipeline: ComputePipeline,
    pub f32_q4_linear_pipeline: ComputePipeline,
    pub f32_nvfp4_linear_pipeline: ComputePipeline,
    pub f32_q4_linear_grouped_pipeline: ComputePipeline,
    pub f32_nvfp4_linear_grouped_pipeline: ComputePipeline,
    pub f32_q6_linear_pipeline: ComputePipeline,
    pub f32_q6_linear_grouped_pipeline: ComputePipeline,
    pub f32_q8_linear_pipeline: ComputePipeline,
    pub f32_q8_linear_kxn_pipeline: ComputePipeline,
    /// When true, encoder `.dgq` MoE uses grouped GPU GEMM (`DGQ_ENCODER_GPU_MOE=0` to opt out).
    encoder_gpu_moe: Cell<bool>,
    pub kernels: GpuKernels,
    pub attention: GpuAttentionKernels,
    pub sampler_kernels: GpuSamplerKernels,
    telemetry: Rc<RefCell<ForwardTelemetry>>,
    telemetry_enabled: Cell<bool>,
}

impl GpuDecoderEngine {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let pool = BufferPool::new();
        let f32_bf16_linear_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_BF16_LINEAR_ENTRY)?;
        let f32_f32_linear_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_F32_LINEAR_ENTRY)?;
        let prod = crate::shaders::variant::KernelVariant::PRODUCTION;
        let f32_q4_linear_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::Q4Affine,
            prod,
        )?;
        let f32_nvfp4_linear_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::NvFp4,
            prod,
        )?;
        let f32_q4_linear_grouped_pipeline = crate::shaders::gemm_linear_grouped::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::Q4Affine,
            prod,
        )?;
        let f32_nvfp4_linear_grouped_pipeline = crate::shaders::gemm_linear_grouped::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::NvFp4,
            prod,
        )?;
        let f32_q6_linear_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::Q6,
            prod,
        )?;
        let f32_q6_linear_grouped_pipeline = crate::shaders::gemm_linear_grouped::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::Q6,
            prod,
        )?;
        let f32_q8_linear_pipeline = crate::shaders::gemm_q8_linear_f32::pipeline_for(&ctx, prod)?;
        let f32_q8_linear_kxn_pipeline =
            crate::shaders::gemm_q8_linear_kxn_f32::pipeline_for(&ctx, prod)?;
        let encoder_gpu_moe = Cell::new(encoder_gpu_moe_from_env());
        let kernels = GpuKernels::new(&ctx)?;
        let attention = GpuAttentionKernels::new(&ctx)?;
        let sampler_kernels = GpuSamplerKernels::new(&ctx)?;
        crate::metal::pipeline_cache::PipelineArchiveCache::flush_global();
        Ok(Self {
            ctx,
            pool,
            f32_bf16_linear_pipeline,
            f32_f32_linear_pipeline,
            f32_q4_linear_pipeline,
            f32_nvfp4_linear_pipeline,
            f32_q4_linear_grouped_pipeline,
            f32_q6_linear_pipeline,
            f32_q6_linear_grouped_pipeline,
            f32_nvfp4_linear_grouped_pipeline,
            f32_q8_linear_pipeline,
            f32_q8_linear_kxn_pipeline,
            encoder_gpu_moe,
            kernels,
            attention,
            sampler_kernels,
            telemetry: Rc::new(RefCell::new(ForwardTelemetry::default())),
            telemetry_enabled: Cell::new(false),
        })
    }

    pub fn reset_forward_telemetry(&self) {
        *self.telemetry.borrow_mut() = ForwardTelemetry::default();
        self.telemetry_enabled.set(true);
    }

    pub fn take_forward_telemetry(&self) -> ForwardTelemetry {
        self.telemetry_enabled.set(false);
        std::mem::take(&mut *self.telemetry.borrow_mut())
    }

    pub fn telemetry_enabled(&self) -> bool {
        self.telemetry_enabled.get()
    }

    pub fn telemetry_handle(&self) -> Rc<RefCell<ForwardTelemetry>> {
        Rc::clone(&self.telemetry)
    }

    pub fn batch_telemetry(&self) -> Option<Rc<RefCell<ForwardTelemetry>>> {
        self.telemetry_enabled().then(|| self.telemetry_handle())
    }

    pub fn encoder_gpu_moe(&self) -> bool {
        self.encoder_gpu_moe.get()
    }

    pub fn set_encoder_gpu_moe(&self, enabled: bool) {
        self.encoder_gpu_moe.set(enabled);
    }
}

fn encoder_gpu_moe_from_env() -> bool {
    crate::flags::encoder_gpu_moe_enabled()
}
