use crate::metal::attention::GpuAttentionKernels;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::kernels::GpuKernels;
use crate::metal::mps_gemm::MpsMatmulCache;
use crate::metal::sampler_kernels::GpuSamplerKernels;
use crate::metal::telemetry::ForwardTelemetry;
use crate::safetensors::Error;
use std::cell::Cell;
use std::rc::Rc;
use std::cell::RefCell;

const GEMM_SHADER: &str = include_str!("../../shaders/gemm.metal");
const GEMM_ENTRY: &str = "bf16_gemm";
const F32_BF16_GEMM_ENTRY: &str = "f32_bf16_gemm";
const F32_BF16_LINEAR_ENTRY: &str = "f32_bf16_linear";
const F32_F32_LINEAR_ENTRY: &str = "f32_f32_linear";

pub struct GpuDecoderEngine {
    pub ctx: MetalContext,
    pub pool: BufferPool,
    pub gemm_pipeline: ComputePipeline,
    /// PyTorch `[out,in]` weights: `y = x @ W^T` without offline transpose.
    pub f32_bf16_linear_pipeline: ComputePipeline,
    pub f32_f32_linear_pipeline: ComputePipeline,
    pub f32_q4_linear_pipeline: ComputePipeline,
    pub f32_nvfp4_linear_pipeline: ComputePipeline,
    pub f32_q4_linear_grouped_pipeline: ComputePipeline,
    pub f32_q8_linear_pipeline: ComputePipeline,
    pub f32_q8_linear_kxn_pipeline: ComputePipeline,
    pub dequant_q4_matrix_pipeline: ComputePipeline,
    pub dequant_nvfp4_matrix_pipeline: ComputePipeline,
    pub mps_matmul: MpsMatmulCache,
    /// When false, `.dgq` q4 linears use the native Metal kernel instead of MPS (deterministic).
    use_mps_q4: Cell<bool>,
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
        let gemm_pipeline = ctx.compile_kernel(GEMM_SHADER, GEMM_ENTRY)?;
        let f32_bf16_linear_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_BF16_LINEAR_ENTRY)?;
        let f32_f32_linear_pipeline = ctx.compile_kernel(GEMM_SHADER, F32_F32_LINEAR_ENTRY)?;
        let prod = crate::kernels::sub::variant::KernelVariant::PRODUCTION;
        let f32_q4_linear_pipeline = crate::kernels::sub::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::Q4Affine,
            prod,
        )?;
        let f32_nvfp4_linear_pipeline = crate::kernels::sub::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::NvFp4,
            prod,
        )?;
        let f32_q4_linear_grouped_pipeline = crate::kernels::sub::gemm_linear_grouped::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::Q4Affine,
            prod,
        )?;
        let f32_q8_linear_pipeline =
            crate::kernels::sub::gemm_q8_linear_f32::pipeline_for(&ctx, prod)?;
        let f32_q8_linear_kxn_pipeline =
            crate::kernels::sub::gemm_q8_linear_kxn_f32::pipeline_for(&ctx, prod)?;
        let dequant_q4_matrix_pipeline =
            crate::kernels::sub::dequant_block_matrix::pipeline_for(
                &ctx,
                crate::kernels::sub::QuantFormat::Q4Affine,
                crate::kernels::sub::variant::KernelVariant::PRODUCTION,
            )?;
        let dequant_nvfp4_matrix_pipeline =
            crate::kernels::sub::dequant_block_matrix::pipeline_for(
                &ctx,
                crate::kernels::sub::QuantFormat::NvFp4,
                crate::kernels::sub::variant::KernelVariant::PRODUCTION,
            )?;
        let mps_matmul = MpsMatmulCache::new(ctx.device.clone());
        let use_mps_q4 = Cell::new(mps_q4_default_from_env());
        let kernels = GpuKernels::new(&ctx)?;
        let attention = GpuAttentionKernels::new(&ctx)?;
        let sampler_kernels = GpuSamplerKernels::new(&ctx)?;
        crate::metal::pipeline_cache::PipelineArchiveCache::flush_global();
        Ok(Self {
            ctx,
            pool,
            gemm_pipeline,
            f32_bf16_linear_pipeline,
            f32_f32_linear_pipeline,
            f32_q4_linear_pipeline,
            f32_nvfp4_linear_pipeline,
            f32_q4_linear_grouped_pipeline,
            f32_q8_linear_pipeline,
            f32_q8_linear_kxn_pipeline,
            dequant_q4_matrix_pipeline,
            dequant_nvfp4_matrix_pipeline,
            mps_matmul,
            use_mps_q4,
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
        self.telemetry_enabled()
            .then(|| self.telemetry_handle())
    }

    /// MPS dense path for `.dgq` block linears (dequant scratch + MPSMatrixMultiplication).
    pub fn q4_mps_triple(
        &mut self,
        enabled: bool,
    ) -> Option<(
        &mut MpsMatmulCache,
        &ComputePipeline,
        &ComputePipeline,
    )> {
        if enabled {
            Some((
                &mut self.mps_matmul,
                &self.dequant_q4_matrix_pipeline,
                &self.dequant_nvfp4_matrix_pipeline,
            ))
        } else {
            None
        }
    }

    pub fn use_mps_q4(&self) -> bool {
        self.use_mps_q4.get()
    }

    pub fn set_use_mps_q4(&self, enabled: bool) {
        self.use_mps_q4.set(enabled);
    }
}

fn mps_q4_default_from_env() -> bool {
    match std::env::var("DGQ_MPS_Q4") {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => true,
    }
}

/// Default MPS Q4 dense path for production (`true` unless `DGQ_MPS_Q4=0`).
pub fn default_use_mps_q4() -> bool {
    mps_q4_default_from_env()
}
