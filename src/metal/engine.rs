use crate::metal::attention::GpuAttentionKernels;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::kernels::GpuKernels;
use crate::metal::mps_gemm::MpsMatmulCache;
use crate::metal::telemetry::ForwardTelemetry;
use crate::safetensors::Error;
use std::cell::Cell;
use std::rc::Rc;
use std::cell::RefCell;

const GEMM_SHADER: &str = include_str!("../../shaders/gemm.metal");
const QGEMM_SHADER: &str = include_str!("../../shaders/qgemm.metal");
const GEMM_ENTRY: &str = "bf16_gemm";
const F32_BF16_GEMM_ENTRY: &str = "f32_bf16_gemm";
const F32_BF16_LINEAR_ENTRY: &str = "f32_bf16_linear";
const F32_F32_LINEAR_ENTRY: &str = "f32_f32_linear";
const F32_Q4_LINEAR_ENTRY: &str = "f32_q4_linear";
const F32_Q4_LINEAR_GROUPED_ENTRY: &str = "f32_q4_linear_grouped";
const F32_Q8_LINEAR_ENTRY: &str = "f32_q8_linear";
const DEQUANT_Q4_MATRIX_ENTRY: &str = "dequant_q4_matrix";

pub struct GpuDecoderEngine {
    pub ctx: MetalContext,
    pub pool: BufferPool,
    pub gemm_pipeline: ComputePipeline,
    /// PyTorch `[out,in]` weights: `y = x @ W^T` without offline transpose.
    pub f32_bf16_linear_pipeline: ComputePipeline,
    pub f32_f32_linear_pipeline: ComputePipeline,
    pub f32_q4_linear_pipeline: ComputePipeline,
    pub f32_q4_linear_grouped_pipeline: ComputePipeline,
    pub f32_q8_linear_pipeline: ComputePipeline,
    pub dequant_q4_matrix_pipeline: ComputePipeline,
    pub mps_matmul: MpsMatmulCache,
    pub kernels: GpuKernels,
    pub attention: GpuAttentionKernels,
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
        let f32_q4_linear_pipeline = ctx.compile_kernel(QGEMM_SHADER, F32_Q4_LINEAR_ENTRY)?;
        let f32_q4_linear_grouped_pipeline =
            ctx.compile_kernel(QGEMM_SHADER, F32_Q4_LINEAR_GROUPED_ENTRY)?;
        let f32_q8_linear_pipeline = ctx.compile_kernel(QGEMM_SHADER, F32_Q8_LINEAR_ENTRY)?;
        let dequant_q4_matrix_pipeline =
            ctx.compile_kernel(QGEMM_SHADER, DEQUANT_Q4_MATRIX_ENTRY)?;
        let mps_matmul = MpsMatmulCache::new(ctx.device.clone());
        let kernels = GpuKernels::new(&ctx)?;
        let attention = GpuAttentionKernels::new(&ctx)?;
        Ok(Self {
            ctx,
            pool,
            gemm_pipeline,
            f32_bf16_linear_pipeline,
            f32_f32_linear_pipeline,
            f32_q4_linear_pipeline,
            f32_q4_linear_grouped_pipeline,
            f32_q8_linear_pipeline,
            dequant_q4_matrix_pipeline,
            mps_matmul,
            kernels,
            attention,
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

    /// MPS dense path for `.dgq` q4 linears (dequant scratch + MPSMatrixMultiplication).
    pub fn q4_mps_pair(
        &mut self,
        enabled: bool,
    ) -> Option<(&mut MpsMatmulCache, &ComputePipeline)> {
        if enabled {
            Some((&mut self.mps_matmul, &self.dequant_q4_matrix_pipeline))
        } else {
            None
        }
    }
}
