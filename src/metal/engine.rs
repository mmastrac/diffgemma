use crate::metal::attention::GpuAttentionKernels;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::kernels::GpuKernels;
use crate::metal::telemetry::ForwardTelemetry;
use crate::safetensors::Error;
use std::cell::Cell;
use std::rc::Rc;
use std::cell::RefCell;

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
    telemetry: Rc<RefCell<ForwardTelemetry>>,
    telemetry_enabled: Cell<bool>,
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
}
