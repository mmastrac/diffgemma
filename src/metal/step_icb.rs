//! P2.2: record monolithic denoise-step compute dispatches into an indirect command buffer (ICB)
//! and replay on subsequent steps (fused Q4/nvfp4 path only).

use crate::metal::device::ComputePipeline;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLDevice,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLResourceOptions, MTLSize,
};

/// Max indirect compute commands for one full denoise step.
const ICB_MAX_COMMANDS: usize = 16384;

/// Append-only CPU pool mirrored into `const_buf` at record finish (ICB binds keep offsets).
const ICB_CONST_BYTES: usize = 4 * 1024 * 1024;
const ICB_CONST_ALIGN: usize = 256;

pub struct StepIcbPlan {
    pub icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    pub command_count: u32,
    pub const_bytes: u32,
}

pub struct StepIcbPair {
    pub no_sc: Option<StepIcbPlan>,
    /// Recorded lazily after step 1 (needs prefilled KV + step-1 canvas state).
    pub with_sc: Option<StepIcbPlan>,
    /// `StepParams.kv_len` when `no_sc` was recorded (audit only; replay reads live params).
    pub no_sc_kv_len: u32,
    /// `StepPipelineKey` bits at record time; invalidate cached plans when assert/deep toggles.
    pub pipeline_key: u8,
}

pub struct IcbRecorder {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    cmd_idx: u32,
    const_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    const_pool: Vec<u8>,
    cmd_open: bool,
}

impl IcbRecorder {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTLIndirectCommandBufferDescriptor::new();
        desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        desc.setInheritPipelineState(false);
        desc.setInheritBuffers(false);
        desc.setMaxKernelBufferBindCount(8);
        let icb = unsafe {
            device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &desc,
                    ICB_MAX_COMMANDS,
                    MTLResourceOptions::empty(),
                )
        }
        .ok_or(Error::Format("ICB alloc failed"))?;
        let const_buf = device
            .newBufferWithLength_options(ICB_CONST_BYTES, MTLResourceOptions::empty())
            .ok_or(Error::Format("ICB const pool alloc failed"))?;
        Ok(Self {
            icb,
            cmd_idx: 0,
            const_buf,
            const_pool: Vec::new(),
            cmd_open: false,
        })
    }

    fn icmd(&self) -> Retained<ProtocolObject<dyn MTLIndirectComputeCommand>> {
        assert!((self.cmd_idx as usize) < ICB_MAX_COMMANDS);
        unsafe { self.icb.indirectComputeCommandAtIndex(self.cmd_idx as usize) }
    }

    fn sync_const_pool(&mut self) {
        if self.const_pool.is_empty() {
            return;
        }
        assert!(
            self.const_pool.len() <= ICB_CONST_BYTES,
            "ICB const pool overflow ({} bytes)",
            self.const_pool.len()
        );
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                self.const_buf.contents().as_ptr() as *mut u8,
                self.const_pool.len(),
            )
        };
        dst.copy_from_slice(&self.const_pool);
    }

    fn bind_constant<T: Copy>(
        &mut self,
        icmd: &ProtocolObject<dyn MTLIndirectComputeCommand>,
        val: &T,
        index: usize,
    ) {
        let size = std::mem::size_of::<T>();
        let pad = (ICB_CONST_ALIGN - (self.const_pool.len() % ICB_CONST_ALIGN)) % ICB_CONST_ALIGN;
        self.const_pool.extend(std::iter::repeat_n(0u8, pad));
        let off = self.const_pool.len();
        self.const_pool.extend_from_slice(unsafe {
            std::slice::from_raw_parts(val as *const T as *const u8, size)
        });
        // Defer GPU upload until finish() — syncing on every bind was O(n^2) and froze the host
        // while recording a 30L step (~thousands of small constants).
        unsafe {
            icmd.setKernelBuffer_offset_atIndex(&self.const_buf, off, index);
        }
    }

    pub fn set_pipeline(&mut self, ps: &ComputePipeline) {
        self.cmd_open = true;
        self.icmd().setComputePipelineState(&ps.pipeline);
    }

    pub fn set_buffer(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        self.cmd_open = true;
        unsafe {
            self.icmd()
                .setKernelBuffer_offset_atIndex(buf, offset, index);
        }
    }

    pub fn set_bytes<T: Copy>(&mut self, val: &T, index: usize) {
        self.cmd_open = true;
        let icmd = self.icmd();
        self.bind_constant(&icmd, val, index);
    }

    pub fn dispatch_threadgroups(&mut self, grid: MTLSize, tg: MTLSize) {
        self.icmd()
            .concurrentDispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        self.cmd_idx += 1;
        self.cmd_open = false;
    }

    pub fn finish(mut self) -> Result<StepIcbPlan, Error> {
        self.sync_const_pool();
        if self.cmd_open {
            // Discard a trailing set_pipeline/bind without dispatch — do not execute a no-op ICB slot.
            self.icmd().reset();
            self.cmd_open = false;
        }
        Ok(StepIcbPlan {
            icb: self.icb,
            command_count: self.cmd_idx,
            const_bytes: self.const_pool.len() as u32,
        })
    }
}

pub fn step_icb_enabled() -> bool {
    match std::env::var("DGQ_STEP_ICB") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

/// Opt-in `with_sc` ICB replay (step ≥ 2). Default off until 30L parity is stable.
pub fn step_icb_with_sc_enabled() -> bool {
    match std::env::var("DGQ_STEP_ICB_SC") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

pub fn replay_step_icb( 
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    plan: &StepIcbPlan,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, Error> {
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Format("compute encoder alloc failed"))?;
    if plan.command_count > 0 {
        unsafe {
            enc.executeCommandsInBuffer_withRange(
                &plan.icb,
                NSRange::new(0, plan.command_count as usize),
            );
        }
    }
    Ok(enc)
}
