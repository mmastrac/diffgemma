//! P2.2 Phase B: record monolithic denoise-step dispatches into an indirect command buffer (ICB)
//! and replay with live MPS Q4 GEMM segments interleaved.

use crate::metal::device::ComputePipeline;
use crate::metal::mps_gemm::MpsMatmulCache;
use crate::metal::step_kernel::StepBuffers;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLResourceOptions, MTLSize,
};

/// Max indirect compute commands for one full denoise step.
const ICB_MAX_COMMANDS: usize = 16384;

/// Append-only CPU pool mirrored into `const_buf` at record finish (ICB binds keep offsets).
const ICB_CONST_BYTES: usize = 4 * 1024 * 1024;
const ICB_CONST_ALIGN: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct MpsQ4GemmOp {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub enum StepReplayOp {
    Icb { start: u32, count: u32 },
    MpsQ4(MpsQ4GemmOp),
}

pub struct StepIcbPlan {
    pub icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    pub command_count: u32,
    pub ops: Vec<StepReplayOp>,
    pub const_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub struct StepIcbPair {
    pub no_sc: StepIcbPlan,
    pub with_sc: StepIcbPlan,
}

pub struct IcbRecorder {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    cmd_idx: u32,
    segment_start: u32,
    ops: Vec<StepReplayOp>,
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
            segment_start: 0,
            ops: Vec::new(),
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
        self.sync_const_pool();
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

    pub fn note_mps_q4_gemm(&mut self, op: MpsQ4GemmOp) {
        if self.cmd_open {
            self.cmd_idx += 1;
            self.cmd_open = false;
        }
        if self.cmd_idx > self.segment_start {
            self.ops.push(StepReplayOp::Icb {
                start: self.segment_start,
                count: self.cmd_idx - self.segment_start,
            });
        }
        self.ops.push(StepReplayOp::MpsQ4(op));
        self.segment_start = self.cmd_idx;
    }

    pub fn finish(mut self) -> Result<StepIcbPlan, Error> {
        self.sync_const_pool();
        if self.cmd_open {
            self.cmd_idx += 1;
            self.cmd_open = false;
        }
        if self.cmd_idx > self.segment_start {
            self.ops.push(StepReplayOp::Icb {
                start: self.segment_start,
                count: self.cmd_idx - self.segment_start,
            });
        }
        Ok(StepIcbPlan {
            icb: self.icb,
            command_count: self.cmd_idx,
            ops: self.ops,
            const_buf: self.const_buf,
        })
    }
}

pub fn step_icb_enabled() -> bool {
    match std::env::var("DGQ_STEP_ICB") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

pub fn replay_step_icb(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    plan: &StepIcbPlan,
    bufs: &StepBuffers,
    mps: &mut MpsMatmulCache,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, Error> {
    let mut enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Format("compute encoder alloc failed"))?;

    for op in &plan.ops {
        match op {
            StepReplayOp::Icb { start, count } if *count > 0 => unsafe {
                enc.executeCommandsInBuffer_withRange(
                    &plan.icb,
                    NSRange::new(*start as usize, *count as usize),
                );
            },
            StepReplayOp::MpsQ4(g) => {
                enc.endEncoding();
                mps.encode_f32_linear(
                    cmd,
                    &bufs.mps_x,
                    &bufs.mps_w,
                    &bufs.mps_c,
                    g.m,
                    g.k,
                    g.n,
                );
                enc = cmd
                    .computeCommandEncoder()
                    .ok_or(Error::Format("compute encoder resume failed"))?;
            }
            StepReplayOp::Icb { .. } => {}
        }
    }
    Ok(enc)
}
