//! GPU debug-status buffer readback (P3.7). Used with `--assert` on generate-monolithic.

use crate::Error;
use crate::shaders::dbg_kernel::{DbgErrorCode, DbgKernelId};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

/// Matches `shaders/include/debug_status.metal` (`DebugStatus`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDebugStatus {
    pub code: u32,
    pub kernel_id: u32,
    pub tg_id: u32,
    pub value: u32,
}

pub const DEBUG_STATUS_BYTES: usize = std::mem::size_of::<GpuDebugStatus>();

pub fn zero_buffer(buf: &ProtocolObject<dyn MTLBuffer>) {
    let ptr = buf.contents().as_ptr() as *mut GpuDebugStatus;
    unsafe {
        ptr.write(GpuDebugStatus::default());
    }
}

pub fn read_buffer(buf: &ProtocolObject<dyn MTLBuffer>) -> GpuDebugStatus {
    let ptr = buf.contents().as_ptr() as *const GpuDebugStatus;
    unsafe { ptr.read() }
}

pub fn check_status(st: GpuDebugStatus) -> Result<(), Error> {
    if st.code == 0 {
        return Ok(());
    }
    let kernel = DbgKernelId::from_u32(st.kernel_id);
    let code = DbgErrorCode::from_u32(st.code);
    let msg = format!(
        "GPU assert: {code:?} in {kernel:?} (tg={:#x}, value={:#x})",
        st.tg_id, st.value
    );
    Err(Error::Format(Box::leak(msg.into_boxed_str())))
}

impl DbgKernelId {
    fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::EmbedGather,
            25 => Self::QkRopeKv,
            26 => Self::Attention,
            27 => Self::MoeRouter,
            29 => Self::MoeBucketFill,
            34 => Self::ScProbs,
            35 => Self::ScSoftembed,
            37 => Self::SampleRowstats,
            38 => Self::SampleCommit,
            39 => Self::SampleApply,
            40 => Self::SampleWrite,
            41 => Self::LogitRowstats,
            13 => Self::SoftcapHalf,
            _ => Self::Unknown,
        }
    }
}

impl DbgErrorCode {
    fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::IndexOob,
            2 => Self::RouteTokenOob,
            3 => Self::SoftmaxNormZero,
            4 => Self::NonFinite,
            5 => Self::EntropyTooHigh,
            6 => Self::QuantUnsupported,
            7 => Self::ArenaOob,
            8 => Self::BadDim,
            9 => Self::NegativeEntropy,
            _ => Self::None,
        }
    }
}
