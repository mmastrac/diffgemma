//! One-shot launch helpers and the kernel-argument builder.

use super::context::{Context, Kernel};
use super::driver::CUdeviceptr;
use crate::Error;

pub fn div_up(value: usize, group: usize) -> usize {
    value.div_ceil(group)
}

/// Threads per block for the 1D helpers (matches the Metal backend's 256).
pub const THREADS_PER_BLOCK: u32 = 256;

/// Typed kernel arguments. Values are stored in aligned slots and pointed at
/// only once the argument list is complete, so a cuLaunchKernel always sees
/// correctly aligned storage.
#[derive(Default)]
pub struct KernelArgs {
    args: Vec<Arg>,
}

enum Arg {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    /// Byte blob in 8-byte-aligned storage (for small POD structs).
    Aligned(Box<[u64]>),
}

impl Arg {
    fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match self {
            Arg::U32(v) => std::ptr::from_mut(v).cast(),
            Arg::I32(v) => std::ptr::from_mut(v).cast(),
            Arg::F32(v) => std::ptr::from_mut(v).cast(),
            Arg::U64(v) => std::ptr::from_mut(v).cast(),
            Arg::Aligned(v) => v.as_mut_ptr().cast(),
        }
    }
}

impl KernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.args.push(Arg::U32(v));
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.args.push(Arg::I32(v));
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.args.push(Arg::F32(v));
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.args.push(Arg::U64(v));
        self
    }

    /// Bind a device pointer (a kernel T* argument).
    pub fn device_ptr(&mut self, p: CUdeviceptr) -> &mut Self {
        self.u64(p)
    }

    /// Two consecutive u32s (a uint2 kernel parameter).
    pub fn u32x2(&mut self, a: u32, b: u32) -> &mut Self {
        let packed = [(a as u64) | ((b as u64) << 32)];
        self.args.push(Arg::Aligned(packed.to_vec().into_boxed_slice()));
        self
    }

    /// Bind a small POD value of arbitrary layout. Storage is 8-byte aligned;
    /// argument types needing stricter alignment are not supported.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        let mut words = vec![0u64; div_up(bytes.len(), 8)].into_boxed_slice();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                words.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        self.args.push(Arg::Aligned(words));
        self
    }

    pub(crate) fn pointers(&mut self) -> Vec<*mut std::ffi::c_void> {
        self.args.iter_mut().map(Arg::as_mut_ptr).collect()
    }
}

/// 1D grid, one thread per element (256-thread blocks).
pub fn launch_1d(
    ctx: &Context,
    kernel: &Kernel,
    count: usize,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    let block = THREADS_PER_BLOCK.min(count.max(1) as u32);
    let grid = div_up(count, block as usize) as u32;
    ctx.launch(kernel, (grid, 1, 1), (block, 1, 1), 0, args)
}

/// 1D grid over [base, base+count) chunked to the CUDA grid-x limit.
pub fn launch_1d_ranged(
    ctx: &Context,
    kernel: &Kernel,
    count: usize,
    mut bind: impl FnMut(&mut KernelArgs, u32, u32),
) -> Result<(), Error> {
    const MAX_GRID_X: usize = i32::MAX as usize;
    let chunk = MAX_GRID_X * THREADS_PER_BLOCK as usize;
    let mut base = 0usize;
    while base < count {
        let len = (count - base).min(chunk);
        let mut args = KernelArgs::new();
        bind(&mut args, base as u32, len as u32);
        launch_1d(ctx, kernel, len, &mut args)?;
        base += len;
    }
    Ok(())
}

/// One block per row, block_width threads per block.
pub fn launch_rows(
    ctx: &Context,
    kernel: &Kernel,
    rows: usize,
    block_width: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    ctx.launch(kernel, ctx_grid(1, rows), (block_width, 1, 1), 0, args)
}

/// 2D grid with a fixed block width (one block per (x, y)).
pub fn launch_grid(
    ctx: &Context,
    kernel: &Kernel,
    width: usize,
    height: usize,
    block_width: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    ctx.launch(kernel, ctx_grid(width, height), (block_width, 1, 1), 0, args)
}

fn ctx_grid(width: usize, height: usize) -> (u32, u32, u32) {
    (width as u32, height as u32, 1)
}
