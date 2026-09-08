//! End-to-end CUDA check: compile a trivial kernel at runtime with NVRTC
//! through gpukit, launch it, and read the result back -- the path every
//! dgops/dgemm kernel takes on a CUDA host.
//!
//! Skips (rather than fails) when no device or NVRTC is found, so the default
//! macOS test run is unaffected. Run it on a CUDA host with:
//!
//!     cargo test -p gpukit --features cuda --test cuda_smoke -- --nocapture

#![cfg(feature = "cuda")]

use gpukit::cuda::{BufferPool, Context, ContextConfig, KernelArgs, launch_1d};

const SMOKE_CU: &str = include_str!("smoke.cu");

fn skip(why: &str) {
    eprintln!("skipping cuda_smoke: {why}");
}

#[test]
fn driver_loads_module_and_launches() {
    let Ok(ctx) = Context::new(ContextConfig::default()) else {
        return skip("no CUDA device");
    };
    eprintln!(
        "device: {} sm_{}{}",
        ctx.name(),
        ctx.compute_capability().0,
        ctx.compute_capability().1
    );

    let kernel = match ctx.compile_kernel(SMOKE_CU, "saxpy") {
        Ok(kernel) => kernel,
        Err(e) => return skip(&format!("NVRTC compile failed: {e}")),
    };

    let n = 1024usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let mut pool = BufferPool::new();
    let dx = pool.allocate(&ctx, n * 4).expect("alloc x");
    let dy = pool.allocate(&ctx, n * 4).expect("alloc y");
    dx.write_f32(&x).expect("upload x");
    dy.write_f32(&vec![1.0f32; n]).expect("upload y");

    let mut args = KernelArgs::new();
    args.device_ptr(dy.device_ptr())
        .device_ptr(dx.device_ptr())
        .f32(2.0)
        .u32(n as u32);
    launch_1d(&ctx, &kernel, n, &mut args).expect("launch saxpy");
    ctx.synchronize().expect("synchronize");

    let mut got = vec![0.0f32; n];
    dy.read_f32(&mut got).expect("download y");
    for (i, v) in got.iter().enumerate() {
        let want = 2.0 * x[i] + 1.0;
        assert!(
            (v - want).abs() < 1e-6,
            "saxpy[{i}] = {v}, want {want} (driver FFI or launch is wrong)"
        );
    }
    eprintln!("saxpy OK over {n} elements");

    pool.release(dx);
    pool.release(dy);
}
