//! End-to-end CUDA driver check: compile a trivial kernel with nvcc, load the
//! cubin through gpukit, launch it, and read the result back.
//!
//! Skips (rather than fails) when no nvcc is found, so the default macOS test
//! run is unaffected. Run it on a CUDA host with:
//!
//!     cargo test -p gpukit --features cuda --test cuda_smoke -- --nocapture

#![cfg(feature = "cuda")]

use gpukit::cuda::{BufferPool, Context, ContextConfig, KernelArgs, launch_1d};
use std::path::PathBuf;
use std::process::Command;

const SMOKE_CU: &str = include_str!("smoke.cu");

fn nvcc() -> Option<String> {
    if let Ok(explicit) = std::env::var("DGQ_NVCC") {
        return Some(explicit);
    }
    for candidate in ["nvcc", "/usr/local/cuda/bin/nvcc"] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

fn build_cubin(nvcc: &str) -> Option<PathBuf> {
    let arch = std::env::var("DGQ_CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
    let dir = std::env::temp_dir().join("gpukit-cuda-smoke");
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("smoke.cu");
    let cubin = dir.join("smoke.cubin");
    std::fs::write(&src, SMOKE_CU).ok()?;
    let status = Command::new(nvcc)
        .args(["-cubin", "-O3", "-arch", &arch, "-o"])
        .arg(&cubin)
        .arg(&src)
        .status()
        .ok()?;
    status.success().then_some(cubin)
}

fn skip(why: &str) {
    eprintln!("skipping cuda_smoke: {why}");
}

#[test]
fn driver_loads_module_and_launches() {
    let Some(nvcc) = nvcc() else {
        return skip("nvcc not found (set DGQ_NVCC or install the CUDA toolkit)");
    };
    let Some(cubin) = build_cubin(&nvcc) else {
        return skip("nvcc could not build smoke.cubin");
    };

    let ctx = Context::new(ContextConfig::default()).expect("open CUDA context");
    eprintln!(
        "device: {} sm_{}{}",
        ctx.name(),
        ctx.compute_capability().0,
        ctx.compute_capability().1
    );

    let image = std::fs::read(&cubin).expect("read cubin");
    let module = ctx.load_module(&image).expect("load module");
    let kernel = module.function("saxpy").expect("resolve saxpy");

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
