//! Device bandwidth and GEMM throughput probes (Q0).

use crate::metal::buffer::BufferPool;
use crate::metal::device::MetalContext;
use crate::metal::gemm::Bf16Gemm;
use crate::safetensors::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLSize,
};

const PROBE_SHADER: &str = include_str!("../../shaders/probe.metal");
const MEMCPY_ENTRY: &str = "memcpy_f32";

pub struct DeviceProbeResult {
    pub device_name: String,
    pub unified_memory: bool,
    pub recommended_working_set_gib: f64,
    pub memcpy_gib_per_s: f64,
    pub gemm_dense_gflops: f64,
    pub gemm_moe_gflops: f64,
}

pub fn probe_device() -> Result<DeviceProbeResult, Error> {
    let ctx = MetalContext::new()?;
    let name = ctx.device.name().to_string();
    let unified = ctx.device.hasUnifiedMemory();
    let working_set = ctx.device.recommendedMaxWorkingSetSize() as f64 / (1024.0_f64.powi(3));

    let memcpy_gib_per_s = probe_memcpy_bandwidth(&ctx)?;
    let (dense_gflops, moe_gflops) = probe_gemm_tflops()?;

    Ok(DeviceProbeResult {
        device_name: name,
        unified_memory: unified,
        recommended_working_set_gib: working_set,
        memcpy_gib_per_s,
        gemm_dense_gflops: dense_gflops,
        gemm_moe_gflops: moe_gflops,
    })
}

fn probe_memcpy_bandwidth(ctx: &MetalContext) -> Result<f64, Error> {
    const COPY_MIB: usize = 256;
    let n = COPY_MIB * 1024 * 1024 / 4;
    let bytes = n * 4;
    let mut pool = BufferPool::new();
    let src = pool
        .allocate(&ctx.device, bytes)
        .ok_or(Error::Format("probe src alloc failed"))?;
    let dst = pool
        .allocate(&ctx.device, bytes)
        .ok_or(Error::Format("probe dst alloc failed"))?;
    BufferPool::write_f32(&src, &vec![1.0f32; n]);

    let pipeline = ctx.compile_kernel(PROBE_SHADER, MEMCPY_ENTRY)?;
    let iters = 20usize;
    let warmup = 3usize;

    for _ in 0..warmup {
        run_memcpy_kernel(ctx, &pipeline.pipeline, &src, &dst, n)?;
    }

    let started = std::time::Instant::now();
    for _ in 0..iters {
        run_memcpy_kernel(ctx, &pipeline.pipeline, &src, &dst, n)?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    // Read + write ≈ 2× bytes moved.
    let total_bytes = bytes as f64 * 2.0 * iters as f64;
    Ok(total_bytes / elapsed / (1024.0_f64.powi(3)))
}

fn run_memcpy_kernel(
    ctx: &MetalContext,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n: usize,
) -> Result<(), Error> {
    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or(Error::Format("probe cmd buffer failed"))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Format("probe encoder failed"))?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let n_u32 = n as u32;
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(&n_u32).cast(),
            std::mem::size_of_val(&n_u32),
            2,
        );
    }
    const TG: usize = 256;
    let grid = MTLSize {
        width: (n + TG - 1) / TG,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: TG,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

/// Returns (dense_shape GFLOP/s, moe_shape GFLOP/s) using `f32_bf16_linear`.
fn probe_gemm_tflops() -> Result<(f64, f64), Error> {
    let mut gemm = Bf16Gemm::new()?;
    let iters = 10usize;
    let warmup = 3usize;

    // Dense-ish: 256 × 2816 @ 2816 (lm_head / large linear).
    let (m_dense, k_dense, n_dense) = (256usize, 2816usize, 2816usize);
    let dense_gflops = bench_linear_gflops(&mut gemm, m_dense, k_dense, n_dense, warmup, iters)?;

    // MoE expert gate_up: ~16 tokens × hidden @ (2×moe_inter); use 1408 as PLAN2 reference.
    let (m_moe, k_moe, n_moe) = (16usize, 2816usize, 1408usize);
    let moe_gflops = bench_linear_gflops(&mut gemm, m_moe, k_moe, n_moe, warmup, iters)?;

    Ok((dense_gflops, moe_gflops))
}

fn bench_linear_gflops(
    gemm: &mut Bf16Gemm,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iters: usize,
) -> Result<f64, Error> {
    let a = vec![0.01f32; m * k];
    let w = vec![0x3f80u16; n * k]; // bf16 1.0
    let mut c = vec![0.0f32; m * n];
    for _ in 0..warmup {
        gemm.matmul_f32_bf16_linear(&a, &w, &mut c, m, k, n)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iters {
        gemm.matmul_f32_bf16_linear(&a, &w, &mut c, m, k, n)?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * k as f64 * n as f64 * iters as f64;
    Ok(flops / elapsed / 1e9)
}

pub fn print_probe_result(r: &DeviceProbeResult) {
    println!("probe-device ok");
    println!("  device:              {}", r.device_name);
    println!("  unified memory:      {}", r.unified_memory);
    println!(
        "  working_set (rec):   {:.1} GiB",
        r.recommended_working_set_gib
    );
    println!("  memcpy bandwidth:    {:.1} GiB/s", r.memcpy_gib_per_s);
    println!(
        "  GEMM dense 256×2816×2816: {:.1} GFLOP/s",
        r.gemm_dense_gflops
    );
    println!(
        "  GEMM moe 16×2816×1408:    {:.1} GFLOP/s",
        r.gemm_moe_gflops
    );
    let dense_tflops = r.gemm_dense_gflops / 1000.0;
    let moe_tflops = r.gemm_moe_gflops / 1000.0;
    let step_tflop = 1.95;
    println!(
        "  dense-shape throughput: {:.3} TFLOP/s (~{:.0} ms for {:.2} TFLOP/step, kernel only)",
        dense_tflops,
        step_tflop / dense_tflops.max(0.001) * 1000.0,
        step_tflop,
    );
    println!(
        "  moe-shape throughput:   {:.3} TFLOP/s (16×2816×1408)",
        moe_tflops,
    );
}
