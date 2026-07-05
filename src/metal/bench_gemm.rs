//! GEMM throughput bench: full-call vs resident compute vs MPSGraph oracle.

use crate::kernels::sub::bf16;
use crate::metal::buffer::BufferPool;
use crate::metal::gemm::Bf16Gemm;
use crate::safetensors::Error;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct GemmShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct GemmBenchRow {
    pub shape: GemmShape,
    pub label: String,
    pub gflops: f64,
}

pub fn parse_shapes(spec: &str) -> Result<Vec<GemmShape>, Error> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dims: Vec<usize> = part
            .split('x')
            .map(|s| {
                s.parse().map_err(|_| {
                    Error::Format("bench-gemm shape must be MxKxN, e.g. 256x2816x2816")
                })
            })
            .collect::<Result<_, _>>()?;
        if dims.len() != 3 {
            return Err(Error::Format("bench-gemm shape must be MxKxN"));
        }
        out.push(GemmShape {
            m: dims[0],
            k: dims[1],
            n: dims[2],
        });
    }
    if out.is_empty() {
        return Err(Error::Format("bench-gemm: no shapes"));
    }
    Ok(out)
}

pub fn bench_custom_kernel(shapes: &[GemmShape], iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    let mut gemm = Bf16Gemm::new()?;
    let warmup = 3usize;
    let mut rows = Vec::new();

    for &shape in shapes {
        let GemmShape { m, k, n } = shape;
        let full = bench_full_call(&mut gemm, m, k, n, warmup, iters)?;
        rows.push(GemmBenchRow {
            shape,
            label: "custom/full_call".into(),
            gflops: full,
        });
        let compute = bench_resident(&mut gemm, m, k, n, warmup, iters, false)?;
        rows.push(GemmBenchRow {
            shape,
            label: "custom/compute_only".into(),
            gflops: compute,
        });
        let batched = bench_resident(&mut gemm, m, k, n, warmup, iters, true)?;
        rows.push(GemmBenchRow {
            shape,
            label: format!("custom/batched_x{iters}"),
            gflops: batched,
        });
    }
    Ok(rows)
}

/// Fused Q4 `gemm_block` (step-kernel dense path): bf16 activations, on-the-fly dequant.
pub fn bench_gemm_block_q4(shapes: &[GemmShape], iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    use crate::kernels::sub::gemm_common;
    use crate::kernels::sub::gemm_q4;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

    let ctx = MetalContext::new()?;
    let warmup = 3usize;
    let mut rows = Vec::new();
    let mut pool = BufferPool::new();

    for &shape in shapes {
        let GemmShape { m, k, n } = shape;
        let pipeline = gemm_q4::pipeline_for(&ctx, n as u32, k as u32)?;
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.0007).cos() * 0.02)
            .collect();
        let fixture = gemm_q4::Fixture {
            x: vec![0.01f32; m * k],
            w_f32,
            m,
            n,
            k,
        };
        let w_q4 = fixture.w_q4();
        let buf_x = pool
            .allocate(&ctx.device, m * k * 2)
            .ok_or(Error::Format("bench gemm_block x"))?;
        let buf_y = pool
            .allocate(&ctx.device, m * n * 2)
            .ok_or(Error::Format("bench gemm_block y"))?;
        let buf_w = pool
            .allocate(&ctx.device, w_q4.len())
            .ok_or(Error::Format("bench gemm_block w"))?;
        BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&fixture.x));
        BufferPool::write_bytes(&buf_w, &w_q4);
        let (grid, tg) = gemm_common::dispatch_shape(m, n);

        let dispatch = |count: usize| -> Result<(), Error> {
            let cmd = ctx
                .queue
                .commandBuffer()
                .ok_or(Error::Format("cmd"))?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or(Error::Format("enc"))?;
            for _ in 0..count {
                enc.setComputePipelineState(&pipeline.pipeline);
                gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, m as u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            Ok(())
        };

        dispatch(warmup)?;
        let started = Instant::now();
        dispatch(iters)?;
        let rate = gflops(m, k, n, iters, started.elapsed().as_secs_f64());
        rows.push(GemmBenchRow {
            shape,
            label: format!("gemm_block/batched_x{iters}"),
            gflops: rate,
        });

        pool.release(m * k * 2, buf_x);
        pool.release(m * n * 2, buf_y);
        pool.release(w_q4.len(), buf_w);
    }
    Ok(rows)
}


/// Tunable GEMM (task #19): sweeps TUNE_BM/TUNE_BN configs via
/// #define prepend; per-lane fragment loads (no simdgroup_load from tgmem).
/// Each config is correctness-checked against gemm_block (expected BIT-exact:
/// same K-chain, dequant, and bf16 rounding).
pub fn bench_gemm_tunable(shapes: &[GemmShape], iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    use crate::kernels::sub::gemm_q4;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};

    const SHADER_TUNE: &str = shader_include::include_metal!("kernels/gemm_tunable.metal");
    const CONFIGS: &[(usize, usize)] = &[(32, 32), (64, 32), (32, 64), (64, 64)];
    let ctx = MetalContext::new()?;
    let warmup = 3usize;
    let mut rows = Vec::new();
    let mut pool = BufferPool::new();

    for &shape in shapes {
        let GemmShape { m, k, n } = shape;
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.0007).cos() * 0.02)
            .collect();
        let fixture = gemm_q4::Fixture {
            x: (0..m * k).map(|i| ((i as f32) * 0.013).sin() * 0.2).collect(),
            w_f32,
            m,
            n,
            k,
        };
        let w_q4 = fixture.w_q4();
        let buf_x = pool
            .allocate(&ctx.device, m * k * 2)
            .ok_or(Error::Format("bench tunable x"))?;
        let buf_y = pool
            .allocate(&ctx.device, m * n * 2)
            .ok_or(Error::Format("bench tunable y"))?;
        let buf_ref = pool
            .allocate(&ctx.device, m * n * 2)
            .ok_or(Error::Format("bench tunable ref"))?;
        let buf_w = pool
            .allocate(&ctx.device, w_q4.len())
            .ok_or(Error::Format("bench tunable w"))?;
        BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&fixture.x));
        BufferPool::write_bytes(&buf_w, &w_q4);

        // Reference output from the production kernel.
        {
            let ref_pipe = gemm_q4::pipeline_for(&ctx, n as u32, k as u32)?;
            let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
            enc.setComputePipelineState(&ref_pipe.pipeline);
            gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_ref, &buf_w, 0, m as u32);
            let (rgrid, rtg) = crate::kernels::sub::gemm_common::dispatch_shape(m, n);
            enc.dispatchThreadgroups_threadsPerThreadgroup(rgrid, rtg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        for &(bm, bn) in CONFIGS {
            let src = format!("#define TUNE_BM {bm}\n#define TUNE_BN {bn}\n{SHADER_TUNE}");
            let pipeline = ctx.compile_gemm_subkernel(
                &src,
                "gemm_tunable",
                n as u32,
                k as u32,
                false,
                crate::kernels::sub::QuantFormat::Q4Affine as u32,
                false,
            )?;
            let grid = MTLSize { width: n.div_ceil(bn), height: m.div_ceil(bm), depth: 1 };
            let tg = MTLSize { width: 128, height: 1, depth: 1 };
            let dispatch = |count: usize| -> Result<(), Error> {
                let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
                let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
                for _ in 0..count {
                    enc.setComputePipelineState(&pipeline.pipeline);
                    gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, m as u32);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                Ok(())
            };
            dispatch(1)?;
            // Correctness: bitwise compare vs production output.
            let mut mismatches = 0usize;
            let mut maxd = 0f32;
            {
                let ry = buf_ref.contents().as_ptr() as *const u16;
                let sy = buf_y.contents().as_ptr() as *const u16;
                for i in 0..(m * n) {
                    let rb = unsafe { *ry.add(i) };
                    let sb = unsafe { *sy.add(i) };
                    if rb != sb {
                        mismatches += 1;
                        let a = bf16::bf16_bits_to_f32(rb);
                        let b = bf16::bf16_bits_to_f32(sb);
                        maxd = maxd.max((a - b).abs());
                    }
                }
            }
            dispatch(warmup)?;
            let started = Instant::now();
            dispatch(iters)?;
            let rate = gflops(m, k, n, iters, started.elapsed().as_secs_f64());
            let bits = if mismatches == 0 {
                "BITEXACT".to_string()
            } else {
                format!("MISMATCH x{mismatches} max|d|={maxd:.5}")
            };
            rows.push(GemmBenchRow {
                shape,
                label: format!("tunable_{bm}x{bn}/x{iters} [{bits}]"),
                gflops: rate,
            });
        }

        pool.release(w_q4.len(), buf_w);

        // Raw (bf16 weights) 64x64 row: reference = gemm_block Raw (the
        // production dense/lm_head path), bitwise-compared.
        {
            let w_bits = bf16::f32_slice_to_bf16_bits(&fixture.w_f32);
            let buf_wr = pool
                .allocate(&ctx.device, w_bits.len() * 2)
                .ok_or(Error::Format("bench tunable wraw"))?;
            BufferPool::write_bf16(&buf_wr, &w_bits);
            {
                let ref_pipe = crate::kernels::sub::gemm_bf16::pipeline_for(&ctx, n as u32, k as u32)?;
                let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
                let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
                enc.setComputePipelineState(&ref_pipe.pipeline);
                gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_ref, &buf_wr, 0, m as u32);
                let (rgrid, rtg) = crate::kernels::sub::gemm_common::dispatch_shape(m, n);
                enc.dispatchThreadgroups_threadsPerThreadgroup(rgrid, rtg);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            let src = format!("#define TUNE_BM 64\n#define TUNE_BN 64\n{SHADER_TUNE}");
            let pipeline = ctx.compile_gemm_subkernel(
                &src,
                "gemm_tunable",
                n as u32,
                k as u32,
                false,
                crate::kernels::sub::QuantFormat::Raw as u32,
                false,
            )?;
            let grid = MTLSize { width: n.div_ceil(64), height: m.div_ceil(64), depth: 1 };
            let tg = MTLSize { width: 128, height: 1, depth: 1 };
            let dispatch = |count: usize| -> Result<(), Error> {
                let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
                let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
                for _ in 0..count {
                    enc.setComputePipelineState(&pipeline.pipeline);
                    gemm_q4::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_wr, 0, m as u32);
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
                Ok(())
            };
            dispatch(1)?;
            let mut mismatches = 0usize;
            {
                let ry = buf_ref.contents().as_ptr() as *const u16;
                let sy = buf_y.contents().as_ptr() as *const u16;
                for i in 0..(m * n) {
                    if unsafe { *ry.add(i) } != unsafe { *sy.add(i) } {
                        mismatches += 1;
                    }
                }
            }
            dispatch(warmup)?;
            let started = Instant::now();
            dispatch(iters)?;
            let rate = gflops(m, k, n, iters, started.elapsed().as_secs_f64());
            let bits = if mismatches == 0 {
                "BITEXACT".to_string()
            } else {
                format!("MISMATCH x{mismatches}")
            };
            rows.push(GemmBenchRow {
                shape,
                label: format!("tunable_raw_64x64/x{iters} [{bits}]"),
                gflops: rate,
            });
            pool.release(w_bits.len() * 2, buf_wr);
        }

        pool.release(m * k * 2, buf_x);
        pool.release(m * n * 2, buf_y);
        pool.release(m * n * 2, buf_ref);
    }
    Ok(rows)
}

/// Tiled bf16-weight `gemm_bf16` (step-kernel mixed-precision attention/FFN path):
/// bf16 activations, bf16 weights read straight into the half tile (no dequant).
pub fn bench_gemm_bf16(shapes: &[GemmShape], iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    use crate::kernels::sub::gemm_bf16;
    use crate::kernels::sub::gemm_common;
    use crate::kernels::sub::gemm_q8;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

    let ctx = MetalContext::new()?;
    let warmup = 3usize;
    let mut rows = Vec::new();
    let mut pool = BufferPool::new();

    for &shape in shapes {
        let GemmShape { m, k, n } = shape;
        let pipeline = gemm_bf16::pipeline_for(&ctx, n as u32, k as u32)?;
        let buf_x = pool
            .allocate(&ctx.device, m * k * 2)
            .ok_or(Error::Format("bench gemm_bf16 x"))?;
        let buf_y = pool
            .allocate(&ctx.device, m * n * 2)
            .ok_or(Error::Format("bench gemm_bf16 y"))?;
        let buf_w = pool
            .allocate(&ctx.device, n * k * 2)
            .ok_or(Error::Format("bench gemm_bf16 w"))?;
        BufferPool::write_bf16(&buf_x, &vec![0x3f00u16; m * k]);
        BufferPool::write_bf16(&buf_w, &vec![0x3f80u16; n * k]);
        let (grid, tg) = gemm_common::dispatch_shape(m, n);

        let dispatch = |count: usize| -> Result<(), Error> {
            let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
            for _ in 0..count {
                enc.setComputePipelineState(&pipeline.pipeline);
                gemm_q8::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, m as u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            Ok(())
        };

        dispatch(warmup)?;
        let started = Instant::now();
        dispatch(iters)?;
        let rate = gflops(m, k, n, iters, started.elapsed().as_secs_f64());
        rows.push(GemmBenchRow {
            shape,
            label: format!("gemm_bf16/batched_x{iters}"),
            gflops: rate,
        });

        pool.release(m * k * 2, buf_x);
        pool.release(m * n * 2, buf_y);
        pool.release(n * k * 2, buf_w);
    }
    Ok(rows)
}

fn bench_full_call(
    gemm: &mut Bf16Gemm,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iters: usize,
) -> Result<f64, Error> {
    let a = vec![0.01f32; m * k];
    let w = vec![0x3f80u16; n * k];
    let mut c = vec![0.0f32; m * n];
    for _ in 0..warmup {
        gemm.matmul_f32_bf16_linear(&a, &w, &mut c, m, k, n)?;
    }
    let started = Instant::now();
    for _ in 0..iters {
        gemm.matmul_f32_bf16_linear(&a, &w, &mut c, m, k, n)?;
    }
    Ok(gflops(m, k, n, iters, started.elapsed().as_secs_f64()))
}

fn bench_resident(
    gemm: &mut Bf16Gemm,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iters: usize,
    batched: bool,
) -> Result<f64, Error> {
    let a_bytes = m * k * 4;
    let w_bytes = n * k * 2;
    let c_bytes = m * n * 4;

    let (buf_a, buf_w, buf_c) = {
        let device = gemm.context().device.clone();
        let pool = gemm.pool_mut();
        let buf_a = pool
            .allocate(&device, a_bytes)
            .ok_or(Error::Format("bench buf_a failed"))?;
        let buf_w = pool
            .allocate(&device, w_bytes)
            .ok_or(Error::Format("bench buf_w failed"))?;
        let buf_c = pool
            .allocate(&device, c_bytes)
            .ok_or(Error::Format("bench buf_c failed"))?;
        BufferPool::write_f32(&buf_a, &vec![0.01f32; m * k]);
        BufferPool::write_bf16(&buf_w, &vec![0x3f80u16; n * k]);
        BufferPool::write_f32(&buf_c, &vec![0.0f32; m * n]);
        (buf_a, buf_w, buf_c)
    };

    for _ in 0..warmup {
        if batched {
            gemm.dispatch_f32_bf16_linear_batched(&buf_a, &buf_w, &buf_c, m, k, n, 1)?;
        } else {
            gemm.dispatch_f32_bf16_linear(&buf_a, &buf_w, &buf_c, m, k, n)?;
        }
    }
    let started = Instant::now();
    if batched {
        gemm.dispatch_f32_bf16_linear_batched(&buf_a, &buf_w, &buf_c, m, k, n, iters)?;
    } else {
        for _ in 0..iters {
            gemm.dispatch_f32_bf16_linear(&buf_a, &buf_w, &buf_c, m, k, n)?;
        }
    }
    let rate = gflops(m, k, n, iters, started.elapsed().as_secs_f64());

    let pool = gemm.pool_mut();
    pool.release(a_bytes, buf_a);
    pool.release(w_bytes, buf_w);
    pool.release(c_bytes, buf_c);
    Ok(rate)
}

fn gflops(m: usize, k: usize, n: usize, iters: usize, secs: f64) -> f64 {
    let flops = 2.0 * m as f64 * k as f64 * n as f64 * iters as f64;
    flops / secs / 1e9
}

/// Run `bench/mpsgraph_gemm.swift` (MetalPerformanceShaders matmul oracle).
pub fn bench_mpsgraph_oracle(shapes: &[GemmShape], iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    let script = Path::new("bench/mpsgraph_gemm.swift");
    if !script.is_file() {
        return Err(Error::Format(
            "MPSGraph oracle script missing: bench/mpsgraph_gemm.swift",
        ));
    }
    let mut rows = Vec::new();
    for &shape in shapes {
        let GemmShape { m, k, n } = shape;
        let out = Command::new("swift")
            .arg(script)
            .arg(format!("{m}"))
            .arg(format!("{k}"))
            .arg(format!("{n}"))
            .arg(format!("{iters}"))
            .output()
            .map_err(Error::Io)?;
        if !out.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
            return Err(Error::Format("mps oracle swift failed"));
        }
        let gflops: f64 = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .map_err(|_| Error::Format("mps oracle bad stdout"))?;
        rows.push(GemmBenchRow {
            shape,
            label: "mps/matmul".into(),
            gflops,
        });
    }
    Ok(rows)
}

pub fn print_bench_rows(rows: &[GemmBenchRow]) {
    println!("bench-gemm ok");
    for r in rows {
        let GemmShape { m, k, n } = r.shape;
        let tflops = r.gflops / 1000.0;
        println!(
            "  {m}×{k}×{n}  {:<24} {:.1} GFLOP/s ({:.3} TFLOP/s)",
            r.label, r.gflops, tflops
        );
    }
}
