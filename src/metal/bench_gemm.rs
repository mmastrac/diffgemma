//! GEMM throughput bench: full-call vs resident compute vs MPS oracle.

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

/// Run `bench/mps_gemm.swift` (MPSGraph matmul oracle).
pub fn bench_mps_oracle(shapes: &[GemmShape], iters: usize) -> Result<Vec<GemmBenchRow>, Error> {
    let script = Path::new("bench/mps_gemm.swift");
    if !script.is_file() {
        return Err(Error::Format(
            "MPS oracle script missing: bench/mps_gemm.swift",
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
