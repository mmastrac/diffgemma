//! Tier-1 gate for the f32 GEMM: a tile-boundary row must compute the same
//! value regardless of how many rows the problem has. The tile is BM rows
//! tall, so a kernel that resolves the tile's row offset from the problem's
//! height (rather than from the block index) silently changes row 0 when m
//! changes — which is exactly the failure mode a "same input, different seq"
//! divergence looks like.
#![cfg(feature = "cuda")]

use dgemm::{BM, BN, CUDA, ENTRY, Problem, THREADS, abi_params};
use gpukit::cuda::{DeviceBuffer, KernelArgs, cached_context, cached_source_kernel};

fn gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let ctx = cached_context().expect("ctx");
    let ba = DeviceBuffer::alloc(&ctx, a.len() * 4).unwrap();
    ba.write_f32(a).unwrap();
    let bb = DeviceBuffer::alloc(&ctx, b.len() * 4).unwrap();
    bb.write_f32(b).unwrap();
    let bc = DeviceBuffer::alloc(&ctx, m * n * 4).unwrap();
    let kernel = cached_source_kernel(CUDA, ENTRY).expect("kernel");
    let problem = Problem::linear(m, n, k);
    let p = abi_params(&problem);
    let mut args = KernelArgs::new();
    args.device_ptr(ba.device_ptr())
        .device_ptr(bb.device_ptr())
        .device_ptr(bc.device_ptr())
        .bytes(&p);
    let grid = (n.div_ceil(BN) as u32, m.div_ceil(BM) as u32, 1);
    ctx.launch(&kernel, grid, (THREADS, 1, 1), 0, &mut args)
        .expect("launch");
    ctx.synchronize().unwrap();
    let mut out = vec![0.0f32; m * n];
    bc.read_f32(&mut out).unwrap();
    out
}

#[test]
fn first_row_is_independent_of_m() {
    let k = 2816usize;
    let n = 4096usize;
    let a: Vec<f32> = (0..64 * k)
        .map(|i| ((i as f32 * 0.0013).sin() * 0.7) + 0.05)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 * 0.0007).cos() * 0.5) - 0.02)
        .collect();

    let mut reference: Option<Vec<f32>> = None;
    for m in [1usize, 2, 4, 8, 16, 20, 22, 32, 64, 65, 128] {
        let got = gemm(m, n, k, &a, &b);
        let row0 = &got[..n];
        match &reference {
            None => reference = Some(row0.to_vec()),
            Some(r) => {
                let mut worst = 0.0f32;
                for i in 0..n {
                    worst = worst.max((r[i] - row0[i]).abs());
                }
                assert!(worst < 1e-3, "m={m}: row 0 differs by {worst}");
            }
        }
    }
}
