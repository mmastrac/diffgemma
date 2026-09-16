//! dgq_rms_norm on a large-magnitude row (real mlp_down scale).
#![cfg(feature = "cuda")]

use gpukit::cuda::{DeviceBuffer, KernelArgs, cached_context, cached_source_kernel};

#[test]
fn rms_large_row() {
    let (hidden, eps) = (2816usize, 1e-6f32);
    let row: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.0031).sin() * 40.0 + ((i as f32) * 0.0007).cos() * 20.0)
        .collect();
    let w: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32) * 0.0004).collect();
    let ss: f32 = row.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / hidden as f32 + eps).sqrt();
    let ctx = cached_context().expect("ctx");
    let bx = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();
    bx.write_f32(&row).unwrap();
    let bw = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();
    bw.write_f32(&w).unwrap();
    let bo = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();
    let k = cached_source_kernel(dgqcuda::KERNELS, "dgq_rms_norm").expect("kernel");
    let mut args = KernelArgs::new();
    args.device_ptr(bx.device_ptr())
        .device_ptr(bw.device_ptr())
        .device_ptr(bo.device_ptr())
        .u32(1)
        .u32(hidden as u32)
        .f32(eps);
    ctx.launch(&k, (1, 1, 1), (256, 1, 1), 0, &mut args)
        .expect("launch");
    ctx.synchronize().unwrap();
    let mut got = vec![0.0f32; 2];
    bo.read_f32(&mut got).unwrap();
    let want = row[0] * inv * w[0];
    println!("ss={ss} inv={inv} got={} want={want}", got[0]);
    assert!((got[0] - want).abs() < 1e-3, "got {} want {want}", got[0]);
}
