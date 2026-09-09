//! dgq_rms_norm on a non-zero input, checked against the CPU formula.
#![cfg(feature = "cuda")]

use gpukit::cuda::{DeviceBuffer, KernelArgs, cached_context, cached_source_kernel};

#[test]
fn rms_norm_matches_cpu() {
    let (seq, hidden) = (4usize, 2816usize);
    // Non-zero everywhere, including index 0.
    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i as f32) * 0.0013).sin() + 0.7)
        .collect();
    let w: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32) * 0.0004).collect();
    let eps = 1e-6f32;

    let ctx = cached_context().expect("ctx");
    let bx = DeviceBuffer::alloc(&ctx, x.len() * 4).unwrap();
    bx.write_f32(&x).unwrap();
    let bw = DeviceBuffer::alloc(&ctx, w.len() * 4).unwrap();
    bw.write_f32(&w).unwrap();
    let bo = DeviceBuffer::alloc(&ctx, x.len() * 4).unwrap();
    let k = cached_source_kernel(dgqcuda::KERNELS, "dgq_rms_norm").expect("kernel");
    let mut args = KernelArgs::new();
    args.device_ptr(bx.device_ptr())
        .device_ptr(bw.device_ptr())
        .device_ptr(bo.device_ptr())
        .u32(seq as u32)
        .u32(hidden as u32)
        .f32(eps);
    ctx.launch(&k, (seq as u32, 1, 1), (256, 1, 1), 0, &mut args)
        .expect("launch");
    ctx.synchronize().unwrap();
    let mut got = vec![0.0f32; x.len()];
    bo.read_f32(&mut got).unwrap();

    for s in 0..seq {
        let ss: f32 = x[s * hidden..(s + 1) * hidden].iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / hidden as f32 + eps).sqrt();
        let want = x[s * hidden] * inv * w[0];
        println!("row {s}: got {} want {want} inv {inv}", got[s * hidden]);
        assert!(
            (got[s * hidden] - want).abs() < 1e-4,
            "row {s}: got {} want {want}",
            got[s * hidden]
        );
    }
}
