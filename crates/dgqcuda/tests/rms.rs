//! dgq_rms_norm_dbg: is the reduction correct for a constant row?
#![cfg(feature = "cuda")]

use gpukit::cuda::{DeviceBuffer, KernelArgs, cached_context, cached_source_kernel};

#[test]
fn reduction_constant_row() {
    let hidden = 2816usize;
    let row = vec![1.0f32; hidden];
    let w = vec![1.0f32; hidden];
    let ctx = cached_context().expect("ctx");
    let bx = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();
    bx.write_f32(&row).unwrap();
    let bw = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();
    bw.write_f32(&w).unwrap();
    let bo = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();
    let k = cached_source_kernel(dgqcuda::KERNELS, "dgq_rms_norm_dbg").expect("kernel");
    let mut args = KernelArgs::new();
    args.device_ptr(bx.device_ptr())
        .device_ptr(bw.device_ptr())
        .device_ptr(bo.device_ptr())
        .u32(1)
        .u32(hidden as u32)
        .f32(1e-6);
    ctx.launch(&k, (1, 1, 1), (256, 1, 1), 0, &mut args)
        .expect("launch");
    ctx.synchronize().unwrap();
    let mut all = vec![0.0f32; hidden];
    bo.read_f32(&mut all).unwrap();
    println!("tail = {:?}", &all[hidden - 8..hidden]);
    println!("out0 = {}", all[0]);
    // Expected: red0 = 2816, inv = 1/sqrt(1 + 1e-6), out0 = inv
    assert!(
        (all[hidden - 6] - 2816.0).abs() < 1.0,
        "red0 = {}",
        all[hidden - 6]
    );
}
