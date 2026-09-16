//! One MoE expert: CPU reference vs the CUDA path's GEMM + swiglu + GEMM.
#![cfg(feature = "cuda")]

use dgqcuda::config::ModelConfig;
use dgqcuda::weights::{LayerKeys, Weights};
use gpukit::cuda::{DeviceBuffer, KernelArgs, cached_context, cached_source_kernel};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

#[test]
fn expert_matches_cpu() {
    let dir = match std::env::var_os("DGQ_MODEL_DIR") {
        Some(d) => std::path::PathBuf::from(d),
        None => std::path::PathBuf::from("../../model/diffgemma-26b-a4b-it-q4"),
    };
    if !dir.join("model.dgq.bin").is_file() {
        eprintln!("skip: no local pack");
        return;
    }
    let cfg = ModelConfig::load(&dir).expect("config");
    let w = Weights::open(&dir, &cfg).expect("pack");
    let t = &cfg.text_config;
    let hidden = t.hidden_size;
    let moe_inter = t.moe_intermediate_size;
    let k = LayerKeys::new(0);
    let gu = w.tensor_f32(&k.experts_gate_up).expect("gu");
    let dn = w.tensor_f32(&k.experts_down).expect("dn");

    // A deterministic input vector.
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.001).sin() * 0.5)
        .collect();
    let expert = 79usize;

    // CPU reference: gate_up = x @ gu[e]^T, act = gelu(gate)*up, out = act @ dn[e]^T
    let gu_stride = moe_inter * 2 * hidden;
    let down_stride = hidden * moe_inter;
    let gu_e = &gu[expert * gu_stride..(expert + 1) * gu_stride];
    let dn_e = &dn[expert * down_stride..(expert + 1) * down_stride];
    let mut gu_out = vec![0.0f32; moe_inter * 2];
    for o in 0..moe_inter * 2 {
        let wr = &gu_e[o * hidden..(o + 1) * hidden];
        gu_out[o] = x.iter().zip(wr.iter()).map(|(a, b)| a * b).sum();
    }
    let mut act = vec![0.0f32; moe_inter];
    for i in 0..moe_inter {
        let g = gu_out[i];
        let u = gu_out[moe_inter + i];
        let x3 = g * g * g;
        let uu = 0.797_884_6 * (g + 0.044_715 * x3);
        let th = if uu > 8.0 {
            1.0
        } else if uu < -8.0 {
            -1.0
        } else {
            uu.tanh()
        };
        act[i] = 0.5 * g * (1.0 + th) * u;
    }
    let mut want = vec![0.0f32; hidden];
    for o in 0..hidden {
        let wr = &dn_e[o * moe_inter..(o + 1) * moe_inter];
        want[o] = act.iter().zip(wr.iter()).map(|(a, b)| a * b).sum();
    }

    // GPU: same three steps with the crate's kernels.
    let ctx = cached_context().expect("ctx");
    let bx = DeviceBuffer::alloc(&ctx, x.len() * 4).unwrap();
    bx.write_f32(&x).unwrap();
    let bgu = DeviceBuffer::alloc(&ctx, gu_e.len() * 4).unwrap();
    bgu.write_f32(gu_e).unwrap();
    let bdn = DeviceBuffer::alloc(&ctx, dn_e.len() * 4).unwrap();
    bdn.write_f32(dn_e).unwrap();
    let bguo = DeviceBuffer::alloc(&ctx, moe_inter * 2 * 4).unwrap();
    let bact = DeviceBuffer::alloc(&ctx, moe_inter * 4).unwrap();
    let bout = DeviceBuffer::alloc(&ctx, hidden * 4).unwrap();

    // gate_up GEMM (linear layout: W is [out, in])
    let problem = dgemm::Problem::linear(1, moe_inter * 2, hidden);
    let abi = dgemm::abi_params(&problem);
    let kernel = cached_source_kernel(dgemm::CUDA, dgemm::ENTRY).expect("gemm");
    let mut args = KernelArgs::new();
    args.device_ptr(bx.device_ptr())
        .device_ptr(bgu.device_ptr())
        .device_ptr(bguo.device_ptr())
        .bytes(&abi);
    gpukit::cuda::launch_grid(
        &ctx,
        &kernel,
        (moe_inter * 2).div_ceil(dgemm::BN),
        1,
        dgemm::THREADS,
        &mut args,
    )
    .expect("launch gu");

    // swiglu
    let sw = cached_source_kernel(dgqcuda::KERNELS, "dgq_swiglu_weighted").expect("swiglu");
    let mut args = KernelArgs::new();
    args.device_ptr(bguo.device_ptr())
        .device_ptr(unsafe { bguo.device_ptr() + (moe_inter as u64) * 4 })
        .f32(1.0)
        .device_ptr(bact.device_ptr())
        .u32(moe_inter as u32);
    let grid = (moe_inter as u32).div_ceil(256);
    ctx.launch(&sw, (grid, 1, 1), (256, 1, 1), 0, &mut args)
        .expect("swiglu");

    // down GEMM
    let problem = dgemm::Problem::linear(1, hidden, moe_inter);
    let abi = dgemm::abi_params(&problem);
    let mut args = KernelArgs::new();
    args.device_ptr(bact.device_ptr())
        .device_ptr(bdn.device_ptr())
        .device_ptr(bout.device_ptr())
        .bytes(&abi);
    gpukit::cuda::launch_grid(
        &ctx,
        &kernel,
        hidden.div_ceil(dgemm::BN),
        1,
        dgemm::THREADS,
        &mut args,
    )
    .expect("launch dn");
    ctx.synchronize().unwrap();
    let mut got = vec![0.0f32; hidden];
    bout.read_f32(&mut got).unwrap();

    let cos = cosine(&want, &got);
    println!("expert cos {cos}");
    println!("want[0..4] {:?}", &want[..4]);
    println!("got [0..4] {:?}", &got[..4]);
    assert!(cos > 0.9999, "expert cos {cos}");
}
