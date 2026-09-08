//! AdamW update: p_new, m_new and v_new in one pass.

use crate::Error;

pub const ENTRY: &str = "adamw";
pub const METAL: &str = include_str!("adamw.metal");

#[cfg(all(feature = "cuda", dgops_cuda_kernels))]
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda/ops/adamw/adamw.cubin"));
#[cfg(all(feature = "cuda", not(dgops_cuda_kernels)))]
const CUBIN: &[u8] = &[];

/// Must stay layout-identical to AdamwParams in adamw.metal and adamw.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AdamwParams {
    pub step: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub p: Vec<f32>,
    pub g: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    /// 1-based optimizer step, used for bias correction.
    pub step: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.p.len()
    }

    pub fn params(&self) -> AdamwParams {
        AdamwParams {
            step: self.step,
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
}

/// The op returns [p_new, m_new, v_new], each of len elements.
pub fn out_len(f: &Fixture) -> usize {
    3 * f.len()
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        p: vec![1.0, -2.0, 0.5, 3.0],
        g: vec![0.1, -0.2, 0.3, 0.0],
        m: vec![0.0; 4],
        v: vec![0.0; 4],
        step: 1,
        lr: 3e-3,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.1,
    }
}

/// Non-zero moment estimates and a late step, to exercise bias correction.
pub fn bias_correction_fixture() -> Fixture {
    let len = 1024;
    Fixture {
        p: (0..len).map(|i| ((i as f32) * 0.37).sin() * 1.5).collect(),
        g: (0..len).map(|i| ((i as f32) * 0.71).cos() * 0.2).collect(),
        m: vec![0.01; len],
        v: vec![0.001; len],
        step: 100,
        lr: 3e-3,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.1,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let n = fix.len();
    let mut out = vec![0.0f32; 3 * n];
    let bc1 = 1.0 - fix.beta1.powi(fix.step as i32);
    let bc2 = 1.0 - fix.beta2.powi(fix.step as i32);
    for i in 0..n {
        let m_new = fix.beta1 * fix.m[i] + (1.0 - fix.beta1) * fix.g[i];
        let v_new = fix.beta2 * fix.v[i] + (1.0 - fix.beta2) * fix.g[i] * fix.g[i];
        let m_hat = m_new / bc1;
        let v_hat = v_new / bc2;
        let update = m_hat / (v_hat.sqrt() + fix.eps) + fix.weight_decay * fix.p[i];
        out[i] = fix.p[i] - fix.lr * update;
        out[n + i] = m_new;
        out[2 * n + i] = v_new;
    }
    out
}

pub fn gpu(fix: &Fixture) -> Result<Vec<f32>, Error> {
    #[cfg(target_os = "macos")]
    {
        metal::gpu(fix)
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        cuda::gpu(fix)
    }
    #[cfg(not(any(target_os = "macos", all(feature = "cuda", not(target_os = "macos")))))]
    {
        let _ = fix;
        Err(Error::Gpu(
            "no GPU backend enabled (build with --features cuda on a CUDA host)",
        ))
    }
}

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(test)]
mod tests {
    crate::op_oracle_matrix! {
        mod tiny,
        cpu = crate::ops::adamw::cpu,
        gpu = crate::ops::adamw::gpu,
        fixture = crate::ops::adamw::tiny_fixture,
        out_len = crate::ops::adamw::out_len,
        max_tol = 1e-6,
        min_cos = 0.999999,
    }

    crate::op_oracle_matrix! {
        mod bias_correction,
        cpu = crate::ops::adamw::cpu,
        gpu = crate::ops::adamw::gpu,
        fixture = crate::ops::adamw::bias_correction_fixture,
        out_len = crate::ops::adamw::out_len,
        max_tol = 1e-6,
        min_cos = 0.999999,
    }
}
