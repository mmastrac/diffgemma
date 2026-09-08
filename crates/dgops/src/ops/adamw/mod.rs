//! AdamW update: p_new, m_new and v_new in one pass.

crate::op_kernel! {
    name = "adamw",
    metal = "adamw.metal",
    cuda = "adamw.cu",
    fixture = Fixture => fix,
    abi = [
        in(buf_p = fix.p),
        in(buf_g = fix.g),
        in(buf_m = fix.m),
        in(buf_v = fix.v),
        out(buf_out = out_len(fix)),
        pod(fix.params()),
        u32(fix.len()),
    ],
    launch = 1d(fix.len()),
    result = (buf_out, out_len(fix)),
    tests = [
        tiny => { fixture = tiny_fixture, gpu = gpu, out_len = out_len, tol = 1e-6, cos = 0.999999 },
        bias_correction => { fixture = bias_correction_fixture, gpu = gpu, out_len = out_len, tol = 1e-6, cos = 0.999999 },
    ],
}

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
