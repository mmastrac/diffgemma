//! Backward pass of the tanh-approximation GELU.

crate::op_kernel! {
    name = "gelu_backward",
    metal = "gelu_backward.metal",
    cuda = "gelu_backward.cu",
    fixture = Fixture => fix,
    abi = [
        in(buf_g = fix.g),
        in(buf_dy = fix.dy),
        out(buf_out = fix.len()),
        u32(fix.len()),
    ],
    launch = 1d(fix.len()),
    result = (buf_out, fix.len()),
    tests = [
        tiny => tiny_fixture => (1e-5, 0.99999),
        long => long_fixture => (1e-4, 0.99999),
    ],
}

/// Matches the engine's gelu_tanh in include/activations.metal.
const GELU_TANH_COEF: f32 = 0.797_884_6;

#[derive(Debug, Clone)]
pub struct Fixture {
    /// Pre-activation input to gelu.
    pub g: Vec<f32>,
    pub dy: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.g.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        g: vec![-2.0, -1.0, 0.0, 0.5, 1.5, 3.0, 10.229641],
        dy: vec![1.0, 0.5, -1.0, 2.0, 0.25, 1.0, 3.0],
    }
}

/// Several blocks' worth of elements across the tanh saturation knee.
pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        g: (0..len).map(|i| ((i as f32) * 0.001).sin() * 3.0).collect(),
        dy: (0..len).map(|i| ((i as f32) * 0.002).cos() * 0.5).collect(),
    }
}

/// d/dx gelu_tanh(x), with the tanh clamped to +-1 outside |u| > 8.
pub fn gelu_tanh_grad(x: f32) -> f32 {
    let x3 = x * x * x;
    let u = GELU_TANH_COEF * (x + 0.044_715 * x3);
    let t = if u > 8.0 {
        1.0
    } else if u < -8.0 {
        -1.0
    } else {
        u.tanh()
    };
    let du = GELU_TANH_COEF * (1.0 + 0.134_145 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.g
        .iter()
        .zip(fix.dy.iter())
        .map(|(&g, &dy)| dy * gelu_tanh_grad(g))
        .collect()
}
