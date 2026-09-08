//! SwiGLU gate/up activation: \`out = gelu_tanh(gate) * up\` elementwise.
//!
//! This is the dense-MLP activation of the diffusion decoder layer (the engine
//! runs \`gelu\` then \`swiglu_mul\`; one fused body computes the same function).

crate::op_kernel! {
    name = "swiglu_gelu",
    metal = "swiglu_gelu.metal",
    cuda = "swiglu_gelu.cu",
    fixture = Fixture => fix,
    abi = [
        in(buf_gate = fix.gate),
        in(buf_up = fix.up),
        out(buf_out = fix.len()),
        u32(fix.len()),
    ],
    launch = 1d(fix.len()),
    result = (buf_out, fix.len()),
    tests = [
        tiny => tiny_fixture => (1e-6, 0.999999),
        mlp_shape => mlp_shape_fixture => (1e-6, 0.999999),
    ],
}

/// Matches the engine's gelu_tanh coefficient (include/activations.metal).
const GELU_TANH_COEF: f32 = 0.797_884_6;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.gate.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        gate: vec![-3.0, -1.0, -0.25, 0.0, 0.5, 1.5, 3.0, 10.229641],
        up: vec![2.0, -1.0, 0.5, 3.0, -2.0, 1.25, 0.0, -0.5],
    }
}

/// The real MLP intermediate width (2112), well past one thread block.
pub fn mlp_shape_fixture() -> Fixture {
    let len = 4 * 2112;
    Fixture {
        gate: (0..len).map(|i| ((i as f32) * 0.011).sin() * 2.5).collect(),
        up: (0..len).map(|i| ((i as f32) * 0.007).cos() * 1.5).collect(),
    }
}

pub fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    let u = GELU_TANH_COEF * (x + 0.044_715 * x3);
    let t = if u > 8.0 {
        1.0
    } else if u < -8.0 {
        -1.0
    } else {
        u.tanh()
    };
    0.5 * x * (1.0 + t)
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.gate
        .iter()
        .zip(fix.up.iter())
        .map(|(&g, &u)| gelu_tanh(g) * u)
        .collect()
}
