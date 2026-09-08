//! PyTorch tanh-approximation GELU, in place.

crate::op_kernel! {
    name = "gelu",
    metal = "gelu.metal",
    cuda = "gelu.cu",
    fixture = Fixture,
    tests = [
        tiny => tiny_fixture => (1e-5, 0.99999),
        mlp_shape => mlp_shape_fixture => (1e-5, 0.99999),
    ],
}

/// Matches the engine's gelu_tanh in include/activations.metal.
const GELU_TANH_COEF: f32 = 0.797_884_6;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![-2.0, -1.0, 0.0, 0.5, 1.5, 3.0, 10.229641],
    }
}

/// Exercises the MLP intermediate width, well past one thread block.
pub fn mlp_shape_fixture() -> Fixture {
    let len = 16 * 2112;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.001).sin()).collect(),
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
    fix.x.iter().copied().map(gelu_tanh).collect()
}
