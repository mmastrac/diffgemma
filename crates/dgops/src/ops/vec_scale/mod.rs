//! x *= scale, elementwise.

crate::op_kernel! {
    name = "vec_scale_inplace",
    metal = "vec_scale.metal",
    cuda = "vec_scale.cu",
    fixture = Fixture,
    tests = [
        tiny => tiny_fixture => (0.0, 1.0),
        long => long_fixture => (0.0, 1.0),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub scale: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, -2.0, 3.0, 0.0],
        scale: 0.5,
    }
}

pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.003).sin()).collect(),
        scale: -1.25,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.x.iter().map(|v| v * fix.scale).collect()
}
