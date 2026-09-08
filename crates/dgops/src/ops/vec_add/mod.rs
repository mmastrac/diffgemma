//! out += addend, elementwise.

crate::op_kernel! {
    name = "vec_add_inplace",
    metal = "vec_add.metal",
    cuda = "ops/vec_add/vec_add",
    fixture = Fixture,
    tests = [
        tiny => tiny_fixture => (0.0, 1.0),
        long => long_fixture => (0.0, 1.0),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub addend: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0],
        addend: vec![0.5, -1.0, 2.0, 0.0],
    }
}

/// Long enough to cross several thread blocks.
pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.003).sin()).collect(),
        addend: (0..len)
            .map(|i| ((i as f32) * 0.007).cos() * 0.25)
            .collect(),
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    fix.x
        .iter()
        .zip(fix.addend.iter())
        .map(|(x, a)| x + a)
        .collect()
}
