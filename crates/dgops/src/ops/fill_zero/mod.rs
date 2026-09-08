//! Zero a range of an f32 buffer.

crate::op_kernel! {
    name = "vec_fill_zero",
    metal = "fill_zero.metal",
    cuda = "fill_zero.cu",
    fixture = Fixture => fix,
    abi = [
        inout(buf = fix.x),
        u32x2(fix.base, fix.count),
    ],
    launch = 1d(fix.count),
    result = (buf, fix.len()),
    tests = [
        tiny => tiny_fixture => (0.0, 1.0),
        long => long_fixture => (0.0, 1.0),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub base: usize,
    pub count: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        base: 2,
        count: 3,
    }
}

/// Offset range spanning several thread blocks.
pub fn long_fixture() -> Fixture {
    let len = 8192;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.003).sin() + 2.0).collect(),
        base: 1000,
        count: 6000,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.x.clone();
    for v in out[fix.base..fix.base + fix.count].iter_mut() {
        *v = 0.0;
    }
    out
}
