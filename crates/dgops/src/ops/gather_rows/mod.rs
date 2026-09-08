//! Gather rows by index from a row-major [tokens, hidden] f32 source.

crate::op_kernel! {
    name = "gather_rows",
    metal = "gather_rows.metal",
    cuda = "ops/gather_rows/gather_rows",
    fixture = Fixture,
    tests = [
        tiny => tiny_fixture => (0.0, 1.0),
        moe => moe_fixture => (0.0, 1.0),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub src: Vec<f32>,
    pub indices: Vec<u32>,
    pub hidden: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.indices.len() * self.hidden
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        src: vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        indices: vec![2, 0, 1],
        hidden: 4,
    }
}

/// Repeated and out-of-order indices over a hidden wider than one block.
pub fn moe_fixture() -> Fixture {
    let tokens = 64;
    let hidden = 512;
    Fixture {
        src: (0..tokens * hidden)
            .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
            .collect(),
        indices: (0..32).map(|i| ((i * 7 + 3) % tokens) as u32).collect(),
        hidden,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for (t, &row) in fix.indices.iter().enumerate() {
        let src_off = row as usize * fix.hidden;
        let dst_off = t * fix.hidden;
        out[dst_off..dst_off + fix.hidden].copy_from_slice(&fix.src[src_off..src_off + fix.hidden]);
    }
    out
}
