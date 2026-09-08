//! Scatter-add whole rows into a row-major destination with atomic adds.

crate::op_kernel! {
    name = "scatter_add_rows",
    metal = "scatter_add_rows.metal",
    cuda = "scatter_add_rows.cu",
    fixture = Fixture => fix,
    abi = [
        inout(buf_dst = fix.dst),
        in_u32(buf_idx = fix.indices),
        in(buf_src = fix.src),
        u32x2(fix.indices.len(), fix.hidden),
    ],
    launch = rows(fix.indices.len()),
    result = (buf_dst, fix.len()),
    tests = [
        tiny => tiny_fixture => (0.0, 1.0),
        repeated => repeated_fixture => (1e-6, 1.0),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    /// Destination rows, [dst.len()/hidden, hidden].
    pub dst: Vec<f32>,
    pub indices: Vec<u32>,
    pub src: Vec<f32>,
    pub hidden: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.dst.len()
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        dst: vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        indices: vec![0, 2, 0],
        src: vec![
            0.5, 0.25, 0.125, 0.0625, //
            1.0, 2.0, 3.0, 4.0, //
            0.5, 0.5, 0.5, 0.5,
        ],
        hidden: 4,
    }
}

/// Repeated and out-of-order indices over a hidden wider than one block.
pub fn repeated_fixture() -> Fixture {
    let dst_rows = 8;
    let hidden = 512;
    let indices: Vec<u32> = vec![0, 1, 1, 2, 3, 3, 3, 0];
    Fixture {
        dst: (0..dst_rows * hidden)
            .map(|i| ((i as f32) * 0.0013).cos() * 0.25)
            .collect(),
        src: (0..indices.len() * hidden)
            .map(|i| ((i as f32) * 0.0023).sin() * 0.5)
            .collect(),
        indices,
        hidden,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.dst.clone();
    for (t, &row) in fix.indices.iter().enumerate() {
        let dst_off = row as usize * fix.hidden;
        let src_off = t * fix.hidden;
        for d in 0..fix.hidden {
            out[dst_off + d] += fix.src[src_off + d];
        }
    }
    out
}
