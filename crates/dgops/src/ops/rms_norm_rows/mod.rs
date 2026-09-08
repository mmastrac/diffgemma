//! Per-row Gemma RMSNorm over \`[seq_len, hidden]\` (f32 I/O, optional affine
//! scale). Ported from the engine kernel of the same name so the diffusion
//! pipeline can run on CUDA against the same CPU oracle the engine uses.

crate::op_kernel! {
    name = "rms_norm_rows",
    metal = "rms_norm_rows.metal",
    cuda = "rms_norm_rows.cu",
    fixture = Fixture => fix,
    abi = [
        in(buf_x = fix.x),
        in(buf_w = fix.weight),
        out(buf_o = fix.len()),
        u32x2(fix.seq_len, fix.hidden),
        f32(fix.eps),
    ],
    launch = rows(fix.seq_len),
    result = (buf_o, fix.len()),
    tests = [
        tiny => tiny_fixture => (1e-5, 0.9999),
        mlp_shape => mlp_shape_fixture => (1e-5, 0.9999),
        gemma_shape => gemma_shape_fixture => (1e-5, 0.9999),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub weight: Vec<f32>,
    pub seq_len: usize,
    pub hidden: usize,
    pub eps: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.seq_len * self.hidden
    }
}

pub fn tiny_fixture() -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: vec![1.0, 0.5, 2.0, 1.5],
        seq_len: 2,
        hidden: 4,
        eps: 1e-6,
    }
}

/// Rows wider than one thread block, and wider than one warp.
pub fn mlp_shape_fixture() -> Fixture {
    let (seq_len, hidden) = (3, 2112);
    let len = seq_len * hidden;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect(),
        weight: (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect(),
        seq_len,
        hidden,
        eps: 1e-6,
    }
}

/// The model's real hidden width (DiffusionGemma-26B: hidden_size 2816).
pub fn gemma_shape_fixture() -> Fixture {
    let (seq_len, hidden) = (5, 2816);
    let len = seq_len * hidden;
    Fixture {
        x: (0..len)
            .map(|i| ((i as f32) * 0.0031).sin() * 1.7 + ((i as f32) * 0.0007).cos() * 0.3)
            .collect(),
        weight: (0..hidden).map(|i| 0.5 + (i as f32) * 0.0005).collect(),
        seq_len,
        hidden,
        eps: 1e-6,
    }
}

/// Per-row RMSNorm: \`out = x / sqrt(mean(x^2) + eps) * weight\`.
/// The oracle for both GPU bodies, and the engine's \`cpu::rms_norm_rows\`.
pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for s in 0..fix.seq_len {
        let off = s * fix.hidden;
        let row = &fix.x[off..off + fix.hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let rms_inv = 1.0 / (sum_sq / fix.hidden as f32 + fix.eps).sqrt();
        let dst = &mut out[off..off + fix.hidden];
        for i in 0..fix.hidden {
            dst[i] = row[i] * rms_inv * fix.weight[i];
        }
    }
    out
}
