//! Split-half / proportional RoPE on \`[seq, heads, head_dim]\` (f32, in place).
//!
//! One thread per (head, position). The rotation pairs are \`(d, d + rot/2)\` for
//! sliding layers (\`rotary_dim == head_dim\`) and \`(d, head_dim/2 + d)\` for the
//! full-attention layers' proportional partial RoPE (\`rotary_dim < head_dim\`).
//! \`elem_offset\` rotates a K slice inside a larger Q+K buffer.

crate::op_kernel! {
    name = "apply_rope_heads",
    metal = "apply_rope_heads.metal",
    cuda = "apply_rope_heads.cu",
    fixture = Fixture => fix,
    abi = [
        inout(buf_x = fix.x),
        in(freqs = fix.freqs),
        pod(fix.params()),
        u32x2(fix.num_heads, fix.seq_len),
    ],
    launch = 1d(fix.num_heads * fix.seq_len),
    result = (buf_x, fix.len()),
    tests = [
        tiny => tiny_fixture => (1e-5, 0.9999),
        sliding_prefill => sliding_prefill_fixture => (1e-5, 0.9999),
        full_partial => full_partial_fixture => (1e-5, 0.9999),
        k_offset => k_offset_fixture => (1e-5, 0.9999),
    ],
}

/// Must stay layout-identical to GqaRopeParams in apply_rope_heads.metal / .cu
/// (and to the engine's GqaParams, which carries the same 13 fields).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GqaRopeParams {
    pub seq_len: u32,
    pub total_kv: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub n_groups: u32,
    pub mask_kind: u32,
    pub sliding_window: u32,
    pub kv_cache_len: u32,
    pub mask_neg: f32,
    pub rotary_dim: u32,
    pub num_heads_rope: u32,
    pub elem_offset: u32,
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub freqs: Vec<f32>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub elem_offset: u32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn params(&self) -> GqaRopeParams {
        GqaRopeParams {
            seq_len: self.seq_len as u32,
            total_kv: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: self.head_dim as u32,
            n_groups: 0,
            mask_kind: 0,
            sliding_window: 0,
            kv_cache_len: 0,
            mask_neg: -1.0e30,
            rotary_dim: self.rotary_dim as u32,
            num_heads_rope: self.num_heads as u32,
            elem_offset: self.elem_offset,
        }
    }
}

fn fill_tensor(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.37 + ((i as f32) * seed * 0.7).cos() * 0.21)
        .collect()
}

/// Interleaved [cos, sin] pairs, one per frequency — the engine's
/// \`cpu::compute_rope_freqs\`.
pub fn rope_freqs(seq_len: usize, rotary_dim: usize, full_head_dim: usize, theta: f32) -> Vec<f32> {
    let mut freqs = vec![0.0f32; seq_len * rotary_dim];
    let half = rotary_dim / 2;
    for s in 0..seq_len {
        let p = s as f32;
        let base = s * rotary_dim;
        for d in 0..half {
            let exponent = (2 * d) as f32 / full_head_dim as f32;
            let freq = 1.0 / theta.powf(exponent);
            let angle = p * freq;
            freqs[base + 2 * d] = angle.cos();
            freqs[base + 2 * d + 1] = angle.sin();
        }
    }
    freqs
}

pub fn tiny_fixture() -> Fixture {
    let (seq_len, num_heads, head_dim) = (4usize, 2usize, 8usize);
    Fixture {
        x: fill_tensor(seq_len * num_heads * head_dim, 0.11),
        freqs: rope_freqs(seq_len, head_dim, head_dim, 10_000.0),
        seq_len,
        num_heads,
        head_dim,
        rotary_dim: head_dim,
        elem_offset: 0,
    }
}

/// Sliding-layer prefill: 16 heads x 128 seq x head_dim 256.
pub fn sliding_prefill_fixture() -> Fixture {
    let (seq_len, num_heads, head_dim) = (128usize, 16usize, 256usize);
    Fixture {
        x: fill_tensor(seq_len * num_heads * head_dim, 0.017),
        freqs: rope_freqs(seq_len, head_dim, head_dim, 10_000.0),
        seq_len,
        num_heads,
        head_dim,
        rotary_dim: head_dim,
        elem_offset: 0,
    }
}

/// Full-attention layer: proportional partial RoPE (rotary 128 on head_dim 512).
pub fn full_partial_fixture() -> Fixture {
    let (seq_len, num_heads, head_dim, rotary_dim) = (64usize, 16usize, 512usize, 128usize);
    Fixture {
        x: fill_tensor(seq_len * num_heads * head_dim, 0.013),
        freqs: rope_freqs(seq_len, rotary_dim, head_dim, 1_000_000.0),
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        elem_offset: 0,
    }
}

/// RoPE only on the K slice inside a larger Q+K buffer.
pub fn k_offset_fixture() -> Fixture {
    let (seq_len, n_q_heads, n_kv_heads, head_dim) = (32usize, 16usize, 8usize, 256usize);
    let q_len = seq_len * n_q_heads * head_dim;
    let k_len = seq_len * n_kv_heads * head_dim;
    Fixture {
        x: fill_tensor(q_len + k_len, 0.019),
        freqs: rope_freqs(seq_len, head_dim, head_dim, 10_000.0),
        seq_len,
        num_heads: n_kv_heads,
        head_dim,
        rotary_dim: head_dim,
        elem_offset: q_len as u32,
    }
}

/// Rotate one head in place (the CPU oracle and the per-thread GPU body).
pub fn apply_rope(vec: &mut [f32], freqs: &[f32], rotary_dim: usize) {
    let head_dim = vec.len();
    let half = rotary_dim / 2;
    let half_head = head_dim / 2;
    let proportional = rotary_dim < head_dim;
    for d in 0..half {
        let cos = freqs[2 * d];
        let sin = freqs[2 * d + 1];
        let i1 = if proportional {
            half_head + d
        } else {
            d + half
        };
        let x0 = vec[d];
        let x1 = vec[i1];
        vec[d] = x0 * cos - x1 * sin;
        vec[i1] = x0 * sin + x1 * cos;
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.x.clone();
    let k_off = fix.elem_offset as usize;
    for s in 0..fix.seq_len {
        for h in 0..fix.num_heads {
            let off = k_off + (s * fix.num_heads + h) * fix.head_dim;
            let foff = s * fix.rotary_dim;
            apply_rope(
                &mut out[off..off + fix.head_dim],
                &fix.freqs[foff..foff + fix.rotary_dim],
                fix.rotary_dim,
            );
        }
    }
    out
}
