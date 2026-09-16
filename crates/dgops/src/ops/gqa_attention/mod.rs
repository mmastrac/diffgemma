//! Grouped-query causal attention over a paged K/V region (f32).
//!
//! The KV region holds, per position \`t\` and KV head \`h\`, a head_dim K vector
//! followed by a head_dim V vector: \`kv[t * n_kv_heads * head_dim * 2 + ...]\`.
//! Query head \`qh\` reads KV head \`qh / n_groups\`. The mask is causal with an
//! optional sliding window; the online softmax is computed in one pass.

crate::op_kernel! {
    name = "gqa_attention",
    metal = "gqa_attention.metal",
    cuda = "gqa_attention.cu",
    fixture = Fixture => fix,
    abi = [
        in(q = fix.q),
        in(kv = fix.kv),
        out(out = fix.len()),
        pod(fix.params()),
        u32x2(fix.seq_len, fix.n_heads),
    ],
    launch = rows(fix.seq_len * fix.n_heads),
    result = (out, fix.len()),
    tests = [
        tiny => tiny_fixture => (1e-5, 0.9999),
        sliding_window => sliding_window_fixture => (1e-5, 0.9999),
        full_gqa => full_gqa_fixture => (1e-5, 0.9999),
        kv_prefix => kv_prefix_fixture => (1e-5, 0.9999),
    ],
}

/// Must stay layout-identical to GqaAttentionParams in .metal / .cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GqaAttentionParams {
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
    /// [seq_len, n_heads, head_dim] queries, already RoPE'd.
    pub q: Vec<f32>,
    /// [total_kv, n_kv_heads, 2 * head_dim]: K then V per head.
    pub kv: Vec<f32>,
    pub seq_len: usize,
    pub total_kv: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Keys outside the last \`sliding_window\` are masked (0 = full causal).
    pub sliding_window: usize,
    /// Keys with index < kv_cache_len are a shared prefix (prefill reuse).
    pub kv_cache_len: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.seq_len * self.n_heads * self.head_dim
    }

    pub fn n_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub fn params(&self) -> GqaAttentionParams {
        GqaAttentionParams {
            seq_len: self.seq_len as u32,
            total_kv: self.total_kv as u32,
            n_heads: self.n_heads as u32,
            n_kv_heads: self.n_kv_heads as u32,
            head_dim: self.head_dim as u32,
            n_groups: self.n_groups() as u32,
            mask_kind: 0,
            sliding_window: self.sliding_window as u32,
            kv_cache_len: self.kv_cache_len as u32,
            mask_neg: -1.0e30,
            rotary_dim: self.head_dim as u32,
            num_heads_rope: 0,
            elem_offset: 0,
        }
    }
}

fn fill(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.4 + ((i as f32) * seed * 0.37).cos() * 0.2)
        .collect()
}

/// Whether key \`ki\` is visible to query \`qi\` under the causal+window mask.
fn visible(p: &Fixture, qi: usize, ki: usize) -> bool {
    if ki >= p.kv_cache_len + qi + 1 {
        return false;
    }
    if p.sliding_window > 0 && ki + p.sliding_window <= p.kv_cache_len + qi {
        return false;
    }
    true
}

pub fn tiny_fixture() -> Fixture {
    let (seq_len, n_heads, n_kv_heads, head_dim) = (3usize, 4usize, 2usize, 8usize);
    let total_kv = seq_len;
    Fixture {
        q: fill(seq_len * n_heads * head_dim, 0.07),
        kv: fill(total_kv * n_kv_heads * 2 * head_dim, 0.11),
        seq_len,
        total_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        sliding_window: 0,
        kv_cache_len: 0,
    }
}

/// Sliding window 4 over 16 positions: only the last few keys stay visible.
pub fn sliding_window_fixture() -> Fixture {
    let (seq_len, n_heads, n_kv_heads, head_dim) = (16usize, 8usize, 2usize, 32usize);
    let total_kv = seq_len;
    Fixture {
        q: fill(seq_len * n_heads * head_dim, 0.017),
        kv: fill(total_kv * n_kv_heads * 2 * head_dim, 0.023),
        seq_len,
        total_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        sliding_window: 4,
        kv_cache_len: 0,
    }
}

/// The full-attention layer geometry: 16 heads, 2 KV heads, head_dim 512.
pub fn full_gqa_fixture() -> Fixture {
    let (seq_len, n_heads, n_kv_heads, head_dim) = (5usize, 16usize, 2usize, 512usize);
    let total_kv = seq_len;
    Fixture {
        q: fill(seq_len * n_heads * head_dim, 0.0031),
        kv: fill(total_kv * n_kv_heads * 2 * head_dim, 0.0073),
        seq_len,
        total_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        sliding_window: 0,
        kv_cache_len: 0,
    }
}

/// 3 new query positions attending to a 5-position resident prefix.
pub fn kv_prefix_fixture() -> Fixture {
    let (seq_len, n_heads, n_kv_heads, head_dim) = (3usize, 8usize, 2usize, 16usize);
    let kv_cache_len = 5usize;
    let total_kv = kv_cache_len + seq_len;
    Fixture {
        q: fill(seq_len * n_heads * head_dim, 0.013),
        kv: fill(total_kv * n_kv_heads * 2 * head_dim, 0.019),
        seq_len,
        total_kv,
        n_heads,
        n_kv_heads,
        head_dim,
        sliding_window: 0,
        kv_cache_len,
    }
}

/// Online-softmax GQA attention — the CPU oracle for both GPU bodies.
pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let hd = fix.head_dim;
    let nkv = fix.n_kv_heads;
    let n_groups = fix.n_groups();
    let t_total = fix.total_kv;
    let mut out = vec![0.0f32; fix.len()];

    for tok in 0..fix.seq_len {
        for qh in 0..fix.n_heads {
            let kvh = qh / n_groups;
            let q_off = (tok * fix.n_heads + qh) * hd;
            let qv = &fix.q[q_off..q_off + hd];

            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; hd];
            for t in 0..t_total {
                if !visible(fix, tok, t) {
                    continue;
                }
                let k_off = t * nkv * hd * 2 + kvh * hd;
                let kk = &fix.kv[k_off..k_off + hd];
                let d: f32 = qv.iter().zip(kk.iter()).map(|(a, b)| a * b).sum();
                let mn = m.max(d);
                let corr = (m - mn).exp();
                let p = (d - mn).exp();
                for a in acc.iter_mut() {
                    *a *= corr;
                }
                l = l * corr + p;
                m = mn;
                let v_off = k_off + nkv * hd;
                let vv = &fix.kv[v_off..v_off + hd];
                for (a, &vv_i) in acc.iter_mut().zip(vv.iter()) {
                    *a += p * vv_i;
                }
            }

            let o_off = (tok * fix.n_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = a / l;
            }
        }
    }
    out
}
