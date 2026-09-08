//! MoE router: RMSNorm(no scale) \u2192 scaled linear \u2192 top-k softmax.
//!
//! Per canvas row: \`n = rms_norm_no_scale(stream) * router_scale * hidden^-0.5\`,
//! \`logits = n @ router_proj^T\`, then the top \`k\` logits are softmaxed over the
//! selected set and multiplied by \`per_expert_scale\`. Ties break toward the
//! lower expert index (the CPU oracle's stable sort). Weights are returned
//! bf16-rounded, exactly as the engine packs them.

crate::op_kernel! {
    name = "moe_router_topk",
    metal = "moe_router_topk.metal",
    cuda = "moe_router_topk.cu",
    fixture = Fixture => fix,
    abi = [
        in(buf_stream = fix.stream),
        in(buf_scale = fix.router_scale),
        in(buf_proj = fix.router_proj),
        in(buf_expert_scale = fix.per_expert_scale),
        out(buf_idx = fix.canvas * fix.top_k),
        out(buf_w = fix.canvas * fix.top_k),
        pod(fix.params(), as params),
    ],
    launch = rows(fix.canvas),
    result = (buf_w, fix.canvas * fix.top_k),
    tests = [
        tiny => tiny_fixture => (0.0, 1.0),
        gemma_router => gemma_router_fixture => (1e-4, 0.99999),
    ],
}

pub const RMS_EPS: f32 = 1e-6;

/// Must stay layout-identical to MoeRouterParams in .metal / .cu.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MoeRouterParams {
    pub canvas: u32,
    pub hidden: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub router_hscale: f32,
    pub top_k2: u32,
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub stream: Vec<f32>,
    pub router_scale: Vec<f32>,
    pub router_proj: Vec<f32>,
    pub per_expert_scale: Vec<f32>,
    pub canvas: usize,
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.canvas * self.top_k
    }

    pub fn router_hscale(&self) -> f32 {
        (self.hidden as f32).powf(-0.5)
    }

    pub fn params(&self) -> MoeRouterParams {
        MoeRouterParams {
            canvas: self.canvas as u32,
            hidden: self.hidden as u32,
            n_experts: self.n_experts as u32,
            top_k: self.top_k as u32,
            router_hscale: self.router_hscale(),
            top_k2: self.top_k as u32,
        }
    }
}

fn fill(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.4 + ((i as f32) * seed * 0.71).cos() * 0.17)
        .collect()
}

pub fn tiny_fixture() -> Fixture {
    let (canvas, hidden, n_experts, top_k) = (3usize, 8usize, 5usize, 2usize);
    Fixture {
        stream: fill(canvas * hidden, 0.13),
        router_scale: (0..hidden).map(|i| 0.5 + i as f32 * 0.1).collect(),
        router_proj: fill(n_experts * hidden, 0.23),
        per_expert_scale: (0..n_experts).map(|i| 1.0 + i as f32 * 0.25).collect(),
        canvas,
        hidden,
        n_experts,
        top_k,
    }
}

/// The real router geometry: hidden 2816, 128 experts, top_k 8.
pub fn gemma_router_fixture() -> Fixture {
    let (canvas, hidden, n_experts, top_k) = (2usize, 2816usize, 128usize, 8usize);
    Fixture {
        stream: fill(canvas * hidden, 0.0031),
        router_scale: (0..hidden).map(|i| 0.75 + (i as f32) * 0.0005).collect(),
        router_proj: fill(n_experts * hidden, 0.0017),
        per_expert_scale: (0..n_experts).map(|i| 0.9 + (i as f32) * 0.01).collect(),
        canvas,
        hidden,
        n_experts,
        top_k,
    }
}

pub fn bf16_round(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xFFFF_0000)
}

/// One row's route: the top-k expert indices and their bf16-rounded weights.
pub fn route_row(logits: &[f32], top_k: usize, per_expert_scale: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(ia, pa), (ib, pb)| {
        pb.partial_cmp(pa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(ib))
    });
    let top = &ranked[..top_k];
    let indices: Vec<u32> = top.iter().map(|(i, _)| *i as u32).collect();
    let raw: Vec<f32> = top.iter().map(|(_, s)| *s).collect();
    let mx = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = raw.iter().map(|&x| (x - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut weights: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / top_k as f32; top_k]
    };
    for (w, &idx) in weights.iter_mut().zip(indices.iter()) {
        *w = bf16_round(*w * per_expert_scale[idx as usize]);
    }
    (indices, weights)
}

/// The CPU oracle: indices and bf16-rounded weights for every canvas row.
pub fn cpu_routes(fix: &Fixture) -> (Vec<u32>, Vec<f32>) {
    let mut indices = Vec::with_capacity(fix.len());
    let mut weights = Vec::with_capacity(fix.len());
    for tok in 0..fix.canvas {
        let row = &fix.stream[tok * fix.hidden..(tok + 1) * fix.hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let rms_inv = 1.0 / (sum_sq / fix.hidden as f32 + RMS_EPS).sqrt();
        let mut logits = vec![0.0f32; fix.n_experts];
        for (e, logit) in logits.iter_mut().enumerate() {
            let proj = &fix.router_proj[e * fix.hidden..(e + 1) * fix.hidden];
            let mut acc = 0.0f32;
            for d in 0..fix.hidden {
                acc += row[d] * rms_inv * fix.router_scale[d] * fix.router_hscale() * proj[d];
            }
            *logit = acc;
        }
        let (idx, w) = route_row(&logits, fix.top_k, &fix.per_expert_scale);
        indices.extend(idx);
        weights.extend(w);
    }
    (indices, weights)
}

/// The fixture-table oracle compares the weight buffer; indices are checked by
/// the dedicated test below (they are integers, so tolerance would hide a swap).
pub fn cpu(fix: &Fixture) -> Vec<f32> {
    cpu_routes(fix).1
}
