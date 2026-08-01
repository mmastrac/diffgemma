//! CPU reference for grouped MoE expert forward (Q4 + NVFP4 dequant loops).

use crate::dgq::block::q4_weight_at;
use crate::dgq::layout::{NVFP4_HEADER_BYTES, q4_row_bytes};
use crate::dgq::nvfp4::nvfp4_weight_at;
use crate::shaders::bf16;
use crate::shaders::cpu::gelu_pytorch_tanh_f32;

#[derive(Debug, Clone, Copy)]
pub struct MoeGroupedDims {
    pub canvas: u32,
    pub hidden: u32,
    pub moe_ff: u32,
    pub n_experts: u32,
}

/// CPU mirror of `moe_grouped` per-token math (Q4 dequant loop).
pub fn expert_forward_q4_mirror(
    x: &[f32],
    gate_up: &[u8],
    down: &[u8],
    moe_ff: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), hidden);
    let mut act = vec![0.0f32; moe_ff];
    for (r, act_r) in act.iter_mut().enumerate() {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for (k, &xk) in x.iter().enumerate() {
            g += q4_weight_at(gate_up, r, k, hidden) * xk;
            u += q4_weight_at(gate_up, r + moe_ff, k, hidden) * xk;
        }
        *act_r = bf16::round_bf16_f32(gelu_pytorch_tanh_f32(g) * u);
    }
    let mut out = vec![0.0f32; hidden];
    for (d, out_d) in out.iter_mut().enumerate() {
        let mut o = 0.0f32;
        for (k, &ak) in act.iter().enumerate() {
            o += q4_weight_at(down, d, k, moe_ff) * ak;
        }
        *out_d = bf16::round_bf16_f32(o);
    }
    out
}

/// CPU mirror of `moe_grouped_nvfp4` per-token math.
pub fn expert_forward_nvfp4_mirror(
    x: &[f32],
    gate_up: &[u8],
    down: &[u8],
    moe_ff: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), hidden);
    let gu_scale = f32::from_le_bytes(gate_up[0..4].try_into().expect("gu header"));
    let dn_scale = f32::from_le_bytes(down[0..4].try_into().expect("dn header"));
    let gu_body = &gate_up[NVFP4_HEADER_BYTES..];
    let dn_body = &down[NVFP4_HEADER_BYTES..];

    let mut act = vec![0.0f32; moe_ff];
    for (r, act_r) in act.iter_mut().enumerate() {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for (k, &xk) in x.iter().enumerate() {
            g += nvfp4_weight_at(gu_body, r, k, hidden, gu_scale) * xk;
            u += nvfp4_weight_at(gu_body, r + moe_ff, k, hidden, gu_scale) * xk;
        }
        *act_r = bf16::round_bf16_f32(gelu_pytorch_tanh_f32(g) * u);
    }
    let mut out = vec![0.0f32; hidden];
    for (d, out_d) in out.iter_mut().enumerate() {
        let mut o = 0.0f32;
        for (k, &ak) in act.iter().enumerate() {
            o += nvfp4_weight_at(dn_body, d, k, moe_ff, dn_scale) * ak;
        }
        *out_d = bf16::round_bf16_f32(o);
    }
    out
}

#[derive(Debug, Clone)]
pub struct GroupedRoute {
    pub offset: Vec<u32>,
    pub num_slots: u32,
    pub token_list: Vec<u32>,
    pub slot_list: Vec<u32>,
    pub weight: Vec<Vec<f32>>,
}

/// Accumulate weighted expert outputs into `moe_out` (canvas × hidden).
pub fn moe_grouped_q4(
    moe_out: &mut [f32],
    moe_in: &[f32],
    gate_up_blob: &[u8],
    down_blob: &[u8],
    route: &GroupedRoute,
    dims: MoeGroupedDims,
) {
    let canvas = dims.canvas as usize;
    let hidden = dims.hidden as usize;
    let moe_ff = dims.moe_ff as usize;
    let n_experts = dims.n_experts as usize;
    assert_eq!(moe_in.len(), canvas * hidden);
    assert_eq!(moe_out.len(), canvas * hidden);
    assert_eq!(route.offset.len(), n_experts);

    let gu_stride = (moe_ff * 2) * q4_row_bytes(hidden);
    let dn_stride = hidden * q4_row_bytes(moe_ff);

    for e in 0..n_experts {
        let end = if e + 1 < n_experts {
            route.offset[e + 1]
        } else {
            route.num_slots
        };
        for slot in route.offset[e]..end {
            let tok = route.token_list[slot as usize] as usize;
            // The kernel writes UNWEIGHTED per-SLOT rows (slot_out[slot] =
            // f32_round_bf16(o)); the router weight is applied downstream by
            // moe_scatter_weighted. The old weighted per-token accumulate here
            // predated that split and made these oracles fail (cos ~0.34).
            let x = &moe_in[tok * hidden..(tok + 1) * hidden];
            let gu = &gate_up_blob[e * gu_stride..(e + 1) * gu_stride];
            let dn = &down_blob[e * dn_stride..(e + 1) * dn_stride];
            let expert_out = expert_forward_q4_mirror(x, gu, dn, moe_ff, hidden);
            let out_row = &mut moe_out[slot as usize * hidden..(slot as usize + 1) * hidden];
            for (o, &v) in out_row.iter_mut().zip(expert_out.iter()) {
                *o = bf16::round_bf16_f32(v);
            }
        }
    }
}

pub fn moe_grouped_nvfp4(
    moe_out: &mut [f32],
    moe_in: &[f32],
    gate_up_blob: &[u8],
    down_blob: &[u8],
    route: &GroupedRoute,
    dims: MoeGroupedDims,
) {
    let canvas = dims.canvas as usize;
    let hidden = dims.hidden as usize;
    let moe_ff = dims.moe_ff as usize;
    let n_experts = dims.n_experts as usize;
    assert_eq!(moe_in.len(), canvas * hidden);
    assert_eq!(moe_out.len(), canvas * hidden);
    assert_eq!(route.offset.len(), n_experts);

    let gu_stride = crate::dgq::layout::nvfp4_matrix_bytes(moe_ff * 2, hidden);
    let dn_stride = crate::dgq::layout::nvfp4_matrix_bytes(hidden, moe_ff);

    for e in 0..n_experts {
        let end = if e + 1 < n_experts {
            route.offset[e + 1]
        } else {
            route.num_slots
        };
        for slot in route.offset[e]..end {
            let tok = route.token_list[slot as usize] as usize;
            // Kernel contract: unweighted per-SLOT rows (see moe_grouped_q4).
            let x = &moe_in[tok * hidden..(tok + 1) * hidden];
            let gu = &gate_up_blob[e * gu_stride..(e + 1) * gu_stride];
            let dn = &down_blob[e * dn_stride..(e + 1) * dn_stride];
            let expert_out = expert_forward_nvfp4_mirror(x, gu, dn, moe_ff, hidden);
            let out_row = &mut moe_out[slot as usize * hidden..(slot as usize + 1) * hidden];
            for (o, &v) in out_row.iter_mut().zip(expert_out.iter()) {
                *o = bf16::round_bf16_f32(v);
            }
        }
    }
}
