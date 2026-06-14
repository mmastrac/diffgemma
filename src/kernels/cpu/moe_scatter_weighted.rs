//! CPU oracle for weighted MoE scatter from batched expert outputs.

use crate::kernels::sub::f16;
use crate::metal::{RouteScratch, CANVAS};

pub fn moe_scatter_weighted(
    expert_out: &[f32],
    route: &RouteScratch,
    hidden: usize,
) -> Vec<f32> {
    let mut moe_out = vec![0.0f32; CANVAS * hidden];
    let slots = route.num_slots as usize;
    for slot in 0..slots {
        let tok = route.token_list[slot] as usize;
        let kk = route.slot_list[slot] as usize;
        let w = f16::f16_bits_to_f32(route.weight[tok][kk]);
        let src = slot * hidden;
        let dst = tok * hidden;
        for d in 0..hidden {
            moe_out[dst + d] += w * expert_out[src + d];
        }
    }
    moe_out
}
