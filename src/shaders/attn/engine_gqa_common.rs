//! Shared params and Metal dispatch for encoder-path GQA kernels.

use crate::model::attention::{AttentionParams, MASK_NEG};

pub const MASK_CAUSAL_SLIDING: u32 = 0;
pub const MASK_ENCODER_EXTEND: u32 = 1;
pub const MASK_DECODER_BITMAP: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GqaParams {
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

impl GqaParams {
    pub fn for_rope(
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        elem_offset: u32,
    ) -> Self {
        Self {
            seq_len: seq_len as u32,
            total_kv: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: head_dim as u32,
            n_groups: 0,
            mask_kind: 0,
            sliding_window: 0,
            kv_cache_len: 0,
            mask_neg: MASK_NEG,
            rotary_dim: rotary_dim as u32,
            num_heads_rope: num_heads as u32,
            elem_offset,
        }
    }

    pub fn for_attention(
        seq_len: usize,
        total_kv: usize,
        params: &AttentionParams,
        mask_kind: u32,
        kv_cache_len: usize,
    ) -> Self {
        Self {
            seq_len: seq_len as u32,
            total_kv: total_kv as u32,
            n_heads: params.n_heads as u32,
            n_kv_heads: params.n_kv_heads as u32,
            head_dim: params.head_dim as u32,
            n_groups: params.n_groups as u32,
            mask_kind,
            sliding_window: params.sliding_window.unwrap_or(0) as u32,
            kv_cache_len: kv_cache_len as u32,
            mask_neg: MASK_NEG,
            rotary_dim: params.rotary_dim as u32,
            num_heads_rope: 0,
            elem_offset: 0,
        }
    }
}

#[cfg(target_os = "macos")]
pub fn set_params(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    params: &GqaParams,
    index: usize,
) {
    use objc2_metal::MTLComputeCommandEncoder;
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(params).cast(),
            std::mem::size_of::<GqaParams>(),
            index,
        );
    }
}

/// One thread per (head, query) — grid can exceed any fixed threadgroup width.
#[cfg(target_os = "macos")]
pub fn dispatch_head_query_2d(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    n_heads: usize,
    seq_len: usize,
) {
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads,
            height: seq_len,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}
