//! Attention kernels batched through `GpuBatch` (shared engine pool/queue).

use crate::metal::batch::{set_bytes, GpuBatch};
use crate::model::attention::{AttentionParams, GqaMask, MASK_NEG};
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use super::attention::GpuAttentionKernels;

const MASK_CAUSAL_SLIDING: u32 = 0;
const MASK_ENCODER_EXTEND: u32 = 1;
const MASK_DECODER_BITMAP: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct GqaParams {
    seq_len: u32,
    total_kv: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_groups: u32,
    mask_kind: u32,
    sliding_window: u32,
    kv_cache_len: u32,
    mask_neg: f32,
    rotary_dim: u32,
    num_heads_rope: u32,
    elem_offset: u32,
}

pub fn rope_qk_batched(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(), Error> {
    if q.len() != seq_len * n_heads * head_dim
        || k.len() != seq_len * n_kv_heads * head_dim
        || freqs.len() != seq_len * rotary_dim
    {
        return Err(Error::Format("rope qk shape mismatch"));
    }

    let buf_q = batch.alloc_f32(q)?;
    let buf_k = batch.alloc_f32(k)?;
    let buf_f = batch.alloc_f32(freqs)?;

    let rope_q = GqaParams {
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
        num_heads_rope: n_heads as u32,
        elem_offset: 0,
    };
    let rope_k = GqaParams {
        num_heads_rope: n_kv_heads as u32,
        ..rope_q
    };

    encode_rope(batch, &kernels.rope_pipeline.pipeline, &buf_q, &buf_f, &rope_q, n_heads, seq_len);
    encode_rope(batch, &kernels.rope_pipeline.pipeline, &buf_k, &buf_f, &rope_k, n_kv_heads, seq_len);

    batch.register_read(buf_q, q);
    batch.register_read(buf_k, k);
    Ok(())
}

/// RoPE canvas K in-place at `k_canvas_elem_offset` within `k_buf`, RoPE Q in pool buffer, then GQA.
/// `k_buf`/`v_buf` must already hold encoder prefix; canvas V written at the same suffix offset.
pub fn decoder_gqa_gpu_kv_batched(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    attn_out: Option<&mut [f32]>,
    q_pre_rope: &[f32],
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_canvas_elem_offset: usize,
    freqs: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, Error> {
    let q_dim = seq_len * params.n_heads * params.head_dim;
    let out_dim = seq_len * params.n_heads * params.head_dim;
    if q_pre_rope.len() != q_dim {
        return Err(Error::Format("decoder gpu kv gqa shape mismatch"));
    }
    if let Some(out) = &attn_out {
        if out.len() != out_dim {
            return Err(Error::Format("decoder gpu kv gqa shape mismatch"));
        }
    }

    let buf_q = batch.alloc_f32(q_pre_rope)?;
    let buf_f = batch.alloc_f32(freqs)?;

    let rope_base = GqaParams {
        seq_len: seq_len as u32,
        total_kv: 0,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: params.head_dim as u32,
        n_groups: 0,
        mask_kind: 0,
        sliding_window: 0,
        kv_cache_len: 0,
        mask_neg: MASK_NEG,
        rotary_dim: params.rotary_dim as u32,
        num_heads_rope: params.n_heads as u32,
        elem_offset: 0,
    };
    let rope_k = GqaParams {
        num_heads_rope: params.n_kv_heads as u32,
        elem_offset: k_canvas_elem_offset as u32,
        ..rope_base
    };

    encode_rope(
        batch,
        &kernels.rope_pipeline.pipeline,
        &buf_q,
        &buf_f,
        &rope_base,
        params.n_heads,
        seq_len,
    );
    encode_rope(
        batch,
        &kernels.rope_pipeline.pipeline,
        &k_buf,
        &buf_f,
        &rope_k,
        params.n_kv_heads,
        seq_len,
    );

    gqa_batched_inner(
        batch,
        kernels,
        attn_out,
        q_pre_rope,
        None,
        Some((k_buf, v_buf)),
        seq_len,
        total_kv,
        params,
        mask,
        Some(buf_q),
    )
}

/// RoPE + GQA over GPU KV; returns attention output buffer when `attn_out` is `None`.
pub fn decoder_gqa_gpu_kv_batched_chained(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    q_pre_rope: &[f32],
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_canvas_elem_offset: usize,
    freqs: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    decoder_gqa_gpu_kv_batched(
        batch,
        kernels,
        None,
        q_pre_rope,
        k_buf,
        v_buf,
        k_canvas_elem_offset,
        freqs,
        seq_len,
        total_kv,
        params,
        mask,
    )?
    .ok_or(Error::Format("gqa gpu kv missing output buffer"))
}

pub fn gqa_batched(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    attn_out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<(), Error> {
    gqa_batched_inner(
        batch,
        kernels,
        Some(attn_out),
        q,
        Some((k, v)),
        None,
        seq_len,
        total_kv,
        params,
        mask,
        None,
    )?;
    Ok(())
}

pub fn gqa_batched_chained(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    gqa_batched_inner(
        batch,
        kernels,
        None,
        q,
        Some((k, v)),
        None,
        seq_len,
        total_kv,
        params,
        mask,
        None,
    )?
    .ok_or(Error::Format("gqa missing output buffer"))
}

/// GQA with K/V already resident in GPU buffers (`k_buf`/`v_buf` hold prefix+canvas).
pub fn gqa_batched_gpu_kv(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    attn_out: &mut [f32],
    q: &[f32],
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<(), Error> {
    gqa_batched_inner(
        batch,
        kernels,
        Some(attn_out),
        q,
        None,
        Some((k_buf, v_buf)),
        seq_len,
        total_kv,
        params,
        mask,
        None,
    )?;
    Ok(())
}

fn gqa_batched_inner(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    attn_out: Option<&mut [f32]>,
    q: &[f32],
    kv_cpu: Option<(&[f32], &[f32])>,
    kv_gpu: Option<(Retained<ProtocolObject<dyn MTLBuffer>>, Retained<ProtocolObject<dyn MTLBuffer>>)>,
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
    q_gpu: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, Error> {
    let q_dim = seq_len * params.n_heads * params.head_dim;
    let kv_dim_elems = total_kv * params.n_kv_heads * params.head_dim;
    let out_dim = seq_len * params.n_heads * params.head_dim;

    if q.len() != q_dim {
        return Err(Error::Format("gqa attention shape mismatch"));
    }
    if let Some(out) = &attn_out {
        if out.len() != out_dim {
            return Err(Error::Format("gqa attention shape mismatch"));
        }
    }
    if let Some((k, v)) = kv_cpu {
        if k.len() != kv_dim_elems || v.len() != kv_dim_elems {
            return Err(Error::Format("gqa attention shape mismatch"));
        }
    } else if kv_gpu.is_none() {
        return Err(Error::Format("gqa gpu kv buffers missing"));
    }

    let (mask_kind, kv_cache_len, positions, decoder_mask) = match mask {
        GqaMask::CausalSliding => (MASK_CAUSAL_SLIDING, 0usize, None, None),
        GqaMask::EncoderExtend {
            kv_cache_len,
            positions,
        } => (MASK_ENCODER_EXTEND, kv_cache_len, Some(positions), None),
        GqaMask::DecoderBitmap(m) => (MASK_DECODER_BITMAP, m.kv_cache_len, None, Some(m)),
    };

    let buf_q = if let Some(buf) = q_gpu {
        buf
    } else {
        batch.alloc_f32(q)?
    };
    let (buf_k, buf_v) = match kv_cpu {
        Some((k, v)) => (batch.alloc_f32(k)?, batch.alloc_f32(v)?),
        None => kv_gpu.expect("gpu kv"),
    };
    let buf_o = batch.alloc_f32_out(out_dim)?;

    let mut mask_buf = None;
    let mut pos_buf = None;

    if let Some(m) = decoder_mask {
        let packed: Vec<u8> = m.attend.iter().map(|&b| u8::from(b)).collect();
        mask_buf = Some(batch.alloc_bytes(&packed)?);
    }

    if let Some(pos) = positions {
        pos_buf = Some(batch.alloc_i64(pos)?);
    }

    let gpu_params = GqaParams {
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
    };

    encode_gqa(
        batch,
        &kernels.attn_pipeline.pipeline,
        &buf_q,
        &buf_k,
        &buf_v,
        &buf_o,
        mask_buf.as_deref(),
        pos_buf.as_deref(),
        &gpu_params,
        params.n_heads,
        seq_len,
    );

    if let Some(out) = attn_out {
        batch.register_read(buf_o, out);
        Ok(None)
    } else {
        Ok(Some(buf_o))
    }
}

fn encode_rope(
    batch: &GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buf_x: &ProtocolObject<dyn MTLBuffer>,
    buf_f: &ProtocolObject<dyn MTLBuffer>,
    params: &GqaParams,
    num_heads: usize,
    seq_len: usize,
) {
    batch.encoder().setComputePipelineState(pipeline);
    unsafe {
        batch.encoder().setBuffer_offset_atIndex(Some(buf_x), 0, 0);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_f), 0, 1);
    }
    set_bytes(batch.encoder(), params, 2);
    dispatch_2d(batch.encoder(), num_heads, seq_len);
}

fn encode_gqa(
    batch: &GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buf_q: &ProtocolObject<dyn MTLBuffer>,
    buf_k: &ProtocolObject<dyn MTLBuffer>,
    buf_v: &ProtocolObject<dyn MTLBuffer>,
    buf_o: &ProtocolObject<dyn MTLBuffer>,
    mask_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    pos_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    params: &GqaParams,
    n_heads: usize,
    seq_len: usize,
) {
    batch.encoder().setComputePipelineState(pipeline);
    unsafe {
        batch.encoder().setBuffer_offset_atIndex(Some(buf_q), 0, 0);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_k), 0, 1);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_v), 0, 2);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_o), 0, 3);
    }
    if let Some(mask) = mask_buf {
        unsafe {
            batch.encoder().setBuffer_offset_atIndex(Some(mask), 0, 4);
        }
    }
    if let Some(pos) = pos_buf {
        unsafe {
            batch.encoder().setBuffer_offset_atIndex(Some(pos), 0, 5);
        }
    }
    set_bytes(batch.encoder(), params, 6);
    dispatch_2d(batch.encoder(), n_heads, seq_len);
}

fn dispatch_2d(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, width: usize, height: usize) {
    let grid = MTLSize {
        width,
        height,
        depth: 1,
    };
    let tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}
