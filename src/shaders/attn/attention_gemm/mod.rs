//! GEMM-attention for full-layer prefill (and the QK stage of top-k
//! attention, `DGQ_ATTN_TOPK_DECODE`).
//!
//! The MMA path is constrained by hd=512 / 8x8-fragment occupancy shape.
//! During prefill the score matrix `S = Q.K^T` fits in memory per 256-row
//! sub-chunk, so full attention decomposes into two big GEMMs at higher rates
//! plus a rowwise softmax:
//!   1. `attn_gemm_qk`      NT-GEMM  S[i,t] = <Q_i, K_t>
//!   2. `attn_gemm_softmax` rowwise  P = exp(S - rowmax) (masked), denom L
//!   3. `attn_gemm_pv`      NN-GEMM  O[i,d] = sum_t P[i,t] V[t,d] / L_i
//!      No 1/sqrt(d) (folded into QK-norm upstream). P is left unnormalized; PV
//!      divides by L at store — mirroring `attention_mma_full`'s final divide so the
//!      two share numerics (f16 K/V, f32 accumulate). Not bit-identical.
//!      Prefill runs the full decomposition; denoise reaches the QK stage through
//!      top-k attention (default on), with `attention_mma_full` as the fallback.

#[cfg(target_os = "macos")]
use crate::Error;

pub const ENTRY_QK: &str = "attn_gemm_qk";
pub const ENTRY_SOFTMAX: &str = "attn_gemm_softmax";
pub const ENTRY_PV: &str = "attn_gemm_pv";

pub const SHADER: &str = include_str!("attention_gemm.metal");

/// Fragment-tile geometry (must match the shader constants).
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const SOFTMAX_TPG: usize = 256;

/// Tunable config for a prefill-attention sweep. The two GEMMs have
/// different shapes (QK: M=canvas,K=hd,N=T; PV: M=canvas,K=T,N=hd), so their
/// tiles tune independently — compiled by prepending `#define AG_*`.
#[derive(Clone, Copy, Debug)]
pub struct TuneCfg {
    pub hc: usize,
    pub qk_bm: usize,
    pub qk_bn: usize,
    pub pv_bm: usize,
    pub pv_bn: usize,
    pub sm_tpg: usize,
}

impl Default for TuneCfg {
    fn default() -> Self {
        Self {
            hc: 4,
            qk_bm: 64,
            qk_bn: 64,
            pv_bm: 64,
            pv_bn: 64,
            sm_tpg: 256,
        }
    }
}

impl TuneCfg {
    /// Compile-time validity: BN divides 128 (loader thread split) and is a
    /// multiple of 16; BM a multiple of 16; SM_TPG a power of two in [32,1024].
    pub fn valid(&self) -> bool {
        let ok_bn = |bn: usize| bn.is_multiple_of(16) && 128 % bn == 0;
        let ok_bm = |bm: usize| bm.is_multiple_of(16) && bm > 0 && bm <= 128;
        ok_bn(self.qk_bn)
            && ok_bn(self.pv_bn)
            && ok_bm(self.qk_bm)
            && ok_bm(self.pv_bm)
            && self.sm_tpg.is_power_of_two()
            && (32..=1024).contains(&self.sm_tpg)
            && (1..=16).contains(&self.hc)
    }
}

/// Prepend the tile #defines to specialize the kernels for a sweep. Un-prepended
/// (production) source keeps the shipped 64x64/256 defaults, so golden is safe.
pub fn tuned_source(bm: usize, bn: usize, sm_tpg: usize) -> String {
    format!("#define AG_BM {bm}\n#define AG_BN {bn}\n#define AG_SM_TPG {sm_tpg}\n{SHADER}")
}

/// Host mirror of the shader `AttnGemmDims`. All buffer base offsets (per head,
/// per KV head) are applied host-side via buffer offset args; this struct
/// carries shapes, row strides, and the softmax mask parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnGemmDims {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_row_stride: u32,
    pub b_row_stride: u32,
    pub s_row_stride: u32,
    pub out_row_stride: u32,
    pub causal: u32,
    pub kv_len: u32,
    pub hd: u32,
    pub group: u32,
    pub nkv: u32,
    pub s_head_stride: u32,
    pub head_base: u32,
}

/// Padded score-matrix width: keys rounded up to the 64-wide N-tile so the QK
/// grid, the softmax pad region, and the PV BK=32 contraction all align.
#[inline]
pub fn n_pad(t_total: usize) -> usize {
    t_total.next_multiple_of(BN)
}

#[cfg(target_os = "macos")]
pub fn pipelines(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
) -> Result<[crate::metal::device::ComputePipeline; 3], Error> {
    Ok([
        ctx.compile_subkernel(SHADER, ENTRY_QK, variant)?,
        ctx.compile_subkernel(SHADER, ENTRY_SOFTMAX, variant)?,
        ctx.compile_subkernel(SHADER, ENTRY_PV, variant)?,
    ])
}

/// F32-side-KV variant (FC30) — Q/K/V from the f32 side ring, all-float
/// MMA, f32 probs. Matches the MMA-full side path precision.
#[cfg(target_os = "macos")]
pub fn pipelines_side(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
) -> Result<[crate::metal::device::ComputePipeline; 3], Error> {
    let bools = [crate::shaders::variant::FcBool {
        index: 30,
        value: true,
    }];
    Ok([
        ctx.compile_subkernel_ex(SHADER, ENTRY_QK, variant, "side", &bools, &[])?,
        ctx.compile_subkernel_ex(SHADER, ENTRY_SOFTMAX, variant, "side", &bools, &[])?,
        ctx.compile_subkernel_ex(SHADER, ENTRY_PV, variant, "side", &bools, &[])?,
    ])
}

/// Production pipelines specialized for a tunable tile config: QK +
/// softmax compiled with the QK tile, PV with the PV tile (independent shapes).
/// The default cfg (64x64/64x64/256) yields byte-identical kernels to
/// `pipelines`/`pipelines_side`, so a default-flag build is unchanged.
#[cfg(target_os = "macos")]
pub fn pipelines_cfg(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
    cfg: TuneCfg,
    side: bool,
) -> Result<[crate::metal::device::ComputePipeline; 3], Error> {
    use crate::shaders::variant::FcBool;
    let bools: &[FcBool] = if side {
        &[FcBool {
            index: 30,
            value: true,
        }]
    } else {
        &[]
    };
    let label = if side { "cfg_side" } else { "cfg" };
    let src_qk = tuned_source(cfg.qk_bm, cfg.qk_bn, cfg.sm_tpg);
    let src_pv = tuned_source(cfg.pv_bm, cfg.pv_bn, cfg.sm_tpg);
    Ok([
        ctx.compile_subkernel_ex(&src_qk, ENTRY_QK, variant, label, bools, &[])?,
        ctx.compile_subkernel_ex(&src_qk, ENTRY_SOFTMAX, variant, label, bools, &[])?,
        ctx.compile_subkernel_ex(&src_pv, ENTRY_PV, variant, label, bools, &[])?,
    ])
}

// --------------------------------------------------------------------------
// GPU oracle: run the 3-kernel decomposition over an `attention::Fixture` and
// return bf16-rounded outputs, to cross-check against the CPU attention oracle
// (same target as `attention::gpu_mma_full`). Full layers (hd=512) only.
// --------------------------------------------------------------------------
#[cfg(target_os = "macos")]
pub fn gpu(
    f: &crate::shaders::attention::Fixture,
    causal: bool,
    side: bool,
) -> Result<Vec<f32>, Error> {
    use crate::shaders::attn::harness::AttnRig;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    // Standard rig: bf16 Q/out, f16 KV +8 pad key rows (direct-load whole
    // 8-key tiles), f32 side-ring mirror for the FC30 path.
    let mut rig = AttnRig::from_fixture(f, 8, true)?;
    let (canvas, n_q_heads, hd) = (rig.canvas, rig.n_q_heads, rig.hd);
    let t_total = rig.t_total;
    let np = n_pad(t_total);

    let [pipe_qk, pipe_sm, pipe_pv] = if side {
        pipelines_side(&rig.ctx, KernelVariant::PRODUCTION)?
    } else {
        pipelines(&rig.ctx, KernelVariant::PRODUCTION)?
    };
    // Scratch: S (f32), P (half or f32 in the side path), lrow (f32) — HC slices.
    let hc = 4.min(n_q_heads);
    let p_elem = if side { 4 } else { 2 };
    let buf_s = rig.alloc(hc * canvas * np * 4, "alloc s")?;
    let buf_p = rig.alloc(hc * canvas * np * p_elem, "alloc p")?;
    let buf_lrow = rig.alloc(hc * canvas * 4, "alloc lrow")?;

    let mut dims = AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: rig.kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: u32::from(causal),
        kv_len: f.kv_len,
        hd: hd as u32,
        group: rig.group as u32,
        nkv: rig.nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };

    let tg128 = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let tg_sm = MTLSize {
        width: SOFTMAX_TPG,
        height: 1,
        depth: 1,
    };

    rig.run_once(|enc| {
        let mut h0 = 0usize;
        while h0 < n_q_heads {
            let hb = (n_q_heads - h0).min(hc);
            dims.head_base = h0 as u32;
            let dims_pv = AttnGemmDims {
                n: hd as u32,
                k: t_total as u32,
                a_row_stride: np as u32,
                ..dims
            };
            let grid_qk = MTLSize {
                width: t_total.div_ceil(BN),
                height: canvas.div_ceil(BM),
                depth: hb,
            };
            let grid_sm = MTLSize {
                width: canvas,
                height: hb,
                depth: 1,
            };
            let grid_pv = MTLSize {
                width: hd.div_ceil(BN),
                height: canvas.div_ceil(BM),
                depth: hb,
            };
            enc.setComputePipelineState(&pipe_qk.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&rig.buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
                if side {
                    enc.setBuffer_offset_atIndex(Some(rig.kvf()), 0, 9);
                }
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

            enc.setComputePipelineState(&pipe_sm.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 2);
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

            enc.setComputePipelineState(&pipe_pv.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_out), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
                if side {
                    enc.setBuffer_offset_atIndex(Some(rig.kvf()), 0, 9);
                }
            }
            gpu_common::set_bytes(enc, &dims_pv, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            h0 += hb;
        }
    })?;

    Ok(rig.read_out_f32(f.out_len()))
}

/// Causal CPU reference (prefill mask): query row `tok` sits at absolute
/// position `kv_len + tok` and attends keys `[0, kv_len + tok]`. Same input
/// rounding as `attention::cpu` (Q -> bf16, KV -> f16, out -> bf16) so the
/// tolerance matches the all-valid oracle. Prefill-only — the production
/// non-causal path is covered by the `attention::cpu` oracle.
#[cfg(all(test, target_os = "macos"))]
pub fn cpu_causal(f: &crate::shaders::attention::Fixture, round_kv_f16: bool) -> Vec<f32> {
    use crate::shaders::bf16;
    use crate::shaders::f16;
    let hd = f.head_dim();
    let nkv = f.n_kv();
    let n_q_heads = f.n_q_heads;
    let canvas = f.canvas;
    let kv_len = f.kv_len as usize;
    let group = n_q_heads / nkv;
    let q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    // f16 path rounds KV to f16 (main cache); f32-side path keeps raw f32.
    let kv: Vec<f32> = if round_kv_f16 {
        f.kvcache
            .iter()
            .map(|&v| f16::f16_bits_to_f32(f16::f32_to_f16_bits(v)))
            .collect()
    } else {
        f.kvcache.clone()
    };
    let mut out = vec![0.0f32; f.out_len()];
    for tok in 0..canvas {
        for qh in 0..n_q_heads {
            let kvh = qh / group;
            let q_off = (tok * n_q_heads + qh) * hd;
            let qv = &q[q_off..q_off + hd];
            let t_stop = kv_len + tok; // inclusive causal cutoff
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; hd];
            for t in 0..=t_stop {
                let k_off = t * nkv * hd * 2 + kvh * hd;
                let d: f32 = qv
                    .iter()
                    .zip(kv[k_off..k_off + hd].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let mn = m.max(d);
                let corr = (m - mn).exp();
                let p = (d - mn).exp();
                let v_off = t * nkv * hd * 2 + nkv * hd + kvh * hd;
                for (a, &vv) in acc.iter_mut().zip(kv[v_off..v_off + hd].iter()) {
                    *a = *a * corr + p * vv;
                }
                l = l * corr + p;
                m = mn;
            }
            let o_off = (tok * n_q_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = bf16::store_bf16_round_half(a / l);
            }
        }
    }
    out
}

/// Model-shaped full-layer fixture (canvas=256, 16 Q / 2 KV, hd=512) for
/// benching the GEMM decomposition against `attention_mma_full` at realistic KV lengths.
#[cfg(target_os = "macos")]
pub fn model_full_fixture(kv_len: u32) -> crate::shaders::attention::Fixture {
    use crate::shaders::cpu::attention::LayerAttnParams;
    let (canvas, n_q_heads, nkv, hd) = (256usize, 16usize, 2usize, 512usize);
    model_full_fixture_with(
        kv_len,
        canvas,
        n_q_heads,
        nkv,
        hd,
        LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: nkv as u32,
            is_full: true,
            v_proj: 0,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: 0,
        },
    )
}

/// Same as `model_full_fixture` but with the production prefill super-chunk
/// canvas (M=1024 = 4 sub-chunks). Matches the shape `bench-prefill-super`
/// routes through `encode_attn_topk` in production.
pub fn model_full_fixture_prod(kv_len: u32) -> crate::shaders::attention::Fixture {
    use crate::shaders::cpu::attention::LayerAttnParams;
    let (canvas, n_q_heads, nkv, hd) = (1024usize, 16usize, 2usize, 512usize);
    model_full_fixture_with(
        kv_len,
        canvas,
        n_q_heads,
        nkv,
        hd,
        LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: nkv as u32,
            is_full: true,
            v_proj: 0,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: 0,
        },
    )
}

fn model_full_fixture_with(
    kv_len: u32,
    canvas: usize,
    n_q_heads: usize,
    nkv: usize,
    hd: usize,
    layer: crate::shaders::cpu::attention::LayerAttnParams,
) -> crate::shaders::attention::Fixture {
    let t_total = kv_len as usize + canvas;
    let mut kvcache = vec![0.0f32; t_total * nkv * hd * 2];
    for (i, x) in kvcache.iter_mut().enumerate() {
        *x = ((i % 4096) as f32 * 0.0007).sin() * 0.35;
    }
    crate::shaders::attention::Fixture {
        q: (0..canvas * n_q_heads * hd)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect(),
        kvcache,
        layer,
        canvas,
        n_q_heads,
        kv_len,
    }
}

/// Time `iters` back-to-back runs of the full 3-kernel sequence over all
/// heads (one full-attention layer, one prefill sub-chunk) in a single command
/// buffer; 1 warm-up round + min over timed rounds (factors out clock ramp).
/// Returns mean ms per layer. `causal` mirrors `gpu()` (benches historically
/// dispatched the prefill mask, causal=1).
#[cfg(target_os = "macos")]
pub fn bench_gpu(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    hc: usize,
    causal: bool,
) -> Result<f64, Error> {
    use crate::shaders::attn::harness::AttnRig;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    let mut rig = AttnRig::from_fixture(f, 8, false)?;
    let (canvas, n_q_heads, hd) = (rig.canvas, rig.n_q_heads, rig.hd);
    let t_total = rig.t_total;
    let np = n_pad(t_total);

    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&rig.ctx, KernelVariant::PRODUCTION)?;
    let hc = hc.clamp(1, n_q_heads);
    let buf_s = rig.alloc(hc * canvas * np * 4, "alloc s")?;
    let buf_p = rig.alloc(hc * canvas * np * 2, "alloc p")?;
    let buf_lrow = rig.alloc(hc * canvas * 4, "alloc lrow")?;

    let mut dims = AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: rig.kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: u32::from(causal),
        kv_len: f.kv_len,
        hd: hd as u32,
        group: rig.group as u32,
        nkv: rig.nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };
    let tg128 = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let tg_sm = MTLSize {
        width: SOFTMAX_TPG,
        height: 1,
        depth: 1,
    };

    rig.time_rounds(iters, |enc| {
        let mut h0 = 0usize;
        while h0 < n_q_heads {
            let hb = (n_q_heads - h0).min(hc);
            dims.head_base = h0 as u32;
            let dims_pv = AttnGemmDims {
                n: hd as u32,
                k: t_total as u32,
                a_row_stride: np as u32,
                ..dims
            };
            let grid_qk = MTLSize {
                width: t_total.div_ceil(BN),
                height: canvas.div_ceil(BM),
                depth: hb,
            };
            let grid_sm = MTLSize {
                width: canvas,
                height: hb,
                depth: 1,
            };
            let grid_pv = MTLSize {
                width: hd.div_ceil(BN),
                height: canvas.div_ceil(BM),
                depth: hb,
            };
            enc.setComputePipelineState(&pipe_qk.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&rig.buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipe_sm.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 2);
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipe_pv.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_out), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
            }
            gpu_common::set_bytes(enc, &dims_pv, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            h0 += hb;
        }
    })
}

/// Sweep bench: compile the kernels for a `TuneCfg` (QK/softmax
/// with the QK tile, PV with the PV tile) and time the full head-chunked prefill
/// sequence at model shape. Returns mean ms/layer (min over warmed rounds), or
/// Err if the config fails to compile (bad tile / register spill). `side` = the
/// f32 side-KV variant. `causal` mirrors `gpu()` (the sweep historically
/// dispatched the prefill mask, causal=1).
#[cfg(target_os = "macos")]
pub fn bench_tuned(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    cfg: TuneCfg,
    side: bool,
    causal: bool,
) -> Result<f64, Error> {
    use crate::shaders::attn::harness::AttnRig;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::{FcBool, KernelVariant};
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    let mut rig = AttnRig::from_fixture(f, 8, true)?;
    let (canvas, n_q_heads, hd) = (rig.canvas, rig.n_q_heads, rig.hd);
    let t_total = rig.t_total;
    let np = n_pad(t_total);
    let hc = cfg.hc.clamp(1, n_q_heads);

    let prod = KernelVariant::PRODUCTION;
    let side_fc: &[FcBool] = if side {
        &[FcBool {
            index: 30,
            value: true,
        }]
    } else {
        &[]
    };
    let src_qk = tuned_source(cfg.qk_bm, cfg.qk_bn, cfg.sm_tpg);
    let src_pv = tuned_source(cfg.pv_bm, cfg.pv_bn, cfg.sm_tpg);
    let pipe_qk = rig
        .ctx
        .compile_subkernel_ex(&src_qk, ENTRY_QK, prod, "tune", side_fc, &[])?;
    let pipe_sm =
        rig.ctx
            .compile_subkernel_ex(&src_qk, ENTRY_SOFTMAX, prod, "tune", side_fc, &[])?;
    let pipe_pv = rig
        .ctx
        .compile_subkernel_ex(&src_pv, ENTRY_PV, prod, "tune", side_fc, &[])?;

    let p_elem = if side { 4 } else { 2 };
    let buf_s = rig.alloc(hc * canvas * np * 4, "alloc s")?;
    let buf_p = rig.alloc(hc * canvas * np * p_elem, "alloc p")?;
    let buf_lrow = rig.alloc(hc * canvas * 4, "alloc lrow")?;

    let mut dims = AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: rig.kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: u32::from(causal),
        kv_len: f.kv_len,
        hd: hd as u32,
        group: rig.group as u32,
        nkv: rig.nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };
    let tg_qk = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let tg_pv = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let tg_sm = MTLSize {
        width: cfg.sm_tpg,
        height: 1,
        depth: 1,
    };

    rig.time_rounds(iters, |enc| {
        let mut h0 = 0usize;
        while h0 < n_q_heads {
            let hb = (n_q_heads - h0).min(hc);
            dims.head_base = h0 as u32;
            let dims_pv = AttnGemmDims {
                n: hd as u32,
                k: t_total as u32,
                a_row_stride: np as u32,
                ..dims
            };
            let grid_qk = MTLSize {
                width: t_total.div_ceil(cfg.qk_bn),
                height: canvas.div_ceil(cfg.qk_bm),
                depth: hb,
            };
            let grid_sm = MTLSize {
                width: canvas,
                height: hb,
                depth: 1,
            };
            let grid_pv = MTLSize {
                width: hd.div_ceil(cfg.pv_bn),
                height: canvas.div_ceil(cfg.pv_bm),
                depth: hb,
            };
            enc.setComputePipelineState(&pipe_qk.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&rig.buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
                if side {
                    enc.setBuffer_offset_atIndex(Some(rig.kvf()), 0, 9);
                }
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg_qk);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipe_sm.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 2);
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipe_pv.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_out), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
                if side {
                    enc.setBuffer_offset_atIndex(Some(rig.kvf()), 0, 9);
                }
            }
            gpu_common::set_bytes(enc, &dims_pv, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg_pv);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            h0 += hb;
        }
    })
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "attention_gemm",
    entry: "attn_gemm_qk",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// GEMM-attention decomposition vs the CPU attention oracle (the target
    /// `attention::gpu_mma_full` uses). Full-layer shape (hd=512, GQA group 8),
    /// non-causal (denoise-equivalent, all-valid).
    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp8_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, false, false).expect("gpu attn_gemm");
        let oracle = crate::shaders::attention::cpu(&f);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp2_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, false, false).expect("gpu attn_gemm");
        let oracle = crate::shaders::attention::cpu(&f);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Causal prefill mask vs the causal CPU reference (f16 KV). Full-layer
    /// shape, GQA group 8; kv_len=28, canvas=16 exercises per-row cutoffs
    /// across ragged 64-key N-tiles.
    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp8_causal_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, false).expect("gpu attn_gemm causal");
        let oracle = cpu_causal(&f, true);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// F32-side-KV path (FC30): all-float MMA reading the f32 side ring;
    /// vs the causal CPU reference with RAW f32 KV (no f16 rounding).
    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp8_causal_side_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, true).expect("gpu attn_gemm causal side");
        let oracle = cpu_causal(&f, false);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Premise check: GEMM-attention vs `attention_mma_full` at real
    /// full-layer shape (canvas=256, 16 Q / 2 KV, hd=512). One command buffer
    /// each, min-of-rounds. Ignored (timing).
    /// Run: `cargo test --release attn_gemm_bench -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn attn_gemm_bench() {
        let iters = 10usize;
        println!("  kv    mma_full   e17(hc16)  e17(hc8)   e17(hc4)   e17(hc2)  best-x");
        for kv_len in [8192u32, 30000, 60000] {
            let f = model_full_fixture(kv_len);
            let mma_full = crate::shaders::attention::bench_path(&f, iters, 3).unwrap();
            let hc16 = bench_gpu(&f, iters, 16, true).unwrap();
            let hc8 = bench_gpu(&f, iters, 8, true).unwrap();
            let hc4 = bench_gpu(&f, iters, 4, true).unwrap();
            let hc2 = bench_gpu(&f, iters, 2, true).unwrap();
            let best = hc16.min(hc8).min(hc4).min(hc2);
            println!(
                "{kv_len:>6}  {mma_full:9.3}  {hc16:9.3}  {hc8:9.3}  {hc4:9.3}  {hc2:9.3}  {:.2}x",
                mma_full / best
            );
        }
    }
}
