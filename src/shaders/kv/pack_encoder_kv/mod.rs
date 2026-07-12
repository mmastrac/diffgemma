//! Pack engine GpuKvCache f32 K/V into monolithic b4 layout.

use crate::safetensors::Error;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "pack_encoder_kv";

pub const SHADER: &str = include_str!("pack_encoder_kv.metal");

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

/// Quantized-KV variant (uint function constant 4 = KvFormat code): grid
/// depth = head_dim/32 groups. `fmt` must be a quantized format (q8/q4).
#[cfg(target_os = "macos")]
pub fn pipeline_fmt_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let uints = [crate::shaders::variant::FcUInt {
        index: 4,
        value: fmt.code(),
    }];
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, fmt.label(), &[], &uints)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    keys: &ProtocolObject<dyn MTLBuffer>,
    values: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    token_count: u32,
    dst_pos: u32,
    nkv: u32,
    hd: u32,
    kv_region_bytes: u64,
    src_pos: u32,
    kv_ring_mask: u32,
) {
    let shape = [token_count, dst_pos, nkv, hd];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(keys), 0, 0);
        enc.setBuffer_offset_atIndex(Some(values), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 2);
    }
    crate::shaders::gpu_common::set_bytes(enc, &shape, 3);
    crate::shaders::gpu_common::set_bytes(enc, &kv_region_bytes, 4);
    crate::shaders::gpu_common::set_bytes(enc, &src_pos, 5);
    crate::shaders::gpu_common::set_bytes(enc, &kv_ring_mask, 6);
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(
    token_count: usize,
    nkv: usize,
    hd: usize,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use crate::shaders::kv_quant::KvFormat;
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: token_count,
            height: nkv,
            // quantized: one thread per 32-group (quantizes K + V); f16: per element.
            depth: if fmt == KvFormat::F16 { hd } else { hd / 32 },
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    )
}

// Armor for the KV-plumbing family (`pack_encoder_kv` / `unpack_encoder_kv` /
// `kv_f32_side_hydrate`) — the three kernels behind the 2026-07-10 ring-write
// hazard postmortem that had NO coverage. These round-trip the encoder f32
// K/V through the byte-packed monolithic cache and back, exercising the ring
// slot arithmetic (`slot = pos & mask`) that clobbered live window slots when
// it was wrong. Rounding-mode-robust by construction (see the module doc on
// each test): f16 asserts bit-exact on f16-exact inputs, q8 asserts within the
// lossy-format tolerance, and hydrate is pinned bit-exact GPU-to-GPU against
// unpack (identical packed bytes, identical dequant helpers — no CPU f16/q8
// rounding model in the loop).
#[cfg(all(test, target_os = "macos"))]
mod roundtrip_tests {
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::f16::round_half;
    use crate::shaders::gpu_common::set_bytes;
    use crate::shaders::kv_quant::{KvFormat, q8_roundtrip, rel_rms};
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    const NKV: usize = 2;
    const HD: usize = 64; // 2 groups of 32 for q8

    fn wave(n: usize, phase: f32, amp: f32, freq: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * freq + phase).sin() * amp)
            .collect()
    }

    /// Pack `keys`/`values` (linear `[token][head][dim]` layout, source
    /// positions `[0, token_count)`) into a byte-packed monolithic region at
    /// abs base `dst_pos` with ring `mask`, then unpack from the same abs base
    /// back into a linear f32 buffer. Returns `(keys_out, values_out)`.
    /// Single batch: pack→unpack ordering holds via Metal hazard tracking on
    /// the shared `dst` buffer (the same guarantee the engine's multi-kernel
    /// batches rely on).
    fn pack_then_unpack(
        keys: &[f32],
        values: &[f32],
        token_count: usize,
        dst_pos: usize,
        mask: u32,
        fmt: KvFormat,
        ring_slots: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        use crate::shaders::unpack_encoder_kv as unpack;
        let per_token = NKV * HD;
        let slot_stride = 2u64 * NKV as u64 * fmt.row_bytes(HD as u32);
        let dst_bytes = ring_slots * slot_stride as usize;

        let ctx = MetalContext::new().expect("metal");
        let pack_pipe =
            super::pipeline_fmt_for(&ctx, KernelVariant::PRODUCTION, fmt).expect("pack pipe");
        let unpack_pipe =
            unpack::pipeline_fmt_for(&ctx, KernelVariant::PRODUCTION, fmt).expect("unpack pipe");

        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf_keys = batch.alloc_f32(keys).expect("keys");
        let buf_values = batch.alloc_f32(values).expect("values");
        let buf_dst = batch.alloc_bytes(&vec![0u8; dst_bytes]).expect("dst");

        let (pgrid, ptg) = super::dispatch_shape(token_count, NKV, HD, fmt);
        batch.dispatch_with_grid(&pack_pipe.pipeline, pgrid, ptg, |enc| {
            super::bind_gpu_buffers(
                enc,
                &buf_keys,
                &buf_values,
                &buf_dst,
                token_count as u32,
                dst_pos as u32,
                NKV as u32,
                HD as u32,
                0,
                0,
                mask,
            );
        });

        let buf_ko = batch.alloc_f32_out(token_count * per_token).expect("ko");
        let buf_vo = batch.alloc_f32_out(token_count * per_token).expect("vo");
        let (ugrid, utg) = unpack::dispatch_shape(token_count, NKV, HD);
        batch.dispatch_with_grid(&unpack_pipe.pipeline, ugrid, utg, |enc| {
            unpack::bind_gpu_buffers(
                enc,
                &buf_dst,
                &buf_ko,
                &buf_vo,
                token_count as u32,
                0,
                NKV as u32,
                HD as u32,
                0,
                dst_pos as u32,
                mask,
            );
        });

        let mut ko = vec![0.0f32; token_count * per_token];
        let mut vo = vec![0.0f32; token_count * per_token];
        batch.register_read(buf_ko, &mut ko);
        batch.register_read(buf_vo, &mut vo);
        batch.end().expect("end");
        (ko, vo)
    }

    /// Pack (linear, mask 0) then hydrate the f32 side ring from the same
    /// bytes. Returns the side ring de-interleaved into `(keys, values)` in
    /// linear `[token][head][dim]` order (slot == pos for mask 0, pos0 0).
    fn pack_then_hydrate(
        keys: &[f32],
        values: &[f32],
        token_count: usize,
        fmt: KvFormat,
    ) -> (Vec<f32>, Vec<f32>) {
        let per_token = NKV * HD;
        let slot_stride = 2u64 * NKV as u64 * fmt.row_bytes(HD as u32);
        let dst_bytes = token_count * slot_stride as usize;

        let ctx = MetalContext::new().expect("metal");
        let pack_pipe =
            super::pipeline_fmt_for(&ctx, KernelVariant::PRODUCTION, fmt).expect("pack pipe");
        let hyd_pipe = crate::shaders::kv_f32_side_hydrate::pipeline_for_kv(
            &ctx,
            KernelVariant::PRODUCTION,
            fmt,
        )
        .expect("hydrate pipe");

        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf_keys = batch.alloc_f32(keys).expect("keys");
        let buf_values = batch.alloc_f32(values).expect("values");
        let buf_dst = batch.alloc_bytes(&vec![0u8; dst_bytes]).expect("dst");

        let (pgrid, ptg) = super::dispatch_shape(token_count, NKV, HD, fmt);
        batch.dispatch_with_grid(&pack_pipe.pipeline, pgrid, ptg, |enc| {
            super::bind_gpu_buffers(
                enc,
                &buf_keys,
                &buf_values,
                &buf_dst,
                token_count as u32,
                0,
                NKV as u32,
                HD as u32,
                0,
                0,
                0,
            );
        });

        // side ring: one slot = K plane (per_token) then V plane (per_token).
        let side_len = token_count * per_token * 2;
        let buf_side = batch.alloc_f32_out(side_len).expect("side");
        let grid = MTLSize {
            width: token_count,
            height: NKV,
            depth: HD,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        batch.dispatch_with_grid(&hyd_pipe.pipeline, grid, tg, |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_dst), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_side), 0, 1);
            }
            let shape = [token_count as u32, 0u32, NKV as u32, HD as u32];
            set_bytes(enc, &shape, 2);
            let region = 0u64;
            set_bytes(enc, &region, 3);
            let mask = 0u32;
            set_bytes(enc, &mask, 4);
        });

        let mut side = vec![0.0f32; side_len];
        batch.register_read(buf_side, &mut side);
        batch.end().expect("end");

        let mut ko = vec![0.0f32; token_count * per_token];
        let mut vo = vec![0.0f32; token_count * per_token];
        for pos in 0..token_count {
            let sbase = pos * per_token * 2;
            let obase = pos * per_token;
            ko[obase..obase + per_token].copy_from_slice(&side[sbase..sbase + per_token]);
            vo[obase..obase + per_token]
                .copy_from_slice(&side[sbase + per_token..sbase + 2 * per_token]);
        }
        (ko, vo)
    }

    fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: len");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.to_bits() == e.to_bits(),
                "{label}: bit mismatch at {i}: got {a} want {e}"
            );
        }
    }

    /// f16 pack→unpack is an exact identity on f16-representable inputs — pins
    /// the linear slot indexing with zero rounding-mode ambiguity.
    #[test]
    fn pack_unpack_roundtrip_f16_linear() {
        let tc = 6;
        let n = tc * NKV * HD;
        let keys: Vec<f32> = wave(n, 0.0, 3.0, 0.017)
            .iter()
            .map(|&v| round_half(v))
            .collect();
        let values: Vec<f32> = wave(n, 1.3, 2.0, 0.011)
            .iter()
            .map(|&v| round_half(v))
            .collect();
        let (ko, vo) = pack_then_unpack(&keys, &values, tc, 0, 0, KvFormat::F16, tc);
        assert_bits_eq(&ko, &keys, "f16 linear keys");
        assert_bits_eq(&vo, &values, "f16 linear values");
    }

    /// f16 round-trip across a ring-buffer wrap (mask 7 → ring size 8, abs
    /// positions 6,7,8,9 → slots 6,7,0,1). This is the exact arithmetic that
    /// clobbered live slots in the fast-prefill pad-tail bug; no slot collides
    /// so the identity must still hold bit-for-bit.
    #[test]
    fn pack_unpack_roundtrip_f16_ring_wrap() {
        let tc = 4;
        let n = tc * NKV * HD;
        let keys: Vec<f32> = wave(n, 0.5, 3.0, 0.017)
            .iter()
            .map(|&v| round_half(v))
            .collect();
        let values: Vec<f32> = wave(n, 2.1, 2.0, 0.011)
            .iter()
            .map(|&v| round_half(v))
            .collect();
        // dst_pos 6, mask 7, ring capacity 8.
        let (ko, vo) = pack_then_unpack(&keys, &values, tc, 6, 7, KvFormat::F16, 8);
        assert_bits_eq(&ko, &keys, "f16 wrap keys");
        assert_bits_eq(&vo, &values, "f16 wrap values");
    }

    /// q8 pack→unpack is lossy; assert the round-trip stays within the
    /// group-32 quantization tolerance (a layout/ring bug scrambles which
    /// token lands where → error explodes far past this bound).
    #[test]
    fn pack_unpack_roundtrip_q8_linear() {
        let tc = 6;
        let n = tc * NKV * HD;
        let keys = wave(n, 0.0, 3.0, 0.017);
        let values = wave(n, 1.3, 2.0, 0.011);
        let (ko, vo) = pack_then_unpack(&keys, &values, tc, 0, 0, KvFormat::Q8, tc);
        assert!(
            rel_rms(&keys, &ko) < 0.02,
            "q8 keys rel_rms {}",
            rel_rms(&keys, &ko)
        );
        assert!(
            rel_rms(&values, &vo) < 0.02,
            "q8 values rel_rms {}",
            rel_rms(&values, &vo)
        );
        // Cross-check the numerics against the CPU q8 mirror (tolerant: 1-ULP
        // scale-rounding differences are allowed, gross layout bugs are not).
        let per_token = NKV * HD;
        for t in 0..tc {
            for h in 0..NKV {
                let base = t * per_token + h * HD;
                let mut expect = vec![0.0f32; HD];
                q8_roundtrip(&keys[base..base + HD], &mut expect);
                assert!(
                    rel_rms(&expect, &ko[base..base + HD]) < 0.01,
                    "q8 vs CPU mirror tok {t} head {h}"
                );
            }
        }
    }

    /// `kv_f32_side_hydrate` and `unpack_encoder_kv` read the SAME packed bytes
    /// with the SAME dequant helpers — only the destination addressing differs
    /// (ring-slot side buffer vs linear engine buffer). At mask 0 / pos0 0 the
    /// two coincide, so hydrate must equal unpack BIT-FOR-BIT. GPU-to-GPU, so
    /// this armor is independent of any CPU f16/q8 rounding model.
    #[test]
    fn hydrate_matches_unpack_f16() {
        let tc = 6;
        let n = tc * NKV * HD;
        let keys: Vec<f32> = wave(n, 0.0, 3.0, 0.017)
            .iter()
            .map(|&v| round_half(v))
            .collect();
        let values: Vec<f32> = wave(n, 1.3, 2.0, 0.011)
            .iter()
            .map(|&v| round_half(v))
            .collect();
        let (uk, uv) = pack_then_unpack(&keys, &values, tc, 0, 0, KvFormat::F16, tc);
        let (hk, hv) = pack_then_hydrate(&keys, &values, tc, KvFormat::F16);
        assert_bits_eq(&hk, &uk, "hydrate vs unpack keys (f16)");
        assert_bits_eq(&hv, &uv, "hydrate vs unpack values (f16)");
    }

    #[test]
    fn hydrate_matches_unpack_q8() {
        let tc = 6;
        let n = tc * NKV * HD;
        let keys = wave(n, 0.0, 3.0, 0.017);
        let values = wave(n, 1.3, 2.0, 0.011);
        let (uk, uv) = pack_then_unpack(&keys, &values, tc, 0, 0, KvFormat::Q8, tc);
        let (hk, hv) = pack_then_hydrate(&keys, &values, tc, KvFormat::Q8);
        assert_bits_eq(&hk, &uk, "hydrate vs unpack keys (q8)");
        assert_bits_eq(&hv, &uv, "hydrate vs unpack values (q8)");
    }
}
