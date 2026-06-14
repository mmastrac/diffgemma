// diffgemma_step.metal — full DiffusionGemma denoise step as one encoder (~130 dispatches).
// rev2: fixed per audit against qgemm.metal / kernels/cpu.rs / sample.rs / generate.rs.
//
// ============================== BUFFER ABI =================================
// b0  blob     device const uchar*        model.dgq.bin, mmap'd, bytesNoCopy
// b1  layout   device const ModelLayout*  built from model.dgq.json (abi.rs)
// b2  params   device const StepParams*
// b3  arena    bound per-dispatch at BYTE offsets below (setBuffer offset, 16B-aligned)
// b4  kvcache  device half*               NEW layout: per-layer region, [pos][K|V][kv_head][dim]
//                                         (requires new prefill writer — see NOTE-KV)
// b5  state    device CanvasState*
// b6  logits   device half* [256][262144] step N lm_head write -> step N+1 SC read.
//                                         Safe: serial dispatches within an encoder + command
//                                         buffer completion order across replays.
// b7  route    device RouteScratch*
//
// FORMATS (authoritative: src/dgq/block.rs, shaders/qgemm.metal):
//   q4_block group (20B): [scale bf16:2][min bf16:2][nibbles:16]; w = scale*q + min;
//                         q for col j at byte 4+j/2; even j = low nibble.   (VERIFY-N: nibble parity)
//   q8_row   row (K+2B):  [scale bf16:2][i8 weights:K]; w = scale*q.
//   raw: bf16. All scales bf16 — no half mode.
//
// ===================== REMAINING VERIFY / DRIVER ITEMS =====================
// VERIFY-N   q4 nibble parity (even j = low nibble) — confirm against q4_weight_at tail.
// VERIFY-SC  soft mix gets sqrt(hidden) scale like token embeds; softmax over stored
//            (post-softcap, t=1) logits via k_logit_rowstats.
// V1(ok'd)   full layers: V aliased from k_proj output, rms_norm_no_scale, no RoPE.
// DRIVER     pipeline table (function constants per shape/layer-type), ICB encode,
//            prefill/extend writer for the NEW kv layout (NOTE-KV), CanvasState init
//            (abi.rs: init_canvas_state, rng = seed+1 per sample.rs Rng::new).
// ===========================================================================

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint  HID = 2816;
constant uint  VOCAB = 262144;
constant uint  CANVAS = 256;
constant uint  NQ_HEADS = 16;
constant uint  MOE_FF = 704;
constant uint  N_EXPERTS = 128;
constant uint  TOP_K = 8;
constant float RMS_EPS = 1e-6f;
constant float SOFTCAP = 30.0f;
constant float EMBED_SCALE = 53.06599664569466f;   // sqrt(2816)
constant float ROUTER_HSCALE = 0.018844940515378f; // 2816^-0.5

constant bool IS_FULL_LAYER [[function_constant(1)]];
constant uint GEMM_N [[function_constant(2)]];
constant uint GEMM_K [[function_constant(3)]];

// ---- arena BYTE offsets (driver binds b3 sub-ranges; all 16B-aligned) ----
// half planes unless noted f32
constant uint A_HIDDEN  = 0;         // [256][2816] h
constant uint A_RESID   = 1441792;   // [256][2816] h
constant uint A_ATTNQ   = 2883584;   // [256][8192] h (max: full-layer Q)
constant uint A_ATTNK   = 7077888;   // [256][2048] h (max: sliding K)
constant uint A_ATTNV   = 8126464;   // [256][2048] h
constant uint A_ATTNO   = 9175040;   // [256][8192] h
constant uint A_FFG     = 13369344;  // [256][2112] h
constant uint A_FFU     = 14450688;  // [256][2112] h
constant uint A_MOEIN   = 15532032;  // [256][2816] h
constant uint A_DENSE   = 16973824;  // [256][2816] h
constant uint A_MOEOUT  = 18415616;  // [256][2816] f32
constant uint A_SOFT    = 21299200;  // [256][2816] h
constant uint A_STREAM  = 22740992;  // [256][2816] h
constant uint A_TMP     = 24182784;  // [256][2816] h
constant uint A_RS_SC   = 25624576;  // [256][2]   f32  (t=1 stats for SC)
constant uint A_RS_SAMP = 25626624;  // [256][2]   f32  (tempered stats for sampler)
// arena total: 25,628,672 B

struct LayerOffsets {
    ulong input_ln, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj, post_attn_ln;
    ulong pre_ff_ln, mlp_gate, mlp_up, mlp_down, post_ff_ln_1;
    ulong router_scale, router_proj, per_expert_scale, pre_ff_ln_2;
    ulong experts_gate_up, experts_down, post_ff_ln_2, post_ff_ln, layer_scalar;
    ulong kv_region;
    uint head_dim; uint n_kv_heads; uint is_full; uint _pad;
};
struct ModelLayout {
    ulong embed, sc_pre_norm, sc_gate, sc_up, sc_down, final_norm;
    LayerOffsets layers[30];
};
struct StepParams {
    uint kv_len; uint max_steps;
    float entropy_bound; float t_min; float t_max; float conf_threshold;
    uint stability_threshold;
    uint min_early_stop_steps;
};
struct CanvasState {
    uint ids[256]; uint prev_argmax[256]; uint new_sample[256];
    float entropy[256]; uint sorted_idx[256]; uint accept[256]; float u_cat[256];
    ulong rng_state;
    uint step;                     // steps completed (0 before first)
    uint stop_flag;
    uint argmax_stable;            // consecutive unchanged steps
    uint argmax_changed;           // per-step scratch flag (atomic)
    float mean_entropy; uint _pad2;
};
struct RouteScratch {
    half weight[256][8]; uint expert[256][8];
    uint count[128]; uint offset[128];
    uint num_slots; uint _pad_route;
    uint token_list[2048]; uint slot_list[2048];
};

// ============================ helpers ============================
inline float bf16_bytes(device const uchar* p) {
    return as_type<float>((uint(p[0]) | (uint(p[1]) << 8)) << 16);
}
inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }
inline float gelu_tanh(float x) {
    float t = clamp(0.7978845608028654f * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
    return 0.5f * x * (1.0f + tanh(t));
}
// device atomic_float fetch_add is unreliable on MPS for moe_out scatter; CAS on uint bits.
inline void atomic_add_f32(device atomic_uint* bits, float val) {
    uint old = atomic_load_explicit(bits, memory_order_relaxed);
    for (;;) {
        float new_f = as_type<float>(old) + val;
        uint new_bits = as_type<uint>(new_f);
        if (atomic_compare_exchange_weak_explicit(
                bits, &old, new_bits, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}
// q4_block: [scale:2][min:2][nibbles:16], w = scale*q + min  (audit item 1)
inline void dequant_q4_group(device const uchar* g, thread float* out32) {
    float s  = bf16_bytes(g);
    float mn = bf16_bytes(g + 2);
    for (uint i = 0; i < 16; ++i) {
        uchar b = g[4 + i];
        out32[2*i]   = s * float(b & 0x0F) + mn;   // VERIFY-N
        out32[2*i+1] = s * float(b >> 4)   + mn;
    }
}
inline ulong q4_row_bytes(uint K) { return ulong(K/32) * 20ul; }
// nvfp4_block: [f32 global_scale:4] + per row [data:ceil(K/2)][scales:ceil(K/16)]
inline float fp16_bits_to_f32(ushort bits) {
    uint sign = (bits >> 15) & 1u;
    uint exp = (bits >> 10) & 0x1fu;
    uint mant = bits & 0x3ffu;
    if (exp == 0u) {
        if (mant == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        return (sign ? -1.0f : 1.0f) * float(mant) * exp2(-24.0f);
    }
    if (exp == 0x1fu) {
        if (mant == 0u) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    uint f32_bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    return as_type<float>(f32_bits);
}
inline float fp8_e4m3_to_f32(uchar b) {
    ushort v = ushort(b & 127u) << 7;
    float converted = fp16_bits_to_f32(v) * 256.0f;
    return (b & 128u) ? -converted : converted;
}
inline float e2m1_to_f32(uint q) {
    float mag = 0.f;
    switch (q & 7u) {
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        case 7: mag = 6.0f; break;
        default: mag = 0.f; break;
    }
    return (q & 8u) ? -mag : mag;
}
inline ulong nvfp4_row_bytes(uint K) {
    return ulong((K + 1u) / 2u + (K + 15u) / 16u);
}
inline ulong nvfp4_matrix_bytes(uint out_dim, uint K) {
    return 4ul + ulong(out_dim) * nvfp4_row_bytes(K);
}
inline void dequant_nvfp4_group(device const uchar* row, uint K, uint g,
                              thread float* out16, float gscale) {
    uint data_len = (K + 1u) / 2u;
    float scale = fp8_e4m3_to_f32(row[data_len + g]) * gscale;
    device const uchar* packed = row + g * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out16[i] = e2m1_to_f32(q) * scale;
    }
}
inline void dequant_nvfp4_tile(device const uchar* row, uint K, uint k0,
                               thread float* out32, float gscale) {
    dequant_nvfp4_group(row, K, k0 / 16u, out32, gscale);
    dequant_nvfp4_group(row, K, k0 / 16u + 1u, out32 + 16, gscale);
}
// q8_row: [scale:2][i8:K]  (audit item 2)
inline ulong q8_row_bytes(uint K) { return ulong(K) + 2ul; }
inline float q8_at(device const uchar* row_base, uint col, float s) {
    return float(*((device const char*)(row_base + 2 + col))) * s;
}
inline ulong lcg_next(ulong s) { return s * 6966169279ul + 1039523323ul; }
inline float lcg_f32(ulong s)  { return float(uint(s >> 32)) * (1.0f/4294967296.0f); }
// CPU: cur_step counts max..1; t = t_min + (t_max-t_min)*(cur/n)   (audit item 7)
inline float temp_at(uint steps_done, constant StepParams& P) {
    float cur = float(P.max_steps - steps_done);
    return P.t_min + (P.t_max - P.t_min) * (cur / float(P.max_steps));
}

// ===================== k_rmsnorm (row; w_off==0 -> no-scale) =====================
kernel void k_rmsnorm(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      device const uchar* blob [[buffer(2)]],
                      constant ulong& w_off [[buffer(3)]],
                      constant uint& dim [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tpg [[threads_per_threadgroup]]) {
    threadgroup float red[8];
    device const half* xr = x + (ulong)row * dim;
    float acc = 0.f;
    for (uint i = lid; i < dim; i += tpg) { float v = float(xr[i]); acc += v*v; }
    acc = simd_sum(acc);
    uint sg = lid / 32, nsg = (tpg + 31) / 32;
    if ((lid & 31) == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float t = 0.f;
        for (uint i = 0; i < nsg; ++i) t += red[i];
        red[0] = rsqrt(t / float(dim) + RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];
    for (uint i = lid; i < dim; i += tpg) {
        float v = float(xr[i]) * inv;
        if (w_off != 0) v *= bf16_bytes(blob + w_off + 2ul*i);
        y[(ulong)row * dim + i] = half(v);
    }
}

// RMSNorm reading f32 activations (MoE scatter output) and writing half.
kernel void k_rmsnorm_f32(device const float* x [[buffer(0)]],
                          device half* y [[buffer(1)]],
                          device const uchar* blob [[buffer(2)]],
                          constant ulong& w_off [[buffer(3)]],
                          constant uint& dim [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint tpg [[threads_per_threadgroup]]) {
    threadgroup float red[8];
    device const float* xr = x + (ulong)row * dim;
    float acc = 0.f;
    for (uint i = lid; i < dim; i += tpg) { float v = xr[i]; acc += v*v; }
    acc = simd_sum(acc);
    uint sg = lid / 32, nsg = (tpg + 31) / 32;
    if ((lid & 31) == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float t = 0.f;
        for (uint i = 0; i < nsg; ++i) t += red[i];
        red[0] = rsqrt(t / float(dim) + RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];
    for (uint i = lid; i < dim; i += tpg) {
        float v = xr[i] * inv;
        if (w_off != 0) v *= bf16_bytes(blob + w_off + 2ul*i);
        y[(ulong)row * dim + i] = half(v);
    }
}

// Tile GEMM kernels below require threadgroup size (128, 1, 1): ltid = lid.x, loop stride 128.
// ===================== k_gemm_q4: y[M,N] = x[M,K] @ Wq4[N,K]^T =====================
kernel void k_gemm_q4(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      device const uchar* blob [[buffer(2)]],
                      constant ulong& w_off [[buffer(3)]],
                      constant uint& M [[buffer(4)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]],
                      uint sgid [[simdgroup_index_in_threadgroup]]) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half tx[32][32];
    threadgroup half tw[32][32];
    uint m0 = tgid.y * 32, n0 = tgid.x * 32;
    uint ltid = lid.x;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    const ulong rowB = q4_row_bytes(K);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = ltid; i < 32*32; i += 128) {
            uint mm = i/32, kk = i%32;
            tx[mm][kk] = (m0+mm < M) ? x[(ulong)(m0+mm)*K + k0+kk] : half(0);
        }
        for (uint r = ltid; r < 32; r += 128) {
            float tmp[32];
            dequant_q4_group(blob + w_off + (ulong)(n0+r)*rowB + (ulong)(k0/32)*20ul, tmp);
            for (uint kk = 0; kk < 32; ++kk) tw[r][kk] = half(tmp[kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a, b0, b1, b2, b3;
            simdgroup_load(a,  &tx[8*sgid][kk], 32);
            simdgroup_load(b0, &tw[0][kk],  32, ulong2(0,0), true);
            simdgroup_load(b1, &tw[8][kk],  32, ulong2(0,0), true);
            simdgroup_load(b2, &tw[16][kk], 32, ulong2(0,0), true);
            simdgroup_load(b3, &tw[24][kk], 32, ulong2(0,0), true);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a, b3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float ty[32][32];
    simdgroup_store(acc0,&ty[8*sgid][0],32); simdgroup_store(acc1,&ty[8*sgid][8],32);
    simdgroup_store(acc2,&ty[8*sgid][16],32); simdgroup_store(acc3,&ty[8*sgid][24],32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < 32*32; i += 128) {
        uint mm = i/32, nn = i%32;
        if (m0+mm < M && n0+nn < N) y[(ulong)(m0+mm)*N + n0+nn] = half(ty[mm][nn]);
    }
}

// ===================== k_gemm_nvfp4: y[M,N] = x[M,K] @ Wnvfp4[N,K]^T =====================
kernel void k_gemm_nvfp4(device const half* x [[buffer(0)]],
                         device half* y [[buffer(1)]],
                         device const uchar* blob [[buffer(2)]],
                         constant ulong& w_off [[buffer(3)]],
                         constant uint& M [[buffer(4)]],
                         uint3 tgid [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]]) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half tx[32][32];
    threadgroup half tw[32][32];
    uint m0 = tgid.y * 32, n0 = tgid.x * 32;
    uint ltid = lid.x;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    float gscale = as_type<float>(*(device const uint*)(blob + w_off));
    const ulong body = w_off + 4ul;
    const ulong rowB = nvfp4_row_bytes(K);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = ltid; i < 32*32; i += 128) {
            uint mm = i/32, kk = i%32;
            tx[mm][kk] = (m0+mm < M) ? x[(ulong)(m0+mm)*K + k0+kk] : half(0);
        }
        for (uint r = ltid; r < 32; r += 128) {
            float tmp[32];
            device const uchar* row = blob + body + (ulong)(n0+r)*rowB;
            dequant_nvfp4_tile(row, K, k0, tmp, gscale);
            for (uint kk = 0; kk < 32; ++kk) tw[r][kk] = half(tmp[kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a, b0, b1, b2, b3;
            simdgroup_load(a,  &tx[8*sgid][kk], 32);
            simdgroup_load(b0, &tw[0][kk],  32, ulong2(0,0), true);
            simdgroup_load(b1, &tw[8][kk],  32, ulong2(0,0), true);
            simdgroup_load(b2, &tw[16][kk], 32, ulong2(0,0), true);
            simdgroup_load(b3, &tw[24][kk], 32, ulong2(0,0), true);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a, b3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float ty[32][32];
    simdgroup_store(acc0,&ty[8*sgid][0],32); simdgroup_store(acc1,&ty[8*sgid][8],32);
    simdgroup_store(acc2,&ty[8*sgid][16],32); simdgroup_store(acc3,&ty[8*sgid][24],32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < 32*32; i += 128) {
        uint mm = i/32, nn = i%32;
        if (m0+mm < M && n0+nn < N) y[(ulong)(m0+mm)*N + n0+nn] = half(ty[mm][nn]);
    }
}

// ===================== k_gemm_q8 (lm_head, SC projections) =====================
kernel void k_gemm_q8(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      device const uchar* blob [[buffer(2)]],
                      constant ulong& w_off [[buffer(3)]],
                      constant uint& M [[buffer(4)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]],
                      uint sgid [[simdgroup_index_in_threadgroup]]) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half tx[32][32];
    threadgroup half tw[32][32];
    uint m0 = tgid.y * 32, n0 = tgid.x * 32;
    uint ltid = lid.x;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    const ulong rowB = q8_row_bytes(K);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = ltid; i < 32*32; i += 128) {
            uint mm = i/32, kk = i%32;
            tx[mm][kk] = (m0+mm < M) ? x[(ulong)(m0+mm)*K + k0+kk] : half(0);
        }
        for (uint i = ltid; i < 32*32; i += 128) {
            uint nn = i/32, kk = i%32;
            device const uchar* rb = blob + w_off + (ulong)(n0+nn)*rowB;
            tw[nn][kk] = half(q8_at(rb, k0+kk, bf16_bytes(rb)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a, b0, b1, b2, b3;
            simdgroup_load(a,  &tx[8*sgid][kk], 32);
            simdgroup_load(b0, &tw[0][kk],  32, ulong2(0,0), true);
            simdgroup_load(b1, &tw[8][kk],  32, ulong2(0,0), true);
            simdgroup_load(b2, &tw[16][kk], 32, ulong2(0,0), true);
            simdgroup_load(b3, &tw[24][kk], 32, ulong2(0,0), true);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a, b3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float ty[32][32];
    simdgroup_store(acc0,&ty[8*sgid][0],32); simdgroup_store(acc1,&ty[8*sgid][8],32);
    simdgroup_store(acc2,&ty[8*sgid][16],32); simdgroup_store(acc3,&ty[8*sgid][24],32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < 32*32; i += 128) {
        uint mm = i/32, nn = i%32;
        if (m0+mm < M && n0+nn < N) y[(ulong)(m0+mm)*N + n0+nn] = half(ty[mm][nn]);
    }
}

// k_gemm_q8_rowk: weight rows indexed by K (vocab); cols N (hidden). For SC softembed.
kernel void k_gemm_q8_rowk(device const half* x [[buffer(0)]],
                           device half* y [[buffer(1)]],
                           device const uchar* blob [[buffer(2)]],
                           constant ulong& w_off [[buffer(3)]],
                           constant uint& M [[buffer(4)]],
                           uint3 tgid [[threadgroup_position_in_grid]],
                           uint3 lid [[thread_position_in_threadgroup]],
                           uint sgid [[simdgroup_index_in_threadgroup]]) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half tx[32][32];
    threadgroup half tw[32][32];
    uint m0 = tgid.y * 32, n0 = tgid.x * 32;
    uint ltid = lid.x;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    const ulong rowB = q8_row_bytes(N);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = ltid; i < 32*32; i += 128) {
            uint mm = i/32, kk = i%32;
            tx[mm][kk] = (m0+mm < M) ? x[(ulong)(m0+mm)*K + k0+kk] : half(0);
        }
        for (uint i = ltid; i < 32*32; i += 128) {
            uint nn = i/32, kk = i%32;
            device const uchar* rb = blob + w_off + (ulong)(k0+kk)*rowB;
            tw[nn][kk] = half(q8_at(rb, n0+nn, bf16_bytes(rb)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a, b0, b1, b2, b3;
            simdgroup_load(a,  &tx[8*sgid][kk], 32);
            simdgroup_load(b0, &tw[0][kk],  32, ulong2(0,0), true);
            simdgroup_load(b1, &tw[8][kk],  32, ulong2(0,0), true);
            simdgroup_load(b2, &tw[16][kk], 32, ulong2(0,0), true);
            simdgroup_load(b3, &tw[24][kk], 32, ulong2(0,0), true);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a, b3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float ty[32][32];
    simdgroup_store(acc0,&ty[8*sgid][   0],32); simdgroup_store(acc1,&ty[8*sgid][8],32);
    simdgroup_store(acc2,&ty[8*sgid][16],32); simdgroup_store(acc3,&ty[8*sgid][24],32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < 32*32; i += 128) {
        uint mm = i/32, nn = i%32;
        if (m0+mm < M && n0+nn < N) y[(ulong)(m0+mm)*N + n0+nn] = half(ty[mm][nn]);
    }
}

// ============ k_qk_rope_kv: per-head QK-norm + split-half RoPE + KV write ============
// GRID: x = CANVAS tokens, y = NQ_HEADS + 2*n_kv_heads head slots (32 sliding / 20 full).
// On full layers (L->v_proj == 0) the V slot reads from the K GEMM output (V1, audit item 6) —
// driver does NOT alias buffers; selection happens here.
// RoPE (audit items 4,5): split-half pairs (d, d+rot/2); inv_freq = theta^(-2d/head_dim)
// — denominator is the FULL head_dim (512 on full layers), rotation spans rot dims only.
kernel void k_qk_rope_kv(device half* q [[buffer(0)]],
                         device half* k [[buffer(1)]],
                         device half* v [[buffer(2)]],
                         device half* kvcache [[buffer(3)]],
                         device const uchar* blob [[buffer(4)]],
                         device const LayerOffsets* L [[buffer(5)]],
                         constant StepParams& P [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint hd = L->head_dim, nkv = L->n_kv_heads;
    const uint tok = gid.x, h = gid.y;
    const uint pos = P.kv_len + tok;
    const bool full = L->is_full != 0;
    const uint rot = full ? hd / 4 : hd;            // partial_rotary_factor 0.25
    const float theta = full ? 1.0e6f : 1.0e4f;

    bool isQ = h < NQ_HEADS;
    bool isK = !isQ && h < NQ_HEADS + nkv;
    uint hh = isQ ? h : (h - NQ_HEADS) % nkv;
    device half* src = isQ ? (q + (ulong)tok*NQ_HEADS*hd + hh*hd)
                     : isK ? (k + (ulong)tok*nkv*hd + hh*hd)
                     : ((L->v_proj != 0 ? v : k) + (ulong)tok*nkv*hd + hh*hd);

    // per-head RMSNorm: Q/K learned scale; V no-scale (V11/V1)
    float ss = 0.f;
    for (uint i = 0; i < hd; ++i) { float t = float(src[i]); ss += t*t; }
    float inv = rsqrt(ss / float(hd) + RMS_EPS);
    ulong noff = isQ ? L->q_norm : isK ? L->k_norm : 0ul;
    // V path must not mutate the shared K buffer on full layers: compute into cache directly.
    float tmp; // per-element rewrite below
    if (isQ || isK) {
        for (uint i = 0; i < hd; ++i) {
            tmp = float(src[i]) * inv * bf16_bytes(blob + noff + 2ul*i);
            src[i] = half(tmp);
        }
        const uint half_rot = rot / 2;
        for (uint d = 0; d < half_rot; ++d) {
            float inv_freq = pow(theta, -2.0f * float(d) / float(hd));   // audit item 5
            float a = float(pos) * inv_freq, c = cos(a), s = sin(a);
            float x0 = float(src[d]), x1 = float(src[d + half_rot]);
            src[d]            = half(x0*c - x1*s);                       // audit item 4
            src[d + half_rot] = half(x0*s + x1*c);
        }
        if (isK) {
            device half* dst = kvcache + L->kv_region/2 + (ulong)pos*nkv*hd*2 + hh*hd;
            for (uint i = 0; i < hd; ++i) dst[i] = src[i];
        }
    } else {
        device half* dst = kvcache + L->kv_region/2 + (ulong)pos*nkv*hd*2 + (ulong)nkv*hd + hh*hd;
        for (uint i = 0; i < hd; ++i) dst[i] = half(float(src[i]) * inv);
    }
}

// ===================== k_attention (canvas queries; all_valid mask) =====================
kernel void k_attention(device const half* q [[buffer(0)]],
                        device const half* kvcache [[buffer(1)]],
                        device half* out [[buffer(2)]],
                        device const LayerOffsets* L [[buffer(3)]],
                        constant StepParams& P [[buffer(4)]],
                        uint3 tgid [[threadgroup_position_in_grid]],
                        uint3 lid [[thread_position_in_threadgroup]],
                        uint3 tpg [[threads_per_threadgroup]]) {       // tpg = 64
    const uint hd = L->head_dim, nkv = L->n_kv_heads;
    const uint tok = tgid.x, qh = tgid.y;
    const uint ltid = lid.x, tpg_w = tpg.x;
    const uint kvh = qh / (NQ_HEADS / nkv);
    const uint T = P.kv_len + CANVAS;
    device const half* qv = q + (ulong)tok*NQ_HEADS*hd + qh*hd;
    device const half* base = kvcache + L->kv_region/2;
    threadgroup float red[8];
    float m = -INFINITY, l = 0.f;
    float acc[8];                                  // hd<=512, tpg=64 -> per<=8
    const uint per = (hd + tpg_w - 1) / tpg_w;
    for (uint i = 0; i < per; ++i) acc[i] = 0.f;
    for (uint t = 0; t < T; ++t) {
        device const half* kk = base + (ulong)t*nkv*hd*2 + kvh*hd;
        float d = 0.f;
        for (uint i = ltid; i < hd; i += tpg_w) d += float(qv[i]) * float(kk[i]);
        d = simd_sum(d);
        uint sg = ltid/32, nsg = (tpg_w+31)/32;
        if ((ltid&31)==0) red[sg] = d;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ltid == 0) { float s = 0.f; for (uint i=0;i<nsg;++i) s += red[i]; red[0] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d = red[0];                                // raw dot product: no 1/sqrt(d)
        float mn = max(m, d), corr = exp(m - mn), p = exp(d - mn);
        l = l*corr + p; m = mn;
        device const half* vv = base + (ulong)t*nkv*hd*2 + (ulong)nkv*hd + kvh*hd;
        for (uint i = 0; i < per; ++i) {
            uint idx = ltid + i*tpg_w;
            if (idx < hd) acc[i] = acc[i]*corr + p*float(vv[idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device half* ov = out + (ulong)tok*NQ_HEADS*hd + qh*hd;
    for (uint i = 0; i < per; ++i) {
        uint idx = ltid + i*tpg_w;
        if (idx < hd) ov[idx] = half(acc[i] / l);
    }
}

// ===================== k_residual (+optional layer_scalar, V7) =====================
kernel void k_residual(device const half* a [[buffer(0)]],
                       device const half* b [[buffer(1)]],
                       device half* y [[buffer(2)]],
                       device const uchar* blob [[buffer(3)]],
                       constant ulong& scal_off [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
    float s = scal_off ? bf16_bytes(blob + scal_off) : 1.0f;
    y[i] = half((float(a[i]) + float(b[i])) * s);
}
// add f32 source variant for moe_out
kernel void k_residual_f32b(device const half* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device half* y [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    y[i] = half(float(a[i]) + b[i]);
}
kernel void k_glu(device const half* gate [[buffer(0)]],
                  device const half* up [[buffer(1)]],
                  device half* y [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    y[i] = half(gelu_tanh(float(gate[i])) * float(up[i]));
}

// ===================== k_router (V10; audit item 16: two-level reduction) =====================
kernel void k_router(device const half* stream [[buffer(0)]],
                     device const uchar* blob [[buffer(1)]],
                     device const LayerOffsets* L [[buffer(2)]],
                     device RouteScratch* R [[buffer(3)]],
                     uint tok [[threadgroup_position_in_grid]],
                     uint e [[thread_position_in_threadgroup]]) {     // tpg = 128
    threadgroup float logits[N_EXPERTS];
    threadgroup float red[4];
    device const half* x = stream + (ulong)tok * HID;
    float ss = 0.f;
    for (uint i = e; i < HID; i += N_EXPERTS) { float t = float(x[i]); ss += t*t; }
    ss = simd_sum(ss);
    if ((e & 31) == 0) red[e/32] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) red[0] = rsqrt((red[0]+red[1]+red[2]+red[3]) / float(HID) + RMS_EPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm_inv = red[0];
    device const uchar* rs = blob + L->router_scale;
    device const uchar* wr = blob + L->router_proj + (ulong)e * HID * 2ul;
    float acc = 0.f;
    for (uint d = 0; d < HID; ++d) {
        float xn = float(x[d]) * norm_inv * bf16_bytes(rs + 2ul*d) * ROUTER_HSCALE;
        acc += xn * bf16_bytes(wr + 2ul*d);
    }
    logits[e] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        float mx = logits[0];
        for (uint i = 1; i < N_EXPERTS; ++i) mx = max(mx, logits[i]);
        float sum = 0.f;
        for (uint i = 0; i < N_EXPERTS; ++i) { logits[i] = exp(logits[i]-mx); sum += logits[i]; }
        float wsum = 0.f; uint pick[TOP_K];
        for (uint kk = 0; kk < TOP_K; ++kk) {
            float best = -1.f; uint bi = 0;
            for (uint i = 0; i < N_EXPERTS; ++i) {
                bool taken = false;
                for (uint p = 0; p < kk; ++p) taken = taken || (pick[p] == i);
                if (!taken && logits[i] > best) { best = logits[i]; bi = i; }  // tie: lower idx
            }
            pick[kk] = bi; wsum += logits[bi];
        }
        device const uchar* pes = blob + L->per_expert_scale;
        for (uint kk = 0; kk < TOP_K; ++kk) {
            R->expert[tok][kk] = pick[kk];
            R->weight[tok][kk] = half((logits[pick[kk]] / wsum) * bf16_bytes(pes + 2ul*pick[kk]));
        }
    }
}

// ===================== bucketing (3 phases) =====================
kernel void k_bucket_count(device RouteScratch* R [[buffer(0)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < N_EXPERTS) R->count[i] = 0;
}
kernel void k_bucket_fill(device RouteScratch* R [[buffer(0)]],
                          constant uint& phase [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (phase == 0) {
        uint tok = i / TOP_K, kk = i % TOP_K;
        atomic_fetch_add_explicit((device atomic_uint*)&R->count[R->expert[tok][kk]],
                                  1u, memory_order_relaxed);
    } else if (phase == 1) {
        if (i == 0) {
            uint s = 0;
            for (uint e = 0; e < N_EXPERTS; ++e) {
                R->offset[e] = s;
                s += R->count[e];
                R->count[e] = 0;
            }
            R->num_slots = s;
        }
    } else {
        uint tok = i / TOP_K, kk = i % TOP_K, e = R->expert[tok][kk];
        uint slot = R->offset[e] + atomic_fetch_add_explicit((device atomic_uint*)&R->count[e],
                                                             1u, memory_order_relaxed);
        R->token_list[slot] = tok; R->slot_list[slot] = kk;
    }
}

// ===================== k_moe_grouped =====================
// grid.x = token-in-group, grid.y = expert. moe_out (f32) zeroed by k_memzero first.
// NOTE: device-atomic f32 scatter — deterministic per (tok,d) only because each routed
// (tok,expert-slot) pair adds once; FP addition order across experts is NOT fixed.
// For bit-exact parity vs CPU use the serialized variant (slot loop) or per-slot
// staging + ordered reduce; acceptable divergence is ~1 ulp f32 — gate in fixtures.
kernel void k_moe_grouped(device const half* moe_in [[buffer(0)]],
                          device float* moe_out [[buffer(1)]],
                          device const uchar* blob [[buffer(2)]],
                          device const LayerOffsets* L [[buffer(3)]],
                          device const RouteScratch* R [[buffer(4)]],
                          uint3 tgid [[threadgroup_position_in_grid]],
                          uint3 lid [[thread_position_in_threadgroup]],
                          uint3 tpg [[threads_per_threadgroup]]) {
    const uint e = tgid.y;
    const uint ltid = lid.x, tpg_w = tpg.x;
    const uint end = (e+1 < N_EXPERTS) ? R->offset[e+1] : R->num_slots;
    const uint n_tok = end - R->offset[e];
    if (tgid.x >= n_tok) return;
    const uint slot = R->offset[e] + tgid.x;
    const uint tok = R->token_list[slot];
    const float w = float(R->weight[tok][R->slot_list[slot]]);
    device const half* x = moe_in + (ulong)tok * HID;
    const ulong gu = L->experts_gate_up + (ulong)e * 1408ul * q4_row_bytes(HID);
    const ulong dn = L->experts_down    + (ulong)e * (ulong)HID * q4_row_bytes(MOE_FF);
    threadgroup float act[MOE_FF];
    for (uint r = ltid; r < MOE_FF; r += tpg_w) {                  // gate||up split at 704
        float g = 0.f, u = 0.f;
        device const uchar* grow = blob + gu + (ulong)r * q4_row_bytes(HID);
        device const uchar* urow = blob + gu + (ulong)(r + MOE_FF) * q4_row_bytes(HID);
        for (uint k0 = 0; k0 < HID; k0 += 32) {
            float wg[32], wu[32];
            dequant_q4_group(grow + (k0/32)*20ul, wg);
            dequant_q4_group(urow + (k0/32)*20ul, wu);
            for (uint i = 0; i < 32; ++i) {
                float xv = float(x[k0+i]);
                g += wg[i]*xv; u += wu[i]*xv;
            }
        }
        act[r] = gelu_tanh(g) * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = ltid; d < HID; d += tpg_w) {
        float o = 0.f;
        device const uchar* drow = blob + dn + (ulong)d * q4_row_bytes(MOE_FF);
        for (uint k0 = 0; k0 < MOE_FF; k0 += 32) {
            float wd[32];
            dequant_q4_group(drow + (k0/32)*20ul, wd);
            for (uint i = 0; i < 32; ++i) o += wd[i]*act[k0+i];
        }
        atomic_add_f32((device atomic_uint*)&moe_out[(ulong)tok*HID + d], w*o);
    }
}

// Debug variant: dump threadgroup act[704] after barrier (scratch[0..704]) and again at
// down-loop entry from ltid==0,d==0 (scratch[704..1408]) for barrier/visibility bisection.
kernel void k_moe_grouped_act_probe(device const half* moe_in [[buffer(0)]],
                                    device float* moe_out [[buffer(1)]],
                                    device const uchar* blob [[buffer(2)]],
                                    device const LayerOffsets* L [[buffer(3)]],
                                    device const RouteScratch* R [[buffer(4)]],
                                    device float* act_scratch [[buffer(5)]],
                                    uint3 tgid [[threadgroup_position_in_grid]],
                                    uint3 lid [[thread_position_in_threadgroup]],
                                    uint3 tpg [[threads_per_threadgroup]]) {
    const uint e = tgid.y;
    const uint ltid = lid.x, tpg_w = tpg.x;
    const uint end = (e+1 < N_EXPERTS) ? R->offset[e+1] : R->num_slots;
    const uint n_tok = end - R->offset[e];
    if (tgid.x >= n_tok) return;
    const uint slot = R->offset[e] + tgid.x;
    const uint tok = R->token_list[slot];
    const float w = float(R->weight[tok][R->slot_list[slot]]);
    device const half* x = moe_in + (ulong)tok * HID;
    if (ltid == 0) {
        const uint meta = MOE_FF * 2u;
        act_scratch[meta + 0] = float(tok);
        act_scratch[meta + 1] = float(slot);
        act_scratch[meta + 2] = float(e);
        act_scratch[meta + 3] = w;
        for (uint i = 0; i < 8u; ++i) act_scratch[meta + 4u + i] = float(x[i]);
        for (uint i = 0; i < 8u; ++i) act_scratch[meta + 12u + i] = float(moe_in[i]);
    }
    const ulong gu = L->experts_gate_up + (ulong)e * 1408ul * q4_row_bytes(HID);
    const ulong dn = L->experts_down    + (ulong)e * (ulong)HID * q4_row_bytes(MOE_FF);
    threadgroup float act[MOE_FF];
    for (uint r = ltid; r < MOE_FF; r += tpg_w) {
        float g = 0.f, u = 0.f;
        device const uchar* grow = blob + gu + (ulong)r * q4_row_bytes(HID);
        device const uchar* urow = blob + gu + (ulong)(r + MOE_FF) * q4_row_bytes(HID);
        for (uint k0 = 0; k0 < HID; k0 += 32) {
            float wg[32], wu[32];
            dequant_q4_group(grow + (k0/32)*20ul, wg);
            dequant_q4_group(urow + (k0/32)*20ul, wu);
            for (uint i = 0; i < 32; ++i) {
                float xv = float(x[k0+i]);
                g += wg[i]*xv; u += wu[i]*xv;
            }
        }
        act[r] = gelu_tanh(g) * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ltid == 0) {
        for (uint r = 0; r < MOE_FF; ++r) act_scratch[r] = act[r];
    }
    for (uint d = ltid; d < HID; d += tpg_w) {
        if (d == 0 && ltid == 0) {
            for (uint r = 0; r < MOE_FF; ++r) act_scratch[MOE_FF + r] = act[r];
        }
        float o = 0.f;
        device const uchar* drow = blob + dn + (ulong)d * q4_row_bytes(MOE_FF);
        for (uint k0 = 0; k0 < MOE_FF; k0 += 32) {
            float wd[32];
            dequant_q4_group(drow + (k0/32)*20ul, wd);
            for (uint i = 0; i < 32; ++i) o += wd[i]*act[k0+i];
        }
        if (ltid == 0 && d < 8u) {
            const uint meta = MOE_FF * 2u;
            act_scratch[meta + 20u + d] = w * o;
        }
        // Probe uses direct store (single writer per (tok,d)); production uses atomic_add_f32.
        moe_out[(ulong)tok * HID + d] = w * o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ltid == 0) {
        const uint meta = MOE_FF * 2u;
        for (uint i = 0; i < 8u; ++i) {
            act_scratch[meta + 28u + i] = moe_out[(ulong)tok * HID + i];
        }
    }
}

// Debug: K-order decode of one 32-wide Q4 group — path A = dequant_q4_group (k_moe_grouped),
// path B = col-indexed q4_weight_at-style decode (f32_q4_linear / qgemm).
inline float q4_at_col(device const uchar* row_base, uint col, uint row_stride) {
    uint g = col / 32u;
    uint j = col % 32u;
    device const uchar* blk = row_base + ulong(g) * 20ul;
    float delta = bf16_bytes(blk);
    float mn = bf16_bytes(blk + 2);
    uchar byte = blk[4u + j / 2u];
    float q = (j & 1u) ? float(byte >> 4) : float(byte & 0x0fu);
    return delta * q + mn;
}

kernel void k_q4_group_k_order_probe(device const uchar* row_base [[buffer(0)]],
                                     constant uint& k0 [[buffer(1)]],
                                     constant uint& in_dim [[buffer(2)]],
                                     device float* out [[buffer(3)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i > 0) return;
    device const uchar* grp = row_base + ulong(k0 / 32u) * 20ul;
    thread float via_dequant[32];
    dequant_q4_group(grp, via_dequant);
    for (uint m = 0; m < 32; ++m) out[m] = via_dequant[m];
    for (uint m = 0; m < 32; ++m) out[32 + m] = q4_at_col(row_base, k0 + m, q4_row_bytes(in_dim));
}

kernel void k_moe_grouped_nvfp4(device const half* moe_in [[buffer(0)]],
                                device float* moe_out [[buffer(1)]],
                                device const uchar* blob [[buffer(2)]],
                                device const LayerOffsets* L [[buffer(3)]],
                                device const RouteScratch* R [[buffer(4)]],
                                uint3 tgid [[threadgroup_position_in_grid]],
                                uint3 lid [[thread_position_in_threadgroup]],
                                uint3 tpg [[threads_per_threadgroup]]) {
    const uint e = tgid.y;
    const uint ltid = lid.x, tpg_w = tpg.x;
    const uint end = (e+1 < N_EXPERTS) ? R->offset[e+1] : R->num_slots;
    const uint n_tok = end - R->offset[e];
    if (tgid.x >= n_tok) return;
    const uint slot = R->offset[e] + tgid.x;
    const uint tok = R->token_list[slot];
    const float w = float(R->weight[tok][R->slot_list[slot]]);
    device const half* x = moe_in + (ulong)tok * HID;
    const ulong gu = L->experts_gate_up + (ulong)e * nvfp4_matrix_bytes(1408u, HID);
    const ulong dn = L->experts_down    + (ulong)e * nvfp4_matrix_bytes(HID, MOE_FF);
    float gu_scale = as_type<float>(*(device const uint*)(blob + gu));
    float dn_scale = as_type<float>(*(device const uint*)(blob + dn));
    const ulong gu_body = gu + 4ul;
    const ulong dn_body = dn + 4ul;
    const ulong hid_row = nvfp4_row_bytes(HID);
    const ulong ff_row = nvfp4_row_bytes(MOE_FF);
    threadgroup float act[MOE_FF];
    for (uint r = ltid; r < MOE_FF; r += tpg_w) {
        float g = 0.f, u = 0.f;
        device const uchar* grow = blob + gu_body + (ulong)r * hid_row;
        device const uchar* urow = blob + gu_body + (ulong)(r + MOE_FF) * hid_row;
        for (uint k0 = 0; k0 < HID; k0 += 32) {
            float wg[32], wu[32];
            dequant_nvfp4_tile(grow, HID, k0, wg, gu_scale);
            dequant_nvfp4_tile(urow, HID, k0, wu, gu_scale);
            for (uint i = 0; i < 32; ++i) {
                float xv = float(x[k0+i]);
                g += wg[i]*xv; u += wu[i]*xv;
            }
        }
        act[r] = gelu_tanh(g) * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = ltid; d < HID; d += tpg_w) {
        float o = 0.f;
        device const uchar* drow = blob + dn_body + (ulong)d * ff_row;
        for (uint k0 = 0; k0 < MOE_FF; k0 += 32) {
            float wd[32];
            dequant_nvfp4_tile(drow, MOE_FF, k0, wd, dn_scale);
            for (uint i = 0; i < 32; ++i) o += wd[i]*act[k0+i];
        }
        atomic_add_f32((device atomic_uint*)&moe_out[(ulong)tok*HID + d], w*o);
    }
}

// ============ k_embed_gather (q8: [scale:2][i8:K], audit item 2) ============
kernel void k_embed_gather(device const uchar* blob [[buffer(0)]],
                           device const ModelLayout* ML [[buffer(1)]],
                           device const CanvasState* S [[buffer(2)]],
                           device half* out [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {  // x=dim, y=token
    uint tok = gid.y, d = gid.x;
    device const uchar* row = blob + ML->embed + (ulong)S->ids[tok] * q8_row_bytes(HID);
    out[(ulong)tok*HID + d] = half(q8_at(row, d, bf16_bytes(row)) * EMBED_SCALE);
}

// ============ k_logit_rowstats (audit item: was missing) ============
// max + sumexp over STORED logits (post-softcap, t=1) -> A_RS_SC, for self-conditioning.
kernel void k_logit_rowstats(device const half* logits [[buffer(0)]],
                             device float* rowstat [[buffer(1)]],     // [256][2]
                             uint row [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]]) {
    threadgroup float r_mx[8]; threadgroup float r_sum[8];
    device const half* lr = logits + (ulong)row * VOCAB;
    float mx = -INFINITY;
    for (uint v = lid; v < VOCAB; v += tpg) mx = max(mx, float(lr[v]));
    mx = simd_max(mx);
    uint sg = lid/32, nsg = (tpg+31)/32;
    if ((lid&31)==0) r_mx[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) { for (uint i=1;i<nsg;++i) r_mx[0] = max(r_mx[0], r_mx[i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = r_mx[0];
    float sum = 0.f;
    for (uint v = lid; v < VOCAB; v += tpg) sum += exp(float(lr[v]) - mx);
    sum = simd_sum(sum);
    if ((lid&31)==0) r_sum[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float s = 0.f; for (uint i=0;i<nsg;++i) s += r_sum[i];
        rowstat[row*2] = mx; rowstat[row*2+1] = s;
    }
}

// ============ k_sc_probs: materialize softmax rows for SC GEMM fast path (M3.2) ============
kernel void k_sc_probs(device const half* logits [[buffer(0)]],
                       device const float* rowstat [[buffer(1)]],
                       device half* probs [[buffer(2)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]]) {
    float mx = rowstat[row*2], sum = rowstat[row*2+1];
    device const half* lr = logits + (ulong)row * VOCAB;
    for (uint v = lid; v < VOCAB; v += tpg) {
        probs[(ulong)row * VOCAB + v] = half(exp(float(lr[v]) - mx) / sum);
    }
}

kernel void k_half_scale(device half* y [[buffer(0)]],
                         constant uint& n [[buffer(1)]],
                         constant float& scale [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    y[gid] = half(float(y[gid]) * scale);
}

// ============ k_sc_softembed: soft[m,d] = (softmax(prev logits) @ embed)[m,d] * sqrt(H) ============
// Uses A_RS_SC (t=1 stats). first_step -> zeros (SC MLP still runs; VERIFY-SC).
kernel void k_sc_softembed(device const half* logits [[buffer(0)]],
                           device const float* rowstat [[buffer(1)]],
                           device const uchar* blob [[buffer(2)]],
                           device const ModelLayout* ML [[buffer(3)]],
                           device half* soft [[buffer(4)]],
                           constant uint& first_step [[buffer(5)]],
                           uint3 tgid [[threadgroup_position_in_grid]],  // x: dim/64, y: token
                           uint3 lid [[thread_position_in_threadgroup]]) { // 64
    uint tok = tgid.y, d = tgid.x*64 + lid.x;
    if (first_step) { soft[(ulong)tok*HID + d] = half(0); return; }
    float mx = rowstat[tok*2], sum = rowstat[tok*2+1];
    device const half* lr = logits + (ulong)tok * VOCAB;
    float acc = 0.f;
    for (uint v = 0; v < VOCAB; ++v) {
        float p = exp(float(lr[v]) - mx) / sum;
        device const uchar* row = blob + ML->embed + (ulong)v * q8_row_bytes(HID);
        acc += p * q8_at(row, d, bf16_bytes(row));
    }
    soft[(ulong)tok*HID + d] = half(acc * EMBED_SCALE);
    // PERF: O(vocab*hid) restream per step; replace with materialized-probs tiled q8 GEMM
    // (reuse k_gemm_q8 with probs as x) once parity passes.
}

kernel void k_softcap(device half* logits [[buffer(0)]],
                      constant uint& base [[buffer(1)]],
                      constant uint& len [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= len) return;
    uint i = base + gid;
    float v = float(logits[i]);
    // Metal fast-math tanh overflows for large |x|; clamp preserves softcap saturation.
    float x = clamp(v / SOFTCAP, -20.0f, 20.0f);
    logits[i] = half(tanh(x) * SOFTCAP);
}

// ======================= sampler =======================
// pass 1: tempered row stats -> A_RS_SAMP; entropy (nats); argmax + changed flag.
kernel void k_sample_rowstats(device const half* logits [[buffer(0)]],
                              device float* rowstat [[buffer(1)]],    // A_RS_SAMP
                              device CanvasState* S [[buffer(2)]],
                              constant StepParams& P [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint tpg [[threads_per_threadgroup]]) {
    float t = temp_at(S->step, P);                  // steps completed so far (audit item 7)
    device const half* lr = logits + (ulong)row * VOCAB;
    threadgroup float r_mx[8]; threadgroup float r_sum[8]; threadgroup float r_ent[8];
    threadgroup uint r_am[8]; threadgroup float r_amv[8];
    float mx = -INFINITY; uint am = 0; float amv = -INFINITY;
    for (uint v = lid; v < VOCAB; v += tpg) {
        float x = float(lr[v]) / t;
        if (x > amv) { amv = x; am = v; }           // first hit -> lower idx on exact tie
        mx = max(mx, x);
    }
    mx = simd_max(mx);
    for (uint o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_down(amv, o); uint oi = simd_shuffle_down(am, o);
        if (ov > amv || (ov == amv && oi < am)) { amv = ov; am = oi; }
    }
    uint sg = lid/32, nsg = (tpg+31)/32;
    if ((lid&31)==0) { r_mx[sg]=mx; r_am[sg]=am; r_amv[sg]=amv; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        for (uint i = 1; i < nsg; ++i) {
            r_mx[0] = max(r_mx[0], r_mx[i]);
            if (r_amv[i] > r_amv[0] || (r_amv[i] == r_amv[0] && r_am[i] < r_am[0]))
                { r_amv[0] = r_amv[i]; r_am[0] = r_am[i]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = r_mx[0];
    float sum = 0.f, ent = 0.f;
    for (uint v = lid; v < VOCAB; v += tpg) {
        float x = float(lr[v]) / t;
        float e = exp(x - mx);
        sum += e; ent += e * (x - mx);
    }
    sum = simd_sum(sum); ent = simd_sum(ent);
    if ((lid&31)==0) { r_sum[sg]=sum; r_ent[sg]=ent; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float ts = 0.f, te = 0.f;
        for (uint i = 0; i < nsg; ++i) { ts += r_sum[i]; te += r_ent[i]; }
        S->entropy[row] = log(ts) - te/ts;          // H in nats
        rowstat[row*2] = mx; rowstat[row*2+1] = ts;
        uint prev = S->prev_argmax[row];
        S->prev_argmax[row] = r_am[0];
        if (prev != r_am[0])
            atomic_store_explicit((device atomic_uint*)&S->argmax_changed, 1u,
                                  memory_order_relaxed);
    }
}

// pass 2 (1 threadgroup): LCG draws (position order), entropy sort,
// CPU-exact accept rule (audit item 8), stability + early stop, step++.
kernel void k_sample_commit(device CanvasState* S [[buffer(0)]],
                            constant StepParams& P [[buffer(1)]],
                            uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float ent[CANVAS];
    if (lid < CANVAS) { ent[lid] = S->entropy[lid]; S->sorted_idx[lid] = lid; S->accept[lid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        ulong st = S->rng_state;
        for (uint i = 0; i < CANVAS; ++i) { st = lcg_next(st); S->u_cat[i] = lcg_f32(st); }
        S->rng_state = st;
        for (uint i = 1; i < CANVAS; ++i) {         // insertion sort, ascending entropy
            uint id = S->sorted_idx[i]; float e = ent[id]; int j = int(i) - 1;
            while (j >= 0 && ent[S->sorted_idx[j]] > e) { S->sorted_idx[j+1] = S->sorted_idx[j]; --j; }
            S->sorted_idx[j+1] = id;
        }
        // HF/MLX: on the final denoise step (cur_step==1) record stats but do not accept.
        bool final_step = (S->step + 1 >= P.max_steps);
        float prefix = 0.f;
        if (!final_step) {
            for (uint i = 0; i < CANVAS; ++i) {
                uint id = S->sorted_idx[i];
                if (prefix <= P.entropy_bound) { S->accept[id] = 1; prefix += ent[id]; }
                else break;
            }
        }
        float mean = 0.f;
        for (uint i = 0; i < CANVAS; ++i) mean += ent[i];
        S->mean_entropy = mean / float(CANVAS);
        uint changed = atomic_load_explicit((device atomic_uint*)&S->argmax_changed,
                                            memory_order_relaxed);
        S->argmax_stable = changed ? 0u : (S->argmax_stable + 1u);
        atomic_store_explicit((device atomic_uint*)&S->argmax_changed, 0u, memory_order_relaxed);
        S->step += 1;
        bool degenerate = true;
        uint real_count = 0u;
        for (uint i = 0; i < CANVAS; ++i) {
            uint t = S->prev_argmax[i];
            if (t != 0u && t != 262143u) { degenerate = false; real_count++; }
        }
        bool confident_stable = S->mean_entropy < P.conf_threshold
            && S->argmax_stable >= P.stability_threshold;
        bool allowed = !degenerate
            && (S->step >= P.min_early_stop_steps || real_count >= 8u);
        if (confident_stable && allowed)
            S->stop_flag = 1;
        if (S->step >= P.max_steps) S->stop_flag = 1;
    }
}

// pass 3: categorical inverse-CDF per row (tempered; tpg MUST be 256: per = VOCAB/256 = 1024)
kernel void k_sample_apply(device const half* logits [[buffer(0)]],
                           device const float* rowstat [[buffer(1)]],   // A_RS_SAMP
                           device CanvasState* S [[buffer(2)]],
                           constant StepParams& P [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]]) {
    float t = temp_at(S->step - 1, P);              // commit already incremented step
    float mx = rowstat[row*2], Z = rowstat[row*2+1];
    float target = S->u_cat[row] * Z;
    device const half* lr = logits + (ulong)row * VOCAB;
    threadgroup float chunk[256];
    uint per = VOCAB / tpg;
    float local = 0.f;
    for (uint v = lid*per; v < (lid+1)*per; ++v) local += exp(float(lr[v])/t - mx);
    chunk[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float cum = 0.f; uint pick = VOCAB - 1;
        for (uint c = 0; c < tpg; ++c) {
            if (cum + chunk[c] >= target) {
                for (uint v = c*per; v < (c+1)*per; ++v) {
                    cum += exp(float(lr[v])/t - mx);
                    if (cum >= target) { pick = v; break; }
                }
                break;
            }
            cum += chunk[c];
        }
        S->new_sample[row] = pick;
    }
}

// pass 4 (1 thread): accepted -> new sample; rejected -> fresh uniform id.
// Renoise iterates positions 0..255 in order — matches renoise_canvas (audit item 11).
kernel void k_sample_write(device CanvasState* S [[buffer(0)]],
                           uint lid [[thread_position_in_threadgroup]]) {
    if (lid == 0) {
        ulong st = S->rng_state;
        for (uint i = 0; i < CANVAS; ++i) {
            if (S->accept[i]) S->ids[i] = S->new_sample[i];
            else { st = lcg_next(st); S->ids[i] = uint(st >> 32) % VOCAB; }
        }
        S->rng_state = st;
    }
}

// ======================= DISPATCH SCHEDULE (encode once, ICB-replay) =======================
// NOTE-KV: kvcache layout here is NEW ([pos][K|V][kvh][dim] per layer-region, half).
// Existing GpuKvCache (f32, split K/V) is incompatible — prefill/extend writer must target
// this layout before integration. Encoder prefill kernels are intentionally out of scope.
//
// step start:
//   0. [step>0] k_logit_rowstats(logits)                      -> A_RS_SC
//   1. k_sc_softembed(logits, A_RS_SC, first_step=S.step==0)  -> A_SOFT
//   2. k_rmsnorm(A_SOFT, sc_pre_norm)                         -> A_TMP
//      k_gemm_q8(sc_gate: N=2112,K=2816) -> A_FFG ; k_gemm_q8(sc_up) -> A_FFU
//      k_glu -> A_FFG ; k_gemm_q8(sc_down: N=2816,K=2112)     -> A_DENSE (reuse)
//   3. k_embed_gather -> A_HIDDEN ; k_residual(A_HIDDEN, A_DENSE, scal=0) -> A_HIDDEN
//      k_rmsnorm(A_HIDDEN, w=0 no-scale)                      -> A_HIDDEN
// per layer L (pipelines specialized by IS_FULL_LAYER + GEMM_N/K):
//   4. k_rmsnorm(input_ln) -> A_TMP
//      k_gemm_q4 q (N=4096|8192) -> A_ATTNQ ; k (N=2048|1024) -> A_ATTNK
//      [sliding only] k_gemm_q4 v -> A_ATTNV
//   5. k_qk_rope_kv (grid.y = 16 + 2*nkv) ; k_attention (grid = 256 x 16, tpg 64)
//      k_gemm_q4 o_proj (N=2816, K=4096|8192) -> A_TMP
//   6. k_rmsnorm(A_TMP, post_attn) -> A_TMP ; k_residual(A_HIDDEN, A_TMP, 0) -> A_STREAM
//   7. k_rmsnorm(A_STREAM, pre_ff) -> A_TMP
//      k_gemm_q4 mlp_gate -> A_FFG ; mlp_up -> A_FFU ; k_glu -> A_FFG
//      k_gemm_q4 mlp_down -> A_DENSE ; k_rmsnorm(A_DENSE, post_ff_1) -> A_DENSE
//   8. k_router (grid = 256, tpg 128)
//      k_bucket_count(128) ; k_bucket_fill phase0(2048) ; phase1(1) ; phase2(2048)
//   9. k_rmsnorm(A_STREAM, pre_ff_2) -> A_MOEIN ; k_memzero(A_MOEOUT)
//      k_moe_grouped (grid = maxM x 128, tpg 128)
//      [A_MOEOUT f32] -> k_rmsnorm_f32(post_ff_2) -> A_MOEIN
//  10. k_residual(A_DENSE, A_MOEIN, 0) -> A_TMP ; k_rmsnorm(A_TMP, post_ff) -> A_TMP
//      k_residual(A_STREAM, A_TMP, layer_scalar) -> A_HIDDEN
// finish:
//  11. k_rmsnorm(A_HIDDEN, final_norm) -> A_TMP
//      k_gemm_q8 lm_head (N=262144, K=2816, weights=embed) -> logits ; k_softcap
//  12. k_sample_rowstats(tpg 256) ; k_sample_commit(tpg 256) ;
//      k_sample_apply(grid 256, tpg 256) ; k_sample_write(1)
// CPU: poll S->stop_flag every step (or every N); on stop read S->ids -> incremental prefill.
