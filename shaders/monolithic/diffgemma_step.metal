// diffgemma_step.metal — full DiffusionGemma denoise step as one encoder (~130 dispatches).
// Device math primitives come from shaders/include/ via Rust concat in step_kernel.rs:
//   common.metal, dequant.metal, activations.metal — no duplicate decode/activation bodies here.
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

// ============================ monolith-only helpers ============================
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

// gemm_q4 -> shaders/kernels/gemm_q4.metal
// gemm_nvfp4 -> shaders/kernels/gemm_nvfp4.metal
// gemm_q8 -> shaders/kernels/gemm_q8.metal
// gemm_q8_rowk -> shaders/kernels/gemm_q8_rowk.metal

// qk_rope_kv -> shaders/kernels/qk_rope_kv.metal
// attention -> shaders/kernels/attention.metal

// moe_router -> shaders/kernels/moe_router.metal
// moe_bucket_count -> shaders/kernels/moe_bucket_count.metal
// moe_bucket_fill -> shaders/kernels/moe_bucket_fill.metal

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

// embed_gather -> shaders/kernels/embed_gather.metal

// logit_rowstats -> shaders/kernels/logit_rowstats.metal
// sc_probs -> shaders/kernels/sc_probs.metal

// sc_softembed -> shaders/kernels/sc_softembed.metal

// sample_rowstats -> shaders/kernels/sample_rowstats.metal
// sample_commit -> shaders/kernels/sample_commit.metal
// sample_apply -> shaders/kernels/sample_apply.metal
// sample_write -> shaders/kernels/sample_write.metal

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
//   5. qk_rope_kv (grid.y = 16 + 2*nkv) ; attention (grid = 256 x 16, tpg 64)
//      k_gemm_q4 o_proj (N=2816, K=4096|8192) -> A_TMP
//   6. k_rmsnorm(A_TMP, post_attn) -> A_TMP ; k_residual(A_HIDDEN, A_TMP, 0) -> A_STREAM
//   7. k_rmsnorm(A_STREAM, pre_ff) -> A_TMP
//      k_gemm_q4 mlp_gate -> A_FFG ; mlp_up -> A_FFU ; k_glu -> A_FFG
//      k_gemm_q4 mlp_down -> A_DENSE ; k_rmsnorm(A_DENSE, post_ff_1) -> A_DENSE
//   8. moe_router (grid = 256, tpg 128)
//      moe_bucket_count(128) ; moe_bucket_fill phase0(2048) ; phase1(1) ; phase2(2048)
//   9. k_rmsnorm(A_STREAM, pre_ff_2) -> A_MOEIN ; k_memzero(A_MOEOUT)
//      k_moe_grouped (grid = maxM x 128, tpg 128)
//      [A_MOEOUT f32] -> k_rmsnorm_f32(post_ff_2) -> A_MOEIN
//  10. k_residual(A_DENSE, A_MOEIN, 0) -> A_TMP ; k_rmsnorm(A_TMP, post_ff) -> A_TMP
//      k_residual(A_STREAM, A_TMP, layer_scalar) -> A_HIDDEN
// finish:
//  11. k_rmsnorm(A_HIDDEN, final_norm) -> A_TMP
//      k_gemm_q8 lm_head (N=262144, K=2816, weights=embed) -> logits ; k_softcap
//  12. sample_rowstats(tpg 256) ; sample_commit(tpg 256) ;
//      sample_apply(grid 256, tpg 256) ; sample_write(1)
// CPU: poll S->stop_flag every step (or every N); on stop read S->ids -> incremental prefill.
