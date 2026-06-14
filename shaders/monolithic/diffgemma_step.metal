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
    uint count[128]; uint row_start[129];
    uint num_slots; uint _pad_route;
    uint token_list[2048]; uint slot_list[2048];
};

// ============================ monolith-only helpers ============================

// gemm_block (Q4+NVFP4 via K_QUANT_FORMAT) -> shaders/kernels/gemm_block.metal
// gemm_q8 -> shaders/kernels/gemm_q8.metal
// gemm_q8_rowk -> shaders/kernels/gemm_q8_rowk.metal

// qk_rope_kv -> shaders/kernels/qk_rope_kv.metal
// attention -> shaders/kernels/attention.metal

// moe_router -> shaders/kernels/moe_router.metal
// moe_bucket_count -> shaders/kernels/moe_bucket_count.metal
// moe_bucket_fill -> shaders/kernels/moe_bucket_fill.metal

// moe_grouped (Q4+NVFP4 via K_QUANT_FORMAT) -> shaders/kernels/moe_grouped.metal
// q4_group_k_order -> shaders/kernels/q4_group_k_order.metal (K_DUMP_STAGE)

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
//      moe_grouped (grid = maxM x 128, tpg 128) — production K_DUMP_STAGE=0 only
//      [A_MOEOUT f32] -> k_rmsnorm_f32(post_ff_2) -> A_MOEIN
//  10. k_residual(A_DENSE, A_MOEIN, 0) -> A_TMP ; k_rmsnorm(A_TMP, post_ff) -> A_TMP
//      k_residual(A_STREAM, A_TMP, layer_scalar) -> A_HIDDEN
// finish:
//  11. k_rmsnorm(A_HIDDEN, final_norm) -> A_TMP
//      k_gemm_q8 lm_head (N=262144, K=2816, weights=embed) -> logits ; k_softcap
//  12. sample_rowstats(tpg 256) ; sample_commit(tpg 256) ;
//      sample_apply(grid 256, tpg 256) ; sample_write(1)
// CPU: poll S->stop_flag every step (or every N); on stop read S->ids -> incremental prefill.
