// diffgemma_step.metal — monolithic step ABI + dispatch schedule (~130 dispatches).
// Shared device math comes from shaders/include/ via #include (Rust: include_metal! on this file).
// rev2: fixed per audit against kernels/cpu.rs / shaders/include/dequant.metal / sample.rs / generate.rs.
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
// b8  arena_layout  device const ArenaLayout*  scratch plane byte offsets (host-built)
//
// FORMATS (authoritative: src/dgq/block.rs, shaders/include/dequant.metal):
//   q4_block group (20B): [scale bf16:2][min bf16:2][nibbles:16]; w = scale*q + min;
//                         q for col j at byte 4+j/2; even j = low nibble.   (VERIFY-N: nibble parity)
//   q8_row   row (K+2B):  [scale bf16:2][i8 weights:K]; w = scale*q.
//   raw: bf16. All scales bf16 — no half mode.
//
// ===================== REMAINING VERIFY / DRIVER ITEMS =====================
// VERIFY-N   q4 nibble parity (even j = low nibble) — confirm against q4_weight_at tail.
// VERIFY-SC  soft mix gets sqrt(hidden) scale like token embeds; softmax over stored
//            post-softcap logits (temperature-scaled at finish for MLX parity).
// V1(ok'd)   full layers: V aliased from k_proj output, rms_norm_no_scale, no RoPE.
// DRIVER     pipeline table (function constants per shape/layer-type), ICB encode,
//            prefill/extend writer for the NEW kv layout (NOTE-KV), CanvasState init
//            (abi.rs: init_canvas_state, rng = seed+1 per sample.rs Rng::new).
// ===========================================================================

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "dequant.metal"
#include "activations.metal"
#include "arena_layout.metal"

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

// Arena plane offsets: see ArenaLayout (b8), built by Rust build_arena_layout().

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
    uint accept_plateau_threshold;
    float plateau_prefix_mean_max;
    uint eos_token_id;
};
struct CanvasState {
    uint ids[256]; uint prev_argmax[256]; uint new_sample[256];
    float entropy[256]; uint sorted_idx[256]; uint accept[256]; float u_cat[256];
    ulong rng_state;
    uint step;                     // steps completed (0 before first)
    uint stop_flag;
    uint argmax_hist_len;
    uint argmax_hist_base;
    uint argmax_hist[2048];
    uint canvas_stable;
    float mean_entropy; uint accept_plateau; uint prev_accept_sig;
    uint frozen[8];
};
struct RouteScratch {
    half weight[256][8]; uint expert[256][8];
    uint count[128]; uint row_start[129];
    uint num_slots; uint num_active_experts; uint active_expert[128];
    uint token_list[2048]; uint slot_list[2048];
    uint token_slot[256][8];
};

// ============================ monolith-only helpers ============================

// gemm_block (Q4+Q8+NVFP4 via K_QUANT_FORMAT) -> shaders/kernels/gemm_block.metal
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
//   0. [step>0] k_logit_rowstats(logits)                      -> arena.a_rs_sc
//   1. k_sc_softembed(logits, a_rs_sc, first_step=S.step==0)  -> arena.a_soft
//   2. k_rmsnorm(a_soft, sc_pre_norm)                         -> arena.a_tmp
//      k_gemm_q8(sc_gate: N=2112,K=2816) -> a_ffg ; k_gemm_q8(sc_up) -> a_ffu
//      k_glu -> a_ffg ; k_gemm_q8(sc_down: N=2816,K=2112)     -> a_dense (reuse)
//   3. k_embed_gather -> a_hidden ; k_residual(a_hidden, a_dense, scal=0) -> a_hidden
//      k_rmsnorm(a_hidden, w=0 no-scale)                      -> a_hidden
// per layer L (pipelines specialized by IS_FULL_LAYER + GEMM_N/K):
//   4. k_rmsnorm(input_ln) -> a_tmp
//      k_gemm_q4 q (N=4096|8192) -> a_attnq ; k (N=2048|1024) -> a_attnk
//      [sliding only] k_gemm_q4 v -> a_attnv
//   5. qk_rope_kv (grid.y = 16 + 2*nkv) ; attention (grid = 256 x 16, tpg 64)
//      k_gemm_q4 o_proj (N=2816, K=4096|8192) -> a_tmp
//   6. k_rmsnorm(a_tmp, post_attn) -> a_tmp ; k_residual(a_hidden, a_tmp, 0) -> a_stream
//   7. k_rmsnorm(a_stream, pre_ff) -> a_tmp
//      k_gemm_q4 mlp_gate -> a_ffg ; mlp_up -> a_ffu ; k_glu -> a_ffg
//      k_gemm_q4 mlp_down -> a_dense ; k_rmsnorm(a_dense, post_ff_1) -> a_dense
//   8. moe_router (grid = 256, tpg 128)
//      moe_bucket_count(128) ; moe_bucket_fill phase0(2048) ; phase1(1) ; phase2(2048)
//   9. k_rmsnorm(a_stream, pre_ff_2) -> a_moein ; k_memzero(a_moeout)
//      moe_grouped (grid = maxM x 128, tpg 128) — production K_DUMP_STAGE=0 only
//      [a_moeout f32] -> k_rmsnorm_f32(post_ff_2) -> a_moein
//  10. k_residual(a_dense, a_moein, 0) -> a_tmp ; k_rmsnorm(a_tmp, post_ff) -> a_tmp
//      k_residual(a_stream, a_tmp, layer_scalar) -> a_hidden
// finish:
//  11. k_rmsnorm(a_hidden, final_norm) -> a_tmp
//      k_gemm_q8 lm_head (N=262144, K=2816, weights=embed) -> logits ; k_softcap
//  12. sample_rowstats(tpg 256) ; sample_commit(tpg 256) ;
//      sample_apply(grid 256, tpg 256) ; sample_write(1)
// CPU: poll S->stop_flag every step (or every N); on stop read S->ids -> incremental prefill.
