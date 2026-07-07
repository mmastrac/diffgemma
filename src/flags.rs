//! Central registry of every `DGQ_*` environment flag.
//!
//! All runtime env-flag accessors live in this module — grep here first.
//! Conventions:
//! - Production toggles default ON and are opt-OUT (`=0` disables); each one
//!   names a shipped, evidence-backed default. They exist for A/B triage.
//! - Sampler-semantics toggles carry their sign-off history in the doc.
//! - Debug/probe flags default OFF and are opt-IN (`=1` or a path enables).
//!
//! One deliberate exception lives outside this module:
//! `DGQ_METAL_PIPELINE_CACHE` (pipeline_cache.rs) — dual bool/path semantics
//! coupled to that module's directory layout.

/// `=0`/`false` disables; anything else (or unset) leaves it ON.
fn on_unless_zero(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

/// `=1`/`true` enables; anything else (or unset) leaves it OFF.
fn on_if_one(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Sampler semantics (quality-affecting; changes need multi-seed gate + census
// + user sign-off — see the quality-ratchet rule)
// ---------------------------------------------------------------------------

/// Hard-freeze of accepted canvas rows (`DGQ_FREEZE=1` re-enables the legacy
/// behavior). Default OFF since 2026-07-05 (user sign-off): the freeze was
/// PROVEN to be the flat-row wart driver (census 4/10 warty -> 0/10). OFF =
/// MLX/HF reference semantics (matches the CPU sampler in `sample.rs`): the
/// accept set is re-decided from fresh entropies every step, accepted rows
/// take that step's fresh denoiser token, dropped rows renoise, and the final
/// commit is the true full-canvas last-step argmax. ON additionally feeds the
/// partial-lm_head row skip (dormant at default).
pub fn freeze_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_if_one("DGQ_FREEZE"))
}

/// Commit the row argmax instead of the tempered categorical sample
/// (`DGQ_DENOISER_ARGMAX=0` restores HF categorical). Default ON since
/// 2026-07-05 (user sign-off): matches MLX's default user temperature=0
/// denoiser — the linear schedule temperature only shapes entropy and the SC
/// soft-embed, never the committed token. With no-freeze this is the
/// MLX-exact config (gate 16,16,11; census 0/10 warty).
pub fn denoiser_argmax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_unless_zero("DGQ_DENOISER_ARGMAX"))
}

/// Entropy-only early stop (`DGQ_EARLY_STOP_MEAN_ENT=<nats>`; 0 disables).
/// A denoise block stops as soon as the full-canvas mean entropy falls below
/// this (after `min_early_stop_steps`), WITHOUT waiting for full argmax
/// stability. Our forward converges the answer text long before the argmax
/// fully locks (the last ~10 steps only micro-flip near-tie positions and
/// shrink entropy from ~0.1 to ~0.0001), so 0.05 cuts ~25-35% of steps on
/// long replies. DEFAULT ON at 0.05 (user sign-off 2026-07-06; evidence:
/// probe answers byte-identical, multi-seed gate + wart census neutral).
pub fn early_stop_mean_ent() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DGQ_EARLY_STOP_MEAN_ENT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|&x| x >= 0.0)
            .unwrap_or(0.05)
    })
}

/// Empty/degenerate-reply retry (E6). On the FIRST denoise block only, if the
/// committed canvas is degenerate (position 0 is a stop/eos/control token → the
/// reply would trim to empty or lead with ceremony), re-roll the initial canvas
/// from the advancing seed stream and re-run the block, up to N times. The empty
/// reply is an intrinsic (prompt, initial-canvas)→eos attractor shared with MLX
/// (E6 2026-07-07: forward is MLX-faithful, token-suppression recovers ceremony
/// junk, template is load-bearing) — a fresh canvas draw escapes it for ~most
/// seeds. Retries stay deterministic per `--seed` (canvas derives from the same
/// rng chain). DEFAULT 3 (user sign-off 2026-07-07; seed-123 gate answers
/// 13→17, calibration seeds 7/42 unchanged 17/17 = pure no-op on good replies,
/// suite 714/0). `DGQ_EMPTY_REPLY_RETRY=0` disables.
pub fn empty_reply_retry() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DGQ_EMPTY_REPLY_RETRY")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3)
    })
}

// ---------------------------------------------------------------------------
// Production perf toggles (all default ON, opt-out for A/B triage)
// ---------------------------------------------------------------------------

/// Tunable GEMM (fragment-level 64x64 kernels for Raw/q8/stacked/sparse-MoE).
/// BIT-EXACT vs the legacy kernels; step 1.22 -> 0.97s. `DGQ_GEMM_TUNABLE=0`
/// falls back to the legacy block GEMMs.
pub fn gemm_tunable_enabled() -> bool {
    on_unless_zero("DGQ_GEMM_TUNABLE")
}

/// Block-sparse (megablocks-style) MoE expert GEMM: ragged M pre-tiled into
/// <=32-row blocks, one block per threadgroup. Bit-identical, ~6%/step.
/// `DGQ_MOE_BLOCK_SPARSE=0` falls back to the grouped path.
pub fn moe_block_sparse_enabled() -> bool {
    on_unless_zero("DGQ_MOE_BLOCK_SPARSE")
}

/// Adaptive M-tile for block-sparse MoE (BIT-IDENTICAL; perf wash today,
/// latent win if the kernels go compute-bound — default ON per user
/// 2026-07-02). Requires block-sparse. `DGQ_MOE_TILE_ADAPT=0` opts out.
pub fn moe_tile_adapt_enabled() -> bool {
    moe_block_sparse_enabled() && on_unless_zero("DGQ_MOE_TILE_ADAPT")
}

/// Fuse the MoE gather into the gate_up block-sparse GEMM (bit-identical,
/// ~28ms/step). Requires block-sparse. `DGQ_MOE_FUSE_GATHER=0` opts out.
pub fn moe_fuse_gather_enabled() -> bool {
    moe_block_sparse_enabled() && on_unless_zero("DGQ_MOE_FUSE_GATHER")
}

/// Weight-stationary expert GEMM for batched-prefill super-chunks
/// (`DGQ_MOE_PREFILL_BM=64|128` opts in; default 32 = OFF). BUILT +
/// DISPROVEN 2026-07-07 (ROADMAP E1): taller blocks consolidate an expert's
/// ~64 routed rows at M=1024 into one dequantized weight stream (bit-
/// identical — 6.5k needle KV dump byte-equal at 32/64/128), but 64 is a
/// WASH (21.1s vs 21.1s prefill) and 128 is 3.6x SLOWER (76s, register
/// spill: TM=8 = 64 f32 accumulators/lane). Roofline says why: at M=1024
/// the expert GEMM is COMPUTE-bound at the ~2.3 TF/s kernel wall (~48
/// ms/layer MMA vs ~7.6 ms/layer weight bytes even with the 2.4x per-block
/// re-reads) — cutting W bytes can't speed a compute-bound kernel. The
/// eb0edba "capped by weight re-reads" attribution was wrong; the cap is
/// kernel TF/s. Machinery kept for re-tests.
pub fn moe_prefill_block_m() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match std::env::var("DGQ_MOE_PREFILL_BM")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(bm @ (64 | 128)) => bm,
            _ => 32,
        }
    })
}

/// GQA matrix-unit attention on sliding layers (~3-4.5%/step). Non-bit-
/// identical (f16 MMA vs f32 scalar) but quality-neutral (multi-seed + MLX
/// bench). `DGQ_ATTN_MMA=0` restores the scalar kernel.
pub fn attn_mma_enabled() -> bool {
    on_unless_zero("DGQ_ATTN_MMA")
}

/// Matrix-unit attention on FULL layers (hd=512; attention -29% at kv=512).
/// Non-bit-identical but quality-neutral (sign-off 2026-06-28).
/// `DGQ_ATTN_MMA_FULL=0` restores the scalar kernel.
pub fn attn_mma_full_enabled() -> bool {
    on_unless_zero("DGQ_ATTN_MMA_FULL")
}

/// Router-as-GEMM (~30ms/step). Non-bit-identical (near-tie expert flips =
/// trajectory re-roll, not bias); accepted 2026-07-02 on multi-seed evidence
/// and the gate was re-baselined with it. `DGQ_ROUTER_GEMM=0` restores the
/// serial router kernel.
pub fn router_gemm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_unless_zero("DGQ_ROUTER_GEMM"))
}

/// Sparse SC softembed (top-survivors gather instead of the full vocab GEMM).
/// APPROXIMATE (drops prob tail below e^-10 of row max); signed off ~16%/step,
/// output-level equivalent to MLX-4bit. Needs bf16 embed (gated at call
/// site). `DGQ_SC_SPARSE=0` uses the exact chunked path.
pub fn sc_sparse_enabled() -> bool {
    on_unless_zero("DGQ_SC_SPARSE")
}

/// Sliding-window masking on sliding-attention layers (matches MLX
/// `_make_decoder_masks`; bit-identical within the window, more correct AND
/// O(window) beyond it). `DGQ_ATTN_WINDOW=0` restores unwindowed for A/B.
pub fn attn_window_enabled() -> bool {
    on_unless_zero("DGQ_ATTN_WINDOW")
}

/// Algebraic fusion: QKV and dense gate+up as one stacked GEMM (bit-identical,
/// ~1.3%/step). `DGQ_FUSED_ALGEBRA=0` disables both for A/B.
pub fn fused_algebra_enabled() -> bool {
    on_unless_zero("DGQ_FUSED_ALGEBRA")
}

pub fn fused_qkv_enabled() -> bool {
    fused_algebra_enabled()
}

pub fn fused_gate_up_enabled() -> bool {
    fused_algebra_enabled()
}

/// Partial lm_head: skip vocab-GEMM rows for frozen canvas rows. Only
/// meaningful under `DGQ_FREEZE=1` (no rows freeze at default semantics).
/// `DGQ_PARTIAL_LM_HEAD=0` opts out.
pub fn partial_lm_head_enabled() -> bool {
    on_unless_zero("DGQ_PARTIAL_LM_HEAD")
}

/// Engine (f32) prefill runs MoE expert GEMMs on GPU (`=0` = CPU mirror, the
/// exact `.dgq` oracle — the triage lever for engine-prefill drift).
pub fn encoder_gpu_moe_enabled() -> bool {
    on_unless_zero("DGQ_ENCODER_GPU_MOE")
}

/// Cross-turn KV reuse in chat (`DGQ_KV_REUSE=0` disables): keep the causal KV
/// for the longest common prefix of the prior sequence and the new prompt, and
/// prefill only the delta at that offset. Big multi-turn win (turn N no longer
/// re-prefills the whole conversation history). NOT byte-identical to a full
/// re-prefill (the reused prefix and the engine-prefilled delta can differ in
/// precision from a single fast prefill of the whole prompt) but quality-
/// equivalent — same class as the fast-vs-engine prefill approximation.
pub fn kv_reuse_enabled() -> bool {
    on_unless_zero("DGQ_KV_REUSE")
}

/// GPU working-set cap (Metal `recommendedMaxWorkingSetSize`), captured once
/// at MetalContext init so the pure `kv_q8` policy can scale to the device's
/// RAM. 0 (unset, e.g. CPU-only tests) → the q8 auto-policy stays off.
static GPU_WORKING_SET_BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Record the device working-set cap (idempotent; first writer wins).
pub fn set_gpu_working_set_cap(bytes: u64) {
    let _ = GPU_WORKING_SET_BYTES.set(bytes);
}

/// q8 KV cache storage. Group-32 symmetric i8 + f16 scales (0.92% rel-RMS on
/// real KV), halves KV memory. **AUTO at long context since 2026-07-07**:
/// enabled when the estimated f16 resident at `max_seq` would approach the
/// GPU working-set cap, so very long sessions stay resident instead of
/// swapping. `DGQ_KV_Q8=1` forces on, `=0` forces off (override the policy).
///
/// Why auto (measured): at kv=262k (model max), f16 KV puts resident at 91%
/// of the ~27 GiB cap and the machine SWAPS (+704 MB, pressure WARN, caught
/// by DGQ_MEM_WATCH); q8 lands at 82% and doesn't. Below the cap f16 is the
/// default — it's issue-bound-faster (E4: q8 is +9%/+54% at 33k, −6% at
/// 105k), so q8 only earns its keep once memory pressure would otherwise
/// force swap. The switch scales with the cap, so smaller Macs flip earlier.
/// Constants measured on this checkpoint 2026-07-07 (f16 session-open:
/// 105k→21.63 GiB, 262k→24.61 GiB → resident ≈ 19.73 GiB + 19.0 KiB/token).
///
/// Not a SPEED lever below the cliff (issue-bound kernels; same physics rules
/// out rotated lower-bit KV for speed). Quality: forced-q8 gate 45/51 =
/// baseline; needle exact 33k/105k. Format is fixed per session at open;
/// every KV writer/reader compiles a matching function-constant variant.
pub fn kv_format(max_seq: usize) -> crate::kernels::sub::kv_quant::KvFormat {
    use crate::kernels::sub::kv_quant::KvFormat;
    // Explicit override (any set value): 1/true → q8, else → f16.
    if let Ok(v) = std::env::var("DGQ_KV_Q8") {
        return if v == "1" || v.eq_ignore_ascii_case("true") {
            KvFormat::Q8
        } else {
            KvFormat::F16
        };
    }
    // Auto: q8 when f16 resident would exceed a safe fraction of the cap.
    let Some(&cap) = GPU_WORKING_SET_BYTES.get() else {
        return KvFormat::F16; // cap unknown (tests / CPU) → keep f16
    };
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const F16_BASE_BYTES: f64 = 19.73 * GIB; // non-KV resident (weights+arena)
    const F16_KV_BYTES_PER_TOKEN: f64 = 19.0 * 1024.0; // f16 KV linear growth
    const SAFE_FRACTION: f64 = 0.85; // switch before the >90% swap regime
    let f16_resident = F16_BASE_BYTES + F16_KV_BYTES_PER_TOKEN * max_seq as f64;
    if f16_resident > SAFE_FRACTION * cap as f64 {
        KvFormat::Q8
    } else {
        KvFormat::F16
    }
}

/// KV block size for sequential-block full-attention (`DGQ_ATTN_KV_BLOCK=
/// <keys>`; DEFAULT 0 = OFF). When on and kv_len+canvas exceeds the block,
/// full-attention layers run as in-order dispatches over block-sized key
/// ranges with f32 online-softmax state persisted in `attn_state` —
/// BIT-IDENTICAL to the monolithic pass (verified: 6.5k traces byte-equal).
/// DISPROVEN AS A PERF LEVER 2026-07-06: the hoped-for SLC-locality win does
/// not exist — threadgroups already sweep keys in near-lockstep, so the SLC
/// already serves the shared stream (measured effective bandwidth 171 GB/s >
/// DRAM's ~150 at kv 32k). Blocking only adds state round-trips: 33k prefill
/// 165.7 -> 177.7 s, denoise unchanged. Kept as scaffolding (the state
/// machinery is the base for a future parallel-split or quantized-KV pass).
pub fn attn_kv_block() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DGQ_ATTN_KV_BLOCK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Batched prefill super-chunks (`DGQ_PREFILL_BATCH=0` restores plain
/// 256-token chunks): up to PREFILL_SUBS causal sub-chunks run as ONE forward
/// — MoE expert weights are streamed once per super-chunk instead of once per
/// chunk (the dominant prefill GEMM cost), and the dense GEMMs run at M=1024.
/// Bit-identical to sequential chunks (row-independent batched stages;
/// attention/rope keep per-sub causal dims + per-sub params slots).
pub fn prefill_batch_enabled() -> bool {
    on_unless_zero("DGQ_PREFILL_BATCH")
}

/// MMA attention in the fast prefill (`DGQ_PREFILL_MMA=0` restores the scalar
/// prefill attention). The MMA kernels honor the causal mask + sliding window;
/// scalar prefill was kept from the freeze era when f16 prefill attention
/// scored 11/16 vs 14/16 — re-validated after no-freeze (which fixed the
/// fast-prefill degeneration class): gate-neutral, and scalar prefill is
/// O(kv_len) serial per query = unusable at long context (6.5k prompt: 98.6s).
pub fn prefill_mma_enabled() -> bool {
    on_unless_zero("DGQ_PREFILL_MMA")
}

/// Fast (quantized) between-block KV extend (`DGQ_FAST_BLOCK_EXTEND=0` restores
/// the f32 engine extend). After a canvas block commits, its 256 tokens are
/// causally extended into KV via `prefill_chunks_from` (~0.85s) instead of the
/// engine (~10s/block — the dominant cost of multi-block replies; a 4-block
/// reply paid ~30s in extends). Same quality class as the fast-vs-engine
/// prefill approximation and cross-turn reuse; gate-validated on flip.
pub fn fast_block_extend_enabled() -> bool {
    on_unless_zero("DGQ_FAST_BLOCK_EXTEND")
}

/// Buffer-resident engine prefill (merged batches, no CPU round-trips).
/// `DGQ_PREFILL_RESIDENT=0` restores the legacy per-batch readback path.
pub fn prefill_resident_enabled() -> bool {
    on_unless_zero("DGQ_PREFILL_RESIDENT")
}

// ---------------------------------------------------------------------------
// Prefill path selection
// ---------------------------------------------------------------------------

/// Min prompt length (tokens) for the fast quantized prefill under the
/// heuristic default. Short prompts stay on the accurate f32 engine (prefill
/// is cheap there, and bf16 prompt KV can flip MoE routing on borderline
/// short-answer prompts); long prompts take the ~10x-faster quantized path.
pub const FAST_PREFILL_MIN_TOKENS: usize = 256;

/// Fast monolithic prefill (quantized + causal) vs the f32 engine for a
/// `prompt_len`-token prompt. `DGQ_FAST_PREFILL=1|0` forces on/off for all
/// lengths; unset uses the length heuristic.
pub fn should_fast_prefill(prompt_len: usize) -> bool {
    match std::env::var("DGQ_FAST_PREFILL").as_deref() {
        Ok("1") | Ok("true") => true,
        Ok("0") | Ok("false") => false,
        _ => prompt_len > FAST_PREFILL_MIN_TOKENS,
    }
}

// ---------------------------------------------------------------------------
// Debug / probe flags (all opt-in)
// ---------------------------------------------------------------------------

/// Suppress step-generate/prefill progress logs (`DGQ_QUIET=1`).
pub fn progress_enabled() -> bool {
    match std::env::var("DGQ_QUIET") {
        Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
        Err(_) => true,
    }
}

/// Opt-in logits NaN guard on the generate hot path (`DGQ_CHECK_LOGITS=1`).
pub fn logits_finite_check_enabled() -> bool {
    on_if_one("DGQ_CHECK_LOGITS")
}

/// Sample count for the logits NaN guard (`DGQ_CHECK_LOGITS_SAMPLES`).
pub fn logits_finite_sample_count() -> usize {
    match std::env::var("DGQ_CHECK_LOGITS_SAMPLES") {
        Ok(v) => v.parse().unwrap_or(4096),
        Err(_) => 4096,
    }
}

/// Per-step entropy traces in generate output (`DGQ_TRACE_ENTROPY=1|full`).
pub fn trace_entropy_enabled() -> bool {
    match std::env::var("DGQ_TRACE_ENTROPY") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full"),
        Err(_) => false,
    }
}

/// Full-canvas (vs 16-prefix) entropy trace payloads (`DGQ_TRACE_ENTROPY=full`).
pub fn trace_entropy_full() -> bool {
    std::env::var("DGQ_TRACE_ENTROPY")
        .map(|v| v.eq_ignore_ascii_case("full"))
        .unwrap_or(false)
}

/// Per-position entropy at end of a denoise block (`DGQ_LOG_FINAL_ENTROPY=1`).
pub fn final_entropy_log_enabled() -> bool {
    match std::env::var("DGQ_LOG_FINAL_ENTROPY") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full"),
        Err(_) => false,
    }
}

/// Decode answer-prefix argmax each denoise step (`DGQ_LOG_STEP_TEXT=1`).
pub fn step_text_log_enabled() -> bool {
    on_if_one("DGQ_LOG_STEP_TEXT")
}

/// GPU-vs-CPU accept-mask parity log per step (`DGQ_LOG_DENOISE=1`).
pub fn denoise_parity_log_enabled() -> bool {
    on_if_one("DGQ_LOG_DENOISE")
}

/// Positions printed by the denoise parity log (`DGQ_LOG_DENOISE_POS`).
pub fn denoise_parity_log_positions() -> usize {
    match std::env::var("DGQ_LOG_DENOISE_POS") {
        Ok(v) => v.parse().unwrap_or(8),
        Err(_) => 8,
    }
}

/// Early-stop decision log + GPU/CPU mismatch check (`DGQ_LOG_EARLY_STOP=1`).
pub fn log_early_stop_enabled() -> bool {
    on_if_one("DGQ_LOG_EARLY_STOP")
}

/// SC soft-embed stage logging (`DGQ_LOG_SC=1`).
pub fn sc_log_enabled() -> bool {
    std::env::var("DGQ_LOG_SC").ok().as_deref() == Some("1")
}

/// Per-stage bf16 activation-range trace (`DGQ_TRACE_RANGES=1`): answers
/// "does any activation exceed f16's 65504 range?" before precision
/// experiments — measure BEFORE concluding (recorded feedback).
pub fn trace_ranges_enabled() -> bool {
    std::env::var("DGQ_TRACE_RANGES").is_ok()
}

/// UMA memory-pressure watch (`DGQ_MEM_WATCH=1`): per-section (session-open
/// / prefill / denoise / profile) report of Metal working-set occupancy,
/// swap-used delta, and the kernel memory-pressure level; prints a LOUD
/// "TIMINGS SUSPECT" line when swap grew, pressure is elevated, or GPU
/// allocation exceeds 90% of recommendedMaxWorkingSetSize — wall-clock from
/// such sections is unified-memory noise, not kernel cost.
pub fn mem_watch_enabled() -> bool {
    on_if_one("DGQ_MEM_WATCH")
}

/// Dump per-layer prefill KV to `<path>` (`DGQ_DUMP_KV=<path>`) — the probe
/// that pinned the fast-prefill routing-flip mechanism.
pub fn dump_kv_path() -> Option<String> {
    std::env::var("DGQ_DUMP_KV").ok()
}

/// Dump MoE routes to `<path>` (`DGQ_MOE_ROUTE_REF=<path>`) for the
/// GPU-vs-CPU route comparison tooling.
pub fn moe_route_ref_path() -> Option<String> {
    std::env::var("DGQ_MOE_ROUTE_REF").ok()
}

/// Engine per-layer hidden dump (`DGQ_ENGINE_LAYER_DUMP=<json path>`).
pub fn engine_layer_dump_path() -> Option<String> {
    std::env::var("DGQ_ENGINE_LAYER_DUMP").ok()
}

/// Canvas row probed by the engine layer dump (`DGQ_ENGINE_LAYER_POS`).
pub fn engine_layer_dump_pos() -> usize {
    std::env::var("DGQ_ENGINE_LAYER_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(129)
}

/// Per-layer engine prefill phase timings (`DGQ_PREFILL_PROFILE=1`).
pub fn prefill_profile_enabled() -> bool {
    std::env::var("DGQ_PREFILL_PROFILE").is_ok()
}

/// Extra step-parity diagnostics (argmax agreement) (`DGQ_PARITY_DEBUG=1`).
pub fn parity_debug_enabled() -> bool {
    std::env::var_os("DGQ_PARITY_DEBUG").is_some()
}
