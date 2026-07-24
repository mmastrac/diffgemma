//! Flag accessors: the `crate::flags::<fn>()` read surface over
//! [`config`](super::config). Split from flags.rs (see mod.rs for the
//! config structs, env parse, and test override machinery).

use super::*;

// ===========================================================================

/// Hard-freeze of accepted canvas rows (`DGQ_FREEZE=1` re-enables the legacy
/// behavior). Default OFF since 2026-07-05 (user sign-off): the freeze was
/// PROVEN to be the flat-row wart driver (census 4/10 warty -> 0/10). OFF =
/// MLX/HF reference semantics (matches the CPU sampler in `sample.rs`): the
/// accept set is re-decided from fresh entropies every step, accepted rows
/// take that step's fresh denoiser token, dropped rows renoise, and the final
/// commit is the true full-canvas last-step argmax. ON additionally feeds the
/// partial-lm_head row skip (dormant at default).
pub fn freeze_enabled() -> bool {
    config().sampler.freeze
}

/// Commit the row argmax instead of the tempered categorical sample
/// (`DGQ_DENOISER_ARGMAX=0` restores HF categorical). Default ON since
/// 2026-07-05 (user sign-off): matches MLX's default user temperature=0
/// denoiser — the linear schedule temperature only shapes entropy and the SC
/// soft-embed, never the committed token. With no-freeze this is the
/// MLX-exact config (gate 16,16,11; census 0/10 warty).
pub fn denoiser_argmax_enabled() -> bool {
    config().sampler.denoiser_argmax
}

/// Entropy-only early stop (`DGQ_EARLY_STOP_MEAN_ENT=<nats>`; 0 disables).
/// A denoise block stops as soon as the full-canvas mean entropy falls below
/// this (after `min_early_stop_steps`), WITHOUT waiting for full argmax
/// stability. DEFAULT ON at 0.05 (user sign-off 2026-07-06; probe answers
/// byte-identical, multi-seed gate + wart census neutral).
pub fn early_stop_mean_ent() -> f32 {
    config().sampler.early_stop_mean_ent
}

/// Non-convergence commit guard threshold (nats); 0.0 = guard disabled.
pub fn block_commit_max_ent() -> f32 {
    config().sampler.block_commit_max_ent
}

/// Fresh-noise re-rolls before the commit guard ends the turn.
pub fn block_commit_retry() -> u32 {
    config().sampler.block_commit_retry
}

/// Empty/degenerate-reply retry (E6). On the FIRST denoise block only, if the
/// committed canvas is degenerate, re-roll the initial canvas from the
/// advancing seed stream and re-run, up to N times. DEFAULT 3 (user sign-off
/// 2026-07-07; seed-123 gate answers 13→17, seeds 7/42 unchanged 17/17).
/// `DGQ_EMPTY_REPLY_RETRY=0` disables.
pub fn empty_reply_retry() -> u32 {
    config().sampler.empty_reply_retry
}

/// Whitespace-collapse STOPGAP (`DGQ_WS_BLOCK_STOP=1` enables). A committed
/// block whose text is PURE whitespace (or all pad/filler) is dropped and the
/// turn ends. Default OFF while the newline-attractor is treated as an UNFIXED
/// BUG — a default-on stopper would truncate the evidence needed to root-cause
/// it. Flip to default ON (with gate + sign-off) only if intrinsic.
pub fn ws_block_stop_enabled() -> bool {
    config().sampler.ws_block_stop
}

/// TEST override for the denoise canvas width (E3/E6 shrink machinery). When
/// set (`DGQ_FORCE_CANVAS=64|128|...`), every denoise step runs at this active
/// canvas width instead of 256. `None` = normal (256). Clamped to [1, CANVAS]
/// at the use site. Diagnostic only; not a product flag.
pub fn force_canvas() -> Option<u32> {
    config().sampler.force_canvas
}

/// Unconditional hard-tier trim floor (`DGQ_COMMIT_CONF_HARD`; 0 = off).
/// Independent of the dup tier since the split.
pub fn commit_conf_hard() -> f32 {
    config().sampler.commit_conf_hard
}

/// Dup-conjunctive commit-time confidence trim τ (`DGQ_COMMIT_CONF_TRIM`;
/// 0 = off). Trims only rows that are BOTH below τ and argmax-duplicating.
pub fn commit_conf_trim() -> f32 {
    config().sampler.commit_conf_trim
}

/// Prefix-exit head mean-entropy threshold (`DGQ_PREFIX_EXIT`; 0 = off).
pub fn prefix_exit() -> f32 {
    config().sampler.prefix_exit
}

// ===========================================================================
// Production perf toggles (all default ON, opt-out for A/B triage)
// ===========================================================================

/// Fuse the MoE gather into the gate_up tunable expert GEMM (bit-identical,
/// ~28ms/step). `DGQ_MOE_FUSE_GATHER=0` opts out.
pub fn moe_fuse_gather_enabled() -> bool {
    config().perf.moe_fuse_gather
}

/// Weight-stationary expert GEMM for batched-prefill super-chunks
/// (`DGQ_MOE_PREFILL_BM=64|128` opts in; default 32 = OFF). BUILT + DISPROVEN
/// 2026-07-07 (ROADMAP E1): at M=1024 the expert GEMM is COMPUTE-bound at the
/// ~2.3 TF/s kernel wall, so cutting weight bytes can't speed it (64 = wash,
/// 128 = 3.6x slower via register spill). Machinery kept for re-tests.
pub fn moe_prefill_block_m() -> u32 {
    config().perf.moe_prefill_block_m
}

/// GQA matrix-unit attention on sliding layers (~3-4.5%/step). Non-bit-
/// identical (f16 MMA vs f32 scalar) but quality-neutral (multi-seed + MLX
/// bench). `DGQ_ATTN_MMA=0` restores the scalar kernel.
pub fn attn_mma_enabled() -> bool {
    config().perf.attn_mma
}

/// Matrix-unit attention on FULL layers (hd=512; attention -29% at kv=512).
/// Non-bit-identical but quality-neutral (sign-off 2026-06-28).
/// `DGQ_ATTN_MMA_FULL=0` restores the scalar kernel.
pub fn attn_mma_full_enabled() -> bool {
    config().perf.attn_mma_full
}

/// E17: route full-layer PREFILL attention through the GEMM decomposition
/// (attn_gemm_qk/softmax/pv) instead of attention_mma_full. ~1.78x on the
/// attention kernel; not bit-identical (full quality gate). `DGQ_GEMM_ATTN=1`.
pub fn gemm_attn_enabled() -> bool {
    config().perf.gemm_attn
}

/// E17a: Q heads processed per E17 dispatch batch (`DGQ_GEMM_ATTN_HC`, default
/// 16 = all heads; drop to 4 for very long contexts if memory-pressured).
/// Bounds the S/P prefill scratch; clamped to n_q_heads at dispatch. HC is
/// numerically invariant (per-head disjoint scratch).
pub fn gemm_attn_head_chunk() -> usize {
    config().perf.gemm_attn_head_chunk
}

/// Holistic sweep (task #88): dense tunable-GEMM tile (bm, bn); default 64x64.
pub fn gemm_tune_tile() -> (usize, usize) {
    let p = &config().perf;
    (p.gemm_tune_bm, p.gemm_tune_bn)
}

/// Holistic sweep (task #88): MoE-sparse N-tile; default 128 (block height
/// stays 32, baked into moe_bucket_fill).
pub fn moe_sparse_bn() -> usize {
    config().perf.moe_sparse_bn
}

/// `DGQ_ATTN_MMA_FULL_QK_ILP2`: split the QK dot in `attention_mma_full`
/// into two interleaved accumulator chains (FC31). Default OFF.
pub fn attn_mma_full_qk_ilp2() -> bool {
    config().perf.attn_mma_full_qk_ilp2
}

/// E20 enabled predicate (the only check `step_kernel` needs for routing).
pub fn attn_topk_enabled() -> bool {
    config().perf.attn_topk
}

/// E20 k per query row. Clamped to K_PAD at compile time host-side.
pub fn attn_topk_k() -> usize {
    config().perf.attn_topk_k.clamp(1, 1024)
}
/// Compile-time slot capacity for the top-k P/Idx planes: next power of two of
/// the requested k, floored at the shipped 64 (kernel default) and capped at
/// 1024. Pipeline compile (`tuned_source` AG_K_PAD) and the Rust-side plane
/// allocations must BOTH use this — same value, one source (#97's lesson).
pub fn attn_topk_k_pad() -> usize {
    if attn_topk_dyn() {
        return 512; // dyn k caps at 512 (see attn_topk_k_cfg)
    }
    attn_topk_k().next_power_of_two().clamp(64, 1024)
}
/// `DGQ_ATTN_TOPK_DYN` (kv-adaptive k).
pub fn attn_topk_dyn() -> bool {
    config().perf.attn_topk_dyn
}
/// `DGQ_ATTN_TOPK_DECODE`: top-k sparse attention on full-layer DENOISE
/// dispatches. Default ON (signed off 2026-07-16); =0 restores dense.
pub fn attn_topk_decode_enabled() -> bool {
    config().perf.attn_topk_decode
}
/// E20 k config handed to `attn_topk_softmax`: `(fixed_k, dyn_divisor,
/// k_min, k_max)`, with `dyn_divisor == 0` selecting `fixed_k`.
///
/// Under kv-adaptive k (`DGQ_ATTN_TOPK_DYN`) the DIVISOR ships and the
/// KERNEL derives k per row, from that row's own causal key count rounded
/// UP to the canvas grid — the value an aligned fresh-prefill chunk would
/// have had for that position. So k is a function of the absolute position
/// alone (phase-invariant) AND numerically unchanged from the shipped,
/// quality-gated behaviour.
///
/// It used to be resolved host-side from the DISPATCH's `t_total`
/// (`clamp(t_total/128, 64, 512)`, one k for every row in the chunk), which
/// made k depend on how the prefill happened to be chunked: the same query
/// got k=114 on a fresh prefill and k=115 when resumed mid-context, so a
/// reused-KV turn attended a different number of keys than a fresh one and
/// forked away from it. That host-side helper is deliberately GONE rather
/// than left for reuse. See `kv_lineage_tests` and the PLAN lineage item.
///
/// Do NOT "simplify" the kernel to use the raw per-row count: that is also
/// phase-invariant but SHRINKS k for early rows in a chunk, which measured
/// as worse long-context prose (a dropped word in doc_20k on 2/3 seeds,
/// invisible to the retrieval gate).
pub fn attn_topk_k_cfg() -> [u32; 4] {
    let pad = attn_topk_k_pad() as u32;
    if attn_topk_dyn() {
        // `.x` carries the CHUNK GRID (the fast prefill's fixed CANVAS stride)
        // the kernel rounds each row's key count up to; `fixed_k` is unused in
        // this mode. It must be the prefill chunk stride, not the dispatch row
        // count, or k stops matching the aligned fresh-prefill value.
        [crate::metal::CANVAS as u32, 128, 64, 512.min(pad)]
    } else {
        [(attn_topk_k() as u32).min(pad), 0, 0, 0]
    }
}

/// E18 fused flash prefill. `.0` = enabled, `.1` = BQ (query-row tile), `.2` =
/// BK (streamed key block). Default off / 16 / 64.
pub fn flash_prefill() -> (bool, usize, usize) {
    let p = &config().perf;
    (p.flash_prefill, p.flash_prefill_bq, p.flash_prefill_bk)
}

/// E17 tunable tile config (task #87). Defaults reproduce the shipped
/// 64x64/256 kernel. Returns (qk_bm, qk_bn, pv_bm, pv_bn, sm_tpg).
pub fn gemm_attn_tile() -> (usize, usize, usize, usize, usize) {
    let p = &config().perf;
    (
        p.gemm_attn_qk_bm,
        p.gemm_attn_qk_bn,
        p.gemm_attn_pv_bm,
        p.gemm_attn_pv_bn,
        p.gemm_attn_sm_tpg,
    )
}

/// Router-as-GEMM (~30ms/step). Non-bit-identical (near-tie expert flips =
/// trajectory re-roll, not bias); accepted 2026-07-02 on multi-seed evidence
/// and the gate was re-baselined with it. `DGQ_ROUTER_GEMM=0` restores the
/// serial router kernel.
pub fn router_gemm_enabled() -> bool {
    config().perf.router_gemm
}

/// Sparse SC softembed (top-survivors gather instead of the full vocab GEMM).
/// APPROXIMATE (drops prob tail below e^-10 of row max); signed off ~16%/step,
/// output-level equivalent to MLX-4bit. Needs bf16 embed (gated at call
/// site). `DGQ_SC_SPARSE=0` uses the exact chunked path.
pub fn sc_sparse_enabled() -> bool {
    config().perf.sc_sparse
}

/// Sliding-window masking on sliding-attention layers (matches MLX
/// `_make_decoder_masks`; bit-identical within the window, more correct AND
/// O(window) beyond it). `DGQ_ATTN_WINDOW=0` restores unwindowed for A/B.
pub fn attn_window_enabled() -> bool {
    config().perf.attn_window
}

/// Algebraic fusion: QKV and dense gate+up as one stacked GEMM (bit-identical,
/// ~1.3%/step). `DGQ_FUSED_ALGEBRA=0` disables both for A/B.
pub fn fused_algebra_enabled() -> bool {
    config().perf.fused_algebra
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
    config().perf.partial_lm_head
}

/// Engine (f32) prefill runs MoE expert GEMMs on GPU (`=0` = CPU mirror, the
/// exact `.dgq` oracle — the triage lever for engine-prefill drift).
pub fn encoder_gpu_moe_enabled() -> bool {
    config().perf.encoder_gpu_moe
}

/// Cross-turn KV reuse in chat (`DGQ_KV_REUSE=0` disables): keep the causal KV
/// for the longest common prefix of the prior sequence and the new prompt, and
/// prefill only the delta at that offset. NOT byte-identical to a full
/// re-prefill but quality-equivalent (same class as fast-vs-engine prefill).
pub fn kv_reuse_enabled() -> bool {
    config().kv.reuse
}

/// Byte budget for the server's multi-conversation KV snapshot pool
/// (`DGQ_CONV_CACHE_GB`). Default 0 = minimal (single hot conversation).
pub fn conv_cache_bytes() -> usize {
    config().server.conv_cache_bytes
}

/// Disk byte budget for the server's SSD conversation-snapshot tier
/// (`DGQ_CONV_DISK_GB`). Default 0 = no disk tier.
pub fn conv_disk_bytes() -> usize {
    config().server.conv_disk_bytes
}

/// Directory for the SSD conversation-snapshot tier (`DGQ_CONV_CACHE_DIR`).
pub fn conv_cache_dir() -> PathBuf {
    config().server.conv_cache_dir.clone()
}

/// `DGQ_CONTINUE_PAST_STOP=1`: restore serve's defer-past-stop bet (default
/// OFF; the premature stop is a strain symptom, not a formatting hiccup).
pub fn continue_past_stop_enabled() -> bool {
    config().server.continue_past_stop
}

/// `DGQ_PREFILL_STATUS` (default ON, `=0` disables): streamed serve requests
/// emit a synthetic `reasoning_content` status line + elapsed heartbeat while
/// a large prompt delta prefills — the dry-start would otherwise be silent
/// until the first denoise step.
pub fn prefill_status_enabled() -> bool {
    config().server.prefill_status
}

/// `DGQ_TOOL_REPAIR=1`: model-guided tool-call repair (error tool-response →
/// corrected call → corrupt exchange rewound out of KV). Equivalent to
/// `serve --tool-repair`. Default OFF.
pub fn tool_repair_enabled() -> bool {
    config().server.tool_repair
}

/// `DGQ_TOOL_VALIDATE=1`: serve rewinds + regenerates (bumped seed) when a
/// reply's tool-call grammar is malformed. Opt-IN and default OFF
/// (quality-affecting: a retry replaces the reply). Equivalent to
/// `serve --tool-validate`.
pub fn tool_validate_enabled() -> bool {
    config().server.tool_validate
}

/// `DGQ_TOOL_COMPACT=1`: enable the serve tool-output compactor (KV rewinder).
/// Opt-IN and default OFF (quality-affecting, not yet gate-signed-off).
/// Equivalent to `serve --tool-compact`.
/// Paced commit streaming (`DGQ_PACED_STREAM`; default ON, `0` disables).
pub fn paced_stream_enabled() -> bool {
    config().server.paced_stream
}

pub fn tool_compact_enabled() -> bool {
    config().server.tool_compact
}

/// Token threshold above which a tool response is compacted
/// (`DGQ_TOOL_COMPACT_THRESHOLD`). Default 384.
pub fn tool_compact_threshold() -> usize {
    config().server.tool_compact_threshold
}

/// Directory for full tool outputs stored by the compactor
/// (`DGQ_TOOL_COMPACT_DIR`).
pub fn tool_compact_dir() -> PathBuf {
    config().server.tool_compact_dir.clone()
}

/// `DGQ_KV_MMAP=1`: back the session KV cache with a `MAP_SHARED` temp-file mmap
/// (wrapped no-copy as the Metal buffer). Opt-in experiment. Default OFF.
pub fn kv_mmap() -> bool {
    config().kv.mmap
}

/// Directory for the `DGQ_KV_MMAP` backing file. Defaults to the system temp dir.
pub fn kv_mmap_dir() -> PathBuf {
    config().kv.mmap_dir.clone()
}

// ===========================================================================
// GPU working-set cap (a device fact, not env config)
// ===========================================================================

/// GPU working-set cap (Metal `recommendedMaxWorkingSetSize`), captured once
/// at MetalContext init so the pure `kv_q8` policy can scale to the device's
/// RAM. 0 (unset, e.g. CPU-only tests) → the q8 auto-policy stays off.
static GPU_WORKING_SET_BYTES: OnceLock<u64> = OnceLock::new();

/// Record the device working-set cap (idempotent; first writer wins).
pub fn set_gpu_working_set_cap(bytes: u64) {
    let _ = GPU_WORKING_SET_BYTES.set(bytes);
}

/// q8 KV cache storage. Group-32 symmetric i8 + f16 scales (0.92% rel-RMS),
/// halves KV memory. **AUTO at long context since 2026-07-07**: enabled when
/// the estimated f16 resident at `max_seq` would approach the GPU working-set
/// cap, so very long sessions stay resident instead of swapping. `DGQ_KV_Q8=1`
/// forces on, `=0` forces off. Format is fixed per session at open; every KV
/// writer/reader compiles a matching function-constant variant.
pub fn kv_format(max_seq: usize) -> crate::shaders::kv_quant::KvFormat {
    use crate::shaders::kv_quant::KvFormat;
    // Explicit override wins.
    if let Some(force_q8) = config().kv.q8_override {
        return if force_q8 {
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

// Resident-memory model for the q4emb checkpoint (measured 2026-07-07; see
// `kv_format`): resident ≈ 19.73 GiB weights+arena + KV linear in tokens.
const CTX_BASE_BYTES: f64 = 19.73 * 1024.0 * 1024.0 * 1024.0;
const CTX_F16_KV_PER_TOKEN: f64 = 19.0 * 1024.0;
/// q8 KV (half) auto-enables once f16 resident exceeds this fraction of budget
/// — matches the `kv_format` auto policy so the guard mirrors the engine.
const CTX_Q8_AUTO_FRACTION: f64 = 0.85;
/// Safe resident ceiling fraction of the budget: past this Metal swaps / the
/// KV alloc fails.
pub const CTX_SAFE_FRACTION: f64 = 0.90;

/// Estimated GPU-resident bytes at `max_seq`, mirroring what the engine will
/// actually allocate: q8 KV (≈half) auto-enables once the f16 resident would
/// exceed `CTX_Q8_AUTO_FRACTION` of `budget_bytes`. `DGQ_KV_Q8` forces the
/// format either way — the SAME rule as `kv_format`.
pub fn estimate_resident_bytes(max_seq: usize, budget_bytes: u64) -> u64 {
    let f16 = CTX_BASE_BYTES + CTX_F16_KV_PER_TOKEN * max_seq as f64;
    let q8 = match config().kv.q8_override {
        Some(force_q8) => force_q8,
        None => budget_bytes > 0 && f16 > CTX_Q8_AUTO_FRACTION * budget_bytes as f64,
    };
    let per_tok = if q8 {
        CTX_F16_KV_PER_TOKEN * 0.5
    } else {
        CTX_F16_KV_PER_TOKEN
    };
    (CTX_BASE_BYTES + per_tok * max_seq as f64) as u64
}

/// Guard for a user-requested context length: `Some((needed, ceiling))` bytes
/// when the estimated resident at `max_seq` would exceed `CTX_SAFE_FRACTION` of
/// `budget_bytes`. `None` = fits, or `budget_bytes == 0` (unknown).
pub fn ctx_over_budget(max_seq: usize, budget_bytes: u64) -> Option<(u64, u64)> {
    if budget_bytes == 0 {
        return None;
    }
    let ceiling = (CTX_SAFE_FRACTION * budget_bytes as f64) as u64;
    let needed = estimate_resident_bytes(max_seq, budget_bytes);
    (needed > ceiling).then_some((needed, ceiling))
}

/// Largest `--ctx` (max_seq) whose estimate stays under the safe ceiling, for
/// the "reduce to <= N" hint. At the ceiling ctx is large, so q8 KV is
/// auto-selected. 0 when the budget can't even hold the base weights.
pub fn max_feasible_ctx(budget_bytes: u64) -> usize {
    let headroom = CTX_SAFE_FRACTION * budget_bytes as f64 - CTX_BASE_BYTES;
    (headroom / (CTX_F16_KV_PER_TOKEN * 0.5)).max(0.0) as usize
}

/// KV block size for sequential-block full-attention (`DGQ_ATTN_KV_BLOCK`;
/// DEFAULT 0 = OFF). BIT-IDENTICAL to the monolithic pass but DISPROVEN AS A
/// PERF LEVER 2026-07-06 (the SLC already serves the shared key stream in
/// near-lockstep). Kept as scaffolding for a future parallel-split/quantized-KV pass.
pub fn attn_kv_block() -> usize {
    config().perf.attn_kv_block
}

/// Batched prefill super-chunks (`DGQ_PREFILL_BATCH=0` restores plain 256-token
/// chunks): up to PREFILL_SUBS causal sub-chunks run as ONE forward. Bit-
/// identical to sequential chunks.
pub fn prefill_batch_enabled() -> bool {
    config().perf.prefill_batch
}

/// MMA attention in the fast prefill (`DGQ_PREFILL_MMA=0` restores the scalar
/// prefill attention). Scalar prefill is O(kv_len) serial per query = unusable
/// at long context.
pub fn prefill_mma_enabled() -> bool {
    config().perf.prefill_mma
}

/// Fast (quantized) between-block KV extend (`DGQ_FAST_BLOCK_EXTEND=0` restores
/// the f32 engine extend). After a canvas block commits, its 256 tokens are
/// causally extended via `prefill_chunks_from` (~0.85s) instead of the engine
/// (~10s/block). Same quality class as fast-vs-engine prefill.
pub fn fast_block_extend_enabled() -> bool {
    config().perf.fast_block_extend
}

/// Buffer-resident engine prefill (merged batches, no CPU round-trips).
/// `DGQ_PREFILL_RESIDENT=0` restores the legacy per-batch readback path.
pub fn prefill_resident_enabled() -> bool {
    config().perf.prefill_resident
}

// ===========================================================================
// Prefill path selection
// ===========================================================================

/// Min prompt length (tokens) for the fast quantized prefill under the
/// heuristic default. Short prompts stay on the accurate f32 engine.
pub const FAST_PREFILL_MIN_TOKENS: usize = 256;

/// E11 (DISPROVEN as the long-prompt fix; kept as a diagnostic): fast prefill
/// runs on fp16 activation-arena pipelines instead of bf16. The real cause of
/// task #64 was the spurious encoder RmsNormHidden (fixed 2b0d12b). Opt-in.
pub fn prefill_f16_enabled() -> bool {
    config().prefill.f16
}

/// E14: during fast prefill, sliding layers write/read K/V through an f32 side
/// ring (window-sized) instead of the f16 monolithic cache — kills the
/// chunk-boundary rounding that compounds through the causal chain and
/// destroyed long-prompt comprehension (task #64/#67). Requires the MMA
/// prefill attention (default).
pub fn prefill_kv_f32_enabled() -> bool {
    config().prefill.kv_f32
}

/// Max prompt length (tokens) the fast quantized prefill handles
/// (`DGQ_FAST_PREFILL_MAX`; 0 = no cap). DEFAULT 0 (uncapped) since the task
/// #64/#68 root-cause fix (2b0d12b, 2026-07-10): the length-dependent
/// comprehension collapse was a spurious encoder-pass RmsNormHidden, not
/// accumulating precision error.
pub fn fast_prefill_max_tokens() -> usize {
    config().prefill.fast_prefill_max
}

/// Fast monolithic prefill (quantized + causal) vs the f32 engine for a
/// `prompt_len`-token prompt. `DGQ_FAST_PREFILL=1|0` forces on/off for all
/// lengths; unset uses the length band (see [`fast_prefill_max_tokens`]).
pub fn should_fast_prefill(prompt_len: usize) -> bool {
    match config().prefill.fast_prefill_force {
        Some(v) => v,
        None => {
            let max = fast_prefill_max_tokens();
            prompt_len > FAST_PREFILL_MIN_TOKENS && (max == 0 || prompt_len <= max)
        }
    }
}

// ===========================================================================
// Debug / probe flags (all opt-in)
// ===========================================================================

/// Suppress step-generate/prefill progress logs (`DGQ_QUIET=1`).
pub fn progress_enabled() -> bool {
    !config().debug.quiet
}

/// Opt-in logits NaN guard on the generate hot path (`DGQ_CHECK_LOGITS=1`).
pub fn logits_finite_check_enabled() -> bool {
    config().debug.logits_finite_check
}

/// Sample count for the logits NaN guard (`DGQ_CHECK_LOGITS_SAMPLES`).
pub fn logits_finite_sample_count() -> usize {
    config().debug.logits_finite_samples
}

/// Per-step entropy traces in generate output (`DGQ_TRACE_ENTROPY=1|full`).
pub fn trace_entropy_enabled() -> bool {
    config().debug.trace_entropy
}

/// Full-canvas (vs 16-prefix) entropy trace payloads (`DGQ_TRACE_ENTROPY=full`).
pub fn trace_entropy_full() -> bool {
    config().debug.trace_entropy_full
}

/// Per-position entropy at end of a denoise block (`DGQ_LOG_FINAL_ENTROPY=1`).
pub fn final_entropy_log_enabled() -> bool {
    config().debug.final_entropy_log
}

/// Decode the answer-region argmax each denoise step and show it in the step
/// progress line. Log-only — generation is bit-identical either way. DEFAULT ON
/// since 2026-07-09. `DGQ_LOG_STEP_TEXT=0` disables.
pub fn step_text_log_enabled() -> bool {
    config().debug.step_text_log
}

/// GPU-vs-CPU accept-mask parity log per step (`DGQ_LOG_DENOISE=1`).
pub fn denoise_parity_log_enabled() -> bool {
    config().debug.denoise_parity_log
}

/// Positions printed by the denoise parity log (`DGQ_LOG_DENOISE_POS`).
pub fn denoise_parity_log_positions() -> usize {
    config().debug.denoise_parity_positions
}

/// Early-stop decision log + GPU/CPU mismatch check (`DGQ_LOG_EARLY_STOP=1`).
pub fn log_early_stop_enabled() -> bool {
    config().debug.log_early_stop
}

/// SC soft-embed stage logging (`DGQ_LOG_SC=1`).
pub fn sc_log_enabled() -> bool {
    config().debug.sc_log
}

/// Per-stage bf16 activation-range trace (`DGQ_TRACE_RANGES=1`): answers "does
/// any activation exceed f16's 65504 range?" before precision experiments.
pub fn trace_ranges_enabled() -> bool {
    config().debug.trace_ranges
}

/// UMA memory-pressure watch (`DGQ_MEM_WATCH=1`): per-section working-set /
/// swap-delta / pressure report; prints a LOUD "TIMINGS SUSPECT" when swap grew
/// or allocation exceeds 90% of the working-set cap.
pub fn mem_watch_enabled() -> bool {
    config().debug.mem_watch
}

/// Dump per-layer prefill KV to `<path>` (`DGQ_DUMP_KV=<path>`).
pub fn dump_kv_path() -> Option<String> {
    config().debug.dump_kv_path.clone()
}

/// E7 M0 p_max trace JSONL path (`DGQ_TRACE_PMAX_JSONL=<path>`).
pub fn trace_pmax_jsonl() -> Option<String> {
    config().debug.trace_pmax_jsonl.clone()
}

/// Structural delimiter/parity check on committed blocks (`DGQ_DELIM_CHECK=1`).
/// Observational: it never steers generation.
pub fn delim_check_enabled() -> bool {
    config().debug.delim_check || config().debug.delim_check_jsonl.is_some()
}

/// Per-block delimiter-check JSONL path (`DGQ_DELIM_CHECK_JSONL=<path>`).
pub fn delim_check_jsonl() -> Option<String> {
    config().debug.delim_check_jsonl.clone()
}

/// Path to an output-token classification probe (`DGQ_TOKEN_CLASS=<file>`).
/// `None` means the classifier is off and NO hidden readback is performed.
pub fn token_class_probe() -> Option<String> {
    config().debug.token_class_probe.clone()
}

/// Dump MoE routes to `<path>` (`DGQ_MOE_ROUTE_REF=<path>`).
pub fn moe_route_ref_path() -> Option<String> {
    config().debug.moe_route_ref_path.clone()
}

/// Engine per-layer hidden dump (`DGQ_ENGINE_LAYER_DUMP=<json path>`).
pub fn engine_layer_dump_path() -> Option<String> {
    config().debug.engine_layer_dump_path.clone()
}

/// Canvas row probed by the engine layer dump (`DGQ_ENGINE_LAYER_POS`).
pub fn engine_layer_dump_pos() -> usize {
    config().debug.engine_layer_dump_pos
}

/// Per-layer engine prefill phase timings (`DGQ_PREFILL_PROFILE=1`).
pub fn prefill_profile_enabled() -> bool {
    config().debug.prefill_profile
}

/// Extra step-parity diagnostics (argmax agreement) (`DGQ_PARITY_DEBUG=1`).
pub fn parity_debug_enabled() -> bool {
    config().debug.parity_debug
}

/// KV-noise sensitivity probe rel-eps (`DGQ_KV_NOISE=<f32>`; None = off).
pub fn kv_noise() -> Option<f32> {
    config().debug.kv_noise
}

/// Sliding-layer KV ring slots (`DGQ_KV_RING`, default 4096; power of two,
/// min 2048). See the field doc for the slack arithmetic.
pub fn kv_ring_slots() -> usize {
    config().debug.kv_ring_slots
}

/// Linear (no-wrap) sliding KV storage (`DGQ_KV_RING_UNCAPPED`).
pub fn kv_ring_uncapped_enabled() -> bool {
    config().debug.kv_ring_uncapped
}

/// Build the MAIN activation set fp16 too (`DGQ_ARENA_F16_ALL`).
pub fn arena_f16_all_enabled() -> bool {
    config().debug.arena_f16_all
}

/// On-disk Metal pipeline cache enabled (`DGQ_METAL_PIPELINE_CACHE`; `0`/`false`
/// disables, unset or any other value enables).
pub fn metal_pipeline_cache_enabled() -> bool {
    match config().debug.metal_pipeline_cache.as_deref() {
        Some(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        None => true,
    }
}

/// Explicit Metal pipeline-cache root dir from `DGQ_METAL_PIPELINE_CACHE` when
/// it names a (non-empty, non-off) path; `None` falls back to the XDG/HOME
/// default (kept in pipeline_cache.rs — those are system env, not DGQ config).
pub fn metal_pipeline_cache_dir_override() -> Option<PathBuf> {
    config()
        .debug
        .metal_pipeline_cache
        .as_deref()
        .filter(|v| !v.is_empty() && *v != "0" && !v.eq_ignore_ascii_case("false"))
        .map(PathBuf::from)
}

#[cfg(test)]
mod ctx_budget_tests {
    use super::*;
    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn unknown_budget_never_trips() {
        assert!(ctx_over_budget(1_000_000, 0).is_none());
    }

    #[test]
    fn small_ctx_fits_large_ctx_refused_on_36gib() {
        let budget = 26 * GIB; // ~72% of a 36 GiB machine
        assert!(ctx_over_budget(8_192, budget).is_none(), "8k must fit");
        assert!(
            ctx_over_budget(900_000, budget).is_some(),
            "900k must be refused"
        );
    }

    #[test]
    fn max_feasible_is_a_real_boundary() {
        let budget = 26 * GIB;
        let n = max_feasible_ctx(budget);
        assert!(n > 0);
        // At the reported max it fits; just past it, it's refused.
        assert!(ctx_over_budget(n.saturating_sub(1024), budget).is_none());
        assert!(ctx_over_budget(n + 50_000, budget).is_some());
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn default_matches_documented_shipped_defaults() {
        let c = RuntimeConfig::default();
        // Sampler sign-offs.
        assert!(!c.sampler.freeze);
        assert!(c.sampler.denoiser_argmax);
        assert_eq!(c.sampler.early_stop_mean_ent, 0.05);
        assert_eq!(c.sampler.empty_reply_retry, 3);
        // Perf toggles ship ON.
        assert!(c.perf.attn_mma && c.perf.fused_algebra);
        assert_eq!(c.perf.moe_prefill_block_m, 32);
        // Prefill uncapped, KV reuse on, tool-compact off.
        assert_eq!(c.prefill.fast_prefill_max, 0);
        assert!(c.kv.reuse);
        assert!(!c.server.tool_compact);
        assert_eq!(c.server.tool_compact_threshold, 384);
    }

    #[test]
    fn install_for_test_overrides_and_restores() {
        let before = freeze_enabled();
        {
            let mut cfg = RuntimeConfig::default();
            cfg.sampler.freeze = !before;
            let _g = install_for_test(cfg);
            assert_eq!(freeze_enabled(), !before, "override takes effect");
        }
        assert_eq!(freeze_enabled(), before, "prior config restored on drop");
    }

    #[test]
    fn set_quiet_flips_progress() {
        let _g = install_for_test(RuntimeConfig::default());
        set_quiet(true);
        assert!(!progress_enabled());
        set_quiet(false);
        assert!(progress_enabled());
    }

    #[test]
    fn kv_q8_override_is_consistent_across_consumers() {
        use crate::shaders::kv_quant::KvFormat;
        let budget = 26 * 1024 * 1024 * 1024u64;
        let q8_bytes = {
            let mut cfg = RuntimeConfig::default();
            cfg.kv.q8_override = Some(true);
            let _g = install_for_test(cfg);
            assert!(matches!(kv_format(8192), KvFormat::Q8), "force-on → q8");
            estimate_resident_bytes(100_000, budget)
        };
        let f16_bytes = {
            let mut cfg = RuntimeConfig::default();
            cfg.kv.q8_override = Some(false);
            let _g = install_for_test(cfg);
            assert!(matches!(kv_format(8192), KvFormat::F16), "force-off → f16");
            estimate_resident_bytes(100_000, budget)
        };
        // Both consumers honor the SAME override: q8 halves per-token KV.
        assert!(q8_bytes < f16_bytes, "q8 override must shrink the estimate");
    }

    #[test]
    fn metal_pipeline_cache_dual_semantics() {
        {
            let _g = install_for_test(RuntimeConfig::default());
            assert!(metal_pipeline_cache_enabled(), "unset → enabled");
            assert!(metal_pipeline_cache_dir_override().is_none());
        }
        for off in ["0", "false", "FALSE"] {
            let mut c = RuntimeConfig::default();
            c.debug.metal_pipeline_cache = Some(off.to_string());
            let _g = install_for_test(c);
            assert!(!metal_pipeline_cache_enabled(), "{off} disables");
            assert!(metal_pipeline_cache_dir_override().is_none());
        }
        let mut c = RuntimeConfig::default();
        c.debug.metal_pipeline_cache = Some("/tmp/pc".to_string());
        let _g = install_for_test(c);
        assert!(metal_pipeline_cache_enabled(), "a path enables");
        assert_eq!(
            metal_pipeline_cache_dir_override(),
            Some(PathBuf::from("/tmp/pc"))
        );
    }

    #[test]
    fn override_is_thread_local() {
        // An override installed here must NOT be visible on another thread
        // (which sees the shared base) — the isolation the design guarantees.
        let mut cfg = RuntimeConfig::default();
        cfg.sampler.freeze = true;
        let _g = install_for_test(cfg);
        assert!(freeze_enabled(), "override visible on this thread");
        let base_freeze = std::thread::spawn(freeze_enabled).join().unwrap();
        assert!(!base_freeze, "other thread sees the base, not the override");
    }

    #[test]
    fn should_fast_prefill_respects_force_and_band() {
        let mut cfg = RuntimeConfig::default();
        cfg.prefill.fast_prefill_force = Some(false);
        {
            let _g = install_for_test(cfg.clone());
            assert!(!should_fast_prefill(10_000), "force-off wins at any length");
        }
        cfg.prefill.fast_prefill_force = None;
        cfg.prefill.fast_prefill_max = 0;
        let _g = install_for_test(cfg);
        assert!(
            !should_fast_prefill(FAST_PREFILL_MIN_TOKENS),
            "at floor = engine"
        );
        assert!(
            should_fast_prefill(FAST_PREFILL_MIN_TOKENS + 1),
            "above floor = fast"
        );
    }
}
