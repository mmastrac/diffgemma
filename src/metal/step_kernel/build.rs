//! `build_step_runtime` + the shared-pipeline cache and bench skip-set.
//! Split from exec.rs; constructs `StepRuntime` (runtime.rs) whose encode
//! path is `StepEnc` (enc.rs).

use super::*;

static STEP_PIPELINES_CACHE: std::sync::OnceLock<
    std::sync::Mutex<HashMap<StepPipelineKey, &'static StepPipelines>>,
> = std::sync::OnceLock::new();

/// Diagnostic stage-ablation skip set (bench_prefill_super_stages only). Empty
/// in production → `prefill_bench_skipped` is a fast empty-slice check.
static PREFILL_BENCH_SKIP: std::sync::OnceLock<std::sync::Mutex<Vec<step_schedule::StepStage>>> =
    std::sync::OnceLock::new();

pub(super) fn set_prefill_bench_skip(stages: &[step_schedule::StepStage]) {
    let m = PREFILL_BENCH_SKIP.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    *m.lock().unwrap() = stages.to_vec();
}

pub(super) fn prefill_bench_skipped(stage: step_schedule::StepStage) -> bool {
    match PREFILL_BENCH_SKIP.get() {
        Some(m) => m.lock().unwrap().contains(&stage),
        None => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StepPipelineKey(u8);

fn step_pipeline_key(
    variant: crate::shaders::variant::KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> StepPipelineKey {
    // KV format code occupies bits 3-4 (0=f16, 1=q8, 2=q4); bit 5 = fp16 arena.
    StepPipelineKey(
        u8::from(variant.shape_assert)
            | (u8::from(variant.debug_fast) << 1)
            | (u8::from(variant.debug_deep) << 2)
            | ((fmt.code() as u8) << 3)
            | (u8::from(variant.arena_f16) << 5),
    )
}

fn shared_step_pipelines(
    ctx: &MetalContext,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<&'static StepPipelines, Error> {
    let variant = crate::shaders::variant::runtime_step_variant();
    let key = step_pipeline_key(variant, fmt);
    let cache = STEP_PIPELINES_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| Error::Gpu("step pipelines cache poisoned"))?;
    if let Some(&pipelines) = guard.get(&key) {
        return Ok(pipelines);
    }

    let pipelines = StepPipelines::new(ctx, variant, fmt)?;
    let leaked: &'static StepPipelines = Box::leak(Box::new(pipelines));
    guard.insert(key, leaked);
    crate::metal::pipeline_cache::PipelineArchiveCache::flush_global();
    Ok(leaked)
}

pub fn log_step_memory_budget(blob_bytes: u64, max_seq: usize, layout: &ModelLayout) {
    let kv = kv_cache_bytes(layout, max_seq);
    let logits = (CANVAS * VOCAB * 2) as u64;
    let sc_probs = sc_probs_buffer_bytes() as u64;
    let arena = step_arena_layout().bytes();
    let (mx, mw) = gemm_scratch_bytes();
    let gemm_scratch = (mx + mw) as u64;
    let gpu_static = kv + logits + sc_probs + arena + gemm_scratch;
    let total = blob_bytes + gpu_static;
    if crate::flags::progress_enabled() {
        eprintln!("step-kernel memory budget:");
        eprintln!(
            "  blob:       {:.2} GiB",
            blob_bytes as f64 / (1024.0_f64.powi(3))
        );
        eprintln!("  arena:      {:.2} MiB", arena as f64 / (1024.0 * 1024.0));
        eprintln!(
            "  kv cache:   {:.2} MiB (max_seq={max_seq})",
            kv as f64 / (1024.0 * 1024.0)
        );
        eprintln!("  logits:     {:.2} MiB", logits as f64 / (1024.0 * 1024.0));
        eprintln!(
            "  sc_probs:   {:.2} MiB",
            sc_probs as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  gemm scratch:{:.2} MiB",
            gemm_scratch as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  gpu static: {:.2} GiB (excl. blob)",
            gpu_static as f64 / (1024.0_f64.powi(3))
        );
        eprintln!(
            "  total est:  {:.2} GiB",
            total as f64 / (1024.0_f64.powi(3))
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepRuntimeBuildTiming {
    pub compile: std::time::Duration,
    pub total: std::time::Duration,
}

pub fn build_step_runtime(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<(StepRuntime, StepRuntimeBuildTiming), Error> {
    let build_started = Instant::now();
    let validated = crate::metal::step_config::validate_step_model(model_dir)?;
    crate::metal::step_config::log_validated_step_model(&validated);

    let store = DgqStore::open(model_dir)?;
    let offsets = build_offsets_from_store(&store);
    let layout = build_layout(&offsets, cfg.max_seq);
    if crate::flags::progress_enabled() {
        let fmt = crate::flags::kv_format(cfg.max_seq);
        if fmt != crate::shaders::kv_quant::KvFormat::F16 {
            eprintln!(
                "step-kernel: {} KV cache (auto at long context, max_seq={}) — shrinks KV to stay under the GPU working-set cap",
                fmt.label(),
                cfg.max_seq
            );
        }
    }
    let layers = cfg.layers.min(validated.num_layers).max(1);

    // Mixed-precision .dgq stores attention + dense-FFN as bf16 (Raw) — or q8 on
    // earlier checkpoints. Detect from the actual stored kind so any vintage
    // dispatches correctly (bf16 / q8 / uniform-q4).
    let attn_ffn_kind = store
        .get_entry("model.decoder.layers.0.self_attn.q_proj.weight")
        .and_then(|e| crate::dgq::layout::parse_quant_kind(&e.meta.kind).ok());
    let attn_ffn_q8 = attn_ffn_kind == Some(crate::dgq::layout::QuantKind::Q8Row);
    let attn_ffn_bf16 = attn_ffn_kind == Some(crate::dgq::layout::QuantKind::Raw);

    // Embed (tied lm_head + SC soft-embed) is q8-per-row on most checkpoints, bf16
    // (Raw) on newer ones. Detect from the stored kind so all three embed consumers
    // (input gather, lm_head, SC softembed) dispatch the matching precision.
    let embed_bf16 = store
        .get_entry("model.decoder.embed_tokens.weight")
        .and_then(|e| crate::dgq::layout::parse_quant_kind(&e.meta.kind).ok())
        == Some(crate::dgq::layout::QuantKind::Raw);
    let block_profile = StepBlockProfile::from_store_profile(store.profile());
    if crate::flags::progress_enabled() {
        if embed_bf16 {
            eprintln!("step-kernel: bf16 embed (tied lm_head + SC)");
        }
        match block_profile.format {
            QuantFormat::NvFp4 => eprintln!("step-kernel: nvfp4 block weights"),
            QuantFormat::Q4Affine => eprintln!("step-kernel: q4 block weights"),
            _ => eprintln!("step-kernel: block weights ({:?})", block_profile.format),
        }
        match block_profile.moe_style() {
            MoeExecutionStyle::BatchedGrouped => {
                eprintln!("step-kernel: batched grouped MoE");
            }
            MoeExecutionStyle::ScalarPerExpert => {
                eprintln!("step-kernel: scalar per-expert MoE");
            }
        }
    }

    // Take the machine-wide memory grant BEFORE any model-scale allocation.
    // Blocks (FIFO, cross-process aware) until the footprint fits; see
    // `membudget` for the contract. Estimate: the weights blob (GPU upload;
    // the mmap side is clean file-backed and evictable), the KV cache at this
    // max_seq, and a fixed slack for arena/scratch/encoder-cache structures.
    let estimate = store.blob_bytes() as usize
        + crate::metal::step_kv::kv_cache_total_bytes(&layout, cfg.max_seq) as usize
        + (2usize << 30);
    let mem_permit = crate::membudget::MemBudget::global()
        .acquire(estimate, "step-runtime")
        .map_err(|e| {
            eprintln!("membudget: {e}");
            Error::Runtime("membudget: timed out waiting for memory grant (holders on stderr)")
        })?;

    let ctx = MetalContext::new()?;
    let compile_started = Instant::now();
    let kv_fmt = crate::flags::kv_format(cfg.max_seq);
    // TEMP diagnostic (E11 bring-up): DGQ_ARENA_F16_ALL=1 builds the MAIN set
    // fp16 too — bisects kernel-level breakage from mode-switch wiring.
    let f16_all = crate::flags::arena_f16_all_enabled();
    if f16_all {
        crate::shaders::variant::set_arena_f16_compile(true);
    }
    let pipelines = shared_step_pipelines(&ctx, kv_fmt)?;
    // f16_all: LEAVE the compile-mode atomic on — lazy dispatch-time compiles
    // (stacked GEMM) must also build f16 for the whole session.
    let pipelines_prefill_f16 = if crate::flags::prefill_f16_enabled() {
        crate::shaders::variant::set_arena_f16_compile(true);
        let out = shared_step_pipelines(&ctx, kv_fmt);
        crate::shaders::variant::set_arena_f16_compile(false);
        Some(out?)
    } else {
        None
    };
    let compile = compile_started.elapsed();

    let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device)?;
    let gpu_blob = std::sync::Arc::clone(&gpu_blob);
    let kv_bytes = kv_cache_bytes(&layout, cfg.max_seq) as usize;
    let logits_bytes = CANVAS * VOCAB * 2;
    let sc_probs_bytes = sc_probs_buffer_bytes();

    log_step_memory_budget(store.blob_bytes(), cfg.max_seq, &layout);

    let sampler = crate::sample::sampler_for_steps(cfg.steps.max(1), cfg.no_early_stop);
    let prefill_len = cfg
        .prefill_token_ids
        .as_ref()
        .map(|t| t.len() as u32)
        .unwrap_or(cfg.kv_len);
    let model_cfg = ModelConfig::load(model_dir)?;
    let eos_token_id = model_cfg.eos_token_id_u32();
    let params = step_params_from_sampler(&sampler, prefill_len, cfg.no_early_stop, eos_token_id);
    let state = init_canvas_state(cfg.seed, VOCAB);
    let (gemm_a_bytes, gemm_b_bytes) = gemm_scratch_bytes();

    let text_config = model_cfg.text_config;
    let weight_store = WeightStore::open(model_dir)?;
    let weight_cache = GpuDecoderWeightCache::load_with_dgq_blob(
        &weight_store,
        &text_config,
        &ctx.device,
        std::sync::Arc::clone(&gpu_blob),
    )?;

    let arena_map = step_arena_layout();
    // KV cache: optionally file-backed (DGQ_KV_MMAP) so dirty pages evict to a
    // real file instead of anonymous swap. `kv_mmap_backing` must outlive
    // `kvcache` — it is declared after it in `StepBuffers` so it drops later.
    let (kvcache, kv_mmap_backing) = alloc_kv_buffer(&ctx.device, kv_bytes)?;
    // E14 f32 side K/V ring for sliding layers (DGQ_PREFILL_KV_F32).
    let mut kv_f32_side_offs = [u64::MAX; N_LAYERS];
    let kv_f32_side = if crate::flags::prefill_kv_f32_enabled() {
        let mut off = 0u64;
        for i in 0..N_LAYERS {
            let l = &layout.layers[i];
            kv_f32_side_offs[i] = off;
            off += (layer_kv_slots(l.is_full != 0, cfg.max_seq)
                * (l.n_kv_heads * l.head_dim) as usize
                * 2
                * 4) as u64;
        }
        if crate::flags::progress_enabled() {
            eprintln!(
                "step-kernel: f32 side KV ring {:.0} MiB (DGQ_PREFILL_KV_F32)",
                off as f64 / (1024.0 * 1024.0)
            );
        }
        Some(alloc_buffer(&ctx.device, off as usize)?)
    } else {
        None
    };
    let bufs = StepBuffers {
        kv_f32_side,
        kv_f32_side_offs,
        blob: gpu_blob.buffer.clone(),
        blob_experts: {
            let (b, _) = gpu_blob.expert_region();
            objc2::rc::Retained::from(b)
        },
        blob_expert_base: gpu_blob.expert_region().1,
        layout: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ModelLayout>())?;
            write_struct(&b, &layout);
            b
        },
        params: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<StepParams>())?;
            write_struct(&b, &params);
            b
        },
        arena: alloc_buffer(&ctx.device, arena_map.bytes() as usize)?,
        kvcache,
        _kv_mmap: kv_mmap_backing,
        state: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<CanvasState>())?;
            write_struct(&b, &state);
            b
        },
        logits: alloc_buffer(&ctx.device, logits_bytes)?,
        sc_probs: alloc_buffer(&ctx.device, sc_probs_bytes)?,
        route: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<RouteScratch>())?;
            zero_buffer(&b);
            b
        },
        dummy_dump: alloc_buffer(&ctx.device, 4)?,
        debug_status: if crate::shaders::variant::runtime_kernel_debug_enabled() {
            Some(alloc_buffer(
                &ctx.device,
                crate::metal::debug_status::DEBUG_STATUS_BYTES,
            )?)
        } else {
            None
        },
        gemm_a: alloc_buffer(&ctx.device, gemm_a_bytes)?,
        gemm_b: alloc_buffer(&ctx.device, gemm_b_bytes)?,
        expert_layer_unique: alloc_buffer(&ctx.device, N_LAYERS * std::mem::size_of::<u32>())?,
        moe_grouped_indirect: alloc_buffer(&ctx.device, MOE_GROUPED_INDIRECT_BYTES)?,
        // {m,l} + O[512] per (16 heads x 256 rows), f32: ~8.4 MiB.
        attn_state: alloc_buffer(
            &ctx.device,
            STEP_NQ_HEADS * CANVAS * (2 + 512) * std::mem::size_of::<f32>(),
        )?,
        // E17 GEMM-attention prefill scratch (opt-in). Head-chunked (E17a): the
        // score matrix S/P holds only HC heads at a time —
        // [HC][CANVAS][n_pad(max_seq)]. Allocated when DGQ_GEMM_ATTN OR
        // DGQ_ATTN_TOPK is set (E20 reuses the same S plane; the two paths are
        // never both live on the same layer).
        attn_gemm_s: if crate::flags::gemm_attn_enabled()
            || crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let np = crate::shaders::attention_gemm::n_pad(cfg.max_seq);
            // Both E17 and E20 encoders batch heads by DGQ_GEMM_ATTN_HC (task
            // #97 unified them); size from the same flag so the encoder can
            // never outrun the scratch.
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * np * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        attn_gemm_p: if crate::flags::gemm_attn_enabled() {
            let np = crate::shaders::attention_gemm::n_pad(cfg.max_seq);
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            // f32-side path stores f32 probs; size for f32 (4B) so either path fits.
            Some(alloc_buffer(&ctx.device, hc * CANVAS * np * 4)?)
        } else {
            None
        },
        attn_gemm_lrow: if crate::flags::gemm_attn_enabled() {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        // E20 top-k scratch: compressed P [HC][CANVAS][K_PAD] (f32), indices
        // Idx [HC][CANVAS][K_PAD] (u32), lrow [HC][CANVAS] (f32), and the u16
        // key plane [HC][CANVAS][n_pad] (FC32 output of QK, read by the
        // selection passes — half the bytes of the S plane). HC comes from the
        // same flag the encoder batches by (task #97). The S plane is shared
        // with attn_gemm_s (allocated above when either flag is set).
        attn_topk_p: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * crate::flags::attn_topk_k_pad() * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        attn_topk_idx: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * crate::flags::attn_topk_k_pad() * std::mem::size_of::<u32>(),
            )?)
        } else {
            None
        },
        attn_topk_lrow: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        attn_topk_pat: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let np = crate::shaders::attention_gemm::n_pad(cfg.max_seq);
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * np * std::mem::size_of::<u16>(),
            )?)
        } else {
            None
        },
        params_sub: alloc_buffer(
            &ctx.device,
            PREFILL_SUBS * std::mem::size_of::<StepParams>(),
        )?,
        arena_map,
        arena_layout_buf: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ArenaLayout>())?;
            write_struct(&b, &arena_map);
            b
        },
    };
    zero_buffer(&bufs.expert_layer_unique);
    zero_buffer(&bufs.moe_grouped_indirect);
    zero_buffer(&bufs.arena);
    zero_buffer(&bufs.kvcache);
    zero_buffer(&bufs.logits);

    // Fast-prefill (DGQ_FAST_PREFILL) runs on the step kernels AFTER `rt` is built
    // (prefill_chunks is a StepRuntime method); the slow f32-engine prefill runs
    // here at open time otherwise.
    if let Some(ref token_ids) = cfg.prefill_token_ids
        && !should_fast_prefill(token_ids.len())
    {
        let mut encoder = crate::metal::step_kv::MonolithicEncoderCache::open_opt(
            model_dir,
            CANVAS,
            cfg.max_seq,
            Some(std::sync::Arc::clone(&gpu_blob)),
        )?;
        let (kv_len, _) = crate::metal::step_kv::prefill_monolithic_kv_with_cache(
            &mut encoder,
            token_ids,
            &bufs.kvcache,
            &layout,
            cfg.max_seq,
            layers,
        )?;
        if kv_len as u32 != prefill_len {
            return Err(Error::Runtime("prefill kv_len mismatch"));
        }
        eprintln!("step-kernel: prefilled kv_len={kv_len} tokens");
    }

    let build = StepRuntimeBuildTiming {
        compile,
        total: build_started.elapsed(),
    };
    if crate::flags::progress_enabled() {
        eprintln!(
            "step-kernel: runtime built (total={:.2?}, compile={:.2?})",
            build.total, build.compile
        );
    }
    let mut rt = StepRuntime {
        _mem_permit: mem_permit,
        ctx,
        pipelines,
        pipelines_prefill_f16,
        arena_f16_mode: false,
        kv_f32_side_valid: 0,
        bufs,
        gpu_blob,
        weight_cache,
        text_config,
        block_profile,
        attn_ffn_q8,
        attn_ffn_bf16,
        embed_bf16,
        layout,
        tensor_offsets: offsets,
        layers,
        max_seq: cfg.max_seq,
        active_canvas: CANVAS,
    };
    if crate::flags::progress_enabled() {
        if rt.embed_bf16 && sc_sparse_enabled() {
            eprintln!(
                "step-kernel: sparse SC softembed (DGQ_SC_SPARSE=0 for the exact chunked path)"
            );
        } else {
            eprintln!("step-kernel: chunked SC softembed");
        }
    }
    if let Some(ref token_ids) = cfg.prefill_token_ids
        && should_fast_prefill(token_ids.len())
    {
        let started = Instant::now();
        let kv_len = rt.prefill_chunks(token_ids)?;
        if kv_len as u32 != prefill_len {
            return Err(Error::Runtime("fast-prefill kv_len mismatch"));
        }
        if crate::flags::progress_enabled() {
            eprintln!(
                "step-kernel: fast-prefilled kv_len={kv_len} tokens ({:.2?})",
                started.elapsed()
            );
        }
    }
    if let Some(path) = crate::flags::dump_kv_path() {
        dump_buffer_raw(&rt.bufs.kvcache, &path);
    }
    Ok((rt, build))
}
