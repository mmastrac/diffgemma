//! Diffgemma's policy layer over the gpukit Metal context: flag-driven cache
//! configuration, the include table, and the mapping from [`KernelVariant`] /
//! GEMM axes onto function-constant values. The mechanism (compilation,
//! specialization, cache-label derivation) lives in `gpukit`.

use crate::Error;
use crate::shaders::variant::KernelVariant;
use gpukit::metal as gk;
use objc2_metal::MTLDevice;

pub use gpukit::metal::ComputePipeline;

/// Part of the archive file key; a change here orphans every on-disk
/// archive. For label-scheme or FC-numbering changes the tree hash cannot
/// see.
const CACHE_BUNDLE_TAG: &str = "diffgemma-v10-gpukit";

fn cache_config() -> gk::CacheConfig {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // Whole-shader-tree hash from build.rs; a kernel edit changes the archive
    // file name, so an edited kernel can never be served stale.
    env!("DGQ_SHADER_TREE_HASH").hash(&mut h);
    CACHE_BUNDLE_TAG.hash(&mut h);
    gk::CacheConfig {
        enabled: crate::flags::metal_pipeline_cache_enabled(),
        dir: crate::flags::metal_pipeline_cache_dir_override(),
        namespace: "diffgemma",
        key: h.finish(),
        verbose: crate::flags::progress_enabled(),
    }
}

/// Caller-chosen function-constant axes for a GEMM subkernel compile. The
/// rest (shape_assert / debug_fast / debug_deep / arena_f16 / n_tile) come
/// from the runtime [`KernelVariant`].
#[derive(Clone, Copy)]
struct GemmCompileConfig {
    gemm_n: u32,
    gemm_k: u32,
    is_full_layer: bool,
    quant_format: u32,
    /// FC10: f16 activation (A) input.
    x_fp16: bool,
    /// FC28 GATHER_A: fused-MoE gate_up gathers bf16 `moein` rows in the A-load.
    gather_a: bool,
    /// FC29: force bf16 output (lm_head logits — f16 acts, bf16-range logits).
    out_bf16: bool,
    /// FC30 K_ROWK_OUT_ARENA: arena-overwrite output mode of `gemm_rowk`.
    out_arena: bool,
}

impl GemmCompileConfig {
    fn raw(gemm_n: u32, gemm_k: u32, is_full_layer: bool, quant_format: u32, x_fp16: bool) -> Self {
        Self {
            gemm_n,
            gemm_k,
            is_full_layer,
            quant_format,
            x_fp16,
            gather_a: false,
            out_bf16: false,
            out_arena: false,
        }
    }

    /// Base GEMM axes (FC1–11) plus the conditional output-mode axes. The
    /// conditional constants are set only when true: their absence is what
    /// `is_function_constant_defined` guards test in the shaders.
    fn fc_values(&self) -> gk::FcValues {
        let variant = crate::shaders::variant::runtime_step_variant();
        let mut fc = gk::FcValues::new();
        fc.set_bool(1, variant.shape_assert)
            .set_uint(2, 0)
            .set_uint(3, self.quant_format)
            .set_bool(7, variant.debug_fast)
            .set_bool(8, variant.debug_deep)
            .set_bool(9, variant.arena_f16)
            .set_bool(4, self.is_full_layer)
            .set_uint(5, self.gemm_n)
            .set_uint(6, self.gemm_k)
            .set_bool(10, self.x_fp16)
            .set_uint(11, crate::shaders::gemm_common::n_tile() as u32);
        if self.gather_a {
            fc.set_bool(28, true);
        }
        if self.out_bf16 {
            fc.set_bool(29, true);
        }
        if self.out_arena {
            fc.set_bool(30, true);
        }
        fc
    }
}

/// A gpukit Metal context opened with diffgemma's configuration. Derefs to
/// [`gk::Context`], so mechanism-level methods (`compile_kernel`,
/// `compile_library`, …) are called directly on it.
pub struct MetalContext {
    inner: gk::Context,
}

impl std::ops::Deref for MetalContext {
    type Target = gk::Context;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl MetalContext {
    pub fn new() -> Result<Self, Error> {
        let inner = gk::Context::new(cache_config())?;
        // Record the working-set cap so the q8-KV auto policy (flags::kv_q8)
        // can scale to this device's RAM.
        crate::flags::set_gpu_working_set_cap(inner.device.recommendedMaxWorkingSetSize());
        Ok(Self { inner })
    }

    /// Specialize an isolated subkernel (function constants 1–3 per AGENTS.md §4).
    pub fn compile_subkernel(
        &self,
        source: &str,
        entry: &str,
        variant: KernelVariant,
    ) -> Result<ComputePipeline, Error> {
        self.compile_subkernel_ex(source, entry, variant, "", &[], &[])
    }

    pub fn compile_subkernel_ex(
        &self,
        source: &str,
        entry: &str,
        variant: KernelVariant,
        extra_label: &str,
        extra_bools: &[crate::shaders::variant::FcBool],
        extra_uints: &[crate::shaders::variant::FcUInt],
    ) -> Result<ComputePipeline, Error> {
        self.compile_subkernel_ex_floats(
            source,
            entry,
            variant,
            extra_label,
            extra_bools,
            extra_uints,
            &[],
        )
    }

    /// As `compile_subkernel_ex` with float axes (e.g. RoPE theta).
    #[allow(clippy::too_many_arguments)]
    pub fn compile_subkernel_ex_floats(
        &self,
        source: &str,
        entry: &str,
        variant: KernelVariant,
        extra_label: &str,
        extra_bools: &[crate::shaders::variant::FcBool],
        extra_uints: &[crate::shaders::variant::FcUInt],
        extra_floats: &[crate::shaders::variant::FcFloat],
    ) -> Result<ComputePipeline, Error> {
        let mut fc = gk::FcValues::new();
        fc.set_bool(1, variant.shape_assert)
            .set_uint(2, variant.dump_stage)
            .set_uint(3, variant.quant_format as u32)
            .set_bool(7, variant.debug_fast || variant.shape_assert)
            .set_bool(8, variant.debug_deep)
            .set_bool(9, variant.arena_f16);
        for extra in extra_bools {
            fc.set_bool(extra.index, extra.value);
        }
        for extra in extra_uints {
            fc.set_uint(extra.index, extra.value);
        }
        for extra in extra_floats {
            fc.set_float(extra.index, extra.value);
        }
        let label = variant.cache_label_extra(entry, extra_label);
        Ok(self.inner.compile_specialized(source, entry, &fc, &label)?)
    }

    /// Specialize a tiled quant GEMM subkernel (FC1–3 global, FC4–6 shape/format, FC9 I/O layout).
    pub fn compile_gemm_subkernel(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        is_full_layer: bool,
        quant_format: u32,
        x_fp16: bool,
    ) -> Result<ComputePipeline, Error> {
        let cfg = GemmCompileConfig::raw(gemm_n, gemm_k, is_full_layer, quant_format, x_fp16);
        self.compile_gemm(source, entry, &cfg)
    }

    /// As `compile_gemm_subkernel` but forces bf16 output (FC29) — lm_head logits
    /// (f16 activation input, bf16-range logits output).
    pub fn compile_gemm_subkernel_out_bf16(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
    ) -> Result<ComputePipeline, Error> {
        let cfg = GemmCompileConfig {
            out_bf16: true,
            ..GemmCompileConfig::raw(gemm_n, gemm_k, false, quant_format, false)
        };
        self.compile_gemm(source, entry, &cfg)
    }

    /// As `compile_gemm_subkernel` but sets GATHER_A (FC28) — the fused-MoE
    /// gate_up path that gathers bf16 `moein` rows in the A-load (buffer 7).
    pub fn compile_gemm_subkernel_gather(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
    ) -> Result<ComputePipeline, Error> {
        let cfg = GemmCompileConfig {
            gather_a: true,
            ..GemmCompileConfig::raw(gemm_n, gemm_k, false, quant_format, false)
        };
        self.compile_gemm(source, entry, &cfg)
    }

    /// As `compile_gemm_subkernel` but sets K_ROWK_OUT_ARENA (FC30) — the
    /// arena-overwrite output mode of `gemm_rowk` (SC softembed); accumulate
    /// (f32, tied-embed lm_head) is the FC30-unset default.
    pub fn compile_gemm_subkernel_rowk_arena(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
        x_fp16: bool,
    ) -> Result<ComputePipeline, Error> {
        let cfg = GemmCompileConfig {
            out_arena: true,
            ..GemmCompileConfig::raw(gemm_n, gemm_k, false, quant_format, x_fp16)
        };
        self.compile_gemm(source, entry, &cfg)
    }

    fn compile_gemm(
        &self,
        source: &str,
        entry: &str,
        cfg: &GemmCompileConfig,
    ) -> Result<ComputePipeline, Error> {
        Ok(self
            .inner
            .compile_specialized(source, entry, &cfg.fc_values(), entry)?)
    }

    /// Specialize stacked GEMM: base FC1–11 plus segment table FC12–27.
    pub fn compile_gemm_stacked_subkernel(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
        stacked: &crate::shaders::gemm_block_stacked::StackedSegFc,
    ) -> Result<ComputePipeline, Error> {
        let cfg = GemmCompileConfig::raw(gemm_n, gemm_k, false, quant_format, false);
        let mut fc = cfg.fc_values();
        fc.set_uint(12, stacked.n_segs)
            .set_uint(13, stacked.end0)
            .set_uint(14, stacked.end1)
            .set_uint(15, stacked.end2);
        for (i, w_off) in stacked.w_off.iter().enumerate() {
            fc.set_ulong(16 + i as u32, *w_off);
        }
        for (i, y_off) in stacked.y_byte_off.iter().enumerate() {
            fc.set_ulong(19 + i as u32, *y_off);
        }
        for (i, col0) in stacked.y_col0.iter().enumerate() {
            fc.set_uint(22 + i as u32, *col0);
        }
        for (i, row_cols) in stacked.y_row_cols.iter().enumerate() {
            fc.set_uint(25 + i as u32, *row_cols);
        }
        Ok(self.inner.compile_specialized(source, entry, &fc, entry)?)
    }
}
