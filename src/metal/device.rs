use crate::Error;
use crate::metal::pipeline_cache::PipelineArchiveCache;
use crate::shaders::variant::KernelVariant;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};

/// Caller-chosen function-constant axes for a GEMM subkernel compile. The rest
/// (shape_assert / debug_fast / debug_deep / arena_f16 / n_tile) come from the
/// runtime [`KernelVariant`]. Named fields replace what was a trailing run of
/// four positional bools on `compile_gemm_subkernel_on_device`.
#[derive(Clone, Copy)]
pub(crate) struct GemmCompileConfig {
    pub gemm_n: u32,
    pub gemm_k: u32,
    pub is_full_layer: bool,
    pub quant_format: u32,
    /// FC10: f16 activation (A) input.
    pub x_fp16: bool,
    /// FC28 GATHER_A: fused-MoE gate_up gathers bf16 `moein` rows in the A-load.
    pub gather_a: bool,
    /// FC29: force bf16 output (lm_head logits — f16 acts, bf16-range logits).
    pub out_bf16: bool,
    /// FC30 K_ROWK_OUT_ARENA: arena-overwrite output mode of `gemm_rowk`.
    pub out_arena: bool,
    /// Hash of the compiled SOURCE. The tunable tile geometry (TUNE_BM/TUNE_BN)
    /// is baked into the source `#define` prepend, NOT a function constant, so
    /// it does NOT appear in the FC-derived label. Without this, two different
    /// tiles at the same (n, k, format) produce the SAME cache label and the
    /// PipelineArchiveCache silently returns the first-compiled pipeline — a
    /// bm=32 kernel fed 64-row blocks then zeroes rows 32..63 (found via the
    /// sparse_block_m_invariant oracle). Folding the source hash into the label
    /// makes any source-define difference cache-distinct.
    pub src_hash: u64,
}

/// Stable (no-random-seed) hash of a shader source for cache-label disambiguation.
pub(crate) fn source_hash(source: &str) -> u64 {
    // FNV-1a: deterministic across runs (std DefaultHasher is randomized).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in source.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl GemmCompileConfig {
    /// Default raw path (no gather / bf16-out / arena override).
    pub fn raw(
        gemm_n: u32,
        gemm_k: u32,
        is_full_layer: bool,
        quant_format: u32,
        x_fp16: bool,
    ) -> Self {
        Self {
            gemm_n,
            gemm_k,
            is_full_layer,
            quant_format,
            x_fp16,
            gather_a: false,
            out_bf16: false,
            out_arena: false,
            src_hash: 0,
        }
    }

    /// FC29 bf16 output (lm_head logits).
    pub fn out_bf16(gemm_n: u32, gemm_k: u32, quant_format: u32) -> Self {
        Self {
            out_bf16: true,
            ..Self::raw(gemm_n, gemm_k, false, quant_format, false)
        }
    }

    /// FC28 GATHER_A (fused-MoE gate_up).
    pub fn gather(gemm_n: u32, gemm_k: u32, quant_format: u32) -> Self {
        Self {
            gather_a: true,
            ..Self::raw(gemm_n, gemm_k, false, quant_format, false)
        }
    }

    /// FC30 arena-overwrite output (`gemm_rowk` SC softembed).
    pub fn rowk_arena(gemm_n: u32, gemm_k: u32, quant_format: u32, x_fp16: bool) -> Self {
        Self {
            out_arena: true,
            ..Self::raw(gemm_n, gemm_k, false, quant_format, x_fp16)
        }
    }
}

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Gpu("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Gpu("failed to create Metal command queue"))?;
        // Record the working-set cap so the q8-KV auto policy (flags::kv_q8)
        // can scale to this device's RAM.
        crate::flags::set_gpu_working_set_cap(device.recommendedMaxWorkingSetSize());
        Ok(Self { device, queue })
    }

    pub fn compile_library(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Error> {
        // Quoted #include expansion happens here (src/shaders/common/expand.rs).
        let ns_source = NSString::from_str(&crate::shaders::expand::expand(source));
        self.device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| shader_compile_error(e))
    }

    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        self.compile_kernel_from_library(&library, entry)
    }

    pub fn compile_kernel_from_library(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
    ) -> Result<ComputePipeline, Error> {
        Self::compile_kernel_from_library_on_device(&self.device, library, entry)
    }

    pub fn compile_kernel_from_library_on_device(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
    ) -> Result<ComputePipeline, Error> {
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName(&name)
            .ok_or(Error::Runtime("Metal kernel not found"))?;
        let cache = PipelineArchiveCache::shared(device)?;
        let pipeline = cache.compile_compute(device, &function, entry)?;
        Ok(ComputePipeline { pipeline })
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
        let library = self.compile_library(source)?;
        let mut cfg = GemmCompileConfig::raw(gemm_n, gemm_k, is_full_layer, quant_format, x_fp16);
        cfg.src_hash = source_hash(source);
        Self::compile_gemm_subkernel_on_device(&self.device, &library, entry, &cfg)
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
        let library = self.compile_library(source)?;
        let mut cfg = GemmCompileConfig::out_bf16(gemm_n, gemm_k, quant_format);
        cfg.src_hash = source_hash(source);
        Self::compile_gemm_subkernel_on_device(&self.device, &library, entry, &cfg)
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
        let library = self.compile_library(source)?;
        let mut cfg = GemmCompileConfig::gather(gemm_n, gemm_k, quant_format);
        cfg.src_hash = source_hash(source);
        Self::compile_gemm_subkernel_on_device(&self.device, &library, entry, &cfg)
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
        let library = self.compile_library(source)?;
        let mut cfg = GemmCompileConfig::rowk_arena(gemm_n, gemm_k, quant_format, x_fp16);
        cfg.src_hash = source_hash(source);
        Self::compile_gemm_subkernel_on_device(&self.device, &library, entry, &cfg)
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
        let library = self.compile_library(source)?;
        Self::compile_gemm_stacked_subkernel_on_device(
            &self.device,
            &library,
            entry,
            gemm_n,
            gemm_k,
            quant_format,
            stacked,
            source_hash(source),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_gemm_stacked_subkernel_on_device(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
        stacked: &crate::shaders::gemm_block_stacked::StackedSegFc,
        src_hash: u64,
    ) -> Result<ComputePipeline, Error> {
        let variant = crate::shaders::variant::runtime_step_variant();
        let fc = MTLFunctionConstantValues::new();
        let shape_assert = variant.shape_assert;
        let dump_stage = 0u32;
        let debug_fast = variant.debug_fast;
        let debug_deep = variant.debug_deep;
        let arena_f16 = variant.arena_f16;
        let gemm_n_tile = crate::shaders::gemm_common::n_tile() as u32;
        let is_full_layer = false;
        let x_fp16 = false;
        unsafe {
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&shape_assert).cast(),
                MTLDataType::Bool,
                1,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&dump_stage).cast(),
                MTLDataType::UInt,
                2,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&quant_format).cast(),
                MTLDataType::UInt,
                3,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&debug_fast).cast(),
                MTLDataType::Bool,
                7,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&debug_deep).cast(),
                MTLDataType::Bool,
                8,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&arena_f16).cast(),
                MTLDataType::Bool,
                9,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&is_full_layer).cast(),
                MTLDataType::Bool,
                4,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&gemm_n).cast(),
                MTLDataType::UInt,
                5,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&gemm_k).cast(),
                MTLDataType::UInt,
                6,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&x_fp16).cast(),
                MTLDataType::Bool,
                10,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&gemm_n_tile).cast(),
                MTLDataType::UInt,
                11,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&stacked.n_segs).cast(),
                MTLDataType::UInt,
                12,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&stacked.end0).cast(),
                MTLDataType::UInt,
                13,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&stacked.end1).cast(),
                MTLDataType::UInt,
                14,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&stacked.end2).cast(),
                MTLDataType::UInt,
                15,
            );
            for (i, w_off) in stacked.w_off.iter().enumerate() {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(w_off).cast(),
                    MTLDataType::ULong,
                    16 + i,
                );
            }
            for (i, y_off) in stacked.y_byte_off.iter().enumerate() {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(y_off).cast(),
                    MTLDataType::ULong,
                    19 + i,
                );
            }
            for (i, col0) in stacked.y_col0.iter().enumerate() {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(col0).cast(),
                    MTLDataType::UInt,
                    22 + i,
                );
            }
            for (i, row_cols) in stacked.y_row_cols.iter().enumerate() {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(row_cols).cast(),
                    MTLDataType::UInt,
                    25 + i,
                );
            }
        }
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &fc)
            .map_err(|e| shader_compile_error(e))?;
        let label = format!(
            "{entry}_qf{quant_format}_n{gemm_n}_k{gemm_k}_nt{gemm_n_tile}_af{}_ns{}_e{}_{}_{}_w{}_{}_{}_y{}_{}_{}_s{src_hash:x}",
            u8::from(arena_f16),
            stacked.n_segs,
            stacked.end0,
            stacked.end1,
            stacked.end2,
            stacked.w_off[0],
            stacked.w_off[1],
            stacked.w_off[2],
            stacked.y_byte_off[0],
            stacked.y_byte_off[1],
            stacked.y_byte_off[2],
        );
        let cache = PipelineArchiveCache::shared(device)?;
        let pipeline = cache.compile_compute(device, &function, &label)?;
        Ok(ComputePipeline { pipeline })
    }

    fn compile_gemm_subkernel_on_device(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
        cfg: &GemmCompileConfig,
    ) -> Result<ComputePipeline, Error> {
        let GemmCompileConfig {
            gemm_n,
            gemm_k,
            is_full_layer,
            quant_format,
            x_fp16,
            gather_a,
            out_bf16,
            out_arena,
            src_hash,
        } = *cfg;
        let variant = crate::shaders::variant::runtime_step_variant();
        let fc = MTLFunctionConstantValues::new();
        let shape_assert = variant.shape_assert;
        let dump_stage = 0u32;
        let debug_fast = variant.debug_fast;
        let debug_deep = variant.debug_deep;
        let arena_f16 = variant.arena_f16;
        let gemm_n_tile = crate::shaders::gemm_common::n_tile() as u32;
        unsafe {
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&shape_assert).cast(),
                MTLDataType::Bool,
                1,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&dump_stage).cast(),
                MTLDataType::UInt,
                2,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&quant_format).cast(),
                MTLDataType::UInt,
                3,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&debug_fast).cast(),
                MTLDataType::Bool,
                7,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&debug_deep).cast(),
                MTLDataType::Bool,
                8,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&arena_f16).cast(),
                MTLDataType::Bool,
                9,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&is_full_layer).cast(),
                MTLDataType::Bool,
                4,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&gemm_n).cast(),
                MTLDataType::UInt,
                5,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&gemm_k).cast(),
                MTLDataType::UInt,
                6,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&x_fp16).cast(),
                MTLDataType::Bool,
                10,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&gemm_n_tile).cast(),
                MTLDataType::UInt,
                11,
            );
            if gather_a {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(&gather_a).cast(),
                    MTLDataType::Bool,
                    28,
                );
            }
            if out_bf16 {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(&out_bf16).cast(),
                    MTLDataType::Bool,
                    29,
                );
            }
            if out_arena {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(&out_arena).cast(),
                    MTLDataType::Bool,
                    30,
                );
            }
        }
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &fc)
            .map_err(|e| shader_compile_error(e))?;
        let label = format!(
            "{entry}_qf{quant_format}_n{gemm_n}_k{gemm_k}_nt{gemm_n_tile}_xfp16{}_sa{}_df{}_dd{}_g{}_o{}_oa{}_af{}_s{src_hash:x}",
            u8::from(x_fp16),
            u8::from(shape_assert),
            u8::from(debug_fast),
            u8::from(debug_deep),
            u8::from(gather_a),
            u8::from(out_bf16),
            u8::from(out_arena),
            u8::from(arena_f16),
        );
        let cache = PipelineArchiveCache::shared(device)?;
        let pipeline = cache.compile_compute(device, &function, &label)?;
        Ok(ComputePipeline { pipeline })
    }

    pub fn compile_kernels(
        &self,
        source: &str,
        entries: &[&str],
    ) -> Result<Vec<ComputePipeline>, Error> {
        let library = self.compile_library(source)?;
        let mut pipelines = Vec::with_capacity(entries.len());
        for entry in entries {
            pipelines.push(self.compile_kernel_from_library(&library, entry)?);
        }
        Ok(pipelines)
    }

    /// Specialize an isolated subkernel (function constants 1–3 per AGENTS.md §4).
    pub fn compile_subkernel(
        &self,
        source: &str,
        entry: &str,
        variant: KernelVariant,
    ) -> Result<ComputePipeline, Error> {
        Self::compile_subkernel_on_device(&self.device, source, entry, variant, "", &[], &[])
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
        Self::compile_subkernel_on_device(
            &self.device,
            source,
            entry,
            variant,
            extra_label,
            extra_bools,
            extra_uints,
        )
    }

    pub fn compile_subkernel_on_device(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        entry: &str,
        variant: KernelVariant,
        extra_label: &str,
        extra_bools: &[crate::shaders::variant::FcBool],
        extra_uints: &[crate::shaders::variant::FcUInt],
    ) -> Result<ComputePipeline, Error> {
        let library = {
            // Quoted #include expansion happens here (src/shaders/common/expand.rs).
            let ns_source = NSString::from_str(&crate::shaders::expand::expand(source));
            device
                .newLibraryWithSource_options_error(&ns_source, None)
                .map_err(|e| shader_compile_error(e))?
        };
        let fc = MTLFunctionConstantValues::new();
        let shape_assert = variant.shape_assert;
        let dump_stage = variant.dump_stage;
        let quant_format = variant.quant_format as u32;
        let debug_fast = variant.debug_fast || variant.shape_assert;
        let debug_deep = variant.debug_deep;
        let arena_f16 = variant.arena_f16;
        unsafe {
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&shape_assert).cast(),
                MTLDataType::Bool,
                1,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&dump_stage).cast(),
                MTLDataType::UInt,
                2,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&quant_format).cast(),
                MTLDataType::UInt,
                3,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&debug_fast).cast(),
                MTLDataType::Bool,
                7,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&debug_deep).cast(),
                MTLDataType::Bool,
                8,
            );
            fc.setConstantValue_type_atIndex(
                std::ptr::NonNull::from_ref(&arena_f16).cast(),
                MTLDataType::Bool,
                9,
            );
            for extra in extra_bools {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(&extra.value).cast(),
                    MTLDataType::Bool,
                    extra.index as usize,
                );
            }
            for extra in extra_uints {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(&extra.value).cast(),
                    MTLDataType::UInt,
                    extra.index as usize,
                );
            }
        }
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &fc)
            .map_err(|e| shader_compile_error(e))?;
        let label = variant.cache_label_extra(entry, extra_label);
        let cache = PipelineArchiveCache::shared(device)?;
        let pipeline = cache.compile_compute(device, &function, &label)?;
        Ok(ComputePipeline { pipeline })
    }
}

pub struct ComputePipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Clone for ComputePipeline {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
        }
    }
}

fn shader_compile_error(err: Retained<NSError>) -> Error {
    Error::NotFound(format!(
        "Metal shader compile failed: {}",
        err.localizedDescription()
    ))
}
