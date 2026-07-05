use crate::kernels::sub::variant::KernelVariant;
use crate::metal::pipeline_cache::PipelineArchiveCache;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType,
    MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Format("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Format("failed to create Metal command queue"))?;
        Ok(Self { device, queue })
    }

    pub fn compile_library(&self, source: &str) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Error> {
        let ns_source = NSString::from_str(source);
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
            .ok_or(Error::Format("Metal kernel not found"))?;
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
        Self::compile_gemm_subkernel_on_device(
            &self.device,
            &library,
            entry,
            gemm_n,
            gemm_k,
            is_full_layer,
            quant_format,
            x_fp16,
            false,
            false,
            false,
        )
    }

    /// As `compile_gemm_subkernel` but sets GEMM_M_ADAPT (FC32) — the
    /// block-sparse MoE pipeline picks the smallest simdgroup M-mapping
    /// (8/16/32 rows) per block at runtime (DGQ_MOE_TILE_ADAPT).
    pub fn compile_gemm_subkernel_adaptive(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
    ) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        Self::compile_gemm_subkernel_on_device(
            &self.device,
            &library,
            entry,
            gemm_n,
            gemm_k,
            false,
            quant_format,
            false,
            false,
            false,
            true,
        )
    }

    /// GATHER_A + GEMM_M_ADAPT combined (adaptive fused-gather MoE gate_up).
    pub fn compile_gemm_subkernel_gather_adaptive(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
    ) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        Self::compile_gemm_subkernel_on_device(
            &self.device,
            &library,
            entry,
            gemm_n,
            gemm_k,
            false,
            quant_format,
            false,
            true,
            false,
            true,
        )
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
        Self::compile_gemm_subkernel_on_device(
            &self.device,
            &library,
            entry,
            gemm_n,
            gemm_k,
            false,
            quant_format,
            false,
            false,
            true,
            false,
        )
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
        Self::compile_gemm_subkernel_on_device(
            &self.device,
            &library,
            entry,
            gemm_n,
            gemm_k,
            false,
            quant_format,
            false,
            true,
            false,
            false,
        )
    }

    /// Specialize stacked GEMM: base FC1–11 plus segment table FC12–27.
    pub fn compile_gemm_stacked_subkernel(
        &self,
        source: &str,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
        stacked: &crate::kernels::sub::gemm_block_stacked::StackedSegFc,
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
        )
    }

    fn compile_gemm_stacked_subkernel_on_device(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
        gemm_n: u32,
        gemm_k: u32,
        quant_format: u32,
        stacked: &crate::kernels::sub::gemm_block_stacked::StackedSegFc,
    ) -> Result<ComputePipeline, Error> {
        let variant = crate::kernels::sub::variant::runtime_step_variant();
        let fc = MTLFunctionConstantValues::new();
        let shape_assert = variant.shape_assert;
        let dump_stage = 0u32;
        let debug_fast = variant.debug_fast;
        let debug_deep = variant.debug_deep;
        let gemm_n_tile = crate::kernels::sub::gemm_common::n_tile() as u32;
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
            "{entry}_qf{quant_format}_n{gemm_n}_k{gemm_k}_nt{gemm_n_tile}_ns{}_e{}_{}_{}_w{}_{}_{}_y{}_{}_{}",
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
        gemm_n: u32,
        gemm_k: u32,
        is_full_layer: bool,
        quant_format: u32,
        x_fp16: bool,
        gather_a: bool,
        out_bf16: bool,
        m_adapt: bool,
    ) -> Result<ComputePipeline, Error> {
        let variant = crate::kernels::sub::variant::runtime_step_variant();
        let fc = MTLFunctionConstantValues::new();
        let shape_assert = variant.shape_assert;
        let dump_stage = 0u32;
        let debug_fast = variant.debug_fast;
        let debug_deep = variant.debug_deep;
        let gemm_n_tile = crate::kernels::sub::gemm_common::n_tile() as u32;
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
            if m_adapt {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(&m_adapt).cast(),
                    MTLDataType::Bool,
                    32,
                );
            }
        }
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &fc)
            .map_err(|e| shader_compile_error(e))?;
        let label = format!(
            "{entry}_qf{quant_format}_n{gemm_n}_k{gemm_k}_nt{gemm_n_tile}_xfp16{}_sa{}_df{}_dd{}_g{}_o{}_ad{}",
            u8::from(x_fp16),
            u8::from(shape_assert),
            u8::from(debug_fast),
            u8::from(debug_deep),
            u8::from(gather_a),
            u8::from(out_bf16),
            u8::from(m_adapt),
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

    /// Specialize an isolated subkernel (function constants 1–3 per STRATEGY.md §4).
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
        extra_bools: &[crate::kernels::sub::variant::FcBool],
        extra_uints: &[crate::kernels::sub::variant::FcUInt],
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
        extra_bools: &[crate::kernels::sub::variant::FcBool],
        extra_uints: &[crate::kernels::sub::variant::FcUInt],
    ) -> Result<ComputePipeline, Error> {
        let library = {
            let ns_source = NSString::from_str(source);
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
