//! Compile-time subkernel variant tuple (AGENTS.md §4).

use std::sync::atomic::{AtomicBool, Ordering};

static RUNTIME_ASSERT: AtomicBool = AtomicBool::new(false);
static RUNTIME_DEBUG_DEEP: AtomicBool = AtomicBool::new(false);
static ARENA_F16_COMPILE: AtomicBool = AtomicBool::new(false);

/// Compile-mode switch: while true, `runtime_step_variant()` yields variants
/// with `arena_f16` set, so a whole pipeline set (incl. the internal-variant
/// GEMM compile paths) builds against fp16 activation-arena slots (FC9).
/// Bracket the build of the E11 prefill pipeline set with this.
pub fn set_arena_f16_compile(enabled: bool) {
    ARENA_F16_COMPILE.store(enabled, Ordering::Relaxed);
}

pub fn arena_f16_compile_enabled() -> bool {
    ARENA_F16_COMPILE.load(Ordering::Relaxed)
}

/// Enable GPU fast/shape asserts for step pipelines (`--assert` on generate-monolithic).
pub fn set_runtime_assert_enabled(enabled: bool) {
    RUNTIME_ASSERT.store(enabled, Ordering::Relaxed);
}

pub fn runtime_assert_enabled() -> bool {
    RUNTIME_ASSERT.load(Ordering::Relaxed)
}

/// Enable expensive NaN/Inf scans (`--debug-deep`, typically with `--assert`).
pub fn set_runtime_debug_deep_enabled(enabled: bool) {
    RUNTIME_DEBUG_DEEP.store(enabled, Ordering::Relaxed);
}

pub fn runtime_debug_deep_enabled() -> bool {
    RUNTIME_DEBUG_DEEP.load(Ordering::Relaxed)
}

/// Either fast asserts or deep finite scans (debug-status buffer + pipeline variant).
pub fn runtime_kernel_debug_enabled() -> bool {
    runtime_assert_enabled() || runtime_debug_deep_enabled()
}

/// Configure both debug flags (call before first step pipeline compile).
pub fn set_runtime_kernel_debug(assert: bool, deep: bool) {
    set_runtime_assert_enabled(assert);
    set_runtime_debug_deep_enabled(deep);
}

/// Subkernel variant for monolithic step pipelines.
pub fn runtime_step_variant() -> KernelVariant {
    let assert = runtime_assert_enabled();
    let deep = runtime_debug_deep_enabled();
    KernelVariant {
        shape_assert: assert,
        debug_fast: assert,
        debug_deep: deep,
        dump_stage: 0,
        quant_format: QuantFormat::INERT,
        arena_f16: arena_f16_compile_enabled(),
    }
}

/// Quantized weight format axis (FC3). One uint, no invalid bool combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantFormat {
    Q4Affine = 0,
    Q8 = 1,
    MxFp4 = 2,
    NvFp4 = 3,
    /// bf16 weights, no dequant (Raw .dgq path; unifies gemm_bf16 into gemm_block).
    Raw = 4,
    /// Affine int6 groups of 32 (`q6_block`, experts).
    Q6 = 5,
}

impl QuantFormat {
    pub const INERT: Self = Self::Q4Affine;
}

/// Element dtype axis (FC4+ on kernels that use K_ELEM_DTYPE / K_IO_DTYPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ElemDtype {
    F32 = 0,
    Half = 1,
}

/// Function-constant specialization for one logical kernel body (FC 1–3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelVariant {
    /// Extra GPU-side bounds checks (tier-2 debug).
    pub shape_assert: bool,
    /// Cheap shape/dim/index asserts (P3.7 FC7).
    pub debug_fast: bool,
    /// Expensive NaN/Inf scans (P3.7 FC8).
    pub debug_deep: bool,
    /// 0 = off; >0 writes chosen intermediates to optional dump buffer.
    pub dump_stage: u32,
    /// Quant format selector (FC3). Inert on elementwise bodies (must be Q4Affine).
    pub quant_format: QuantFormat,
    /// Activation-arena slots are fp16 instead of bf16 (FC9; E11 prefill stream).
    pub arena_f16: bool,
}

impl KernelVariant {
    pub const PRODUCTION: Self = Self {
        shape_assert: false,
        debug_fast: false,
        debug_deep: false,
        dump_stage: 0,
        quant_format: QuantFormat::INERT,
        arena_f16: false,
    };

    pub const TEST_ASSERT: Self = Self {
        shape_assert: true,
        debug_fast: true,
        debug_deep: false,
        dump_stage: 0,
        quant_format: QuantFormat::INERT,
        arena_f16: false,
    };

    pub const TEST_DUMP: Self = Self {
        shape_assert: true,
        debug_fast: true,
        debug_deep: false,
        dump_stage: 1,
        quant_format: QuantFormat::INERT,
        arena_f16: false,
    };

    pub fn cache_label(&self, entry: &str) -> String {
        format!(
            "{entry}_sa{}_df{}_dd{}_d{}_qf{}_af{}",
            u8::from(self.shape_assert),
            u8::from(self.debug_fast),
            u8::from(self.debug_deep),
            self.dump_stage,
            self.quant_format as u32,
            u8::from(self.arena_f16),
        )
    }

    pub fn cache_label_extra(&self, entry: &str, extra: &str) -> String {
        if extra.is_empty() {
            self.cache_label(entry)
        } else {
            format!("{}{}", self.cache_label(entry), extra)
        }
    }
}

/// Optional bool function constant beyond the shared triple (index ≥ 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FcBool {
    pub index: u32,
    pub value: bool,
}

/// Optional uint function constant beyond the shared triple (index ≥ 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FcUInt {
    pub index: u32,
    pub value: u32,
}
