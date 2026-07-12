//! Legacy dense block-GEMM oracle (shared by the gemm_bf16/q4/q8/nvfp4
//! format fixtures). Validation-only; production dense GEMM is gemm_tunable.
pub const ENTRY: &str = "gemm_block";
pub const SHADER: &str = include_str!("gemm_block.metal");

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "gemm_block",
    entry: "gemm_block",
    quant_formats: &[
        crate::shaders::variant::QuantFormat::Q4Affine,
        crate::shaders::variant::QuantFormat::Q8,
        crate::shaders::variant::QuantFormat::NvFp4,
    ],
    fc: &[(4, "IS_FULL_LAYER"), (5, "GEMM_N"), (6, "GEMM_K")],
    variants: crate::shaders::manifest::KernelVariants::GemmBlock,
};
