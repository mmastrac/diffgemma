//! Legacy dense block-GEMM oracle (shared by the gemm_bf16/q4/q8/nvfp4
//! format fixtures). Validation-only; production dense GEMM is gemm_tunable.
pub const ENTRY: &str = "gemm_block";
pub const SHADER: &str = include_str!("gemm_block.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "gemm_block",
        entry: "gemm_block",
        quant_formats: &[
            QuantFormat::Q4Affine,
            QuantFormat::Q8,
            QuantFormat::NvFp4,
        ],
        fc: &[(4, "IS_FULL_LAYER"), (5, "GEMM_N"), (6, "GEMM_K")],
        variants: KernelVariants::GemmBlock,
    }
}
