//! Legacy dense block-GEMM oracle (shared by the gemm_bf16/q4/q8/nvfp4
//! format fixtures). Validation-only; production dense GEMM is gemm_tunable.
crate::shader_kernel! {
    name = "gemm_block",
    metal = "gemm_block.metal",
    spec = {
            quant_formats: &[
                QuantFormat::Q4Affine,
                QuantFormat::Q8,
                QuantFormat::NvFp4,
            ],
            fc: &[(4, "IS_FULL_LAYER"), (5, "GEMM_N"), (6, "GEMM_K")],
            variants: KernelVariants::GemmBlock,
    },
    tests = {},
}
