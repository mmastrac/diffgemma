//! Compile-time subkernel variant tuple (STRATEGY.md §4).

/// Function-constant specialization for one logical kernel body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelVariant {
    /// Extra GPU-side bounds checks (tier-2 debug).
    pub shape_assert: bool,
    /// 0 = off; >0 writes chosen intermediates to optional dump buffer.
    pub dump_stage: u32,
    /// Quant path selector (GEMM kernels; inert on elementwise bodies).
    pub use_fp4: bool,
}

impl KernelVariant {
    pub const PRODUCTION: Self = Self {
        shape_assert: false,
        dump_stage: 0,
        use_fp4: false,
    };

    pub const TEST_ASSERT: Self = Self {
        shape_assert: true,
        dump_stage: 0,
        use_fp4: false,
    };

    pub const TEST_DUMP: Self = Self {
        shape_assert: true,
        dump_stage: 1,
        use_fp4: false,
    };

    pub fn cache_label(&self, entry: &str) -> String {
        format!(
            "{entry}_sa{}_d{}_fp{}",
            u8::from(self.shape_assert),
            self.dump_stage,
            u8::from(self.use_fp4),
        )
    }
}
