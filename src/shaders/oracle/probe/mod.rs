//! Device bandwidth probe kernel (diagnostics; src/metal/probe.rs).
pub const MEMCPY_ENTRY: &str = "memcpy_f32";
pub const SHADER: &str = include_str!("probe.metal");

crate::kernel_spec! {
    pub const SPEC {
        name: "probe",
        entry: "memcpy_f32",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
