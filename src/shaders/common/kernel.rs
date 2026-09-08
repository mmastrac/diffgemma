//! `shader_kernel!` — the engine binding of `gpukit::kernel!`.
//!
//! Binds this crate's error type, Metal context/pipeline types, variant type,
//! the scattered manifest macro (`kernel_spec!`) and the tier-1 helpers, so a
//! kernel's registration block names only what is specific to that kernel.
//! `name` defaults the entry point; pass `entry` as well only when the manifest
//! name and the Metal entry point differ.

#[macro_export]
macro_rules! shader_kernel {
    // Manifest name and Metal entry point differ.
    ( name = $name:literal, entry = $entry:literal, $($rest:tt)* ) => {
        gpukit::kernel! {
            shader,
            error = crate::Error,
            ctx = crate::metal::device::MetalContext,
            pipeline_ty = crate::metal::device::ComputePipeline,
            variant = crate::shaders::variant::KernelVariant,
            spec_macro = crate::kernel_spec,
            assert_oracle = crate::shaders::test_util::assert_oracle,
            skip_gpu = crate::shaders::test_util::skip_gpu_on_ci,
            elem_f32 = crate::shaders::test_util::ElemFormat::F32,
            name = $name,
            entry = $entry,
            $($rest)*
        }
    };
    // Manifest name doubles as the entry point (the common case).
    ( name = $name:literal, $($rest:tt)* ) => {
        crate::shader_kernel! { name = $name, entry = $name, $($rest)* }
    };
}
