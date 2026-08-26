//! Diffgemma's shared-header table for gpukit's `#include` expansion.
//!
//! Kernels embed their Metal source with a file-relative `include_str!` (the
//! .metal sits next to its mod.rs); every shared header under
//! `src/shaders/include/` gets a row here. Adding a header = add the file AND
//! a row (the `include_table_matches_dir` test enforces both directions).

/// Every shared header under `src/shaders/include/`, by include name.
pub static INCLUDES: &[(&str, &str)] = &[
    (
        "activations.metal",
        include_str!("../include/activations.metal"),
    ),
    ("arena.metal", include_str!("../include/arena.metal")),
    ("arena_fc.metal", include_str!("../include/arena_fc.metal")),
    (
        "attention_device.metal",
        include_str!("../include/attention_device.metal"),
    ),
    ("common.metal", include_str!("../include/common.metal")),
    (
        "debug_status.metal",
        include_str!("../include/debug_status.metal"),
    ),
    ("dequant.metal", include_str!("../include/dequant.metal")),
    (
        "dgq_kernels.metal",
        include_str!("../include/dgq_kernels.metal"),
    ),
    ("fc_axes.metal", include_str!("../include/fc_axes.metal")),
    (
        "gemm_block_tile.metal",
        include_str!("../include/gemm_block_tile.metal"),
    ),
    ("gemm_fc.metal", include_str!("../include/gemm_fc.metal")),
    (
        "gemm_frag_tile.metal",
        include_str!("../include/gemm_frag_tile.metal"),
    ),
    (
        "gemm_stacked.metal",
        include_str!("../include/gemm_stacked.metal"),
    ),
    (
        "gemm_stacked_fc.metal",
        include_str!("../include/gemm_stacked_fc.metal"),
    ),
    (
        "gqa_device.metal",
        include_str!("../include/gqa_device.metal"),
    ),
    (
        "moe_grouped_device.metal",
        include_str!("../include/moe_grouped_device.metal"),
    ),
    (
        "moe_grouped_dispatch.metal",
        include_str!("../include/moe_grouped_dispatch.metal"),
    ),
    (
        "moe_router_device.metal",
        include_str!("../include/moe_router_device.metal"),
    ),
    (
        "qgemm_grouped.metal",
        include_str!("../include/qgemm_grouped.metal"),
    ),
    (
        "sampler_device.metal",
        include_str!("../include/sampler_device.metal"),
    ),
    (
        "sc_prob_scale.metal",
        include_str!("../include/sc_prob_scale.metal"),
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn include_table_matches_dir() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/include");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .expect("read include dir")
            .map(|e| {
                e.expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .filter(|n| n.ends_with(".metal"))
            .collect();
        on_disk.sort();
        let mut in_table: Vec<String> = INCLUDES.iter().map(|(n, _)| n.to_string()).collect();
        in_table.sort();
        assert_eq!(
            on_disk, in_table,
            "src/shaders/include/ and expand::INCLUDES must list the same headers"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_resolves_every_production_include() {
        let s = gpukit::metal::expand(crate::shaders::moe_grouped::SHADER, INCLUDES);
        assert!(s.contains("kernel void moe_grouped"));
        assert!(s.len() > 8000);
    }
}
