//! Runtime `#include "x.metal"` expansion for shader sources.
//!
//! Kernels embed their Metal source with a file-relative `include_str!`
//! (the .metal sits next to its mod.rs); the shared headers live in the
//! table below. Expansion runs once per library compile in device.rs —
//! string work measured in microseconds against a millisecond Metal
//! compile. Emits `#line` directives so Metal compile errors map back to
//! the original files (the entry file is labeled `kernel.metal`; its line
//! numbers still match the original source).
//!
//! Only quoted local includes (`#include "name.metal"`) are expanded;
//! system includes (`#include <metal_stdlib>`) pass through to the Metal
//! runtime compiler untouched.

/// Every shared header under `src/shaders/include/`, by include name.
/// Adding a header = add the file AND a row here (the
/// `include_table_matches_dir` test enforces both directions).
static INCLUDES: &[(&str, &str)] = &[
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

fn include_source(name: &str) -> Option<&'static str> {
    INCLUDES.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

/// `#include "name.metal"` → `Some(name)`; anything else (system includes,
/// code) → `None`. Mirrors the retired shader-include proc macro.
fn parse_local_include(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("#include \"")?;
    let name = &rest[..rest.find('"')?];
    name.ends_with(".metal").then_some(name)
}

/// Recursively expand quoted includes, emitting `#line` directives.
pub fn expand(source: &str) -> String {
    expand_labeled("kernel.metal", source)
}

fn expand_labeled(label: &str, source: &str) -> String {
    let mut out = String::with_capacity(source.len() * 2);
    out.push_str(&format!("#line 1 \"{label}\"\n"));
    let mut line_num: u32 = 1;
    for line in source.split_inclusive('\n') {
        if let Some(name) = parse_local_include(line) {
            let content = include_source(name).unwrap_or_else(|| {
                panic!("expand: unknown include {name:?} (add it to src/shaders/include/ and the INCLUDES table)")
            });
            let resume = line_num + 1;
            out.push_str(&expand_labeled(&format!("include/{name}"), content));
            out.push_str(&format!("#line {resume} \"{label}\"\n"));
        } else {
            out.push_str(line);
        }
        line_num += 1;
    }
    out
}

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

    #[test]
    fn expand_moe_grouped() {
        let s = expand(crate::shaders::moe_grouped::SHADER);
        assert!(s.contains("kernel void moe_grouped"));
        assert!(s.len() > 8000);
    }

    #[test]
    fn emits_line_directives() {
        let s = expand(crate::shaders::gelu::SHADER);
        assert!(
            s.contains("#line 1 \"kernel.metal\""),
            "entry #line missing"
        );
        assert!(
            s.contains("#line 1 \"include/fc_axes.metal\""),
            "include #line missing"
        );
        assert!(
            s.contains("#line 1 \"include/activations.metal\""),
            "nested include #line missing"
        );
        // After the fc_axes include on line 4, gelu.metal resumes at line 5.
        assert!(
            s.contains("#line 5 \"kernel.metal\""),
            "resume #line missing: {}",
            &s[..s.len().min(800)]
        );
    }

    #[test]
    fn passthrough_without_local_includes() {
        let s = expand(crate::shaders::gemm::ENGINE_LINEAR_SHADER);
        // System includes untouched; content preserved verbatim after the label.
        assert!(s.contains("#include <metal_stdlib>"));
        assert!(s.contains("kernel void bf16_gemm"));
    }
}
