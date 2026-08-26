//! Registers `src/shaders/include/` for gpukit's `#include` expansion.
//!
//! Every shared header in that folder is embedded and resolvable by name —
//! adding a header is adding the file. build.rs emits `rerun-if-changed`
//! for the shader tree, which covers the embedded folder.

gpukit::register_includes!("$CARGO_MANIFEST_DIR/src/shaders/include");

#[cfg(test)]
mod tests {
    #[test]
    fn every_production_include_resolves() {
        let table = gpukit::includes::include_table().expect("include table");
        assert!(!table.is_empty(), "no include folders registered");
        let s = gpukit::metal::expand(crate::shaders::moe_grouped::SHADER, &table);
        assert!(s.contains("kernel void moe_grouped"));
        assert!(s.len() > 8000);
    }
}
