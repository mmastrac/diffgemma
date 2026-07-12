use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Hash every shader source under `src/shaders/` (sorted walk): the Metal
/// pipeline cache keys on the WHOLE shader tree (DGQ_SHADER_TREE_HASH), so a
/// stale metallib can never be served after a kernel edit. (Shader sources
/// are embedded via file-relative include_str!, which already registers each
/// .metal as a cargo dependency — but the build script itself must re-run to
/// recompute the hash env var, hence the rerun-if-changed lines below.)
///
/// rerun-if-changed is emitted per DIRECTORY + per .metal FILE (not the tree
/// root) so .rs edits under src/shaders/ do not re-run the build script;
/// directory entries catch .metal file adds/removes.
fn hash_shader_tree(dir: &Path, h: &mut DefaultHasher) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            hash_shader_tree(&path, h);
        } else if path.extension().is_some_and(|e| e == "metal") {
            println!("cargo:rerun-if-changed={}", path.display());
            path.to_string_lossy().hash(h);
            std::fs::read(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
                .hash(h);
        }
    }
}

fn main() {
    let mut h = DefaultHasher::new();
    hash_shader_tree(Path::new("src/shaders"), &mut h);
    println!("cargo:rustc-env=DGQ_SHADER_TREE_HASH={:016x}", h.finish());

    if std::env::var("CARGO_FEATURE_BLAS").is_ok() {
        let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target == "macos" {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        } else if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
            // Optional: link system OpenBLAS when explicitly requested on Linux.
            println!("cargo:rustc-link-lib=openblas");
        }
    }
}
