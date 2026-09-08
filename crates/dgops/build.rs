//! Compile every ops/**/*.cu to a cubin for the CUDA backend.
//!
//! Only runs when the cuda feature is on. A missing nvcc is a warning, not an
//! error: the Rust backend still builds (and type-checks) without kernels, and
//! a CUDA dispatch reports that they were not built rather than failing to
//! compile. DGQ_NVCC / DGQ_CUDA_ARCH override the compiler and the -arch value.

use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<String> {
    if let Ok(explicit) = std::env::var("DGQ_NVCC") {
        return Some(explicit);
    }
    for candidate in ["nvcc", "/usr/local/cuda/bin/nvcc"] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

/// sm_XXX for -arch. Prefers an explicit override, then the compute capability
/// nvidia-smi reports, then nvcc's own native detection.
fn cuda_arch() -> String {
    if let Ok(explicit) = std::env::var("DGQ_CUDA_ARCH") {
        return explicit;
    }
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        && out.status.success()
        && let Some(first) = String::from_utf8_lossy(&out.stdout).lines().next()
    {
        let digits: String = first.chars().filter(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return format!("sm_{digits}");
        }
    }
    "native".to_string()
}

fn collect_cu(dir: &Path, out: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_cu(&path, out);
        } else if path.extension().is_some_and(|e| e == "cu") {
            println!("cargo:rerun-if-changed={}", path.display());
            out.push(path);
        }
    }
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(dgops_cuda_kernels)");
    println!("cargo:rerun-if-env-changed=DGQ_NVCC");
    println!("cargo:rerun-if-env-changed=DGQ_CUDA_ARCH");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        println!(
            "cargo:warning=nvcc not found; dgops builds without CUDA kernels \
             (set DGQ_NVCC or install the CUDA toolkit, then rebuild)"
        );
        return;
    };

    let arch = cuda_arch();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("cuda");
    let mut sources = Vec::new();
    collect_cu(Path::new("src"), &mut sources);

    for src in &sources {
        let rel = src.strip_prefix("src").expect("src-relative");
        let dst = out_dir.join(rel).with_extension("cubin");
        std::fs::create_dir_all(dst.parent().expect("parent")).expect("mkdir");
        let status = Command::new(&nvcc)
            .args(["-cubin", "-O3", "-std=c++17", "-arch", &arch, "-o"])
            .arg(&dst)
            .arg(src)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(
            status.success(),
            "nvcc failed for {} (arch {arch})",
            src.display()
        );
    }

    println!(
        "cargo:warning=dgops: compiled {} CUDA kernel source(s) for {arch}",
        sources.len()
    );
    println!("cargo:rustc-cfg=dgops_cuda_kernels");
}
