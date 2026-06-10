fn main() {
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
