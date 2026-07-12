//! Expand `#include "name.metal"` in shader sources at compile time (Rust concat path).
//!
//! Emits `#line N "path.metal"` so Metal compile errors map back to the original file.
//! Native Metal builds resolve the same includes via `-I src/shaders/include`.

use proc_macro::TokenStream;
use quote::quote;
use std::path::{Path, PathBuf};
use syn::{parse_macro_input, LitStr};

fn shaders_root() -> Result<PathBuf, String> {
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?,
    );
    let direct = manifest.join("src/shaders");
    if direct.is_dir() {
        return Ok(direct);
    }
    if let Some(parent) = manifest.parent() {
        let nested = parent.join("src/shaders");
        if nested.is_dir() {
            return Ok(nested);
        }
    }
    Err(format!(
        "shader-include: src/shaders/ not found under {} (set CARGO_MANIFEST_DIR?)",
        manifest.display()
    ))
}

fn line_path(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
}

fn line_directive(root: &Path, file: &Path, line: u32) -> String {
    format!("#line {line} \"{}\"\n", line_path(root, file))
}

fn resolve_include(root: &Path, name: &str) -> Result<PathBuf, String> {
    let include = root.join("include").join(name);
    if include.is_file() {
        return Ok(include);
    }
    Err(format!(
        "shader-include: unknown include {name:?} (expected {}/include/{name})",
        root.display()
    ))
}

fn parse_local_include(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("#include \"")?;
    let end = rest.find('"')?;
    let name = &rest[..end];
    name.ends_with(".metal").then_some(name)
}

fn expand_file(root: &Path, file: &Path, source: &str) -> Result<String, String> {
    let mut out = String::with_capacity(source.len() * 2);
    out.push_str(&line_directive(root, file, 1));

    let mut line_num: u32 = 1;
    for line in source.split_inclusive('\n') {
        if let Some(name) = parse_local_include(line) {
            let inc_path = resolve_include(root, name)?;
            let content = std::fs::read_to_string(&inc_path)
                .map_err(|e| format!("read {}: {e}", inc_path.display()))?;
            let resume_line = line_num + 1;
            out.push_str(&expand_file(root, &inc_path, &content)?);
            out.push_str(&line_directive(root, file, resume_line));
        } else {
            out.push_str(line);
        }
        line_num += 1;
    }
    Ok(out)
}

fn read_entry(root: &Path, rel: &str) -> Result<String, String> {
    let path = root.join(rel);
    if !path.is_file() {
        return Err(format!("shader-include: entry not found: {}", path.display()));
    }
    let source = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    expand_file(root, &path, &source)
}

/// `include_metal!("kernels/gelu.metal")` → fully expanded `&str` literal.
#[proc_macro]
pub fn include_metal(input: TokenStream) -> TokenStream {
    let path_lit = parse_macro_input!(input as LitStr);
    let rel = path_lit.value();
    if Path::new(&rel).is_absolute() || rel.contains("..") {
        return syn::Error::new(
            path_lit.span(),
            format!("shader-include: path must be relative under src/shaders/: {rel:?}"),
        )
        .to_compile_error()
        .into();
    }
    let root = match shaders_root() {
        Ok(r) => r,
        Err(e) => {
            return syn::Error::new(path_lit.span(), e)
                .to_compile_error()
                .into();
        }
    };
    let expanded = match read_entry(&root, &rel) {
        Ok(s) => s,
        Err(e) => {
            return syn::Error::new(path_lit.span(), e)
                .to_compile_error()
                .into();
        }
    };
    let lit = LitStr::new(&expanded, path_lit.span());
    quote! { #lit }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_moe_grouped() {
        let root = shaders_root().expect("root");
        let s = read_entry(&root, "moe/moe_grouped/moe_grouped.metal").expect("expand");
        assert!(s.contains("kernel void moe_grouped"));
        assert!(s.len() > 8000);
    }

    #[test]
    fn emits_line_directives() {
        let root = shaders_root().expect("root");
        let s = read_entry(&root, "elementwise/gelu/gelu.metal").expect("expand");
        assert!(
            s.contains("#line 1 \"elementwise/gelu/gelu.metal\""),
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
        // After fc_axes include, gelu.metal resumes at line 5 (#include on line 4).
        assert!(
            s.contains("#line 5 \"elementwise/gelu/gelu.metal\""),
            "resume #line missing: {}",
            &s[..s.len().min(800)]
        );
    }
}
