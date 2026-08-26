//! Runtime `#include "x.metal"` expansion for shader sources.
//!
//! The Metal runtime compiler takes one source string, so shared headers are
//! spliced in before compilation — string work measured in microseconds
//! against a millisecond Metal compile. Emits `#line` directives so compile
//! errors map back to the original files (the entry file is labeled
//! `kernel.metal`; its line numbers match the original source).
//!
//! Only quoted local includes (`#include "name.metal"`) are expanded; system
//! includes (`#include <metal_stdlib>`) pass through to the Metal compiler
//! untouched.

/// `#include "name.metal"` → `Some(name)`; anything else (system includes,
/// code) → `None`.
fn parse_local_include(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("#include \"")?;
    let name = &rest[..rest.find('"')?];
    name.ends_with(".metal").then_some(name)
}

/// Recursively expand quoted includes against the caller's header table.
///
/// Panics on an include name absent from `includes` — a missing header is a
/// build defect, not a runtime condition.
pub fn expand(source: &str, includes: &[(&str, &str)]) -> String {
    expand_labeled("kernel.metal", source, includes)
}

fn expand_labeled(label: &str, source: &str, includes: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(source.len() * 2);
    out.push_str(&format!("#line 1 \"{label}\"\n"));
    for (line_num, line) in (1_u32..).zip(source.split_inclusive('\n')) {
        if let Some(name) = parse_local_include(line) {
            let content = includes
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, s)| *s)
                .unwrap_or_else(|| {
                    panic!("expand: unknown include {name:?} (missing from the include table)")
                });
            let resume = line_num + 1;
            out.push_str(&expand_labeled(
                &format!("include/{name}"),
                content,
                includes,
            ));
            out.push_str(&format!("#line {resume} \"{label}\"\n"));
        } else {
            out.push_str(line);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const INCLUDES: &[(&str, &str)] = &[
        ("outer.metal", "#include \"inner.metal\"\nfloat outer();\n"),
        ("inner.metal", "float inner();\n"),
    ];

    #[test]
    fn emits_line_directives() {
        let s = expand("line1\n#include \"outer.metal\"\nline3\n", INCLUDES);
        assert!(
            s.contains("#line 1 \"kernel.metal\""),
            "entry #line missing"
        );
        assert!(
            s.contains("#line 1 \"include/outer.metal\""),
            "include #line missing"
        );
        assert!(
            s.contains("#line 1 \"include/inner.metal\""),
            "nested #line missing"
        );
        // The include sits on line 2, so the entry file resumes at line 3.
        assert!(
            s.contains("#line 3 \"kernel.metal\""),
            "resume #line missing:\n{s}"
        );
        assert!(s.contains("float outer();"));
        assert!(s.contains("float inner();"));
    }

    #[test]
    fn passthrough_without_local_includes() {
        let s = expand("#include <metal_stdlib>\nkernel void k() {}\n", INCLUDES);
        assert!(s.contains("#include <metal_stdlib>"));
        assert!(s.contains("kernel void k() {}"));
    }

    #[test]
    #[should_panic(expected = "unknown include")]
    fn unknown_include_panics() {
        expand("#include \"missing.metal\"\n", INCLUDES);
    }
}
