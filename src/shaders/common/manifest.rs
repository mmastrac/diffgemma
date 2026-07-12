//! Kernel FC axis manifest — validity contract for pipeline specialization.
//!
//! Each registered kernel declares its own `SPEC` const in its kernel
//! directory (colocated with the Metal source); `MANIFEST` below collects
//! them. This module is the enforcement layer: invalid tuples cannot be
//! compiled, and tier-1 tests enumerate every valid variant row.
//!
//! The human-readable TOML form is GENERATED from these specs at runtime
//! (`render_toml()`, `diffgemma-mps manifest`) — there is no checked-in
//! manifest file to drift. FC 1–3 are reserved globally
//! (src/shaders/include/fc_axes.metal); `fc` lists each kernel's LOCAL
//! axes (index ≥ 4). Axes of non-registered kernels (the tunable GEMM
//! family's FC9–11/28–30, convert_scale's FC4/5, …) are documented in
//! ARCHITECTURE.md.

use super::variant::{ElemDtype, FcBool, FcUInt, KernelVariant, QuantFormat};
use crate::safetensors::Error;

/// Collected kernel specs (each lives beside its kernel).
pub struct Manifest {
    pub kernels: &'static [&'static KernelSpec],
}

pub struct KernelSpec {
    pub name: &'static str,
    pub entry: &'static str,
    /// Valid FC3 values. Empty means inert-only (must be `QuantFormat::Q4Affine`).
    pub quant_formats: &'static [QuantFormat],
    /// Local function-constant axes: (index ≥ 4, name). FC1–3 are implied.
    pub fc: &'static [(u32, &'static str)],
    pub variants: KernelVariants,
}

pub enum KernelVariants {
    RmsNormRows {
        rows: &'static [RmsNormRowsVariant],
    },
    RmsNormRowsTiled {
        rows: &'static [RmsNormRowsTiledVariant],
    },
    SwigluSplit {
        rows: &'static [SwigluSplitVariant],
    },
    SwigluMoeGateUp,
    Gelu,
    GemmBlock,
    GemmRowk,
    Elementwise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNormRowsVariant {
    pub affine: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNormRowsTiledVariant {
    pub in_dtype: ElemDtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwigluSplitVariant {
    pub io_dtype: ElemDtype,
    pub gelu_gate: bool,
    pub in_place: bool,
}

pub static MANIFEST: Manifest = Manifest {
    kernels: &[
        &crate::shaders::rms_norm_rows::SPEC,
        &crate::shaders::rms_norm_rows_tiled::SPEC,
        &crate::shaders::swiglu::SPEC,
        &crate::shaders::swiglu::SPEC_MOE_GATE_UP,
        &crate::shaders::gelu::SPEC,
        &crate::shaders::embed_gather::SPEC,
        &crate::shaders::logit_rowstats::SPEC,
        &crate::shaders::sc_prob_cols::SPEC,
        &crate::shaders::sample_rowstats::SPEC,
        &crate::shaders::sample_commit::SPEC,
        &crate::shaders::sample_apply::SPEC,
        &crate::shaders::sample_write::SPEC,
        &crate::shaders::qk_rope_kv::SPEC,
        &crate::shaders::attention::SPEC,
        &crate::shaders::moe_router::SPEC,
        &crate::shaders::moe_bucket_count::SPEC,
        &crate::shaders::moe_bucket_fill::SPEC,
        &crate::shaders::moe_grouped::SPEC,
        &crate::shaders::moe_scatter_weighted::SPEC,
        &crate::shaders::gemm_linear_grouped::SPEC,
        &crate::shaders::gemm_block_grouped::SPEC,
        &crate::shaders::gemm_linear_f32::SPEC,
        &crate::shaders::gemm_q8_linear_f32::SPEC,
        &crate::shaders::oracle::gemm::gemm_block::SPEC,
        &crate::shaders::gemm_block_stacked::SPEC,
        &crate::shaders::gemm_rowk::SPEC,
    ],
};

pub fn spec_by_entry(entry: &str) -> Option<&'static KernelSpec> {
    MANIFEST.kernels.iter().copied().find(|k| k.entry == entry)
}

/// Render the collected specs as the human-readable TOML manifest.
/// (`diffgemma-mps manifest` — generated on demand, never checked in.)
pub fn render_toml() -> String {
    let mut out = String::new();
    out.push_str(
        "# Kernel function-constant manifest (GENERATED: diffgemma-mps manifest).\n\
         #\n\
         # FC 1\u{2013}3 are reserved globally (see src/shaders/include/fc_axes.metal).\n\
         # FC 4+ are per-kernel; each kernel lists its full index map under [kernel.fc].\n\
         # Pipeline construction in Rust MUST only build tuples listed under [[kernel.variants]].\n\
         # Tier-1 tests MUST cover every [[kernel.variants]] row (see manifest.rs tests).\n",
    );
    for k in MANIFEST.kernels {
        out.push_str(&format!(
            "\n[[kernel]]\nname = \"{}\"\nentry = \"{}\"\n",
            k.name, k.entry
        ));
        let formats: Vec<String> = k
            .quant_formats
            .iter()
            .map(|f| (*f as u32).to_string())
            .collect();
        out.push_str(&format!("quant_formats = [{}]\n", formats.join(", ")));
        out.push_str(
            "\n[kernel.fc]\n1 = \"K_SHAPE_ASSERT\"\n2 = \"K_DUMP_STAGE\"\n3 = \"K_QUANT_FORMAT\"\n",
        );
        for (idx, name) in k.fc {
            out.push_str(&format!("{idx} = \"{name}\"\n"));
        }
        match k.variants {
            KernelVariants::RmsNormRows { rows } => {
                for r in rows {
                    out.push_str(&format!("\n[[kernel.variants]]\naffine = {}\n", r.affine));
                }
            }
            KernelVariants::RmsNormRowsTiled { rows } => {
                for r in rows {
                    out.push_str(&format!(
                        "\n[[kernel.variants]]\nin_dtype = {}\n",
                        r.in_dtype as u32
                    ));
                }
            }
            KernelVariants::SwigluSplit { rows } => {
                for r in rows {
                    out.push_str(&format!(
                        "\n[[kernel.variants]]\nio_dtype = {}\ngelu_gate = {}\nin_place = {}\n",
                        r.io_dtype as u32, r.gelu_gate, r.in_place
                    ));
                }
            }
            KernelVariants::SwigluMoeGateUp
            | KernelVariants::Gelu
            | KernelVariants::GemmBlock
            | KernelVariants::GemmRowk
            | KernelVariants::Elementwise => {
                out.push_str("\n[[kernel.variants]]\n");
            }
        }
    }
    out
}

pub fn validate_shared(entry: &str, variant: KernelVariant) -> Result<(), Error> {
    let spec = spec_by_entry(entry)
        .ok_or_else(|| Error::NotFound(format!("manifest: unknown kernel entry {entry:?}")))?;
    if !spec.quant_formats.is_empty() {
        if !spec.quant_formats.contains(&variant.quant_format) {
            return Err(Error::NotFound(format!(
                "manifest: {entry} does not support quant_format {:?}",
                variant.quant_format
            )));
        }
    } else if variant.quant_format != QuantFormat::Q4Affine {
        return Err(Error::NotFound(format!(
            "manifest: {entry} is quant-inert; quant_format must be Q4Affine (0)"
        )));
    }
    Ok(())
}

pub fn rms_norm_rows_variant(affine: bool) -> Result<RmsNormRowsVariant, Error> {
    let v = RmsNormRowsVariant { affine };
    let spec = spec_by_entry("rms_norm_rows").expect("manifest");
    if let KernelVariants::RmsNormRows { rows } = spec.variants {
        if rows.contains(&v) {
            return Ok(v);
        }
    }
    Err(Error::NotFound(format!(
        "manifest: invalid rms_norm_rows affine={affine}"
    )))
}

pub fn rms_norm_rows_tiled_variant(in_dtype: ElemDtype) -> Result<RmsNormRowsTiledVariant, Error> {
    let v = RmsNormRowsTiledVariant { in_dtype };
    let spec = spec_by_entry("rms_norm_rows_tiled").expect("manifest");
    if let KernelVariants::RmsNormRowsTiled { rows } = spec.variants {
        if rows.contains(&v) {
            return Ok(v);
        }
    }
    Err(Error::NotFound(format!(
        "manifest: invalid rms_norm_rows_tiled in_dtype={in_dtype:?}"
    )))
}

pub fn swiglu_split_variant(v: SwigluSplitVariant) -> Result<SwigluSplitVariant, Error> {
    let spec = spec_by_entry("swiglu").expect("manifest");
    if let KernelVariants::SwigluSplit { rows } = spec.variants {
        if rows.contains(&v) {
            return Ok(v);
        }
    }
    Err(Error::NotFound(format!(
        "manifest: invalid swiglu split {v:?}"
    )))
}

impl RmsNormRowsVariant {
    pub fn local_fcs(self) -> [FcBool; 1] {
        [FcBool {
            index: 4,
            value: self.affine,
        }]
    }

    pub fn cache_suffix(self) -> &'static str {
        if self.affine { "_aff1" } else { "_aff0" }
    }
}

impl RmsNormRowsTiledVariant {
    pub fn local_fcs(self) -> [FcUInt; 1] {
        [FcUInt {
            index: 4,
            value: self.in_dtype as u32,
        }]
    }

    pub fn cache_suffix(self) -> &'static str {
        match self.in_dtype {
            ElemDtype::F32 => "_dt0",
            ElemDtype::Half => "_dt1",
        }
    }
}

impl SwigluSplitVariant {
    pub fn local_fcs(self) -> ([FcUInt; 1], [FcBool; 2]) {
        (
            [FcUInt {
                index: 4,
                value: self.io_dtype as u32,
            }],
            [
                FcBool {
                    index: 5,
                    value: self.gelu_gate,
                },
                FcBool {
                    index: 6,
                    value: self.in_place,
                },
            ],
        )
    }

    pub fn cache_suffix(self) -> String {
        format!(
            "_dt{}_g{}_p{}",
            self.io_dtype as u32,
            u8::from(self.gelu_gate),
            u8::from(self.in_place),
        )
    }

    pub const DECODER_MUL: Self = Self {
        io_dtype: ElemDtype::F32,
        gelu_gate: false,
        in_place: true,
    };

    pub const MONOLITH_GLU: Self = Self {
        io_dtype: ElemDtype::Half,
        gelu_gate: true,
        in_place: false,
    };
}

/// Assert no duplicate FC indices across shared + local axes for one kernel.
pub fn assert_no_fc_collisions(entry: &str, local_indices: &[u32]) -> Result<(), Error> {
    for &idx in local_indices {
        if (1..=3).contains(&idx) {
            return Err(Error::NotFound(format!(
                "manifest: {entry} local FC{idx} collides with reserved FC 1–3"
            )));
        }
    }
    let mut seen = [false; 32];
    for idx in 1..=3 {
        seen[idx as usize] = true;
    }
    for &idx in local_indices {
        if idx as usize >= seen.len() {
            continue;
        }
        if seen[idx as usize] {
            return Err(Error::NotFound(format!(
                "manifest: {entry} duplicate FC index {idx}"
            )));
        }
        seen[idx as usize] = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_entries_unique() {
        let mut names = std::collections::HashSet::new();
        for k in MANIFEST.kernels {
            assert!(names.insert(k.entry), "duplicate entry {}", k.entry);
        }
    }

    #[test]
    fn all_fc_maps_collision_free() {
        for k in MANIFEST.kernels {
            let indices: Vec<u32> = k.fc.iter().map(|(idx, _)| *idx).collect();
            assert_no_fc_collisions(k.entry, &indices).unwrap();
        }
    }

    #[test]
    fn all_manifest_variants_enumerated() {
        for k in MANIFEST.kernels {
            match k.variants {
                KernelVariants::RmsNormRows { rows } => assert_eq!(rows.len(), 2),
                KernelVariants::RmsNormRowsTiled { rows } => assert_eq!(rows.len(), 2),
                KernelVariants::SwigluSplit { rows } => assert_eq!(rows.len(), 2),
                KernelVariants::SwigluMoeGateUp => {}
                KernelVariants::Gelu => {}
                KernelVariants::GemmBlock => {}
                KernelVariants::GemmRowk => {}
                KernelVariants::Elementwise => {}
            }
        }
    }

    #[test]
    fn rendered_toml_covers_every_spec() {
        let toml = render_toml();
        assert_eq!(
            toml.matches("[[kernel]]").count(),
            MANIFEST.kernels.len(),
            "one [[kernel]] block per spec"
        );
        assert!(toml.contains("K_QUANT_FORMAT"));
        assert!(toml.contains("name = \"rms_norm_rows\""));
        assert!(toml.contains("4 = \"K_AFFINE\""));
        // Two enumerated swiglu variant rows survive the round trip.
        assert!(toml.contains("io_dtype = 0"));
        assert!(toml.contains("io_dtype = 1"));
        // The old hand-maintained file's drift cases stay dead.
        assert!(!toml.contains("sc_softembed"));
    }

    #[test]
    fn metal_shader_basenames_unique() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let files = collect_metal_files(&root);
        let mut basenames = std::collections::HashSet::new();
        for path in &files {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert!(
                basenames.insert(name.to_string()),
                "duplicate .metal basename: {name} ({path:?})"
            );
        }
    }

    #[test]
    fn include_dir_has_no_kernels() {
        let include = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/include");
        for path in collect_metal_files(&include) {
            let text = std::fs::read_to_string(&path).expect("read include metal");
            assert!(
                !text.contains("kernel void"),
                "include/ must be device-fn headers only: {}",
                path.display()
            );
        }
    }

    #[test]
    fn kernel_dir_entry_points_present() {
        // Every .metal outside include/ is an entry file (kernel or oracle)
        // and must declare at least one entry point.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let include = root.join("include");
        for path in collect_metal_files(&root) {
            if path.starts_with(&include) {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read kernel metal");
            assert!(
                text.contains("kernel void"),
                "non-include .metal must declare at least one entry point: {}",
                path.display()
            );
        }
    }

    fn collect_metal_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        collect_metal_files_rec(dir, &mut out);
        out.sort();
        out
    }

    fn collect_metal_files_rec(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                collect_metal_files_rec(&path, out);
            } else if path.extension().is_some_and(|e| e == "metal") {
                out.push(path);
            }
        }
    }

    #[test]
    fn reject_invalid_quant_format() {
        let v = KernelVariant {
            quant_format: QuantFormat::NvFp4,
            ..KernelVariant::PRODUCTION
        };
        assert!(validate_shared("gelu", v).is_err());
    }

    #[test]
    fn reject_invalid_swiglu_split_tuple() {
        let bad = SwigluSplitVariant {
            io_dtype: ElemDtype::F32,
            gelu_gate: true,
            in_place: true,
        };
        assert!(super::swiglu_split_variant(bad).is_err());
    }
}
