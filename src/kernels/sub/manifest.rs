//! Kernel FC axis manifest — validity contract for pipeline specialization.
//!
//! `build/manifest.toml` is the human-readable spec; this module is the
//! enforcement layer. Invalid tuples cannot be compiled; tier-1 tests enumerate
//! every valid variant row.

use super::variant::{ElemDtype, FcBool, FcUInt, KernelVariant, QuantFormat};
use crate::safetensors::Error;

/// Parsed manifest (static mirror of `build/manifest.toml`).
pub struct Manifest {
    pub kernels: &'static [KernelSpec],
}

pub struct KernelSpec {
    pub name: &'static str,
    pub entry: &'static str,
    /// Valid FC3 values. Empty means inert-only (must be `QuantFormat::Q4Affine`).
    pub quant_formats: &'static [QuantFormat],
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
    GemmQ4,
    GemmNvfp4,
    GemmQ8,
    GemmQ8Rowk,
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
        KernelSpec {
            name: "rms_norm_rows",
            entry: "rms_norm_rows",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::RmsNormRows {
                rows: &[
                    RmsNormRowsVariant { affine: false },
                    RmsNormRowsVariant { affine: true },
                ],
            },
        },
        KernelSpec {
            name: "rms_norm_rows_tiled",
            entry: "rms_norm_rows_tiled",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::RmsNormRowsTiled {
                rows: &[
                    RmsNormRowsTiledVariant {
                        in_dtype: ElemDtype::F32,
                    },
                    RmsNormRowsTiledVariant {
                        in_dtype: ElemDtype::Half,
                    },
                ],
            },
        },
        KernelSpec {
            name: "swiglu",
            entry: "swiglu",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::SwigluSplit {
                rows: &[
                    SwigluSplitVariant {
                        io_dtype: ElemDtype::F32,
                        gelu_gate: false,
                        in_place: true,
                    },
                    SwigluSplitVariant {
                        io_dtype: ElemDtype::Half,
                        gelu_gate: true,
                        in_place: false,
                    },
                ],
            },
        },
        KernelSpec {
            name: "swiglu_moe_gate_up",
            entry: "swiglu_moe_gate_up",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::SwigluMoeGateUp,
        },
        KernelSpec {
            name: "gelu",
            entry: "gelu",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::Gelu,
        },
        KernelSpec {
            name: "embed_gather",
            entry: "embed_gather",
            quant_formats: &[QuantFormat::Q8],
            variants: KernelVariants::Elementwise,
        },
        KernelSpec {
            name: "logit_rowstats",
            entry: "logit_rowstats",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::Elementwise,
        },
        KernelSpec {
            name: "sc_probs",
            entry: "sc_probs",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::Elementwise,
        },
        KernelSpec {
            name: "gemm_q4",
            entry: "gemm_q4",
            quant_formats: &[QuantFormat::Q4Affine],
            variants: KernelVariants::GemmQ4,
        },
        KernelSpec {
            name: "gemm_nvfp4",
            entry: "gemm_nvfp4",
            quant_formats: &[QuantFormat::NvFp4],
            variants: KernelVariants::GemmNvfp4,
        },
        KernelSpec {
            name: "gemm_q8",
            entry: "gemm_q8",
            quant_formats: &[QuantFormat::Q8],
            variants: KernelVariants::GemmQ8,
        },
        KernelSpec {
            name: "gemm_q8_rowk",
            entry: "gemm_q8_rowk",
            quant_formats: &[QuantFormat::Q8],
            variants: KernelVariants::GemmQ8Rowk,
        },
    ],
};

pub fn spec_by_entry(entry: &str) -> Option<&'static KernelSpec> {
    MANIFEST.kernels.iter().find(|k| k.entry == entry)
}

pub fn validate_shared(entry: &str, variant: KernelVariant) -> Result<(), Error> {
    let spec = spec_by_entry(entry).ok_or_else(|| {
        Error::NotFound(format!("manifest: unknown kernel entry {entry:?}"))
    })?;
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
    fn rms_norm_rows_fc_map_no_collisions() {
        assert_no_fc_collisions("rms_norm_rows", &[4]).unwrap();
    }

    #[test]
    fn swiglu_fc_map_no_collisions() {
        assert_no_fc_collisions("swiglu", &[4, 5, 6]).unwrap();
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
                KernelVariants::GemmQ4 => {}
                KernelVariants::GemmNvfp4 => {}
                KernelVariants::GemmQ8 => {}
                KernelVariants::GemmQ8Rowk => {}
                KernelVariants::Elementwise => {}
            }
        }
    }

    #[test]
    fn toml_file_present() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("build/manifest.toml");
        assert!(path.exists(), "build/manifest.toml missing");
        let text = std::fs::read_to_string(path).expect("read manifest");
        assert!(text.contains("K_QUANT_FORMAT"));
        assert!(text.contains("rms_norm_rows"));
        assert!(!text.contains("K_INTERLEAVED"));
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
