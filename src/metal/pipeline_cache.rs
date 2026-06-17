//! Runtime `MTLBinaryArchive` cache for compiled compute pipelines.
//!
//! Persists device-specific pipeline ISA to disk so cold starts can skip
//! recompilation after the first run (pure runtime API; no Metal SDK required).

use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSURL, NSString};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLPipelineOption,
};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};

const CACHE_BUNDLE_TAG: &str = "diffgemma-mps-v8-stacked-seg-fc";

fn shader_bundle_token() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for src in [
        include_str!("../../shaders/include/common.metal"),
        include_str!("../../shaders/include/dequant.metal"),
        include_str!("../../shaders/include/activations.metal"),
        include_str!("../../shaders/gemm.metal"),
        include_str!("../../shaders/include/fc_axes.metal"),
        include_str!("../../shaders/include/gemm_fc.metal"),
        include_str!("../../shaders/include/gemm_block_tile.metal"),
        include_str!("../../shaders/include/qgemm_grouped.metal"),
        include_str!("../../shaders/kernels/dequant_block_matrix.metal"),
        include_str!("../../shaders/kernels/gemm_block.metal"),
        include_str!("../../shaders/kernels/gemm_block_stacked.metal"),
        include_str!("../../shaders/kernels/gemm_block_grouped.metal"),
        include_str!("../../shaders/include/gemm_stacked.metal"),
        include_str!("../../shaders/include/gemm_stacked_fc.metal"),
        include_str!("../../shaders/kernels/gemm_linear_f32.metal"),
        include_str!("../../shaders/kernels/gemm_linear_grouped.metal"),
        include_str!("../../shaders/kernels/gemm_q8_linear_f32.metal"),
        include_str!("../../shaders/kernels/gemm_q8_linear_kxn_f32.metal"),
        include_str!("../../shaders/kernels/gemm_q8.metal"),
        include_str!("../../shaders/kernels/gemm_q8_rowk.metal"),
        include_str!("../../shaders/kernels/embed_gather.metal"),
        include_str!("../../shaders/kernels/gather_prob_cols.metal"),
        include_str!("../../shaders/kernels/gather_rows.metal"),
        include_str!("../../shaders/kernels/gelu.metal"),
        include_str!("../../shaders/kernels/swiglu.metal"),
        include_str!("../../shaders/kernels/swiglu_moe_gate_up.metal"),
        include_str!("../../shaders/kernels/half_scale.metal"),
        include_str!("../../shaders/kernels/half_to_f32.metal"),
        include_str!("../../shaders/kernels/pack_encoder_kv.metal"),
        include_str!("../../shaders/kernels/logit_rowstats.metal"),
        include_str!("../../shaders/kernels/sc_prob_cols.metal"),
        include_str!("../../shaders/kernels/sc_probs.metal"),
        include_str!("../../shaders/kernels/sc_softembed.metal"),
        include_str!("../../shaders/kernels/memzero_bytes.metal"),
        include_str!("../../shaders/kernels/residual_f32b.metal"),
        include_str!("../../shaders/kernels/residual_half.metal"),
        include_str!("../../shaders/kernels/rms_norm_rows.metal"),
        include_str!("../../shaders/kernels/rms_norm_rows_tiled.metal"),
        include_str!("../../shaders/kernels/router_scale_rows.metal"),
        include_str!("../../shaders/kernels/router_top_k_rows.metal"),
        include_str!("../../shaders/kernels/softcap_half.metal"),
        include_str!("../../shaders/kernels/softmax_rows.metal"),
        include_str!("../../shaders/kernels/vec_add_inplace.metal"),
        include_str!("../../shaders/kernels/vec_fill_zero.metal"),
        include_str!("../../shaders/kernels/vec_mul_inplace.metal"),
        include_str!("../../shaders/kernels/vec_scale_inplace.metal"),
        include_str!("../../shaders/include/gqa_device.metal"),
        include_str!("../../shaders/kernels/apply_rope_heads.metal"),
        include_str!("../../shaders/kernels/argmax_rows.metal"),
        include_str!("../../shaders/kernels/copy_f32.metal"),
        include_str!("../../shaders/kernels/gqa_attention.metal"),
        include_str!("../../shaders/kernels/logit_softcapping.metal"),
        include_str!("../../shaders/kernels/row_entropy.metal"),
        include_str!("../../shaders/kernels/sample_from_probs_rows.metal"),
        include_str!("../../shaders/kernels/scale_logits.metal"),
        include_str!("../../shaders/kernels/scatter_vocab_chunk.metal"),
        include_str!("../../shaders/kernels/compact_active_rows.metal"),
        include_str!("../../shaders/kernels/gather_rows_bf16.metal"),
        include_str!("../../shaders/kernels/scatter_logits_rows.metal"),
        include_str!("../../shaders/monolithic/diffgemma_step.metal"),
    ] {
        src.hash(&mut h);
    }
    CACHE_BUNDLE_TAG.hash(&mut h);
    h.finish()
}

fn cache_enabled() -> bool {
    match std::env::var("DGQ_METAL_PIPELINE_CACHE") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
}

fn cache_root_dir() -> PathBuf {
    if let Ok(v) = std::env::var("DGQ_METAL_PIPELINE_CACHE") {
        if !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false") {
            return PathBuf::from(v);
        }
    }
    std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join("diffgemma-mps")
        .join("metal-pipelines")
}

fn sanitize_device_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub struct PipelineArchiveCache {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    cache_file: PathBuf,
    dirty: AtomicBool,
    persist: bool,
}

// Metal pipeline objects are used from the main inference thread only.
unsafe impl Send for PipelineArchiveCache {}
unsafe impl Sync for PipelineArchiveCache {}

static GLOBAL_CACHE: OnceLock<Result<Arc<PipelineArchiveCache>, String>> = OnceLock::new();

impl PipelineArchiveCache {
    pub fn shared(device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>, Error> {
        match GLOBAL_CACHE.get_or_init(|| Self::open(device).map(Arc::new).map_err(|e| e.to_string()))
        {
            Ok(c) => Ok(Arc::clone(c)),
            Err(msg) => Err(Error::NotFound(msg.clone())),
        }
    }

    pub fn flush_global() {
        if let Some(Ok(c)) = GLOBAL_CACHE.get() {
            c.flush_if_dirty();
        }
    }

    fn open(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        if !cache_enabled() {
            return Self::open_ephemeral(device);
        }

        let cache_dir = cache_root_dir();
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            Error::NotFound(format!("metal pipeline cache mkdir failed: {e}"))
        })?;

        let device_name = sanitize_device_name(&device.name().to_string());
        let token = shader_bundle_token();
        let cache_file = cache_dir.join(format!("{device_name}-{token:016x}.metallibarchive"));
        let existed = cache_file.exists();

        let desc = MTLBinaryArchiveDescriptor::new();
        if existed {
            if let Some(url) = NSURL::from_file_path(&cache_file) {
                desc.setUrl(Some(&url));
            }
        }

        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|e| {
                Error::NotFound(format!(
                    "MTLBinaryArchive open failed: {}",
                    e.localizedDescription()
                ))
            })?;

        eprintln!(
            "metal pipeline cache: {} ({}, {:.2?})",
            cache_file.display(),
            if existed { "loaded" } else { "new" },
            open_started.elapsed()
        );

        Ok(Self {
            archive,
            cache_file,
            dirty: AtomicBool::new(false),
            persist: true,
        })
    }

    fn open_ephemeral(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|_| Error::Format("MTLBinaryArchive create failed"))?;
        Ok(Self {
            archive,
            cache_file: PathBuf::new(),
            dirty: AtomicBool::new(false),
            persist: false,
        })
    }

    pub fn compile_compute(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(function));
        desc.setLabel(Some(&NSString::from_str(label)));
        desc.setSupportIndirectCommandBuffers(true);
        let archives = NSArray::from_slice(&[&*self.archive]);
        desc.setBinaryArchives(Some(&archives));

        let pipeline = device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|_| Error::Format("Metal pipeline compile failed"))?;

        if self.persist
            && self
                .archive
                .addComputePipelineFunctionsWithDescriptor_error(&desc)
                .is_ok()
        {
            self.dirty.store(true, Ordering::Relaxed);
        }

        Ok(pipeline)
    }

    pub fn flush_if_dirty(&self) {
        if !self.persist || !self.dirty.swap(false, Ordering::Relaxed) {
            return;
        }
        let Some(url) = NSURL::from_file_path(&self.cache_file) else {
            return;
        };
        let save_started = std::time::Instant::now();
        match self.archive.serializeToURL_error(&url) {
            Ok(()) => eprintln!(
                "metal pipeline cache: saved {} ({:.2?})",
                self.cache_file.display(),
                save_started.elapsed()
            ),
            Err(e) => eprintln!(
                "warning: metal pipeline cache serialize failed: {}",
                e.localizedDescription()
            ),
        }
    }
}
