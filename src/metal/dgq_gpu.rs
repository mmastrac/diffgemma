//! Zero-copy `.dgq` blob → `MTLBuffer` and per-tensor GPU views.

use crate::dgq::layout::{
    dgq_version_supported, nvfp4_matrix_bytes, nvfp4_row_bytes, q4_matrix_bytes, q8_row_bytes,
    QuantKind, NVFP4_HEADER_BYTES, MANIFEST_FILE,
};
use crate::dgq::DgqStore;
use crate::safetensors::Error;
use memmap2::Mmap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

/// Keeps mmap + file alive while the GPU buffer references the mapping.
pub struct DgqGpuBlob {
    _file: File,
    _mmap: Mmap,
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
}

impl DgqGpuBlob {
    pub fn from_store(
        store: &DgqStore,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Arc<Self>, Error> {
        let model_dir = store.model_dir.clone();
        let manifest_path = model_dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest: crate::dgq::layout::DgqManifest = serde_json::from_str(&manifest_json)?;
        if !dgq_version_supported(manifest.version) {
            return Err(Error::Format("unsupported .dgq version"));
        }
        let blob_path = model_dir.join(&manifest.blob_file);
        let file = File::open(&blob_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len();
        let ptr = NonNull::new(mmap.as_ptr() as *mut c_void).ok_or(Error::Format("dgq mmap null"))?;
        let buffer = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Error::Format("dgq gpu blob alloc failed"))?
        };
        Ok(Arc::new(Self {
            _file: file,
            _mmap: mmap,
            buffer,
            len,
        }))
    }

    pub fn open(model_dir: impl AsRef<Path>, device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>, Error> {
        let store = DgqStore::open(model_dir)?;
        Self::from_store(&store, device)
    }
}

/// Quantized linear weight view into a shared blob buffer (PyTorch `[out, in]`).
#[derive(Clone)]
pub struct Q4LinearGpu {
    pub kind: QuantKind,
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q4LinearGpu {
    pub fn from_entry(
        blob: Arc<DgqGpuBlob>,
        offset: u64,
        out_dim: usize,
        in_dim: usize,
        kind: QuantKind,
    ) -> Self {
        Self {
            kind,
            blob,
            byte_offset: offset,
            out_dim,
            in_dim,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        self.kind == QuantKind::Nvfp4Block
    }

    pub fn matrix_byte_len(&self) -> usize {
        match self.kind {
            QuantKind::Q4Block => q4_matrix_bytes(self.out_dim, self.in_dim),
            QuantKind::Nvfp4Block => nvfp4_matrix_bytes(self.out_dim, self.in_dim),
            _ => panic!("not a block linear"),
        }
    }

    pub fn row_byte_len(&self) -> usize {
        match self.kind {
            QuantKind::Q4Block => crate::dgq::layout::q4_row_bytes(self.in_dim),
            QuantKind::Nvfp4Block => nvfp4_row_bytes(self.in_dim),
            _ => panic!("not a block linear"),
        }
    }

    /// Payload after optional nvfp4 global-scale header.
    pub fn body_offset(&self) -> u64 {
        self.byte_offset
            + if self.is_nvfp4() {
                NVFP4_HEADER_BYTES as u64
            } else {
                0
            }
    }

    pub fn q4_byte_len(&self) -> usize {
        self.matrix_byte_len()
    }

    pub fn groups_per_row(&self) -> u32 {
        match self.kind {
            QuantKind::Q4Block => self.in_dim.div_ceil(32) as u32,
            QuantKind::Nvfp4Block => self.in_dim.div_ceil(16) as u32,
            _ => panic!("not a block linear"),
        }
    }

    pub fn global_scale_f32(&self) -> f32 {
        if !self.is_nvfp4() {
            return 1.0;
        }
        let ptr = unsafe {
            self.blob
                .buffer
                .contents()
                .as_ptr()
                .add(self.byte_offset as usize) as *const u8
        };
        f32::from_le_bytes([
            unsafe { *ptr },
            unsafe { *ptr.add(1) },
            unsafe { *ptr.add(2) },
            unsafe { *ptr.add(3) },
        ])
    }

    pub fn weight_buffer(&self) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        (&self.blob.buffer, self.byte_offset)
    }

    /// CPU-readable view of Q4 bytes (shared mmap; matches GPU blob layout).
    pub fn src_slice(&self) -> &[u8] {
        let len = self.q4_byte_len();
        unsafe {
            let ptr = self
                .blob
                .buffer
                .contents()
                .as_ptr()
                .add(self.byte_offset as usize) as *const u8;
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

/// Raw bf16/f32 payload in the blob (norms, router).
#[derive(Clone)]
pub struct RawBlobView {
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub numel: usize,
}

/// Expert stack `[E, out, in]` q4 tensor; slice per expert without copy.
#[derive(Clone)]
pub struct Q4ExpertStackGpu {
    pub kind: QuantKind,
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub experts: usize,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q4ExpertStackGpu {
    pub fn matrix_stride(&self) -> usize {
        match self.kind {
            QuantKind::Q4Block => q4_matrix_bytes(self.out_dim, self.in_dim),
            QuantKind::Nvfp4Block => nvfp4_matrix_bytes(self.out_dim, self.in_dim),
            _ => panic!("not a block expert stack"),
        }
    }

    pub fn expert_linear(&self, expert: usize) -> Q4LinearGpu {
        let per = self.matrix_stride();
        Q4LinearGpu {
            kind: self.kind,
            blob: Arc::clone(&self.blob),
            byte_offset: self.byte_offset + expert as u64 * per as u64,
            out_dim: self.out_dim,
            in_dim: self.in_dim,
        }
    }
}

/// Q8 row-major matrix view into the shared blob (embed / lm_head).
#[derive(Clone)]
pub struct Q8LinearGpu {
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub row_offset: usize,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q8LinearGpu {
    pub fn from_entry(blob: Arc<DgqGpuBlob>, offset: u64, out_dim: usize, in_dim: usize) -> Self {
        Self {
            blob,
            byte_offset: offset,
            row_offset: 0,
            out_dim,
            in_dim,
        }
    }

    pub fn row_slice(&self, row_offset: usize, out_dim: usize) -> Self {
        assert!(row_offset + out_dim <= self.out_dim);
        Self {
            blob: Arc::clone(&self.blob),
            byte_offset: self.byte_offset,
            row_offset: self.row_offset + row_offset,
            out_dim,
            in_dim: self.in_dim,
        }
    }

    pub fn row_stride(&self) -> usize {
        q8_row_bytes(self.in_dim)
    }

    pub fn weight_buffer(&self) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        let off = self.byte_offset + self.row_offset as u64 * self.row_stride() as u64;
        (&self.blob.buffer, off)
    }

    /// Decode per-row bf16 scale at row `r` (CPU reference for q8 row layout).
    pub fn row_scale_f32(&self, row: usize) -> f32 {
        let stride = self.row_stride();
        let (_, base) = self.weight_buffer();
        let byte = base as usize + row * stride;
        let ptr = unsafe {
            self.blob.buffer.contents().as_ptr().add(byte) as *const u8
        };
        let bits = u16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }]);
        bf16_bits_to_f32(bits)
    }
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32) * 2f32.powi(-24);
        return if sign == 1 { -val } else { val };
    }
    if exp == 0x1f {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

/// Histogram q8 row scales for chunk; also show scales the old f32_q8_linear kernel would read.
pub fn log_q8_chunk_scale_histogram(w: &Q8LinearGpu, k_dim: usize, hidden: usize) {
    let rows = w.out_dim;
    let mut ok = 0usize;
    let mut huge = 0usize;
    let mut max_scale = 0.0f32;
    for r in 0..rows {
        let s = w.row_scale_f32(r);
        max_scale = max_scale.max(s.abs());
        if s.is_finite() && s.abs() < 1e3 {
            ok += 1;
        } else {
            huge += 1;
        }
    }
    eprintln!(
        "q8 chunk scales (correct row_stride={}): rows={rows} ok={ok} huge={huge} max_abs={max_scale:.6}",
        w.row_stride()
    );

    // Old kernel: row index = hidden col, row_stride = 2 + k_dim (vocab chunk size).
    let wrong_stride = 2 + k_dim;
    let wrong_rows = hidden.min(rows);
    let mut wrong_huge = 0usize;
    let mut wrong_max = 0.0f32;
    let (_, base) = w.weight_buffer();
    for col in 0..wrong_rows {
        let byte = base as usize + col * wrong_stride;
        let ptr = unsafe {
            w.blob.buffer.contents().as_ptr().add(byte) as *const u8
        };
        let bits = u16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }]);
        let s = bf16_bits_to_f32(bits);
        wrong_max = wrong_max.max(s.abs());
        if !s.is_finite() || s.abs() >= 1e3 {
            wrong_huge += 1;
        }
    }
    eprintln!(
        "q8 chunk scales (old f32_q8_linear indexing, stride={wrong_stride}): sampled={wrong_rows} huge={wrong_huge} max_abs={wrong_max:.6e}"
    );
}

pub fn parse_kind(s: &str) -> Result<QuantKind, Error> {
    match s {
        "q4_block" => Ok(QuantKind::Q4Block),
        "nvfp4_block" => Ok(QuantKind::Nvfp4Block),
        "q8_row" => Ok(QuantKind::Q8Row),
        "raw" => Ok(QuantKind::Raw),
        _ => Err(Error::Format("unknown dgq tensor kind")),
    }
}

pub fn load_q8_linear(store: &DgqStore, blob: Arc<DgqGpuBlob>, name: &str) -> Result<Q8LinearGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if parse_kind(&entry.meta.kind)? != QuantKind::Q8Row {
        return Err(Error::Format("expected q8_row linear"));
    }
    if entry.meta.shape.len() != 2 {
        return Err(Error::Format("q8 linear expects rank 2"));
    }
    let out = entry.meta.shape[0] as usize;
    let inp = entry.meta.shape[1] as usize;
    Ok(Q8LinearGpu::from_entry(blob, entry.meta.offset, out, inp))
}

pub fn load_block_linear(store: &DgqStore, blob: Arc<DgqGpuBlob>, name: &str) -> Result<Q4LinearGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    let kind = parse_kind(&entry.meta.kind)?;
    if kind != QuantKind::Q4Block && kind != QuantKind::Nvfp4Block {
        return Err(Error::Format("expected q4_block or nvfp4_block linear"));
    }
    if entry.meta.shape.len() != 2 {
        return Err(Error::Format("block linear expects rank 2"));
    }
    let out = entry.meta.shape[0] as usize;
    let inp = entry.meta.shape[1] as usize;
    Ok(Q4LinearGpu::from_entry(blob, entry.meta.offset, out, inp, kind))
}

pub fn load_q4_linear(store: &DgqStore, blob: Arc<DgqGpuBlob>, name: &str) -> Result<Q4LinearGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if parse_kind(&entry.meta.kind)? != QuantKind::Q4Block {
        return Err(Error::Format("expected q4_block linear"));
    }
    if entry.meta.shape.len() != 2 {
        return Err(Error::Format("q4 linear expects rank 2"));
    }
    let out = entry.meta.shape[0] as usize;
    let inp = entry.meta.shape[1] as usize;
    Ok(Q4LinearGpu::from_entry(
        blob,
        entry.meta.offset,
        out,
        inp,
        QuantKind::Q4Block,
    ))
}

pub fn load_block_expert_stack(
    store: &DgqStore,
    blob: Arc<DgqGpuBlob>,
    name: &str,
) -> Result<Q4ExpertStackGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    let kind = parse_kind(&entry.meta.kind)?;
    if kind != QuantKind::Q4Block && kind != QuantKind::Nvfp4Block {
        return Err(Error::Format("expected q4_block or nvfp4_block expert stack"));
    }
    if entry.meta.shape.len() != 3 {
        return Err(Error::Format("block expert expects rank 3"));
    }
    Ok(Q4ExpertStackGpu {
        kind,
        blob,
        byte_offset: entry.meta.offset,
        experts: entry.meta.shape[0] as usize,
        out_dim: entry.meta.shape[1] as usize,
        in_dim: entry.meta.shape[2] as usize,
    })
}

pub fn load_q4_expert_stack(
    store: &DgqStore,
    blob: Arc<DgqGpuBlob>,
    name: &str,
) -> Result<Q4ExpertStackGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if parse_kind(&entry.meta.kind)? != QuantKind::Q4Block {
        return Err(Error::Format("expected q4_block expert stack"));
    }
    if entry.meta.shape.len() != 3 {
        return Err(Error::Format("q4 expert expects rank 3"));
    }
    Ok(Q4ExpertStackGpu {
        kind: QuantKind::Q4Block,
        blob,
        byte_offset: entry.meta.offset,
        experts: entry.meta.shape[0] as usize,
        out_dim: entry.meta.shape[1] as usize,
        in_dim: entry.meta.shape[2] as usize,
    })
}

pub fn load_raw_view(store: &DgqStore, blob: Arc<DgqGpuBlob>, name: &str) -> Result<RawBlobView, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if parse_kind(&entry.meta.kind)? != QuantKind::Raw {
        return Err(Error::Format("expected raw tensor"));
    }
    let numel: usize = entry.meta.shape.iter().product::<i64>() as usize;
    Ok(RawBlobView {
        blob,
        byte_offset: entry.meta.offset,
        byte_len: entry.meta.byte_len,
        numel,
    })
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod q4_gpu_tests {
    use super::*;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::linear::f32_q4_linear_gpu_bufs;

    #[test]
    fn q4_gpu_linear_matches_cpu_dequant() {
        let dgq_dir = std::path::Path::new("/tmp/quantized-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q4 = load_q4_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.layers.0.self_attn.q_proj.weight",
        )
        .expect("q4 view");

        let m = 4usize;
        let k = q4.in_dim;
        let n = q4.out_dim;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001 - 0.5).sin() * 0.01).collect();

        let f32_w = store
            .tensor_f32("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("dequant");
        let mut cpu_out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * k + p] * f32_w[col * k + p];
                }
                cpu_out[row * n + col] = sum;
            }
        }

        let mut pool = BufferPool::new();
        let prod = crate::kernels::sub::variant::KernelVariant::PRODUCTION;
        let pipeline = crate::kernels::sub::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::Q4Affine,
            prod,
        )
        .expect("pipeline");
        let nvfp4_pipeline = crate::kernels::sub::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::NvFp4,
            prod,
        )
        .expect("nvfp4 pipeline");
        let mut batch = GpuBatch::begin(&ctx.queue, &mut pool, &ctx.device).expect("batch");
        let buf_a = batch.alloc_f32(&a).expect("a");
        let buf_c = f32_q4_linear_gpu_bufs(
            &mut batch,
            &pipeline,
            &nvfp4_pipeline,
            &buf_a,
            &q4,
            m,
            k,
            n,
        )
        .expect("gemm");
        let mut gpu_out = vec![0.0f32; m * n];
        batch.register_read(buf_c, &mut gpu_out);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("q4 gpu vs cpu dequant max_err={max_err:.6}");
        assert!(max_err < 0.05, "max_err={max_err}");
    }

    #[test]
    fn q8_gpu_linear_matches_cpu_dequant() {
        let dgq_dir = std::path::Path::new("/tmp/quantized-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q8 = load_q8_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.embed_tokens.weight",
        )
        .expect("q8 embed");
        let q8 = q8.row_slice(100, 8);

        let m = 4usize;
        let k = q8.in_dim;
        let n = q8.out_dim;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001 - 0.5).sin() * 0.01).collect();

        let f32_w = store
            .tensor_f32("model.decoder.embed_tokens.weight")
            .expect("dequant");
        let row_off = 100usize;
        let mut cpu_out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * k + p] * f32_w[(row_off + col) * k + p];
                }
                cpu_out[row * n + col] = sum;
            }
        }

        let mut pool = BufferPool::new();
        let prod = crate::kernels::sub::variant::KernelVariant::PRODUCTION;
        let pipeline =
            crate::kernels::sub::gemm_q8_linear_f32::pipeline_for(&ctx, prod).expect("pipeline");
        let mut batch = GpuBatch::begin(&ctx.queue, &mut pool, &ctx.device).expect("batch");
        let buf_a = batch.alloc_f32(&a).expect("a");
        let buf_c =
            crate::metal::linear::f32_q8_linear_gpu_bufs(&mut batch, &pipeline, &buf_a, &q8, m, k, n)
                .expect("gemm");
        let mut gpu_out = vec![0.0f32; m * n];
        batch.register_read(buf_c, &mut gpu_out);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("q8 gpu vs cpu dequant max_err={max_err:.6}");
        assert!(max_err < 0.05, "max_err={max_err}");
    }

    #[test]
    fn nvfp4_gpu_linear_matches_cpu_dequant() {
        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q4 = load_block_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.layers.0.self_attn.q_proj.weight",
        )
        .expect("nvfp4 view");
        assert!(q4.is_nvfp4());

        let m = 4usize;
        let k = q4.in_dim;
        let n = q4.out_dim;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001 - 0.5).sin() * 0.01).collect();

        let f32_w = store
            .tensor_f32("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("dequant");
        let mut cpu_out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * k + p] * f32_w[col * k + p];
                }
                cpu_out[row * n + col] = sum;
            }
        }

        let mut pool = BufferPool::new();
        let prod = crate::kernels::sub::variant::KernelVariant::PRODUCTION;
        let q4_pipeline = crate::kernels::sub::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::Q4Affine,
            prod,
        )
        .expect("pipeline");
        let nvfp4_pipeline = crate::kernels::sub::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::NvFp4,
            prod,
        )
        .expect("nvfp4 pipeline");
        let mut batch = GpuBatch::begin(&ctx.queue, &mut pool, &ctx.device).expect("batch");
        let buf_a = batch.alloc_f32(&a).expect("a");
        let buf_c = f32_q4_linear_gpu_bufs(
            &mut batch,
            &q4_pipeline,
            &nvfp4_pipeline,
            &buf_a,
            &q4,
            m,
            k,
            n,
        )
        .expect("gemm");
        let mut gpu_out = vec![0.0f32; m * n];
        batch.register_read(buf_c, &mut gpu_out);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("nvfp4 gpu vs cpu dequant max_err={max_err:.6}");
        assert!(max_err < 0.08, "max_err={max_err}");
    }

    #[test]
    fn nvfp4_gpu_dequant_matrix_matches_cpu() {
        use crate::metal::mps_gemm::dispatch_dequant_nvfp4_matrix;
        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q4 = load_block_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.layers.0.self_attn.q_proj.weight",
        )
        .expect("nvfp4 view");
        let cpu = store
            .tensor_f32("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("cpu dequant");

        let prod = crate::kernels::sub::variant::KernelVariant::PRODUCTION;
        let pipeline = crate::kernels::sub::dequant_block_matrix::pipeline_for(
            &ctx,
            crate::kernels::sub::QuantFormat::NvFp4,
            prod,
        )
        .expect("pipeline");
        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin(&ctx.queue, &mut pool, &ctx.device).expect("batch");
        let buf_out = batch.alloc_f32_out(cpu.len()).expect("out");
        dispatch_dequant_nvfp4_matrix(batch.encoder(), &pipeline, &q4, &buf_out);
        let mut gpu = vec![0.0f32; cpu.len()];
        batch.register_read(buf_out, &mut gpu);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let err = (a - b).abs();
            if err > max_err {
                max_err = err;
                worst = (i / q4.in_dim, i % q4.in_dim, *a, *b);
            }
        }
        eprintln!(
            "nvfp4 gpu dequant matrix max_err={max_err:.6} worst=({},{}) cpu={:.6} gpu={:.6}",
            worst.0, worst.1, worst.2, worst.3
        );
        assert!(max_err < 0.01, "max_err={max_err}");
    }
}
