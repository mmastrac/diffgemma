//! GPU port of the MTP draft head: same math as `mtp_head`, one command
//! buffer per draft token, weights resident as bf16, activations f32.
//! Kernels live in `mtp_head_gpu.metal`; per-token host work is the embed-row
//! write, the logits argmax, and the recurrent hidden copy.
//!
//! Experiment infra, retained after the free-run drafting line was cut
//! (see ARCHITECTURE.md Negative Knowledge): only the ignored `mtp_*` probe
//! tests exercise it.
#![allow(dead_code)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};
use std::path::Path;

use crate::Error;
use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::mtp_head::{BACKBONE_HID, BackboneKv, HEAD_HID, TARGET_EMBED_SCALE};
use crate::safetensors::SafetensorsFile;

const SHADER: &str = include_str!("mtp_head_gpu.metal");
const N_Q_HEADS: usize = 16;
const FFW: usize = 8192;
const VOCAB: usize = 262144;
const EPS: f32 = 1e-6;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

#[repr(C)]
struct AttnDims {
    n_q: u32,
    n_kv: u32,
    hd: u32,
    seq: u32,
    kv_len: u32,
}

struct GpuLayer {
    input_ln: Buf,
    post_attn_ln: Buf,
    pre_ffw_ln: Buf,
    post_ffw_ln: Buf,
    layer_scalar: f32,
    q_proj: Buf,
    q_norm: Buf,
    o_proj: Buf,
    gate: Buf,
    up: Buf,
    down: Buf,
    head_dim: usize,
    is_full: bool,
}

struct Scratch {
    x: Buf,
    h: Buf,
    hn: Buf,
    q: Buf,
    attn: Buf,
    a: Buf,
    an: Buf,
    g: Buf,
    u: Buf,
    m: Buf,
    logits: Buf,
    hidden_next: Buf,
}

struct GpuKv {
    k_swa: Buf,
    v_swa: Buf,
    k_full: Buf,
    v_full: Buf,
    seq: usize,
}

pub struct MtpHeadGpu {
    ctx: MetalContext,
    pool: BufferPool,
    p_matvec: ComputePipeline,
    p_rmsnorm: ComputePipeline,
    p_rope: ComputePipeline,
    p_attend: ComputePipeline,
    p_gelu_mul: ComputePipeline,
    p_add_scale: ComputePipeline,
    layers: Vec<GpuLayer>,
    final_norm: Buf,
    embed: Buf,
    pre_projection: Buf,
    post_projection: Buf,
    scratch: Scratch,
    kv: Option<GpuKv>,
}

fn write_bytes(buf: &Buf, bytes: &[u8]) {
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents().as_ptr() as *mut u8, bytes.len());
    }
}

fn set_bytes<T>(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(value as *const T as *mut std::ffi::c_void).unwrap(),
            std::mem::size_of::<T>(),
            index,
        );
    }
}

fn tensor_bytes<'a>(st: &'a SafetensorsFile, name: &str) -> Result<&'a [u8], Error> {
    let info = st
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or(Error::Runtime("missing MTP head tensor"))?;
    Ok(st.data(info))
}

/// Matrix weights stay bf16 (the matvec kernel widens per element).
fn upload_bf16(
    pool: &mut BufferPool,
    ctx: &MetalContext,
    st: &SafetensorsFile,
    name: &str,
) -> Result<Buf, Error> {
    let bytes = tensor_bytes(st, name)?;
    let buf = pool
        .allocate(&ctx.device, bytes.len())
        .ok_or(Error::Gpu("MTP weight alloc failed"))?;
    write_bytes(&buf, bytes);
    Ok(buf)
}

/// Norm weight vectors widen to f32 (the rmsnorm kernel reads f32).
fn upload_f32(
    pool: &mut BufferPool,
    ctx: &MetalContext,
    st: &SafetensorsFile,
    name: &str,
) -> Result<Buf, Error> {
    let f: Vec<f32> = tensor_bytes(st, name)?
        .chunks_exact(2)
        .map(|c| crate::shaders::cpu::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let buf = pool
        .allocate(&ctx.device, f.len() * 4)
        .ok_or(Error::Gpu("MTP norm alloc failed"))?;
    BufferPool::write_f32(&buf, &f);
    Ok(buf)
}

impl MtpHeadGpu {
    pub fn load(safetensors_path: &Path) -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let library = ctx.compile_library(SHADER)?;
        let pipe = |entry: &str| ctx.compile_kernel_from_library(&library, entry);
        let st = SafetensorsFile::open(safetensors_path)?;
        let mut pool = BufferPool::new();

        let mut layers = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("model.layers.{i}");
            let is_full = i == 3;
            let scalar_bytes = tensor_bytes(&st, &format!("{p}.layer_scalar"))?;
            layers.push(GpuLayer {
                input_ln: upload_f32(&mut pool, &ctx, &st, &format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: upload_f32(
                    &mut pool,
                    &ctx,
                    &st,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                pre_ffw_ln: upload_f32(
                    &mut pool,
                    &ctx,
                    &st,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                )?,
                post_ffw_ln: upload_f32(
                    &mut pool,
                    &ctx,
                    &st,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                )?,
                layer_scalar: crate::shaders::cpu::bf16_to_f32(u16::from_le_bytes([
                    scalar_bytes[0],
                    scalar_bytes[1],
                ])),
                q_proj: upload_bf16(&mut pool, &ctx, &st, &format!("{p}.self_attn.q_proj.weight"))?,
                q_norm: upload_f32(&mut pool, &ctx, &st, &format!("{p}.self_attn.q_norm.weight"))?,
                o_proj: upload_bf16(&mut pool, &ctx, &st, &format!("{p}.self_attn.o_proj.weight"))?,
                gate: upload_bf16(&mut pool, &ctx, &st, &format!("{p}.mlp.gate_proj.weight"))?,
                up: upload_bf16(&mut pool, &ctx, &st, &format!("{p}.mlp.up_proj.weight"))?,
                down: upload_bf16(&mut pool, &ctx, &st, &format!("{p}.mlp.down_proj.weight"))?,
                head_dim: if is_full { 512 } else { 256 },
                is_full,
            });
        }
        let final_norm = upload_f32(&mut pool, &ctx, &st, "model.norm.weight")?;
        let embed = upload_bf16(&mut pool, &ctx, &st, "model.embed_tokens.weight")?;
        let pre_projection = upload_bf16(&mut pool, &ctx, &st, "pre_projection.weight")?;
        let post_projection = upload_bf16(&mut pool, &ctx, &st, "post_projection.weight")?;

        let alloc = |pool: &mut BufferPool, elems: usize| -> Result<Buf, Error> {
            pool.allocate(&ctx.device, elems * 4)
                .ok_or(Error::Gpu("MTP scratch alloc failed"))
        };
        let scratch = Scratch {
            x: alloc(&mut pool, 2 * BACKBONE_HID)?,
            h: alloc(&mut pool, HEAD_HID)?,
            hn: alloc(&mut pool, HEAD_HID)?,
            q: alloc(&mut pool, N_Q_HEADS * 512)?,
            attn: alloc(&mut pool, N_Q_HEADS * 512)?,
            a: alloc(&mut pool, HEAD_HID)?,
            an: alloc(&mut pool, HEAD_HID)?,
            g: alloc(&mut pool, FFW)?,
            u: alloc(&mut pool, FFW)?,
            m: alloc(&mut pool, HEAD_HID)?,
            logits: alloc(&mut pool, VOCAB)?,
            hidden_next: alloc(&mut pool, BACKBONE_HID)?,
        };

        Ok(Self {
            p_matvec: pipe("mtp_matvec_bf16")?,
            p_rmsnorm: pipe("mtp_rmsnorm")?,
            p_rope: pipe("mtp_rope")?,
            p_attend: pipe("mtp_attend")?,
            p_gelu_mul: pipe("mtp_gelu_mul")?,
            p_add_scale: pipe("mtp_add_scale")?,
            layers,
            final_norm,
            embed,
            pre_projection,
            post_projection,
            scratch,
            kv: None,
            ctx,
            pool,
        })
    }

    pub fn upload_kv(&mut self, kv: &BackboneKv) -> Result<(), Error> {
        let up = |pool: &mut BufferPool, data: &[f32]| -> Result<Buf, Error> {
            let buf = pool
                .allocate(&self.ctx.device, data.len() * 4)
                .ok_or(Error::Gpu("MTP kv alloc failed"))?;
            BufferPool::write_f32(&buf, data);
            Ok(buf)
        };
        let mut pool = std::mem::replace(&mut self.pool, BufferPool::new());
        let kv = GpuKv {
            k_swa: up(&mut pool, &kv.k_swa)?,
            v_swa: up(&mut pool, &kv.v_swa)?,
            k_full: up(&mut pool, &kv.k_full)?,
            v_full: up(&mut pool, &kv.v_full)?,
            seq: kv.seq,
        };
        self.pool = pool;
        self.kv = Some(kv);
        Ok(())
    }

    /// GPU mirror of `mtp_head::draft_tokens`; requires `upload_kv` first.
    /// `stop_tok` ends the draft after emitting that token (eos for
    /// whole-answer drafting).
    pub fn draft_tokens(
        &self,
        pos: usize,
        init_hidden: &[f32],
        init_tok: u32,
        k_draft: usize,
        stop_tok: Option<u32>,
        embed_fn: &mut dyn FnMut(u32) -> Option<Vec<f32>>,
    ) -> Result<Vec<u32>, Error> {
        self.draft_tokens_conf(pos, init_hidden, init_tok, k_draft, stop_tok, embed_fn, None)
    }

    /// As `draft_tokens`, optionally recording each drafted token's softmax
    /// probability (the head's own acceptance-confidence signal).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_tokens_conf(
        &self,
        pos: usize,
        init_hidden: &[f32],
        init_tok: u32,
        k_draft: usize,
        stop_tok: Option<u32>,
        embed_fn: &mut dyn FnMut(u32) -> Option<Vec<f32>>,
        mut confidences: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, Error> {
        let kv = self.kv.as_ref().ok_or(Error::Runtime("upload_kv first"))?;
        let kv_len = pos + 1;
        assert!(kv_len <= kv.seq);
        BufferPool::write_f32_at_offset(&self.scratch.x, BACKBONE_HID * 4, init_hidden);
        let mut tok = init_tok;
        let mut out = Vec::new();
        let timing = std::env::var("DGQ_MTP_TIMING").is_ok();
        let (mut t_pre, mut t_gpu, mut t_post) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..k_draft {
            let t0 = std::time::Instant::now();
            let Some(mut emb) = embed_fn(tok) else { break };
            for v in emb.iter_mut() {
                *v *= TARGET_EMBED_SCALE;
            }
            BufferPool::write_f32(&self.scratch.x, &emb);
            let t1 = std::time::Instant::now();
            self.encode_forward(kv, pos as u32, kv_len as u32)?;
            let t2 = std::time::Instant::now();
            let mut logits = vec![0.0f32; VOCAB];
            BufferPool::read_f32(&self.scratch.logits, &mut logits);
            let mut best = f32::NEG_INFINITY;
            for (i, &s) in logits.iter().enumerate() {
                if s > best {
                    best = s;
                    tok = i as u32;
                }
            }
            if let Some(confs) = confidences.as_deref_mut() {
                let denom: f32 = logits.iter().map(|&s| (s - best).exp()).sum();
                confs.push(1.0 / denom);
            }
            let mut hidden = vec![0.0f32; BACKBONE_HID];
            BufferPool::read_f32(&self.scratch.hidden_next, &mut hidden);
            BufferPool::write_f32_at_offset(&self.scratch.x, BACKBONE_HID * 4, &hidden);
            out.push(tok);
            t_pre += (t1 - t0).as_secs_f64();
            t_gpu += (t2 - t1).as_secs_f64();
            t_post += t2.elapsed().as_secs_f64();
            if stop_tok == Some(tok) {
                break;
            }
        }
        if timing && !out.is_empty() {
            let n = out.len() as f64;
            eprintln!(
                "mtp-gpu timing: embed+write {:.2}ms  encode+wait {:.2}ms  readback+argmax {:.2}ms per token",
                1e3 * t_pre / n,
                1e3 * t_gpu / n,
                1e3 * t_post / n
            );
        }
        Ok(out)
    }

    /// Timing probe: encode the forward chain `n` times in one command
    /// buffer (results are garbage; measures in-chain vs per-buffer cost).
    pub fn debug_time_chains(&self, n: usize) -> Result<f64, Error> {
        let kv = self.kv.as_ref().ok_or(Error::Runtime("upload_kv first"))?;
        let kv_len = (kv.seq / 2) as u32;
        let started = std::time::Instant::now();
        self.encode_forward_n(kv, kv_len.saturating_sub(1), kv_len, n)?;
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    fn encode_forward(&self, kv: &GpuKv, pos: u32, kv_len: u32) -> Result<(), Error> {
        self.encode_forward_n(kv, pos, kv_len, 1)
    }

    /// Timing probe: `n` repeats of one op at production sizes.
    pub fn debug_time_op(&self, op: &str, n: usize) -> Result<f64, Error> {
        let kv = self.kv.as_ref().ok_or(Error::Runtime("upload_kv first"))?;
        let l3 = &self.layers[3];
        let started = std::time::Instant::now();
        let cmd = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Gpu("command buffer failed"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(Error::Gpu("encoder failed"))?;
        let matvec = |w: &Buf, x: &Buf, y: &Buf, out_dim: usize, in_dim: usize| {
            enc.setComputePipelineState(&self.p_matvec.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(w), 0, 0);
                enc.setBuffer_offset_atIndex(Some(x), 0, 1);
                enc.setBuffer_offset_atIndex(Some(y), 0, 2);
            }
            set_bytes(&enc, &[out_dim as u32, in_dim as u32], 3);
            let grid = MTLSize {
                width: out_dim.div_ceil(8),
                height: 1,
                depth: 1,
            };
            let tgs = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgs);
        };
        let s = &self.scratch;
        for _ in 0..n {
            match op {
                "lm" => matvec(&self.embed, &s.hn, &s.logits, VOCAB, HEAD_HID),
                "gate" => matvec(&l3.gate, &s.hn, &s.g, FFW, HEAD_HID),
                "down" => matvec(&l3.down, &s.g, &s.m, HEAD_HID, FFW),
                "qproj" => matvec(&l3.q_proj, &s.hn, &s.q, N_Q_HEADS * 512, HEAD_HID),
                "preproj" => matvec(&self.pre_projection, &s.x, &s.h, HEAD_HID, 2 * BACKBONE_HID),
                "postproj" => {
                    matvec(&self.post_projection, &s.hn, &s.hidden_next, BACKBONE_HID, HEAD_HID)
                }
                "attend" => {
                    enc.setComputePipelineState(&self.p_attend.pipeline);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&s.q), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&kv.k_full), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&kv.v_full), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&s.attn), 0, 3);
                    }
                    set_bytes(
                        &enc,
                        &AttnDims {
                            n_q: N_Q_HEADS as u32,
                            n_kv: 2,
                            hd: 512,
                            seq: kv.seq as u32,
                            kv_len: kv.seq as u32,
                        },
                        4,
                    );
                    let grid = MTLSize {
                        width: N_Q_HEADS.div_ceil(8),
                        height: 1,
                        depth: 1,
                    };
                    let tgs = MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgs);
                }
                _ => unreachable!(),
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    /// Timing probe: `n` dependent trivial dispatches (add_scale on h).
    pub fn debug_time_trivial(&self, n: usize) -> Result<f64, Error> {
        let started = std::time::Instant::now();
        let cmd = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Gpu("command buffer failed"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(Error::Gpu("encoder failed"))?;
        for _ in 0..n {
            enc.setComputePipelineState(&self.p_add_scale.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.scratch.h), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.scratch.hn), 0, 1);
            }
            set_bytes(&enc, &(HEAD_HID as u32), 2);
            set_bytes(&enc, &1.0f32, 3);
            let one = MTLSize {
                width: 4,
                height: 1,
                depth: 1,
            };
            let tgs = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(one, tgs);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    fn encode_forward_n(&self, kv: &GpuKv, pos: u32, kv_len: u32, n: usize) -> Result<(), Error> {
        let cmd = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Gpu("command buffer failed"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(Error::Gpu("encoder failed"))?;

        let dispatch = |pipeline: &ComputePipeline, threads: usize| {
            enc.setComputePipelineState(&pipeline.pipeline);
            let tg = 256usize.min(threads.max(1));
            let grid = MTLSize {
                width: threads.div_ceil(tg),
                height: 1,
                depth: 1,
            };
            let tgs = MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgs);
        };
        let bind = |bufs: &[&Buf]| {
            for (i, b) in bufs.iter().enumerate() {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(b), 0, i);
                }
            }
        };
        let matvec = |w: &Buf, x: &Buf, y: &Buf, out_dim: usize, in_dim: usize| {
            assert_eq!(in_dim % 4, 0);
            bind(&[w, x, y]);
            set_bytes(&enc, &[out_dim as u32, in_dim as u32], 3);
            // Simdgroup-per-row: 8 rows per 256-thread threadgroup.
            enc.setComputePipelineState(&self.p_matvec.pipeline);
            let grid = MTLSize {
                width: out_dim.div_ceil(8),
                height: 1,
                depth: 1,
            };
            let tgs = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgs);
        };
        let rmsnorm = |x: &Buf, w: &Buf, y: &Buf, rows: usize, hidden: usize| {
            bind(&[x, w, y]);
            set_bytes(&enc, &[rows as u32, hidden as u32], 3);
            set_bytes(&enc, &EPS, 4);
            dispatch(&self.p_rmsnorm, rows);
        };
        let add_scale = |h: &Buf, x: &Buf, len: usize, scale: f32| {
            bind(&[h, x]);
            set_bytes(&enc, &(len as u32), 2);
            set_bytes(&enc, &scale, 3);
            dispatch(&self.p_add_scale, len);
        };

        let s = &self.scratch;
        for _ in 0..n {
        matvec(&self.pre_projection, &s.x, &s.h, HEAD_HID, 2 * BACKBONE_HID);
        for l in &self.layers {
            let hd = l.head_dim;
            rmsnorm(&s.h, &l.input_ln, &s.hn, 1, HEAD_HID);
            matvec(&l.q_proj, &s.hn, &s.q, N_Q_HEADS * hd, HEAD_HID);
            rmsnorm(&s.q, &l.q_norm, &s.q, N_Q_HEADS, hd);
            let (rot, theta) = if l.is_full {
                (hd / 4, 1e6f32)
            } else {
                (hd, 1e4f32)
            };
            bind(&[&s.q]);
            set_bytes(&enc, &[N_Q_HEADS as u32, hd as u32, rot as u32, pos], 1);
            set_bytes(&enc, &theta, 2);
            dispatch(&self.p_rope, N_Q_HEADS * rot / 2);
            let (k, v, n_kv) = if l.is_full {
                (&kv.k_full, &kv.v_full, 2u32)
            } else {
                (&kv.k_swa, &kv.v_swa, 8u32)
            };
            bind(&[&s.q, k, v, &s.attn]);
            set_bytes(
                &enc,
                &AttnDims {
                    n_q: N_Q_HEADS as u32,
                    n_kv,
                    hd: hd as u32,
                    seq: kv.seq as u32,
                    kv_len,
                },
                4,
            );
            enc.setComputePipelineState(&self.p_attend.pipeline);
            let grid = MTLSize {
                width: N_Q_HEADS.div_ceil(8),
                height: 1,
                depth: 1,
            };
            let tgs = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgs);
            matvec(&l.o_proj, &s.attn, &s.a, HEAD_HID, N_Q_HEADS * hd);
            rmsnorm(&s.a, &l.post_attn_ln, &s.an, 1, HEAD_HID);
            add_scale(&s.h, &s.an, HEAD_HID, 1.0);
            rmsnorm(&s.h, &l.pre_ffw_ln, &s.hn, 1, HEAD_HID);
            matvec(&l.gate, &s.hn, &s.g, FFW, HEAD_HID);
            matvec(&l.up, &s.hn, &s.u, FFW, HEAD_HID);
            bind(&[&s.g, &s.u]);
            set_bytes(&enc, &(FFW as u32), 2);
            dispatch(&self.p_gelu_mul, FFW);
            matvec(&l.down, &s.g, &s.m, HEAD_HID, FFW);
            rmsnorm(&s.m, &l.post_ffw_ln, &s.an, 1, HEAD_HID);
            add_scale(&s.h, &s.an, HEAD_HID, l.layer_scalar);
        }
        rmsnorm(&s.h, &self.final_norm, &s.hn, 1, HEAD_HID);
        matvec(&self.embed, &s.hn, &s.logits, VOCAB, HEAD_HID);
        matvec(&self.post_projection, &s.hn, &s.hidden_next, BACKBONE_HID, HEAD_HID);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }
}
