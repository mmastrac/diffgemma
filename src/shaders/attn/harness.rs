//! Shared Fixture→GPU rig for the attention harnesses (`attention_gemm`,
//! `attention_topk`, `attention_flash`). Every GPU oracle / bench in those
//! modules built the same skeleton by hand: `MetalContext::new` → `BufferPool`
//! → q/out/kv(/kvf) buffers filled from the fixture → per-kernel scratch →
//! encode → wait → bf16 readback or min-of-rounds timing. `AttnRig` owns that
//! skeleton once; the harnesses keep only their kernel-specific dims structs
//! and encode bodies.

use crate::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

/// Retained handle to a pooled `MTLBuffer`.
pub(crate) type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Fixture-derived context, standard buffers, and geometry shared by the
/// attention GPU oracles and benches.
pub(crate) struct AttnRig {
    pub ctx: crate::metal::device::MetalContext,
    pool: crate::metal::buffer::BufferPool,
    /// Q: `[canvas][n_q_heads][hd]` bf16 (arena default), filled from `f.q`.
    pub buf_q: Buf,
    /// Output: same shape/format as Q (bf16), zeroed.
    pub buf_out: Buf,
    /// KV: f16 main cache + `kv_slack_slots` zero pad key rows (device tile
    /// loads over-read whole tiles past T).
    pub buf_kv: Buf,
    /// f32 side-ring mirror of the KV cache (same pad), for the FC30 paths.
    /// Present only when built `with_kvf`.
    buf_kvf: Option<Buf>,
    pub canvas: usize,
    pub n_q_heads: usize,
    pub nkv: usize,
    pub hd: usize,
    pub group: usize,
    pub kv_len: usize,
    pub t_total: usize,
    /// Elements between per-token KV rows (K then V).
    pub kstride: usize,
}

impl AttnRig {
    /// Build the rig: fresh `MetalContext`, pool, and the standard buffers
    /// filled from the fixture exactly as the hand-rolled harnesses did.
    /// `kv_slack_slots` is the number of zero pad key rows appended to the KV
    /// cache (8 for the GEMM/top-k tiles, `BK` for flash); `with_kvf` also
    /// allocates + fills the f32 mirror (skip it in benches that never bind
    /// it — the mirror is large at real KV lengths).
    pub(crate) fn from_fixture(
        f: &crate::shaders::attention::Fixture,
        kv_slack_slots: usize,
        with_kvf: bool,
    ) -> Result<Self, Error> {
        use crate::metal::buffer::BufferPool;

        let canvas = f.canvas;
        let n_q_heads = f.n_q_heads;
        let nkv = f.n_kv();
        let hd = f.head_dim();
        let group = n_q_heads / nkv;
        let kv_len = f.kv_len as usize;
        let t_total = kv_len + canvas;
        let kstride = nkv * hd * 2;

        let ctx = crate::metal::device::MetalContext::new()?;
        let mut pool = BufferPool::new();
        let buf_q = pool
            .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
            .ok_or(Error::Gpu("alloc q"))?;
        BufferPool::write_bf16(&buf_q, &crate::shaders::bf16::f32_slice_to_bf16_bits(&f.q));
        let buf_out = pool
            .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
            .ok_or(Error::Gpu("alloc out"))?;
        let buf_kv = pool
            .allocate(
                &ctx.device,
                (f.kvcache.len() + kv_slack_slots * kstride) * 2,
            )
            .ok_or(Error::Gpu("alloc kv"))?;
        {
            let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
            bits.resize(bits.len() + kv_slack_slots * kstride, 0);
            BufferPool::write_bf16(&buf_kv, &bits);
        }
        let buf_kvf = if with_kvf {
            let buf = pool
                .allocate(
                    &ctx.device,
                    (f.kvcache.len() + kv_slack_slots * kstride) * 4,
                )
                .ok_or(Error::Gpu("alloc kvf"))?;
            let mut fs = f.kvcache.clone();
            fs.resize(fs.len() + kv_slack_slots * kstride, 0.0);
            BufferPool::write_f32(&buf, &fs);
            Some(buf)
        } else {
            None
        };
        Ok(Self {
            ctx,
            pool,
            buf_q,
            buf_out,
            buf_kv,
            buf_kvf,
            canvas,
            n_q_heads,
            nkv,
            hd,
            group,
            kv_len,
            t_total,
            kstride,
        })
    }

    /// The f32 side-ring mirror; panics if the rig was built without it.
    pub(crate) fn kvf(&self) -> &Buf {
        self.buf_kvf.as_ref().expect("rig built without kvf")
    }

    /// Per-kernel scratch allocation from the rig's pool (zeroed).
    pub(crate) fn alloc(&mut self, size: usize, label: &'static str) -> Result<Buf, Error> {
        self.pool
            .allocate(&self.ctx.device, size)
            .ok_or(Error::Gpu(label))
    }

    /// One command buffer: `encode` once, end, commit, wait.
    pub(crate) fn run_once(
        &self,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<(), Error> {
        let cmd = self.ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
        encode(&enc);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Shared timing statistic: 6 rounds of one command buffer with `iters`
    /// back-to-back encodes each; round 0 is warm-up (factors out clock ramp),
    /// min over the rest. Returns mean ms per iter of the best round.
    pub(crate) fn time_rounds(
        &self,
        iters: usize,
        mut encode: impl FnMut(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<f64, Error> {
        let mut best = f64::INFINITY;
        for round in 0..6 {
            let t = std::time::Instant::now();
            let cmd = self.ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
            for _ in 0..iters {
                encode(&enc);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if round > 0 {
                best = best.min(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
            }
        }
        Ok(best)
    }

    /// Read back `len` bf16 elements of `buf_out` as f32.
    pub(crate) fn read_out_f32(&self, len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        let ptr = self.buf_out.contents().as_ptr() as *const u16;
        for (i, o) in out.iter_mut().enumerate() {
            *o = crate::shaders::bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) });
        }
        out
    }
}
