//! Grouped MoE expert GEMM (q4 weights): kernels, the host-side bucket layout,
//! and a CPU oracle the tier-1 test pins them to.

use crate::config::Error;
use gpukit::cuda::{Context, DeviceBuffer, KernelArgs, cached_source_kernel};

/// The grouped-MoE kernel source, compiled by NVRTC on first use.
pub const MOE_KERNELS: &str = include_str!("moe_grouped.cu");

/// Block tile of the grouped kernel; must match the #defines in moe_grouped.cu.
pub const BM: usize = 32;
pub const BN: usize = 64;

/// Q4 row stride in bytes (32 weights per group: 4 bytes of scales + 16 bytes
/// of nibbles). Must match `dgemm::format::layout::q4_row_bytes`.
pub fn q4_row_bytes(k_dim: usize) -> usize {
    k_dim.div_ceil(32) * 20
}

/// One expert bucket: the expert index and the tokens routed to it.
#[derive(Debug, Clone)]
pub struct Bucket {
    pub expert: usize,
    pub tokens: Vec<u32>,
}

/// The expert-major flat row layout a grouped launch reads: per-row token id,
/// per-row routing weight, and the row offset of every bucket.
#[derive(Debug, Clone)]
pub struct GroupedPlan {
    pub tok_idx: Vec<u32>,
    pub row_w: Vec<f32>,
    /// `num_jobs + 1` entries; bucket `j` owns rows `starts[j]..starts[j+1]`.
    pub starts: Vec<u32>,
    /// The expert each bucket holds. A bucket with no tokens still occupies a
    /// job slot, so the kernel and the oracle must read the expert index from
    /// here rather than assume job j is expert j.
    pub experts: Vec<u32>,
}

impl GroupedPlan {
    /// Build the plan from per-token expert choices (the router top-k output,
    /// both arrays `[seq * top_k]` in token-major order). Buckets keep expert
    /// order and, within an expert, token order — the same deterministic order
    /// the CPU oracle sums in.
    pub fn new(idx: &[u32], wts: &[f32], seq: usize, top_k: usize, n_experts: usize) -> Self {
        // Per-token weight of each routing entry, so a bucket can record the
        // weight that belongs to its (token, expert) pair.
        let mut entry_w = vec![0.0f32; seq * n_experts];
        for s in 0..seq {
            for k in 0..top_k {
                let e = idx[s * top_k + k] as usize;
                entry_w[s * n_experts + e] = wts[s * top_k + k];
            }
        }
        let mut buckets: Vec<Bucket> = (0..n_experts)
            .map(|expert| Bucket {
                expert,
                tokens: Vec::new(),
            })
            .collect();
        for s in 0..seq {
            for k in 0..top_k {
                let e = idx[s * top_k + k] as usize;
                if e < n_experts {
                    buckets[e].tokens.push(s as u32);
                }
            }
        }
        let mut tok_idx = Vec::new();
        let mut row_w = Vec::new();
        let mut starts = vec![0u32];
        let mut experts = Vec::with_capacity(n_experts);
        for b in &buckets {
            if b.tokens.is_empty() {
                continue;
            }
            experts.push(b.expert as u32);
            for &t in &b.tokens {
                tok_idx.push(t);
                row_w.push(entry_w[t as usize * n_experts + b.expert]);
            }
            starts.push(tok_idx.len() as u32);
        }
        Self {
            tok_idx,
            row_w,
            starts,
            experts,
        }
    }

    pub fn rows(&self) -> usize {
        self.tok_idx.len()
    }

    /// Buckets that hold at least one row. The empty ones are dropped from the
    /// expert list (their job slot disappears) so the kernel never indexes a
    /// value the host did not fill in.
    pub fn num_jobs(&self) -> usize {
        self.experts.len()
    }
}

/// A grouped GEMM shape: the expert-major input rows, the q4 expert weights
/// (`[n_experts, n_dim, k_dim]` rows of q4), and the plan mapping rows back to
/// tokens.
pub struct GroupedGemm {
    pub k_dim: usize,
    pub n_dim: usize,
    pub row_bytes: usize,
    /// NVRTC entry point (`dgq_moe_gate_up` or `dgq_moe_down`).
    pub entry: &'static str,
}

impl GroupedGemm {
    /// `entry` selects the kernel entry; the gate/up and down bodies differ
    /// only in how the A rows are indexed.
    pub fn new(k_dim: usize, n_dim: usize, entry: &'static str) -> Self {
        Self {
            k_dim,
            n_dim,
            row_bytes: q4_row_bytes(k_dim),
            entry,
        }
    }

    /// `c[global_row, :] = a[tok_idx[global_row], :] @ w[expert, :, :]^T`.
    ///
    /// `a` holds `[n_tokens, k]` rows; `rows` is the bucketed (expert-major)
    /// row count and `num_jobs` the number of buckets.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        ctx: &Context,
        a: &DeviceBuffer,
        w_blob: &DeviceBuffer,
        starts: &DeviceBuffer,
        tok_idx: &DeviceBuffer,
        experts: &DeviceBuffer,
        c: &DeviceBuffer,
        rows: usize,
        num_jobs: usize,
    ) -> Result<(), Error> {
        let kernel = cached_source_kernel(MOE_KERNELS, self.entry)?;
        let mut args = KernelArgs::new();
        args.device_ptr(a.device_ptr())
            .device_ptr(w_blob.device_ptr())
            .device_ptr(c.device_ptr())
            .device_ptr(starts.device_ptr())
            .device_ptr(tok_idx.device_ptr())
            .device_ptr(experts.device_ptr())
            .u32(self.k_dim as u32)
            .u32(self.n_dim as u32)
            .u32(num_jobs as u32)
            .u32(self.row_bytes as u32);
        let grid = (self.n_dim.div_ceil(BN) as u32, rows.div_ceil(BM) as u32, 1);
        ctx.launch(&kernel, grid, (128, 1, 1), 0, &mut args)?;
        Ok(())
    }
}

pub fn upload_u32(ctx: &Context, data: &[u32]) -> Result<DeviceBuffer, Error> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let b = DeviceBuffer::alloc(ctx, data.len().max(1) * 4)?;
    b.write_bytes(bytes)?;
    Ok(b)
}

/// CPU oracle: for every bucket, `c[row, :] = a[token, :] @ w[expert, :, :]^T`
/// with `w` already decoded to f32, in bucket order.
pub fn grouped_gemm_cpu(
    a: &[f32],
    w: &[f32],
    plan: &GroupedPlan,
    k_dim: usize,
    n_dim: usize,
) -> Vec<f32> {
    let rows = plan.rows();
    let mut c = vec![0.0f32; rows * n_dim];
    for j in 0..plan.num_jobs() {
        let start = plan.starts[j] as usize;
        let end = plan.starts[j + 1] as usize;
        let w_off = plan.experts[j] as usize * n_dim * k_dim;
        for r in start..end {
            for o in 0..n_dim {
                let mut acc = 0.0f32;
                for k in 0..k_dim {
                    // A is already in bucketed (expert-major) row order: the
                    // down projection reads the gate/up output, which the
                    // grouped GEMM wrote in this order. The gather happens on
                    // the way in, not here.
                    acc += a[r * k_dim + k] * w[w_off + o * k_dim + k];
                }
                c[r * n_dim + o] = acc;
            }
        }
    }
    c
}
