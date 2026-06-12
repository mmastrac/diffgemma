//! Metal GPU runtime (macOS only, `metal` feature).

mod attention;
mod attention_batch;
mod batch;
mod batched_kernels;
mod buffer;
mod decoder;
mod decoder_attention;
mod decoder_layer;
mod device;
mod expert_cache;
mod encoder_extend;
mod engine;
mod gemm;
mod kernels;
mod linear;
mod kv_cache;
mod memory;
mod moe;
mod probe;
mod bench_gemm;
mod telemetry;
mod weights;

pub use memory::{
    estimate_decoder_forward, estimate_paged_layer_bytes, estimate_weight_cache,
    log_expert_cache_stats, MemoryEstimate,
};

pub use attention::{GpuAttention, GpuAttentionKernels};
pub use decoder::{
    bench_forward, forward as decoder_forward, load_weight_cache, BenchConfig, GpuDecoderScratch,
};
pub use encoder_extend::{extend_prefill_gpu, prefill_gpu};
pub use weights::GpuDecoderWeightCache;
pub use kv_cache::GpuKvCache;
pub use engine::GpuDecoderEngine;
pub use gemm::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};
pub use probe::{print_probe_result, probe_device, DeviceProbeResult};
pub use bench_gemm::{bench_custom_kernel, bench_mps_oracle, parse_shapes, print_bench_rows};
pub use telemetry::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
