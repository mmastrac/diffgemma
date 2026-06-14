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
mod embed;
mod lm_head;
mod kv_cache;
mod memory;
mod moe;
mod mps_gemm;
mod pipeline_cache;
mod probe;
mod dgq_gpu;
mod bench_gemm;
mod router;
mod sampler;
mod sampler_kernels;
mod step_config;
mod step_generate;
mod step_kv;
mod step_m0;
mod step_logits_dump;
mod step_preamble_dump;
mod step_attn_dump;
mod step_moe_dump;
mod step_moe_single_dump;
mod step_kernel;
mod self_conditioning;
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
pub use sampler::{sampler_step_gpu, upload_logits_gpu};
pub use step_kernel::{
    bench_step_kernel, build_step_runtime, hello_chat_prefill_token_ids, logits_finite_check_enabled,
    run_denoise_steps, run_single_denoise_step, run_step_attn_layer_capture, run_step_forward,
    run_step_layer_hidden_probe, run_step_moe_layer_capture, run_step_moe_single_expert_capture,
    run_step_preamble_capture,
    run_step_probe, run_step_smoke, run_embed_row_gpu, run_step_q4_mps_parity, step_use_mps_q4_default,
    step_use_sc_gemm_default,
    DenoiseStepStats, trace_entropy_enabled, LayerAttnCapture, LayerMoeCapture,
    PreambleCapture, SingleExpertMoeCapture,
    StepBenchResult, StepFinishMode, StepForwardOutput, StepProbeResult, StepQ4MpsParityResult,
    StepSmokeConfig, StepSmokeResult, CANVAS,
};
pub use step_preamble_dump::{
    hidden_cosine as preamble_hidden_cosine, run_step_preamble_dump, write_step_preamble_dump,
    StepPreambleDump,
};
pub use step_attn_dump::{
    hidden_cosine, run_step_attn_layer_dump, write_step_attn_layer_dump, StepAttnLayerDump,
};
pub use step_moe_dump::{run_step_moe_layer_dump, write_step_moe_layer_dump, StepMoeLayerDump};
pub use step_moe_single_dump::{
    run_step_moe_single_expert_dump, write_step_moe_single_expert_dump, StepMoeSingleExpertDump,
};
pub use step_logits_dump::{
    parse_positions, run_step_layer_hidden_dump, run_step_logits_dump, write_step_layer_hidden_dump,
    write_step_logits_dump, StepLayerHiddenDump, StepLogitsDump,
};
pub use step_m0::{run_step_parity, run_step_verify, M0VerifyResult, StepParityConfig, StepParityResult};
pub use step_config::{log_validated_step_model, validate_step_model, ValidatedStepModel};
pub use step_generate::{generate_monolithic, generate_with_session, StepGenerateConfig, StepGenerateSession};
pub use step_kv::{
    extend_monolithic_kv, monolithic_kv_prefix_max_diff, prefill_monolithic_kv_with_cache,
    run_step_kv_audit, run_step_kv_mps_parity, run_step_attn_probe, MonolithicEncoderCache, StepKvAuditResult,
    StepKvMpsParityResult,
};
pub use telemetry::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
