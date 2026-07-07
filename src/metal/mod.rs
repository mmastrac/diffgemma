//! Metal GPU runtime (macOS only, `metal` feature).

mod attention;
mod attention_batch;
pub(crate) mod batch;
mod batched_kernels;
pub(crate) mod buffer;
mod decoder;
mod decoder_attention;
mod decoder_layer;
pub(crate) mod arena_layout;
pub(crate) mod debug_status;
pub(crate) mod device;
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
mod dequant_matrix;
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
mod step_moe_batched_pin_dump;
mod step_moe_route_dump;
mod step_moe_single_dump;
mod step_kernel;
mod step_quant;
mod self_conditioning;
mod telemetry;
mod weights;


pub use dgq_gpu::DgqGpuBlob;
pub use memory::{estimate_decoder_forward, estimate_paged_layer_bytes};

pub use attention::GpuAttention;
pub use decoder::{
    forward as decoder_forward, load_weight_cache, BenchConfig, GpuDecoderScratch,
};
pub use encoder_extend::prefill_gpu;
pub use weights::GpuDecoderWeightCache;
pub use kv_cache::GpuKvCache;
pub use engine::GpuDecoderEngine;
pub use gemm::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};
pub use probe::{print_probe_result, probe_device};
pub use bench_gemm::{bench_custom_kernel, bench_gemm_bf16, bench_gemm_block_q4, bench_gemm_tunable, bench_gemm_tunable_sparse, bench_mpsgraph_oracle, parse_shapes, print_bench_rows};
pub use step_kernel::{
    bench_fused_gemm_dispatches, bench_step_kernel, build_offsets_from_store, build_step_runtime,
    run_step_probe, run_step_smoke, run_embed_row_gpu, layer_moe_block_jobs, trace_entropy_enabled,
    bench_step_kernel_profile, bench_step_kernel_profile_steps, bench_step_kernel_encode_subprofile,
    EncodeSubProfileResult, LayerEncodeSubProfile, MoeEncodeSubProfile, StepFinishMode,
    StepSmokeConfig, CANVAS, PREFILL_M, PREFILL_SUBS, CanvasState, LayerOffsets, RouteScratch, StepParams,
    fill_token_slot, N_EXPERTS, TOP_K, HID, MOE_FF, FROZEN_WORDS, MOE_MAX_BLOCKS,
};
pub use step_preamble_dump::{
    run_step_preamble_dump, write_step_preamble_dump,
};
pub use step_attn_dump::{
    run_step_attn_layer_dump, write_step_attn_layer_dump,
};
pub use step_moe_dump::{run_step_moe_layer_dump, write_step_moe_layer_dump};
pub use step_moe_batched_pin_dump::{
    print_pin_summary as print_batched_pin_summary, run_step_moe_batched_pin_dump,
    write_step_moe_batched_pin_dump,
};
pub use step_moe_route_dump::{
    print_route_summary, run_step_moe_route_dump, write_step_moe_route_dump,
};
pub use step_moe_single_dump::{
    run_step_moe_single_expert_dump, write_step_moe_single_expert_dump,
};
pub use step_logits_dump::{
    parse_positions, run_step_bf16_oracle_logits_dump, run_step_bf16_oracle_logits_dump_gpu_kv,
    run_step_layer_hidden_dump,
    run_step_logits_dump, write_step_layer_hidden_dump, write_step_logits_dump,
};
pub use step_m0::{run_step_parity, run_step_verify, StepParityConfig};
pub use step_config::{log_validated_step_model, validate_step_model};
pub use step_generate::{generate_monolithic, generate_with_session, StepGenerateConfig, StepGenerateSession, StepObserver, StepProgressEvent};
pub use step_quant::BlockGroupedJob;
pub use step_kv::{
    prefill_monolithic_kv_with_cache,
    run_step_kv_audit, run_step_kv_parity, run_step_kv_bf16_cross_parity,
    run_step_attn_probe, MonolithicEncoderCache,
};
pub use telemetry::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
