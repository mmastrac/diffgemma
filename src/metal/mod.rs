//! Metal GPU runtime (macOS only, `metal` feature).

pub(crate) mod arena_layout;
mod attention;
mod attention_batch;
pub(crate) mod batch;
mod batched_kernels;
mod bench_gemm;
pub(crate) mod buffer;
pub(crate) mod debug_status;
mod decoder;
mod decoder_attention;
mod decoder_layer;
pub(crate) mod device;
mod dgq_gpu;
mod embed;
mod encoder_extend;
mod engine;
mod expert_cache;
mod gemm;
mod kernels;
mod kv_cache;
mod linear;
mod lm_head;
mod memory;
pub mod memwatch;
mod moe;
mod pipeline_cache;
mod probe;
mod router;
mod sampler;
mod sampler_kernels;
mod self_conditioning;
mod step_attn_dump;
mod step_config;
mod step_generate;
mod step_kernel;
mod step_kv;
mod step_logits_dump;
mod step_m0;
mod step_moe_batched_pin_dump;
mod step_moe_dump;
mod step_moe_route_dump;
mod step_moe_single_dump;
mod step_preamble_dump;
mod step_quant;
mod telemetry;
mod weights;

// Re-exports consumed only by in-crate tests. Test-gated so the non-test build
// stays warning-clean under `-D warnings` (the binary never uses these paths).
#[cfg(test)]
pub use dgq_gpu::DgqGpuBlob;
#[cfg(test)]
pub use step_kernel::build_offsets_from_store;

pub use attention::GpuAttention;
pub use bench_gemm::{
    GemmShape, bench_custom_kernel, bench_gemm_bf16, bench_gemm_block_q4, bench_gemm_int_sparse,
    bench_gemm_tunable, bench_gemm_tunable_db, bench_gemm_tunable_sparse, bench_mpsgraph_oracle,
    parse_shapes, print_bench_rows,
};
pub use decoder::GpuDecoderScratch;
pub use engine::GpuDecoderEngine;
pub use gemm::{Bf16Gemm, bf16_matmul_cpu, f32_to_bf16};
pub use kv_cache::GpuKvCache;
pub use probe::{print_probe_result, probe_device};
pub use step_attn_dump::{run_step_attn_layer_dump, write_step_attn_layer_dump};
pub use step_config::{log_validated_step_model, validate_step_model};
pub use step_generate::{
    BlockOutcome, CancelToken, KvSnapshot, ProposedBlock, StepGenerateConfig, StepGenerateSession,
    StepObserver, StepProgressEvent, TurnState, begin_turn, commit_block, finish_turn,
    generate_monolithic, generate_with_session, propose_block,
};
pub use step_kernel::{
    CANVAS, CanvasState, EncodeSubProfileResult, FROZEN_WORDS, HID, LayerEncodeSubProfile,
    LayerOffsets, MOE_FF, MOE_MAX_BLOCKS, MoeEncodeSubProfile, N_EXPERTS, PREFILL_M, PREFILL_SUBS,
    RouteScratch, StepFinishMode, StepParams, StepSmokeConfig, TOP_K, bench_fused_gemm_dispatches,
    bench_step_kernel, bench_step_kernel_encode_subprofile, bench_step_kernel_prefill_super,
    bench_step_kernel_prefill_super_stages, bench_step_kernel_profile,
    bench_step_kernel_profile_steps, fill_token_slot, layer_moe_block_jobs, run_embed_row_gpu,
    run_step_attn_qk_plane_dump, run_step_probe, run_step_smoke,
};
pub use step_kv::{
    run_step_attn_probe, run_step_kv_audit, run_step_kv_bf16_cross_parity, run_step_kv_parity,
};
pub use step_logits_dump::{
    parse_positions, run_step_bf16_oracle_logits_dump, run_step_bf16_oracle_logits_dump_gpu_kv,
    run_step_layer_hidden_dump, run_step_logits_dump, write_step_layer_hidden_dump,
    write_step_logits_dump,
};
pub use step_m0::{StepParityConfig, run_step_parity, run_step_verify};
pub use step_moe_batched_pin_dump::{
    print_pin_summary as print_batched_pin_summary, run_step_moe_batched_pin_dump,
    write_step_moe_batched_pin_dump,
};
pub use step_moe_dump::{run_step_moe_layer_dump, write_step_moe_layer_dump};
pub use step_moe_route_dump::{
    print_route_summary, run_step_moe_route_dump, write_step_moe_route_dump,
};
pub use step_moe_single_dump::{
    run_step_moe_single_expert_dump, write_step_moe_single_expert_dump,
};
pub use step_preamble_dump::{run_step_preamble_dump, write_step_preamble_dump};
pub use step_quant::BlockGroupedJob;
pub use telemetry::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
