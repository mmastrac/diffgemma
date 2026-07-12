//! KV-cache movement and probes: encoder KV pack/unpack, f32 side-KV hydrate,
//! plus the KV-quant / Hadamard-rotation diagnostics.
pub mod hadamard;
pub mod kv_f32_side_hydrate;
pub mod kv_quant;
pub mod pack_encoder_kv;
pub mod unpack_encoder_kv;
