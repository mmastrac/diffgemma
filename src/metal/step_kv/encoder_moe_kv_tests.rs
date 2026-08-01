//! Tests for `encoder_moe_kv_tests`, extracted from step_kv.rs (backlog item 3).

use super::*;
use crate::chat_template::{ChatFormatOptions, ChatTurn, format_chat_token_ids};
use crate::metal::device::MetalContext;
use crate::tokenizer::Tokenizer;

fn calgary_prefill(model_dir: &Path) -> Vec<u32> {
    let tok = Tokenizer::load(model_dir.join("tokenizer.json")).expect("tokenizer");
    format_chat_token_ids(
        &tok,
        &[ChatTurn::user("How can I get from Calgary to Namibia?")],
        &ChatFormatOptions::default(),
    )
    .expect("prefill ids")
}

#[test]
fn nvfp4_encoder_prefill_long_prompt_kv_finite() {
    let dir = std::path::Path::new("/tmp/nvfp4-weights");
    if !dir.join("model.dgq.json").exists() {
        eprintln!("skip: /tmp/nvfp4-weights missing");
        return;
    }
    let ids = calgary_prefill(dir);
    assert!(
        ids.len() >= 20,
        "expected long chat prompt, got {}",
        ids.len()
    );
    let ctx = MetalContext::new().expect("metal");
    let layout = build_layout(
        &build_offsets_from_store(&DgqStore::open(dir).expect("dgq")),
        512,
    );
    let kv_bytes = kv_cache_total_bytes(&layout, 512) as usize;
    let kv_buf = ctx
        .device
        .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
        .expect("kv buf");
    let mut cache =
        MonolithicEncoderCache::open_opt(dir, CANVAS, 512, None).expect("encoder cache");
    cache.engine.set_encoder_gpu_moe(true);
    let (kv_len, _) = prefill_monolithic_kv_with_cache(&mut cache, &ids, &kv_buf, &layout, 512, 2)
        .expect("prefill");
    for layer in 0..2 {
        let k_max = kvcache_plane_max_abs(&kv_buf, &layout, layer, kv_len, 0);
        eprintln!("nvfp4 prefill L{layer}: kv_len={kv_len} k_max={k_max:.4}");
        assert!(
            k_max > 1e-4,
            "layer {layer} K prefix looks unset (max={k_max})"
        );
    }
}
