//! Shared test-only helpers for hand-building tiny synthetic HF safetensors
//! fixtures — never the real 19 GiB model. Used by overlay/repack round-trip
//! tests and the head-splice plan tests so they stay Tier-1 (milliseconds,
//! blob-free).

use std::collections::BTreeMap;
use std::path::Path;

/// Deterministic pseudo-random bf16 bit patterns — never all-zero, so a
/// "read the wrong tensor" bug shows up as a mismatch, not a trivially-equal
/// all-zero comparison.
pub(crate) fn bf16_payload(numel: usize, seed: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(numel * 2);
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    for _ in 0..numel {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let bits = ((x >> 16) & 0x7fff) as u16 | 0x3c00; // small positive bf16-ish bit pattern
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// Write one safetensors shard: `tensors` is `(name, shape, byte payload)`,
/// written to the file in the given order with ZERO padding between them
/// (matching a real safetensors writer's back-to-back `data_offsets`).
///
/// Pads the header with a `__metadata__` filler string so `8 + header_len`
/// (the absolute file offset of the FIRST tensor) lands on a 64-byte
/// boundary — real HF shards land wherever their header JSON happens to
/// end, which is essentially arbitrary, but the VA-splice writer can only
/// splice a run whose file offset is ITSELF 64-byte aligned (see
/// `dgq::convert::quantize_model`'s doc comment on why 64-byte alignment
/// and page-congruence are mutually exclusive otherwise). Tests that need
/// to prove "the writer's layout is fully spliceable" need a fixture that
/// COULD be, same as this model's real shards usually are (weight byte
/// lengths here are already 64-byte multiples, matching real hidden-dim-
/// sized tensors) — a fixture that's merely lucky or unlucky about header
/// length would be testing header-length parity, not the writer.
pub(crate) fn write_shard(path: &Path, tensors: &[(&str, Vec<i64>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        let end = data.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    // Filler `__metadata__` entry, then pad the value string until
    // `8 + header_len` is a multiple of 64.
    header.insert("__metadata__".to_string(), serde_json::json!({ "pad": "" }));
    let mut pad_len = 0usize;
    loop {
        header.insert(
            "__metadata__".to_string(),
            serde_json::json!({ "pad": "x".repeat(pad_len) }),
        );
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header.clone())).unwrap();
        if (8 + header_json.len()).is_multiple_of(64) {
            break;
        }
        pad_len += 1;
        assert!(
            pad_len < 128,
            "could not find a padding length within 128 bytes"
        );
    }
    let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    assert!((8 + header_json.len()).is_multiple_of(64));
    let mut out = Vec::with_capacity(8 + header_json.len() + data.len());
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write shard");
}

pub(crate) fn write_index(dir: &Path, shard_name: &str, tensor_names: &[&str]) {
    let weight_map: BTreeMap<&str, &str> = tensor_names.iter().map(|&n| (n, shard_name)).collect();
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .expect("write index");
}
