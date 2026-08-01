//! Token-fusion oracle tests. Split from step_kv_bench_tests.rs;
//! shares its helpers via the sibling module.

use super::engine_extend_bench_tests::{model_dir, synth_ids};
use super::*;
use crate::metal::device::MetalContext;

/// How MERGEABLE are neighboring full-layer KV rows? Token fusion
/// (CaM/KVMerger/Compressive-Transformer class) would coalesce aged rows of
/// the 5 FULL layers (the only long-range memory = all the KV bytes and the
/// O(kv_len) step cost). Merging by averaging is promising iff neighbors are
/// correlated — and structurally they should be: proportional RoPE ropes only
/// rot=hd/4 dims, so 75% of every full-layer K row is position-independent.
/// Measures adjacent-row and block-of-4 cosine (K whole / K roped-dims / K
/// unroped-dims / V), by age band, plus the constant-row-norm check (scalar
/// k_norm → rows on a sphere). Full layers store linearly (slot == pos at
/// any length), so a longer prompt is fine here.
#[test]
#[ignore = "model-gated: cargo test --release e16_fusion_mergeability_stats -- --ignored --nocapture"]
fn e16_fusion_mergeability_stats() {
    use crate::metal::step_kernel::{StepFinishMode, StepSmokeConfig, build_step_runtime};
    use crate::tokenizer::Tokenizer;
    let Some(dir) = model_dir() else { return };
    // Carrier selection (`DGQ_E16_CARRIER`): a text file path, or "random"
    // for pseudo-random token ids — the null control that separates
    // content redundancy from model-intrinsic KV structure. Default = the
    // English technical markdown fixture. Similarity structure may differ
    // by content class (code / logs / non-English), so the M0b verdict
    // must not rest on one carrier.
    let carrier =
        std::env::var("DGQ_E16_CARRIER").unwrap_or_else(|_| "fixtures/smoketest/longdoc.md".into());
    let tok = Tokenizer::load(dir.join("tokenizer.json")).expect("tokenizer");
    let max_seq = 8192usize;
    let mut ids = if carrier == "random" {
        synth_ids(max_seq)
    } else {
        tok.encode(
            &std::fs::read_to_string(&carrier).expect("carrier fixture"),
            false,
        )
    };
    ids.truncate(max_seq - CANVAS - 36);
    let n = ids.len();
    eprintln!("e16-m0b carrier: {carrier}");
    let cfg = StepSmokeConfig {
        layers: N_LAYERS,
        steps: 1,
        kv_len: 0,
        seed: 7,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: true,
    };
    let (mut rt, _) = build_step_runtime(&dir, &cfg).expect("runtime");
    let kv = rt.prefill_chunks(&ids).expect("fast prefill");
    assert_eq!(kv, n);
    let layout = *rt.layout();

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            d += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            d / (na * nb).sqrt()
        }
    }

    eprintln!("e16-m0b: full-layer KV mergeability ({n} tokens; aged = all but last 2048)");
    for layer in 0..N_LAYERS {
        let l = &layout.layers[layer];
        if l.is_full == 0 {
            continue;
        }
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let rot = hd / 4;
        let half_head = hd / 2;
        // roped dims = {0..rot/2} ∪ {half_head..half_head+rot/2}
        let roped: Vec<usize> = (0..rot / 2).chain(half_head..half_head + rot / 2).collect();
        let unroped: Vec<usize> = (0..hd).filter(|d| !roped.contains(d)).collect();
        let token_stride = nkv * hd * 2;
        let slot_base = l.kv_region as usize / 2;
        // read all K and V rows for this layer: [pos][head][hd]
        let read_row = |pos: usize, hh: usize, is_v: bool| -> Vec<f32> {
            let base = slot_base + pos * token_stride + if is_v { nkv * hd } else { 0 };
            (0..hd)
                .map(|d| f16_bits_to_f32(read_half_at(rt.kvcache(), (base + hh * hd + d) * 2)))
                .collect()
        };
        let sub = |v: &[f32], idx: &[usize]| -> Vec<f32> { idx.iter().map(|&i| v[i]).collect() };
        let bands: [(usize, usize, &str); 2] = [
            (0, n.saturating_sub(2048), "aged  "),
            (n.saturating_sub(2048), n, "recent"),
        ];
        for (b0, b1, label) in bands {
            if b1 - b0 < 8 {
                continue;
            }
            let (mut ck, mut ckr, mut cku, mut cv, mut c4k, mut c4v) =
                (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
            let (mut n_adj, mut n_blk) = (0usize, 0usize);
            let mut norm_sum = 0f64;
            let mut norm_sq = 0f64;
            let mut n_norm = 0usize;
            // sample every head, stride positions by 4 to bound CPU time
            for hh in 0..nkv {
                let mut pos = b0;
                while pos + 4 < b1 {
                    let k: Vec<Vec<f32>> = (0..4).map(|i| read_row(pos + i, hh, false)).collect();
                    let v: Vec<Vec<f32>> = (0..4).map(|i| read_row(pos + i, hh, true)).collect();
                    ck += cos(&k[0], &k[1]);
                    ckr += cos(&sub(&k[0], &roped), &sub(&k[1], &roped));
                    cku += cos(&sub(&k[0], &unroped), &sub(&k[1], &unroped));
                    cv += cos(&v[0], &v[1]);
                    n_adj += 1;
                    // mean pairwise cos within the block of 4
                    let (mut sk, mut sv, mut np) = (0f64, 0f64, 0usize);
                    for i in 0..4 {
                        for j in i + 1..4 {
                            sk += cos(&k[i], &k[j]);
                            sv += cos(&v[i], &v[j]);
                            np += 1;
                        }
                    }
                    c4k += sk / np as f64;
                    c4v += sv / np as f64;
                    n_blk += 1;
                    let nn = k[0]
                        .iter()
                        .map(|&x| (x as f64) * (x as f64))
                        .sum::<f64>()
                        .sqrt();
                    norm_sum += nn;
                    norm_sq += nn * nn;
                    n_norm += 1;
                    pos += 16;
                }
            }
            let m = |s: f64, c: usize| s / c.max(1) as f64;
            let nm = norm_sum / n_norm.max(1) as f64;
            let nsd = (norm_sq / n_norm.max(1) as f64 - nm * nm).max(0.0).sqrt();
            eprintln!(
                "  L{layer:02} {label} adjK {:+.3} (roped {:+.3} unroped {:+.3}) adjV {:+.3} | blk4 K {:+.3} V {:+.3} | ‖K‖ {nm:.2}±{nsd:.2}",
                m(ck, n_adj),
                m(ckr, n_adj),
                m(cku, n_adj),
                m(cv, n_adj),
                m(c4k, n_blk),
                m(c4v, n_blk)
            );
        }
    }
}

/// Quality frontier of count-weighted KV fusion with ZERO kernel changes.
/// Trick: prefill normally, then rewrite each aged full-layer block of r
/// rows as r DUPLICATES of its mean-K / mean-V — duplicated keys contribute
/// r·exp(q·k̄) to the softmax, which is EXACTLY the count-weighted
/// merged-attention semantics a real fused kernel would implement (and
/// duplicated V̄ gives the right weighted average). Then re-enter generation
/// on the doctored cache (restore → mutate → generate re-entry skips prefill
/// since kv_valid == prompt) and judge the doc_13k ladder question. Faithful
/// to a real implementation including the merged-RoPE-position effect (we
/// merge stored post-RoPE rows). Sliding layers untouched (they never see
/// aged tokens anyway).
#[test]
#[ignore = "model-gated: cargo test --release e16_fusion_oracle_replay -- --ignored --nocapture"]
fn e16_fusion_oracle_replay() {
    use crate::chat_template;
    use crate::config;
    use crate::metal::step_generate::{
        StepGenerateConfig, StepGenerateSession, generate_with_session,
    };
    use crate::sample;
    use crate::shaders::f16::f32_to_f16_bits;
    use crate::tokenizer::Tokenizer;
    let Some(dir) = model_dir() else { return };
    let tok = Tokenizer::load(dir.join("tokenizer.json")).expect("tokenizer");
    let text = std::fs::read_to_string("fixtures/smoketest/longdoc.md").expect("probe fixture");
    let doc_ids = tok.encode(&text, false);
    let excerpt = tok.decode(&doc_ids[..13300]);
    // Facts at doc depths ~979 ("1085") and ~7716 ("20.25") — DEEP inside
    // the fused aged region at every grid W (recent window = last 1-2k).
    // (First run of this oracle asked about facts at ~12.7-13.0k, which sit
    // INSIDE the protected recent window — all cells passed identically and
    // proved only that the harness works. Aim questions at fused depths.)
    let question = "(a) How many seconds did the 105k-token needle prefill take, and (b) \
                        what is the Metal single-buffer allocation cap in GiB?";
    let prompt_text =
        format!("{excerpt}\n\n[end of document]\nAnswer from the document above: {question}");
    let turns = [chat_template::ChatTurn::user(&prompt_text)];
    let prompt = chat_template::format_chat_token_ids(
        &tok,
        &turns,
        &chat_template::ChatFormatOptions::default(),
    )
    .expect("chat prompt");
    let n = prompt.len();
    let max_seq = 16384usize;
    let sampler = sample::sampler_for_steps(48, false);
    let mut cfg = StepGenerateConfig::from_generate(7, 512, max_seq, N_LAYERS, sampler, false);
    cfg.stop_token_ids = config::load_generation_stop_tokens(&dir);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
    session.extend_kv(&prompt).expect("prefill");
    let snap = session.snapshot_kv();
    let layout = *session.layout_for_test();
    eprintln!("e16-oracle: prompt {n} tokens; facts at ~979 / ~7716, aged region = [0, n-W)");

    // Word-run match with digit<->letter splits (mirror of smoke_normalize).
    fn normalize(s: &str) -> String {
        let mut out = String::new();
        let mut prev = 0u8; // 0 space, 1 digit, 2 letter
        for c in s.chars() {
            if c.is_alphanumeric() {
                let cur = if c.is_ascii_digit() { 1 } else { 2 };
                if prev != 0 && prev != cur {
                    out.push(' ');
                }
                for lc in c.to_lowercase() {
                    out.push(lc);
                }
                prev = cur;
            } else if prev != 0 {
                out.push(' ');
                prev = 0;
            }
        }
        out.trim().to_string()
    }
    let matches = |reply: &str, k: &str| {
        format!(" {} ", normalize(reply)).contains(&format!(" {} ", normalize(k)))
    };

    fn write_half_at(
        buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        byte_off: usize,
        bits: u16,
    ) {
        use objc2_metal::MTLBuffer as _;
        unsafe {
            *(buf.contents().as_ptr() as *mut u8)
                .add(byte_off)
                .cast::<u16>() = bits;
        }
    }

    // Count-weighted fusion via duplication over aged range [a0, a1):
    // each block of r rows -> r copies of the block mean (K and V). With
    // `tau`, a block merges ONLY if the mean pairwise K-cosine of its rows
    // (per head) clears it — sharp singleton rows stay exact (similarity-
    // gated merging). Returns (candidate_rows, effective_rows): what a
    // real M1 layout would store (merged block of m -> 1 row).
    let fuse_range = |session: &StepGenerateSession,
                      a0: usize,
                      a1: usize,
                      r: usize,
                      tau: Option<f64>|
     -> (usize, usize) {
        let buf = session.kv_buffer_for_test();
        let (mut candidates, mut effective) = (0usize, 0usize);
        for layer in 0..N_LAYERS {
            let l = &layout.layers[layer];
            if l.is_full == 0 {
                continue;
            }
            let nkv = l.n_kv_heads as usize;
            let hd = l.head_dim as usize;
            let token_stride = nkv * hd * 2;
            let half_base = l.kv_region as usize / 2;
            let read_row = |pos: usize, hh: usize, off: usize| -> Vec<f64> {
                let base = half_base + pos * token_stride + off + hh * hd;
                (0..hd)
                    .map(|d| f16_bits_to_f32(read_half_at(buf, (base + d) * 2)) as f64)
                    .collect()
            };
            let mut b0 = a0;
            while b0 < a1 {
                let b1 = (b0 + r).min(a1);
                let m = b1 - b0;
                if m >= 2 {
                    for hh in 0..nkv {
                        candidates += m;
                        let ks: Vec<Vec<f64>> = (b0..b1).map(|p| read_row(p, hh, 0)).collect();
                        let merge = match tau {
                            None => true,
                            Some(t) => {
                                let (mut s, mut c) = (0f64, 0usize);
                                for i in 0..m {
                                    for j in i + 1..m {
                                        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
                                        #[allow(clippy::needless_range_loop)]
                                        // x indexes ks[i] and ks[j] in lockstep
                                        for x in 0..hd {
                                            d += ks[i][x] * ks[j][x];
                                            na += ks[i][x] * ks[i][x];
                                            nb += ks[j][x] * ks[j][x];
                                        }
                                        s += d / (na * nb).sqrt().max(1e-12);
                                        c += 1;
                                    }
                                }
                                s / c as f64 >= t
                            }
                        };
                        if !merge {
                            effective += m;
                            continue;
                        }
                        effective += 1;
                        for off in [0usize, nkv * hd] {
                            let rows: Vec<Vec<f64>> = if off == 0 {
                                ks.clone()
                            } else {
                                (b0..b1).map(|p| read_row(p, hh, off)).collect()
                            };
                            let mut mean = vec![0f64; hd];
                            for row in &rows {
                                for (d, m) in mean.iter_mut().enumerate() {
                                    *m += row[d];
                                }
                            }
                            for mm in mean.iter_mut() {
                                *mm /= m as f64;
                            }
                            for pos in b0..b1 {
                                let base = half_base + pos * token_stride + off + hh * hd;
                                for (d, &mm) in mean.iter().enumerate() {
                                    write_half_at(buf, (base + d) * 2, f32_to_f16_bits(mm as f32));
                                }
                            }
                        }
                    }
                } else {
                    effective += m;
                    candidates += m;
                }
                b0 = b1;
            }
        }
        (candidates, effective)
    };

    // Cells: uniform r=2 (known-clean), TIERED (r=4 for distant, r=2 for
    // mid-age — facts sit one per tier), similarity-GATED r=4/r=8 at
    // tau=0.5 (merge only redundant runs; sharp singletons stay exact).
    let w = 2048usize;
    let names = [
        "control",
        "uniform_r2",
        "tiered_r4_r2",
        "gated_t50_r4",
        "gated_t50_r8",
    ];
    for name in names {
        session.restore_kv(&snap);
        let (cand, eff) = match name {
            "control" => (0, 0),
            "uniform_r2" => fuse_range(&session, 0, n - w, 2, None),
            "tiered_r4_r2" => {
                let far = fuse_range(&session, 0, n.saturating_sub(8192), 4, None);
                let mid = fuse_range(&session, n.saturating_sub(8192), n - w, 2, None);
                (far.0 + mid.0, far.1 + mid.1)
            }
            "gated_t50_r4" => fuse_range(&session, 0, n - w, 4, Some(0.50)),
            "gated_t50_r8" => fuse_range(&session, 0, n - w, 8, Some(0.50)),
            _ => unreachable!(),
        };
        let out = generate_with_session(&mut session, &prompt, &cfg, "e16-oracle").expect("gen");
        let new_ids = sample::strip_degenerate_token_ids(out.token_ids.get(n..).unwrap_or(&[]));
        let reply = chat_template::sanitize_model_reply(&tok.decode(&new_ids));
        let r1 = matches(&reply, "1085");
        let r2 = matches(&reply, "20.25");
        let keep = if cand > 0 {
            format!("{:.0}%", 100.0 * eff as f64 / cand as f64)
        } else {
            "100%".into()
        };
        let prev = reply
            .chars()
            .take(80)
            .collect::<String>()
            .replace('\n', " ");
        eprintln!(
            "e16-oracle {name:<13} keys-kept {keep:<4} 1085:{} 20.25:{} | {prev}",
            if r1 { "PASS" } else { "FAIL" },
            if r2 { "PASS" } else { "FAIL" },
        );
    }
}

/// Multi-needle oracle: binomial-strength fusion-quality stats,
/// carrier-selectable (`DGQ_E16_CARRIER`: text file path; default = the
/// English markdown fixture — run it on code/log carriers too, the
/// mergeability census showed raw K-similarity is mostly model-intrinsic
/// common-mode, so only task quality can validate a carrier class).
/// Plants 8 synthetic city-code needles at even depths through the FUSED
/// region of a ~13k-token carrier; one generation asks for the full list
/// -> k-of-8 per cell instead of a coin flip per cell.
#[test]
#[ignore = "model-gated: cargo test --release e16_fusion_multineedle_oracle -- --ignored --nocapture"]
fn e16_fusion_multineedle_oracle() {
    use crate::chat_template;
    use crate::config;
    use crate::metal::step_generate::{
        StepGenerateConfig, StepGenerateSession, generate_with_session,
    };
    use crate::sample;
    use crate::shaders::f16::f32_to_f16_bits;
    use crate::tokenizer::Tokenizer;
    let Some(dir) = model_dir() else { return };
    let tok = Tokenizer::load(dir.join("tokenizer.json")).expect("tokenizer");
    let carrier =
        std::env::var("DGQ_E16_CARRIER").unwrap_or_else(|_| "fixtures/smoketest/longdoc.md".into());
    let mut raw = std::fs::read_to_string(&carrier).expect("carrier fixture");
    while tok.encode(&raw, false).len() < 14000 {
        let again = raw.clone();
        raw.push_str("\n\n");
        raw.push_str(&again);
    }
    const NEEDLES: [(&str, &str); 8] = [
        ("Lisbon", "48291"),
        ("Osaka", "57364"),
        ("Nairobi", "81437"),
        ("Quito", "29586"),
        ("Tromso", "63917"),
        ("Adelaide", "74128"),
        ("Reykjavik", "36852"),
        ("Valparaiso", "95274"),
    ];
    for (city, code) in NEEDLES {
        assert!(!raw.contains(code), "carrier already contains {code}");
        assert!(!raw.contains(city), "carrier already contains {city}");
    }
    // Truncate the carrier to ~13.1k tokens, then plant needles at even
    // char depths within the first 85% (all needles stay in the aged
    // region for W=2048), on line boundaries.
    let ids = tok.encode(&raw, false);
    let base_text = tok.decode(&ids[..13100.min(ids.len())]);
    let mut text = String::new();
    let bytes = base_text.len();
    let mut cursor = 0usize;
    for (i, (city, code)) in NEEDLES.iter().enumerate() {
        let mut target = bytes * 85 * (2 * i + 1) / (100 * 2 * NEEDLES.len());
        target = target.min(bytes - 1);
        while !base_text.is_char_boundary(target) {
            target += 1;
        }
        let cut = base_text[target..]
            .find('\n')
            .map(|o| target + o + 1)
            .unwrap_or(bytes);
        text.push_str(&base_text[cursor..cut]);
        text.push_str(&format!("\nThe secret access code for {city} is {code}.\n"));
        cursor = cut;
    }
    text.push_str(&base_text[cursor..]);
    let prompt_text = format!(
        "{text}\n\n[end of document]\nAnswer from the document above: List every city that \
             has a secret access code in the document, together with its code."
    );
    let turns = [chat_template::ChatTurn::user(&prompt_text)];
    let prompt = chat_template::format_chat_token_ids(
        &tok,
        &turns,
        &chat_template::ChatFormatOptions::default(),
    )
    .expect("chat prompt");
    let n = prompt.len();
    // Report each needle's token depth (verify they sit in the fused region).
    let depths: Vec<usize> = NEEDLES
        .iter()
        .map(|(_, code)| {
            let off = text.find(code).expect("planted");
            tok.encode(&text[..off], false).len()
        })
        .collect();
    let w = 2048usize;
    eprintln!(
        "e16-needles carrier={carrier} prompt {n} tok; needle depths {depths:?}; aged=[0,{})",
        n - w
    );

    let max_seq = 16384usize;
    let sampler = sample::sampler_for_steps(48, false);
    let mut cfg = StepGenerateConfig::from_generate(7, 512, max_seq, N_LAYERS, sampler, false);
    cfg.stop_token_ids = config::load_generation_stop_tokens(&dir);
    let (mut session, _) = StepGenerateSession::open(&dir, &cfg, None).expect("session");
    session.extend_kv(&prompt).expect("prefill");
    let snap = session.snapshot_kv();
    let layout = *session.layout_for_test();

    fn write_half_at(
        buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        byte_off: usize,
        bits: u16,
    ) {
        use objc2_metal::MTLBuffer as _;
        unsafe {
            *(buf.contents().as_ptr() as *mut u8)
                .add(byte_off)
                .cast::<u16>() = bits;
        }
    }
    // Count-weighted fusion of full-layer rows in [a0, a1) at ratio r
    // (duplication trick; see e16_fusion_oracle_replay). With `resid_tau`,
    // a block merges only if the mean pairwise cosine of its K RESIDUALS
    // (row minus the 256-position band mean — the model-intrinsic
    // common-mode that dominates raw cosine even on random tokens) clears
    // it: distinctive rows (needles) refuse to merge, boilerplate merges
    // freely. Returns (candidate_rows, effective_rows).
    let fuse_range = |session: &StepGenerateSession,
                      a0: usize,
                      a1: usize,
                      r: usize,
                      resid_tau: Option<f64>|
     -> (usize, usize) {
        let buf = session.kv_buffer_for_test();
        let (mut cand, mut eff) = (0usize, 0usize);
        for layer in 0..N_LAYERS {
            let l = &layout.layers[layer];
            if l.is_full == 0 {
                continue;
            }
            let nkv = l.n_kv_heads as usize;
            let hd = l.head_dim as usize;
            let token_stride = nkv * hd * 2;
            let half_base = l.kv_region as usize / 2;
            let read_k = |pos: usize, hh: usize| -> Vec<f64> {
                let base = half_base + pos * token_stride + hh * hd;
                (0..hd)
                    .map(|d| f16_bits_to_f32(read_half_at(buf, (base + d) * 2)) as f64)
                    .collect()
            };
            // Per-head band means of K over 256-position windows (common-mode).
            const BAND: usize = 256;
            let n_bands = a1.div_ceil(BAND);
            let mut band_mean: Vec<Vec<Vec<f64>>> = vec![vec![vec![0f64; hd]; n_bands]; nkv];
            if resid_tau.is_some() {
                let mut band_cnt = vec![vec![0usize; n_bands]; nkv];
                for pos in a0..a1 {
                    let b = pos / BAND;
                    for hh in 0..nkv {
                        let k = read_k(pos, hh);
                        for (d, &v) in k.iter().enumerate() {
                            band_mean[hh][b][d] += v;
                        }
                        band_cnt[hh][b] += 1;
                    }
                }
                for hh in 0..nkv {
                    for b in 0..n_bands {
                        let c = band_cnt[hh][b].max(1) as f64;
                        for v in band_mean[hh][b].iter_mut() {
                            *v /= c;
                        }
                    }
                }
            }
            let mut b0 = a0;
            while b0 < a1 {
                let b1 = (b0 + r).min(a1);
                let m = b1 - b0;
                if m >= 2 {
                    #[allow(clippy::needless_range_loop)]
                    // hh indexes several parallel per-kv-head arrays
                    for hh in 0..nkv {
                        cand += m;
                        let merge = match resid_tau {
                            None => true,
                            Some(t) => {
                                let mu = &band_mean[hh][b0 / BAND];
                                let rs: Vec<Vec<f64>> = (b0..b1)
                                    .map(|p| {
                                        let k = read_k(p, hh);
                                        k.iter().zip(mu).map(|(&a, &b)| a - b).collect()
                                    })
                                    .collect();
                                let (mut s, mut c) = (0f64, 0usize);
                                for i in 0..m {
                                    for j in i + 1..m {
                                        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
                                        #[allow(clippy::needless_range_loop)]
                                        // x indexes rs[i] and rs[j] in lockstep
                                        for x in 0..hd {
                                            d += rs[i][x] * rs[j][x];
                                            na += rs[i][x] * rs[i][x];
                                            nb += rs[j][x] * rs[j][x];
                                        }
                                        s += d / (na * nb).sqrt().max(1e-12);
                                        c += 1;
                                    }
                                }
                                s / c as f64 >= t
                            }
                        };
                        if !merge {
                            eff += m;
                            continue;
                        }
                        eff += 1;
                        for off in [0usize, nkv * hd] {
                            let mut mean = vec![0f64; hd];
                            for pos in b0..b1 {
                                let base = half_base + pos * token_stride + off + hh * hd;
                                for (d, mm) in mean.iter_mut().enumerate() {
                                    *mm +=
                                        f16_bits_to_f32(read_half_at(buf, (base + d) * 2)) as f64;
                                }
                            }
                            for mm in mean.iter_mut() {
                                *mm /= m as f64;
                            }
                            for pos in b0..b1 {
                                let base = half_base + pos * token_stride + off + hh * hd;
                                for (d, &mm) in mean.iter().enumerate() {
                                    write_half_at(buf, (base + d) * 2, f32_to_f16_bits(mm as f32));
                                }
                            }
                        }
                    }
                } else {
                    cand += m;
                    eff += m;
                }
                b0 = b1;
            }
        }
        (cand, eff)
    };

    for name in [
        "control",
        "uniform_r2",
        "tiered_r4_r2",
        "uniform_r4",
        "resid_t20_r2",
        "resid_t20_r4",
    ] {
        session.restore_kv(&snap);
        let (cand, eff) = match name {
            "control" => (0, 0),
            "uniform_r2" => fuse_range(&session, 0, n - w, 2, None),
            "tiered_r4_r2" => {
                let a = fuse_range(&session, 0, n.saturating_sub(8192), 4, None);
                let b = fuse_range(&session, n.saturating_sub(8192), n - w, 2, None);
                (a.0 + b.0, a.1 + b.1)
            }
            "uniform_r4" => fuse_range(&session, 0, n - w, 4, None),
            "resid_t20_r2" => fuse_range(&session, 0, n - w, 2, Some(0.20)),
            "resid_t20_r4" => fuse_range(&session, 0, n - w, 4, Some(0.20)),
            _ => unreachable!(),
        };
        let keep = if cand > 0 {
            format!("{:.0}%", 100.0 * eff as f64 / cand as f64)
        } else {
            "100%".into()
        };
        let out = generate_with_session(&mut session, &prompt, &cfg, "e16-needles").expect("gen");
        let new_ids = sample::strip_degenerate_token_ids(out.token_ids.get(n..).unwrap_or(&[]));
        let reply = chat_template::sanitize_model_reply(&tok.decode(&new_ids));
        let hits: Vec<bool> = NEEDLES
            .iter()
            .map(|(_, code)| reply.contains(code))
            .collect();
        let k = hits.iter().filter(|&&h| h).count();
        let missed: Vec<&str> = NEEDLES
            .iter()
            .zip(&hits)
            .filter(|(_, h)| !**h)
            .map(|((city, _), _)| *city)
            .collect();
        eprintln!("e16-needles {name:<13} keys-kept {keep:<4} {k}/8 retrieved; missed {missed:?}");
    }
}

/// One resident engine prefill of 1024 synthetic tokens with the per-layer
/// phase profile on — splits waitA (attention+dense GEMMs+router GPU) from
/// waitB (MoE GPU) to locate the ms/tok. Diagnostic only.
#[test]
#[ignore = "model-gated bench: cargo test --release engine_prefill_profile -- --ignored --nocapture"]
fn engine_prefill_profile() {
    let Some(dir) = model_dir() else { return };
    let mut cfg = crate::flags::RuntimeConfig::default();
    cfg.debug.prefill_profile = true;
    let _g = crate::flags::install_for_test(cfg);
    let max_seq = 2048usize;
    let store = DgqStore::open(&dir).expect("dgq");
    let layout = build_layout(&build_offsets_from_store(&store), max_seq);
    let ctx = MetalContext::new().expect("metal");
    let kv_bytes = kv_cache_total_bytes(&layout, max_seq) as usize;
    let kv_buf = ctx
        .device
        .newBufferWithLength_options(kv_bytes, MTLResourceOptions::StorageModeShared)
        .expect("kv buf");
    let mut cache = MonolithicEncoderCache::open(&dir, CANVAS, max_seq).expect("cache");
    let ids = synth_ids(1024);
    let t = std::time::Instant::now();
    let (kv_len, _) =
        prefill_monolithic_kv_with_cache(&mut cache, &ids, &kv_buf, &layout, max_seq, N_LAYERS)
            .expect("prefill");
    eprintln!(
        "engine_prefill_profile: kv_len={kv_len} total={:.2}s ({:.1} ms/tok)",
        t.elapsed().as_secs_f64(),
        t.elapsed().as_secs_f64() * 1000.0 / kv_len as f64
    );
}
