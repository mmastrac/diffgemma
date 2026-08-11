//! `mtp-dump`: MTP training-state corpus dump.
//!
//! Reads a corpus JSONL (`{"id", "prompt", "reply"?, "source"?}`); a record
//! with a reply is teacher-forced as-is, one without gets its reply generated
//! by this model first. Agentic records (`{"id", "context", "reply",
//! "source"?}`) encode both sides specials-aware: `context` is a full
//! model-format prompt (serve-log or converted trace) and `reply` the
//! assistant turn verbatim, thought ceremony included. Each kept record
//! writes layer-29 hiddens plus
//! last-sliding/last-full K/V as f32 bins and appends a manifest.jsonl line.
//! Records whose hidden bin already exists are skipped, so an interrupted
//! run resumes where it left off.

use std::io::{BufRead, Write};
use std::path::Path;

use crate::Error;
use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
use crate::{chat_template, tokenizer};

const MAX_SEQ: usize = 4096;

fn write_f32(path: &Path, data: &[f32]) -> Result<(), Error> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

pub(crate) fn run_mtp_dump_cmd(
    model_dir: &Path,
    input: &Path,
    out_dir: &Path,
    limit: Option<usize>,
) -> Result<(), Error> {
    std::fs::create_dir_all(out_dir)?;
    let file = std::fs::File::open(input)?;
    let records: Vec<serde_json::Value> = std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l))
        .collect::<Result<_, _>>()
        .map_err(|_| Error::Runtime("corpus jsonl parse failed"))?;
    let n = limit.unwrap_or(records.len()).min(records.len());

    let layers = crate::commands::resolve_model_layers(model_dir, None)?;
    let tok = tokenizer::Tokenizer::load(model_dir.join("tokenizer.json"))?;
    let mut cfg = StepGenerateConfig::from_generate(
        7,
        crate::metal::CANVAS,
        MAX_SEQ,
        layers,
        crate::sample::sampler_for_steps(48, false),
        false,
    );
    let (mut session, _) = StepGenerateSession::open(model_dir, &cfg, None)?;
    let eos = session.eos_token_id();

    let manifest_path = out_dir.join("manifest.jsonl");
    let mut manifest = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&manifest_path)?;

    let mut kept = 0usize;
    let mut skipped = 0usize;
    for (ri, rec) in records.iter().take(n).enumerate() {
        let id = rec["id"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| format!("r{ri}"));
        let prompt = rec["prompt"].as_str().unwrap_or("");
        if prompt.is_empty() && rec["context"].as_str().is_none() {
            eprintln!("mtp-dump: {id}: missing prompt/context, skipped");
            skipped += 1;
            continue;
        }
        let hidden_path = out_dir.join(format!("{id}_hidden.bin"));
        if hidden_path.exists() {
            continue;
        }
        let source = rec["source"].as_str().unwrap_or("self");

        let prompt_ids = match rec["context"].as_str() {
            // Agentic trace: the context is already model-format text.
            Some(context) => tok.encode_with_specials(context),
            None => chat_template::format_chat_token_ids(
                &tok,
                &[chat_template::ChatTurn::user(prompt)],
                &chat_template::ChatFormatOptions::default(),
            )?,
        };
        let reply_ids: Vec<u32> = match rec["reply"].as_str() {
            // Specials-aware: teacher replies may carry thought-ceremony
            // markers, which must land as protocol ids (corpus text is
            // trusted, authored by our own builder).
            Some(reply) => tok.encode_with_specials(reply),
            None => {
                cfg.seed = 7 + ri as u64;
                let out = generate_with_session(&mut session, &prompt_ids, &cfg, prompt)?;
                let reply = &out.token_ids[prompt_ids.len()..];
                let end = reply.iter().position(|&t| t == eos).unwrap_or(reply.len());
                reply[..end].to_vec()
            }
        };
        let seq: Vec<u32> = prompt_ids.iter().chain(&reply_ids).copied().collect();
        // max_seq minus canvas headroom for the prefill chunk writes.
        if reply_ids.len() < 8 || seq.len() > MAX_SEQ - 512 {
            eprintln!(
                "mtp-dump: {id}: skipped (reply={}, seq={})",
                reply_ids.len(),
                seq.len()
            );
            skipped += 1;
            continue;
        }

        let (hidden, kv) = session.capture_mtp_states(&seq)?;
        write_f32(&hidden_path, &hidden)?;
        write_f32(&out_dir.join(format!("{id}_k_swa.bin")), &kv.k_swa)?;
        write_f32(&out_dir.join(format!("{id}_v_swa.bin")), &kv.v_swa)?;
        write_f32(&out_dir.join(format!("{id}_k_full.bin")), &kv.k_full)?;
        write_f32(&out_dir.join(format!("{id}_v_full.bin")), &kv.v_full)?;
        let line = serde_json::json!({
            "id": id,
            "source": source,
            "prompt": prompt,
            "seq": seq,
            "ans_start": prompt_ids.len(),
        });
        writeln!(manifest, "{line}")?;
        manifest.flush()?;
        kept += 1;
        eprintln!(
            "mtp-dump: {}/{n} {id} [{source}] seq={} reply={}",
            ri + 1,
            seq.len(),
            reply_ids.len()
        );
    }
    eprintln!(
        "mtp-dump: kept {kept}, skipped {skipped}, manifest {}",
        manifest_path.display()
    );
    Ok(())
}
