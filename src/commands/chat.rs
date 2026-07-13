//! Interactive `chat` subcommand.

use super::*;

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_chat_cmd(
    model_dir: &std::path::Path,
    initial_prompt: Option<String>,
    seed: u64,
    steps: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    raw_prompt: bool,
    verbose: bool,
    events_path: Option<PathBuf>,
    json: bool,
    ctx: Option<usize>,
) -> ExitCode {
    use metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
    use std::io::{self, IsTerminal, Write};

    // Quiet by default: chat is a UI, not a log. `--verbose` restores the
    // step/prefill/session logs; `--json` routes the event stream to stdout so
    // no human chrome may pollute it.
    if !verbose && !flags::quiet_set_by_user() {
        flags::set_quiet(true);
    }
    // `--json`: JSONL to stdout, all human output suppressed. Otherwise the
    // spinner/streaming renderer runs on an interactive tty (not under
    // --verbose). `--events <path>` tees the JSONL to a file either way.
    let interactive = !json && !verbose && io::stdout().is_terminal();
    let events_file = match &events_path {
        Some(p) => match std::fs::File::create(p) {
            Ok(f) => Some(f),
            Err(err) => {
                eprintln!("error: cannot open --events {}: {err}", p.display());
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    // Per-turn JSON sink factory: stdout for --json, a cloned handle to the
    // events file otherwise. `None` when neither is requested.
    let make_json_sink = |turn_active: bool| -> Option<Box<dyn Write + Send>> {
        if !turn_active {
            return None;
        }
        if json {
            Some(Box::new(io::stdout()))
        } else {
            events_file
                .as_ref()
                .and_then(|f| f.try_clone().ok())
                .map(|f| Box::new(f) as Box<dyn Write + Send>)
        }
    };
    let want_json = json || events_file.is_some();

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: chat requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }

    let layers = match resolve_model_layers(model_dir, max_layers) {
        Ok(n) => n,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // Full-message chat: generate until the model emits its end-of-turn token,
    // limited only by the context window (no per-turn token cap — long replies
    // stream freely). The KV arena is sized once at session open and fixed
    // thereafter, so pick a roomy `max_seq` up front. `--ctx N` overrides for
    // long-context sessions (ring-buffer sliding KV keeps this cheap:
    // ~20 KB/token + ~410 MiB fixed; 131072 ≈ 3 GiB). `--max-new-tokens`, if
    // the caller raises it above the default, becomes an optional hard ceiling.
    #[allow(non_snake_case)]
    let CHAT_MAX_SEQ: usize = ctx.unwrap_or(8192);
    // Fail-fast --ctx budget guard: refuse a context whose weights+KV would
    // swap / fail to allocate, before loading the model (see check_ctx_budget).
    if let Err(msg) = check_ctx_budget(CHAT_MAX_SEQ) {
        eprintln!("error: {msg}");
        return ExitCode::FAILURE;
    }
    let explicit_cap = if max_new_tokens > 256 {
        Some(max_new_tokens)
    } else {
        None
    };

    let stop_token_ids = config::load_generation_stop_tokens(model_dir);
    if verbose {
        eprintln!("chat: full-message stop tokens = {stop_token_ids:?}, cap = {explicit_cap:?}");
    }

    let sampler = sample::sampler_for_steps(steps, no_early_stop);
    let mut step_cfg = StepGenerateConfig::from_generate(
        seed,
        CHAT_MAX_SEQ,
        CHAT_MAX_SEQ,
        layers,
        sampler,
        no_early_stop,
    );
    step_cfg.degenerate_reply_check =
        chat_template::empty_reply_check(model_dir, stop_token_ids.clone());
    step_cfg.stop_token_ids = stop_token_ids;

    if interactive {
        print!("loading model… ");
        let _ = io::stdout().flush();
    }
    let open_started = std::time::Instant::now();
    let mut session = match StepGenerateSession::open(model_dir, &step_cfg, None) {
        Ok((s, compile)) => {
            if interactive {
                println!("ready ({:.1}s)", open_started.elapsed().as_secs_f64());
            } else if verbose {
                eprintln!("chat: session ready ({compile:.2?}, layers={layers})");
            }
            s
        }
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = match tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => std::sync::Arc::new(t),
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // (No warm-up: COLD-START-1 — the first fresh-session generation returning an
    // empty/EOS reply — is fixed at the root by the deterministic first-step SC
    // seed. The throwaway warm-up generation is no longer needed.)

    let mut history: Vec<chat_template::ChatTurn> = Vec::new();
    let mut turn_idx = 0u64;

    let mut run_turn = |history: &mut Vec<chat_template::ChatTurn>,
                        turn_idx: &mut u64|
     -> Result<(), crate::Error> {
        let prompt = build_chat_prompt_tokens(model_dir, history, raw_prompt)?;
        let prompt_len = prompt.len();
        // KV arena is fixed at session open (CHAT_MAX_SEQ); the reply may use
        // the whole remaining context (no artificial cap), or the caller's
        // explicit ceiling if one was given. Reserve one CANVAS block: each
        // denoise block writes a CANVAS-wide canvas at [kv_len..kv_len+CANVAS],
        // so kv_len + CANVAS must stay within max_seq — this keeps the
        // `set_kv_len` overflow assert unreachable from a near-full context.
        let budget = CHAT_MAX_SEQ.saturating_sub(prompt_len + metal::CANVAS);
        if budget == 0 {
            if !json {
                println!(
                    "model> (prompt leaves no room for a reply within the {CHAT_MAX_SEQ}-token context; cannot generate)"
                );
            }
            history.push(chat_template::ChatTurn::model(String::new()));
            *turn_idx = turn_idx.wrapping_add(1);
            return Ok(());
        }
        step_cfg.max_new_tokens = explicit_cap.map_or(budget, |c| c.min(budget));
        step_cfg.seed = seed.wrapping_add(*turn_idx);
        let this_turn = *turn_idx;
        *turn_idx = turn_idx.wrapping_add(1);

        let started = std::time::Instant::now();
        let stream = chat_ui::ChatStream::start(
            std::sync::Arc::clone(&tokenizer),
            step_cfg.stop_token_ids.clone(),
            interactive,
            make_json_sink(want_json),
            this_turn,
            prompt_len,
        );
        step_cfg.step_observer = Some(stream.observer());
        let out = generate_with_session(&mut session, &prompt, &step_cfg, "chat")?;
        step_cfg.step_observer = None;
        let elapsed = started.elapsed();

        let new_ids =
            sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
        let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
        let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
        let secs = elapsed.as_secs_f64();
        stream.finish(
            &reply,
            new_tokens,
            out.denoise_steps_run,
            secs,
            out.stopped_on_eot,
        );

        if !interactive && !json {
            // Non-interactive human output (piped / --verbose): print the reply
            // once, since the terminal renderer was disabled.
            if reply.is_empty() {
                println!("model> (empty response)");
            } else {
                println!("model> {reply}");
            }
        }
        if !json {
            let cap_note = if out.stopped_on_eot {
                ""
            } else {
                " · hit context limit"
            };
            println!(
                "  ({new_tokens} tok · {} steps · {secs:.1}s · {:.1} tok/s{cap_note})",
                out.denoise_steps_run,
                new_tokens as f64 / secs.max(1e-9),
            );
        }
        history.push(chat_template::ChatTurn::model(reply));
        Ok(())
    };

    if let Some(first) = initial_prompt {
        let first = first.trim();
        if !first.is_empty() {
            history.push(chat_template::ChatTurn::user(first));
            if let Err(err) = run_turn(&mut history, &mut turn_idx) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    }

    if !json {
        println!("chat ready (type 'exit' or 'quit' to end; Ctrl-D also exits)");
    }
    let stdin = io::stdin();
    loop {
        if !json {
            print!("you> ");
            let _ = io::stdout().flush();
        }
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        history.push(chat_template::ChatTurn::user(line));
        if let Err(err) = run_turn(&mut history, &mut turn_idx) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
