// diffgemma-mps is an Apple-Silicon Metal inference engine. Metal is the only
// backend — there is no CPU forward/generate path — so the whole program is
// macOS-only. Fail fast on other targets with one clear message rather than a
// wall of "cannot find `metal`" errors from the GPU-referencing kernel code.
#[cfg(not(target_os = "macos"))]
compile_error!(
    "diffgemma-mps requires macOS on Apple Silicon: Metal is the only backend \
     and there is no CPU inference path."
);

mod buffer;
#[cfg(target_os = "macos")]
mod chat_protocol;
mod chat_template;
#[cfg(target_os = "macos")]
mod chat_ui;
mod config;
#[cfg(target_os = "macos")]
mod conversation;
mod denoise_trace;
mod dgq;
mod fast_slice;
mod flags;
mod generate;
mod generate_golden;
mod golden;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod safetensors;
mod sample;
#[cfg(target_os = "macos")]
mod server;
#[allow(dead_code)]
mod shaders;
mod tensor;
mod tokenizer;
mod toolcompact;
mod tools;
mod weights;

mod cli;
mod commands;

use std::process::ExitCode;

fn main() -> ExitCode {
    commands::dispatch(cli::parse_cli())
}
