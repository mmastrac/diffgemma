//! HTTP request parsing + SSE / JSON response layer (extracted from server.rs,
//! backlog item 10). A child module: `use super::*` + ancestry reach the wire
//! types, `Job`/`ServerEvent`, and the shared `build_turns`/`needs_tool_rendering`
//! helpers in the parent.

use serde::Serialize;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::mpsc;

use super::*;

pub(crate) fn cap_tokens(
    tokenizer: &crate::tokenizer::Tokenizer,
    text: &str,
    max_tokens: usize,
) -> String {
    let max_tokens = max_tokens.max(16);
    let ids = tokenizer.encode(text, false);
    if ids.len() <= max_tokens {
        return text.to_string();
    }
    let mut s = tokenizer.decode(&ids[..max_tokens]);
    s.push_str("\n[excerpt truncated]");
    s
}

// ===========================================================================
// HTTP layer.
// ===========================================================================

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// Parse an HTTP/1.1 request: request line, headers, and (Content-Length) body.
fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None); // connection closed
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break; // end of headers
        }
        if let Some(v) = h
            .split_once(':')
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.trim())
        {
            content_length = v.parse().unwrap_or(0);
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Ok(Some(Request {
            method,
            path,
            body: Vec::new(),
        }));
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn write_json(stream: &mut TcpStream, status: &str, value: &serde_json::Value) {
    let body = serde_json::to_vec(value).unwrap_or_default();
    write_response(stream, status, "application/json", &body);
}

fn error_json(msg: &str, kind: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": msg, "type": kind } })
}

pub(crate) fn handle_connection(
    mut stream: TcpStream,
    jobs: mpsc::Sender<Job>,
    model_name: Arc<str>,
    think_default: bool,
) {
    let req = match read_request(&mut stream) {
        Ok(Some(r)) => r,
        _ => return,
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            write_json(
                &mut stream,
                "200 OK",
                &serde_json::json!({ "status": "ok" }),
            );
        }
        ("GET", "/v1/models") => {
            write_json(
                &mut stream,
                "200 OK",
                &serde_json::json!({
                    "object": "list",
                    "data": [
                        { "id": &*model_name, "object": "model", "owned_by": "local" },
                        { "id": format!("{}:think", model_name), "object": "model", "owned_by": "local" },
                        { "id": format!("{}:think=false", model_name), "object": "model", "owned_by": "local" },
                    ],
                }),
            );
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat(&mut stream, &req.body, &jobs, &model_name, think_default);
        }
        ("OPTIONS", _) => {
            write_response(&mut stream, "204 No Content", "text/plain", b"");
        }
        _ => {
            write_json(
                &mut stream,
                "404 Not Found",
                &error_json("unknown route", "invalid_request_error"),
            );
        }
    }
}

fn handle_chat(
    stream: &mut TcpStream,
    body: &[u8],
    jobs: &mpsc::Sender<Job>,
    model_name: &Arc<str>,
    think_default: bool,
) {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(err) => {
            write_json(
                stream,
                "400 Bad Request",
                &error_json(
                    &format!("invalid JSON body: {err}"),
                    "invalid_request_error",
                ),
            );
            return;
        }
    };

    if req.messages.is_empty() {
        write_json(
            stream,
            "400 Bad Request",
            &error_json("no messages", "invalid_request_error"),
        );
        return;
    }

    let enable_thinking = resolve_enable_thinking(
        req.model.as_deref(),
        req.enable_thinking,
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking),
        think_default,
    );
    let response_model: Arc<str> = Arc::from(
        req.model
            .as_deref()
            .unwrap_or(model_name.as_ref())
            .to_string(),
    );

    let (resp_tx, resp_rx) = mpsc::channel();
    let job = Job {
        messages: req.messages,
        tools: req.tools,
        stream: req.stream,
        max_tokens: req.max_tokens,
        seed: req.seed,
        enable_thinking,
        // Draft deltas only exist for streaming responses; a non-streaming
        // request would decode the full canvas every step just to have
        // respond_json discard the events.
        emit_drafts: req.x_diffusion_drafts.unwrap_or(true) && req.stream,
        resp: resp_tx,
    };
    let streaming = job.stream;
    if jobs.send(job).is_err() {
        write_json(
            stream,
            "503 Service Unavailable",
            &error_json("generation worker unavailable", "server_error"),
        );
        return;
    }

    if streaming {
        stream_sse(stream, resp_rx, &response_model);
    } else {
        respond_json(stream, resp_rx, &response_model);
    }
}

/// Build plain `ChatTurn`s from raw messages for the non-tool prompt path
/// (`format_chat_template` — the gate-validated encoding). system→user,
/// assistant→model; content flattened via `tools::message_text`.
pub(crate) fn build_turns(messages: &[serde_json::Value]) -> Vec<crate::chat_template::ChatTurn> {
    messages
        .iter()
        .filter_map(|m| {
            let role = match m.get("role").and_then(|r| r.as_str()) {
                Some("user") | Some("system") | Some("developer") => {
                    crate::chat_template::ChatRole::User
                }
                Some("assistant") | Some("model") => crate::chat_template::ChatRole::Model,
                _ => return None,
            };
            Some(crate::chat_template::ChatTurn {
                role,
                content: crate::tools::message_text(m),
                thought: None,
            })
        })
        .collect()
}

/// True when a request needs the tool-aware renderer: `tools` present, or any
/// message is a `tool` result or carries `tool_calls`.
pub(crate) fn needs_tool_rendering(
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> bool {
    !tools.is_empty()
        || messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("tool") || m.get("tool_calls").is_some()
        })
}

/// Stream Server-Sent Events: one `chat.completion.chunk` per delta, ending with
/// a `finish_reason` chunk and `data: [DONE]`.
fn stream_sse(stream: &mut TcpStream, rx: mpsc::Receiver<ServerEvent>, model_name: &Arc<str>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let id = format!("chatcmpl-{}", next_id());
    let created = now_secs();
    let model = model_name.to_string();

    // Opening role chunk (OpenAI convention).
    let _ = send_chunk(
        stream,
        &id,
        created,
        &model,
        ChunkChoice {
            index: 0,
            delta: DeltaBody {
                role: Some("assistant"),
                ..empty_delta()
            },
            finish_reason: None,
        },
    );

    let mut finish = "stop";
    for ev in rx {
        match ev {
            ServerEvent::Delta(d) => {
                if send_chunk(
                    stream,
                    &id,
                    created,
                    &model,
                    ChunkChoice {
                        index: 0,
                        delta: d.into_delta_body(),
                        finish_reason: None,
                    },
                )
                .is_err()
                {
                    return; // client hung up
                }
            }
            ServerEvent::Done {
                stopped,
                tool_calls,
                ..
            } => {
                if !tool_calls.is_empty() {
                    finish = "tool_calls";
                    let _ = send_chunk(
                        stream,
                        &id,
                        created,
                        &model,
                        ChunkChoice {
                            index: 0,
                            delta: DeltaBody {
                                tool_calls: Some(tool_calls),
                                ..empty_delta()
                            },
                            finish_reason: None,
                        },
                    );
                } else if !stopped {
                    finish = "length";
                }
                break;
            }
            ServerEvent::Error(msg) => {
                let _ = write_sse_data(stream, &error_json(&msg, "server_error"));
                let _ = stream.write_all(b"data: [DONE]\n\n");
                let _ = stream.flush();
                return;
            }
        }
    }

    let _ = send_chunk(
        stream,
        &id,
        created,
        &model,
        ChunkChoice {
            index: 0,
            delta: empty_delta(),
            finish_reason: Some(finish),
        },
    );
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

fn send_chunk(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    choice: ChunkChoice,
) -> std::io::Result<()> {
    let chunk = Chunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![choice],
    };
    write_sse_data(stream, &chunk)
}

fn write_sse_data<T: Serialize>(stream: &mut TcpStream, value: &T) -> std::io::Result<()> {
    let json = serde_json::to_string(value).unwrap_or_default();
    stream.write_all(b"data: ")?;
    stream.write_all(json.as_bytes())?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

/// Non-streaming: drain the worker to completion, then write one JSON response.
fn respond_json(stream: &mut TcpStream, rx: mpsc::Receiver<ServerEvent>, model_name: &Arc<str>) {
    for ev in rx {
        match ev {
            ServerEvent::Delta(_) => {}
            ServerEvent::Done {
                content,
                reasoning,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                stopped,
            } => {
                let has_calls = !tool_calls.is_empty();
                let mut message = serde_json::json!({
                    "role": "assistant",
                    // OpenAI convention: content is null when only tool_calls.
                    "content": if has_calls && content.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(content)
                    },
                });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = serde_json::Value::String(reasoning);
                }
                if has_calls {
                    message["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                let finish_reason = if has_calls {
                    "tool_calls"
                } else if stopped {
                    "stop"
                } else {
                    "length"
                };
                let value = serde_json::json!({
                    "id": format!("chatcmpl-{}", next_id()),
                    "object": "chat.completion",
                    "created": now_secs(),
                    "model": &**model_name,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                });
                write_json(stream, "200 OK", &value);
                return;
            }
            ServerEvent::Error(msg) => {
                write_json(
                    stream,
                    "500 Internal Server Error",
                    &error_json(&msg, "server_error"),
                );
                return;
            }
        }
    }
}
