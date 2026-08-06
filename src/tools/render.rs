use serde_json::{Map, Value};

use super::Q;
use super::parse::{message_text, strip_thinking};
use crate::tokenizer::CLIENT_GUARD;

/// Wrap client-supplied text in [`CLIENT_GUARD`] sentinels so
/// `Tokenizer::encode_prompt` refuses to special-match inside it: a special-
/// token literal in a message body, tool output, name, key, or description
/// encodes as plain text instead of protocol tokens. Guard chars already in
/// the text are dropped (private-use; meaningless in real content, and a
/// literal one would corrupt range tracking).
fn guard(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push(CLIENT_GUARD);
    out.extend(s.chars().filter(|&c| c != CLIENT_GUARD));
    out.push(CLIENT_GUARD);
    out
}

/// Remove guard sentinels — the display/log form of a guarded render.
pub fn strip_client_guards(s: &str) -> String {
    s.chars().filter(|&c| c != CLIENT_GUARD).collect()
}

// ===========================================================================
// Definitions: OpenAI `tools` → the model's `<|tool>declaration:…<tool|>` grammar
// ===========================================================================

/// Format OpenAI `tools` into the concatenated `<|tool>…<tool|>` declaration
/// blocks the model expects in the system turn. Returns "" for no tools.
#[allow(dead_code)]
pub fn format_tool_declarations(tools: &[Value]) -> String {
    strip_client_guards(
        &tools
            .iter()
            .map(|t| format!("<|tool>{}<tool|>", format_declaration(t)))
            .collect::<String>(),
    )
}

fn format_declaration(tool: &Value) -> String {
    let f = tool.get("function").unwrap_or(tool);
    let name = guard(f.get("name").and_then(Value::as_str).unwrap_or(""));
    let desc = guard(f.get("description").and_then(Value::as_str).unwrap_or(""));
    let mut s = format!("declaration:{name}{{description:{Q}{desc}{Q}");
    if let Some(params) = f.get("parameters").filter(|p| p.is_object()) {
        let mut fields: Vec<String> = Vec::new();
        if let Some(props) = params.get("properties").and_then(Value::as_object) {
            fields.push(format!("properties:{{{}}}", format_parameters(props)));
        }
        let req = required_list(params);
        if !req.is_empty() {
            fields.push(format!("required:[{req}]"));
        }
        if let Some(ty) = params.get("type").and_then(Value::as_str) {
            fields.push(format!("type:{Q}{}{Q}", guard(&ty.to_uppercase())));
        }
        s.push_str(",parameters:{");
        s.push_str(&fields.join(","));
        s.push('}');
    }
    s.push('}');
    s
}

/// `key:{body},key:{body}` for properties, keys in sorted order (jinja `dictsort`).
fn format_parameters(props: &Map<String, Value>) -> String {
    let mut keys: Vec<&String> = props.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| format!("{}:{{{}}}", guard(k), format_property_body(&props[*k])))
        .collect::<Vec<_>>()
        .join(",")
}

/// One property's inner fields: `description`, then type-specific (`enum`/`items`),
/// then `nullable`, then object `properties`/`required`, and `type` always last.
fn format_property_body(v: &Value) -> String {
    let ty = v
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_uppercase();
    let mut fields: Vec<String> = Vec::new();
    if let Some(d) = v.get("description").and_then(Value::as_str) {
        fields.push(format!("description:{Q}{}{Q}", guard(d)));
    }
    if ty == "STRING" {
        if let Some(en) = v.get("enum").filter(|e| e.is_array()) {
            fields.push(format!("enum:{}", format_argument(en, true)));
        }
    } else if ty == "ARRAY"
        && let Some(items) = v.get("items").filter(|i| i.is_object())
    {
        fields.push(format!("items:{{{}}}", format_items_body(items)));
    }
    if v.get("nullable").and_then(Value::as_bool) == Some(true) {
        fields.push("nullable:true".into());
    }
    if ty == "OBJECT"
        && let Some(props) = v.get("properties").and_then(Value::as_object)
    {
        fields.push(format!("properties:{{{}}}", format_parameters(props)));
        let req = required_list(v);
        if !req.is_empty() {
            fields.push(format!("required:[{req}]"));
        }
    }
    fields.push(format!("type:{Q}{}{Q}", guard(&ty)));
    fields.join(",")
}

/// Array `items` body: `type`, `properties`, `required` handled specially, keys sorted.
fn format_items_body(items: &Value) -> String {
    let m = items.as_object().unwrap();
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort();
    let mut parts = Vec::new();
    for k in keys {
        let v = &m[k];
        if v.is_null() {
            continue;
        }
        let part = match k.as_str() {
            "type" => match v.as_str() {
                Some(s) => format!("type:{Q}{}{Q}", guard(&s.to_uppercase())),
                None => continue,
            },
            "properties" => match v.as_object() {
                Some(p) => format!("properties:{{{}}}", format_parameters(p)),
                None => continue,
            },
            "required" => format!("required:[{}]", required_list(items)),
            _ => format!("{}:{}", guard(k), format_argument(v, true)),
        };
        parts.push(part);
    }
    parts.join(",")
}

/// `<|"|>a<|"|>,<|"|>b<|"|>` from an object's `required` list.
fn required_list(obj: &Value) -> String {
    obj.get("required")
        .and_then(Value::as_array)
        .map(|r| {
            r.iter()
                .filter_map(Value::as_str)
                .map(|s| format!("{Q}{}{Q}", guard(s)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default()
}

/// jinja `format_argument`: strings quoted with `<|"|>`, maps/arrays recursive,
/// bools bare. `escape_keys` quotes object keys (definitions) vs not (call args).
fn format_argument(v: &Value, escape_keys: bool) -> String {
    match v {
        Value::String(s) => format!("{Q}{}{Q}", guard(s)),
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Number(n) => n.to_string(),
        Value::Null => String::new(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter()
                .map(|x| format_argument(x, escape_keys))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let body = keys
                .iter()
                .map(|k| {
                    let key = if escape_keys {
                        format!("{Q}{}{Q}", guard(k))
                    } else {
                        guard(k)
                    };
                    format!("{key}:{}", format_argument(&m[*k], escape_keys))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

// ===========================================================================
// Conversation rendering: OpenAI messages+tools → the model's prompt string.
//
// Mirrors the reference chat_template.jinja for the tool-use path (verified
// byte-for-byte against the oracle). Produces a STRING with special-token
// literals; the caller tokenizes it (the specials encode to their ids).
// ===========================================================================

/// Render a full conversation (with optional `tools`) to the model's prompt
/// string. Handles the system/tools block, user/assistant turns, assistant
/// `tool_calls`, and `tool`-role responses (forward-scanned onto the preceding
/// assistant turn). `add_generation_prompt` appends the `<|turn>model` scaffold
/// unless the last thing emitted was a tool call/response (model continues
/// in-turn). Thinking off seeds the empty thought channel, matching the default.
/// Display/log form of [`render_conversation_guarded`] — guard sentinels
/// stripped. Byte-identical to the historical render.
pub fn render_conversation(
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    strip_client_guards(&render_conversation_guarded(
        messages,
        tools,
        add_generation_prompt,
        enable_thinking,
    ))
}

/// Render with client-text guard sentinels intact — feed this to
/// `Tokenizer::encode_prompt` so special-token literals inside client text
/// encode as plain text (token-injection hardening) while template markup
/// becomes real special ids.
pub fn render_conversation_guarded(
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let role = |m: &Value| {
        m.get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let mut out = String::from("<bos>");
    let mut prev = PrevType::Other;

    let first_is_system = messages
        .first()
        .map(|m| matches!(role(m).as_str(), "system" | "developer"))
        .unwrap_or(false);

    if enable_thinking || !tools.is_empty() || first_is_system {
        out.push_str("<|turn>system\n");
        if enable_thinking {
            out.push_str("<|think|>\n");
        }
        if first_is_system {
            out.push_str(&guard(message_text(&messages[0]).trim()));
        }
        for tool in tools {
            out.push_str(&format!("<|tool>{}<tool|>", format_declaration(tool)));
        }
        out.push_str("<turn|>\n");
    }

    let start = if first_is_system { 1 } else { 0 };
    let mut i = start;
    while i < messages.len() {
        let m = &messages[i];
        let r = role(m);
        if r == "tool" {
            // A tool message not consumed by a preceding assistant turn: emit its
            // response standalone (rare; e.g. a leading tool result).
            out.push_str(&format_tool_response(tool_name_for(m, None), m));
            prev = PrevType::ToolResponse;
            i += 1;
            continue;
        }
        let render_role = if r == "assistant" {
            "model"
        } else {
            r.as_str()
        };
        out.push_str(&format!("<|turn>{render_role}\n"));

        let tool_calls = m
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|c| !c.is_empty());
        if let Some(calls) = tool_calls {
            for tc in calls {
                out.push_str(&format_tool_call(tc));
            }
            prev = PrevType::ToolCall;
        }

        // Forward-scan consecutive `tool` messages as responses on this turn.
        let mut responses_emitted = false;
        if tool_calls.is_some() {
            let mut k = i + 1;
            while k < messages.len() && role(&messages[k]) == "tool" {
                let name = tool_name_for(&messages[k], tool_calls);
                out.push_str(&format_tool_response(name, &messages[k]));
                responses_emitted = true;
                prev = PrevType::ToolResponse;
                k += 1;
            }
            i = k - 1; // consumed the tool messages
        }

        let content = if render_role == "model" {
            strip_thinking(&message_text(m))
        } else {
            message_text(m).trim().to_string()
        };
        out.push_str(&guard(&content));
        let has_content = !content.trim().is_empty();

        if prev == PrevType::ToolCall && !responses_emitted {
            out.push_str("<|tool_response>");
        } else if !responses_emitted || has_content {
            out.push_str("<turn|>\n");
            prev = PrevType::Other;
        }
        i += 1;
    }

    if add_generation_prompt && !matches!(prev, PrevType::ToolCall | PrevType::ToolResponse) {
        out.push_str("<|turn>model\n");
        if !enable_thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
    }
    out
}

#[derive(PartialEq)]
enum PrevType {
    Other,
    ToolCall,
    ToolResponse,
}

/// `<|tool_call>call:NAME{key:value,…}<tool_call|>` for an OpenAI tool_call
/// (arguments may be a JSON object or a JSON string), keys sorted, bare keys.
fn format_tool_call(tc: &Value) -> String {
    let f = tc.get("function").unwrap_or(tc);
    let name = f.get("name").and_then(Value::as_str).unwrap_or("");
    let args = f.get("arguments");
    let obj = match args {
        Some(Value::Object(m)) => Some(m.clone()),
        Some(Value::String(s)) => serde_json::from_str::<Value>(s)
            .ok()
            .and_then(|v| v.as_object().cloned()),
        _ => None,
    };
    let body = obj
        .map(|m| {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            keys.iter()
                .map(|k| format!("{}:{}", guard(k), format_argument(&m[*k], false)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    format!("<|tool_call>call:{}{{{body}}}<tool_call|>", guard(name))
}

/// Render one tool response in the canonical grammar (server-side tool
/// execution, e.g. the compactor's `expand_summary`, extends KV with this).
/// Display form — guards stripped; use the `_guarded` twin for encoding.
#[cfg(test)]
pub(crate) fn render_tool_response(name: &str, msg: &Value) -> String {
    strip_client_guards(&format_tool_response(name.to_string(), msg))
}

/// Guarded twin of [`render_tool_response`] for `Tokenizer::encode_prompt`.
pub(crate) fn render_tool_response_guarded(name: &str, msg: &Value) -> String {
    format_tool_response(name.to_string(), msg)
}

/// The forced-open thought marker appended to a tool-mode prompt: it opens the
/// thought channel so the model plans its calls inside the reasoning block
/// rather than narrating them into the visible answer. The decoder's channel
/// split and reasoning cleanup key off the same marker.
pub(crate) const OPEN_THOUGHT: &str = "<|channel>thought\n";

/// Continuation tail for a tool round: each `(name, content)` response rendered
/// as a guarded `<|tool_response>` block, then the reopened thought when
/// `thinking` so the next round keeps planning inside the reasoning block.
pub(crate) fn tool_continuation_tail(responses: &[(String, String)], thinking: bool) -> String {
    let mut tail = String::new();
    for (name, content) in responses {
        tail.push_str(&render_tool_response_guarded(
            name,
            &serde_json::json!({ "content": content }),
        ));
    }
    if thinking {
        tail.push_str(OPEN_THOUGHT);
    }
    tail
}

/// `<|tool_response>response:NAME{…}<tool_response|>`. String content wraps in a
/// `value:` key; object content is rendered field-by-field (bare keys).
fn format_tool_response(name: String, msg: &Value) -> String {
    let content = msg.get("content").cloned().unwrap_or(Value::Null);
    let body = match &content {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            keys.iter()
                .map(|k| format!("{}:{}", guard(k), format_argument(&m[*k], false)))
                .collect::<Vec<_>>()
                .join(",")
        }
        other => {
            let s = other
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| match other {
                    Value::Null => String::new(),
                    _ => other.to_string(),
                });
            format!("value:{}", format_argument(&Value::String(s), false))
        }
    };
    format!(
        "<|tool_response>response:{}{{{body}}}<tool_response|>",
        guard(&name)
    )
}

/// Resolve a `tool` message's function name via its `tool_call_id` against the
/// preceding assistant `tool_calls`, else its `name`, else "unknown".
fn tool_name_for(msg: &Value, calls: Option<&Vec<Value>>) -> String {
    if let (Some(id), Some(calls)) = (msg.get("tool_call_id").and_then(Value::as_str), calls) {
        for tc in calls {
            if tc.get("id").and_then(Value::as_str) == Some(id)
                && let Some(n) = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
            {
                return n.to_string();
            }
        }
    }
    msg.get("name")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string()
}
