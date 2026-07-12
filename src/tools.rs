//! Tool-use (function calling) formatting + parsing for diffusiongemma.
//!
//! The model was trained with a custom tool grammar (its `chat_template.jinja`,
//! reproduced by `python/scripts/tool_format_oracle.py`), NOT plain JSON. This
//! module mirrors that grammar exactly so tool definitions we inject match what
//! the model saw in training, and parses the calls it emits back into the OpenAI
//! `tool_calls` shape. Pure (no GPU) and unit-tested byte-for-byte against the
//! jinja oracle.
//!
//! Grammar (see the oracle):
//! - **definition**: `<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{
//!   properties:{KEY:{description:<|"|>…<|"|>,…,type:<|"|>TYPE<|"|>},…},
//!   required:[<|"|>k<|"|>,…],type:<|"|>OBJECT<|"|>}}<tool|>` — property keys
//!   sorted, string values/descriptions/types wrapped in the `<|"|>` quote token,
//!   types upper-cased, `type` always last within a property.
//! - **call** (model output): `<|tool_call>call:NAME{key:value,…}<tool_call|>`,
//!   one wrapper per call, bare keys, `<|"|>`-quoted string values.

use serde_json::{Map, Value};

/// The `<|"|>` string-quote special token (id 52) used throughout the grammar.
const Q: &str = "<|\"|>";

// ===========================================================================
// Definitions: OpenAI `tools` → the model's `<|tool>declaration:…<tool|>` grammar
// ===========================================================================

/// Format OpenAI `tools` into the concatenated `<|tool>…<tool|>` declaration
/// blocks the model expects in the system turn. Returns "" for no tools.
#[allow(dead_code)]
pub fn format_tool_declarations(tools: &[Value]) -> String {
    tools
        .iter()
        .map(|t| format!("<|tool>{}<tool|>", format_declaration(t)))
        .collect()
}

fn format_declaration(tool: &Value) -> String {
    let f = tool.get("function").unwrap_or(tool);
    let name = f.get("name").and_then(Value::as_str).unwrap_or("");
    let desc = f.get("description").and_then(Value::as_str).unwrap_or("");
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
            fields.push(format!("type:{Q}{}{Q}", ty.to_uppercase()));
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
        .map(|k| format!("{k}:{{{}}}", format_property_body(&props[*k])))
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
        fields.push(format!("description:{Q}{d}{Q}"));
    }
    if ty == "STRING" {
        if let Some(en) = v.get("enum").filter(|e| e.is_array()) {
            fields.push(format!("enum:{}", format_argument(en, true)));
        }
    } else if ty == "ARRAY" {
        if let Some(items) = v.get("items").filter(|i| i.is_object()) {
            fields.push(format!("items:{{{}}}", format_items_body(items)));
        }
    }
    if v.get("nullable").and_then(Value::as_bool) == Some(true) {
        fields.push("nullable:true".into());
    }
    if ty == "OBJECT" {
        if let Some(props) = v.get("properties").and_then(Value::as_object) {
            fields.push(format!("properties:{{{}}}", format_parameters(props)));
            let req = required_list(v);
            if !req.is_empty() {
                fields.push(format!("required:[{req}]"));
            }
        }
    }
    fields.push(format!("type:{Q}{ty}{Q}"));
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
                Some(s) => format!("type:{Q}{}{Q}", s.to_uppercase()),
                None => continue,
            },
            "properties" => match v.as_object() {
                Some(p) => format!("properties:{{{}}}", format_parameters(p)),
                None => continue,
            },
            "required" => format!("required:[{}]", required_list(items)),
            _ => format!("{k}:{}", format_argument(v, true)),
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
                .map(|s| format!("{Q}{s}{Q}"))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default()
}

/// jinja `format_argument`: strings quoted with `<|"|>`, maps/arrays recursive,
/// bools bare. `escape_keys` quotes object keys (definitions) vs not (call args).
fn format_argument(v: &Value, escape_keys: bool) -> String {
    match v {
        Value::String(s) => format!("{Q}{s}{Q}"),
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
                        format!("{Q}{k}{Q}")
                    } else {
                        (*k).clone()
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
pub fn render_conversation(
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
            out.push_str(message_text(&messages[0]).trim());
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
        out.push_str(&content);
        let has_content = !content.trim().is_empty();

        if prev == PrevType::ToolCall && !responses_emitted {
            out.push_str("<|tool_response>");
        } else if !(responses_emitted && !has_content) {
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
                .map(|k| format!("{k}:{}", format_argument(&m[*k], false)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    format!("<|tool_call>call:{name}{{{body}}}<tool_call|>")
}

/// Render one tool response in the canonical grammar (server-side tool
/// execution, e.g. the compactor's `expand_summary`, extends KV with this).
pub(crate) fn render_tool_response(name: &str, msg: &Value) -> String {
    format_tool_response(name.to_string(), msg)
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
                .map(|k| format!("{k}:{}", format_argument(&m[*k], false)))
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
    format!("<|tool_response>response:{name}{{{body}}}<tool_response|>")
}

/// Resolve a `tool` message's function name via its `tool_call_id` against the
/// preceding assistant `tool_calls`, else its `name`, else "unknown".
fn tool_name_for(msg: &Value, calls: Option<&Vec<Value>>) -> String {
    if let (Some(id), Some(calls)) = (msg.get("tool_call_id").and_then(Value::as_str), calls) {
        for tc in calls {
            if tc.get("id").and_then(Value::as_str) == Some(id) {
                if let Some(n) = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                {
                    return n.to_string();
                }
            }
        }
    }
    msg.get("name")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string()
}

/// The user-facing content of a tool-call response: the preamble before the
/// first `<|tool_call>` (the between-call / trailing text is dropped — clients
/// treat `content` as the preamble when `tool_calls` are present).
pub fn content_before_tool_calls(text: &str) -> String {
    match text.find("<|tool_call>") {
        Some(i) => text[..i].trim().to_string(),
        None => text.trim().to_string(),
    }
}

/// Parsed calls → OpenAI `tool_calls` array (`arguments` serialized as a string).
pub fn to_openai_tool_calls(calls: &[ParsedToolCall]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "id": format!("call_{i}"),
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments.to_string()},
            })
        })
        .collect()
}

/// Flatten a message's `content` (string or content-parts array) to text.
pub fn message_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Remove `<|channel>…<channel|>` thought spans (jinja `strip_thinking`).
fn strip_thinking(text: &str) -> String {
    let mut result = String::new();
    for part in text.split("<channel|>") {
        match part.split_once("<|channel>") {
            Some((before, _)) => result.push_str(before),
            None => result.push_str(part),
        }
    }
    result.trim().to_string()
}

// ===========================================================================
// Calls: the model's `<|tool_call>call:NAME{…}<tool_call|>` output → OpenAI shape
// ===========================================================================

/// A tool call parsed from model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    /// Arguments as a JSON object (serialized to a string for the OpenAI field).
    pub arguments: Value,
}

/// Parse every `call:NAME{ARGS}` in model output into structured tool calls,
/// tolerant of the exact wrapper tokens around/between them.
pub fn parse_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("call:") {
        let after = &rest[pos + "call:".len()..];
        let Some(brace) = after.find('{') else { break };
        // A malformed/unbalanced call must not drop the calls that follow it:
        // skip past this `call:` and keep scanning.
        let Some((inner, consumed)) = read_braced(&after[brace..]) else {
            rest = after;
            continue;
        };
        let name = after[..brace].trim().to_string();
        out.push(ParsedToolCall {
            name,
            arguments: parse_object(inner, 0),
        });
        rest = &after[brace + consumed..];
    }
    out
}

/// Given a string starting with '{', return (inner content, bytes consumed incl.
/// both braces), respecting nested braces. `None` if unbalanced.
fn read_braced(s: &str) -> Option<(&str, usize)> {
    let b = s.as_bytes();
    if b.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[1..i], i + 1));
                }
            }
            _ => {}
        }
    }
    None
}

/// Cap on argument nesting depth — past it, a value is kept as a raw string
/// rather than recursed into (guards against runaway `{{{…}}}` output).
const MAX_ARG_DEPTH: u32 = 32;

/// Parse a `key:value,key:value` body (bare keys, `<|"|>`-quoted strings) into a
/// JSON object.
fn parse_object(inner: &str, depth: u32) -> Value {
    let mut map = Map::new();
    for field in split_top_level(inner) {
        let Some(colon) = field.find(':') else {
            continue;
        };
        let key = field[..colon].trim().trim_matches('"').to_string();
        if key.is_empty() {
            continue;
        }
        map.insert(key, parse_value(field[colon + 1..].trim(), depth));
    }
    Value::Object(map)
}

fn parse_value(s: &str, depth: u32) -> Value {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix(Q).and_then(|x| x.strip_suffix(Q)) {
        return Value::String(inner.to_string());
    }
    if let Some(inner) = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        return Value::String(inner.to_string());
    }
    if depth < MAX_ARG_DEPTH {
        if s.starts_with('{') {
            if let Some((body, _)) = read_braced(s) {
                return parse_object(body, depth + 1);
            }
        }
        if let Some(body) = s.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
            return Value::Array(
                split_top_level(body)
                    .into_iter()
                    .map(|v| parse_value(v, depth + 1))
                    .collect(),
            );
        }
    }
    match s {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    if let Ok(n) = serde_json::from_str::<serde_json::Number>(s) {
        return Value::Number(n);
    }
    Value::String(s.to_string())
}

/// Byte-substring search (needle in haystack) → start index.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Split on top-level commas, respecting `{}`/`[]` nesting and `<|"|>` strings.
///
/// Byte-safe: scans `s.as_bytes()` and only ever slices `s` at ASCII delimiter
/// positions (commas/brackets), which are always char boundaries — so multi-byte
/// UTF-8 anywhere in the input can never split mid-character (the previous
/// `s[i..]` scan panicked on e.g. a bare `naïve`).
fn split_top_level(s: &str) -> Vec<&str> {
    let qb = Q.as_bytes();
    let bytes = s.as_bytes();
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    let mut i = 0;
    while i < bytes.len() {
        // Skip a whole `<|"|>…<|"|>` quoted run so delimiters inside don't count.
        if bytes[i..].starts_with(qb) {
            let after = i + qb.len();
            if let Some(rel) = find_bytes(&bytes[after..], qb) {
                i = after + rel + qb.len();
                continue;
            }
        }
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < s.len() || !s.is_empty() && start == s.len() {
        let tail = s[start..].trim();
        if !tail.is_empty() {
            parts.push(tail);
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Oracle strings from python/scripts/tool_format_oracle.py (the reference jinja).

    #[test]
    fn declaration_matches_jinja_oracle_simple() {
        let tools = [json!({"type":"function","function":{
            "name":"list_dir",
            "description":"Lists all files and subdirectories in a specified directory path",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"The absolute path to the directory to list (e.g., /tmp)"}},
                "required":["path"]}}})];
        let expected = "<|tool>declaration:list_dir{description:<|\"|>Lists all files and subdirectories in a specified directory path<|\"|>,parameters:{properties:{path:{description:<|\"|>The absolute path to the directory to list (e.g., /tmp)<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
        assert_eq!(format_tool_declarations(&tools), expected);
    }

    #[test]
    fn declaration_matches_jinja_oracle_rich() {
        let tools = [json!({"type":"function","function":{
            "name":"search",
            "description":"Search the web",
            "parameters":{"type":"object","properties":{
                "query":{"type":"string","description":"the query"},
                "count":{"type":"integer","description":"how many"},
                "safe":{"type":"boolean","description":"safe search"},
                "sort":{"type":"string","description":"sort order","enum":["relevance","date"]},
                "tags":{"type":"array","description":"tags","items":{"type":"string"}}},
                "required":["query"]}}})];
        let expected = "<|tool>declaration:search{description:<|\"|>Search the web<|\"|>,parameters:{properties:{count:{description:<|\"|>how many<|\"|>,type:<|\"|>INTEGER<|\"|>},query:{description:<|\"|>the query<|\"|>,type:<|\"|>STRING<|\"|>},safe:{description:<|\"|>safe search<|\"|>,type:<|\"|>BOOLEAN<|\"|>},sort:{description:<|\"|>sort order<|\"|>,enum:[<|\"|>relevance<|\"|>,<|\"|>date<|\"|>],type:<|\"|>STRING<|\"|>},tags:{description:<|\"|>tags<|\"|>,items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>}},required:[<|\"|>query<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
        assert_eq!(format_tool_declarations(&tools), expected);
    }

    #[test]
    fn parse_single_call() {
        let calls =
            parse_tool_calls("<|tool_call>call:list_dir{path:<|\"|>/tmp<|\"|>}<tool_call|>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_dir");
        assert_eq!(calls[0].arguments, json!({"path":"/tmp"}));
    }

    #[test]
    fn parse_multiple_calls() {
        let calls = parse_tool_calls(
            "<|tool_call>call:list_dir{path:<|\"|>/tmp<|\"|>}<tool_call>call:list_dir{path:<|\"|>/home<|\"|>}<tool_call|>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({"path":"/tmp"}));
        assert_eq!(calls[1].arguments, json!({"path":"/home"}));
    }

    #[test]
    fn parse_typed_args() {
        let calls = parse_tool_calls("call:search{query:<|\"|>rust<|\"|>,count:5,safe:true}");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            json!({"query":"rust","count":5,"safe":true})
        );
    }

    #[test]
    fn parse_none_when_no_calls() {
        assert!(parse_tool_calls("just a normal answer, no tools here").is_empty());
    }

    #[test]
    fn parse_never_panics_on_adversarial_input() {
        let cases = [
            // Multi-byte UTF-8 inside a call (high-temp model can emit these).
            "call:list_dir{path:<|\"|>/tmp/café/naïve/日本<|\"|>}",
            "<|tool_call>call:πcalc{x:<|\"|>±∞<|\"|>}<tool_call|>",
            // Multi-byte OUTSIDE a quoted run (bare key/value) — the hard case.
            "call:x{a:naïve,b:1}",
            "call:f{café:1}",
            "call:g{x:日本語}",
            // Deeply nested braces (runaway generation).
            &format!("call:x{{a:{}{}", "{".repeat(5000), "}".repeat(5000)),
            // Unbalanced / truncated.
            "call:list_dir{path:<|\"|>/tmp",
            "call:a{b:{c:{d:",
            "<|tool_call>call:",
            "call:{}}}}}",
            // Garbage bytes near markers.
            "call:\u{feff}\u{200b}{k\u{0}:v}",
        ];
        for c in cases {
            let _ = parse_tool_calls(c); // must return, not panic
        }
    }

    #[test]
    fn parse_total_under_random_fuzz() {
        // Deterministic LCG (no deps) → strings assembled from tool-call
        // fragments, delimiters, and multi-byte chars. `parse_tool_calls` must
        // return on every one (a panic here = a server crash on model output).
        let frags = [
            "call:",
            "<|tool_call>",
            "<tool_call|>",
            "{",
            "}",
            "[",
            "]",
            ",",
            ":",
            "<|\"|>",
            "path",
            "/tmp",
            "naïve",
            "日本",
            "±∞",
            "true",
            "42",
            " ",
            "\n",
            "x",
        ];
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for _ in 0..20_000 {
            let n = next() % 40;
            let mut s = String::new();
            for _ in 0..n {
                s.push_str(frags[next() % frags.len()]);
            }
            let _ = parse_tool_calls(&s);
            let _ = content_before_tool_calls(&s);
        }
    }

    fn list_dir_tool() -> Value {
        json!({"type":"function","function":{
            "name":"list_dir",
            "description":"Lists all files and subdirectories in a specified directory path",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"The absolute path to the directory to list (e.g., /tmp)"}},
                "required":["path"]}}})
    }

    #[test]
    fn render_tools_plus_user_matches_oracle() {
        let msgs = [json!({"role":"user","content":"List files in /tmp"})];
        let expected = "<bos><|turn>system\n<|tool>declaration:list_dir{description:<|\"|>Lists all files and subdirectories in a specified directory path<|\"|>,parameters:{properties:{path:{description:<|\"|>The absolute path to the directory to list (e.g., /tmp)<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\nList files in /tmp<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        assert_eq!(
            render_conversation(&msgs, &[list_dir_tool()], true, false),
            expected
        );
    }

    #[test]
    fn render_tool_call_and_response_matches_oracle() {
        let msgs = [
            json!({"role":"user","content":"List files in /tmp"}),
            json!({"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"list_dir","arguments":{"path":"/tmp"}}}]}),
            json!({"role":"tool","tool_call_id":"c1","content":"a.txt\nb.txt"}),
        ];
        let expected = "<bos><|turn>system\n<|tool>declaration:list_dir{description:<|\"|>Lists all files and subdirectories in a specified directory path<|\"|>,parameters:{properties:{path:{description:<|\"|>The absolute path to the directory to list (e.g., /tmp)<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\nList files in /tmp<turn|>\n<|turn>model\n<|tool_call>call:list_dir{path:<|\"|>/tmp<|\"|>}<tool_call|><|tool_response>response:list_dir{value:<|\"|>a.txt\nb.txt<|\"|>}<tool_response|>";
        assert_eq!(
            render_conversation(&msgs, &[list_dir_tool()], true, false),
            expected
        );
    }

    #[test]
    fn render_system_plus_tools_matches_oracle() {
        let msgs = [
            json!({"role":"system","content":"You are helpful."}),
            json!({"role":"user","content":"hi"}),
        ];
        let expected = "<bos><|turn>system\nYou are helpful.<|tool>declaration:list_dir{description:<|\"|>Lists all files and subdirectories in a specified directory path<|\"|>,parameters:{properties:{path:{description:<|\"|>The absolute path to the directory to list (e.g., /tmp)<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        assert_eq!(
            render_conversation(&msgs, &[list_dir_tool()], true, false),
            expected
        );
    }
}
