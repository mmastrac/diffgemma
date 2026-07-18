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

/// The `<|"|>` string-quote special token (id 52) used throughout the grammar.
pub(crate) const Q: &str = "<|\"|>";

mod parse;
mod render;

#[cfg(test)]
pub use parse::ParsedToolCall;
pub(crate) use parse::strip_thinking;
pub use parse::{
    content_before_tool_calls, message_text, parse_tool_calls, scan_call_attempts,
    should_continue_past_stop, thinking_call_names, to_openai_tool_calls,
    tool_call_lost_in_thinking, validate_tool_reply,
};
#[cfg(test)]
pub use render::format_tool_declarations;
#[cfg(test)]
pub(crate) use render::render_tool_response;
pub(crate) use render::render_tool_response_guarded;
pub use render::{render_conversation, render_conversation_guarded, strip_client_guards};

#[cfg(test)]
mod tests;
