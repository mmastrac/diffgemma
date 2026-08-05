//! Mini-harness definitions: a JSON file that configures a chat session as a
//! small tool-driven application (`chat --harness <file.json>`).
//!
//! A harness bundles a system prompt, an optional persistent prethink
//! template, file-backed variables, and shell-backed tools with declared
//! parameters. The pieces compose with the session machinery: tools steer via
//! `::end` / `::set` directive lines in their output, `$$var$$` templates
//! expand per turn, and a file-backed variable reads fresh from its file at
//! every use so tools can buffer multi-line state there.
//!
//! ```json
//! {
//!   "prompt": "You are ...",
//!   "prethink": "Scene so far: $$scene$$",
//!   "vars": { "scene": ".story/scene.txt" },
//!   "tools": [
//!     {
//!       "name": "set_scene",
//!       "description": "Replace the scene description.",
//!       "params": [ { "name": "scene", "description": "The new scene." } ],
//!       "command": "printf '%s' \"$scene\" > .story/scene.txt && echo ok"
//!     }
//!   ]
//! }
//! ```

use super::engine::{ShellTool, ToolParam};
use serde::Deserialize;

/// A parsed `--harness` file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Harness {
    /// System prompt, as if set by `/prompt`.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Persistent prethink template, as if set by `/prethink persistent`
    /// (`$$last$$` / `$$prompt$$` / `$$var$$` expand per turn).
    #[serde(default)]
    pub prethink: Option<String>,
    /// File-backed variables: `name` -> path, read fresh at every use. The
    /// file may not exist yet (reads empty until a tool writes it).
    #[serde(default)]
    pub vars: std::collections::HashMap<String, String>,
    #[serde(default)]
    tools: Vec<HarnessTool>,
}

/// One tool definition. `params` may be empty (a no-argument tool); each
/// param arrives in `command` as an env var of the same name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HarnessTool {
    name: String,
    #[serde(default)]
    description: String,
    command: String,
    #[serde(default)]
    params: Vec<HarnessParam>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HarnessParam {
    name: String,
    #[serde(default)]
    description: String,
    /// Params default to required; `"required": false` marks one optional.
    #[serde(default = "default_true")]
    required: bool,
}

fn default_true() -> bool {
    true
}

impl Harness {
    pub fn load(path: &std::path::Path) -> Result<Harness, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|err| format!("cannot read harness {}: {err}", path.display()))?;
        let harness: Harness = serde_json::from_str(&text)
            .map_err(|err| format!("harness {}: {err}", path.display()))?;
        harness.validate(path)?;
        Ok(harness)
    }

    fn validate(&self, path: &std::path::Path) -> Result<(), String> {
        for t in &self.tools {
            if t.name.trim().is_empty() {
                return Err(format!("harness {}: tool with empty name", path.display()));
            }
            if t.command.trim().is_empty() {
                return Err(format!(
                    "harness {}: tool '{}' has an empty command",
                    path.display(),
                    t.name
                ));
            }
            if t.params.iter().any(|p| p.name.trim().is_empty()) {
                return Err(format!(
                    "harness {}: tool '{}' has a param with an empty name",
                    path.display(),
                    t.name
                ));
            }
        }
        Ok(())
    }

    /// The harness tools as session [`ShellTool`]s.
    pub fn shell_tools(&self) -> Vec<ShellTool> {
        self.tools
            .iter()
            .map(|t| {
                let description = if t.description.trim().is_empty() {
                    format!("The {} tool.", t.name)
                } else {
                    t.description.clone()
                };
                ShellTool::new(
                    t.name.clone(),
                    description,
                    t.command.clone(),
                    t.params
                        .iter()
                        .map(|p| ToolParam {
                            name: p.name.clone(),
                            description: p.description.clone(),
                            required: p.required,
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Harness;

    fn parse(json: &str) -> Result<Harness, String> {
        let h: Harness = serde_json::from_str(json).map_err(|e| e.to_string())?;
        h.validate(std::path::Path::new("test.json"))?;
        Ok(h)
    }

    #[test]
    fn parses_the_full_shape() {
        let h = parse(
            r#"{
                "prompt": "You are a narrator.",
                "prethink": "Scene: $$scene$$",
                "vars": { "scene": ".story/scene.txt" },
                "tools": [
                    { "name": "set_scene", "description": "Set the scene.",
                      "params": [ { "name": "scene" } ],
                      "command": "printf %s \"$scene\" > .story/scene.txt" },
                    { "name": "commit", "command": "echo '::end Ready.'" }
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(h.prompt.as_deref(), Some("You are a narrator."));
        assert_eq!(h.vars["scene"], ".story/scene.txt");
        let tools = h.shell_tools();
        assert_eq!(tools.len(), 2);
        // A no-param tool declares an empty properties object.
        let decl = crate::chat::engine::tool_declaration(&tools[1]);
        assert_eq!(decl["function"]["name"], "commit");
        assert!(
            decl["function"]["parameters"]["properties"]
                .as_object()
                .unwrap()
                .is_empty()
        );
        // A default-description tool synthesizes one; params default required.
        let decl = crate::chat::engine::tool_declaration(&tools[0]);
        assert_eq!(decl["function"]["parameters"]["required"][0], "scene");
    }

    #[test]
    fn rejects_bad_shapes() {
        // Unknown fields are typos, not extensions.
        assert!(parse(r#"{ "promt": "x" }"#).is_err());
        // Empty command.
        assert!(parse(r#"{ "tools": [ { "name": "a", "command": " " } ] }"#).is_err());
        // Empty tool name.
        assert!(parse(r#"{ "tools": [ { "name": "", "command": "echo" } ] }"#).is_err());
        // Missing command entirely.
        assert!(parse(r#"{ "tools": [ { "name": "a" } ] }"#).is_err());
    }
}
