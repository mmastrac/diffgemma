//! Mini-harness definitions: a JSON file that configures a chat session as a
//! small tool-driven application (`chat --harness <file.json>`).
//!
//! A harness bundles a system prompt, an optional persistent prethink
//! template, file-backed variables, and shell-backed tools with declared
//! parameters. The pieces compose with the session machinery: tools steer via
//! `::end` / `::set` directive lines in their output, `$$var$$` templates
//! expand per turn, and a file-backed variable reads fresh from its file at
//! every use so tools can buffer multi-line state there. Tool commands run
//! from the harness's own scratch directory (see [`scratch_dir`]), relative
//! var paths anchor there too, and `HARNESS_DIR` in the command environment
//! points back at the harness file's directory for scripts shipped beside
//! it. A harness behaves the same from any invocation directory, and its
//! state never lands in the invoking project.
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
    /// Tool rounds per turn before the loop gives up (default 8). Coding
    /// harnesses with read/edit/verify cycles want more headroom.
    #[serde(default)]
    pub max_rounds: Option<usize>,
    /// Per-turn output budget. Tool results extend the model's own output,
    /// so tool-heavy harnesses exhaust the chat default and end the turn
    /// with an empty reply. An explicit CLI `--max-new-tokens` wins.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
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

/// The harness's scratch directory: tool commands run here and relative
/// file vars resolve here. A tmpdir, so state is disposable, but stable
/// across sessions until the OS cleans it. Keyed by file stem plus a hash
/// of the absolute path so same-named harnesses stay apart.
pub fn scratch_dir(path: &std::path::Path) -> std::path::PathBuf {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in abs.as_os_str().as_encoded_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let stem = abs
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("harness");
    std::env::temp_dir()
        .join("diffgemma-harness")
        .join(format!("{stem}-{:08x}", h & 0xffff_ffff))
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

    /// The harness tools as session [`ShellTool`]s, run from `scratch` with
    /// `harness_dir` exported as `HARNESS_DIR`, when given.
    pub fn shell_tools(
        &self,
        scratch: Option<&std::path::Path>,
        harness_dir: Option<&std::path::Path>,
    ) -> Vec<ShellTool> {
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
                .in_dirs(
                    scratch.map(std::path::Path::to_path_buf),
                    harness_dir.map(std::path::Path::to_path_buf),
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
                "max_new_tokens": 8192,
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
        assert_eq!(h.max_new_tokens, Some(8192));
        let tools = h.shell_tools(None, None);
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
    fn scratch_dirs_are_stable_and_distinct() {
        let a = super::scratch_dir(std::path::Path::new("/proj-a/web.json"));
        let b = super::scratch_dir(std::path::Path::new("/proj-b/web.json"));
        assert_eq!(
            a,
            super::scratch_dir(std::path::Path::new("/proj-a/web.json"))
        );
        assert_ne!(a, b);
        assert!(a.file_name().unwrap().to_str().unwrap().starts_with("web-"));
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
