//! Structured decisions: typed questions answered by ONE denoise forward.
//!
//! A chat request whose system message is a JSON question schema enters this
//! mode. The one user message is the state to judge. The reply is a JSON
//! object of per-question probability distributions. A request is exactly one
//! turn: the reply is read off the logits. It never enters the KV. There is
//! no text generation: the
//! answer template (`id: label` per question) is seeded into the canvas with
//! each label slot left as noise, and the label's distribution is read off
//! the model's logits at that slot after `steps` forwards. Every question is
//! scored in the same forward, and each slot sees the whole template
//! bidirectionally, so a later question conditions an earlier one.
//!
//! Labels are single tokens by construction: `yes`/`no` for a proposition,
//! `A`/`B`/… for a choice, `1`/`2`/… for ordered levels. The tokenizer is the
//! authority on that: [`Schema::resolve_template`] refuses a schema whose
//! labels do not land on exactly one canvas token each.
//!
//! Schema (system message), Jev-shaped:
//!
//! ```json
//! {"instructions": "optional global context",
//!  "questions": [
//!    {"id": "relevant", "type": "noul", "instructions": "Is this on topic?"},
//!    {"id": "bucket", "type": "choice", "instructions": "Which bucket?",
//!     "options": [{"name": "bug", "description": "defect"}, {"name": "feature"}]},
//!    {"id": "depth", "type": "score", "instructions": "How thorough?",
//!     "levels": ["Superficial", "Adequate", "Thorough"]}],
//!  "steps": 1, "hole": "noise", "climb": 0, "climb_mode": "joint", "samples": 4}
//! ```

use serde::Deserialize;
use serde_json::{Value, json};

use crate::tokenizer::Tokenizer;

/// Choices ride on letters. Past this many, a label would need two tokens.
const MAX_LETTER_LABELS: usize = 26;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A proposition: probability it holds (`yes` / `no`).
    Noul,
    /// One of N named options.
    Choice,
    /// One of N ordered levels, also reported as an expected level.
    Score,
}

/// What fills a label slot in the seeded canvas before the forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hole {
    /// A uniform-random token id, the denoiser's training-time "unknown".
    #[default]
    Noise,
    /// The pad token.
    Pad,
    /// The question's first label. Biased toward that label; A/B use only.
    Label,
}

#[derive(Debug, Clone)]
pub struct Choice {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    pub instructions: String,
    /// The user-facing alternatives, in schema order. `yes`/`no` for a noul.
    pub choices: Vec<Choice>,
    /// One single-token label per choice, same order.
    pub labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub instructions: Option<String>,
    pub questions: Vec<Question>,
    /// Denoise forwards before the logits are read. 1 is the one-shot read;
    /// 2 adds a self-conditioned pass.
    pub steps: usize,
    pub hole: Hole,
    /// Denoise canvas width to run, rounded up to a 64-row multiple that
    /// holds the template. `None` picks the smallest such width: the answer
    /// is a fixed number of single-token labels, so it always fits, and a
    /// 64-row forward costs half a 256-row one. Read 32 times per width,
    /// three tickets: unambiguous answers identical at 1.00, a borderline
    /// one within 1.3 standard errors of its 256-row value.
    pub active: Option<usize>,
    /// Hill-climb rounds after the first read: each round writes every
    /// slot's top label into the canvas and re-reads from step 0, stopping
    /// early when a round repeats the last one's labels. 0 is one read.
    pub climb: usize,
    /// Climb rounds re-read each slot in its own forward, with the other
    /// slots filled and its own slot back at its seed filler.
    pub leave_one_out: bool,
    /// Independent reads with different hole noise, which are averaged. A
    /// single read conditions on one noise token in the slot and is sharper
    /// than the marginal over the noise. The mean over reads estimates that
    /// marginal. The prompt is prefilled once, so each extra sample costs
    /// one forward per round.
    pub samples: usize,
}

/// The answer template as canvas tokens, plus where each question's label
/// sits and which token each of its labels is.
#[derive(Debug, Clone)]
pub struct Template {
    pub ids: Vec<u32>,
    pub slots: Vec<Slot>,
}

#[derive(Debug, Clone)]
pub struct Slot {
    pub pos: usize,
    /// Token id per label, in `Question::labels` order.
    pub label_ids: Vec<u32>,
}

/// One scored slot: raw (softcapped, untempered) logits for its label tokens
/// plus the row's own statistics, which tell a confident label from a row
/// whose mass went to some other token.
#[derive(Debug, Clone)]
pub struct SlotScore {
    pub label_logits: Vec<f32>,
    pub argmax: u32,
    pub entropy: f32,
    pub row_logsumexp: f32,
}

#[derive(Deserialize)]
struct SchemaJson {
    #[serde(default)]
    instructions: Option<String>,
    questions: Vec<QuestionJson>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    hole: Option<String>,
    #[serde(default)]
    active: Option<usize>,
    #[serde(default)]
    climb: Option<usize>,
    #[serde(default)]
    climb_mode: Option<String>,
    #[serde(default)]
    samples: Option<usize>,
}

#[derive(Deserialize)]
struct QuestionJson {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    options: Option<Vec<OptionJson>>,
    #[serde(default)]
    levels: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OptionJson {
    Name(String),
    Full {
        name: String,
        #[serde(default)]
        description: Option<String>,
    },
}

/// The mode heuristic: the first message is a system/developer message whose
/// whole text is a JSON object with a `questions` array. `None` means an
/// ordinary chat request; `Some(Err)` means it looked structured but the
/// schema is unusable, which the caller reports instead of falling back.
pub fn detect(messages: &[Value]) -> Option<Result<Schema, String>> {
    let first = messages.first()?;
    let role = first.get("role").and_then(Value::as_str)?;
    if !matches!(role, "system" | "developer") {
        return None;
    }
    let text = crate::tools::message_text(first);
    let text = text.trim();
    if !text.starts_with('{') {
        return None;
    }
    let value: Value = serde_json::from_str(text).ok()?;
    value.get("questions")?.as_array()?;
    Some(Schema::from_value(&value))
}

impl Schema {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let raw: SchemaJson =
            serde_json::from_value(value.clone()).map_err(|e| format!("schema: {e}"))?;
        if raw.questions.is_empty() {
            return Err("schema: no questions".into());
        }
        let mut questions = Vec::with_capacity(raw.questions.len());
        for q in raw.questions {
            questions.push(Question::from_json(q)?);
        }
        let mut seen = std::collections::HashSet::new();
        for q in &questions {
            if !seen.insert(q.id.as_str()) {
                return Err(format!("schema: duplicate question id {:?}", q.id));
            }
        }
        let hole = match raw.hole.as_deref() {
            None | Some("noise") => Hole::Noise,
            Some("pad") => Hole::Pad,
            Some("label") => Hole::Label,
            Some(other) => return Err(format!("schema: unknown hole {other:?}")),
        };
        Ok(Self {
            instructions: raw.instructions,
            questions,
            steps: raw.steps.unwrap_or(1).clamp(1, 8),
            hole,
            active: raw.active,
            climb: raw.climb.unwrap_or(0).min(16),
            leave_one_out: match raw.climb_mode.as_deref() {
                None | Some("joint") => false,
                Some("loo") => true,
                Some(other) => return Err(format!("schema: unknown climb_mode {other:?}")),
            },
            samples: raw.samples.unwrap_or(4).clamp(1, 32),
        })
    }

    /// The system prompt: every question with its labelled alternatives and
    /// the one-line-per-question reply format the template then pins.
    pub fn system_text(&self) -> String {
        let mut s = String::from(
            "Answer a fixed set of questions about the state the user provides. \
             Each question lists its allowed answers; reply with exactly one \
             label per question.\n",
        );
        if let Some(extra) = self.instructions.as_deref().map(str::trim)
            && !extra.is_empty()
        {
            s.push('\n');
            s.push_str(extra);
            s.push('\n');
        }
        for q in &self.questions {
            s.push('\n');
            s.push_str(&format!("Question {}: {}\n", q.id, q.instructions.trim()));
            for (choice, label) in q.choices.iter().zip(&q.labels) {
                match (q.kind, choice.description.as_deref()) {
                    (Kind::Noul, _) => s.push_str(&format!("  {label}\n")),
                    (_, Some(d)) if !d.trim().is_empty() => {
                        s.push_str(&format!("  {label}: {} ({})\n", choice.name, d.trim()))
                    }
                    _ => s.push_str(&format!("  {label}: {}\n", choice.name)),
                }
            }
        }
        s.push_str(
            "\nReply with one line per question, in this order, formatted as \"id: label\".",
        );
        s
    }

    /// The reply the model is held to: one `id: label` line per question.
    /// `labels[i]` indexes `questions[i].labels`.
    pub fn answer_text(&self, labels: &[usize]) -> String {
        self.questions
            .iter()
            .zip(labels)
            .map(|(q, &l)| format!("{}: {}", q.id, q.labels[l]))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Tokenize the answer template and locate every label slot. Each label
    /// must change exactly one token of the template, and all of a question's
    /// labels must land on the same position: that is what makes the slot's
    /// logit row a distribution over the question's answers.
    pub fn resolve_template(&self, tok: &Tokenizer) -> Result<Template, String> {
        let base_labels = vec![0usize; self.questions.len()];
        let base = tok.encode(&self.answer_text(&base_labels), false);
        let mut slots = Vec::with_capacity(self.questions.len());
        for (qi, q) in self.questions.iter().enumerate() {
            let mut pos: Option<usize> = None;
            let mut label_ids = vec![0u32; q.labels.len()];
            for (li, label) in q.labels.iter().enumerate().skip(1) {
                let mut labels = base_labels.clone();
                labels[qi] = li;
                let enc = tok.encode(&self.answer_text(&labels), false);
                if enc.len() != base.len() {
                    return Err(format!(
                        "question {:?}: label {label:?} is not a single token",
                        q.id
                    ));
                }
                let diffs: Vec<usize> = (0..enc.len()).filter(|&i| enc[i] != base[i]).collect();
                let [p] = diffs.as_slice() else {
                    return Err(format!(
                        "question {:?}: label {label:?} changes {} template tokens, expected 1",
                        q.id,
                        diffs.len()
                    ));
                };
                if pos.is_some_and(|prev| prev != *p) {
                    return Err(format!(
                        "question {:?}: labels do not share one template slot",
                        q.id
                    ));
                }
                pos = Some(*p);
                label_ids[li] = enc[*p];
            }
            let pos = pos.ok_or_else(|| format!("question {:?}: needs two labels", q.id))?;
            label_ids[0] = base[pos];
            let mut uniq = label_ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() != label_ids.len() {
                return Err(format!(
                    "question {:?}: two labels tokenize to the same id",
                    q.id
                ));
            }
            slots.push(Slot { pos, label_ids });
        }
        Ok(Template { ids: base, slots })
    }

    /// The seeded canvas: template, turn close, pad to `canvas_len`, with
    /// each label slot replaced per [`Hole`]. Noise draws come from a
    /// seed-keyed LCG so a request is reproducible at a fixed seed.
    pub fn build_canvas(
        &self,
        template: &Template,
        turn_close: u32,
        pad: u32,
        vocab: u32,
        canvas_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        if template.ids.len() + 1 > canvas_len {
            return Err(format!(
                "answer template is {} tokens; the canvas holds {}",
                template.ids.len(),
                canvas_len - 1
            ));
        }
        let mut canvas = template.ids.clone();
        canvas.push(turn_close);
        canvas.resize(canvas_len, pad);
        let mut rng = crate::sample::Rng::new(seed ^ 0x5354_5255_4354);
        for slot in &template.slots {
            canvas[slot.pos] = match self.hole {
                Hole::Noise => rng.next_u32() % vocab,
                Hole::Pad => pad,
                Hole::Label => template.ids[slot.pos],
            };
        }
        Ok(canvas)
    }

    /// Canvas width to dispatch: the schema's `active`, or the template plus
    /// its turn close, rounded up to a multiple of 64 and clamped to the
    /// canvas.
    pub fn active_width(&self, template: &Template, canvas_len: usize) -> usize {
        let need = template.ids.len() + 1;
        self.active
            .unwrap_or(need)
            .max(need)
            .div_ceil(64)
            .saturating_mul(64)
            .clamp(1, canvas_len)
    }

    /// The reply body. Probabilities are a softmax over the label logits at
    /// temperature 1, averaged over `samples` (each sample's last round), and
    /// `confidence` is the top label's probability with `stderr` its standard
    /// error over the samples (absent for one sample). `label_mass` is the
    /// share of the row's full-vocabulary mass the labels hold. It is low when
    /// the model did not read the slot as an answer. Per-slot diagnostics
    /// come from the first sample.
    pub fn answers_json(
        &self,
        template: &Template,
        samples: &[Vec<Vec<SlotScore>>],
        converged: bool,
        tok: &Tokenizer,
        timing: &Value,
    ) -> Value {
        let finals: Vec<&[SlotScore]> = samples
            .iter()
            .filter_map(|rounds| rounds.last().map(Vec::as_slice))
            .collect();
        let scores: &[SlotScore] = finals.first().copied().unwrap_or(&[]);
        let rounds: &[Vec<SlotScore>] = samples.first().map(Vec::as_slice).unwrap_or(&[]);
        let mut answers = serde_json::Map::new();
        let mut diag = serde_json::Map::new();
        for (qi, ((q, slot), s)) in self
            .questions
            .iter()
            .zip(&template.slots)
            .zip(scores)
            .enumerate()
        {
            let per_sample: Vec<Vec<f32>> = finals
                .iter()
                .filter_map(|f| f.get(qi))
                .map(|s| softmax(&s.label_logits))
                .collect();
            let n = per_sample.len().max(1) as f32;
            let probs: Vec<f32> = (0..q.labels.len())
                .map(|li| per_sample.iter().map(|p| p[li]).sum::<f32>() / n)
                .collect();
            let top = argmax(&probs);
            let agreement = per_sample.iter().filter(|p| argmax(p) == top).count() as f32 / n;
            let stderr = (per_sample.len() > 1).then(|| {
                let var = per_sample
                    .iter()
                    .map(|p| (p[top] - probs[top]).powi(2))
                    .sum::<f32>()
                    / (n - 1.0);
                (var / n).sqrt()
            });
            let mut dist = serde_json::Map::new();
            for (choice, p) in q.choices.iter().zip(&probs) {
                dist.insert(choice.name.clone(), json!(p));
            }
            let mut a = serde_json::Map::new();
            a.insert("type".into(), json!(kind_name(q.kind)));
            a.insert("label".into(), json!(q.labels[top]));
            match q.kind {
                Kind::Noul => {
                    a.insert("noul".into(), json!(probs[0]));
                }
                Kind::Choice => {
                    a.insert("choice".into(), json!(q.choices[top].name));
                }
                Kind::Score => {
                    let expected: f32 = probs
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (i as f32 + 1.0) * p)
                        .sum();
                    a.insert("score".into(), json!(expected));
                    a.insert("level".into(), json!(q.choices[top].name));
                }
            }
            a.insert("probabilities".into(), Value::Object(dist));
            a.insert("confidence".into(), json!(probs[top]));
            if let Some(se) = stderr {
                a.insert("stderr".into(), json!(se));
                a.insert("agreement".into(), json!(agreement));
            }
            answers.insert(q.id.clone(), Value::Object(a));

            let label_mass: f32 = s
                .label_logits
                .iter()
                .map(|&l| (l - s.row_logsumexp).exp())
                .sum();
            diag.insert(
                q.id.clone(),
                json!({
                    "argmax_token": tok.id_to_token(s.argmax).unwrap_or("?"),
                    "argmax_is_label": slot.label_ids.contains(&s.argmax),
                    "entropy": s.entropy,
                    "label_mass": label_mass,
                }),
            );
        }
        // One line per round: each question's top label and its probability,
        // so a climb's trajectory reads at a glance.
        let climb: Vec<Value> = rounds
            .iter()
            .map(|round| {
                let mut m = serde_json::Map::new();
                for (q, s) in self.questions.iter().zip(round) {
                    let probs = softmax(&s.label_logits);
                    let top = argmax(&probs);
                    m.insert(q.id.clone(), json!([q.labels[top], probs[top]]));
                }
                Value::Object(m)
            })
            .collect();
        // Each sample's top label, its probability and the row entropy per
        // question, so the spread reads at a glance beside the averaged answer.
        let sample_tops: Vec<Value> = finals
            .iter()
            .map(|f| {
                let mut m = serde_json::Map::new();
                for (q, s) in self.questions.iter().zip(f.iter()) {
                    let probs = softmax(&s.label_logits);
                    let top = argmax(&probs);
                    m.insert(q.id.clone(), json!([q.labels[top], probs[top], s.entropy]));
                }
                Value::Object(m)
            })
            .collect();
        json!({
            "answers": Value::Object(answers),
            "diagnostics": {
                "steps": self.steps,
                "hole": hole_name(self.hole),
                "samples": {"n": finals.len(), "tops": sample_tops},
                "climb": {
                    "mode": if self.leave_one_out { "loo" } else { "joint" },
                    "rounds": climb,
                    "converged": converged,
                },
                "timing": timing,
                "questions": Value::Object(diag),
            },
        })
    }
}

impl Question {
    fn from_json(q: QuestionJson) -> Result<Self, String> {
        if q.id.trim().is_empty() || q.id.contains(['\n', ':']) {
            return Err(format!(
                "question id {:?} must be non-empty, no ':' or newline",
                q.id
            ));
        }
        let kind = match q.kind.as_str() {
            "noul" | "bool" | "boolean" => Kind::Noul,
            "choice" => Kind::Choice,
            "score" => Kind::Score,
            other => return Err(format!("question {:?}: unknown type {other:?}", q.id)),
        };
        let choices: Vec<Choice> = match kind {
            Kind::Noul => vec![
                Choice {
                    name: "yes".into(),
                    description: None,
                },
                Choice {
                    name: "no".into(),
                    description: None,
                },
            ],
            Kind::Choice => q
                .options
                .ok_or_else(|| format!("question {:?}: choice needs options", q.id))?
                .into_iter()
                .map(|o| match o {
                    OptionJson::Name(name) => Choice {
                        name,
                        description: None,
                    },
                    OptionJson::Full { name, description } => Choice { name, description },
                })
                .collect(),
            Kind::Score => q
                .levels
                .ok_or_else(|| format!("question {:?}: score needs levels", q.id))?
                .into_iter()
                .map(|name| Choice {
                    name,
                    description: None,
                })
                .collect(),
        };
        if choices.len() < 2 {
            return Err(format!(
                "question {:?}: needs at least two alternatives",
                q.id
            ));
        }
        if choices.len() > MAX_LETTER_LABELS {
            return Err(format!(
                "question {:?}: at most {MAX_LETTER_LABELS} alternatives",
                q.id
            ));
        }
        let labels: Vec<String> = match kind {
            Kind::Noul => vec!["yes".into(), "no".into()],
            Kind::Score if choices.len() <= 9 => {
                (1..=choices.len()).map(|i| i.to_string()).collect()
            }
            _ => (0..choices.len())
                .map(|i| char::from(b'A' + i as u8).to_string())
                .collect(),
        };
        Ok(Self {
            id: q.id,
            kind,
            instructions: q.instructions.unwrap_or_default(),
            choices,
            labels,
        })
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Noul => "noul",
        Kind::Choice => "choice",
        Kind::Score => "score",
    }
}

fn hole_name(h: Hole) -> &'static str {
    match h {
        Hole::Noise => "noise",
        Hole::Pad => "pad",
        Hole::Label => "label",
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(0, |best, (i, &x)| if x > v[best] { i } else { best })
}

/// The user message must be JSON: the state. Returned as the client wrote it.
pub fn state_text(message: &Value) -> Result<String, String> {
    let text = crate::tools::message_text(message);
    let text = text.trim();
    serde_json::from_str::<Value>(text).map_err(|e| format!("state must be JSON: {e}"))?;
    Ok(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Value {
        json!({
            "instructions": "The state is a support ticket.",
            "questions": [
                {"id": "relevant", "type": "noul", "instructions": "Is this a product question?"},
                {"id": "bucket", "type": "choice", "instructions": "Which bucket?",
                 "options": [{"name": "bug", "description": "Defect"}, {"name": "feature"}, "billing"]},
                {"id": "depth", "type": "score", "instructions": "How thorough?",
                 "levels": ["Superficial", "Adequate", "Thorough"]}
            ]
        })
    }

    #[test]
    fn detect_requires_system_json_with_questions() {
        let msgs = vec![json!({"role": "system", "content": example().to_string()})];
        assert!(matches!(detect(&msgs), Some(Ok(_))));
        let plain = vec![json!({"role": "system", "content": "You are helpful."})];
        assert!(detect(&plain).is_none());
        let other_json = vec![json!({"role": "system", "content": "{\"a\": 1}"})];
        assert!(detect(&other_json).is_none());
        let user_first = vec![json!({"role": "user", "content": example().to_string()})];
        assert!(detect(&user_first).is_none());
        let bad = vec![
            json!({"role": "system", "content": "{\"questions\": [{\"id\": \"x\", \"type\": \"zzz\"}]}"}),
        ];
        assert!(matches!(detect(&bad), Some(Err(_))));
    }

    #[test]
    fn labels_follow_kind() {
        let s = Schema::from_value(&example()).unwrap();
        assert_eq!(s.questions[0].labels, ["yes", "no"]);
        assert_eq!(s.questions[1].labels, ["A", "B", "C"]);
        assert_eq!(s.questions[1].choices[2].name, "billing");
        assert_eq!(s.questions[2].labels, ["1", "2", "3"]);
        assert_eq!(
            s.answer_text(&[1, 2, 0]),
            "relevant: no\nbucket: C\ndepth: 1"
        );
        assert!(s.system_text().contains("  B: feature\n"));
        assert!(s.system_text().contains("  A: bug (Defect)\n"));
    }

    #[test]
    fn template_slots_are_single_tokens_on_the_real_tokenizer() {
        let Some(dir) = crate::shaders::common::test_util::dgq_model_dir() else {
            eprintln!("skipping: no model dir");
            return;
        };
        let tok = Tokenizer::load(dir.join("tokenizer.json")).unwrap();
        let s = Schema::from_value(&example()).unwrap();
        let t = s.resolve_template(&tok).unwrap();
        assert_eq!(t.slots.len(), 3);
        for (q, slot) in s.questions.iter().zip(&t.slots) {
            assert_eq!(slot.label_ids.len(), q.labels.len());
            let mut ids = t.ids.clone();
            for (li, &id) in slot.label_ids.iter().enumerate() {
                ids[slot.pos] = id;
                assert!(
                    tok.decode(&ids)
                        .contains(&format!("{}: {}", q.id, q.labels[li])),
                    "slot decode for {} label {}",
                    q.id,
                    q.labels[li]
                );
            }
        }
        let canvas = s.build_canvas(&t, 106, 0, 262144, 256, 7).unwrap();
        assert_eq!(canvas.len(), 256);
        assert_eq!(canvas[t.ids.len()], 106);
        assert_eq!(s.active_width(&t, 256), 64);
        let mut wide = s.clone();
        wide.active = Some(256);
        assert_eq!(wide.active_width(&t, 256), 256);
        wide.active = Some(1);
        assert_eq!(wide.active_width(&t, 256), 64);
    }
}
