//! Next Token Arena: expose the model's immediate distribution, then fork the
//! prefix under the strongest candidate tokens and generate short previews.

use futures::future::join_all;
use gen_core::{
    GenError, classify_engine_error, resolve_sampler,
    schema::{self, ChatRequest},
};
use inferlet::{
    Context,
    inference::SlotOutput,
    model::Model,
    runtime,
    sample::{Distribution, Entropy, Sampler},
};
use ratio_wire::{Envelope, Event, EventSink, FinishReason, GenResult, RunInput};
use serde::Serialize;
use std::cell::{Cell, RefCell};

const TOP_K: u32 = 8;
const BRANCH_COUNT: usize = 5;
const BRANCH_TOKENS: usize = 12;

struct SessionSink {
    seq: Cell<u32>,
    signal: RefCell<Option<inferlet::types::FutureString>>,
}

impl SessionSink {
    fn new() -> Self {
        Self {
            seq: Cell::new(0),
            signal: RefCell::new(None),
        }
    }
}

impl EventSink for SessionSink {
    fn emit(&self, ev: Event) {
        let seq = self.seq.get();
        self.seq.set(seq + 1);
        inferlet::session::send(&Envelope::new(seq, &ev).to_line());
    }

    fn cancelled(&self) -> bool {
        let mut slot = self.signal.borrow_mut();
        let fut = slot.get_or_insert_with(inferlet::session::receive);
        fut.get().is_some()
    }
}

#[derive(Clone, Debug, Serialize)]
struct ArenaToken {
    id: u32,
    text: String,
    probability: f32,
}

#[derive(Clone, Debug, Serialize)]
struct BranchPreview {
    token: ArenaToken,
    preview: String,
    generated_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ArenaPayload {
    kind: &'static str,
    prompt: String,
    picked: ArenaToken,
    top: Vec<ArenaToken>,
    branches: Vec<BranchPreview>,
    entropy: f32,
    temperature: f32,
    top_p: f32,
}

fn fail(sink: &SessionSink, e: GenError) -> String {
    sink.emit(e.into_event());
    serde_json::to_string(&GenResult::default()).unwrap_or_else(|_| "{}".into())
}

fn last_user_prompt(req: &ChatRequest) -> Result<String, GenError> {
    let prompt = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content_str())
        .unwrap_or("")
        .to_string();
    if prompt.trim().is_empty() {
        return Err(GenError::new(
            "invalid_request",
            "Next Token Arena needs a non-empty user prompt",
        )
        .with_param("messages"));
    }
    Ok(prompt)
}

fn detokenize(model: &Model, tokens: &[u32]) -> String {
    model.tokenizer().decode(tokens).unwrap_or_default()
}

fn arena_token(model: &Model, token_id: u32, probability: f32) -> ArenaToken {
    let mut text = detokenize(model, &[token_id]);
    if text.is_empty() {
        text = format!("<token:{token_id}>");
    }
    ArenaToken {
        id: token_id,
        text,
        probability,
    }
}

async fn branch_preview(
    base: &Context,
    model: &Model,
    token: ArenaToken,
    sampler: Sampler,
) -> Result<BranchPreview, GenError> {
    let mut branch = base
        .fork()
        .map_err(|e| GenError::new(classify_engine_error(&e), e))?;
    branch.append(&[token.id]);
    let generated = branch
        .generate(sampler)
        .max_tokens(BRANCH_TOKENS)
        .collect_tokens()
        .await
        .map_err(|e| GenError::new(classify_engine_error(&e), e))?;

    let mut full = Vec::with_capacity(1 + generated.len());
    full.push(token.id);
    full.extend_from_slice(&generated);
    let mut preview = detokenize(model, &full);
    if preview.trim().is_empty() {
        preview = token.text.clone();
    }
    Ok(BranchPreview {
        token,
        preview,
        generated_tokens: 1 + generated.len() as u32,
    })
}

async fn run_arena(req: &ChatRequest, sink: &SessionSink) -> Result<GenResult, GenError> {
    let models = runtime::models();
    if !models.iter().any(|m| m == &req.model) {
        return Err(GenError::new(
            "model_not_found",
            format!("model {:?} is not served by this engine", req.model),
        ));
    }
    schema::validate_sampling(req).map_err(|(c, m)| GenError::new(c, m))?;

    let model = Model::load(&req.model).map_err(|e| GenError::new("model_load_failed", e))?;
    let prompt = last_user_prompt(req)?;
    let prompt_tokens = model.tokenizer().encode(&prompt);
    if prompt_tokens.is_empty() {
        return Err(GenError::new(
            "invalid_request",
            "prompt did not tokenize into any model tokens",
        )
        .with_param("messages"));
    }

    sink.emit(Event::Ready);
    if sink.cancelled() {
        sink.emit(Event::Finish {
            reason: FinishReason::Cancelled,
        });
        return Ok(GenResult {
            finish_reason: Some(FinishReason::Cancelled),
            ..GenResult::default()
        });
    }

    let temperature = req.temperature_or_default();
    let top_p = req.top_p_or_default();
    let sampler = resolve_sampler(temperature, top_p);
    let distribution_temperature = if temperature <= 0.0 { 1.0 } else { temperature };

    let mut ctx = Context::new(&model)
        .map_err(|e| GenError::new(classify_engine_error(&e), e))?;
    let last_index = prompt_tokens.len() as u32 - 1;
    let mut forward = ctx.forward();
    forward.input(&prompt_tokens);
    let sample_handle = forward.sample(&[last_index], sampler.clone());
    let dist_handle = forward.probe(
        last_index,
        Distribution {
            temperature: distribution_temperature,
            k: TOP_K,
        },
    );
    let entropy_handle = forward.probe(last_index, Entropy);
    let out = forward
        .execute()
        .await
        .map_err(|e| GenError::new(classify_engine_error(&e), e))?;

    let picked_id = out
        .token(sample_handle)
        .ok_or_else(|| GenError::new("context_failed", "sampler returned no token"))?;
    let (ids, probabilities) = out
        .distribution(dist_handle)
        .ok_or_else(|| GenError::new("context_failed", "distribution probe returned no data"))?;
    let entropy = out.entropy(entropy_handle).unwrap_or(0.0);
    if !out.raw().slots.iter().any(|s| matches!(s, SlotOutput::Token(_))) {
        return Err(GenError::new(
            "context_failed",
            "engine returned no sampled token for the prompt",
        ));
    }

    let top: Vec<ArenaToken> = ids
        .iter()
        .zip(probabilities.iter())
        .map(|(&id, &probability)| arena_token(&model, id, probability))
        .collect();
    let picked_probability = top
        .iter()
        .find(|candidate| candidate.id == picked_id)
        .map(|candidate| candidate.probability)
        .unwrap_or(0.0);
    let picked = arena_token(&model, picked_id, picked_probability);

    let branch_inputs: Vec<ArenaToken> = top.iter().take(BRANCH_COUNT).cloned().collect();
    let branch_results = join_all(
        branch_inputs
            .into_iter()
            .map(|token| branch_preview(&ctx, &model, token, sampler.clone())),
    )
    .await;
    let mut branches = Vec::new();
    for result in branch_results {
        branches.push(result?);
    }

    let completion_tokens = 1 + branches.iter().map(|b| b.generated_tokens).sum::<u32>();
    let payload = ArenaPayload {
        kind: "next_token_arena",
        prompt,
        picked,
        top,
        branches,
        entropy,
        temperature,
        top_p,
    };
    let content = serde_json::to_string(&payload)
        .map_err(|e| GenError::new("encode_failed", e.to_string()))?;
    sink.emit(Event::ContentDelta {
        text: content.clone(),
    });
    sink.emit(Event::Finish {
        reason: FinishReason::Stop,
    });
    sink.emit(Event::Usage {
        prompt_tokens: prompt_tokens.len() as u32,
        completion_tokens,
        context_window: (ctx.max_tokens() > 0).then_some(ctx.max_tokens()),
    });

    Ok(GenResult {
        content,
        reasoning: None,
        tool_calls: Vec::new(),
        finish_reason: Some(FinishReason::Stop),
        prompt_tokens: prompt_tokens.len() as u32,
        completion_tokens,
        context_window: (ctx.max_tokens() > 0).then_some(ctx.max_tokens()),
        boundary_found: false,
        reused_tokens: 0,
    })
}

#[inferlet::main]
async fn main(input: RunInput) -> inferlet::Result<String> {
    let sink = SessionSink::new();

    if input.v != ratio_wire::envelope::ENVELOPE_VERSION {
        return Ok(fail(
            &sink,
            GenError::new(
                "unsupported_envelope_version",
                format!("v={} not supported", input.v),
            ),
        ));
    }

    let req: ChatRequest = match serde_json::from_value(input.request) {
        Ok(r) => r,
        Err(e) => {
            return Ok(fail(
                &sink,
                GenError::new("invalid_request", e.to_string()),
            ));
        }
    };

    match run_arena(&req, &sink).await {
        Ok(result) => Ok(serde_json::to_string(&result).unwrap_or_else(|_| "{}".into())),
        Err(e) => Ok(fail(&sink, e)),
    }
}
