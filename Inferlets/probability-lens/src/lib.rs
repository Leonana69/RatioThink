//! Probability Lens: run the normal chat path while recording the probability
//! distribution and entropy at each visible generated token.

use gen_core::{GenError, TokenObservation, schema::ChatRequest};
use ratio_wire::{Envelope, Event, EventSink, GenResult, RunInput};
use serde::Serialize;
use std::cell::{Cell, RefCell};

const TOP_K: u32 = 5;

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
    fn emit(&self, event: Event) {
        let seq = self.seq.get();
        self.seq.set(seq + 1);
        inferlet::session::send(&Envelope::new(seq, &event).to_line());
    }

    fn cancelled(&self) -> bool {
        let mut slot = self.signal.borrow_mut();
        let future = slot.get_or_insert_with(inferlet::session::receive);
        future.get().is_some()
    }
}

/// Pass lifecycle, warnings, errors, and reasoning through while holding back
/// the ordinary content/terminal frames. The outer inferlet replaces them with
/// one structured lens payload followed by the original terminal metadata.
struct LensSink<'a> {
    session: &'a SessionSink,
}

impl EventSink for LensSink<'_> {
    fn emit(&self, event: Event) {
        if !matches!(
            event,
            Event::ContentDelta { .. } | Event::Finish { .. } | Event::Usage { .. }
        ) {
            self.session.emit(event);
        }
    }

    fn cancelled(&self) -> bool {
        self.session.cancelled()
    }
}

#[derive(Debug, Serialize)]
struct LensPayload {
    kind: &'static str,
    text: String,
    tokens: Vec<TokenObservation>,
    average_entropy: Option<f32>,
    temperature: f32,
    top_p: f32,
}

fn fail(sink: &SessionSink, error: GenError) -> String {
    sink.emit(error.into_event());
    serde_json::to_string(&GenResult::default()).unwrap_or_else(|_| "{}".into())
}

fn average_entropy(tokens: &[TokenObservation]) -> Option<f32> {
    let values: Vec<f32> = tokens
        .iter()
        .filter_map(|token| token.entropy.filter(|value| value.is_finite()))
        .collect();
    (!values.is_empty()).then(|| values.iter().sum::<f32>() / values.len() as f32)
}

async fn run_lens(req: &ChatRequest, sink: &SessionSink) -> Result<GenResult, GenError> {
    let observed =
        gen_core::run_chat_with_token_observations(req, &LensSink { session: sink }, TOP_K).await?;
    let mut result = observed.result;
    let payload = LensPayload {
        kind: "probability_lens",
        text: result.content.clone(),
        average_entropy: average_entropy(&observed.tokens),
        tokens: observed.tokens,
        temperature: req.temperature_or_default(),
        top_p: req.top_p_or_default(),
    };
    let content = serde_json::to_string(&payload)
        .map_err(|error| GenError::new("encode_failed", error.to_string()))?;

    sink.emit(Event::ContentDelta {
        text: content.clone(),
    });
    sink.emit(Event::Finish {
        reason: result
            .finish_reason
            .unwrap_or(ratio_wire::FinishReason::Error),
    });
    sink.emit(Event::Usage {
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        context_window: result.context_window,
    });
    result.content = content;
    Ok(result)
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

    let request: ChatRequest = match serde_json::from_value(input.request) {
        Ok(request) => request,
        Err(error) => {
            return Ok(fail(
                &sink,
                GenError::new("invalid_request", error.to_string()),
            ));
        }
    };

    match run_lens(&request, &sink).await {
        Ok(result) => Ok(serde_json::to_string(&result).unwrap_or_else(|_| "{}".into())),
        Err(error) => Ok(fail(&sink, error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_entropy_ignores_missing_values() {
        let token = |entropy| TokenObservation {
            id: 1,
            text: "x".into(),
            probability: Some(0.5),
            entropy,
            alternatives: Vec::new(),
        };
        assert_eq!(
            average_entropy(&[token(Some(1.0)), token(None), token(Some(3.0))]),
            Some(2.0)
        );
        assert_eq!(average_entropy(&[token(None)]), None);
    }
}
