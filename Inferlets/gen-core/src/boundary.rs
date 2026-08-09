//! Cross-mode KV reuse: open the previous turn's canonical boundary, and leave
//! this turn's behind.
//!
//! Ported from `chat-apc/src/chat/prefix_cache.rs` (`plan` :714-765,
//! `finalize` :903-950) with four deliberate corrections:
//!
//! 1. **Names come from `ratio-names`**, so every mode computes the same name
//!    for the same history. chat-apc folds a per-crate `CARGO_PKG_VERSION` into
//!    its marker, which makes cross-inferlet reuse structurally impossible.
//! 2. **`open` verifies `full.starts_with(prefix)`.** The reference compares
//!    LENGTHS only (`prefix_cache.rs:781`), which admits a same-length but
//!    different prefix — i.e. another conversation's KV.
//! 3. **The canonical context is kept, never reconstructed.** Callers generate
//!    on a *fork* and save from the untouched canonical one, so cue, reasoning
//!    and branch tokens can never leak into a boundary.
//! 4. **The input boundary is checkpointed before generation.** A reasoning
//!    model can exhaust its output budget before producing visible content.
//!    The app then omits the empty assistant row from the next request, so the
//!    cue-free input prompt is still the deepest valid reusable boundary even
//!    though there is no assistant boundary to save.
//!
//! ## The canonical shape
//!
//! A boundary always holds `build_prompt_tokens(messages, CueMode::None)` —
//! no cue, no `/no_think`, no per-branch directive. Anything mode-specific is
//! appended *after* the fork. That is what lets a ToT turn and a chat turn
//! agree on a name.

use crate::schema::{ChatMessage, CueMode};
use crate::{GenError, prompt};
use inferlet::{Context, chat, model::Model};
use ratio_names::SnapshotName;
use ratio_wire::KvDiagnostics;
use serde::{Deserialize, Serialize};

/// How far back to look for a usable boundary when the exact one is absent.
/// Matches chat-apc's `LADDER_DEPTH`.
const LADDER_DEPTH: usize = 4;

/// Conversation identity plus caching policy, carried on every generative
/// request. Top-level rather than per-inferlet: ToT and Best-of-N could not
/// name a boundary at all without it.
///
/// `Default` is written out rather than derived so it agrees with what serde
/// produces for `{}`. A derived `Default` gives `enabled: false` while
/// `#[serde(default = "yes")]` gives `true`, so a test constructing
/// `BoundaryDirective { key: k, ..Default::default() }` would silently exercise
/// the reuse-disabled path and assert on code that never ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDirective {
    /// Opaque conversation id. Hashed into the name, never interpolated raw.
    pub key: String,
    /// Bumped by the client to deliberately orphan prior KV.
    #[serde(default)]
    pub compat: String,
    #[serde(default)]
    pub turn: u32,
    /// `false` disables reuse for this turn without changing the wire shape.
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}

impl Default for BoundaryDirective {
    fn default() -> Self {
        Self { key: String::new(), compat: String::new(), turn: 0, enabled: yes() }
    }
}

/// What actually happened, for diagnostics and tests.
///
/// `found` alone is NOT a hit signal. Scheduler eviction calls `suspend` and
/// leaves the name in place, so an evicted boundary **opens successfully and
/// replays the whole prefix** — logging identically to a real hit.
///
/// `reused_tokens` and friends are therefore NAME-level: `reused_tokens = 800`
/// means a boundary with that name covered 800 tokens, not that those tokens
/// were resident. The residency fields below are what make the difference
/// observable, and they come from the engine rather than being inferred:
/// `Context::open_with_report` returns pie's own accounting of how much of the
/// prefix it reused versus regenerated.
///
/// Read them together. `reused_tokens` large with `replayed_pages == Some(0)`
/// is a genuine hit. The same `reused_tokens` with a non-zero `replayed_pages`
/// is a replay wearing a hit's clothes — the tokens were re-prefilled and the
/// only thing reused was the name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenOutcome {
    pub found: bool,
    /// Tokens the opened boundary covered.
    pub reused_tokens: usize,
    /// Tokens we had to append on top.
    pub appended_tokens: usize,
    /// True when the exact boundary was used rather than a ladder rung.
    pub exact: bool,
    /// Committed pages the engine reported as already resident, reused as-is.
    ///
    /// `None` when nothing was opened (a cold start), so "no data" stays
    /// distinguishable from "reported zero".
    pub resident_pages: Option<u32>,
    /// Committed pages the engine had to REGENERATE by replay forward passes.
    ///
    /// This is the field that falsifies a reuse claim: non-zero means the open
    /// paid a prefill that `reused_tokens` alone would have hidden.
    pub replayed_pages: Option<u32>,
    /// Recurrent state was replayed. Tracked on a path independent of KV pages,
    /// so this can be true while `replayed_pages` is `Some(0)` — which is the
    /// normal shape for a hybrid/linear-attention model.
    pub rs_replayed: bool,
}

impl OpenOutcome {
    pub fn cold(total: usize) -> Self {
        Self {
            found: false,
            reused_tokens: 0,
            appended_tokens: total,
            exact: false,
            resident_pages: None,
            replayed_pages: None,
            rs_replayed: false,
        }
    }

    /// True only when the engine confirmed the reused prefix was resident.
    ///
    /// A `found` boundary with `replayed_pages > 0` is NOT a hit: the name
    /// resolved, but the KV behind it was rebuilt. Returns `false` when the
    /// engine reported nothing, so an unknown never reads as a success.
    pub fn is_resident_hit(&self) -> bool {
        self.found && matches!(self.replayed_pages, Some(0)) && !self.rs_replayed
    }

    /// Project onto the wire type every result carries.
    ///
    /// `OpenOutcome` stays the richer host-side record — `appended_tokens` and
    /// `exact` drive sizing decisions and never reach the client. This is the
    /// one place the narrowing happens, so chat, ToT and Best-of-N cannot
    /// disagree about how an engine report becomes a wire field.
    pub fn kv(&self) -> KvDiagnostics {
        KvDiagnostics {
            boundary_found: self.found,
            reused_tokens: self.reused_tokens as u32,
            resident_pages: self.resident_pages,
            replayed_pages: self.replayed_pages,
            rs_replayed: self.rs_replayed,
        }
    }
}

/// A context parked at exactly the canonical prompt, plus the tokens it holds.
///
/// Generate on a **fork** of `ctx`; keep this one untouched so the exit save
/// cannot pick up cue or generation tokens.
pub struct Canonical {
    pub ctx: Context,
    pub tokens: Vec<u32>,
    pub outcome: OpenOutcome,
    /// Whether the exact cue-free input boundary is already durable.
    ///
    /// This is saved by [`open_canonical`] before any mode-specific work. It is
    /// load-bearing for reasoning-only turns: when `content == ""`, the app
    /// drops the empty assistant row, and the next request falls back to this
    /// prompt boundary rather than re-prefilling the whole conversation.
    pub prompt_saved: bool,
}

/// Candidate boundaries, longest first: the exact prompt, then progressively
/// shorter message prefixes.
fn ladder(
    model: &Model,
    directive: &BoundaryDirective,
    model_id: &str,
    messages: &[ChatMessage],
) -> Vec<(Vec<u32>, SnapshotName)> {
    let mut out = Vec::new();
    let deepest = messages.len();
    let shallowest = deepest.saturating_sub(LADDER_DEPTH);
    for cut in (shallowest..=deepest).rev() {
        let Ok(tokens) = prompt::build_prompt_tokens(model, &messages[..cut], CueMode::None) else {
            continue;
        };
        if tokens.is_empty() {
            continue;
        }
        let name = SnapshotName::conv(&directive.key, &directive.compat, model_id, &tokens);
        out.push((tokens, name));
    }
    out
}

/// Entry half: a context holding the canonical prompt, reusing a saved
/// boundary when one is available.
///
/// `model_id` MUST be the resolved, membership-checked engine model id — the
/// same string every mode uses. It is hashed into the name, so a mode that
/// trims, case-folds or default-fills it computes a different name for the same
/// history and orphans the boundary silently, with no error anywhere.
///
/// Returns FLUSHED. The flush is here rather than at the call site because
/// `Context::fork` clones the unflushed buffer: forking first makes both the
/// fork and the later `save_boundary` fork replay the same prefill. Flushing
/// once up front materializes the pages, both forks inherit them, and
/// `save_boundary`'s own flush degrades to a no-op. Token sequences — and so
/// the digest — are unchanged either way; only the work is.
///
/// Opens via `open_with_report` rather than `open`, so the returned
/// [`OpenOutcome`] carries the engine's own account of how much of the prefix
/// was resident versus replayed. Eviction suspends rather than deletes, so a
/// plain `open` succeeds on an evicted boundary and silently re-prefills it;
/// the report is the only way to tell that from a hit.
pub async fn open_canonical(
    model: &Model,
    model_id: &str,
    directive: &BoundaryDirective,
    messages: &[ChatMessage],
) -> Result<Canonical, GenError> {
    let canonical = prompt::build_prompt_tokens(model, messages, CueMode::None)
        .map_err(|(c, m)| GenError::new(c, m))?;

    if directive.enabled && !directive.key.is_empty() {
        for (prefix, name) in ladder(model, directive, model_id, messages) {
            // `open_with_report`, not `open`: the report is the only way to
            // learn whether this boundary was actually resident or was found by
            // name and silently rebuilt.
            let Ok((mut ctx, report)) = Context::open_with_report(model, name.as_str()) else {
                continue;
            };
            // THE CHECK the reference omits. A length-only comparison would
            // accept a same-length prefix from another conversation and serve
            // its KV as ours.
            if !canonical.starts_with(&prefix) {
                continue;
            }
            let suffix = &canonical[prefix.len()..];
            if !suffix.is_empty() {
                ctx.append(suffix);
            }
            ctx.flush()
                .await
                .map_err(|e| GenError::new(crate::classify_engine_error(&e), e))?;
            // Promote a ladder hit to the exact input boundary before any
            // mode-specific fork is made. This is safe even if generation
            // later fails: the snapshot represents only client-sent history,
            // with no cue, reasoning, branch directive or generated token.
            let prompt_saved =
                save_prompt_checkpoint(&ctx, directive, model_id, &canonical);
            return Ok(Canonical {
                ctx,
                outcome: OpenOutcome {
                    found: true,
                    reused_tokens: prefix.len(),
                    appended_tokens: suffix.len(),
                    exact: prefix.len() == canonical.len(),
                    resident_pages: Some(report.resident_prefix_pages),
                    replayed_pages: Some(report.replayed_pages),
                    rs_replayed: report.rs_replayed,
                },
                tokens: canonical,
                prompt_saved,
            });
        }
    }

    let mut ctx = Context::new(model).map_err(|e| GenError::new("context_failed", e))?;
    ctx.append(&canonical);
    ctx.flush()
        .await
        .map_err(|e| GenError::new(crate::classify_engine_error(&e), e))?;
    let prompt_saved = save_prompt_checkpoint(&ctx, directive, model_id, &canonical);
    Ok(Canonical {
        outcome: OpenOutcome::cold(canonical.len()),
        tokens: canonical,
        ctx,
        prompt_saved,
    })
}

/// Save the exact cue-free input boundary, best-effort.
///
/// This deliberately happens in the entry half rather than waiting for a
/// visible answer. A reasoning model may return `finish_reason=length` with all
/// output in `reasoning_content` and `content=""`; the app omits that empty
/// assistant row from history, so this prompt is exactly the boundary the next
/// mode can reuse. Saving the live generation context instead would be wrong:
/// it contains hidden reasoning the app never resends.
fn save_prompt_checkpoint(
    ctx: &Context,
    directive: &BoundaryDirective,
    model_id: &str,
    tokens: &[u32],
) -> bool {
    if !directive.enabled || directive.key.is_empty() || tokens.is_empty() {
        return false;
    }
    save_one(
        ctx,
        &SnapshotName::conv(&directive.key, &directive.compat, model_id, tokens),
    )
}

/// Exit half: leave the visible-answer boundary for the next turn, whatever
/// mode runs it. The prompt boundary was already attempted by
/// [`open_canonical`] and is retried here if that first save failed.
///
/// `canonical` must be the untouched context from [`open_canonical`]. It is
/// forked, not consumed. `answer=None` means the turn produced no visible
/// assistant content; in that case only the prompt boundary is valid because
/// the app excludes the empty assistant row from future request history.
///
/// ORDERING: callers must complete this **before** emitting their terminal
/// event. A client that sees the terminal frame may issue the next request
/// immediately and race the save.
pub async fn save_boundary(
    canonical: &Canonical,
    model: &Model,
    model_id: &str,
    directive: &BoundaryDirective,
    answer: Option<&str>,
) -> Result<SaveOutcome, GenError> {
    if !directive.enabled || directive.key.is_empty() {
        return Ok(SaveOutcome::disabled());
    }

    // `open_canonical` normally landed this before generation. Retry directly
    // from the still-untouched, already-flushed context if the entry save
    // failed. No fork is needed until there is visible assistant text to add.
    let prompt_saved = canonical.prompt_saved
        || save_prompt_checkpoint(&canonical.ctx, directive, model_id, &canonical.tokens);

    let Some(answer) = answer.filter(|s| !s.is_empty()) else {
        return Ok(SaveOutcome { prompt_saved, full_saved: false });
    };

    let mut snap = canonical
        .ctx
        .fork()
        .map_err(|e| GenError::new("boundary_fork_failed", e))?;

    // The generated boundary: what the NEXT turn's history will hash to, given
    // the client persists and resends exactly this visible answer.
    let assistant = chat::assistant(model, answer);
    let mut full = canonical.tokens.clone();
    full.extend_from_slice(&assistant);
    snap.append(&assistant);
    snap.flush()
        .await
        .map_err(|e| GenError::new("boundary_flush_failed", e))?;
    let full_saved = save_one(
        &snap,
        &SnapshotName::conv(&directive.key, &directive.compat, model_id, &full),
    );
    Ok(SaveOutcome { prompt_saved, full_saved })
}

/// Content-addressed names make "already exists" success by construction: the
/// same name can only have been produced by the same tokens.
///
/// Returns whether the name is now present. Callers that treat the save as an
/// optimization ignore it; callers that are about to do something IRREVERSIBLE
/// on the strength of it must not.
fn save_one(ctx: &Context, name: &SnapshotName) -> bool {
    match ctx.save(name.as_str()) {
        Ok(()) => true,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("already exists") {
                return true;
            }
            // Non-fatal for chat/ToT: a failed boundary save costs the next
            // turn a re-prefill, it does not corrupt this one.
            eprintln!("[gen-core] boundary save failed for {}: {msg}", name.as_str());
            false
        }
    }
}

/// Which of the two boundaries actually landed.
///
/// This type exists because `save_boundary`'s `Result` is NOT a report of
/// whether anything was written. It returns `Err` only for fork/flush faults;
/// every `ctx.save` error is swallowed by `save_one`, and the whole function
/// short-circuits to `Ok(())` when reuse is disabled or the key is empty. So
/// `save_boundary(...).await?` succeeding is consistent with zero names having
/// been written.
///
/// For chat and ToT that is the right shape — the save is a cache optimization
/// and a failure costs only a re-prefill. Best-of-N's commit is different: it
/// DELETES the round's candidate snapshots on the strength of the save, and
/// those snapshots are the only recovery state. Gating that on a `Result` that
/// cannot report failure would free the KV after writing nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SaveOutcome {
    /// The prompt-only boundary (this turn's client-sent history, no cue and no
    /// generated reasoning). This may be true when `full_saved` is false.
    pub prompt_saved: bool,
    /// The generated boundary — history + assistant(answer). THIS is the one
    /// the next turn normally hits. False is expected for a reasoning-only turn
    /// whose visible answer is empty; an irreversible follow-up action such as
    /// Best-of-N release must still require it.
    pub full_saved: bool,
}

impl SaveOutcome {
    /// Nothing was attempted: reuse is off, or the directive carries no key.
    pub fn disabled() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_defaults_to_enabled() {
        let d: BoundaryDirective = serde_json::from_value(serde_json::json!({"key": "c1"})).unwrap();
        assert!(d.enabled, "omitting `enabled` must not silently disable reuse");
        assert_eq!(d.compat, "");
    }

    #[test]
    fn directive_can_disable_without_changing_shape() {
        let d: BoundaryDirective =
            serde_json::from_value(serde_json::json!({"key": "c1", "enabled": false})).unwrap();
        assert!(!d.enabled);
    }

    #[test]
    fn cold_outcome_reports_everything_as_appended() {
        let o = OpenOutcome::cold(120);
        assert!(!o.found && !o.exact);
        assert_eq!((o.reused_tokens, o.appended_tokens), (0, 120));
    }

    /// The guard that distinguishes reuse from serving another conversation's
    /// KV. Same length, different content, must be rejected.
    #[test]
    fn same_length_different_prefix_is_not_a_prefix() {
        let canonical = vec![1u32, 2, 3, 4];
        let impostor = vec![9u32, 9, 9];
        assert!(!canonical.starts_with(&impostor));
        assert!(canonical.starts_with(&[1u32, 2, 3]));
    }

    #[test]
    fn prompt_only_save_is_a_successful_partial_checkpoint() {
        // A reasoning-only turn has no assistant boundary, but the cue-free
        // input prompt is still reusable by the next mode. Do not collapse
        // that state into `disabled()` or pretend a full boundary landed.
        let outcome = SaveOutcome { prompt_saved: true, full_saved: false };
        assert!(outcome.prompt_saved);
        assert!(!outcome.full_saved);
    }
}

#[cfg(test)]
mod swift_wire_tests {
    use super::*;

    /// The app sends `ChatCacheDirective` under `boundary`. That type carries
    /// `policy` and `retention` too, which this side has no use for — so the
    /// decode must tolerate them rather than fail the request.
    #[test]
    fn accepts_the_real_swift_directive() {
        let from_app = serde_json::json!({
            "key": "0F1B2C3D-4E5F-6071-8293-A4B5C6D7E8F9",
            "turn": 4,
            "compat": "1",
            "policy": "auto",
            "retention": { "budget_pages": 1024, "reason": "kv_pressure" }
        });
        let d: BoundaryDirective =
            serde_json::from_value(from_app).expect("must decode the app's directive");
        assert_eq!(d.key, "0F1B2C3D-4E5F-6071-8293-A4B5C6D7E8F9");
        assert_eq!(d.compat, "1");
        assert_eq!(d.turn, 4);
        assert!(d.enabled, "the app sends no `enabled`; reuse must stay on");
    }

    /// A request with no directive must degrade to no-reuse, not error.
    #[test]
    fn absent_directive_disables_reuse_quietly() {
        let d = BoundaryDirective::default();
        assert!(d.key.is_empty(), "an empty key is the no-reuse signal");
    }
}
