//! `KvDiagnostics` is flattened into three carriers — `ratio_wire::GenResult`,
//! `tot_core::tree::TreeResult` and `bestofn_core::round::RoundResult` — and is
//! decoded by the gateway from an untyped envelope. These tests pin the two
//! properties that makes safe.
//!
//! The exact byte string below is what the three structs emitted BEFORE the
//! hoist, when each declared the five fields itself. It is checked in as a
//! regression anchor: `#[serde(flatten)]` must be shape-neutral, and nesting
//! the fields under a `kv` object would be a breaking wire change.

use ratio_wire::event::{GenResult, KvDiagnostics};

#[test]
fn flatten_is_wire_neutral() {
    let g = GenResult {
        content: "c".into(),
        prompt_tokens: 1,
        completion_tokens: 2,
        kv: KvDiagnostics {
            boundary_found: true,
            reused_tokens: 22,
            resident_pages: Some(3),
            replayed_pages: Some(0),
            rs_replayed: false,
        },
        ..Default::default()
    };
    assert_eq!(
        serde_json::to_string(&g).unwrap(),
        r#"{"content":"c","finish_reason":null,"prompt_tokens":1,"completion_tokens":2,"boundary_found":true,"reused_tokens":22,"resident_pages":3,"replayed_pages":0,"rs_replayed":false}"#,
        "the five KV fields must stay at the TOP level, in this order — \
         nesting them under `kv` is a breaking change for every consumer"
    );
}

#[test]
fn absent_is_not_zero() {
    // The distinction the whole residency effort turned on: the gateway renders
    // a missing value as -1, so `None` and `Some(0)` must not collapse.
    let measured_zero: KvDiagnostics =
        serde_json::from_str(r#"{"replayed_pages":0}"#).unwrap();
    assert_eq!(measured_zero.replayed_pages, Some(0));

    let reported_nothing: KvDiagnostics = serde_json::from_str("{}").unwrap();
    assert_eq!(reported_nothing.replayed_pages, None);
    assert_eq!(reported_nothing.resident_pages, None);
    // Everything defaults, so a guest predating the struct still decodes —
    // which is what lets the gateway `unwrap_or_default()` its envelope.
    assert_eq!(reported_nothing, KvDiagnostics::default());
}

#[test]
fn skips_absent_options_rather_than_emitting_null() {
    // A `null` would read as present-but-unknown downstream; the gateway probes
    // for the key itself.
    let v = serde_json::to_value(KvDiagnostics::default()).unwrap();
    assert!(v.get("resident_pages").is_none());
    assert!(v.get("replayed_pages").is_none());
    assert!(v.get("boundary_found").is_some(), "non-Option fields always ride");
}

/// The gateway decodes this out of an untyped envelope that also carries
/// `model`, `root`, `resume` and friends. Unknown siblings must be ignored,
/// not rejected.
#[test]
fn decodes_from_a_larger_envelope() {
    let envelope = r#"{
        "model": "m", "breadth": 3, "root": {"id": "root"},
        "boundary_found": true, "reused_tokens": 22,
        "resident_pages": 3, "replayed_pages": 0, "rs_replayed": false,
        "resume": {"kind": "warm"}
    }"#;
    let kv: KvDiagnostics = serde_json::from_str(envelope).unwrap();
    assert_eq!(kv.boundary_found, true);
    assert_eq!(kv.reused_tokens, 22);
    assert_eq!(kv.replayed_pages, Some(0));
}
