/// Integration tests for PostgreSQL-compatible date/time functions added to PQL.
use pieskieo_core::{pql::{Executor, Parser, Value}, PieskieoDb};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

// Helper: run a COMPUTE query over a single-row collection and return the value of field `r`.
fn compute_one(ex: &Executor, db: &Arc<PieskieoDb>, expr: &str) -> Value {
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"x": 1}))
        .unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = {expr} SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    result
        .rows
        .into_iter()
        .next()
        .and_then(|row| row.data.get("r").cloned())
        .unwrap_or(Value::Null)
}

// ── CLOCK_TIMESTAMP / STATEMENT_TIMESTAMP / TRANSACTION_TIMESTAMP ────────────

#[test]
fn test_clock_timestamp_returns_integer() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "CLOCK_TIMESTAMP()");
    match v {
        Value::Integer(ts) => assert!(ts > 1_000_000_000, "expected unix epoch > 1B, got {ts}"),
        other => panic!("expected Integer, got {other:?}"),
    }
}

#[test]
fn test_statement_timestamp_returns_integer() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "STATEMENT_TIMESTAMP()");
    match v {
        Value::Integer(ts) => assert!(ts > 1_000_000_000, "expected unix epoch > 1B, got {ts}"),
        other => panic!("expected Integer, got {other:?}"),
    }
}

#[test]
fn test_transaction_timestamp_returns_integer() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "TRANSACTION_TIMESTAMP()");
    match v {
        Value::Integer(ts) => assert!(ts > 1_000_000_000, "expected unix epoch > 1B, got {ts}"),
        other => panic!("expected Integer, got {other:?}"),
    }
}

// ── LOCALTIME / LOCALTIMESTAMP / TIMEOFDAY ────────────────────────────────────

#[test]
fn test_localtime_returns_integer() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "LOCALTIME()");
    match v {
        Value::Integer(ts) => assert!(ts > 0, "expected positive timestamp"),
        other => panic!("expected Integer, got {other:?}"),
    }
}

#[test]
fn test_localtimestamp_returns_integer() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "LOCALTIMESTAMP()");
    match v {
        Value::Integer(ts) => assert!(ts > 1_000_000_000),
        other => panic!("expected Integer, got {other:?}"),
    }
}

#[test]
fn test_timeofday_returns_integer() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "TIMEOFDAY()");
    match v {
        Value::Integer(ts) => assert!(ts > 1_000_000_000),
        other => panic!("expected Integer, got {other:?}"),
    }
}

// ── CURRENT_TIME ──────────────────────────────────────────────────────────────

#[test]
fn test_current_time_within_day() {
    let (db, ex) = setup();
    let v = compute_one(&ex, &db, "CURRENT_TIME()");
    match v {
        Value::Integer(secs) => {
            assert!(secs >= 0, "seconds since midnight must be >= 0");
            assert!(secs < 86400, "seconds since midnight must be < 86400, got {secs}");
        }
        other => panic!("expected Integer, got {other:?}"),
    }
}

// ── TIMEZONE / AT_TIME_ZONE ───────────────────────────────────────────────────

#[test]
fn test_timezone_passthrough() {
    let (db, ex) = setup();
    // Insert a known timestamp (2024-01-01 00:00:00 UTC = 1704067200)
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"ts": 1704067200}))
        .unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMEZONE(\"UTC\", ts) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(1704067200)));
}

#[test]
fn test_at_time_zone_passthrough() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"ts": 1704067200}))
        .unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = AT_TIME_ZONE(ts, \"UTC\") SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(1704067200)));
}

// ── ISFINITE ──────────────────────────────────────────────────────────────────

#[test]
fn test_isfinite_integer() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"v": 42})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = ISFINITE(v) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(true)));
}

#[test]
fn test_isfinite_float() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"v": 3.14})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = ISFINITE(v) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(true)));
}

// ── TIMESTAMPDIFF ─────────────────────────────────────────────────────────────

#[test]
fn test_timestampdiff_seconds() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    // 100 seconds apart
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"t1": 1000, "t2": 1100})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMESTAMPDIFF(\"seconds\", t1, t2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(100)));
}

#[test]
fn test_timestampdiff_minutes() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    // 3600 seconds = 60 minutes
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"t1": 0, "t2": 3600})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMESTAMPDIFF(\"minutes\", t1, t2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(60)));
}

#[test]
fn test_timestampdiff_hours() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"t1": 0, "t2": 7200})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMESTAMPDIFF(\"hours\", t1, t2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(2)));
}

#[test]
fn test_timestampdiff_days() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    // 172800 seconds = 2 days
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"t1": 0, "t2": 172800})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMESTAMPDIFF(\"days\", t1, t2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(2)));
}

#[test]
fn test_timestampdiff_negative() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"t1": 200, "t2": 100})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMESTAMPDIFF(\"seconds\", t1, t2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(-100)));
}

// ── JUSTIFY_DAYS / JUSTIFY_HOURS / JUSTIFY_INTERVAL ──────────────────────────

#[test]
fn test_justify_days_passthrough() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"v": 86400})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = JUSTIFY_DAYS(v) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(86400)));
}

#[test]
fn test_justify_hours_passthrough() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"v": 3600})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = JUSTIFY_HOURS(v) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(3600)));
}

#[test]
fn test_justify_interval_passthrough() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"v": 7200})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = JUSTIFY_INTERVAL(v) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Integer(7200)));
}

// ── OVERLAPS ──────────────────────────────────────────────────────────────────

#[test]
fn test_overlaps_true() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    // [100, 300) overlaps [200, 400)
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"s1": 100, "e1": 300, "s2": 200, "e2": 400})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = OVERLAPS(s1, e1, s2, e2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(true)));
}

#[test]
fn test_overlaps_false() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    // [100, 200) does not overlap [200, 300)
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"s1": 100, "e1": 200, "s2": 200, "e2": 300})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = OVERLAPS(s1, e1, s2, e2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(false)));
}

#[test]
fn test_overlaps_contained() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    // [100, 500) contains [200, 300)
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(),
        serde_json::json!({"s1": 100, "e1": 500, "s2": 200, "e2": 300})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = OVERLAPS(s1, e1, s2, e2) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(true)));
}

// ── GENERATE_SERIES_TIMESTAMP / TIMESTAMP_SERIES ─────────────────────────────

#[test]
fn test_generate_series_timestamp_basic() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"x": 1})).unwrap();
    // 0 to 2*86400 step 86400 -> [0, 86400, 172800]
    let pql = format!("QUERY {cname} COMPUTE r = GENERATE_SERIES_TIMESTAMP(0, 172800, 86400) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(
        r,
        Some(Value::Array(vec![
            Value::Integer(0),
            Value::Integer(86400),
            Value::Integer(172800),
        ]))
    );
}

#[test]
fn test_timestamp_series_alias() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"x": 1})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = TIMESTAMP_SERIES(0, 3600, 1200) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(
        r,
        Some(Value::Array(vec![
            Value::Integer(0),
            Value::Integer(1200),
            Value::Integer(2400),
            Value::Integer(3600),
        ]))
    );
}

#[test]
fn test_generate_series_timestamp_empty_when_stop_lt_start() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"x": 1})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = GENERATE_SERIES_TIMESTAMP(1000, 500, 100) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Array(vec![])));
}

#[test]
fn test_generate_series_timestamp_capped_at_10000() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"x": 1})).unwrap();
    // step=1, start=0, stop=20000 -> capped at 10000 entries
    let pql = format!("QUERY {cname} COMPUTE r = GENERATE_SERIES_TIMESTAMP(0, 20000, 1) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    match r {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 10_000),
        other => panic!("expected Array, got {other:?}"),
    }
}

// ── Null-safety: empty args return Null/default ───────────────────────────────

#[test]
fn test_isfinite_null_input() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // missing field -> Value::Null -> ISFINITE returns Bool(false)
    let pql = format!("QUERY {cname} COMPUTE r = ISFINITE(missing_field) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(false)));
}

#[test]
fn test_overlaps_insufficient_args() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({"a": 1})).unwrap();
    // Only 2 args instead of 4 -> returns Bool(false)
    let pql = format!("QUERY {cname} COMPUTE r = OVERLAPS(a, a) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Bool(false)));
}

#[test]
fn test_justify_days_null_on_missing() {
    let (db, ex) = setup();
    let cname = format!("t_{}", uuid::Uuid::new_v4().simple());
    db.put_doc_ns(None, Some(&cname), Uuid::new_v4(), serde_json::json!({})).unwrap();
    let pql = format!("QUERY {cname} COMPUTE r = JUSTIFY_DAYS(missing_field) SELECT r;");
    let mut p = Parser::new(&pql);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    let r = result.rows.into_iter().next().and_then(|row| row.data.get("r").cloned());
    assert_eq!(r, Some(Value::Null));
}
