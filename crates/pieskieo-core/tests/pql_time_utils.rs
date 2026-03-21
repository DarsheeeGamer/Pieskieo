/// Integration tests for PQL time arithmetic and duration functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

// ── SECONDS_TO_HMS ────────────────────────────────────────────────────────────

#[test]
fn test_seconds_to_hms_3661() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 3661}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = SECONDS_TO_HMS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("01:01:01".to_string()))
    );
}

#[test]
fn test_seconds_to_hms_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = SECONDS_TO_HMS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("00:00:00".to_string()))
    );
}

#[test]
fn test_seconds_to_hms_86400() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 86400}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = SECONDS_TO_HMS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("24:00:00".to_string()))
    );
}

#[test]
fn test_sec_to_hms_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 7322}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = SEC_TO_HMS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("02:02:02".to_string()))
    );
}

// ── HMS_TO_SECONDS ────────────────────────────────────────────────────────────

#[test]
fn test_hms_to_seconds_010101() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "01:01:01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = HMS_TO_SECONDS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(3661)));
}

#[test]
fn test_hms_to_seconds_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "00:00:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = HMS_TO_SECONDS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(0)));
}

#[test]
fn test_hms_to_sec_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "02:30:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = HMS_TO_SEC(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(9000)));
}

#[test]
fn test_seconds_to_hms_round_trip() {
    // Two-step round-trip: 7384 -> HMS string -> seconds
    // Step 1: convert to HMS
    let (db1, ex1) = setup();
    db1.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 7384}),
    )
    .unwrap();
    let mut p1 = Parser::new("QUERY t COMPUTE nv = SECONDS_TO_HMS(tv) SELECT nv;");
    let r1 = ex1.execute(p1.parse().unwrap()).unwrap();
    let hms = match r1.rows[0].data.get("nv") {
        Some(Value::String(s)) => s.clone(),
        _ => panic!("expected string"),
    };
    // 7384 = 2h 3m 4s => "02:03:04"
    assert_eq!(hms, "02:03:04");
    // Step 2: convert back to seconds
    let (db2, ex2) = setup();
    db2.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": hms}),
    )
    .unwrap();
    let mut p2 = Parser::new("QUERY t COMPUTE nv = HMS_TO_SECONDS(tv) SELECT nv;");
    let r2 = ex2.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(r2.rows[0].data.get("nv"), Some(&Value::Integer(7384)));
}

// ── DURATION_PARTS ────────────────────────────────────────────────────────────

#[test]
fn test_duration_parts_3661_hours() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 3661}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DURATION_PARTS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::Object(obj)) = r.rows[0].data.get("nv") {
        assert_eq!(obj.get("hours"), Some(&Value::Integer(1)));
        assert_eq!(obj.get("minutes"), Some(&Value::Integer(1)));
        assert_eq!(obj.get("seconds"), Some(&Value::Integer(1)));
    } else {
        panic!("expected Object");
    }
}

#[test]
fn test_seconds_to_parts_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 3600}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = SECONDS_TO_PARTS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::Object(obj)) = r.rows[0].data.get("nv") {
        assert_eq!(obj.get("hours"), Some(&Value::Integer(1)));
        assert_eq!(obj.get("minutes"), Some(&Value::Integer(0)));
        assert_eq!(obj.get("seconds"), Some(&Value::Integer(0)));
    } else {
        panic!("expected Object");
    }
}

#[test]
fn test_duration_parts_days() {
    let (db, ex) = setup();
    // 86400 + 3661 = 90061 seconds => 1 day, 1 hour, 1 minute, 1 second
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 90061}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DURATION_PARTS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::Object(obj)) = r.rows[0].data.get("nv") {
        assert_eq!(obj.get("days"), Some(&Value::Integer(1)));
        assert_eq!(obj.get("hours"), Some(&Value::Integer(1)));
    } else {
        panic!("expected Object");
    }
}

// ── TIME_ADD_SECONDS ──────────────────────────────────────────────────────────

#[test]
fn test_time_add_seconds_plus_one_hour() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "01:00:00", "nv": 3600}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE tv = TIME_ADD_SECONDS(t1, nv) SELECT tv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("tv"),
        Some(&Value::String("02:00:00".to_string()))
    );
}

#[test]
fn test_time_add_seconds_overflow_hour() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "23:00:00", "nv": 7200}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE tv = TIME_ADD_SECONDS(t1, nv) SELECT tv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("tv"),
        Some(&Value::String("25:00:00".to_string()))
    );
}

#[test]
fn test_add_seconds_to_time_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "00:00:00", "nv": 90}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE tv = ADD_SECONDS_TO_TIME(t1, nv) SELECT tv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("tv"),
        Some(&Value::String("00:01:30".to_string()))
    );
}

// ── TIME_DIFF_SECONDS ─────────────────────────────────────────────────────────

#[test]
fn test_time_diff_seconds_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "01:00:00", "t2": "02:00:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = TIME_DIFF_SECONDS(t1, t2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(3600)));
}

#[test]
fn test_time_diff_seconds_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "02:00:00", "t2": "01:00:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = TIME_DIFF_SECONDS(t1, t2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(-3600)));
}

#[test]
fn test_time_subtract_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "00:30:00", "t2": "01:00:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = TIME_SUBTRACT(t1, t2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(1800)));
}

// ── IS_VALID_TIME ─────────────────────────────────────────────────────────────

#[test]
fn test_is_valid_time_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "12:30:45"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_VALID_TIME(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_time_invalid_minutes() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "12:60:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_VALID_TIME(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_valid_time_invalid_seconds() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "12:30:60"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_VALID_TIME(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_time_string_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "25:00:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_TIME_STRING(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // hours can exceed 23; only minutes and seconds are range-checked
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

// ── EPOCH_TO_DATE ─────────────────────────────────────────────────────────────

#[test]
fn test_epoch_to_date_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = EPOCH_TO_DATE(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("1970-01-01".to_string()))
    );
}

#[test]
fn test_epoch_to_date_one_day() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 86400}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = EPOCH_TO_DATE(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("1970-01-02".to_string()))
    );
}

#[test]
fn test_epoch_to_date_2021_01_01() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 1609459200i64}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = EPOCH_TO_DATE(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2021-01-01".to_string()))
    );
}

#[test]
fn test_unix_to_date_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = UNIX_TO_DATE_STR(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("1970-01-01".to_string()))
    );
}

// ── DATE_TO_EPOCH ─────────────────────────────────────────────────────────────

#[test]
fn test_date_to_epoch_1970_01_01() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "1970-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_TO_EPOCH(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(0)));
}

#[test]
fn test_date_to_epoch_1970_01_02() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "1970-01-02"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_TO_EPOCH(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(86400)));
}

#[test]
fn test_date_str_to_unix_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": "1970-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_STR_TO_UNIX(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(0)));
}

#[test]
fn test_epoch_to_date_and_back_round_trip() {
    // Two-step round-trip: epoch -> date string -> epoch
    // Step 1: epoch 86400 -> "1970-01-02"
    let (db1, ex1) = setup();
    db1.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 86400}),
    )
    .unwrap();
    let mut p1 = Parser::new("QUERY t COMPUTE nv = EPOCH_TO_DATE(tv) SELECT nv;");
    let r1 = ex1.execute(p1.parse().unwrap()).unwrap();
    let date_str = match r1.rows[0].data.get("nv") {
        Some(Value::String(s)) => s.clone(),
        _ => panic!("expected string"),
    };
    assert_eq!(date_str, "1970-01-02");
    // Step 2: "1970-01-02" -> 86400
    let (db2, ex2) = setup();
    db2.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": date_str}),
    )
    .unwrap();
    let mut p2 = Parser::new("QUERY t COMPUTE nv = DATE_TO_EPOCH(tv) SELECT nv;");
    let r2 = ex2.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(r2.rows[0].data.get("nv"), Some(&Value::Integer(86400)));
}

// ── DATETIME_DIFF_SECONDS ─────────────────────────────────────────────────────

#[test]
fn test_datetime_diff_seconds_one_day() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-01-01", "d2": "2024-01-02"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATETIME_DIFF_SECONDS(d1, d2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(86400)));
}

#[test]
fn test_datetime_diff_sec_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-01-01", "d2": "2024-01-02"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATETIME_DIFF_SEC(d1, d2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(86400)));
}

// ── TIME_OVERLAP ──────────────────────────────────────────────────────────────

#[test]
fn test_time_overlap_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = TIME_OVERLAP(0, 10, 5, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_time_overlap_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = TIME_OVERLAP(0, 5, 10, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_intervals_overlap_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = INTERVALS_OVERLAP(0, 10, 5, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

// ── OVERLAP_DURATION ──────────────────────────────────────────────────────────

#[test]
fn test_overlap_duration_five() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = OVERLAP_DURATION(0, 10, 5, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(5)));
}

#[test]
fn test_overlap_duration_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = OVERLAP_DURATION(0, 5, 10, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(0)));
}

#[test]
fn test_interval_overlap_secs_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = INTERVAL_OVERLAP_SECS(0, 10, 5, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(5)));
}

// ── CLAMP_TO_INTERVAL ─────────────────────────────────────────────────────────

#[test]
fn test_clamp_to_interval_above() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CLAMP_TO_INTERVAL(tv, 0.0, 3.0) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Float(3.0)));
}

#[test]
fn test_clamp_to_interval_below() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": -1.0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CLAMP_TO_INTERVAL(tv, 0.0, 10.0) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Float(0.0)));
}

#[test]
fn test_clip_to_interval_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CLIP_TO_INTERVAL(tv, 0.0, 3.0) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Float(3.0)));
}

// ── Additional edge-case tests ────────────────────────────────────────────────

#[test]
fn test_is_valid_time_not_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 12345}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_VALID_TIME(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_clamp_to_interval_within_range() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CLAMP_TO_INTERVAL(tv, 0.0, 10.0) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Float(5.0)));
}

#[test]
fn test_overlap_duration_full_containment() {
    let (db, ex) = setup();
    // [0, 20] fully contains [5, 15] => overlap is 10
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = OVERLAP_DURATION(0, 20, 5, 15) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(10)));
}

#[test]
fn test_time_diff_seconds_same_time() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "05:30:00", "t2": "05:30:00"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = TIME_DIFF_SECONDS(t1, t2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(0)));
}

#[test]
fn test_duration_parts_keys_present() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tv": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DURATION_PARTS(tv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::Object(obj)) = r.rows[0].data.get("nv") {
        assert!(obj.contains_key("days"));
        assert!(obj.contains_key("hours"));
        assert!(obj.contains_key("minutes"));
        assert!(obj.contains_key("seconds"));
    } else {
        panic!("expected Object with duration keys");
    }
}
