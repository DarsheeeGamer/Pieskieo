/// Integration tests for PQL date/time functions.
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

#[test]
fn test_date_trunc() {
    let (db, ex) = setup();
    // 2024-03-15 14:30:45 = Unix timestamp 1710510645
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ts": 1710510645_i64}),
    )
    .unwrap();

    let mut p = Parser::new("QUERY t COMPUTE d = DATE_TRUNC(\"day\", ts) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Truncate to day start
    match r.rows[0].data.get("d") {
        Some(Value::Integer(v)) => {
            // Should be divisible by 86400
            assert_eq!(
                v % 86400,
                0,
                "DATE_TRUNC('day') result should be divisible by 86400"
            );
        }
        other => panic!("Expected Integer, got {:?}", other),
    }
}

#[test]
fn test_date_part() {
    let (db, ex) = setup();
    // 2024-03-15 14:30:45
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ts": 1710510645_i64}),
    )
    .unwrap();

    let mut p = Parser::new(
        "QUERY t COMPUTE h = DATE_PART(\"hour\", ts) COMPUTE q = QUARTER(ts) SELECT h, q;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => assert!(*v >= 0 && *v < 24, "hour should be 0-23, got {}", v),
        other => panic!("Expected Integer for hour, got {:?}", other),
    }
    match r.rows[0].data.get("q") {
        Some(Value::Integer(v)) => assert!(*v >= 1 && *v <= 4, "quarter should be 1-4, got {}", v),
        other => panic!("Expected Integer for quarter, got {:?}", other),
    }
}

#[test]
fn test_interval_helpers() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();

    let mut p = Parser::new("QUERY t COMPUTE secs = INTERVAL_HOURS(2) COMPUTE days = INTERVAL_DAYS(3) SELECT secs, days;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("secs"), Some(&Value::Integer(7200)));
    assert_eq!(r.rows[0].data.get("days"), Some(&Value::Integer(259200)));
}

#[test]
fn test_day_of_week_and_weekend() {
    let (db, ex) = setup();
    // Unix epoch (1970-01-01) is a Thursday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ts": 0_i64}),
    )
    .unwrap();

    let mut p = Parser::new(
        "QUERY t COMPUTE dow = DAY_OF_WEEK(ts) COMPUTE is_w = IS_WEEKDAY(ts) SELECT dow, is_w;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Thursday = 4 (0=Sunday)
    assert_eq!(r.rows[0].data.get("dow"), Some(&Value::Integer(4)));
    assert_eq!(r.rows[0].data.get("is_w"), Some(&Value::Bool(true)));
}

#[test]
fn test_week_of_year() {
    let (db, ex) = setup();
    // 2024-01-07 = week 1 of the year (approximately)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ts": 1704585600_i64}),
    )
    .unwrap(); // approx 2024-01-07

    let mut p = Parser::new("QUERY t COMPUTE w = WEEK_OF_YEAR(ts) SELECT w;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("w") {
        Some(Value::Integer(v)) => assert!(*v >= 1 && *v <= 53, "week should be 1-53, got {}", v),
        other => panic!("Expected Integer for week, got {:?}", other),
    }
}
