/// Integration tests for PQL window/analytical functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup_scores() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for (name, score) in [
        ("Alice", 90),
        ("Bob", 85),
        ("Carol", 92),
        ("Dave", 78),
        ("Eve", 88),
    ] {
        db.put_doc_ns(
            None,
            Some("scores"),
            Uuid::new_v4(),
            serde_json::json!({"name": name, "score": score}),
        )
        .unwrap();
    }
    (dir, db, ex)
}

#[test]
fn test_lag_lead() {
    let (_dir, _db, ex) = setup_scores();

    // ORDER BY score, then LAG and LEAD using OVER (ORDER BY score)
    let mut p = Parser::new(
        "QUERY scores COMPUTE prev = LAG(score, 1) OVER (ORDER BY score) COMPUTE next_val = LEAD(score, 1) OVER (ORDER BY score) SELECT name, score, prev, next_val ORDER BY score;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 5, "Expected 5 rows");

    // First row (lowest score=78) should have Null for prev (no prior row)
    let first = &r.rows[0];
    assert_eq!(
        first.data.get("prev"),
        Some(&Value::Null),
        "First row LAG should be Null"
    );

    // Last row should have Null for next_val
    let last = &r.rows[r.rows.len() - 1];
    assert_eq!(
        last.data.get("next_val"),
        Some(&Value::Null),
        "Last row LEAD should be Null"
    );
}

#[test]
fn test_first_last_value() {
    let (_dir, _db, ex) = setup_scores();

    let mut p = Parser::new(
        "QUERY scores COMPUTE first_s = FIRST_VALUE(score) OVER (ORDER BY score) COMPUTE last_s = LAST_VALUE(score) OVER (ORDER BY score) SELECT name, score, first_s, last_s;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 5);

    // FIRST_VALUE should be 78 (lowest score after ORDER BY)
    for row in &r.rows {
        assert_eq!(
            row.data.get("first_s"),
            Some(&Value::Integer(78)),
            "FIRST_VALUE should be 78 for all rows"
        );
    }

    // LAST_VALUE should be 92 (highest score)
    for row in &r.rows {
        assert_eq!(
            row.data.get("last_s"),
            Some(&Value::Integer(92)),
            "LAST_VALUE should be 92 for all rows"
        );
    }
}

#[test]
fn test_percent_rank() {
    let (_dir, _db, ex) = setup_scores();

    let mut p = Parser::new(
        "QUERY scores COMPUTE pr = PERCENT_RANK() OVER (ORDER BY score) SELECT name, score, pr ORDER BY score;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 5);

    // First row should have percent_rank = 0.0
    match r.rows[0].data.get("pr") {
        Some(Value::Float(v)) => assert!(
            (*v - 0.0).abs() < 1e-9,
            "First PERCENT_RANK should be 0.0, got {}",
            v
        ),
        other => panic!("Expected Float for PERCENT_RANK, got {:?}", other),
    }

    // Last row should have percent_rank = 1.0
    match r.rows[r.rows.len() - 1].data.get("pr") {
        Some(Value::Float(v)) => assert!(
            (*v - 1.0).abs() < 1e-9,
            "Last PERCENT_RANK should be 1.0, got {}",
            v
        ),
        other => panic!("Expected Float for PERCENT_RANK, got {:?}", other),
    }
}

#[test]
fn test_nth_value() {
    let (_dir, _db, ex) = setup_scores();

    // NTH_VALUE(score, 2) should return the 2nd score in the partition ordered by score (85)
    let mut p = Parser::new(
        "QUERY scores COMPUTE nth = NTH_VALUE(score, 2) OVER (ORDER BY score) SELECT name, score, nth;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 5);

    // All rows in the same partition should get the same 2nd value = 85
    for row in &r.rows {
        assert_eq!(
            row.data.get("nth"),
            Some(&Value::Integer(85)),
            "NTH_VALUE(score, 2) should be 85 for all rows"
        );
    }
}

#[test]
fn test_cume_dist() {
    let (_dir, _db, ex) = setup_scores();

    let mut p = Parser::new(
        "QUERY scores COMPUTE cd = CUME_DIST() OVER (ORDER BY score) SELECT name, score, cd ORDER BY score;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 5);

    // Last row (highest score) should have CUME_DIST = 1.0
    match r.rows[r.rows.len() - 1].data.get("cd") {
        Some(Value::Float(v)) => assert!(
            (*v - 1.0).abs() < 1e-9,
            "Last CUME_DIST should be 1.0, got {}",
            v
        ),
        other => panic!("Expected Float for CUME_DIST, got {:?}", other),
    }

    // First row (lowest score) should have CUME_DIST = 0.2 (1/5)
    match r.rows[0].data.get("cd") {
        Some(Value::Float(v)) => assert!(
            (*v - 0.2).abs() < 1e-9,
            "First CUME_DIST should be 0.2, got {}",
            v
        ),
        other => panic!("Expected Float for CUME_DIST, got {:?}", other),
    }
}
