/// Integration tests for PQL conditional/control-flow functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
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
fn test_iif() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"score": 85, "passing": false})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE grade = IIF(passing, \"A\", \"B\") SELECT grade;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("grade"), Some(&Value::String("B".to_string())));
}

#[test]
fn test_switch() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"status": "active"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE label = SWITCH(status, "active", "Active User", "inactive", "Inactive User", "Unknown") SELECT label;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("label"), Some(&Value::String("Active User".to_string())));
}

#[test]
fn test_between_inclusive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"score": 75})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE in_range = BETWEEN_INCLUSIVE(score, 70, 80) COMPUTE out_range = BETWEEN_INCLUSIVE(score, 80, 90) SELECT in_range, out_range;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("in_range"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("out_range"), Some(&Value::Bool(false)));
}

#[test]
fn test_try_parse() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"valid_int": "42", "invalid_int": "abc"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = TRY_PARSE(valid_int, "INT") COMPUTE bad = TRY_PARSE(invalid_int, "INT") SELECT n, bad;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(42)));
    assert_eq!(r.rows[0].data.get("bad"), Some(&Value::Null));
}

#[test]
fn test_zero_if_null_and_empty_if_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"x": null})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE z = ZERO_IF_NULL(x) COMPUTE e = EMPTY_IF_NULL(x) SELECT z, e;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("z"), Some(&Value::Integer(0)));
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::String(String::new())));
}
