/// Integration tests for PQL numeric/math utility functions.
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
fn test_safe_divide() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 10, "b": 0}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = SAFE_DIVIDE(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Null));
}

#[test]
fn test_factorial() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE f = FACTORIAL(n) SELECT f;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("f"), Some(&Value::Integer(120)));
}

#[test]
fn test_gcd_lcm() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 12, "b": 8}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE g = GCD(a, b) COMPUTE l = LCM(a, b) SELECT g, l;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("g"), Some(&Value::Integer(4)));
    assert_eq!(r.rows[0].data.get("l"), Some(&Value::Integer(24)));
}

#[test]
fn test_lerp() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = LERP(0, 10, 0.5) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Float(5.0)));
}

#[test]
fn test_to_int_to_float() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "42", "f": 3.7}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE i = TO_INT(s) COMPUTE fl = TO_FLOAT(f) SELECT i, fl;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("i"), Some(&Value::Integer(42)));
    assert_eq!(r.rows[0].data.get("fl"), Some(&Value::Float(3.7)));
}

#[test]
fn test_log2_log10() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 8}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE l2 = LOG2(n) COMPUTE l10 = LOG10(n) SELECT l2, l10;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("l2"), Some(&Value::Float(3.0)));
    match r.rows[0].data.get("l10") {
        Some(Value::Float(f)) => assert!((*f - 0.903).abs() < 0.001),
        other => panic!("Expected Float, got {:?}", other),
    }
}
