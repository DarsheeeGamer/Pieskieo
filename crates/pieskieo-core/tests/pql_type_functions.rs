/// Integration tests for PQL type conversion and validation functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_safe_int() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "42", "b": "hello", "c": 3.7}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE x = SAFE_INT(a) COMPUTE y = SAFE_INT(b) COMPUTE z = SAFE_INT(c) SELECT x, y, z;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("x"), Some(&Value::Integer(42)));
    assert_eq!(r.rows[0].data.get("y"), Some(&Value::Null));
    assert_eq!(r.rows[0].data.get("z"), Some(&Value::Integer(3)));
}

#[test]
fn test_is_empty_is_truthy() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"empty_str": "", "nonempty": "hi", "zero": 0, "nonzero": 5}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = IS_EMPTY(empty_str) COMPUTE b = IS_EMPTY(nonempty) COMPUTE c = IS_TRUTHY(nonzero) COMPUTE d = IS_TRUTHY(zero) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_positive_negative_zero() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pos": 5, "neg": -3, "z": 0}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE p = IS_POSITIVE(pos) COMPUTE n = IS_NEGATIVE(neg) COMPUTE iz = IS_ZERO(z) COMPUTE inz = IS_ZERO(pos) SELECT p, n, iz, inz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("p"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("iz"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("inz"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_not_null() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 42}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = IS_NOT_NULL(x) COMPUTE b = IS_NOT_NULL(nonexistent) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_ordinal() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n1": 1, "n2": 2, "n3": 3, "n11": 11}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE o1 = ORDINAL(n1) COMPUTE o2 = ORDINAL(n2) COMPUTE o3 = ORDINAL(n3) COMPUTE o11 = ORDINAL(n11) SELECT o1, o2, o3, o11;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("o1"),
        Some(&Value::String("1st".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o2"),
        Some(&Value::String("2nd".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o3"),
        Some(&Value::String("3rd".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o11"),
        Some(&Value::String("11th".to_string()))
    );
}

#[test]
fn test_safe_bool() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "true", "b": "false", "c": "maybe"}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE x = SAFE_BOOL(a) COMPUTE y = SAFE_BOOL(b) COMPUTE z = SAFE_BOOL(c) SELECT x, y, z;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("x"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("y"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("z"), Some(&Value::Null));
}
