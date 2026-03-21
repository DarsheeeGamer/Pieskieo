/// Integration tests for PQL JSON/document functions.
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
fn test_json_path_query() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"user": {"name": "Alice", "age": 30}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE name = JSON_PATH_QUERY(user, "name") SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("name"),
        Some(&Value::String("Alice".to_string()))
    );
}

#[test]
fn test_json_path_exists() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": {"b": 1}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE exists_ab = JSON_PATH_EXISTS(a, "b") COMPUTE exists_c = JSON_PATH_EXISTS(a, "c") SELECT exists_ab, exists_c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("exists_ab"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("exists_c"), Some(&Value::Bool(false)));
}

#[test]
fn test_json_set() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1, "y": 2}),
    )
    .unwrap();

    // JSON_SET adds a key; field `obj` holds the entire document as an object
    // We'll pass the document fields and build the object with JSON_BUILD_OBJECT, then set a key
    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x, "y", y) COMPUTE obj2 = JSON_SET(base, "z", 3) COMPUTE sz = JSON_SIZE(obj2) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_json_delete() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1, "y": 2, "z": 3}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x, "y", y, "z", z) COMPUTE obj_new = JSON_DELETE(base, "z") COMPUTE sz = JSON_SIZE(obj_new) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_json_rename_key() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"name": "Alice", "age": 30}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("name", name, "age", age) COMPUTE renamed = JSON_RENAME_KEY(base, "name", "full_name") COMPUTE has_new = JSON_CONTAINS_KEY(renamed, "full_name") COMPUTE has_old = JSON_CONTAINS_KEY(renamed, "name") SELECT has_new, has_old;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has_new"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("has_old"), Some(&Value::Bool(false)));
}

#[test]
fn test_json_contains_key() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"name": "Alice", "age": 30}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("name", name, "age", age) COMPUTE has_name = JSON_CONTAINS_KEY(base, "name") COMPUTE has_foo = HAS_KEY(base, "foo") SELECT has_name, has_foo;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has_name"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("has_foo"), Some(&Value::Bool(false)));
}

#[test]
fn test_json_typeof() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 42, "s": "hello", "arr": [1, 2], "obj": {}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE tn = JSON_TYPEOF(n) COMPUTE ts = JSON_TYPEOF(s) COMPUTE ta = JSON_TYPEOF(arr) SELECT tn, ts, ta;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("tn"),
        Some(&Value::String("number".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("ts"),
        Some(&Value::String("string".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("ta"),
        Some(&Value::String("array".to_string()))
    );
}

#[test]
fn test_json_strip_nulls() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "c": "hello"}),
    )
    .unwrap();

    // Build an object that includes a null value, then strip it
    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", null, "c", c) COMPUTE cleaned = JSON_STRIP_NULLS(base) COMPUTE sz = JSON_SIZE(cleaned) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_json_deep_merge() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1, "y": 2, "z": 3}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE obj1 = JSON_BUILD_OBJECT("x", x, "y", y) COMPUTE obj2 = JSON_BUILD_OBJECT("y", z, "w", 4) COMPUTE merged = JSON_DEEP_MERGE(obj1, obj2) COMPUTE sz = JSON_SIZE(merged) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // obj1 has x,y; obj2 has y,w -> merged has x,y,w -> 3 keys
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_json_size() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"name": "Alice", "age": 30, "active": true}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("name", name, "age", age, "active", active) COMPUTE sz = JSON_SIZE(base) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_json_flatten() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nested": {"key": "value"}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE flat = JSON_FLATTEN(nested) COMPUTE has_key = JSON_CONTAINS_KEY(flat, "key") SELECT has_key;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has_key"), Some(&Value::Bool(true)));
}

#[test]
fn test_jsonpath_exists_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"data": {"score": 99}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE e1 = JSONPATH_EXISTS(data, "score") COMPUTE e2 = JSON_PATH_EXISTS(data, "missing") SELECT e1, e2;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("e1"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("e2"), Some(&Value::Bool(false)));
}

#[test]
fn test_json_remove_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE result = JSON_REMOVE(base, "b") COMPUTE sz = JSON_SIZE(result) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}
