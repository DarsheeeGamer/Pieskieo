/// Integration tests for PQL object/map manipulation functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_object_entries() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"data": {"x": 1, "y": 2}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE e = OBJECT_ENTRIES(data) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("e") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "should have 2 entries for 2-key object");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_pick() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"profile": {"name": "Alice", "age": 30, "secret": "hidden"}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE p = PICK(profile, "name", "age") SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("name"), "should have name");
            assert!(obj.contains_key("age"), "should have age");
            assert!(!obj.contains_key("secret"), "should NOT have secret");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_omit() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"data": {"a": 1, "b": 2, "c": 3}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE o = OMIT(data, "c") SELECT o;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("o") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("a"), "should have a");
            assert!(obj.contains_key("b"), "should have b");
            assert!(!obj.contains_key("c"), "should NOT have c");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_object_size() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2, "c": 3}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE sz = OBJECT_SIZE(obj) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_object_has() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"obj": {"x": 42}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE yes = OBJECT_HAS(obj, "x") COMPUTE no = OBJECT_HAS(obj, "z") SELECT yes, no;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("yes"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("no"), Some(&Value::Bool(false)));
}

#[test]
fn test_object_get_with_default() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"cfg": {"color": "blue"}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE c = OBJECT_GET(cfg, "color", "red") COMPUTE d = OBJECT_GET(cfg, "size", "medium") SELECT c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("c"),
        Some(&Value::String("blue".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::String("medium".to_string()))
    );
}

#[test]
fn test_invert_object() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mapping": {"en": "hello", "es": "hola"}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE inv = INVERT_OBJECT(mapping) SELECT inv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inv") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("hello"), "inverted: hello -> en");
            assert!(obj.contains_key("hola"), "inverted: hola -> es");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_flatten_object() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nested": {"a": {"b": 42}}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE f = FLATTEN_OBJECT(nested, ".") SELECT f;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Object(obj)) => {
            assert!(
                obj.contains_key("a.b"),
                "should have flattened key 'a.b', got: {:?}",
                obj.keys().collect::<Vec<_>>()
            );
            assert_eq!(obj.get("a.b"), Some(&Value::Integer(42)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}
