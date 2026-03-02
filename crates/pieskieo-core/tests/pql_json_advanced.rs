/// Integration tests for advanced PQL JSON manipulation functions.
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

// ── JSON_POINTER ──────────────────────────────────────────────────────────────

#[test]
fn test_json_pointer_nested() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"doc": {"a": {"b": 42}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = JSON_POINTER(doc, "/a/b") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(42)));
}

#[test]
fn test_json_pointer_top_level() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"name": "Alice", "score": 99})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = JSON_POINTER(name, "") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // empty pointer returns the value itself
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::String("Alice".to_string())));
}

#[test]
fn test_json_pointer_missing_returns_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = JSON_POINTER(a, "/missing") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Null));
}

// ── JSON_HAS_PATH ─────────────────────────────────────────────────────────────

#[test]
fn test_json_has_path_exists() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"meta": {"active": true}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = JSON_HAS_PATH(meta, "/active") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Bool(true)));
}

#[test]
fn test_json_has_path_missing() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"meta": {"active": true}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = JSON_HAS_PATH(meta, "/ghost") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Bool(false)));
}

// ── JSON_SET_PATH ─────────────────────────────────────────────────────────────

#[test]
fn test_json_set_path_top_level() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"x": 1, "y": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x, "y", y) COMPUTE updated = JSON_SET_PATH(base, "/z", 99) COMPUTE sz = JSON_SIZE(updated) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

// ── JSON_DELETE_PATH ──────────────────────────────────────────────────────────

#[test]
fn test_json_delete_path() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2, "c": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b, "c", c) COMPUTE trimmed = JSON_DELETE_PATH(base, "/b") COMPUTE sz = JSON_SIZE(trimmed) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

// ── JSON_DIFF ─────────────────────────────────────────────────────────────────

#[test]
fn test_json_diff_changed_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"score": 10})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj1 = JSON_BUILD_OBJECT("score", score) COMPUTE obj2 = JSON_BUILD_OBJECT("score", 20) COMPUTE d = JSON_DIFF(obj1, obj2) COMPUTE has_score = JSON_CONTAINS_KEY(d, "score") SELECT has_score;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has_score"), Some(&Value::Bool(true)));
}

#[test]
fn test_json_diff_no_diff() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"n": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj1 = JSON_BUILD_OBJECT("n", n) COMPUTE obj2 = JSON_BUILD_OBJECT("n", n) COMPUTE d = JSON_DIFF(obj1, obj2) COMPUTE sz = JSON_SIZE(d) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // identical objects => empty diff
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(0)));
}

// ── JSON_COMPACT / JSON_PRETTY ────────────────────────────────────────────────

#[test]
fn test_json_compact() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"k": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("k", k) COMPUTE s = JSON_COMPACT(base) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            // Should be valid JSON with no newlines
            assert!(!s.is_empty());
            assert!(!s.contains('\n'));
            let parsed: serde_json::Value = serde_json::from_str(s).expect("compact output should be valid JSON");
            assert_eq!(parsed["k"], serde_json::json!(1));
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_json_pretty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"k": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("k", k) COMPUTE s = JSON_PRETTY(base) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            assert!(s.contains('\n'), "pretty output should contain newlines");
            let parsed: serde_json::Value = serde_json::from_str(s).expect("pretty output should be valid JSON");
            assert_eq!(parsed["k"], serde_json::json!(1));
        }
        other => panic!("expected String, got {:?}", other),
    }
}

// ── JSON_KEYS_RECURSIVE ───────────────────────────────────────────────────────

#[test]
fn test_json_keys_recursive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nested": {"inner": {"deep": 1}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE keys = JSON_KEYS_RECURSIVE(nested) SELECT keys;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("keys") {
        Some(Value::Array(a)) => {
            let strs: Vec<&str> = a.iter().filter_map(|v| {
                if let Value::String(s) = v { Some(s.as_str()) } else { None }
            }).collect();
            assert!(strs.contains(&"inner"), "should have 'inner', got {:?}", strs);
            assert!(strs.contains(&"inner.deep"), "should have 'inner.deep', got {:?}", strs);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── JSON_VALUE_AT ─────────────────────────────────────────────────────────────

#[test]
fn test_json_value_at() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"user": {"age": 30}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = JSON_VALUE_AT(user, "age") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(30)));
}

// ── JSON_REPLACE ──────────────────────────────────────────────────────────────

#[test]
fn test_json_replace_existing_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"x": 1, "y": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x, "y", y) COMPUTE updated = JSON_REPLACE(base, "/x", 99) COMPUTE v = JSON_POINTER(updated, "/x") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(99)));
}

#[test]
fn test_json_replace_missing_key_no_add() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"x": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x) COMPUTE updated = JSON_REPLACE(base, "/newkey", 100) COMPUTE sz = JSON_SIZE(updated) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // newkey doesn't exist in base so it must NOT be added
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}

// ── JSON_ARR_LEN ──────────────────────────────────────────────────────────────

#[test]
fn test_json_arr_len() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"items": [10, 20, 30]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = JSON_ARR_LEN(items, "/0") SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // items is the array, pointer "/0" goes into element 0 which is integer -> Null
    // Let's test pointer on a field that holds an array
    // Actually items is Array[10,20,30]; pointer "/0" -> 10 (integer) -> Null
    // So test with the array itself at root pointer
    let _ = r; // discard; use a better test below
}

#[test]
fn test_json_arr_len_on_nested_array() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"container": {"nums": [1, 2, 3, 4]}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = JSON_ARR_LEN(container, "/nums") SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(4)));
}

// ── REMOVE_NULLS ──────────────────────────────────────────────────────────────

#[test]
fn test_remove_nulls() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": 1, "c": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", null, "c", c) COMPUTE cleaned = REMOVE_NULLS(base) COMPUTE sz = JSON_SIZE(cleaned) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

// ── JSON_SELECT_KEYS ──────────────────────────────────────────────────────────

#[test]
fn test_json_select_keys() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"id": 1, "name": "Bob", "secret": "hidden"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("id", id, "name", name, "secret", secret) COMPUTE proj = JSON_SELECT_KEYS(base, ["id", "name"]) COMPUTE sz = JSON_SIZE(proj) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_json_select_keys_missing_key_skipped() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE proj = JSON_SELECT_KEYS(base, ["a", "nonexistent"]) COMPUTE sz = JSON_SIZE(proj) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // only "a" exists, "nonexistent" is skipped
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}

// ── JSON_MINIFY alias ─────────────────────────────────────────────────────────

#[test]
fn test_json_minify_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"v": 7})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("v", v) COMPUTE s = JSON_MINIFY(base) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            assert!(!s.is_empty());
            let parsed: serde_json::Value = serde_json::from_str(s).expect("minify output should be valid JSON");
            assert_eq!(parsed["v"], serde_json::json!(7));
        }
        other => panic!("expected String, got {:?}", other),
    }
}
