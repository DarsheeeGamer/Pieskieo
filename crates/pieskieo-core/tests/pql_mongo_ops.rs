/// Integration tests for MongoDB-style aggregation pipeline operator functions.
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

// ── COND / IF_THEN_ELSE ──────────────────────────────────────────────────────

#[test]
fn test_cond_true_branch() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"flag": true})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = COND(flag, "pass", "fail") SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::String("pass".to_string())));
}

#[test]
fn test_cond_false_branch() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"flag": false})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = COND(flag, "pass", "fail") SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::String("fail".to_string())));
}

#[test]
fn test_cond_null_is_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"flag": null})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = COND(flag, "yes", "no") SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::String("no".to_string())));
}

#[test]
fn test_if_then_else_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"active": true})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IF_THEN_ELSE(active, "big", "small") SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::String("big".to_string())));
}

// ── ARRAY_ELEM_AT ────────────────────────────────────────────────────────────

#[test]
fn test_array_elem_at_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE e = ARRAY_ELEM_AT(arr, 1) SELECT e;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Integer(20)));
}

#[test]
fn test_array_elem_at_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE e = ARRAY_ELEM_AT(arr, -1) SELECT e;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Integer(30)));
}

#[test]
fn test_array_elem_at_out_of_bounds() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE e = ARRAY_ELEM_AT(arr, 10) SELECT e;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Null));
}

// ── ARRAY_MAP / ARRAY_FILTER / ARRAY_REDUCE ─────────────────────────────────

#[test]
fn test_array_map_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_MAP(arr, arr) COMPUTE sz = ARRAY_COUNT(result) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_array_filter_nulls() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, null, 2, false, 3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_FILTER(arr, arr) COMPUTE sz = ARRAY_COUNT(result) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // null and false should be filtered out
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_array_reduce_sum() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_REDUCE(arr, 0, arr) SELECT result;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Integer(60)));
}

// ── OBJECT_SET / OBJECT_PUT ──────────────────────────────────────────────────

#[test]
fn test_object_set_new_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = OBJECT_SET(obj, "b", 2) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("a"), Some(&Value::Integer(1)));
            assert_eq!(obj.get("b"), Some(&Value::Integer(2)));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_object_put_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"x": 10}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = OBJECT_PUT(obj, "y", 20) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("y"), Some(&Value::Integer(20)));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── OBJECT_REMOVE / OBJECT_UNSET ────────────────────────────────────────────

#[test]
fn test_object_remove() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2, "c": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = OBJECT_REMOVE(obj, "b") SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("a"), Some(&Value::Integer(1)));
            assert!(obj.get("b").is_none(), "key 'b' should be removed");
            assert_eq!(obj.get("c"), Some(&Value::Integer(3)));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_object_unset_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"x": 1, "y": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = OBJECT_UNSET(obj, "x") SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Object(obj)) => {
            assert!(obj.get("x").is_none(), "key 'x' should be removed");
            assert_eq!(obj.get("y"), Some(&Value::Integer(2)));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── MERGE_OBJECTS alias ──────────────────────────────────────────────────────

#[test]
fn test_merge_objects_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": {"x": 1}, "b": {"y": 2}})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = MERGE_OBJECTS(a, b) SELECT result;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("x"), Some(&Value::Integer(1)));
            assert_eq!(obj.get("y"), Some(&Value::Integer(2)));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── VALUES_ARRAY alias ───────────────────────────────────────────────────────

#[test]
fn test_values_array_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = VALUES_ARRAY(obj) COMPUTE sz = ARRAY_COUNT(result) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

// ── IN_ARRAY ─────────────────────────────────────────────────────────────────

#[test]
fn test_in_array_found() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3], "v": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE found = IN_ARRAY(v, arr) SELECT found;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("found"), Some(&Value::Bool(true)));
}

#[test]
fn test_in_array_not_found() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3], "v": 9})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE found = IN_ARRAY(v, arr) SELECT found;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("found"), Some(&Value::Bool(false)));
}

// ── ARRAY_PUSH / ARRAY_POP / ARRAY_SHIFT / ARRAY_UNSHIFT ────────────────────

#[test]
fn test_array_push() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_PUSH(arr, 4) COMPUTE sz = ARRAY_COUNT(result) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(4)));
}

#[test]
fn test_array_pop() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_POP(arr) COMPUTE sz = ARRAY_COUNT(result) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_array_shift() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_SHIFT(arr) SELECT result;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "shift removes first element");
            assert_eq!(arr[0], Value::Integer(20));
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_unshift() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [20, 30]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = ARRAY_UNSHIFT(arr, 10) SELECT result;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("result") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "unshift prepends element");
            assert_eq!(arr[0], Value::Integer(10));
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

// ── LET / MAP / REDUCE ───────────────────────────────────────────────────────

#[test]
fn test_let_returns_last_expr() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"x": 42})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = LET("var", x, x) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Integer(42)));
}

#[test]
fn test_reduce_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE result = REDUCE(arr, 0, arr) SELECT result;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Simplified reduce returns initial value
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Integer(0)));
}
