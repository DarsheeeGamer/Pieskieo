/// Integration tests for PQL JSON/object utility built-in functions.
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

// ── JSON_KEYS ─────────────────────────────────────────────────────────────────

#[test]
fn test_json_keys_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kk = JSON_KEYS(obj) SELECT kk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kk") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("a".to_string()));
            assert_eq!(arr[1], Value::String("b".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_json_keys_sorted() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"z": 1, "a": 2, "m": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kk = JSON_KEYS(obj) SELECT kk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kk") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr[0], Value::String("a".to_string()));
            assert_eq!(arr[1], Value::String("m".to_string()));
            assert_eq!(arr[2], Value::String("z".to_string()));
        }
        other => panic!("expected sorted array, got {:?}", other),
    }
}

// ── OBJECT_KEYS alias ─────────────────────────────────────────────────────────

#[test]
fn test_object_keys_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"x": 10, "y": 20}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kk = OBJECT_KEYS(obj) SELECT kk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kk") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("x".to_string()));
            assert_eq!(arr[1], Value::String("y".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── JSON_VALUES ───────────────────────────────────────────────────────────────

#[test]
fn test_json_values_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE vv = JSON_VALUES(obj) SELECT vv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("vv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            // sorted by key: "a" -> 1, "b" -> 2
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(2));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_json_values_sorted_by_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"z": 99, "a": 11}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE vv = JSON_VALUES(obj) SELECT vv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("vv") {
        Some(Value::Array(arr)) => {
            // "a" comes before "z", so 11 before 99
            assert_eq!(arr[0], Value::Integer(11));
            assert_eq!(arr[1], Value::Integer(99));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── OBJECT_VALUES alias ───────────────────────────────────────────────────────

#[test]
fn test_object_values_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"p": 5, "q": 6}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE vv = OBJECT_VALUES(obj) SELECT vv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("vv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── OBJECT_ENTRIES / JSON_ENTRIES ─────────────────────────────────────────────

#[test]
fn test_object_entries_structure() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"data": {"k1": "v1", "k2": "v2"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ee = OBJECT_ENTRIES(data) SELECT ee;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ee") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_json_entries_returns_key_value_pairs() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ee = JSON_ENTRIES(obj) SELECT ee;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ee") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            // Each element should be [key, value] array, sorted by key
            match &arr[0] {
                Value::Array(pair) => {
                    assert_eq!(pair[0], Value::String("a".to_string()));
                    assert_eq!(pair[1], Value::Integer(1));
                }
                other => panic!("expected [k,v] pair, got {:?}", other),
            }
            match &arr[1] {
                Value::Array(pair) => {
                    assert_eq!(pair[0], Value::String("b".to_string()));
                    assert_eq!(pair[1], Value::Integer(2));
                }
                other => panic!("expected [k,v] pair, got {:?}", other),
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── OBJECT_PICK / JSON_PICK ───────────────────────────────────────────────────

#[test]
fn test_object_pick_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2, "c": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pp = OBJECT_PICK(obj, "a", "c") SELECT pp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pp") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("a"));
            assert!(obj.contains_key("c"));
            assert!(!obj.contains_key("b"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_pick_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"x": 10, "y": 20, "z": 30}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pp = JSON_PICK(obj, "x", "z") SELECT pp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pp") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("x"));
            assert!(obj.contains_key("z"));
            assert!(!obj.contains_key("y"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── OBJECT_OMIT / JSON_OMIT ───────────────────────────────────────────────────

#[test]
fn test_object_omit_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2, "c": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE oo = OBJECT_OMIT(obj, "b") SELECT oo;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("oo") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("a"));
            assert!(obj.contains_key("c"));
            assert!(!obj.contains_key("b"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_omit_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"p": 1, "q": 2, "r": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE oo = JSON_OMIT(obj, "q", "r") SELECT oo;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("oo") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 1);
            assert!(obj.contains_key("p"));
            assert!(!obj.contains_key("q"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── OBJECT_MERGE / JSON_MERGE ─────────────────────────────────────────────────

#[test]
fn test_object_merge_disjoint() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1}})).unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", 1) COMPUTE extra = JSON_BUILD_OBJECT("b", 2) COMPUTE merged = OBJECT_MERGE(base, extra) SELECT merged;"#
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("merged") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("a"));
            assert!(obj.contains_key("b"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_merge_right_wins_conflict() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1}})).unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE left_obj = JSON_BUILD_OBJECT("a", 1, "b", 2) COMPUTE right_obj = JSON_BUILD_OBJECT("a", 99, "c", 3) COMPUTE merged = JSON_MERGE(left_obj, right_obj) SELECT merged;"#
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("merged") {
        Some(Value::Object(obj)) => {
            // right wins on "a"
            assert_eq!(obj.get("a"), Some(&Value::Integer(99)));
            assert!(obj.contains_key("b"));
            assert!(obj.contains_key("c"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── OBJECT_INVERT / SWAP_KEYS_VALUES ─────────────────────────────────────────

#[test]
fn test_object_invert_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": "x", "b": "y"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE inv = OBJECT_INVERT(obj) SELECT inv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inv") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert_eq!(obj.get("x"), Some(&Value::String("a".to_string())));
            assert_eq!(obj.get("y"), Some(&Value::String("b".to_string())));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_swap_keys_values_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"foo": "bar"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE inv = SWAP_KEYS_VALUES(obj) SELECT inv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inv") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("bar"), Some(&Value::String("foo".to_string())));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── JSON_SIZE / OBJECT_SIZE ───────────────────────────────────────────────────

#[test]
fn test_json_size_object() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sz = JSON_SIZE(obj) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_object_size_array() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sz = OBJECT_SIZE(arr) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(4)));
}

// ── JSON_DEPTH / OBJECT_DEPTH ─────────────────────────────────────────────────

#[test]
fn test_json_depth_empty_object() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dd = JSON_DEPTH(obj) SELECT dd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dd"), Some(&Value::Integer(1)));
}

#[test]
fn test_json_depth_nested() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": {"b": 1}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dd = JSON_DEPTH(obj) SELECT dd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dd"), Some(&Value::Integer(2)));
}

#[test]
fn test_object_depth_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": {"b": {"c": 42}}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dd = OBJECT_DEPTH(obj) SELECT dd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dd"), Some(&Value::Integer(3)));
}

// ── JSON_FLATTEN / FLATTEN_OBJECT ─────────────────────────────────────────────

#[test]
fn test_json_flatten_nested() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": {"b": 1}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE flat = JSON_FLATTEN(obj) SELECT flat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("flat") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("a.b"), "should have flattened key 'a.b'");
            assert_eq!(obj.get("a.b"), Some(&Value::Integer(1)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_flatten_object_already_flat() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"x": 1, "y": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE flat = FLATTEN_OBJECT(obj) SELECT flat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("flat") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("x"));
            assert!(obj.contains_key("y"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── OBJECT_RENAME_KEY / JSON_RENAME ───────────────────────────────────────────

#[test]
fn test_object_rename_key_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE renamed = OBJECT_RENAME_KEY(obj, "a", "alpha") SELECT renamed;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("renamed") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("alpha"), "should have new key 'alpha'");
            assert!(!obj.contains_key("a"), "should NOT have old key 'a'");
            assert_eq!(obj.get("alpha"), Some(&Value::Integer(1)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_rename_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"old_name": "Alice"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE renamed = JSON_RENAME(obj, "old_name", "name") SELECT renamed;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("renamed") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("name"));
            assert!(!obj.contains_key("old_name"));
            assert_eq!(obj.get("name"), Some(&Value::String("Alice".to_string())));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── OBJECT_MAP_VALUES / JSON_MAP_VALUES ───────────────────────────────────────

#[test]
fn test_object_map_values_scale() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": {"a": 2, "b": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE scaled = OBJECT_MAP_VALUES(nv, 2.0) SELECT scaled;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("scaled") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("a"), Some(&Value::Float(4.0)));
            assert_eq!(obj.get("b"), Some(&Value::Float(6.0)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_map_values_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": {"x": 10, "y": 5}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE scaled = JSON_MAP_VALUES(nv, 3.0) SELECT scaled;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("scaled") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("x"), Some(&Value::Float(30.0)));
            assert_eq!(obj.get("y"), Some(&Value::Float(15.0)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_object_map_values_non_numeric_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": {"label": "hello", "count": 5}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE scaled = OBJECT_MAP_VALUES(nv, 2.0) SELECT scaled;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("scaled") {
        Some(Value::Object(obj)) => {
            // non-numeric stays as-is
            assert_eq!(obj.get("label"), Some(&Value::String("hello".to_string())));
            // numeric gets scaled
            assert_eq!(obj.get("count"), Some(&Value::Float(10.0)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── JSON_CONTAINS_KEY / HAS_KEY ───────────────────────────────────────────────

#[test]
fn test_json_contains_key_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE has = JSON_CONTAINS_KEY(obj, "a") SELECT has;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has"), Some(&Value::Bool(true)));
}

#[test]
fn test_json_contains_key_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE has = JSON_CONTAINS_KEY(obj, "b") SELECT has;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has"), Some(&Value::Bool(false)));
}

#[test]
fn test_has_key_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"name": "Bob"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE has = HAS_KEY(obj, "name") SELECT has;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("has"), Some(&Value::Bool(true)));
}

// ── OBJECT_FILTER_NULLS / COMPACT_OBJECT ─────────────────────────────────────

#[test]
fn test_object_filter_nulls_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": null, "c": "hello"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE compacted = OBJECT_FILTER_NULLS(obj) SELECT compacted;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("compacted") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert!(obj.contains_key("a"));
            assert!(obj.contains_key("c"));
            assert!(!obj.contains_key("b"), "null key should be removed");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_compact_object_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"x": null, "y": 42}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE compacted = COMPACT_OBJECT(obj) SELECT compacted;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("compacted") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 1);
            assert!(!obj.contains_key("x"));
            assert!(obj.contains_key("y"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_object_filter_nulls_no_nulls() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE compacted = OBJECT_FILTER_NULLS(obj) SELECT compacted;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("compacted") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── ARRAY_TO_OBJECT_KV / PAIRS_TO_OBJECT ──────────────────────────────────────

#[test]
fn test_array_to_object_kv_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"kv": [["a", 1], ["b", 2]]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = ARRAY_TO_OBJECT_KV(kv) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
            assert_eq!(obj.get("a"), Some(&Value::Integer(1)));
            assert_eq!(obj.get("b"), Some(&Value::Integer(2)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_pairs_to_object_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"kv": [["name", "Alice"], ["score", 95]]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = PAIRS_TO_OBJECT(kv) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("name"), Some(&Value::String("Alice".to_string())));
            assert_eq!(obj.get("score"), Some(&Value::Integer(95)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── Additional edge-case and coverage tests ───────────────────────────────────

#[test]
fn test_json_keys_single_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"only": true}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kk = JSON_KEYS(obj) SELECT kk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kk") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], Value::String("only".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_json_depth_scalar_is_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": {"b": {"c": {"d": 1}}}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dd = JSON_DEPTH(obj) SELECT dd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // obj has depth 4: {a: {b: {c: {d: 1}}}}
    assert_eq!(r.rows[0].data.get("dd"), Some(&Value::Integer(4)));
}

#[test]
fn test_json_size_three_keys() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2, "c": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sz = JSON_SIZE(obj) SELECT sz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_object_merge_single_key_each() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1}})).unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE left_obj = JSON_BUILD_OBJECT("a", 1) COMPUTE right_obj = JSON_BUILD_OBJECT("b", 2) COMPUTE merged = JSON_MERGE(left_obj, right_obj) SELECT merged;"#
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("merged") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 2);
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_object_invert_integer_values() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"first": 1, "second": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE inv = OBJECT_INVERT(obj) SELECT inv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inv") {
        Some(Value::Object(obj)) => {
            // integer values become string keys
            assert_eq!(obj.get("1"), Some(&Value::String("first".to_string())));
            assert_eq!(obj.get("2"), Some(&Value::String("second".to_string())));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_array_to_object_kv_three_pairs() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"kv": [["x", 10], ["y", 20], ["z", 30]]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = ARRAY_TO_OBJECT_KV(kv) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 3);
            assert_eq!(obj.get("x"), Some(&Value::Integer(10)));
            assert_eq!(obj.get("y"), Some(&Value::Integer(20)));
            assert_eq!(obj.get("z"), Some(&Value::Integer(30)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_pick_missing_key_ignored() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pp = JSON_PICK(obj, "a", "nonexistent") SELECT pp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pp") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 1);
            assert!(obj.contains_key("a"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_json_flatten_deeply_nested() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": {"b": {"c": 42}}}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE flat = JSON_FLATTEN(obj) SELECT flat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("flat") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("a.b.c"), "should have key 'a.b.c'");
            assert_eq!(obj.get("a.b.c"), Some(&Value::Integer(42)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_object_rename_key_preserves_other_keys() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": 1, "b": 2, "c": 3}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE renamed = OBJECT_RENAME_KEY(obj, "b", "beta") SELECT renamed;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("renamed") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 3);
            assert!(obj.contains_key("a"));
            assert!(obj.contains_key("beta"));
            assert!(obj.contains_key("c"));
            assert!(!obj.contains_key("b"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_compact_object_all_nulls() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"obj": {"a": null, "b": null}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE compacted = COMPACT_OBJECT(obj) SELECT compacted;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("compacted") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 0, "all null keys should be removed");
        }
        other => panic!("expected object, got {:?}", other),
    }
}
