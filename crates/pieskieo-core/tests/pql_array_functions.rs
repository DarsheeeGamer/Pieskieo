/// Integration tests for PQL advanced array functions.
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
fn test_array_intersect_except() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1,2,3,4], "b": [3,4,5,6]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE inter = ARRAY_INTERSECT(a, b) COMPUTE exc = ARRAY_EXCEPT(a, b) SELECT inter, exc;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inter") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2, "Intersection [1,2,3,4] ∩ [3,4,5,6] = [3,4]"),
        other => panic!("Expected Array for intersect, got {:?}", other),
    }
    match r.rows[0].data.get("exc") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2, "Except [1,2,3,4] - [3,4,5,6] = [1,2]"),
        other => panic!("Expected Array for except, got {:?}", other),
    }
}

#[test]
fn test_array_deduplicate() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1,2,2,3,3,3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = ARRAY_DEDUPLICATE(arr) COMPUTE sz = ARRAY_COUNT(d) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_array_contains_all_any() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1,2,3,4,5]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE all_in = ARRAY_CONTAINS_ALL(arr, [2,4]) COMPUTE any_in = ARRAY_CONTAINS_ANY(arr, [9,5]) SELECT all_in, any_in;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("all_in"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("any_in"), Some(&Value::Bool(true)));
}

#[test]
fn test_array_index_of() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": ["a","b","c","d"]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE idx = ARRAY_INDEX_OF(arr, "c") SELECT idx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("idx"), Some(&Value::Integer(2)));
}

#[test]
fn test_array_chunk() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1,2,3,4,5,6]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE chunks = ARRAY_CHUNK(arr, 2) COMPUTE cnt = ARRAY_COUNT(chunks) SELECT cnt;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

#[test]
fn test_array_compact() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, null, 2, null, 3]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE compacted = ARRAY_COMPACT(arr) COMPUTE sz = ARRAY_COUNT(compacted) SELECT sz;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(3)));
}

#[test]
fn test_array_rotate() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1,2,3,4,5]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = ARRAY_ROTATE(arr, 2) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr[0], Value::Integer(3), "After rotating [1,2,3,4,5] by 2, first element should be 3");
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}
