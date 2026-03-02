/// Integration tests for PQL string manipulation functions.
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
fn test_repeat() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "ab"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = REPEAT(s, 3) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("ababab".to_string())));
}

#[test]
fn test_str_split_and_split_part() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "a,b,c,d"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE parts = STR_SPLIT(s, ",") COMPUTE second = SPLIT_PART(s, ",", 2) SELECT parts, second;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("parts") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 4, "Expected 4 parts"),
        other => panic!("Expected Array, got {:?}", other),
    }
    assert_eq!(r.rows[0].data.get("second"), Some(&Value::String("b".to_string())));
}

#[test]
fn test_array_to_string() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"tags": ["rust", "db", "fast"]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE joined = ARRAY_TO_STRING(tags, ", ") SELECT joined;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("joined"), Some(&Value::String("rust, db, fast".to_string())));
}

#[test]
fn test_slugify() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"title": "Hello World! This is a Test"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE slug = SLUGIFY(title) SELECT slug;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("slug"), Some(&Value::String("hello-world-this-is-a-test".to_string())));
}

#[test]
fn test_proper_case() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"name": "john doe smith"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE proper = PROPER_CASE(name) SELECT proper;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("proper"), Some(&Value::String("John Doe Smith".to_string())));
}

#[test]
fn test_string_normalize() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "  Hello   World  "})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE n = STRING_NORMALIZE(s) SELECT n;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::String("hello world".to_string())));
}

#[test]
fn test_format_number() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"n": 1234567.89})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE fmt = FORMAT_NUMBER(n, 2) SELECT fmt;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("fmt"), Some(&Value::String("1,234,567.89".to_string())));
}
