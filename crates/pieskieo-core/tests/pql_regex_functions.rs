/// Integration tests for PQL regex/pattern functions.
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
fn test_regexp_count() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"text": "hello world hello foo hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = REGEXP_COUNT(text, "hello") SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(3)));
}

#[test]
fn test_regexp_like() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"email": "user@example.com"})).unwrap();
    // PQL string: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    // The \\ in the PQL string becomes \ after lexer processing, giving regex \.
    let mut p = Parser::new(
        r#"QUERY t COMPUTE is_email = REGEXP_LIKE(email, "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$") SELECT is_email;"#
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("is_email"), Some(&Value::Bool(true)));
}

#[test]
fn test_regexp_extract() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "price: $42.50"})).unwrap();
    // PQL string: "[0-9]+\\.[0-9]+" -> regex [0-9]+\.[0-9]+
    let mut p = Parser::new(
        r#"QUERY t COMPUTE n = REGEXP_EXTRACT(s, "[0-9]+\\.[0-9]+") SELECT n;"#
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::String("42.50".to_string())));
}

#[test]
fn test_regexp_extract_all() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "cat bat sat"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE matches = REGEXP_EXTRACT_ALL(s, "[a-z]at") COMPUTE cnt = ARRAY_COUNT(matches) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

#[test]
fn test_regexp_instr() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pos = REGEXP_INSTR(s, "world") SELECT pos;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pos"), Some(&Value::Integer(6)));
}

#[test]
fn test_split_regex() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "one  two   three"})).unwrap();
    // PQL string: "\\s+" -> regex \s+
    let mut p = Parser::new(
        r#"QUERY t COMPUTE parts = STRING_SPLIT_REGEX(s, "\\s+") COMPUTE cnt = ARRAY_COUNT(parts) SELECT cnt;"#
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}
