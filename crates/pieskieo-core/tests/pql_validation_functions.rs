/// Integration tests for PQL email, URL, phone and pattern validation functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_is_email() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": "user@example.com", "b": "notanemail", "c": "@nodomain.com"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE va = IS_EMAIL(a) COMPUTE vb = IS_EMAIL(b) COMPUTE vc = IS_EMAIL(c) SELECT va, vb, vc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("va"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("vb"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("vc"), Some(&Value::Bool(false)));
}

#[test]
fn test_email_domain_local() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"email": "alice@example.org"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE dom = EMAIL_DOMAIN(email) COMPUTE loc = EMAIL_LOCAL(email) SELECT dom, loc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dom"), Some(&Value::String("example.org".to_string())));
    assert_eq!(r.rows[0].data.get("loc"), Some(&Value::String("alice".to_string())));
}

#[test]
fn test_url_parts() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://api.example.com/v1/users?page=2&limit=10"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE url_scheme = URL_SCHEME(url) COMPUTE url_host = URL_HOST(url) COMPUTE url_path = URL_PATH(url) COMPUTE url_query = URL_QUERY(url) SELECT url_scheme, url_host, url_path, url_query;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("url_scheme"), Some(&Value::String("https".to_string())));
    assert_eq!(r.rows[0].data.get("url_host"), Some(&Value::String("api.example.com".to_string())));
    assert_eq!(r.rows[0].data.get("url_path"), Some(&Value::String("/v1/users".to_string())));
    assert_eq!(r.rows[0].data.get("url_query"), Some(&Value::String("page=2&limit=10".to_string())));
}

#[test]
fn test_is_url() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": "https://example.com", "b": "not-a-url", "c": "ftp://files.example.com"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE va = IS_URL(a) COMPUTE vb = IS_URL(b) COMPUTE vc = IS_URL(c) SELECT va, vb, vc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("va"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("vb"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("vc"), Some(&Value::Bool(true)));
}

#[test]
fn test_luhn_check() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 4532015112830366 is a valid Luhn number; 1234567890123456 is not
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"valid": "4532015112830366", "invalid": "1234567890123456"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE a = LUHN_CHECK(valid) COMPUTE b = LUHN_CHECK(invalid) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_mask_email() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"email": "alice@example.com"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE m = MASK_EMAIL(email) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::String(s)) => {
            assert!(s.contains('@'), "masked email should still contain @");
            assert!(s.starts_with('a'), "first char should be visible");
            assert!(s.contains('*'), "should contain mask chars");
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_is_phone() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": "+1-800-555-1234", "b": "abc", "c": "123"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE va = IS_PHONE(a) COMPUTE vb = IS_PHONE(b) COMPUTE vc = IS_PHONE(c) SELECT va, vb, vc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("va"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("vb"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("vc"), Some(&Value::Bool(false)));
}
