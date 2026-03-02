/// Integration tests for PQL hash/encoding functions.
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
fn test_fnv_hash() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = FNV_HASH(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "FNV hash should not be 0 for 'hello'"),
        other => panic!("Expected Integer for FNV_HASH, got {:?}", other),
    }
}

#[test]
fn test_base64_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "Hello, World!"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE encoded = BASE64_ENCODE(s) COMPUTE decoded = BASE64_DECODE(encoded) SELECT decoded;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("decoded"), Some(&Value::String("Hello, World!".to_string())));
}

#[test]
fn test_url_encode_decode() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello world & foo=bar"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE enc = URL_ENCODE(s) COMPUTE dec = URL_DECODE(enc) SELECT dec;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dec"), Some(&Value::String("hello world & foo=bar".to_string())));
}

#[test]
fn test_html_encode_decode() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "<script>alert('xss')</script>"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE enc = HTML_ENCODE(s) SELECT enc;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("enc") {
        Some(Value::String(s)) => {
            assert!(!s.contains('<'), "HTML encoded string should not contain <");
            assert!(s.contains("&lt;"), "HTML encoded string should contain &lt;");
        }
        other => panic!("Expected String for HTML_ENCODE, got {:?}", other),
    }
}

#[test]
fn test_rot13() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "Hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = ROT13(s) COMPUTE rr = ROT13(r) SELECT rr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // ROT13 twice = original
    assert_eq!(r.rows[0].data.get("rr"), Some(&Value::String("Hello".to_string())));
}
