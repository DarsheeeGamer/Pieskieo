/// Integration tests for PQL hash/checksum functions (hash2).
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
fn test_murmur3() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = MURMUR3(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer for MURMUR3, got {:?}", other),
    }
}

#[test]
fn test_murmur3_hash_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "world"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = MURMUR3_HASH(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer for MURMUR3_HASH, got {:?}", other),
    }
}

#[test]
fn test_fnv1a() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = FNV1A(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "FNV1A should not be 0 for 'hello'"),
        other => panic!("expected Integer for FNV1A, got {:?}", other),
    }
}

#[test]
fn test_djbhash() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = DJBHASH(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "DJBHASH should not be 0 for 'hello'"),
        other => panic!("expected Integer for DJBHASH, got {:?}", other),
    }
}

#[test]
fn test_adler32() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "Wikipedia"}),
    )
    .unwrap();
    // Known Adler-32 of "Wikipedia" is 0x11E60398 = 300286872
    let mut p = Parser::new("QUERY t COMPUTE h = ADLER32(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => assert_eq!(*v, 300286872, "ADLER32('Wikipedia') = 300286872"),
        other => panic!("expected Integer for ADLER32, got {:?}", other),
    }
}

#[test]
fn test_crc32() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = CRC32(s) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "CRC32 should not be 0 for 'hello'"),
        other => panic!("expected Integer for CRC32, got {:?}", other),
    }
}

#[test]
fn test_hash_combine() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 12345, "b": 67890}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = HASH_COMBINE(a, b) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer for HASH_COMBINE, got {:?}", other),
    }
}

#[test]
fn test_consistent_hash() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"k": 42, "n": 10}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = CONSISTENT_HASH(k, n) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Integer(v)) => {
            assert!(*v >= 0 && *v < 10, "bucket should be in [0, 10), got {}", v)
        }
        other => panic!("expected Integer for CONSISTENT_HASH, got {:?}", other),
    }
}
