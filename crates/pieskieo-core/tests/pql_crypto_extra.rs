/// Integration tests for additional PQL hash, checksum, and encoding functions.
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

// ── CRC32 tests ───────────────────────────────────────────────────────────────

#[test]
fn test_crc32_hello_known_value() {
    // CRC32("hello") = 907060870 (standard CRC32 / IEEE 802.3)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CRC32(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(907060870)));
}

#[test]
fn test_crc32_empty_string_is_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": ""})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CRC32(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(0)));
}

#[test]
fn test_crc32_hash_alias() {
    // CRC32_HASH is an alias for CRC32 and must produce the same result
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = CRC32(sv) COMPUTE b = CRC32_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "CRC32 and CRC32_HASH must produce identical results");
}

// ── CRC16 tests ───────────────────────────────────────────────────────────────

#[test]
fn test_crc16_hello_known_value() {
    // CRC16-CCITT / XMODEM of "hello" = 53870
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = CRC16(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(53870)));
}

#[test]
fn test_crc16_hash_alias() {
    // CRC16_HASH must produce the same result as CRC16
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "world"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = CRC16(sv) COMPUTE b = CRC16_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "CRC16 and CRC16_HASH must produce identical results");
}

// ── ADLER32 tests ─────────────────────────────────────────────────────────────

#[test]
fn test_adler32_hello_known_value() {
    // ADLER32("hello") = 103547413
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = ADLER32(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(103547413)));
}

#[test]
fn test_adler_hash_alias() {
    // ADLER_HASH is an alias for ADLER32
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = ADLER32(sv) COMPUTE b = ADLER_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "ADLER32 and ADLER_HASH must produce identical results");
}

// ── FNV1A tests ───────────────────────────────────────────────────────────────

#[test]
fn test_fnv1a_hello_is_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = FNV1A(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "FNV1A('hello') must not be 0"),
        other => panic!("expected Integer for FNV1A, got {:?}", other),
    }
}

#[test]
fn test_fnv1a_hash_alias() {
    // FNV1A_HASH is an alias for FNV1A
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = FNV1A(sv) COMPUTE b = FNV1A_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "FNV1A and FNV1A_HASH must produce identical results");
}

#[test]
fn test_fnv1a_different_inputs_differ() {
    // FNV1A("") and FNV1A("a") must differ
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "", "sv2": "a"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = FNV1A(sv) COMPUTE b = FNV1A(sv2) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_ne!(a, b, "FNV1A('') and FNV1A('a') must differ");
}

// ── DJBX33A tests ─────────────────────────────────────────────────────────────

#[test]
fn test_djbx33a_hello_is_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DJBX33A(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "DJBX33A('hello') must not be 0"),
        other => panic!("expected Integer for DJBX33A, got {:?}", other),
    }
}

#[test]
fn test_djb2_hash_alias_equals_djbx33a() {
    // DJB2_HASH and DJBX33A must produce the same result
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = DJBX33A(sv) COMPUTE b = DJB2_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "DJBX33A and DJB2_HASH must produce identical results");
}

#[test]
fn test_djbx33a_empty_vs_hello_differ() {
    // DJBX33A("") != DJBX33A("hello")
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "", "sv2": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = DJBX33A(sv) COMPUTE b = DJBX33A(sv2) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_ne!(a, b, "DJBX33A('') and DJBX33A('hello') must differ");
}

// ── MURMUR3 tests ─────────────────────────────────────────────────────────────

#[test]
fn test_murmur3_hello_is_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = MURMUR3(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer for MURMUR3, got {:?}", other),
    }
}

#[test]
fn test_murmur3_hash_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = MURMUR3(sv) COMPUTE b = MURMUR3_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "MURMUR3 and MURMUR3_HASH must produce identical results");
}

// ── JENKINS_HASH tests ────────────────────────────────────────────────────────

#[test]
fn test_jenkins_hash_hello_known_value() {
    // Jenkins OAT("hello") = 3372029979 (as i64 = 3372029979)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = JENKINS_HASH(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(3372029979)));
}

#[test]
fn test_jenkins_oat_alias() {
    // JENKINS_OAT must produce the same result as JENKINS_HASH
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = JENKINS_HASH(sv) COMPUTE b = JENKINS_OAT(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "JENKINS_HASH and JENKINS_OAT must produce identical results");
}

// ── POLYNOMIAL_HASH tests ─────────────────────────────────────────────────────

#[test]
fn test_polynomial_hash_hello_default_args() {
    // POLYNOMIAL_HASH("hello", 31, 1000000007) = 99162322
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = POLYNOMIAL_HASH(sv, 31, 1000000007) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(99162322)));
}

#[test]
fn test_poly_hash_alias() {
    // POLY_HASH must produce the same result as POLYNOMIAL_HASH
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = POLYNOMIAL_HASH(sv) COMPUTE b = POLY_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "POLYNOMIAL_HASH and POLY_HASH must produce identical results");
}

#[test]
fn test_polynomial_hash_explicit_equals_default() {
    // POLYNOMIAL_HASH("hello", 31, 1000000007) == POLYNOMIAL_HASH("hello") with defaults
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = POLYNOMIAL_HASH(sv) COMPUTE b = POLYNOMIAL_HASH(sv, 31, 1000000007) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "POLYNOMIAL_HASH default args must match explicit args");
}

// ── WYHASH tests ──────────────────────────────────────────────────────────────

#[test]
fn test_wyhash_hello_is_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = WYHASH(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer for WYHASH('hello'), got {:?}", other),
    }
}

#[test]
fn test_wy_hash_alias() {
    // WY_HASH must produce the same result as WYHASH
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = WYHASH(sv) COMPUTE b = WY_HASH(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "WYHASH and WY_HASH must produce identical results");
}

#[test]
fn test_wyhash_empty_vs_hello_differ() {
    // WYHASH("") != WYHASH("hello")
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "", "sv2": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = WYHASH(sv) COMPUTE b = WYHASH(sv2) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_ne!(a, b, "WYHASH('') and WYHASH('hello') must differ");
}

// ── ISBN_CHECK tests ──────────────────────────────────────────────────────────

#[test]
fn test_isbn_check_valid_isbn10() {
    // "0306406152" is a valid ISBN-10
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "0306406152"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = ISBN_CHECK(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_isbn_check_invalid_isbn10() {
    // "0306406153" is NOT a valid ISBN-10 (last digit changed)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "0306406153"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = ISBN_CHECK(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_valid_isbn_alias() {
    // IS_VALID_ISBN is an alias for ISBN_CHECK
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "0306406152"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = ISBN_CHECK(sv) COMPUTE b = IS_VALID_ISBN(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "ISBN_CHECK and IS_VALID_ISBN must produce identical results");
}

#[test]
fn test_isbn_check_valid_isbn13() {
    // "9780306406157" is a valid ISBN-13
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "9780306406157"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = ISBN_CHECK(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

// ── VERHOEFF tests ────────────────────────────────────────────────────────────

#[test]
fn test_verhoeff_valid_2363() {
    // "2363" is a valid Verhoeff number
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "2363"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = VERHOEFF(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_verhoeff_invalid_2364() {
    // "2364" is NOT a valid Verhoeff number
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "2364"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = VERHOEFF(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_verhoeff_check_alias() {
    // VERHOEFF_CHECK is an alias for VERHOEFF
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "2363"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = VERHOEFF(sv) COMPUTE b = VERHOEFF_CHECK(sv) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "VERHOEFF and VERHOEFF_CHECK must produce identical results");
}

// ── HASH_COMBINE / COMBINE_HASHES tests ───────────────────────────────────────

#[test]
fn test_hash_combine_returns_integer() {
    // HASH_COMBINE(1234, 5678) must return an integer
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1234, "nv2": 5678})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE res = HASH_COMBINE(nv, nv2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer for HASH_COMBINE, got {:?}", other),
    }
}

#[test]
fn test_combine_hashes_alias() {
    // COMBINE_HASHES must produce the same result as HASH_COMBINE
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1234, "nv2": 5678})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = HASH_COMBINE(nv, nv2) COMPUTE b = COMBINE_HASHES(nv, nv2) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_eq!(a, b, "HASH_COMBINE and COMBINE_HASHES must produce identical results");
}

// ── Additional coverage tests ─────────────────────────────────────────────────

#[test]
fn test_crc32_different_strings_differ() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "foo", "sv2": "bar"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = CRC32(sv) COMPUTE b = CRC32(sv2) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_ne!(a, b, "CRC32('foo') and CRC32('bar') must differ");
}

#[test]
fn test_crc16_different_strings_differ() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "foo", "sv2": "bar"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = CRC16(sv) COMPUTE b = CRC16(sv2) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    assert_ne!(a, b, "CRC16('foo') and CRC16('bar') must differ");
}

#[test]
fn test_jenkins_hash_nonzero_for_nonempty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "test"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = JENKINS_HASH(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "JENKINS_HASH('test') must not be 0"),
        other => panic!("expected Integer for JENKINS_HASH, got {:?}", other),
    }
}

#[test]
fn test_polynomial_hash_nonempty_nonzero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "hello"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = POLYNOMIAL_HASH(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "POLYNOMIAL_HASH('hello') must not be 0"),
        other => panic!("expected Integer for POLYNOMIAL_HASH, got {:?}", other),
    }
}

#[test]
fn test_isbn_check_invalid_short_string() {
    // A short string that is not 10 or 13 digits should return false
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "12345"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = ISBN_CHECK(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_adler32_nonempty_nonzero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"sv": "test"})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = ADLER32(sv) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nv") {
        Some(Value::Integer(v)) => assert_ne!(*v, 0, "ADLER32('test') must not be 0"),
        other => panic!("expected Integer for ADLER32, got {:?}", other),
    }
}
