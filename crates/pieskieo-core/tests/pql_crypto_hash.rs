/// Integration tests for PQL cryptographic hash functions (pure Rust implementations).
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

// ── SHA-256 tests ────────────────────────────────────────────────────────────

#[test]
fn test_sha256_empty_string() {
    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA256(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "e3b0c44298fc1c149afbf4c8996fb924\
             27ae41e4649b934ca495991b7852b855"
                .to_string()
        ))
    );
}

#[test]
fn test_sha256_abc() {
    // SHA-256("abc") well-known prefix check.
    // NIST FIPS 180-4 Example 1: the first 8 hex chars (32 bits) are ba7816bf,
    // and the full digest is 64 hex chars (32 bytes).
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA256(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 64, "SHA-256 hex digest must be 64 chars");
            assert!(
                s.starts_with("ba7816bf"),
                "SHA-256('abc') must start with ba7816bf (NIST vector prefix)"
            );
            assert!(
                s.chars().all(|c| c.is_ascii_hexdigit()),
                "SHA-256 output must be lowercase hex"
            );
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_sha256_hello_world() {
    // SHA-256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576f25d1cf9b0c2a7e6
    // Correct: b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576f25d1cf9b0c2a7e6 is 63 chars
    // Actual SHA-256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576f25d1cf9b0c2a7e6
    // Let's use "hello" which is well known:
    // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA256(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824".to_string()
        ))
    );
}

#[test]
fn test_sha256_alias_sha_256() {
    // SHA_256 and SHA2_256 should produce the same result as SHA256
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "test"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = SHA256(s) COMPUTE b = SHA_256(s) COMPUTE c = SHA2_256(s) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = r.rows[0].data.get("a").cloned();
    let b = r.rows[0].data.get("b").cloned();
    let c = r.rows[0].data.get("c").cloned();
    assert_eq!(a, b, "SHA256 and SHA_256 must produce identical results");
    assert_eq!(a, c, "SHA256 and SHA2_256 must produce identical results");
}

// ── SHA-512 tests ────────────────────────────────────────────────────────────

#[test]
fn test_sha512_empty_string() {
    // SHA-512("") = cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e
    // Full 128-char hex:
    // cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce
    // 47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA512(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 128, "SHA-512 hex digest must be 128 chars");
            assert_eq!(
                s,
                "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
                 47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
                "SHA-512('') known vector mismatch"
            );
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

// ── SHA-1 tests ──────────────────────────────────────────────────────────────

#[test]
fn test_sha1_empty_string() {
    // SHA-1("") = da39a3ee5e6b4b0d3255bfef95601890afd80709
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA1(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "da39a3ee5e6b4b0d3255bfef95601890afd80709".to_string()
        ))
    );
}

#[test]
fn test_sha1_abc() {
    // SHA-1("abc") = a9993e364706816aba3e25717850c26c9cd0d89d
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA1(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "a9993e364706816aba3e25717850c26c9cd0d89d".to_string()
        ))
    );
}

#[test]
fn test_sha1_alias_sha_1() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "pieskieo"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE a = SHA1(s) COMPUTE b = SHA_1(s) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "SHA1 and SHA_1 must match"
    );
}

// ── MD5 tests ────────────────────────────────────────────────────────────────

#[test]
fn test_md5_empty_string() {
    // MD5("") = d41d8cd98f00b204e9800998ecf8427e
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = MD5(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "d41d8cd98f00b204e9800998ecf8427e".to_string()
        ))
    );
}

#[test]
fn test_md5_hello() {
    // MD5("hello") = 5d41402abc4b2a76b9719d911017c592
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = MD5(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "5d41402abc4b2a76b9719d911017c592".to_string()
        ))
    );
}

#[test]
fn test_md5_alias_md5_hash() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "test123"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE a = MD5(s) COMPUTE b = MD5_HASH(s) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "MD5 and MD5_HASH must match"
    );
}

// ── HMAC-SHA256 tests ────────────────────────────────────────────────────────

#[test]
fn test_hmac_sha256_basic() {
    // HMAC-SHA256 with known test vector:
    // key = "key", message = "The quick brown fox jumps over the lazy dog"
    // Expected: f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"msg": "The quick brown fox jumps over the lazy dog", "k": "key"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = HMAC_SHA256(msg, k) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String(
            "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8".to_string()
        ))
    );
}

#[test]
fn test_hmac256_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"msg": "hello", "k": "secret"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = HMAC_SHA256(msg, k) COMPUTE b = HMAC256(msg, k) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "HMAC_SHA256 and HMAC256 must match"
    );
    match r.rows[0].data.get("a") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 64, "HMAC-SHA256 result must be 64 hex chars")
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

// ── SHA256_TRUNCATED tests ───────────────────────────────────────────────────

#[test]
fn test_sha256_truncated_default_length() {
    // SHA256_TRUNCATED("hello") should return first 8 hex chars of SHA-256("hello")
    // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    // First 8 chars: 2cf24dba
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA256_TRUNCATED(s) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String("2cf24dba".to_string()))
    );
}

#[test]
fn test_sha256_truncated_custom_length() {
    // SHA256_TRUNCATED("hello", 16) -> first 16 hex chars
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = SHA256_TRUNCATED(s, 16) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::String("2cf24dba5fb0a30e".to_string()))
    );
}

#[test]
fn test_sha256_short_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = SHA256_TRUNCATED(s, 8) COMPUTE b = SHA256_SHORT(s) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "SHA256_TRUNCATED and SHA256_SHORT must match for default length"
    );
}

// ── CONTENT_HASH tests ───────────────────────────────────────────────────────

#[test]
fn test_content_hash_returns_64_hex_chars() {
    // CONTENT_HASH of any value should return a 64-char hex string
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"val": "some content"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = CONTENT_HASH(val) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 64, "CONTENT_HASH must return 64 hex chars");
            assert!(
                s.chars().all(|c| c.is_ascii_hexdigit()),
                "CONTENT_HASH must be hex"
            );
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_content_fingerprint_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"val": 42}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = CONTENT_HASH(val) COMPUTE b = CONTENT_FINGERPRINT(val) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "CONTENT_HASH and CONTENT_FINGERPRINT must match"
    );
}

#[test]
fn test_content_hash_deterministic() {
    // The same value should always produce the same hash
    let (db, ex) = setup();
    let id = Uuid::new_v4();
    db.put_doc_ns(
        None,
        Some("t"),
        id,
        serde_json::json!({"val": "deterministic"}),
    )
    .unwrap();
    let mut p1 = Parser::new(r#"QUERY t COMPUTE h = CONTENT_HASH(val) SELECT h;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let mut p2 = Parser::new(r#"QUERY t COMPUTE h = CONTENT_HASH(val) SELECT h;"#);
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(
        r1.rows[0].data.get("h"),
        r2.rows[0].data.get("h"),
        "CONTENT_HASH must be deterministic"
    );
}

// ── HASH_PASSWORD / VERIFY_HASH tests ────────────────────────────────────────

#[test]
fn test_hash_password_returns_64_hex_chars() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "mysecret", "salt": "randomsalt"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = HASH_PASSWORD(pw, salt) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 64, "HASH_PASSWORD must return 64 hex chars");
            assert!(
                s.chars().all(|c| c.is_ascii_hexdigit()),
                "HASH_PASSWORD must be hex"
            );
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_hash_password_simple_kdf_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "pass", "salt": "nacl"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = HASH_PASSWORD(pw, salt) COMPUTE b = SIMPLE_KDF(pw, salt) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "HASH_PASSWORD and SIMPLE_KDF must match"
    );
}

#[test]
fn test_verify_hash_correct_password() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "correct_horse_battery", "salt": "stapleSalt"}),
    )
    .unwrap();
    // First compute the hash, then verify it
    let mut p1 = Parser::new(r#"QUERY t COMPUTE h = HASH_PASSWORD(pw, salt) SELECT h;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let hash = match r1.rows[0].data.get("h") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("Expected String, got {:?}", other),
    };

    // Now verify using the literal hash
    let query = format!(
        r#"QUERY t COMPUTE ok = VERIFY_HASH(pw, salt, "{}") SELECT ok;"#,
        hash
    );
    let mut p2 = Parser::new(&query);
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(r2.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_verify_hash_wrong_password() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "wrongpassword", "salt": "somesalt"}),
    )
    .unwrap();
    // Use a hash that definitely won't match
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ok = VERIFY_HASH(pw, salt, "0000000000000000000000000000000000000000000000000000000000000000") SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_hash_verify_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "password", "salt": "salt"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = VERIFY_HASH(pw, salt, "badHash") COMPUTE b = HASH_VERIFY(pw, salt, "badHash") SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        r.rows[0].data.get("b"),
        "VERIFY_HASH and HASH_VERIFY must match"
    );
}
