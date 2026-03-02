/// Integration tests for new PQL data encoding, compression analysis, and parsing functions.
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

// ── BASE64_URL_ENCODE / TO_BASE64_URL ─────────────────────────────────────────

#[test]
fn test_base64_url_encode_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_URL_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // URL-safe base64 of "hello" (no padding): aGVsbG8
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("aGVsbG8".to_string())));
}

#[test]
fn test_to_base64_url_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_BASE64_URL(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("aGVsbG8".to_string())));
}

#[test]
fn test_base64_url_encode_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": ""})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_URL_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("".to_string())));
}

// ── BASE64_URL_DECODE / FROM_BASE64_URL ───────────────────────────────────────

#[test]
fn test_base64_url_decode_basic() {
    let (db, ex) = setup();
    // URL-safe base64 of "hello" (no padding): aGVsbG8
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "aGVsbG8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_URL_DECODE(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("hello".to_string())));
}

#[test]
fn test_from_base64_url_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "aGVsbG8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FROM_BASE64_URL(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("hello".to_string())));
}

#[test]
fn test_base64_url_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE enc = BASE64_URL_ENCODE(msg) COMPUTE dec = BASE64_URL_DECODE(enc) SELECT dec;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("dec"), Some(&Value::String("hello world".to_string())));
}

// ── BASE32_ENCODE / TO_BASE32 ─────────────────────────────────────────────────

#[test]
fn test_base32_encode_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE32_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // base32 of "hello" = "NBSWY3DPEB3W64TMMQ======"
    // Actually RFC 4648: "hello" -> NBSWY3DP
    match result.rows[0].data.get("r") {
        Some(Value::String(s)) => {
            assert!(s.starts_with("NBSWY3DP"), "BASE32_ENCODE(hello) should start with NBSWY3DP, got {}", s);
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_to_base32_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "A"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_BASE32(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // base32 of "A" = "IE======" (A = 0x41, top 5 bits = 01000 = 8 = I, next 3+2 bits...)
    // "A" = 65 = 0b01000001
    // 5 bits: 01000 = 8 = 'I', 3 bits: 001 padded to 00100 = 4 = 'E'
    // Padded to 8 chars: IE======
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("IE======".to_string())));
}

#[test]
fn test_base32_encode_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": ""})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE32_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("".to_string())));
}

// ── BASE32_DECODE / FROM_BASE32 ───────────────────────────────────────────────

#[test]
fn test_base32_decode_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "NBSWY3DP"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE32_DECODE(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("hello".to_string())));
}

#[test]
fn test_from_base32_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "NBSWY3DP"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FROM_BASE32(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("hello".to_string())));
}

#[test]
fn test_base32_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE enc = BASE32_ENCODE(msg) COMPUTE dec = BASE32_DECODE(enc) SELECT dec;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("dec"), Some(&Value::String("hello".to_string())));
}

// ── BASE16_ENCODE / TO_BASE16 ─────────────────────────────────────────────────

#[test]
fn test_base16_encode_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hi"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE16_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // "hi" = 0x68 0x69 -> "6869"
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("6869".to_string())));
}

#[test]
fn test_to_base16_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "AB"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_BASE16(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // "AB" = 0x41 0x42 -> "4142"
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("4142".to_string())));
}

#[test]
fn test_base16_encode_uppercase() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "\u{00FF}"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE16_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // U+00FF in UTF-8 is 0xC3 0xBF -> "C3BF"
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("C3BF".to_string())));
}

// ── BASE58_ENCODE / TO_BASE58 ─────────────────────────────────────────────────

#[test]
fn test_base58_encode_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE58_ENCODE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // base58 of "hello" = "Cn8eVZg"
    match result.rows[0].data.get("r") {
        Some(Value::String(s)) => {
            assert!(!s.is_empty(), "BASE58_ENCODE should return non-empty string");
            // Verify it only contains base58 characters
            let b58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
            assert!(s.chars().all(|c| b58_chars.contains(c)), "Result should only contain base58 chars: {}", s);
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_to_base58_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hi"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_BASE58(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::String(s)) => {
            assert!(!s.is_empty(), "TO_BASE58 should return non-empty string");
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

// ── BASE58_DECODE / FROM_BASE58 ───────────────────────────────────────────────

#[test]
fn test_base58_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE enc = BASE58_ENCODE(msg) COMPUTE dec = BASE58_DECODE(enc) SELECT dec;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("dec"), Some(&Value::String("hello".to_string())));
}

#[test]
fn test_from_base58_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE enc = TO_BASE58(msg) COMPUTE dec = FROM_BASE58(enc) SELECT dec;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("dec"), Some(&Value::String("world".to_string())));
}

// ── URL_ENCODE_COMPONENT / ENCODE_URI_COMPONENT ───────────────────────────────

#[test]
fn test_url_encode_component_space() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_ENCODE_COMPONENT(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("hello%20world".to_string())));
}

#[test]
fn test_encode_uri_component_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "a=1&b=2"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ENCODE_URI_COMPONENT(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("a%3D1%26b%3D2".to_string())));
}

#[test]
fn test_url_encode_component_safe_chars() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "abc-123_ok.~"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_ENCODE_COMPONENT(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("abc-123_ok.~".to_string())));
}

// ── XML_ESCAPE / ESCAPE_XML ───────────────────────────────────────────────────

#[test]
fn test_xml_escape_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "<tag attr=\"val\">"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = XML_ESCAPE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("&lt;tag attr=&quot;val&quot;&gt;".to_string())));
}

#[test]
fn test_escape_xml_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "a & b"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ESCAPE_XML(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("a &amp; b".to_string())));
}

#[test]
fn test_xml_escape_apostrophe() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "it's"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = XML_ESCAPE(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("it&apos;s".to_string())));
}

// ── IS_BASE64 / IS_VALID_BASE64 ───────────────────────────────────────────────

#[test]
fn test_is_base64_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "aGVsbG8="})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_BASE64(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_base64_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "aGVsbG8="})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_BASE64(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_base64_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "not_base64!"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_BASE64(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── IS_HEX_STRING / IS_HEX ───────────────────────────────────────────────────

#[test]
fn test_is_hex_string_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"h": "deadbeef"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_HEX_STRING(h) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_hex_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"h": "ABCDEF0123456789"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_HEX(h) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_hex_string_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"h": "xyz"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_HEX_STRING(h) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── IS_ASCII / IS_ASCII_ONLY ──────────────────────────────────────────────────

#[test]
fn test_is_ascii_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_ASCII(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_ascii_only_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "abc123"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_ASCII_ONLY(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_ascii_false_for_utf8() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "caf\u{00E9}"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_ASCII(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── IS_NUMERIC_STRING ─────────────────────────────────────────────────────────

#[test]
fn test_is_numeric_string_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "42"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_NUMERIC_STRING(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_numeric_string_float() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "3.14"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_NUMERIC_STRING(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_numeric_string_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "-99.5"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_NUMERIC_STRING(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_numeric_string_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_NUMERIC_STRING(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── IS_INTEGER_STRING / IS_INT_STR ────────────────────────────────────────────

#[test]
fn test_is_integer_string_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "42"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_INTEGER_STRING(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_int_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "-7"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_INT_STR(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_integer_string_false_for_float() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": "3.14"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_INTEGER_STRING(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── CHAR_CODE / CHAR_TO_CODE ──────────────────────────────────────────────────

#[test]
fn test_char_code_a() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": "A"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CHAR_CODE(c) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(65)));
}

#[test]
fn test_char_to_code_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": "a"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CHAR_TO_CODE(c) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(97)));
}

#[test]
fn test_char_code_from_longer_string_uses_first_char() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CHAR_CODE(c) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // 'h' = 104
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(104)));
}

// ── CODE_TO_CHAR / FROM_CHAR_CODE ─────────────────────────────────────────────

#[test]
fn test_code_to_char_65() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 65})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CODE_TO_CHAR(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("A".to_string())));
}

#[test]
fn test_from_char_code_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 97})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FROM_CHAR_CODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("a".to_string())));
}

#[test]
fn test_char_code_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": "Z"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE code = CHAR_CODE(c) COMPUTE ch = CODE_TO_CHAR(code) SELECT ch;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("ch"), Some(&Value::String("Z".to_string())));
}

// ── STRING_TO_BYTES / STR_TO_BYTES ────────────────────────────────────────────

#[test]
fn test_string_to_bytes_hi() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hi"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = STRING_TO_BYTES(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // "hi" = [104, 105]
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Array(vec![
        Value::Integer(104), Value::Integer(105),
    ])));
}

#[test]
fn test_str_to_bytes_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "AB"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = STR_TO_BYTES(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // "AB" = [65, 66]
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Array(vec![
        Value::Integer(65), Value::Integer(66),
    ])));
}

// ── BYTES_TO_STRING / BYTES_TO_STR ───────────────────────────────────────────

#[test]
fn test_bytes_to_string_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"b": [72, 101, 108, 108, 111]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BYTES_TO_STRING(b) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("Hello".to_string())));
}

#[test]
fn test_bytes_to_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"b": [65, 66]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BYTES_TO_STR(b) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("AB".to_string())));
}

#[test]
fn test_string_bytes_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "Hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bts = STRING_TO_BYTES(msg) COMPUTE back = BYTES_TO_STRING(bts) SELECT back;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("back"), Some(&Value::String("Hello".to_string())));
}

// ── RLE_ENCODE_ARRAY / RLE_ARR ────────────────────────────────────────────────

#[test]
fn test_rle_encode_array_string() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "AAABBBCC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_ENCODE_ARRAY(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("3A3B2C".to_string())));
}

#[test]
fn test_rle_arr_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "AABCC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_ARR(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("2AB2C".to_string())));
}

#[test]
fn test_rle_encode_array_no_runs() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "ABCD"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_ENCODE_ARRAY(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // Single characters should not have count prefix
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("ABCD".to_string())));
}

// ── RLE_DECODE_STRING / RLD_STR ───────────────────────────────────────────────

#[test]
fn test_rle_decode_string_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "3A3B2C"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_DECODE_STRING(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("AAABBBCC".to_string())));
}

#[test]
fn test_rld_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"encoded": "2AB2C"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLD_STR(encoded) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("AABCC".to_string())));
}

#[test]
fn test_rle_string_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "AAABBBCC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE enc = RLE_ENCODE_ARRAY(msg) COMPUTE dec = RLE_DECODE_STRING(enc) SELECT dec;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("dec"), Some(&Value::String("AAABBBCC".to_string())));
}

// ── COMPRESS_RATIO / COMPRESSION_RATIO ───────────────────────────────────────

#[test]
fn test_compress_ratio_repeated_chars() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "AAAAAAAAAA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = COMPRESS_RATIO(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // "AAAAAAAAAA" (10 chars) -> "10A" (3 chars) = ratio 0.3
    match result.rows[0].data.get("r") {
        Some(Value::Float(f)) => {
            assert!(*f < 1.0, "Highly repeated string should have ratio < 1.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_compression_ratio_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "ABCDE"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = COMPRESSION_RATIO(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // "ABCDE" (5 chars) -> "ABCDE" (5 chars) = ratio 1.0
    match result.rows[0].data.get("r") {
        Some(Value::Float(_)) => {}
        other => panic!("Expected Float, got {:?}", other),
    }
}

// ── DETECT_ENCODING / GUESS_ENCODING ─────────────────────────────────────────

#[test]
fn test_detect_encoding_numeric() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "3.14"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DETECT_ENCODING(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("numeric".to_string())));
}

#[test]
fn test_guess_encoding_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = GUESS_ENCODING(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("ascii".to_string())));
}

#[test]
fn test_detect_encoding_hex() {
    let (db, ex) = setup();
    // 8 hex chars that don't match base64 pattern in edge cases
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"msg": "deadbeef"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DETECT_ENCODING(msg) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("hex".to_string())));
}

// ── IS_JSON_STRING / IS_VALID_JSON ────────────────────────────────────────────

#[test]
fn test_is_json_string_object() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"payload": "{\"key\":\"val\"}"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_JSON_STRING(payload) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_json_alias_array() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"payload": "[1,2,3]"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_JSON(payload) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_json_string_false_for_plain_string() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"payload": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_JSON_STRING(payload) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── IS_CSV_ROW / IS_CSV ───────────────────────────────────────────────────────

#[test]
fn test_is_csv_row_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"line": "a,b,c"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_CSV_ROW(line) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_csv_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"line": "a,b,c"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_CSV(line) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_csv_row_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"line": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_CSV_ROW(line) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// ── PARSE_CSV_ROW ─────────────────────────────────────────────────────────────

#[test]
fn test_parse_csv_row_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"line": "a,b,c"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PARSE_CSV_ROW(line) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Array(vec![
        Value::String("a".to_string()),
        Value::String("b".to_string()),
        Value::String("c".to_string()),
    ])));
}

// ── CSV_QUOTE_FIELD / QUOTE_CSV ───────────────────────────────────────────────

#[test]
fn test_csv_quote_field_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"fld": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CSV_QUOTE_FIELD(fld) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("\"hello world\"".to_string())));
}

#[test]
fn test_quote_csv_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"fld": "say \"hi\""})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = QUOTE_CSV(fld) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // Quotes should be doubled
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::String("\"say \"\"hi\"\"\"".to_string())));
}

// ── PARSE_TSV_ROW ─────────────────────────────────────────────────────────────

#[test]
fn test_parse_tsv_row_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"line": "a\tb\tc"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PARSE_TSV_ROW(line) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Array(vec![
        Value::String("a".to_string()),
        Value::String("b".to_string()),
        Value::String("c".to_string()),
    ])));
}
