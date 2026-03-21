/// Integration tests for PostgreSQL-compatible string functions added to PQL.
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

// ── CONCAT_WS ────────────────────────────────────────────────────────────────

#[test]
fn test_concat_ws_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "foo", "b": "bar", "c": "baz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONCAT_WS(", ", a, b, c) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("foo, bar, baz".to_string()))
    );
}

#[test]
fn test_concat_ws_skips_null() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "foo", "c": "baz"}),
    )
    .unwrap();
    // b is missing (null), should be skipped
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONCAT_WS("-", a, b, c) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("foo-baz".to_string()))
    );
}

#[test]
fn test_concat_ws_empty_args() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONCAT_WS() SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String(String::new())));
}

// ── OVERLAY ──────────────────────────────────────────────────────────────────

#[test]
fn test_overlay_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello world"}),
    )
    .unwrap();
    // OVERLAY("hello world", "there", 7, 5) -> "hello there"
    let mut p = Parser::new(r#"QUERY t COMPUTE r = OVERLAY(s, "there", 7, 5) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("hello there".to_string()))
    );
}

#[test]
fn test_overlay_no_length() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello world"}),
    )
    .unwrap();
    // Without length arg, default is len(replacement)
    let mut p = Parser::new(r#"QUERY t COMPUTE r = OVERLAY(s, "XY", 1, 2) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("XYllo world".to_string()))
    );
}

// ── CONVERT / CONVERT_FROM / CONVERT_TO ─────────────────────────────────────

#[test]
fn test_convert_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONVERT(s, "UTF8", "UTF8") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("hello".to_string()))
    );
}

#[test]
fn test_convert_from_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONVERT_FROM(s, "UTF8") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("world".to_string()))
    );
}

#[test]
fn test_convert_to_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "data"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONVERT_TO(s, "UTF8") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("data".to_string()))
    );
}

// ── ENCODE / DECODE ──────────────────────────────────────────────────────────

#[test]
fn test_encode_hex() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "AB"}),
    )
    .unwrap();
    // "AB" in hex is "4142"
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ENCODE(s, "hex") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("4142".to_string()))
    );
}

#[test]
fn test_encode_base64() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "Man"}),
    )
    .unwrap();
    // base64("Man") = "TWFu"
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ENCODE(s, "base64") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("TWFu".to_string()))
    );
}

#[test]
fn test_encode_escape() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    // Printable ASCII, escape should return unchanged
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ENCODE(s, "escape") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("hello".to_string()))
    );
}

// ── TO_ASCII ─────────────────────────────────────────────────────────────────

#[test]
fn test_to_ascii_from_int() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 65}),
    )
    .unwrap();
    // TO_ASCII(65) -> "A"
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_ASCII(n) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("A".to_string()))
    );
}

#[test]
fn test_to_ascii_from_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_ASCII(s) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("hello".to_string()))
    );
}

// ── REGEXP_SPLIT_TO_TABLE ────────────────────────────────────────────────────

#[test]
fn test_regexp_split_to_table() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "one1two2three"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE parts = REGEXP_SPLIT_TO_TABLE(s, "[0-9]") SELECT parts;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("parts") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::String("one".to_string()));
            assert_eq!(arr[1], Value::String("two".to_string()));
            assert_eq!(arr[2], Value::String("three".to_string()));
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

// ── SUBSTRING_INDEX ──────────────────────────────────────────────────────────

#[test]
fn test_substring_index_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "a.b.c.d"}),
    )
    .unwrap();
    // SUBSTRING_INDEX("a.b.c.d", ".", 2) -> "a.b"
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBSTRING_INDEX(s, ".", 2) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("a.b".to_string()))
    );
}

#[test]
fn test_substring_index_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "a.b.c.d"}),
    )
    .unwrap();
    // SUBSTRING_INDEX("a.b.c.d", ".", -2) -> "c.d"
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBSTRING_INDEX(s, ".", -2) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("c.d".to_string()))
    );
}

// ── OCTET_LENGTH ─────────────────────────────────────────────────────────────

#[test]
fn test_octet_length_ascii() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = OCTET_LENGTH(s) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(5)));
}

#[test]
fn test_octet_length_multibyte() {
    let (db, ex) = setup();
    // "é" is 2 bytes in UTF-8 but 1 character
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "café"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = OCTET_LENGTH(s) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // "café" = c(1) + a(1) + f(1) + é(2) = 5 bytes
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(5)));
}

// ── PARSE_IDENT ──────────────────────────────────────────────────────────────

#[test]
fn test_parse_ident_dotted() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "public.my_table"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PARSE_IDENT(s) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("public".to_string()));
            assert_eq!(arr[1], Value::String("my_table".to_string()));
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_parse_ident_quoted() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": r#""schema"."table""#}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PARSE_IDENT(s) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("schema".to_string()));
            assert_eq!(arr[1], Value::String("table".to_string()));
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

// ── FORMAT_TYPE ──────────────────────────────────────────────────────────────

#[test]
fn test_format_type_known_oids() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ti = FORMAT_TYPE(23, 0) COMPUTE tt = FORMAT_TYPE(25, 0) SELECT ti, tt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ti"),
        Some(&Value::String("integer".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("tt"),
        Some(&Value::String("text".to_string()))
    );
}

#[test]
fn test_format_type_unknown_oid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FORMAT_TYPE(99999, 0) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("type_99999".to_string()))
    );
}

// ── Integration: multiple new functions together ──────────────────────────────

#[test]
fn test_concat_ws_with_encode() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "foo", "b": "bar"}),
    )
    .unwrap();
    // Build a joined string, then encode it in hex
    let mut p = Parser::new(
        r#"QUERY t COMPUTE joined = CONCAT_WS(":", a, b) COMPUTE enc = ENCODE(joined, "hex") SELECT enc;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // "foo:bar" in hex = 666f6f3a626172
    assert_eq!(
        r.rows[0].data.get("enc"),
        Some(&Value::String("666f6f3a626172".to_string()))
    );
}

#[test]
fn test_overlay_with_substring_index() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"filepath": "/home/user/file.txt"}),
    )
    .unwrap();
    // Get first two path segments then get octet length
    let mut p = Parser::new(
        r#"QUERY t COMPUTE first2 = SUBSTRING_INDEX(filepath, "/", 3) COMPUTE r = OCTET_LENGTH(first2) SELECT first2, r;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("first2"),
        Some(&Value::String("/home/user".to_string()))
    );
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(10)));
}

#[test]
fn test_regexp_split_to_table_single_result() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "nodivider"}),
    )
    .unwrap();
    // No match -> single element array
    let mut p =
        Parser::new(r#"QUERY t COMPUTE parts = REGEXP_SPLIT_TO_TABLE(s, "[0-9]+") SELECT parts;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("parts") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 1),
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_parse_ident_single() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "mytable"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PARSE_IDENT(s) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], Value::String("mytable".to_string()));
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_to_ascii_various_codepoints() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n1": 97, "n2": 90, "n3": 48}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = TO_ASCII(n1) COMPUTE z = TO_ASCII(n2) COMPUTE zero = TO_ASCII(n3) SELECT a, z, zero;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("a"),
        Some(&Value::String("a".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("z"),
        Some(&Value::String("Z".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("zero"),
        Some(&Value::String("0".to_string()))
    );
}

#[test]
fn test_encode_base64_padding() {
    let (db, ex) = setup();
    // "M" -> "TQ==" (needs 2 padding chars)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "M"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ENCODE(s, "base64") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("TQ==".to_string()))
    );
}

#[test]
fn test_convert_all_null_on_missing() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // CONVERT with no args returns NULL
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ENCODE(missing_field, "hex") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
}

#[test]
fn test_overlay_start_at_end() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "abc"}),
    )
    .unwrap();
    // OVERLAY("abc", "XYZ", 4, 0) -> "abcXYZ" (start after end)
    let mut p = Parser::new(r#"QUERY t COMPUTE r = OVERLAY(s, "XYZ", 4, 0) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("abcXYZ".to_string()))
    );
}

#[test]
fn test_format_type_boolean_uuid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE b = FORMAT_TYPE(16, 0) COMPUTE u = FORMAT_TYPE(2950, 0) SELECT b, u;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("b"),
        Some(&Value::String("boolean".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("u"),
        Some(&Value::String("uuid".to_string()))
    );
}
