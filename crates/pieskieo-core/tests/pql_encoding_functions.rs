/// Integration tests for PQL text encoding, compression, and data serialization functions.
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

// ── BASE64_ENCODE / TO_BASE64 ─────────────────────────────────────────────────

#[test]
fn test_base64_encode_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("aGVsbG8=".to_string()))
    );
}

#[test]
fn test_to_base64_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = TO_BASE64(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("aGVsbG8=".to_string()))
    );
}

#[test]
fn test_base64_encode_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("".to_string()))
    );
}

#[test]
fn test_base64_encode_world() {
    // "world" -> "d29ybGQ="
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("d29ybGQ=".to_string()))
    );
}

// ── BASE64_DECODE / FROM_BASE64 ───────────────────────────────────────────────

#[test]
fn test_base64_decode_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "aGVsbG8="}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BASE64_DECODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("hello".to_string()))
    );
}

#[test]
fn test_from_base64_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "aGVsbG8="}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FROM_BASE64(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("hello".to_string()))
    );
}

#[test]
fn test_base64_roundtrip() {
    // Encode then decode should return the original string
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE enc = BASE64_ENCODE(s) COMPUTE dec = BASE64_DECODE(enc) SELECT dec;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("dec"),
        Some(&Value::String("hello".to_string()))
    );
}

// ── URL_ENCODE / PERCENT_ENCODE ───────────────────────────────────────────────

#[test]
fn test_url_encode_space() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("hello%20world".to_string()))
    );
}

#[test]
fn test_percent_encode_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PERCENT_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("hello%20world".to_string()))
    );
}

#[test]
fn test_url_encode_special_chars() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "a=1&b=2"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("a%3D1%26b%3D2".to_string()))
    );
}

#[test]
fn test_url_encode_safe_chars() {
    // Letters, digits, -, _, ., ~ are not encoded
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "abc-123_test.ok~"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("abc-123_test.ok~".to_string()))
    );
}

// ── URL_DECODE / PERCENT_DECODE ───────────────────────────────────────────────

#[test]
fn test_url_decode_space() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello%20world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_DECODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("hello world".to_string()))
    );
}

#[test]
fn test_percent_decode_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello%20world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PERCENT_DECODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("hello world".to_string()))
    );
}

#[test]
fn test_url_encode_decode_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "foo bar+baz"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE enc = URL_ENCODE(s) COMPUTE dec = URL_DECODE(enc) SELECT dec;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("dec"),
        Some(&Value::String("foo bar+baz".to_string()))
    );
}

// ── HTML_ENCODE / HTML_ESCAPE ─────────────────────────────────────────────────

#[test]
fn test_html_encode_angle_brackets() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "<b>"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HTML_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("&lt;b&gt;".to_string()))
    );
}

#[test]
fn test_html_escape_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "<b>"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HTML_ESCAPE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("&lt;b&gt;".to_string()))
    );
}

#[test]
fn test_html_encode_ampersand_and_quote() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "a & \"b\""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HTML_ENCODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("a &amp; &quot;b&quot;".to_string()))
    );
}

// ── HTML_DECODE / HTML_UNESCAPE ───────────────────────────────────────────────

#[test]
fn test_html_decode_angle_brackets() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "&lt;b&gt;"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HTML_DECODE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("<b>".to_string()))
    );
}

#[test]
fn test_html_unescape_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "&lt;b&gt;"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HTML_UNESCAPE(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("<b>".to_string()))
    );
}

#[test]
fn test_html_encode_decode_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "<div class=\"x\">a & b</div>"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE enc = HTML_ENCODE(s) COMPUTE dec = HTML_DECODE(enc) SELECT dec;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("dec"),
        Some(&Value::String("<div class=\"x\">a & b</div>".to_string()))
    );
}

// ── RLE_ENCODE / RUN_LENGTH_ENCODE ───────────────────────────────────────────

#[test]
fn test_rle_encode_integers() {
    // [1,1,1,2,2] -> [[1,3],[2,2]]
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1,1,1,2,2]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_ENCODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Array(pairs)) => {
            assert_eq!(pairs.len(), 2, "Expected 2 RLE pairs for [1,1,1,2,2]");
            match &pairs[0] {
                Value::Array(p) => {
                    assert_eq!(p[0], Value::Integer(1));
                    assert_eq!(p[1], Value::Integer(3));
                }
                other => panic!("Expected Array pair, got {:?}", other),
            }
            match &pairs[1] {
                Value::Array(p) => {
                    assert_eq!(p[0], Value::Integer(2));
                    assert_eq!(p[1], Value::Integer(2));
                }
                other => panic!("Expected Array pair, got {:?}", other),
            }
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_run_length_encode_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5,5,5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RUN_LENGTH_ENCODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Array(pairs)) => {
            assert_eq!(pairs.len(), 1, "Expected 1 RLE pair for [5,5,5]");
            match &pairs[0] {
                Value::Array(p) => {
                    assert_eq!(p[0], Value::Integer(5));
                    assert_eq!(p[1], Value::Integer(3));
                }
                other => panic!("Expected Array pair, got {:?}", other),
            }
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_rle_encode_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": []}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_ENCODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Array(vec![])));
}

// ── RLE_DECODE / RUN_LENGTH_DECODE ───────────────────────────────────────────

#[test]
fn test_rle_decode_integers() {
    // [[1,3],[2,2]] -> [1,1,1,2,2]
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [[1,3],[2,2]]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RLE_DECODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(1),
            Value::Integer(1),
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(2),
        ]))
    );
}

#[test]
fn test_run_length_decode_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [[7,2]]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RUN_LENGTH_DECODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::Array(vec![Value::Integer(7), Value::Integer(7)]))
    );
}

#[test]
fn test_rle_encode_decode_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [3,3,1,1,1,2]}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE enc = RLE_ENCODE(arr) COMPUTE dec = RLE_DECODE(enc) SELECT dec;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("dec"),
        Some(&Value::Array(vec![
            Value::Integer(3),
            Value::Integer(3),
            Value::Integer(1),
            Value::Integer(1),
            Value::Integer(1),
            Value::Integer(2),
        ]))
    );
}

// ── DELTA_ENCODE / DELTA_COMPRESS ────────────────────────────────────────────

#[test]
fn test_delta_encode_basic() {
    // [1,3,6,10] -> [1,2,3,4]
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1,3,6,10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DELTA_ENCODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
            Value::Integer(4),
        ]))
    );
}

#[test]
fn test_delta_compress_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10,20,30]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DELTA_COMPRESS(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(10),
            Value::Integer(10),
            Value::Integer(10),
        ]))
    );
}

// ── DELTA_DECODE / DELTA_DECOMPRESS ──────────────────────────────────────────

#[test]
fn test_delta_decode_basic() {
    // [1,2,3,4] -> [1,3,6,10]
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1,2,3,4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DELTA_DECODE(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(1),
            Value::Integer(3),
            Value::Integer(6),
            Value::Integer(10),
        ]))
    );
}

#[test]
fn test_delta_decompress_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10,10,10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DELTA_DECOMPRESS(arr) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
        ]))
    );
}

#[test]
fn test_delta_encode_decode_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5,8,12,15,20]}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE enc = DELTA_ENCODE(arr) COMPUTE dec = DELTA_DECODE(enc) SELECT dec;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("dec"),
        Some(&Value::Array(vec![
            Value::Integer(5),
            Value::Integer(8),
            Value::Integer(12),
            Value::Integer(15),
            Value::Integer(20),
        ]))
    );
}

// ── ZIGZAG_ENCODE / ZZ_ENCODE ────────────────────────────────────────────────

#[test]
fn test_zigzag_encode_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 0}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZIGZAG_ENCODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(0)));
}

#[test]
fn test_zigzag_encode_negative_one() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": -1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZIGZAG_ENCODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(1)));
}

#[test]
fn test_zigzag_encode_one() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 1}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZIGZAG_ENCODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(2)));
}

#[test]
fn test_zz_encode_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": -2}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZZ_ENCODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// ── ZIGZAG_DECODE / ZZ_DECODE ────────────────────────────────────────────────

#[test]
fn test_zigzag_decode_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 0}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZIGZAG_DECODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(0)));
}

#[test]
fn test_zigzag_decode_one_gives_neg_one() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 1}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZIGZAG_DECODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(-1)));
}

#[test]
fn test_zz_decode_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 2}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ZZ_DECODE(n) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("r"), Some(&Value::Integer(1)));
}

#[test]
fn test_zigzag_encode_decode_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": -42}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE enc = ZIGZAG_ENCODE(n) COMPUTE dec = ZIGZAG_DECODE(enc) SELECT dec;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows[0].data.get("dec"), Some(&Value::Integer(-42)));
}

// ── BIT_PACK / PACK_BITS ──────────────────────────────────────────────────────

#[test]
fn test_bit_pack_returns_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1,2,3,4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = BIT_PACK(arr, 4) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::String(s)) => {
            assert!(!s.is_empty(), "BIT_PACK should return non-empty hex string");
            assert!(
                s.chars().all(|c| c.is_ascii_hexdigit()),
                "BIT_PACK output must be hex"
            );
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_pack_bits_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0,1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PACK_BITS(arr, 8) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::String(s)) => {
            assert_eq!(s, "0001", "PACK_BITS([0,1], 8) should be '0001'");
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

// ── BIT_UNPACK / UNPACK_BITS ─────────────────────────────────────────────────

#[test]
fn test_bit_pack_unpack_roundtrip() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5,10,15]}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE packed = BIT_PACK(arr, 8) COMPUTE unpacked = BIT_UNPACK(packed, 8, 3) SELECT unpacked;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("unpacked"),
        Some(&Value::Array(vec![
            Value::Integer(5),
            Value::Integer(10),
            Value::Integer(15),
        ]))
    );
}

#[test]
fn test_unpack_bits_alias() {
    // Pack [7,8] with 8 bits then unpack with alias
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [7,8]}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE packed = PACK_BITS(arr, 8) COMPUTE unpacked = UNPACK_BITS(packed, 8, 2) SELECT unpacked;"#,
    );
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("unpacked"),
        Some(&Value::Array(vec![Value::Integer(7), Value::Integer(8)]))
    );
}

// ── ROT13 / ROTATE13 ─────────────────────────────────────────────────────────

#[test]
fn test_rot13_hello() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ROT13(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("uryyb".to_string()))
    );
}

#[test]
fn test_rotate13_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ROTATE13(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("uryyb".to_string()))
    );
}

#[test]
fn test_rot13_double_application() {
    // ROT13(ROT13("hello")) = "hello"
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r1 = ROT13(s) COMPUTE r2 = ROT13(r1) SELECT r2;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r2"),
        Some(&Value::String("hello".to_string()))
    );
}

#[test]
fn test_rot13_uppercase() {
    // "HELLO" -> "URYYB"
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "HELLO"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ROT13(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("URYYB".to_string()))
    );
}

#[test]
fn test_rot13_preserves_non_alpha() {
    // Numbers and punctuation are unchanged
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello123!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = ROT13(s) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        result.rows[0].data.get("r"),
        Some(&Value::String("uryyb123!".to_string()))
    );
}
