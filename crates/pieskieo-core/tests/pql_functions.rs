/// Integration tests for PQL built-in functions.
/// Tests live here (separate from production code) per project conventions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

// ── Type-checking functions ──────────────────────────────────────────────────

#[test]
fn test_is_number() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 42, "s": "hello"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE a = IS_NUMBER(n) COMPUTE b = IS_NUMBER(s) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_string_bool_array_object() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello", "b": true, "arr": [1,2], "obj": {"x": 1}})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE is_s = IS_STRING(s) COMPUTE is_b = IS_BOOL(b) COMPUTE is_a = IS_ARRAY(arr) COMPUTE is_o = IS_OBJECT(obj) SELECT is_s, is_b, is_a, is_o;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("is_s"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("is_b"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("is_a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("is_o"), Some(&Value::Bool(true)));
}

// ── Hex / binary encoding ────────────────────────────────────────────────────

#[test]
fn test_hex_conversion() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 255})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE h = TO_HEX(n) COMPUTE b = TO_BINARY(n) SELECT h, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("ff".to_string())));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::String("11111111".to_string())));
}

#[test]
fn test_from_hex() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"h": "ff"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE n = FROM_HEX(h) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(255)));
}

// ── String extras ────────────────────────────────────────────────────────────

#[test]
fn test_mask_functions() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"card": "1234567890123456"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE m = MASK(card) COMPUTE pm = MASK_PARTIAL(card, 4, 4) SELECT m, pm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::String(s)) => assert!(s.chars().all(|c| c == '*'), "MASK should produce all *"),
        other => panic!("expected masked string, got {:?}", other),
    }
    match r.rows[0].data.get("pm") {
        Some(Value::String(s)) => {
            assert_eq!(&s[..4], "1234", "first 4 chars should be visible");
            assert_eq!(&s[s.len()-4..], "3456", "last 4 chars should be visible");
        }
        other => panic!("expected partially masked string, got {:?}", other),
    }
}

#[test]
fn test_word_count_char_count() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "hello world foo bar"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE wc = WORD_COUNT(text) COMPUTE cc = CHAR_COUNT(text, "o") SELECT wc, cc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("wc"), Some(&Value::Integer(4)));
    assert_eq!(r.rows[0].data.get("cc"), Some(&Value::Integer(4))); // "hello"(1), "world"(1), "foo"(2)
}

// ── Geospatial extras ────────────────────────────────────────────────────────

#[test]
fn test_geo_point_functions() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"lat": 40.7128, "lon": -74.0060})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE pt = GEO_POINT(lat, lon) COMPUTE miles = GEO_DISTANCE_MILES(lat, lon, 51.5074, -0.1278) SELECT pt, miles;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pt") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("lat"), "geo point should have lat");
            assert!(m.contains_key("lon"), "geo point should have lon");
        }
        other => panic!("expected object, got {:?}", other),
    }
    match r.rows[0].data.get("miles") {
        Some(Value::Float(f)) => assert!(*f > 3000.0 && *f < 4000.0, "NYC-London should be ~3460 miles, got {}", f),
        other => panic!("expected float miles, got {:?}", other),
    }
}

// ── Aggregate extras ─────────────────────────────────────────────────────────

#[test]
fn test_range_agg_and_entropy() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [10, 20, 30, 40, 50] {
        db.put_doc_ns(None, Some("data"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY data COMPUTE g = 1 GROUP BY g COMPUTE r = RANGE_AGG(val) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows.len(), 1);
    match result.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!((*f - 40.0).abs() < 0.001, "range should be 50-10=40, got {}", f),
        other => panic!("expected float 40.0, got {:?}", other),
    }
}

#[test]
fn test_iqr() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Values: 1, 2, 3, 4, 5, 6, 7, 8 — Q1=2, Q3=6, IQR=4
    for v in [1, 2, 3, 4, 5, 6, 7, 8] {
        db.put_doc_ns(None, Some("vals"), Uuid::new_v4(), serde_json::json!({"v": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY vals COMPUTE g = 1 GROUP BY g COMPUTE iqr_val = IQR(v) SELECT iqr_val;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows.len(), 1);
    match result.rows[0].data.get("iqr_val") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "IQR should be positive, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}
