/// Integration tests for PQL parsing and formatting utility functions.
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
fn test_parse_int_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "  42  ", "bad": "abc"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE n = PARSE_INT(s) COMPUTE m = PARSE_INT(bad) SELECT n, m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(42)));
    assert_eq!(r.rows[0].data.get("m"), Some(&Value::Null));
}

#[test]
fn test_parse_int_hex() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"hex_val": "0xff"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = PARSE_INT(hex_val) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(255)));
}

#[test]
fn test_parse_float_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"num_str": "  3.14  ", "bad": "xyz"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE f = PARSE_FLOAT(num_str) COMPUTE g = PARSE_FLOAT(bad) SELECT f, g;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Float(v)) => {
            assert!((v - 3.14).abs() < 1e-9, "expected ~3.14, got {}", v);
        }
        other => panic!("expected Float, got {:?}", other),
    }
    assert_eq!(r.rows[0].data.get("g"), Some(&Value::Null));
}

#[test]
fn test_parse_date_str() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "15/03/2024"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PARSE_DATE_STR(dt, "%d/%m/%Y") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("2024-03-15".to_string()))
    );
}

#[test]
fn test_parse_date_str_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "not-a-date"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PARSE_DATE_STR(dt, "%d/%m/%Y") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Null));
}

#[test]
fn test_parse_csv_line_simple() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"csv_line": "a,b,c"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE arr = PARSE_CSV_LINE(csv_line) SELECT arr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("arr"),
        Some(&Value::Array(vec![
            Value::String("a".to_string()),
            Value::String("b".to_string()),
            Value::String("c".to_string()),
        ]))
    );
}

#[test]
fn test_parse_csv_line_quoted() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"csv_line": "a,\"b,c\",d"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE arr = PARSE_CSV_LINE(csv_line) SELECT arr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("arr"),
        Some(&Value::Array(vec![
            Value::String("a".to_string()),
            Value::String("b,c".to_string()),
            Value::String("d".to_string()),
        ]))
    );
}

#[test]
fn test_parse_kv_ampersand() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"kvstr": "name=John&age=30"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = PARSE_KV(kvstr) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(map)) => {
            assert_eq!(map.get("name"), Some(&Value::String("John".to_string())));
            assert_eq!(map.get("age"), Some(&Value::String("30".to_string())));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_parse_tsv_line() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tsv_line": "alpha\tbeta\tgamma"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE arr = PARSE_TSV_LINE(tsv_line) SELECT arr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("arr"),
        Some(&Value::Array(vec![
            Value::String("alpha".to_string()),
            Value::String("beta".to_string()),
            Value::String("gamma".to_string()),
        ]))
    );
}

#[test]
fn test_format_bytes_kb() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1024}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_BYTES(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1 KB".to_string()))
    );
}

#[test]
fn test_format_bytes_mb() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1048576}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_BYTES(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1 MB".to_string()))
    );
}

#[test]
fn test_format_bytes_partial() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1536}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_BYTES(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1.5 KB".to_string()))
    );
}

#[test]
fn test_format_duration_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"secs": 3661}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_DURATION(secs) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1h 1m 1s".to_string()))
    );
}

#[test]
fn test_format_duration_subsecond() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"secs": 1.5}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_DURATION(secs) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1s 500ms".to_string()))
    );
}

#[test]
fn test_format_percent_default() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ratio": 0.15}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_PERCENT(ratio) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("15.0%".to_string()))
    );
}

#[test]
fn test_format_percent_custom_decimals() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ratio": 0.153}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_PERCENT(ratio, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("15.30%".to_string()))
    );
}

#[test]
fn test_format_currency_usd() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"amt": 1234567.89}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_CURRENCY(amt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("$1,234,567.89".to_string()))
    );
}

#[test]
fn test_format_currency_eur() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"amt": 1000.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_CURRENCY(amt, "EUR") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("\u{20ac}1,000.00".to_string()))
    );
}

#[test]
fn test_format_scientific_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 0.00123}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_SCIENTIFIC(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // 0.00123 = 1.23e-3
    match r.rows[0].data.get("out") {
        Some(Value::String(s)) => {
            assert!(s.contains("1.23"), "expected mantissa 1.23, got: {}", s);
            assert!(
                s.contains("e-") || s.contains("e+"),
                "expected scientific notation, got: {}",
                s
            );
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_format_ordinal() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2, "c": 3, "d": 11, "e": 12, "f": 21}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t
        COMPUTE o1 = FORMAT_ORDINAL(a)
        COMPUTE o2 = FORMAT_ORDINAL(b)
        COMPUTE o3 = FORMAT_ORDINAL(c)
        COMPUTE o11 = FORMAT_ORDINAL(d)
        COMPUTE o12 = FORMAT_ORDINAL(e)
        COMPUTE o21 = FORMAT_ORDINAL(f)
        SELECT o1, o2, o3, o11, o12, o21;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("o1"),
        Some(&Value::String("1st".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o2"),
        Some(&Value::String("2nd".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o3"),
        Some(&Value::String("3rd".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o11"),
        Some(&Value::String("11th".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o12"),
        Some(&Value::String("12th".to_string()))
    );
    assert_eq!(
        r.rows[0].data.get("o21"),
        Some(&Value::String("21st".to_string()))
    );
}

#[test]
fn test_si_prefix_kilo() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1500}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SI_PREFIX(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1.5k".to_string()))
    );
}

#[test]
fn test_si_prefix_mega() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1500000}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SI_PREFIX(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("out"),
        Some(&Value::String("1.5M".to_string()))
    );
}

#[test]
fn test_parse_size_mb() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"sz": "1.5 MB"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PARSE_SIZE(sz) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(1572864)));
}

#[test]
fn test_parse_size_kb() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"sz": "2 KB"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PARSE_SIZE(sz) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(2048)));
}

#[test]
fn test_parse_duration_hours_minutes() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dur": "1h 30m"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PARSE_DURATION(dur) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Float(v)) => {
            assert!((v - 5400.0).abs() < 1e-6, "expected 5400.0, got {}", v);
        }
        other => panic!("expected Float(5400.0), got {:?}", other),
    }
}

#[test]
fn test_parse_duration_seconds_ms() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dur": "2s 500ms"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PARSE_DURATION(dur) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Float(v)) => {
            assert!((v - 2.5).abs() < 1e-6, "expected 2.5, got {}", v);
        }
        other => panic!("expected Float(2.5), got {:?}", other),
    }
}

#[test]
fn test_aliases_str_to_int() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": "-99"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = STR_TO_INT(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(-99)));
}

#[test]
fn test_aliases_atof() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": "2.718"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ATOF(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Float(v)) => {
            assert!((v - 2.718).abs() < 1e-9, "expected ~2.718, got {}", v);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}
