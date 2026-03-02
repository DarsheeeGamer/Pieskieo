/// Integration tests for PQL built-in data validation, data quality, and schema-checking functions.
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

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => panic!("expected numeric, got {:?}", v),
    }
}

// ── IS_EMAIL / VALID_EMAIL ────────────────────────────────────────────────────

#[test]
fn test_is_email_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "user@example.com"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EMAIL(addr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_email_invalid_no_at() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "notanemail"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EMAIL(addr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_email_short_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "a@b.c"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EMAIL(addr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_valid_email_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "user@example.com"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_EMAIL(addr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_email_no_dot_in_domain() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "user@nodot"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EMAIL(addr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_URL / VALID_URL ────────────────────────────────────────────────────────

#[test]
fn test_is_url_https_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"lnk": "https://example.com"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_URL(lnk) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_url_invalid_no_scheme() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"lnk": "notaurl"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_URL(lnk) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_url_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"lnk": "http://example.org/path"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_URL(lnk) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_PHONE / VALID_PHONE ────────────────────────────────────────────────────

#[test]
fn test_is_phone_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ph": "+1-800-555-1234"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_PHONE(ph) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_phone_invalid_letters() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ph": "abc"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_PHONE(ph) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_phone_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ph": "5551234567"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_PHONE(ph) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_CREDIT_CARD / LUHN_CHECK ───────────────────────────────────────────────

#[test]
fn test_is_credit_card_valid_visa() {
    let (db, ex) = setup();
    // Valid Visa test number that passes Luhn
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cc": "4532015112830366"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_CREDIT_CARD(cc) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_credit_card_invalid_short() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cc": "1234567890"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_CREDIT_CARD(cc) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_luhn_check_alias_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cc": "4532015112830366"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = LUHN_CHECK(cc) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_luhn_check_alias_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cc": "4532015112830367"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = LUHN_CHECK(cc) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_POSTAL_CODE / VALID_ZIP ────────────────────────────────────────────────

#[test]
fn test_is_postal_code_us_5digit() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"zip": "12345"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_POSTAL_CODE(zip) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_postal_code_us_plus4() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"zip": "12345-6789"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_POSTAL_CODE(zip) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_valid_zip_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"zip": "90210"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_ZIP(zip) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_DATE_STR / VALID_DATE_STR ──────────────────────────────────────────────

#[test]
fn test_is_date_str_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dt": "2024-01-15"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DATE_STR(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_date_str_wrong_format() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dt": "01-15-2024"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DATE_STR(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_date_str_invalid_month_13() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dt": "2024-13-01"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DATE_STR(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_date_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dt": "2000-12-31"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_DATE_STR(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_TIME_STR / VALID_TIME_STR ──────────────────────────────────────────────

#[test]
fn test_is_time_str_valid_hhmmss() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"tm": "14:30:00"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_TIME_STR(tm) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_time_str_invalid_hour_25() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"tm": "25:00:00"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_TIME_STR(tm) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_time_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"tm": "09:05"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_TIME_STR(tm) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_JSON_STR / VALID_JSON_STR ──────────────────────────────────────────────

#[test]
fn test_is_json_str_valid_object() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"payload": "{\"key\":\"value\"}"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_JSON_STR(payload) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_json_str_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"payload": "not json"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_JSON_STR(payload) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_json_str_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"payload": "[1,2,3]"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_JSON_STR(payload) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_NUMERIC_STR / ALL_DIGITS ───────────────────────────────────────────────

#[test]
fn test_is_numeric_str_digits_only() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"num": "12345"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_NUMERIC_STR(num) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_numeric_str_has_dot() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"num": "123.45"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_NUMERIC_STR(num) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_all_digits_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"num": "000999"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ALL_DIGITS(num) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_ALPHA_ONLY / LETTERS_ONLY ─────────────────────────────────────────────

#[test]
fn test_is_alpha_only_letters() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALPHA_ONLY(txt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_alpha_only_with_digits() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "hello123"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALPHA_ONLY(txt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_letters_only_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "ABCdef"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = LETTERS_ONLY(txt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_ALPHANUMERIC / ALNUM_ONLY ─────────────────────────────────────────────

#[test]
fn test_is_alphanumeric_mixed() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "hello123"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALPHANUMERIC(txt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_alphanumeric_with_special() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "hello!"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALPHANUMERIC(txt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_alnum_only_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "abc123"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ALNUM_ONLY(txt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── COMPLETENESS_SCORE / DATA_COMPLETE ───────────────────────────────────────

#[test]
fn test_completeness_score_all_non_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"rec": {"fname": "Alice", "lname": "Smith", "age": 30}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = COMPLETENESS_SCORE(rec) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(v) => assert!((to_f64(v) - 1.0).abs() < 0.001, "expected 1.0, got {}", to_f64(v)),
        None => panic!("missing sc"),
    }
}

#[test]
fn test_completeness_score_half_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"rec": {"fname": "Bob", "lname": null}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = COMPLETENESS_SCORE(rec) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(v) => assert!((to_f64(v) - 0.5).abs() < 0.001, "expected 0.5, got {}", to_f64(v)),
        None => panic!("missing sc"),
    }
}

#[test]
fn test_data_complete_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"rec": {"x": 1, "y": 2}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = DATA_COMPLETE(rec) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(v) => assert!((to_f64(v) - 1.0).abs() < 0.001, "DATA_COMPLETE alias should give 1.0, got {}", to_f64(v)),
        None => panic!("missing sc"),
    }
}

// ── UNIQUENESS_RATIO / DISTINCT_RATIO ─────────────────────────────────────────

#[test]
fn test_uniqueness_ratio_all_distinct() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"arr": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ur = UNIQUENESS_RATIO(arr) SELECT ur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ur") {
        Some(v) => assert!((to_f64(v) - 1.0).abs() < 0.001, "all distinct → 1.0, got {}", to_f64(v)),
        None => panic!("missing ur"),
    }
}

#[test]
fn test_uniqueness_ratio_all_same() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"arr": [1, 1, 1]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ur = UNIQUENESS_RATIO(arr) SELECT ur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ur") {
        Some(v) => {
            let f = to_f64(v);
            assert!(f > 0.3 && f < 0.4, "all same in 3 → ~0.333, got {}", f);
        }
        None => panic!("missing ur"),
    }
}

#[test]
fn test_distinct_ratio_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"arr": [1, 2, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ur = DISTINCT_RATIO(arr) SELECT ur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ur") {
        Some(v) => {
            let f = to_f64(v);
            // 3 distinct out of 4 → 0.75
            assert!((f - 0.75).abs() < 0.001, "DISTINCT_RATIO: 3 distinct of 4 → 0.75, got {}", f);
        }
        None => panic!("missing ur"),
    }
}

// ── DATA_QUALITY_SCORE / DQ_SCORE ─────────────────────────────────────────────

#[test]
fn test_data_quality_score_full_object() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"rec": {"fname": "Alice", "lname": "Smith", "age": 30}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dq = DATA_QUALITY_SCORE(rec) SELECT dq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dq") {
        Some(v) => {
            let f = to_f64(v);
            assert!(f >= 0.9, "fully populated object should have high DQ score, got {}", f);
        }
        None => panic!("missing dq"),
    }
}

#[test]
fn test_data_quality_score_with_nulls() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"rec": {"fname": "Bob", "lname": null, "age": null}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dq = DATA_QUALITY_SCORE(rec) SELECT dq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dq") {
        Some(v) => {
            let f = to_f64(v);
            assert!(f < 0.9, "object with nulls should have lower DQ score, got {}", f);
        }
        None => panic!("missing dq"),
    }
}

#[test]
fn test_dq_score_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"rec": {"field": "value"}})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dq = DQ_SCORE(rec) SELECT dq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dq") {
        Some(v) => {
            let f = to_f64(v);
            assert!((f - 1.0).abs() < 0.001, "DQ_SCORE alias: single string field → 1.0, got {}", f);
        }
        None => panic!("missing dq"),
    }
}

// ── IS_IBAN / VALID_IBAN ──────────────────────────────────────────────────────

#[test]
fn test_is_iban_valid_gb() {
    let (db, ex) = setup();
    // GB29NWBK60161331926819 is a valid IBAN test value
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"iban": "GB29NWBK60161331926819"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_IBAN(iban) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_iban_invalid_too_short() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"iban": "GB29"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_IBAN(iban) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_iban_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"iban": "GB29NWBK60161331926819"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_IBAN(iban) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}
