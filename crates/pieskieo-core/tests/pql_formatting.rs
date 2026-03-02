/// Integration tests for PQL number formatting, locale, and currency utility functions.
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

// ── FORMAT_NUMBER ──────────────────────────────────────────────────────────

#[test]
fn test_format_number_thousands() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1234567.89})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_NUMBER(nv, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1,234,567.89".to_string())));
}

#[test]
fn test_num_format_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1000.0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NUM_FORMAT(nv, 0) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1,000".to_string())));
}

// ── FORMAT_CURRENCY ────────────────────────────────────────────────────────

#[test]
fn test_format_currency_default_usd() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1234.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_CURRENCY(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("$1,234.50".to_string())));
}

#[test]
fn test_format_currency_eur() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1234.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_CURRENCY(nv, "EUR") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("€1,234.50".to_string())));
}

#[test]
fn test_currency_format_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 500.0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CURRENCY_FORMAT(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("$500.00".to_string())));
}

// ── FORMAT_PERCENT ─────────────────────────────────────────────────────────

#[test]
fn test_format_percent_two_decimals() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 0.4567})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_PERCENT(nv, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("45.67%".to_string())));
}

#[test]
fn test_format_percent_zero_decimals() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 0.1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_PERCENT(nv, 0) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("10%".to_string())));
}

#[test]
fn test_percent_format_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 0.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PERCENT_FORMAT(nv, 1) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("50.0%".to_string())));
}

// ── FORMAT_BYTES ───────────────────────────────────────────────────────────

#[test]
fn test_format_bytes_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_BYTES(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("0 B".to_string())));
}

#[test]
fn test_format_bytes_one_kb() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1024})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_BYTES(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1 KB".to_string())));
}

#[test]
fn test_format_bytes_one_mb() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1048576})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_BYTES(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1 MB".to_string())));
}

#[test]
fn test_humanize_bytes_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 512})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = HUMANIZE_BYTES(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("512 B".to_string())));
}

// ── ORDINAL_SUFFIX ─────────────────────────────────────────────────────────

#[test]
fn test_ordinal_suffix_1st() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1st".to_string())));
}

#[test]
fn test_ordinal_suffix_2nd() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("2nd".to_string())));
}

#[test]
fn test_ordinal_suffix_3rd() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("3rd".to_string())));
}

#[test]
fn test_ordinal_suffix_4th() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 4})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("4th".to_string())));
}

#[test]
fn test_ordinal_suffix_11th_special_case() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 11})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("11th".to_string())));
}

#[test]
fn test_ordinal_suffix_21st() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 21})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("21st".to_string())));
}

#[test]
fn test_ordinal_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("5th".to_string())));
}

// ── INT_TO_ROMAN ───────────────────────────────────────────────────────────

#[test]
fn test_int_to_roman_1() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INT_TO_ROMAN(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("I".to_string())));
}

#[test]
fn test_int_to_roman_4() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 4})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INT_TO_ROMAN(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("IV".to_string())));
}

#[test]
fn test_int_to_roman_14() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 14})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INT_TO_ROMAN(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("XIV".to_string())));
}

#[test]
fn test_int_to_roman_2024() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 2024})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INT_TO_ROMAN(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("MMXXIV".to_string())));
}

#[test]
fn test_to_roman_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 10})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TO_ROMAN(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("X".to_string())));
}

// ── ROMAN_TO_INT ───────────────────────────────────────────────────────────

#[test]
fn test_roman_to_int_xiv() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "XIV"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROMAN_TO_INT(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(14)));
}

#[test]
fn test_roman_to_int_mmxxiv() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "MMXXIV"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROMAN_TO_INT(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(2024)));
}

#[test]
fn test_from_roman_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "IX"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FROM_ROMAN(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(9)));
}

// ── NUM_TO_WORDS ───────────────────────────────────────────────────────────

#[test]
fn test_num_to_words_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NUM_TO_WORDS(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("zero".to_string())));
}

#[test]
fn test_num_to_words_forty_two() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 42})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NUM_TO_WORDS(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("forty-two".to_string())));
}

#[test]
fn test_num_to_words_one_hundred() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 100})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NUM_TO_WORDS(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("one hundred".to_string())));
}

#[test]
fn test_number_to_words_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 7})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NUMBER_TO_WORDS(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("seven".to_string())));
}

// ── FORMAT_DURATION ────────────────────────────────────────────────────────

#[test]
fn test_format_duration_hours_mins_secs() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 3661})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_DURATION(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1h 1m 1s".to_string())));
}

#[test]
fn test_format_duration_hours_only() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 86400})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_DURATION(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("24h".to_string())));
}

#[test]
fn test_humanize_duration_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 60})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = HUMANIZE_DURATION(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("1m".to_string())));
}

// ── CLEAN_PHONE ────────────────────────────────────────────────────────────

#[test]
fn test_clean_phone_strips_formatting() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"ph": "+1 (555) 123-4567"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CLEAN_PHONE(ph) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("15551234567".to_string())));
}

#[test]
fn test_normalize_phone_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"ph": "(800) 555-0199"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NORMALIZE_PHONE(ph) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("8005550199".to_string())));
}

// ── LUHN_CHECK ─────────────────────────────────────────────────────────────

#[test]
fn test_luhn_check_valid_visa() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "4532015112830366"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = LUHN_CHECK(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_luhn_check_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "1234567890123456"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = LUHN_CHECK(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_valid_luhn_alias_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "4532015112830366"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_VALID_LUHN(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_luhn_alias_invalid() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "1234567890123456"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_VALID_LUHN(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(false)));
}

// ── TRUNCATE_TEXT ──────────────────────────────────────────────────────────

#[test]
fn test_truncate_text_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "Hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TRUNCATE_TEXT(txt, 8) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("Hello...".to_string())));
}

#[test]
fn test_truncate_text_no_truncation_needed() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "Hi"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TRUNCATE_TEXT(txt, 10) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("Hi".to_string())));
}

#[test]
fn test_text_truncate_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "Hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TEXT_TRUNCATE(txt, 8) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("Hello...".to_string())));
}

// ── WRAP_TEXT ──────────────────────────────────────────────────────────────

#[test]
fn test_wrap_text_word_boundaries() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "one two three four"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = WRAP_TEXT(txt, 10) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // "one two" = 7 fits; "one two three" = 13 > 10, new line for "three";
    // "three four" = 10 fits on next line
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("one two\nthree four".to_string())));
}

#[test]
fn test_word_wrap_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "hello world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = WORD_WRAP(txt, 20) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("hello world".to_string())));
}

// ── Additional edge case tests ─────────────────────────────────────────────

#[test]
fn test_format_number_integer_no_decimals() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 9999})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_NUMBER(nv, 0) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("9,999".to_string())));
}

#[test]
fn test_format_currency_gbp() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 99.99})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FORMAT_CURRENCY(nv, "GBP") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("£99.99".to_string())));
}

#[test]
fn test_ordinal_suffix_12th_special_case() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 12})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("12th".to_string())));
}

#[test]
fn test_ordinal_suffix_13th_special_case() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 13})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ORDINAL_SUFFIX(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("13th".to_string())));
}

#[test]
fn test_int_to_roman_9() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 9})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INT_TO_ROMAN(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("IX".to_string())));
}

#[test]
fn test_roman_to_int_iv() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "IV"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROMAN_TO_INT(txt) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(4)));
}

#[test]
fn test_clean_phone_digits_only_input() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"ph": "5551234567"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CLEAN_PHONE(ph) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("5551234567".to_string())));
}

#[test]
fn test_num_to_words_nineteen() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 19})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NUM_TO_WORDS(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("nineteen".to_string())));
}

#[test]
fn test_truncate_text_exact_length() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "Hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TRUNCATE_TEXT(txt, 5) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("Hello".to_string())));
}

