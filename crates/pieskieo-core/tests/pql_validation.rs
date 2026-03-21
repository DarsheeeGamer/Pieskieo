/// Integration tests for PQL built-in data validation and quality functions.
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

// ── IS_VALID_DATE / VALIDATE_DATE ─────────────────────────────────────────────

#[test]
fn test_is_valid_date_leap_year_valid() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-02-29", "d2": "2023-02-29", "d3": "invalid"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE v1 = IS_VALID_DATE(d1) COMPUTE v2 = IS_VALID_DATE(d2) COMPUTE v3 = IS_VALID_DATE(d3) SELECT v1, v2, v3;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v1"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("v2"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("v3"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_valid_date_month_bounds() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-04-30", "d2": "2024-04-31"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ok = IS_VALID_DATE(d1) COMPUTE bad = IS_VALID_DATE(d2) SELECT ok, bad;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("bad"), Some(&Value::Bool(false)));
}

#[test]
fn test_validate_date_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2025-12-31"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALIDATE_DATE(d) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_date_invalid_month() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2024-13-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_DATE(d) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_VALID_DATETIME / VALIDATE_DATETIME ─────────────────────────────────────

#[test]
fn test_is_valid_datetime_full() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-06-15 23:59:59"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_DATETIME(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_datetime_date_only() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALIDATE_DATETIME(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_datetime_bad_time() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-01 25:00:00"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_DATETIME(dt) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_DATE_BEFORE / IS_DATE_AFTER ───────────────────────────────────────────

#[test]
fn test_is_date_before() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "2022-01-01", "b": "2023-06-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DATE_BEFORE(a, b) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_date_lt_alias_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "2025-12-31", "b": "2023-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = DATE_LT(a, b) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_date_after() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "2025-06-01", "b": "2020-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DATE_AFTER(a, b) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_DATE_BETWEEN / DATE_IN_RANGE ──────────────────────────────────────────

#[test]
fn test_is_date_between_inclusive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2024-06-15"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ok = IS_DATE_BETWEEN(d, "2024-01-01", "2024-12-31") SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_date_in_range_out() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2025-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ok = DATE_IN_RANGE(d, "2024-01-01", "2024-12-31") SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_FUTURE_DATE / IS_PAST_DATE ─────────────────────────────────────────────

#[test]
fn test_is_future_date() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2030-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_FUTURE_DATE(d) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_past_date() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2020-06-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_PAST_DATE(d) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_in_future_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d": "2019-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_IN_FUTURE(d) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_IN_RANGE / IN_RANGE ────────────────────────────────────────────────────

#[test]
fn test_is_in_range_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 50}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_IN_RANGE(n, 1, 100) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_in_range_boundary() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 100}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IN_RANGE(n, 1, 100) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_in_range_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 101}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_IN_RANGE(n, 1, 100) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_EVEN / IS_ODD ─────────────────────────────────────────────────────────

#[test]
fn test_is_even_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 42}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EVEN(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_odd_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 7}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ODD(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_even_num_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 3}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EVEN_NUM(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_WHOLE_NUMBER / IS_WHOLE ────────────────────────────────────────────────

#[test]
fn test_is_whole_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_WHOLE_NUMBER(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_whole_float_fractional() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 5.5}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_WHOLE(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_DIVISIBLE_BY / DIVISIBLE_BY ───────────────────────────────────────────

#[test]
fn test_is_divisible_by_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 12}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DIVISIBLE_BY(n, 4) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_multiple_of_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 9}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = MULTIPLE_OF(n, 4) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_BETWEEN / IN_BOUNDS ───────────────────────────────────────────────────

#[test]
fn test_is_between_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_BETWEEN(n, 1, 10) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_in_bounds_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 11}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IN_BOUNDS(n, 1, 10) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_PRIME_NUMBER ───────────────────────────────────────────────────────────

#[test]
fn test_is_prime_number_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 17}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_PRIME_NUMBER(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_prime_number_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 18}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_PRIME_NUMBER(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── MATCHES_PATTERN / REGEX_MATCH_SIMPLE (glob) ───────────────────────────────

#[test]
fn test_matches_pattern_star() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = MATCHES_PATTERN(s, "hello*") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_matches_pattern_question_mark() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "cat"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = REGEX_MATCH_SIMPLE(s, "c?t") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_matches_pattern_no_match() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "dog"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = MATCHES_PATTERN(s, "cat") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_ALPHA / IS_ALPHABETIC ─────────────────────────────────────────────────

#[test]
fn test_is_alpha_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "HelloWorld"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALPHA(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_alphabetic_false_with_digit() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "Hello1"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALPHABETIC(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_DIGITS_ONLY / IS_ALL_DIGITS ────────────────────────────────────────────

#[test]
fn test_is_digits_only_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "123456"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_DIGITS_ONLY(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_all_digits_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "123abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALL_DIGITS(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── HAS_UPPERCASE / HAS_LOWERCASE / HAS_DIGIT / HAS_SPECIAL_CHAR ─────────────

#[test]
fn test_has_uppercase_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "helloWorld"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = HAS_UPPERCASE(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_has_lowercase_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "ALLCAPS"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = HAS_LOWERCASE(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_has_digit_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "pass1word"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = HAS_DIGIT(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_has_special_char_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "p@ssword"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = HAS_SPECIAL_CHAR(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_STRONG_PASSWORD / IS_STRONG_PASS ──────────────────────────────────────

#[test]
fn test_is_strong_password_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "Abc123!@#"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_STRONG_PASSWORD(pw) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_strong_pass_too_short() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "Ab1!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_STRONG_PASS(pw) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_strong_password_no_special() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pw": "Abcdef12"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_STRONG_PASSWORD(pw) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_EMPTY_OR_NULL / IS_BLANK_VAL ──────────────────────────────────────────

#[test]
fn test_is_empty_or_null_empty_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "   "}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_EMPTY_OR_NULL(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_blank_val_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_BLANK_VAL(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_NOT_EMPTY / IS_FILLED ──────────────────────────────────────────────────

#[test]
fn test_is_not_empty_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "data"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_NOT_EMPTY(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_filled_false_whitespace() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "  "}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_FILLED(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── STRING_MATCHES_ANY / MATCHES_ANY ─────────────────────────────────────────

#[test]
fn test_string_matches_any_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "foobar", "patterns": ["foo*", "baz*"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = STRING_MATCHES_ANY(s, patterns) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_matches_any_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "qux", "patterns": ["foo*", "bar*"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = MATCHES_ANY(s, patterns) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ASSERT_NOT_NULL / REQUIRE_NOT_NULL ───────────────────────────────────────

#[test]
fn test_assert_not_null_passes_value() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"num": 42}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ASSERT_NOT_NULL(num, "num") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Integer(42)));
}

#[test]
fn test_require_not_null_error_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"missing": null}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE ok = REQUIRE_NOT_NULL(missing, "missing") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ok") {
        Some(Value::String(s)) => assert!(s.contains("VALIDATION_ERROR")),
        other => panic!("expected validation error string, got {:?}", other),
    }
}

// ── ASSERT_IN_RANGE / REQUIRE_IN_RANGE ───────────────────────────────────────

#[test]
fn test_assert_in_range_passes() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"score": 85}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE ok = ASSERT_IN_RANGE(score, 0, 100, "score") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Integer(85)));
}

#[test]
fn test_require_in_range_error() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"score": 150}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE ok = REQUIRE_IN_RANGE(score, 0, 100, "score") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ok") {
        Some(Value::String(s)) => assert!(s.contains("VALIDATION_ERROR")),
        other => panic!("expected validation error string, got {:?}", other),
    }
}

// ── COALESCE_DEFAULT ──────────────────────────────────────────────────────────

#[test]
fn test_coalesce_default_uses_default() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": null}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = COALESCE_DEFAULT(x, 99) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Integer(99)));
}

#[test]
fn test_coalesce_default_returns_value() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 42}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = COALESCE_DEFAULT(x, 99) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Integer(42)));
}

// ── NULLIF_EMPTY ──────────────────────────────────────────────────────────────

#[test]
fn test_nullif_empty_replaces_empty_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = NULLIF_EMPTY(s, "default") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ok"),
        Some(&Value::String("default".to_string()))
    );
}

#[test]
fn test_nullif_empty_keeps_non_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = NULLIF_EMPTY(s, "default") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ok"),
        Some(&Value::String("hello".to_string()))
    );
}

// ── NULLIF_EQUAL / NULL_IF_EQ ─────────────────────────────────────────────────

#[test]
fn test_nullif_equal_returns_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 0}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = NULLIF_EQUAL(n, 0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Null));
}

#[test]
fn test_null_if_eq_returns_value() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = NULL_IF_EQ(n, 0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Integer(5)));
}

// ── GET_TYPE ──────────────────────────────────────────────────────────────────

#[test]
fn test_get_type_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 42}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = GET_TYPE(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ok"),
        Some(&Value::String("integer".to_string()))
    );
}

#[test]
fn test_get_type_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = GET_TYPE(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ok"),
        Some(&Value::String("string".to_string()))
    );
}

#[test]
fn test_get_type_bool() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"b": true}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = GET_TYPE(b) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ok"),
        Some(&Value::String("bool".to_string()))
    );
}

// ── CAST_SAFE / TRY_CAST ──────────────────────────────────────────────────────

#[test]
fn test_cast_safe_string_to_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "42"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = CAST_SAFE(s, "integer") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Integer(42)));
}

#[test]
fn test_try_cast_invalid_returns_null() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "not_a_number"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = TRY_CAST(s, "integer") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Null));
}

#[test]
fn test_cast_safe_integer_to_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 100}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = CAST_SAFE(n, "string") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ok"),
        Some(&Value::String("100".to_string()))
    );
}

#[test]
fn test_cast_safe_string_to_bool() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "true"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = CAST_SAFE(s, "bool") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── VALIDATE_SCHEMA / MATCHES_SCHEMA ─────────────────────────────────────────

#[test]
fn test_validate_schema_pass() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rec": {"age": 25, "name": "Alice"}}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ok = VALIDATE_SCHEMA(rec, {"age": "integer", "name": "string"}) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_matches_schema_fail_wrong_type() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rec": {"age": "twenty-five", "name": "Alice"}}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE ok = MATCHES_SCHEMA(rec, {"age": "integer", "name": "string"}) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ARRAY_IS_SORTED / IS_SORTED ──────────────────────────────────────────────

#[test]
fn test_array_is_sorted_asc() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_IS_SORTED(arr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_sorted_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 3, 2, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_SORTED(arr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_array_is_sorted_desc() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 4, 3, 2, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_IS_SORTED(arr, "desc") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── ARRAY_IS_UNIQUE / IS_ALL_UNIQUE ──────────────────────────────────────────

#[test]
fn test_array_is_unique_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_IS_UNIQUE(arr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_all_unique_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALL_UNIQUE(arr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ARRAY_ALL_MATCH / ALL_SATISFY ────────────────────────────────────────────

#[test]
fn test_array_all_match_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_ALL_MATCH(arr, "positive") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_all_satisfy_even_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [2, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ALL_SATISFY(arr, "even") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ARRAY_ANY_MATCH / ANY_SATISFY ────────────────────────────────────────────

#[test]
fn test_array_any_match_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, -2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_ANY_MATCH(arr, "negative") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_any_satisfy_odd_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [2, 4, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ANY_SATISFY(arr, "odd") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ARRAY_NONE_MATCH / NONE_SATISFY ──────────────────────────────────────────

#[test]
fn test_array_none_match_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_NONE_MATCH(arr, "negative") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_none_satisfy_positive_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, -2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = NONE_SATISFY(arr, "positive") SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ARRAY_COUNT_IF / COUNT_WHERE ─────────────────────────────────────────────

#[test]
fn test_array_count_if_even() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = ARRAY_COUNT_IF(arr, "even") SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

#[test]
fn test_count_where_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [-1, 2, -3, 4, 0]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = COUNT_WHERE(arr, "positive") SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(2)));
}

// ── ARRAY_HAS_NULLS / HAS_NULLS ──────────────────────────────────────────────

#[test]
fn test_array_has_nulls_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, null, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = ARRAY_HAS_NULLS(arr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_has_nulls_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = HAS_NULLS(arr) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── ARRAY_NULL_COUNT / COUNT_NULLS ───────────────────────────────────────────

#[test]
fn test_array_null_count() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [null, 1, null, 2, null]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = ARRAY_NULL_COUNT(arr) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

#[test]
fn test_count_nulls_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = COUNT_NULLS(arr) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(0)));
}

// ── ARRAY_NON_NULL_COUNT / COUNT_NON_NULLS ───────────────────────────────────

#[test]
fn test_array_non_null_count() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [null, 1, null, 2]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = ARRAY_NON_NULL_COUNT(arr) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(2)));
}

#[test]
fn test_count_non_nulls_full_array() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = COUNT_NON_NULLS(arr) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

// ── IS_FINITE_NUM / IS_NOT_A_NUMBER ──────────────────────────────────────────

#[test]
fn test_is_finite_num_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 42}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_FINITE_NUM(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_not_a_number_non_nan() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 3.14}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_NOT_A_NUMBER(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── IS_POS / IS_NEG / IS_ZERO_VAL ────────────────────────────────────────────

#[test]
fn test_is_pos_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_POS(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_neg_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": -3}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_NEG(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_zero_val_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 0}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ZERO_VAL(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_ALNUM ──────────────────────────────────────────────────────────────────

#[test]
fn test_is_alnum_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "abc123"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALNUM(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_alnum_false_with_special() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "abc!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_ALNUM(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── CONTAINS_UPPER / CONTAINS_LOWER / CONTAINS_DIGIT / CONTAINS_SPECIAL ──────

#[test]
fn test_contains_upper_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "helloWorld"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = CONTAINS_UPPER(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_contains_special_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "no_special"}),
    )
    .unwrap();
    // underscore is not alphanumeric, so this should be true
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = CONTAINS_SPECIAL(s) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}
