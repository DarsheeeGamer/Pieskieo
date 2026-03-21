/// Integration tests for PQL calendar and business-date functions.
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

// ── IS_LEAP_YEAR / IS_LEAP ────────────────────────────────────────────────────

#[test]
fn test_is_leap_year_2000() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2000}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_LEAP_YEAR(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_leap_year_1900_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 1900}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_LEAP_YEAR(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_leap_year_2024() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2024}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_LEAP_YEAR(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_leap_year_2023_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2023}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_LEAP_YEAR(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_leap_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2024}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_LEAP(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

// ── DAYS_IN_MONTH ─────────────────────────────────────────────────────────────

#[test]
fn test_days_in_month_feb_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2000, "mo": 2}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAYS_IN_MONTH(yr, mo) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(29)));
}

#[test]
fn test_days_in_month_feb_non_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 1900, "mo": 2}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAYS_IN_MONTH(yr, mo) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(28)));
}

#[test]
fn test_days_in_month_january() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2024, "mo": 1}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAYS_IN_MONTH(yr, mo) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(31)));
}

// ── DAYS_IN_YEAR ─────────────────────────────────────────────────────────────

#[test]
fn test_days_in_year_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2024}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAYS_IN_YEAR(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(366)));
}

#[test]
fn test_days_in_year_non_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"yr": 2023}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAYS_IN_YEAR(yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(365)));
}

// ── LAST_DAY_OF_MONTH / FIRST_DAY_OF_MONTH ───────────────────────────────────

#[test]
fn test_last_day_of_month_feb_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-02-15"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = LAST_DAY_OF_MONTH(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-02-29".to_string()))
    );
}

#[test]
fn test_last_day_of_month_jan() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2023-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = LAST_DAY_OF_MONTH(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2023-01-31".to_string()))
    );
}

#[test]
fn test_first_day_of_month() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-02-15"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = FIRST_DAY_OF_MONTH(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-02-01".to_string()))
    );
}

// ── QUARTER_NUMBER / QUARTER_START / QUARTER_END ─────────────────────────────

#[test]
fn test_quarter_number_q1() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-15"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = QUARTER_NUMBER(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(1)));
}

#[test]
fn test_quarter_number_q2() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-04-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = QUARTER_NUMBER(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(2)));
}

#[test]
fn test_quarter_number_q3() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-07-10"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = QUARTER_NUMBER(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(3)));
}

#[test]
fn test_quarter_number_q4() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-10-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = QUARTER_NUMBER(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(4)));
}

#[test]
fn test_quarter_start() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-05-15"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = QUARTER_START(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-04-01".to_string()))
    );
}

#[test]
fn test_quarter_end() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-05-15"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = QUARTER_END(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-06-30".to_string()))
    );
}

// ── WEEK_NUMBER ───────────────────────────────────────────────────────────────

#[test]
fn test_week_number_jan1() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = WEEK_NUMBER(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // First day of year should be week 1
    let nv = r.rows[0].data.get("nv").cloned();
    assert!(matches!(nv, Some(Value::Integer(1))));
}

// ── DATE_ADD_MONTHS / DATE_ADD_YEARS ─────────────────────────────────────────

#[test]
fn test_date_add_months_clamp() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-31", "mo": 1}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_ADD_MONTHS(dt, mo) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Jan 31 + 1 month = Feb 29 (2024 is a leap year, clamped to 29)
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-02-29".to_string()))
    );
}

#[test]
fn test_date_add_years_clamp() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2020-02-29", "yr": 1}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_ADD_YEARS(dt, yr) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Feb 29 + 1 year = Feb 28 (2021 is not a leap year, clamped to 28)
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2021-02-28".to_string()))
    );
}

// ── IS_WEEKEND_DAY / IS_WORK_DAY ─────────────────────────────────────────────

#[test]
fn test_is_weekend_day_saturday() {
    let (db, ex) = setup();
    // 2024-01-06 is a Saturday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-06"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_WEEKEND_DAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_weekend_day_monday_false() {
    let (db, ex) = setup();
    // 2024-01-08 is a Monday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-08"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_WEEKEND_DAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_work_day_monday() {
    let (db, ex) = setup();
    // 2024-01-08 is a Monday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-08"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_WORK_DAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_work_day_saturday_false() {
    let (db, ex) = setup();
    // 2024-01-06 is a Saturday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-06"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = IS_WORK_DAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Bool(false)));
}

// ── DAY_OF_WEEK_NAME ─────────────────────────────────────────────────────────

#[test]
fn test_day_of_week_name_monday() {
    let (db, ex) = setup();
    // 2024-01-01 is a Monday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAY_OF_WEEK_NAME(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("Monday".to_string()))
    );
}

#[test]
fn test_day_of_week_name_saturday() {
    let (db, ex) = setup();
    // 2024-01-06 is a Saturday
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-06"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DAY_OF_WEEK_NAME(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("Saturday".to_string()))
    );
}

// ── MONTH_NAME ────────────────────────────────────────────────────────────────

#[test]
fn test_month_name_january() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mo": 1}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = MONTH_NAME(mo) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("January".to_string()))
    );
}

#[test]
fn test_month_name_december() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mo": 12}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = MONTH_NAME(mo) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("December".to_string()))
    );
}

// ── DATE_DIFF_MONTHS / DATE_DIFF_YEARS ───────────────────────────────────────

#[test]
fn test_date_diff_months() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-01-01", "d2": "2024-06-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_DIFF_MONTHS(d1, d2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(5)));
}

#[test]
fn test_date_diff_years() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2020-01-01", "d2": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = DATE_DIFF_YEARS(d1, d2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(4)));
}

// ── NEXT_WEEKDAY / PREV_WEEKDAY / PREV_BUSINESS_DAY ──────────────────────────

#[test]
fn test_next_weekday_from_friday() {
    let (db, ex) = setup();
    // 2024-01-05 is a Friday, next weekday should be Monday 2024-01-08
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-05"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = NEXT_WEEKDAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-01-08".to_string()))
    );
}

#[test]
fn test_prev_weekday_from_monday() {
    let (db, ex) = setup();
    // 2024-01-08 is a Monday, previous weekday should be Friday 2024-01-05
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-08"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = PREV_WEEKDAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-01-05".to_string()))
    );
}

#[test]
fn test_prev_business_day_from_monday() {
    let (db, ex) = setup();
    // 2024-01-08 is a Monday, previous business day should be Friday 2024-01-05
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-08"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = PREV_BUSINESS_DAY(dt) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("2024-01-05".to_string()))
    );
}

// ── BIZ_DAYS_BETWEEN ─────────────────────────────────────────────────────────

#[test]
fn test_biz_days_between_mon_to_fri() {
    let (db, ex) = setup();
    // 2024-01-08 (Mon) to 2024-01-12 (Fri) = 4 business days (Mon, Tue, Wed, Thu)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-01-08", "d2": "2024-01-12"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = BIZ_DAYS_BETWEEN(d1, d2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(4)));
}

#[test]
fn test_business_days_diff_alias() {
    let (db, ex) = setup();
    // Same test using alias
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"d1": "2024-01-08", "d2": "2024-01-12"}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE nv = BUSINESS_DAYS_DIFF(d1, d2) SELECT nv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("nv"), Some(&Value::Integer(4)));
}
