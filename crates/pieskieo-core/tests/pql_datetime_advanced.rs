/// Integration tests for advanced PQL date/time functions.
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
fn test_iso_week() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = ISO_WEEK(dt) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Jan 15, 2024 is ISO week 3
    assert_eq!(r.rows[0].data.get("w"), Some(&Value::Integer(3)));
}

#[test]
fn test_iso_week_aliases() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = CALENDAR_WEEK(dt) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("w"), Some(&Value::Integer(3)));
}

#[test]
fn test_iso_year() {
    let (db, ex) = setup();
    // 2024-01-01 is Monday, ISO week 1 of 2024 -> ISO year 2024
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE y = ISO_YEAR(dt) SELECT y;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("y"), Some(&Value::Integer(2024)));
}

#[test]
fn test_iso_year_week_year_alias() {
    let (db, ex) = setup();
    // 2020-12-31 is in ISO week 53 of 2020, ISO year 2020
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2020-12-31"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE y = WEEK_YEAR(dt) SELECT y;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("y"), Some(&Value::Integer(2020)));
}

#[test]
fn test_start_of_month() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-03-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = START_OF_MONTH(dt) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("s"),
        Some(&Value::String("2024-03-01".to_string()))
    );
}

#[test]
fn test_end_of_month() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-03-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE e = END_OF_MONTH(dt) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("e"),
        Some(&Value::String("2024-03-31".to_string()))
    );
}

#[test]
fn test_end_of_month_february_leap() {
    let (db, ex) = setup();
    // Feb 2024 is a leap year -> 29 days
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-02-10"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE e = LAST_DAY(dt) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("e"),
        Some(&Value::String("2024-02-29".to_string()))
    );
}

#[test]
fn test_end_of_month_december() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-12-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE e = MONTH_END(dt) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("e"),
        Some(&Value::String("2024-12-31".to_string()))
    );
}

#[test]
fn test_start_of_year() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-06-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = START_OF_YEAR(dt) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("s"),
        Some(&Value::String("2024-01-01".to_string()))
    );
}

#[test]
fn test_end_of_year() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-06-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE e = END_OF_YEAR(dt) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("e"),
        Some(&Value::String("2024-12-31".to_string()))
    );
}

#[test]
fn test_age_in_years() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "1990-05-15", "dt_to": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE a = AGE_IN_YEARS(dt_from, dt_to) SELECT a;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Birthday in May, end date is Jan 1 -> 33 full years (birthday hasn't passed yet in 2024)
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(33)));
}

#[test]
fn test_years_between_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2000-01-01", "dt_to": "2024-01-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE a = YEARS_BETWEEN(dt_from, dt_to) SELECT a;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(24)));
}

#[test]
fn test_age_in_months() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-03-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = AGE_IN_MONTHS(dt_from, dt_to) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Jan 1 to Mar 15 -> 2 full months (Jan and Feb passed)
    assert_eq!(r.rows[0].data.get("m"), Some(&Value::Integer(2)));
}

#[test]
fn test_months_between_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-15", "dt_to": "2024-03-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = MONTHS_BETWEEN(dt_from, dt_to) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("m"), Some(&Value::Integer(2)));
}

#[test]
fn test_age_in_days() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-01-08"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = AGE_IN_DAYS(dt_from, dt_to) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(7)));
}

#[test]
fn test_date_diff_days_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-02-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DATE_DIFF_DAYS(dt_from, dt_to) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // January 2024 has 31 days
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(31)));
}

#[test]
fn test_date_diff_weeks() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-01-22"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = DATE_DIFF_WEEKS(dt_from, dt_to) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // 21 days / 7 = 3 weeks
    assert_eq!(r.rows[0].data.get("w"), Some(&Value::Integer(3)));
}

#[test]
fn test_weeks_between_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-01-29"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = WEEKS_BETWEEN(dt_from, dt_to) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // 28 days / 7 = 4 weeks
    assert_eq!(r.rows[0].data.get("w"), Some(&Value::Integer(4)));
}

#[test]
fn test_date_diff_business_days() {
    let (db, ex) = setup();
    // Mon Jan 1 to Mon Jan 8, 2024
    // Business days between (exclusive start, inclusive end): Tue 2, Wed 3, Thu 4, Fri 5, Mon 8 = 5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-01-08"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE b = DATE_DIFF_BUSINESS_DAYS(dt_from, dt_to) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(5)));
}

#[test]
fn test_business_days_between_alias() {
    let (db, ex) = setup();
    // Mon Jan 1 to Fri Jan 5, 2024: Tue 2, Wed 3, Thu 4, Fri 5 = 4 business days
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt_from": "2024-01-01", "dt_to": "2024-01-05"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE b = BUSINESS_DAYS_BETWEEN(dt_from, dt_to) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(4)));
}

#[test]
fn test_next_weekday_from_friday() {
    let (db, ex) = setup();
    // Jan 5, 2024 is Friday -> next weekday is Monday Jan 8
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-05"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = NEXT_WEEKDAY(dt) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("n"),
        Some(&Value::String("2024-01-08".to_string()))
    );
}

#[test]
fn test_next_weekday_from_saturday() {
    let (db, ex) = setup();
    // Jan 6, 2024 is Saturday -> next weekday is Monday Jan 8
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-06"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = NEXT_BUSINESS_DAY(dt) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("n"),
        Some(&Value::String("2024-01-08".to_string()))
    );
}

#[test]
fn test_prev_weekday() {
    let (db, ex) = setup();
    // Jan 8, 2024 is Monday -> previous weekday is Friday Jan 5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-08"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE p = PREV_WEEKDAY(dt) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("p"),
        Some(&Value::String("2024-01-05".to_string()))
    );
}

#[test]
fn test_prev_weekday_from_sunday() {
    let (db, ex) = setup();
    // Jan 7, 2024 is Sunday -> previous weekday is Friday Jan 5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-07"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE p = PREVIOUS_BUSINESS_DAY(dt) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("p"),
        Some(&Value::String("2024-01-05".to_string()))
    );
}

#[test]
fn test_nth_weekday_in_month() {
    let (db, ex) = setup();
    // 3rd Monday (0=Mon) in January 2024:
    // Jan 1 = 1st Mon, Jan 8 = 2nd Mon, Jan 15 = 3rd Mon
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = NTH_WEEKDAY_IN_MONTH(2024, 1, 0, 3) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::String("2024-01-15".to_string()))
    );
}

#[test]
fn test_nth_weekday_in_month_friday() {
    let (db, ex) = setup();
    // 1st Friday (4=Fri) in January 2024:
    // Jan 1 = Mon, Jan 2 = Tue, Jan 3 = Wed, Jan 4 = Thu, Jan 5 = Fri -> 1st Friday
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = NTH_DOW_IN_MONTH(2024, 1, 4, 1) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::String("2024-01-05".to_string()))
    );
}

#[test]
fn test_fiscal_quarter_standard() {
    let (db, ex) = setup();
    // Standard fiscal year (Jan start), April = Q2
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-04-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE q = FISCAL_QUARTER(dt, 1) SELECT q;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("q"), Some(&Value::Integer(2)));
}

#[test]
fn test_fiscal_quarter_october_start() {
    let (db, ex) = setup();
    // Fiscal year starts October: Oct=Q1, Jan=Q2, Apr=Q3, Jul=Q4
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-01-15"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE q = FISCAL_Q(dt, 10) SELECT q;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Jan is 3 months after Oct start (Oct, Nov, Dec = Q1), so Jan = Q2
    assert_eq!(r.rows[0].data.get("q"), Some(&Value::Integer(2)));
}

#[test]
fn test_fiscal_quarter_april_start() {
    let (db, ex) = setup();
    // Fiscal year starts April: Apr=Q1
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "2024-04-01"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE q = FISCAL_QUARTER(dt, 4) SELECT q;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("q"), Some(&Value::Integer(1)));
}

#[test]
fn test_is_leap_year_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = IS_LEAP_YEAR(2024) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("l"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_leap_year_false_regular() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LEAP_YEAR(2023) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("l"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_leap_year_false_century() {
    let (db, ex) = setup();
    // 1900 is divisible by 100 but not 400 -> not a leap year
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = IS_LEAP_YEAR(1900) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("l"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_leap_year_true_400() {
    let (db, ex) = setup();
    // 2000 is divisible by 400 -> leap year
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = IS_LEAP_YEAR(2000) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("l"), Some(&Value::Bool(true)));
}

#[test]
fn test_days_in_month_regular() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DAYS_IN_MONTH(2024, 3) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(31)));
}

#[test]
fn test_days_in_month_february_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DAYS_IN_MONTH(2024, 2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(29)));
}

#[test]
fn test_days_in_month_february_non_leap() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = MONTH_DAYS(2023, 2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(28)));
}

#[test]
fn test_make_interval_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // 1 year, 2 months, 3 days
    let mut p = Parser::new(r#"QUERY t COMPUTE iv = MAKE_INTERVAL(1, 2, 3, 0, 0, 0) SELECT iv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("iv"),
        Some(&Value::String("P1Y2M3D".to_string()))
    );
}

#[test]
fn test_make_interval_with_time() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // 0 years, 0 months, 1 day, 2 hours, 30 minutes, 0 seconds
    let mut p = Parser::new(r#"QUERY t COMPUTE iv = MAKE_INTERVAL(0, 0, 1, 2, 30, 0) SELECT iv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("iv"),
        Some(&Value::String("P1DT2H30M".to_string()))
    );
}

#[test]
fn test_make_interval_zeros() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE iv = MAKE_INTERVAL(0, 0, 0, 0, 0, 0) SELECT iv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("iv"),
        Some(&Value::String("P0D".to_string()))
    );
}

#[test]
fn test_make_interval_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE iv = INTERVAL(0, 0, 0, 4, 0, 0) SELECT iv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("iv"),
        Some(&Value::String("PT4H".to_string()))
    );
}

#[test]
fn test_null_on_invalid_date() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dt": "not-a-date"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = ISO_WEEK(dt) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("w"), Some(&Value::Null));
}
