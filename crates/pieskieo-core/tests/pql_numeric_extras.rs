/// Integration tests for PQL numeric precision and math extra functions.
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
fn test_gcd_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 48, "b": 18}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE g = GCD(a, b) SELECT g;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("g"), Some(&Value::Integer(6)));
}

#[test]
fn test_lcm() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 4, "b": 6}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE l = LCM(a, b) SELECT l;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("l"), Some(&Value::Integer(12)));
}

#[test]
fn test_is_prime() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 7}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_PRIME(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_prime_not_prime() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 1}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_PRIME(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

#[test]
fn test_prime_factors() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 12}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE f = PRIME_FACTORS(n) SELECT f;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("f"),
        Some(&Value::Array(vec![
            Value::Integer(2),
            Value::Integer(2),
            Value::Integer(3)
        ]))
    );
}

#[test]
fn test_prime_factors_one() {
    // PRIME_FACTORS(1) should return empty list
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 1}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE f = PRIME_FACTORS(n) SELECT f;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("f"), Some(&Value::Array(vec![])));
}

#[test]
fn test_digits_sum() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 123}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE s = DIGITS_SUM(n) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("s"), Some(&Value::Integer(6)));
}

#[test]
fn test_digital_root() {
    let (db, ex) = setup();
    // 4+9+3=16 -> 1+6=7
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 493}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = DIGITAL_ROOT(n) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(7)));
}

#[test]
fn test_next_power_of_2() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = NEXT_POWER_OF_2(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

#[test]
fn test_next_power_of_2_exact() {
    // 8 is already a power of 2 — should return 8
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 8}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = NEXT_POWER_OF_2(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

#[test]
fn test_is_armstrong() {
    let (db, ex) = setup();
    // 1^3 + 5^3 + 3^3 = 1 + 125 + 27 = 153
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 153}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_ARMSTRONG(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_armstrong_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 100}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_ARMSTRONG(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

#[test]
fn test_base_convert_hex() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 255}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BASE_CONVERT(n, 16) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("ff".to_string()))
    );
}

#[test]
fn test_base_convert_binary() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 10}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BASE_CONVERT(n, 2) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::String("1010".to_string()))
    );
}

#[test]
fn test_round_half_up() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 2.5}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = ROUND_HALF_UP(x, 0) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Float(3.0)));
}

#[test]
fn test_round_half_even() {
    let (db, ex) = setup();
    // 0.5 rounds to 0 (nearest even)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 0.5}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = ROUND_HALF_EVEN(x, 0) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Float(0.0)));
}

#[test]
fn test_truncate_to() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 3.789}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = TRUNCATE_TO(x, 2) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!((*f - 3.78).abs() < 1e-9, "got {}", f),
        other => panic!("expected Float(3.78), got {:?}", other),
    }
}

#[test]
fn test_significant_figures() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1234.5}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = SIGNIFICANT_FIGURES(x, 3) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!((*f - 1230.0).abs() < 1.0, "got {}", f),
        other => panic!("expected Float near 1230, got {:?}", other),
    }
}

#[test]
fn test_safe_mod_negative() {
    let (db, ex) = setup();
    // SAFE_MOD(-7, 3) -> 2
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": -7, "b": 3}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = SAFE_MOD(a, b) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(2)));
}

#[test]
fn test_is_power_of_2() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 16}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_POWER_OF_2(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_power_of_2_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_POWER_OF_2(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

#[test]
fn test_prev_power_of_2() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = PREV_POWER_OF_2(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(4)));
}

#[test]
fn test_coprime() {
    let (db, ex) = setup();
    // GCD(8, 15) = 1
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 8, "b": 15}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = COPRIME(a, b) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

#[test]
fn test_totient() {
    let (db, ex) = setup();
    // phi(9) = 6
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 9}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = TOTIENT(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(6)));
}

#[test]
fn test_next_prime() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 10}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = NEXT_PRIME(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(11)));
}

#[test]
fn test_reverse_number() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 123}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = REVERSE_NUMBER(n) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(321)));
}
