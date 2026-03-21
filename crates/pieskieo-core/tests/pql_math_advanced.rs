/// Integration tests for PQL advanced math built-in functions.
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
fn test_poly_eval() {
    let (db, ex) = setup();
    // 1 + 2*x + 3*x^2 at x=2 = 1+4+12 = 17
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"coeffs": [1,2,3], "xval": 2}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = POLY_EVAL(coeffs, xval) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 17.0).abs() < 0.001),
        other => panic!("expected 17.0, got {:?}", other),
    }
}

#[test]
fn test_polynomial_eval_alias() {
    let (db, ex) = setup();
    // 3 + 0*x + 1*x^2 at x=4 = 3+0+16 = 19
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"coeffs": [3,0,1], "xval": 4}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = POLYNOMIAL_EVAL(coeffs, xval) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 19.0).abs() < 0.001),
        other => panic!("expected 19.0, got {:?}", other),
    }
}

#[test]
fn test_poly_derive() {
    let (db, ex) = setup();
    // [1, 2, 3] = 1 + 2x + 3x^2 -> derivative = [2, 6]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"coeffs": [1,2,3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE deriv = POLY_DERIVE(coeffs) SELECT deriv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("deriv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::Float(2.0));
            assert_eq!(arr[1], Value::Float(6.0));
        }
        other => panic!("expected array [2.0,6.0], got {:?}", other),
    }
}

#[test]
fn test_poly_integrate() {
    let (db, ex) = setup();
    // [2, 6] = 2 + 6x -> integral = [0, 2, 3]  (0 + 2x + 3x^2)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"coeffs": [2,6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE intg = POLY_INTEGRATE(coeffs) SELECT intg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("intg") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Float(0.0));
            assert_eq!(arr[1], Value::Float(2.0));
            assert_eq!(arr[2], Value::Float(3.0));
        }
        other => panic!("expected array [0.0,2.0,3.0], got {:?}", other),
    }
}

#[test]
fn test_catalan_number() {
    let (db, ex) = setup();
    // C(0)=1, C(1)=1, C(2)=2, C(3)=5, C(4)=14
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 4}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = CATALAN_NUMBER(nval) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(14)));
}

#[test]
fn test_catalan_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = CATALAN(nval) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
}

#[test]
fn test_bell_number() {
    let (db, ex) = setup();
    // B(0)=1, B(1)=1, B(2)=2, B(3)=5, B(4)=15
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 4}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE b = BELL_NUMBER(nval) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(15)));
}

#[test]
fn test_digital_root() {
    let (db, ex) = setup();
    // digital root of 493 = 4+9+3=16 -> 1+6=7
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 493}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dr = DIGITAL_ROOT(nval) SELECT dr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dr"), Some(&Value::Integer(7)));
}

#[test]
fn test_digital_root_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dr = DIGITAL_ROOT(nval) SELECT dr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dr"), Some(&Value::Integer(0)));
}

#[test]
fn test_aliquot_sum() {
    let (db, ex) = setup();
    // proper divisors of 6: 1+2+3=6
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 6}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = ALIQUOT_SUM(nval) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("s"), Some(&Value::Integer(6)));
}

#[test]
fn test_euler_totient() {
    let (db, ex) = setup();
    // phi(9) = 6 (1,2,4,5,7,8 are coprime to 9)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 9}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE t = EULER_TOTIENT(nval) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("t"), Some(&Value::Integer(6)));
}

#[test]
fn test_totient_alias() {
    let (db, ex) = setup();
    // phi(12) = 4
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 12}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE t = TOTIENT(nval) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("t"), Some(&Value::Integer(4)));
}

#[test]
fn test_is_perfect() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"p": 28, "np": 10}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = IS_PERFECT(p) COMPUTE b = IS_PERFECT(np) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_abundant() {
    let (db, ex) = setup();
    // 12 is abundant (1+2+3+4+6=16>12)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"abn": 12, "not_abn": 5}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = IS_ABUNDANT(abn) COMPUTE b = IS_ABUNDANT(not_abn) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_deficient() {
    let (db, ex) = setup();
    // 5 is deficient (1<5)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dfn": 5, "not_dfn": 12}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = IS_DEFICIENT(dfn) COMPUTE b = IS_DEFICIENT(not_dfn) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_collatz_length() {
    let (db, ex) = setup();
    // Collatz sequence from 1: length=1
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cl = COLLATZ_LENGTH(nval) SELECT cl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cl"), Some(&Value::Integer(1)));
}

#[test]
fn test_collatz_length_27() {
    let (db, ex) = setup();
    // n=27 has Collatz length 112
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 27}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cl = COLLATZ_LEN(nval) SELECT cl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cl"), Some(&Value::Integer(112)));
}

#[test]
fn test_tribonacci() {
    let (db, ex) = setup();
    // T(7) = 24
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 7}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE t = TRIBONACCI(nval) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("t"), Some(&Value::Integer(24)));
}

#[test]
fn test_padovan_num() {
    let (db, ex) = setup();
    // P(7) = 4
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 7}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = PADOVAN_NUM(nval) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pv"), Some(&Value::Integer(4)));
}

#[test]
fn test_jacobsthal() {
    let (db, ex) = setup();
    // J(0)=0, J(1)=1, J(2)=1, J(3)=3, J(4)=5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 4}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE jv = JACOBSTHAL(nval) SELECT jv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("jv"), Some(&Value::Integer(5)));
}

#[test]
fn test_pell_number() {
    let (db, ex) = setup();
    // P(5) = 29
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 5}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = PELL_NUMBER(nval) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pv"), Some(&Value::Integer(29)));
}

#[test]
fn test_pell_num_alias() {
    let (db, ex) = setup();
    // P(1) = 1
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = PELL_NUM(nval) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pv"), Some(&Value::Integer(1)));
}

#[test]
fn test_sylvester_seq() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 3}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sv = SYLVESTER_SEQ(nval) SELECT sv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Integer(2));
            assert_eq!(arr[1], Value::Integer(3)); // 2+1
            assert_eq!(arr[2], Value::Integer(7)); // 2*3+1
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_continued_fraction() {
    let (db, ex) = setup();
    // 7/3 = [2; 3] -> [2, 3]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pv": 7, "qv": 3}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cf = CONTINUED_FRACTION(pv, qv) SELECT cf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cf") {
        Some(Value::Array(arr)) => assert!(!arr.is_empty()),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_prime_factors() {
    let (db, ex) = setup();
    // 12 = 2 * 2 * 3
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 12}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pf = PRIME_FACTORS(nval) SELECT pf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pf") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!(arr.contains(&Value::Integer(2)));
            assert!(arr.contains(&Value::Integer(3)));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_factorize_alias() {
    let (db, ex) = setup();
    // 13 is prime -> [13]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 13}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pf = FACTORIZE(nval) SELECT pf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pf") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], Value::Integer(13));
        }
        other => panic!("expected [13], got {:?}", other),
    }
}

#[test]
fn test_partition_count() {
    let (db, ex) = setup();
    // p(4) = 5 (partitions of 4: 4, 3+1, 2+2, 2+1+1, 1+1+1+1)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 4}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = PARTITION_COUNT(nval) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pv"), Some(&Value::Integer(5)));
}

#[test]
fn test_num_partitions_alias() {
    let (db, ex) = setup();
    // p(0) = 1 (empty partition)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nval": 0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = NUM_PARTITIONS(nval) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pv"), Some(&Value::Integer(1)));
}
