/// Integration tests for PQL advanced number theory and combinatorics built-in functions.
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

// ── Extended GCD ──────────────────────────────────────────────────────────────

#[test]
fn test_extended_gcd_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // gcd(35, 15) = 5
    let mut p = Parser::new(r#"QUERY t COMPUTE res = EXTENDED_GCD(35, 15) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("gcd"), Some(&Value::Integer(5)));
            // Verify Bezout identity: 35*x + 15*y == 5
            let x = match m.get("x") {
                Some(Value::Integer(v)) => *v,
                _ => panic!("no x"),
            };
            let y = match m.get("y") {
                Some(Value::Integer(v)) => *v,
                _ => panic!("no y"),
            };
            assert_eq!(35 * x + 15 * y, 5, "Bezout identity failed");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_extended_gcd_coprime() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // gcd(13, 7) = 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = EXTENDED_GCD(13, 7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("gcd"), Some(&Value::Integer(1)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_xgcd_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = XGCD(12, 8) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("gcd"), Some(&Value::Integer(4)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── MOD_INVERSE ───────────────────────────────────────────────────────────────

#[test]
fn test_mod_inverse_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 3^(-1) mod 7 = 5 because 3*5 = 15 ≡ 1 (mod 7)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOD_INVERSE(3, 7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(5)));
}

#[test]
fn test_mod_inverse_no_inverse() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // gcd(4, 6) = 2 != 1, so no inverse
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOD_INVERSE(4, 6) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Null));
}

#[test]
fn test_modular_inverse_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 2^(-1) mod 11 = 6 because 2*6 = 12 ≡ 1 (mod 11)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MODULAR_INVERSE(2, 11) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(6)));
}

// ── MOD_POW ───────────────────────────────────────────────────────────────────

#[test]
fn test_mod_pow_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 2^10 % 100 = 1024 % 100 = 24
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOD_POW(2, 10, 100) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(24)));
}

#[test]
fn test_mod_pow_fermat() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Fermat's little theorem: 2^6 ≡ 1 (mod 7)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOD_POW(2, 6, 7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(1)));
}

#[test]
fn test_modular_power_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 3^4 % 5 = 81 % 5 = 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MODULAR_POWER(3, 4, 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(1)));
}

// ── CHINESE_REMAINDER ─────────────────────────────────────────────────────────

#[test]
fn test_crt_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rs": [2, 3], "ms": [3, 5]}),
    )
    .unwrap();
    // x ≡ 2 (mod 3), x ≡ 3 (mod 5) => x = 8
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CHINESE_REMAINDER(rs, ms) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(8)));
}

#[test]
fn test_crt_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rs": [0, 3, 4], "ms": [3, 4, 5]}),
    )
    .unwrap();
    // x ≡ 0 (mod 3), x ≡ 3 (mod 4), x ≡ 4 (mod 5) => x = 39
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CRT(rs, ms) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let val = match r.rows[0].data.get("res") {
        Some(Value::Integer(v)) => *v,
        _ => panic!("not int"),
    };
    assert_eq!(val % 3, 0);
    assert_eq!(val % 4, 3);
    assert_eq!(val % 5, 4);
}

// ── PRIMITIVE_ROOT ────────────────────────────────────────────────────────────

#[test]
fn test_primitive_root_7() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Primitive root of 7 is 3
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PRIMITIVE_ROOT(7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(3)));
}

#[test]
fn test_primitive_root_5() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Primitive root of 5 is 2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIND_PRIMITIVE_ROOT(5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(2)));
}

// ── DISCRETE_LOG ──────────────────────────────────────────────────────────────

#[test]
fn test_discrete_log_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 2^x ≡ 8 (mod 11) -> x = 3 (since 2^3 = 8)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DISCRETE_LOG(2, 8, 11) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(3)));
}

// ── LUCAS_SEQUENCE ────────────────────────────────────────────────────────────

#[test]
fn test_lucas_num_base() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // L(0)=2, L(1)=1
    let mut p = Parser::new(r#"QUERY t COMPUTE a = LUCAS_NUM(0), b = LUCAS_NUM(1) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
}

#[test]
fn test_lucas_num_sequence() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // L(2)=3, L(3)=4, L(4)=7, L(5)=11
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = LUCAS_SEQUENCE(2), b = LUCAS_SEQUENCE(3), c = LUCAS_SEQUENCE(4), d = LUCAS_SEQUENCE(5) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(3)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(4)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(7)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(11)));
}

// ── MOTZKIN_NUMBER ────────────────────────────────────────────────────────────

#[test]
fn test_motzkin_numbers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // M(0)=1, M(1)=1, M(2)=2, M(3)=4, M(4)=9
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = MOTZKIN(0), b = MOTZKIN(1), c = MOTZKIN(2), d = MOTZKIN(3), e = MOTZKIN(4) SELECT a, b, c, d, e;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(4)));
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Integer(9)));
}

// ── DELANNOY_NUMBER ───────────────────────────────────────────────────────────

#[test]
fn test_delannoy_base() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // D(0,0)=1, D(1,0)=1, D(0,1)=1
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = DELANNOY(0, 0), b = DELANNOY(1, 0), c = DELANNOY(0, 1) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
}

#[test]
fn test_delannoy_values() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // D(1,1)=3, D(2,2)=13, D(2,1)=5
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = DELANNOY(1, 1), b = DELANNOY(2, 2), c = DELANNOY_NUMBER(2, 1) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(3)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(13)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(5)));
}

// ── NARAYANA_NUMBER ───────────────────────────────────────────────────────────

#[test]
fn test_narayana_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // N(1,1)=1, N(4,2)=6
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = NARAYANA(1, 1), b = NARAYANA(4, 2) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(6)));
}

// ── EULER_NUMBER ──────────────────────────────────────────────────────────────

#[test]
fn test_euler_numbers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // E(0)=1, E(2)=-1, E(4)=5
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = EULER_NUM(0), b = EULER_NUM(2), c = EULER_NUM(4), d = EULER_NUMBER(1) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(-1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(5)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(0))); // odd => 0
}

// ── BERNOULLI_NUMBER ──────────────────────────────────────────────────────────

#[test]
fn test_bernoulli_numbers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // B(0)=1, B(1)=-0.5, B(2)=1/6
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = BERNOULLI(0), b = BERNOULLI(1), c = BERNOULLI_NUMBER(2) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!((f - 1.0).abs() < 1e-9),
        other => panic!("expected 1.0, got {:?}", other),
    }
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!((f - (-0.5)).abs() < 1e-9),
        other => panic!("expected -0.5, got {:?}", other),
    }
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!((f - (1.0 / 6.0)).abs() < 1e-9),
        other => panic!("expected 1/6, got {:?}", other),
    }
}

// ── TANGENT_NUMBER ────────────────────────────────────────────────────────────

#[test]
fn test_tangent_numbers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // T(1)=1, T(3)=2, T(5)=16
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = TANGENT_NUMBER(1), b = ZAG_NUMBER(3), c = TANGENT_NUMBER(5), d = TANGENT_NUMBER(2) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(16)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(0))); // even => 0
}

// ── N_CHOOSE_K ────────────────────────────────────────────────────────────────

#[test]
fn test_n_choose_k() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // C(5,2)=10, C(10,3)=120, C(6,0)=1
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = N_CHOOSE_K(5, 2), b = N_CHOOSE_K(10, 3), c = N_CHOOSE_K(6, 0) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(10)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(120)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
}

// ── PERMUTATIONS ──────────────────────────────────────────────────────────────

#[test]
fn test_permutations() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // P(5,2)=20, P(4,4)=24
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = PERMUTATIONS(5, 2), b = N_PERM_K(4, 4) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(20)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(24)));
}

// ── STIRLING_FIRST ────────────────────────────────────────────────────────────

#[test]
fn test_stirling_first() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // c(4,2)=11, c(5,3)=35, c(3,1)=2
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = STIRLING_FIRST(4, 2), b = STIRLING1(5, 3), c = STIRLING_FIRST(3, 1) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(11)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(35)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(2)));
}

// ── STIRLING_SECOND ───────────────────────────────────────────────────────────

#[test]
fn test_stirling_second() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // S(4,2)=7, S(5,3)=25, S(3,1)=1
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = STIRLING_SECOND(4, 2), b = STIRLING2(5, 3), c = STIRLING_SECOND(3, 1) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(7)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(25)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
}

// ── NECKLACE_COUNT ────────────────────────────────────────────────────────────

#[test]
fn test_necklace_count() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Necklaces of length 3 with 2 colors = 4
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NECKLACE_COUNT(3, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(4)));
}

#[test]
fn test_necklaces_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Necklaces of length 4 with 2 colors = 6
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NECKLACES(4, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(6)));
}

// ── DERANGEMENT ───────────────────────────────────────────────────────────────

#[test]
fn test_derangement() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // D(0)=1, D(1)=0, D(2)=1, D(3)=2, D(4)=9
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = DERANGEMENT(0), b = DERANGEMENT(1), c = SUBFACTORIAL(2), d = DERANGEMENT(3), e = DERANGEMENT(4) SELECT a, b, c, d, e;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(0)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Integer(9)));
}

// ── SET_PARTITIONS ────────────────────────────────────────────────────────────

#[test]
fn test_set_partitions() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // B(0)=1, B(1)=1, B(2)=2, B(3)=5, B(4)=15
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = SET_PARTITIONS(0), b = BELL_PARTITION(1), c = SET_PARTITIONS(2), d = SET_PARTITIONS(3), e = SET_PARTITIONS(4) SELECT a, b, c, d, e;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(5)));
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Integer(15)));
}

// ── INTEGER_PARTITIONS ────────────────────────────────────────────────────────

#[test]
fn test_integer_partitions() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // p(5,2)=2: {4+1, 3+2}, p(5,3)=2: {3+1+1, 2+2+1}, p(4,2)=2: {3+1,2+2}
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = INTEGER_PARTITIONS(5, 2), b = PARTITION_INTO(5, 3), c = INTEGER_PARTITIONS(4, 2) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(2)));
}

// ── SIGMA_FUNCTION ────────────────────────────────────────────────────────────

#[test]
fn test_sigma_function() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // sigma(1)=1, sigma(6)=12, sigma(12)=28
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = SIGMA_FUNCTION(1), b = PERFECT_NUMBER_SIGMA(6), c = SIGMA_FUNCTION(12) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(12)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(28)));
}

// ── LIOUVILLE_FUNCTION ────────────────────────────────────────────────────────

#[test]
fn test_liouville_function() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // L(1) = 1 (0 prime factors), L(2) = -1 (1 prime factor), L(4) = 1 (2 prime factors)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = LIOUVILLE_FUNCTION(1), b = LIOUVILLE(2), c = LIOUVILLE(4) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(-1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
}

// ── VON_MANGOLDT ──────────────────────────────────────────────────────────────

#[test]
fn test_von_mangoldt() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Lambda(1)=0, Lambda(2)=ln(2), Lambda(4)=ln(2), Lambda(6)=0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = MANGOLDT_FUNCTION(1), b = VON_MANGOLDT(2), c = MANGOLDT_FUNCTION(4), d = VON_MANGOLDT(6) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Float(0.0)));
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!((f - 2f64.ln()).abs() < 1e-9),
        other => panic!("expected ln(2), got {:?}", other),
    }
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!((f - 2f64.ln()).abs() < 1e-9), // 4=2^2, so ln(2)
        other => panic!("expected ln(2) for 4=2^2, got {:?}", other),
    }
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Float(0.0))); // 6=2*3, not prime power
}

// ── MOBIUS_FUNCTION ───────────────────────────────────────────────────────────

#[test]
fn test_mobius_function() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // mu(1)=1, mu(2)=-1, mu(6)=1, mu(4)=0 (4=2^2 has square factor)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = MOBIUS_FUNCTION(1), b = MOBIUS(2), c = MOBIUS(6), d = MOBIUS(4) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(-1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(0)));
}

// ── RADICAL ───────────────────────────────────────────────────────────────────

#[test]
fn test_radical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // rad(12)=2*3=6, rad(1)=1, rad(8)=2
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = RADICAL(12), b = RADICAL_N(1), c = RADICAL(8) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(6)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(2)));
}

// ── PRIMES_UP_TO ──────────────────────────────────────────────────────────────

#[test]
fn test_primes_up_to() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Primes up to 20: 2,3,5,7,11,13,17,19
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PRIMES_UP_TO(20) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 8);
            assert_eq!(arr[0], Value::Integer(2));
            assert_eq!(arr[7], Value::Integer(19));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_sieve_primes_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SIEVE_PRIMES(10) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(
                arr,
                &vec![
                    Value::Integer(2),
                    Value::Integer(3),
                    Value::Integer(5),
                    Value::Integer(7)
                ]
            );
        }
        other => panic!("expected [2,3,5,7], got {:?}", other),
    }
}

// ── NTH_PRIME ─────────────────────────────────────────────────────────────────

#[test]
fn test_nth_prime() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 1st=2, 2nd=3, 3rd=5, 4th=7, 5th=11, 10th=29
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = NTH_PRIME(1), b = NTH_PRIME(2), c = PRIME_N(3), d = NTH_PRIME(10) SELECT a, b, c, d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(2)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(3)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(5)));
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(29)));
}

// ── PRIME_PI ──────────────────────────────────────────────────────────────────

#[test]
fn test_prime_pi() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // pi(10)=4, pi(100)=25, pi(1)=0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = PRIME_PI(10), b = PRIME_COUNT(100), c = PRIME_PI(1) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(4)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(25)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(0)));
}

// ── PREV_PRIME ────────────────────────────────────────────────────────────────

#[test]
fn test_prev_prime() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // prev_prime(10)=7, prev_prime(8)=7, prev_prime(3)=2
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = PREV_PRIME(10), b = PREV_PRIME_BEFORE(8), c = PREV_PRIME(3) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(7)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(7)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(2)));
}

#[test]
fn test_prev_prime_null_for_small() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // prev_prime(2) = Null (no prime < 2)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PREV_PRIME(2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Null));
}

// ── IS_SMOOTH ─────────────────────────────────────────────────────────────────

#[test]
fn test_is_smooth() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 12=2^2*3 is 3-smooth (all factors <=3), 15=3*5 is not 3-smooth
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = IS_SMOOTH(12, 3), b = B_SMOOTH(15, 3), c = IS_SMOOTH(1, 2) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Bool(true)));
}

// ── SMOOTH_NUMBERS ────────────────────────────────────────────────────────────

#[test]
fn test_smooth_numbers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 2-smooth numbers up to 16: 1, 2, 4, 8, 16
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SMOOTH_NUMBERS(16, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(
                arr,
                &vec![
                    Value::Integer(1),
                    Value::Integer(2),
                    Value::Integer(4),
                    Value::Integer(8),
                    Value::Integer(16)
                ]
            );
        }
        other => panic!("expected [1,2,4,8,16], got {:?}", other),
    }
}

#[test]
fn test_find_smooth_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 3-smooth numbers up to 12: 1,2,3,4,6,8,9,12
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIND_SMOOTH(12, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            let expected = vec![1i64, 2, 3, 4, 6, 8, 9, 12];
            let got: Vec<i64> = arr
                .iter()
                .map(|v| match v {
                    Value::Integer(i) => *i,
                    _ => -1,
                })
                .collect();
            assert_eq!(got, expected);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── Bracelet count ────────────────────────────────────────────────────────────

#[test]
fn test_bracelet_count() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Bracelets of length 3 with 2 colors
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BRACELET_COUNT(3, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Should be a positive integer
    match r.rows[0].data.get("res") {
        Some(Value::Integer(v)) => assert!(*v > 0, "expected positive, got {}", v),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── Additional edge cases ──────────────────────────────────────────────────────

#[test]
fn test_motzkin_number_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOTZKIN_NUMBER(5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // M(5) = 21
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(21)));
}

#[test]
fn test_narayana_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NARAYANA_NUMBER(3, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // N(3,2) = C(3,2)*C(3,1)/3 = 3*3/3 = 3
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(3)));
}

#[test]
fn test_mod_pow_exp_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // any^0 mod m = 1 for m > 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOD_POW(5, 0, 7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(1)));
}

#[test]
fn test_nth_prime_large() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 25th prime is 97
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTH_PRIME(25) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(97)));
}

#[test]
fn test_sterling_second_boundary() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // S(n,1)=1 for any n>=1, S(n,n)=1
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = STIRLING2(5, 1), b = STIRLING2(5, 5) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
}

#[test]
fn test_mobius_squarefree_three_primes() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // mu(30) = mu(2*3*5) = (-1)^3 = -1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MOBIUS(30) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(-1)));
}

#[test]
fn test_integer_partitions_edge() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // p(n, 1) = 1 always (only one partition: n itself), p(n, n) = 1 (all ones)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = INTEGER_PARTITIONS(7, 1), b = PARTITION_INTO(4, 4) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Integer(1)));
}

#[test]
fn test_primes_up_to_small() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // No primes below 2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PRIMES_UP_TO(1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Array(vec![])));
}
