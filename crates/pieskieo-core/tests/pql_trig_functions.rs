/// Integration tests for PQL hyperbolic and advanced math functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_sinh_cosh_tanh() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1.0}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = SINH(x) COMPUTE b = COSH(x) COMPUTE c = TANH(x) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.1752_f64).abs() < 0.001,
            "sinh(1) ≈ 1.1752, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.5431_f64).abs() < 0.001,
            "cosh(1) ≈ 1.5431, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.7616_f64).abs() < 0.001,
            "tanh(1) ≈ 0.7616, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_asinh_acosh_atanh() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1.0, "y": 0.5}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = ASINH(x) COMPUTE b = ACOSH(x) COMPUTE c = ATANH(y) SELECT a, b, c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.8814_f64).abs() < 0.001,
            "asinh(1) ≈ 0.8814, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(f.abs() < 0.0001, "acosh(1) = 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.5493_f64).abs() < 0.001,
            "atanh(0.5) ≈ 0.5493, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_erf_erfc() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 1.0}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE a = ERF(x) COMPUTE b = ERFC(x) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let erf_val = match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => {
            assert!(
                (*f - 0.8427_f64).abs() < 0.001,
                "erf(1) ≈ 0.8427, got {}",
                f
            );
            *f
        }
        other => panic!("expected float, got {:?}", other),
    };
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(
            ((erf_val + *f) - 1.0).abs() < 0.001,
            "erf + erfc = 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_log1p_expm1() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 0.0}),
    )
    .unwrap();

    // log1p(0) = 0; expm1(0) = 0
    let mut p = Parser::new(r#"QUERY t COMPUTE a = LOG1P(x) COMPUTE b = EXPM1(x) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "log1p(0) = 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "expm1(0) = 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_gamma_cbrt_square() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 5.0, "y": 8.0}),
    )
    .unwrap();

    // GAMMA(5) = 4! = 24; CBRT(8) = 2; SQUARE(5) = 25
    let mut p = Parser::new(
        r#"QUERY t COMPUTE g = GAMMA(x) COMPUTE c = CBRT(y) COMPUTE sq = SQUARE(x) SELECT g, c, sq;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!((*f - 24.0).abs() < 0.1, "GAMMA(5) = 24, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 0.0001, "CBRT(8) = 2, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
    match r.rows[0].data.get("sq") {
        Some(Value::Float(f)) => assert!((*f - 25.0).abs() < 0.0001, "SQUARE(5) = 25, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_trim_mean() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
        db.put_doc_ns(
            None,
            Some("data"),
            Uuid::new_v4(),
            serde_json::json!({"val": v}),
        )
        .unwrap();
    }

    // 10% trim removes 1 value from each end: [2..9], mean = 5.5
    let mut p = Parser::new(
        r#"QUERY data COMPUTE g = 1 GROUP BY g COMPUTE m = TRIM_MEAN(val, 10) SELECT m;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => assert!(
            (*f - 5.5).abs() < 0.01,
            "trimmed mean should be 5.5, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}
