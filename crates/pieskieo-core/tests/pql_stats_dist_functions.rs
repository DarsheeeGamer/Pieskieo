/// Integration tests for PQL statistical distribution functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db_with_val(
    ns: &str,
    key: &str,
    val: f64,
) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some(ns),
        Uuid::new_v4(),
        serde_json::json!({key: val}),
    )
    .unwrap();
    (dir, db, ex)
}

#[test]
fn test_normal_pdf() {
    // Normal PDF at mean (x=0, mean=0, std=1) should be 1/sqrt(2*pi) ≈ 0.3989
    let (_dir, _db, ex) = make_db_with_val("t", "x", 0.0);
    let mut p = Parser::new("QUERY t COMPUTE pdf = NORMAL_PDF(x, 0.0, 1.0) SELECT pdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pdf") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.3989422804014327).abs() < 0.001,
            "expected ~0.3989 got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_norm_pdf_alias() {
    // NORM_PDF alias should produce same result
    let (_dir, _db, ex) = make_db_with_val("t2", "x", 0.0);
    let mut p = Parser::new("QUERY t2 COMPUTE pdf = NORM_PDF(x, 0.0, 1.0) SELECT pdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pdf") {
        Some(Value::Float(f)) => assert!((*f - 0.3989).abs() < 0.001, "expected ~0.3989 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_normal_cdf_at_zero() {
    // CDF at 0 with standard normal should be exactly 0.5
    let (_dir, _db, ex) = make_db_with_val("t3", "x", 0.0);
    let mut p = Parser::new("QUERY t3 COMPUTE c = NORMAL_CDF(x, 0.0, 1.0) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.01, "expected ~0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_normal_cdf_phi_alias() {
    // PHI alias, CDF at 1.96 ≈ 0.975
    let (_dir, _db, ex) = make_db_with_val("t4", "x", 1.96);
    let mut p = Parser::new("QUERY t4 COMPUTE c = PHI(x, 0.0, 1.0) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!((*f - 0.975).abs() < 0.005, "expected ~0.975 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_poisson_pmf() {
    // P(X=2; lambda=2) = e^-2 * 2^2 / 2! = e^-2 * 2 ≈ 0.2707
    let (_dir, _db, ex) = make_db_with_val("t5", "k", 2.0);
    let mut p = Parser::new("QUERY t5 COMPUTE pmf = POISSON_PMF(k, 2.0) SELECT pmf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pmf") {
        Some(Value::Float(f)) => assert!((*f - 0.2707).abs() < 0.001, "expected ~0.2707 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_poisson_cdf() {
    // P(X<=2; lambda=2) = P(0) + P(1) + P(2) ≈ 0.1353 + 0.2707 + 0.2707 = 0.6767
    let (_dir, _db, ex) = make_db_with_val("t6", "k", 2.0);
    let mut p = Parser::new("QUERY t6 COMPUTE cdf = POISSON_CDF(k, 2.0) SELECT cdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cdf") {
        Some(Value::Float(f)) => assert!((*f - 0.6767).abs() < 0.005, "expected ~0.6767 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_binomial_pmf() {
    // P(X=3; n=10, p=0.5) = C(10,3) * 0.5^3 * 0.5^7 = 120/1024 ≈ 0.1172
    let (_dir, _db, ex) = make_db_with_val("t7", "k", 3.0);
    let mut p = Parser::new("QUERY t7 COMPUTE pmf = BINOMIAL_PMF(k, 10.0, 0.5) SELECT pmf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pmf") {
        Some(Value::Float(f)) => assert!((*f - 0.1172).abs() < 0.001, "expected ~0.1172 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_binomial_cdf() {
    // P(X<=3; n=10, p=0.5) should be about 0.1719
    let (_dir, _db, ex) = make_db_with_val("t8", "k", 3.0);
    let mut p = Parser::new("QUERY t8 COMPUTE cdf = BINOMIAL_CDF(k, 10.0, 0.5) SELECT cdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cdf") {
        Some(Value::Float(f)) => assert!((*f - 0.1719).abs() < 0.002, "expected ~0.1719 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_exponential_pdf() {
    // f(1; rate=1) = 1 * e^-1 ≈ 0.3679
    let (_dir, _db, ex) = make_db_with_val("t9", "x", 1.0);
    let mut p = Parser::new("QUERY t9 COMPUTE pdf = EXPONENTIAL_PDF(x, 1.0) SELECT pdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pdf") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_exponential_cdf() {
    // F(1; rate=1) = 1 - e^-1 ≈ 0.6321
    let (_dir, _db, ex) = make_db_with_val("t10", "x", 1.0);
    let mut p = Parser::new("QUERY t10 COMPUTE cdf = EXPONENTIAL_CDF(x, 1.0) SELECT cdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cdf") {
        Some(Value::Float(f)) => assert!((*f - 0.6321).abs() < 0.001, "expected ~0.6321 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_uniform_pdf_inside() {
    // f(0.5; a=0, b=1) = 1/(1-0) = 1.0
    let (_dir, _db, ex) = make_db_with_val("t11", "x", 0.5);
    let mut p = Parser::new("QUERY t11 COMPUTE pdf = UNIFORM_PDF(x, 0.0, 1.0) SELECT pdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pdf") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-9, "expected 1.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_uniform_pdf_outside() {
    // f(2.0; a=0, b=1) = 0.0  (outside the range)
    let (_dir, _db, ex) = make_db_with_val("t12", "x", 2.0);
    let mut p = Parser::new("QUERY t12 COMPUTE pdf = UNIFORM_PDF(x, 0.0, 1.0) SELECT pdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pdf") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 1e-9, "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_uniform_cdf() {
    // F(0.25; a=0, b=1) = 0.25
    let (_dir, _db, ex) = make_db_with_val("t13", "x", 0.25);
    let mut p = Parser::new("QUERY t13 COMPUTE cdf = UNIFORM_CDF(x, 0.0, 1.0) SELECT cdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cdf") {
        Some(Value::Float(f)) => assert!((*f - 0.25).abs() < 1e-9, "expected 0.25 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_z_score() {
    // Z_SCORE(5, 3, 2) = (5-3)/2 = 1.0
    let (_dir, _db, ex) = make_db_with_val("t14", "x", 5.0);
    let mut p = Parser::new("QUERY t14 COMPUTE z = Z_SCORE(x, 3.0, 2.0) SELECT z;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("z") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-9, "expected 1.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_probit_at_half() {
    // PROBIT(0.5) = 0.0  (median of standard normal)
    let (_dir, _db, ex) = make_db_with_val("t15", "p", 0.5);
    let mut p = Parser::new("QUERY t15 COMPUTE q = PROBIT(p) SELECT q;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("q") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.001, "expected ~0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_probit_at_975() {
    // PROBIT(0.975) ≈ 1.96  (standard 95% CI upper bound)
    let (_dir, _db, ex) = make_db_with_val("t16", "p", 0.975);
    let mut p = Parser::new("QUERY t16 COMPUTE q = PROBIT(p) SELECT q;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("q") {
        Some(Value::Float(f)) => assert!((*f - 1.96).abs() < 0.01, "expected ~1.96 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_probit_invalid_returns_nan() {
    // PROBIT(0) should return NaN
    let (_dir, _db, ex) = make_db_with_val("t17", "p", 0.0);
    let mut p = Parser::new("QUERY t17 COMPUTE q = PROBIT(p) SELECT q;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("q") {
        Some(Value::Float(f)) => assert!(f.is_nan(), "expected NaN got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_exp_pdf_alias() {
    // EXP_PDF alias: f(0; rate=2) = 2 * e^0 = 2.0
    let (_dir, _db, ex) = make_db_with_val("t18", "x", 0.0);
    let mut p = Parser::new("QUERY t18 COMPUTE pdf = EXP_PDF(x, 2.0) SELECT pdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pdf") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 1e-9, "expected 2.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_exp_cdf_alias() {
    // EXP_CDF alias: F(0; rate=1) = 0.0
    let (_dir, _db, ex) = make_db_with_val("t19", "x", 0.0);
    let mut p = Parser::new("QUERY t19 COMPUTE cdf = EXP_CDF(x, 1.0) SELECT cdf;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cdf") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 1e-9, "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_standardize_value_alias() {
    // STANDARDIZE_VALUE alias for Z_SCORE
    let (_dir, _db, ex) = make_db_with_val("t20", "x", 10.0);
    let mut p = Parser::new("QUERY t20 COMPUTE z = STANDARDIZE_VALUE(x, 10.0, 5.0) SELECT z;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("z") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 1e-9, "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}
