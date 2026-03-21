/// Integration tests for PQL probability distribution and statistics functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db_f64(ns: &str, field: &str, val: f64) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some(ns),
        Uuid::new_v4(),
        serde_json::json!({ field: val }),
    )
    .unwrap();
    (dir, db, ex)
}

fn make_db_str(ns: &str, field: &str, val: &str) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some(ns),
        Uuid::new_v4(),
        serde_json::json!({ field: val }),
    )
    .unwrap();
    (dir, db, ex)
}

fn make_db_multi(
    ns: &str,
    fields: &[(&str, f64)],
) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let mut obj = serde_json::Map::new();
    for (k, v) in fields {
        obj.insert(k.to_string(), serde_json::Value::from(*v));
    }
    db.put_doc_ns(
        None,
        Some(ns),
        Uuid::new_v4(),
        serde_json::Value::Object(obj),
    )
    .unwrap();
    (dir, db, ex)
}

// ── Test 1: NORMAL_PDF at the mean ────────────────────────────────────────────
#[test]
fn test_normal_pdf_at_mean() {
    // f(0 | mu=0, sigma=1) = 1/sqrt(2*pi) ≈ 0.3989
    let (_dir, _db, ex) = make_db_f64("nd01", "xv", 0.0);
    let mut p = Parser::new("QUERY nd01 COMPUTE rv = NORMAL_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.3989422804).abs() < 0.001,
            "expected ~0.3989 got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 2: GAUSSIAN_PDF alias ────────────────────────────────────────────────
#[test]
fn test_gaussian_pdf_alias() {
    // GAUSSIAN_PDF is alias for NORMAL_PDF
    let (_dir, _db, ex) = make_db_f64("nd02", "xv", 0.0);
    let mut p = Parser::new("QUERY nd02 COMPUTE rv = GAUSSIAN_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3989).abs() < 0.001, "expected ~0.3989 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 3: NORMAL_CDF at zero ────────────────────────────────────────────────
#[test]
fn test_normal_cdf_at_zero() {
    // CDF(0, 0, 1) = 0.5
    let (_dir, _db, ex) = make_db_f64("nd03", "xv", 0.0);
    let mut p = Parser::new("QUERY nd03 COMPUTE rv = NORMAL_CDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.01, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 4: NORMAL_CDF at 1.96 ────────────────────────────────────────────────
#[test]
fn test_normal_cdf_at_1_96() {
    // CDF(1.96, 0, 1) ≈ 0.975
    let (_dir, _db, ex) = make_db_f64("nd04", "xv", 1.96);
    let mut p = Parser::new("QUERY nd04 COMPUTE rv = NORMAL_CDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.975).abs() < 0.005, "expected ~0.975 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 5: GAUSSIAN_CDF alias ────────────────────────────────────────────────
#[test]
fn test_gaussian_cdf_alias() {
    // GAUSSIAN_CDF(0, 0, 1) = 0.5
    let (_dir, _db, ex) = make_db_f64("nd05", "xv", 0.0);
    let mut p = Parser::new("QUERY nd05 COMPUTE rv = GAUSSIAN_CDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.01, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 6: BINOMIAL_PMF(k=2, n=5, p=0.5) ────────────────────────────────────
#[test]
fn test_binomial_pmf_k2_n5_p05() {
    // C(5,2)*0.5^2*0.5^3 = 10*0.25*0.125 = 0.3125
    let (_dir, _db, ex) = make_db_f64("nd06", "kv", 2.0);
    let mut p = Parser::new("QUERY nd06 COMPUTE rv = BINOMIAL_PMF(kv, 5.0, 0.5) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3125).abs() < 0.0001, "expected 0.3125 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 7: BINOM_PMF alias ───────────────────────────────────────────────────
#[test]
fn test_binom_pmf_alias() {
    // BINOM_PMF(2, 5, 0.5) = 0.3125
    let (_dir, _db, ex) = make_db_f64("nd07", "kv", 2.0);
    let mut p = Parser::new("QUERY nd07 COMPUTE rv = BINOM_PMF(kv, 5.0, 0.5) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3125).abs() < 0.0001, "expected 0.3125 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 8: POISSON_PMF(k=0, lambda=1) ───────────────────────────────────────
#[test]
fn test_poisson_pmf_k0_lambda1() {
    // e^(-1) ≈ 0.3679
    let (_dir, _db, ex) = make_db_f64("nd08", "kv", 0.0);
    let mut p = Parser::new("QUERY nd08 COMPUTE rv = POISSON_PMF(kv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 9: POISSON_PROB alias ────────────────────────────────────────────────
#[test]
fn test_poisson_prob_alias() {
    // POISSON_PROB(0, 1) = e^(-1) ≈ 0.3679
    let (_dir, _db, ex) = make_db_f64("nd09", "kv", 0.0);
    let mut p = Parser::new("QUERY nd09 COMPUTE rv = POISSON_PROB(kv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 10: EXPONENTIAL_PDF(1.0, 1.0) ───────────────────────────────────────
#[test]
fn test_exponential_pdf_x1_lambda1() {
    // lambda * exp(-lambda*x) = 1 * e^(-1) ≈ 0.3679
    let (_dir, _db, ex) = make_db_f64("nd10", "xv", 1.0);
    let mut p = Parser::new("QUERY nd10 COMPUTE rv = EXPONENTIAL_PDF(xv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 11: EXP_PDF alias ────────────────────────────────────────────────────
#[test]
fn test_exp_pdf_alias() {
    // EXP_PDF(1.0, 1.0) = e^(-1) ≈ 0.3679
    let (_dir, _db, ex) = make_db_f64("nd11", "xv", 1.0);
    let mut p = Parser::new("QUERY nd11 COMPUTE rv = EXP_PDF(xv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 12: EXPONENTIAL_CDF(0.0, 1.0) = 0.0 ────────────────────────────────
#[test]
fn test_exponential_cdf_at_zero() {
    // F(0, 1) = 1 - e^0 = 0.0
    let (_dir, _db, ex) = make_db_f64("nd12", "xv", 0.0);
    let mut p = Parser::new("QUERY nd12 COMPUTE rv = EXPONENTIAL_CDF(xv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.0001, "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 13: EXPONENTIAL_CDF(100.0, 1.0) ≈ 1.0 ───────────────────────────────
#[test]
fn test_exponential_cdf_at_large_x() {
    // F(100, 1) = 1 - e^(-100) ≈ 1.0
    let (_dir, _db, ex) = make_db_f64("nd13", "xv", 100.0);
    let mut p = Parser::new("QUERY nd13 COMPUTE rv = EXPONENTIAL_CDF(xv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.0001, "expected ~1.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 14: EXP_CDF alias ────────────────────────────────────────────────────
#[test]
fn test_exp_cdf_alias() {
    // EXP_CDF(1.0, 1.0) = 1 - e^(-1) ≈ 0.6321
    let (_dir, _db, ex) = make_db_f64("nd14", "xv", 1.0);
    let mut p = Parser::new("QUERY nd14 COMPUTE rv = EXP_CDF(xv, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.6321).abs() < 0.001, "expected ~0.6321 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 15: UNIFORM_PDF(0.5, 0.0, 1.0) = 1.0 ───────────────────────────────
#[test]
fn test_uniform_pdf_inside_range() {
    // f(0.5 | a=0, b=1) = 1/(1-0) = 1.0
    let (_dir, _db, ex) = make_db_f64("nd15", "xv", 0.5);
    let mut p = Parser::new("QUERY nd15 COMPUTE rv = UNIFORM_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.0001, "expected 1.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 16: UNIFORM_PDF(2.0, 0.0, 1.0) = 0.0 (outside range) ───────────────
#[test]
fn test_uniform_pdf_outside_range() {
    // x=2.0 is outside [0,1], so pdf = 0
    let (_dir, _db, ex) = make_db_f64("nd16", "xv", 2.0);
    let mut p = Parser::new("QUERY nd16 COMPUTE rv = UNIFORM_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.0001, "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 17: UNIFORM_DENSITY alias ───────────────────────────────────────────
#[test]
fn test_uniform_density_alias() {
    // UNIFORM_DENSITY(0.5, 0.0, 1.0) = 1.0
    let (_dir, _db, ex) = make_db_f64("nd17", "xv", 0.5);
    let mut p = Parser::new("QUERY nd17 COMPUTE rv = UNIFORM_DENSITY(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.0001, "expected 1.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 18: GEOMETRIC_PMF(k=1, p=0.5) = 0.5 ────────────────────────────────
#[test]
fn test_geometric_pmf_k1_p05() {
    // P(X=1) = (1-0.5)^0 * 0.5 = 0.5
    let (_dir, _db, ex) = make_db_f64("nd18", "kv", 1.0);
    let mut p = Parser::new("QUERY nd18 COMPUTE rv = GEOMETRIC_PMF(kv, 0.5) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.0001, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 19: GEOMETRIC_PMF(k=2, p=0.5) = 0.25 ───────────────────────────────
#[test]
fn test_geometric_pmf_k2_p05() {
    // P(X=2) = (1-0.5)^1 * 0.5 = 0.25
    let (_dir, _db, ex) = make_db_f64("nd19", "kv", 2.0);
    let mut p = Parser::new("QUERY nd19 COMPUTE rv = GEOMETRIC_PMF(kv, 0.5) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.25).abs() < 0.0001, "expected 0.25 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 20: GEO_PMF alias ────────────────────────────────────────────────────
#[test]
fn test_geo_pmf_alias() {
    // GEO_PMF(1, 0.5) = 0.5
    let (_dir, _db, ex) = make_db_f64("nd20", "kv", 1.0);
    let mut p = Parser::new("QUERY nd20 COMPUTE rv = GEO_PMF(kv, 0.5) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.0001, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 21: LOG_NORMAL_PDF(1.0, 0.0, 1.0) ───────────────────────────────────
#[test]
fn test_log_normal_pdf() {
    // f(1 | mu=0, sigma=1) = exp(-(ln(1)-0)^2 / 2) / (1*1*sqrt(2*pi))
    //                      = exp(0) / sqrt(2*pi) = 1/sqrt(2*pi) ≈ 0.3989
    let (_dir, _db, ex) = make_db_f64("nd21", "xv", 1.0);
    let mut p = Parser::new("QUERY nd21 COMPUTE rv = LOG_NORMAL_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3989).abs() < 0.001, "expected ~0.3989 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 22: LOGNORMAL_PDF alias ─────────────────────────────────────────────
#[test]
fn test_lognormal_pdf_alias() {
    // LOGNORMAL_PDF(1.0, 0.0, 1.0) ≈ 0.3989
    let (_dir, _db, ex) = make_db_f64("nd22", "xv", 1.0);
    let mut p = Parser::new("QUERY nd22 COMPUTE rv = LOGNORMAL_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3989).abs() < 0.001, "expected ~0.3989 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 23: PARETO_PDF(1.0, 2.0, 1.0) = 2.0 ────────────────────────────────
#[test]
fn test_pareto_pdf() {
    // f(1 | alpha=2, xm=1) = 2 * 1^2 / 1^3 = 2.0
    let (_dir, _db, ex) = make_db_f64("nd23", "xv", 1.0);
    let mut p = Parser::new("QUERY nd23 COMPUTE rv = PARETO_PDF(xv, 2.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 0.0001, "expected 2.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 24: PARETO_DENSITY alias ────────────────────────────────────────────
#[test]
fn test_pareto_density_alias() {
    // PARETO_DENSITY(1.0, 2.0, 1.0) = 2.0
    let (_dir, _db, ex) = make_db_f64("nd24", "xv", 1.0);
    let mut p = Parser::new("QUERY nd24 COMPUTE rv = PARETO_DENSITY(xv, 2.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 0.0001, "expected 2.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 25: WEIBULL_PDF(1.0, 1.0, 1.0) = e^(-1) ≈ 0.3679 ──────────────────
#[test]
fn test_weibull_pdf() {
    // f(1 | k=1, lambda=1) = (1/1)*(1/1)^0 * exp(-1) = e^(-1) ≈ 0.3679
    let (_dir, _db, ex) = make_db_f64("nd25", "xv", 1.0);
    let mut p = Parser::new("QUERY nd25 COMPUTE rv = WEIBULL_PDF(xv, 1.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 26: WEIBULL_DENSITY alias ───────────────────────────────────────────
#[test]
fn test_weibull_density_alias() {
    // WEIBULL_DENSITY(1.0, 1.0, 1.0) = e^(-1)
    let (_dir, _db, ex) = make_db_f64("nd26", "xv", 1.0);
    let mut p = Parser::new("QUERY nd26 COMPUTE rv = WEIBULL_DENSITY(xv, 1.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.3679).abs() < 0.001, "expected ~0.3679 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 27: CAUCHY_PDF(0.0, 0.0, 1.0) = 1/pi ≈ 0.3183 ─────────────────────
#[test]
fn test_cauchy_pdf() {
    // f(0 | x0=0, gamma=1) = 1/(pi*1*(1+0)) = 1/pi ≈ 0.3183
    let (_dir, _db, ex) = make_db_f64("nd27", "xv", 0.0);
    let mut p = Parser::new("QUERY nd27 COMPUTE rv = CAUCHY_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!(
            (*f - std::f64::consts::FRAC_1_PI).abs() < 0.001,
            "expected ~0.3183 got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 28: CAUCHY_DENSITY alias ────────────────────────────────────────────
#[test]
fn test_cauchy_density_alias() {
    // CAUCHY_DENSITY(0.0, 0.0, 1.0) = 1/pi
    let (_dir, _db, ex) = make_db_f64("nd28", "xv", 0.0);
    let mut p = Parser::new("QUERY nd28 COMPUTE rv = CAUCHY_DENSITY(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!(
            (*f - std::f64::consts::FRAC_1_PI).abs() < 0.001,
            "expected ~0.3183 got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 29: LAPLACE_PDF(0.0, 0.0, 1.0) = 0.5 ───────────────────────────────
#[test]
fn test_laplace_pdf_at_mode() {
    // f(0 | mu=0, b=1) = exp(0) / (2*1) = 0.5
    let (_dir, _db, ex) = make_db_f64("nd29", "xv", 0.0);
    let mut p = Parser::new("QUERY nd29 COMPUTE rv = LAPLACE_PDF(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.0001, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 30: LAPLACE_DENSITY alias ───────────────────────────────────────────
#[test]
fn test_laplace_density_alias() {
    // LAPLACE_DENSITY(0.0, 0.0, 1.0) = 0.5
    let (_dir, _db, ex) = make_db_f64("nd30", "xv", 0.0);
    let mut p = Parser::new("QUERY nd30 COMPUTE rv = LAPLACE_DENSITY(xv, 0.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.0001, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 31: BETA_PDF(0.5, 2.0, 2.0) > 0 ────────────────────────────────────
#[test]
fn test_beta_pdf_positive() {
    // BETA_PDF(0.5, 2.0, 2.0) should be a positive value (= 1.5 exactly for symmetric Beta(2,2))
    let (_dir, _db, ex) = make_db_f64("nd31", "xv", 0.5);
    let mut p = Parser::new("QUERY nd31 COMPUTE rv = BETA_PDF(xv, 2.0, 2.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive value got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 32: BETA_DENSITY alias ──────────────────────────────────────────────
#[test]
fn test_beta_density_alias() {
    // BETA_DENSITY(0.5, 2.0, 2.0) > 0
    let (_dir, _db, ex) = make_db_f64("nd32", "xv", 0.5);
    let mut p = Parser::new("QUERY nd32 COMPUTE rv = BETA_DENSITY(xv, 2.0, 2.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive value got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 33: NORMAL_QUANTILE(0.5) = 0.0 ─────────────────────────────────────
#[test]
fn test_normal_quantile_at_half() {
    // probit(0.5) = 0 (median of standard normal)
    let (_dir, _db, ex) = make_db_f64("nd33", "pv", 0.5);
    let mut p = Parser::new("QUERY nd33 COMPUTE rv = NORMAL_QUANTILE(pv) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.01, "expected ~0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 34: NORMAL_QUANTILE(0.975) ≈ 1.96 ───────────────────────────────────
#[test]
fn test_normal_quantile_at_0975() {
    // probit(0.975) ≈ 1.96
    let (_dir, _db, ex) = make_db_f64("nd34", "pv", 0.975);
    let mut p = Parser::new("QUERY nd34 COMPUTE rv = NORMAL_QUANTILE(pv) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 1.96).abs() < 0.02, "expected ~1.96 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 35: PROBIT alias ─────────────────────────────────────────────────────
#[test]
fn test_probit_at_half() {
    // PROBIT(0.5) = 0
    let (_dir, _db, ex) = make_db_f64("nd35", "pv", 0.5);
    let mut p = Parser::new("QUERY nd35 COMPUTE rv = PROBIT(pv) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.01, "expected ~0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 36: DISTRIBUTION_MEAN("normal", 5.0, 1.0) = 5.0 ────────────────────
#[test]
fn test_distribution_mean_normal() {
    // mean of Normal(5, 1) = 5
    let (_dir, _db, ex) = make_db_str("nd36", "dist", "normal");
    let mut p = Parser::new("QUERY nd36 COMPUTE rv = DISTRIBUTION_MEAN(dist, 5.0, 1.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 5.0).abs() < 0.0001, "expected 5.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 37: DISTRIBUTION_MEAN("exponential", 2.0) = 0.5 ────────────────────
#[test]
fn test_distribution_mean_exponential() {
    // mean of Exponential(lambda=2) = 1/2 = 0.5
    let (_dir, _db, ex) = make_db_str("nd37", "dist", "exponential");
    let mut p = Parser::new("QUERY nd37 COMPUTE rv = DISTRIBUTION_MEAN(dist, 2.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.0001, "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 38: DIST_MEAN alias ──────────────────────────────────────────────────
#[test]
fn test_dist_mean_alias() {
    // DIST_MEAN("binomial", 10.0, 0.3) = 10*0.3 = 3.0
    let (_dir, _db, ex) = make_db_str("nd38", "dist", "binomial");
    let mut p = Parser::new("QUERY nd38 COMPUTE rv = DIST_MEAN(dist, 10.0, 0.3) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 3.0).abs() < 0.0001, "expected 3.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 39: DISTRIBUTION_VARIANCE("normal", 0.0, 2.0) = 4.0 ────────────────
#[test]
fn test_distribution_variance_normal() {
    // Var(Normal(0, sigma=2)) = sigma^2 = 4.0
    let (_dir, _db, ex) = make_db_str("nd39", "dist", "normal");
    let mut p =
        Parser::new("QUERY nd39 COMPUTE rv = DISTRIBUTION_VARIANCE(dist, 0.0, 2.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 4.0).abs() < 0.0001, "expected 4.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 40: DISTRIBUTION_VARIANCE("binomial", 10.0, 0.3) = 2.1 ─────────────
#[test]
fn test_distribution_variance_binomial() {
    // Var(Binomial(n=10, p=0.3)) = n*p*(1-p) = 10*0.3*0.7 = 2.1
    let (_dir, _db, ex) = make_db_str("nd40", "dist", "binomial");
    let mut p =
        Parser::new("QUERY nd40 COMPUTE rv = DISTRIBUTION_VARIANCE(dist, 10.0, 0.3) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 2.1).abs() < 0.0001, "expected 2.1 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Test 41: DIST_VARIANCE alias ──────────────────────────────────────────────
#[test]
fn test_dist_variance_alias() {
    // DIST_VARIANCE("exponential", 2.0) = 1/(2^2) = 0.25
    let (_dir, _db, ex) = make_db_str("nd41", "dist", "exponential");
    let mut p = Parser::new("QUERY nd41 COMPUTE rv = DIST_VARIANCE(dist, 2.0) SELECT rv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rv") {
        Some(Value::Float(f)) => assert!((*f - 0.25).abs() < 0.0001, "expected 0.25 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}
