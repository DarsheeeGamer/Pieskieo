/// Integration tests for advanced PQL financial and business calculation functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (dir, db, ex)
}

// ── NPV ──────────────────────────────────────────────────────────────────────

#[test]
fn test_npv_simple_cashflows() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rate": 0.1}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE n = NPV(0.1, [100.0, 100.0, 100.0]) SELECT n;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Float(f)) => {
            // NPV = 100/1.1 + 100/1.21 + 100/1.331 ≈ 248.69
            assert!(*f > 248.0 && *f < 249.5, "expected ~248.69, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_net_present_value_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new("QUERY t COMPUTE n = NET_PRESENT_VALUE(0.1, [100.0, 100.0, 100.0]) SELECT n;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(*f > 248.0 && *f < 249.5, "expected ~248.69, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── IRR ──────────────────────────────────────────────────────────────────────

#[test]
fn test_irr_known_cashflows() {
    let (_dir, db, ex) = setup();
    // cashflows: [-1000, 400, 400, 400] — IRR ≈ 9.7%
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IRR([-1000.0, 400.0, 400.0, 400.0]) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(*f > 0.09 && *f < 0.11, "expected ~9.7% IRR, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_internal_rate_return_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE r = INTERNAL_RATE_RETURN([-1000.0, 400.0, 400.0, 400.0]) SELECT r;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(*f > 0.09 && *f < 0.11, "expected ~9.7% IRR, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── MIRR ─────────────────────────────────────────────────────────────────────

#[test]
fn test_mirr_returns_float() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE m = MIRR([-1000.0, 300.0, 400.0, 500.0], 0.1, 0.12) SELECT m;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => assert!(
            f.is_finite(),
            "MIRR should return a finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_modified_irr_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE m = MODIFIED_IRR([-1000.0, 300.0, 400.0, 500.0], 0.1, 0.12) SELECT m;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => assert!(
            f.is_finite(),
            "MODIFIED_IRR should return a finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── CAGR / COMPOUND_GROWTH_RATE ───────────────────────────────────────────────

#[test]
fn test_cagr_compound_growth_rate_alias() {
    // COMPOUND_GROWTH_RATE(100, 200, 5) = (200/100)^(1/5) - 1 ≈ 0.1487
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new("QUERY t COMPUTE c = COMPOUND_GROWTH_RATE(100.0, 200.0, 5.0) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.1487).abs() < 0.001,
            "expected ~14.87% CAGR, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_cagr_via_cagr_function() {
    // The existing CAGR function for sanity: same math
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE c = CAGR(100.0, 200.0, 5.0) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.1487).abs() < 0.001,
            "expected ~14.87% CAGR, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── SHARPE ───────────────────────────────────────────────────────────────────

#[test]
fn test_sharpe_positive_ratio() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new("QUERY t COMPUTE s = SHARPE([0.05, 0.06, 0.07, 0.08, 0.09], 0.0) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0,
            "Sharpe ratio should be positive for positive returns, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sharpe_ratio_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE s = SHARPE_RATIO([0.05, 0.06, 0.07, 0.08, 0.09], 0.0) SELECT s;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0,
            "SHARPE_RATIO alias should give positive result, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── SORTINO ──────────────────────────────────────────────────────────────────

#[test]
fn test_sortino_positive_with_positive_mean() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Returns all positive except one negative — mean > 0 and target=0, so sortino > 0
    let mut p =
        Parser::new("QUERY t COMPUTE s = SORTINO([0.05, 0.06, 0.07, -0.01, 0.08], 0.0) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.0, "Sortino ratio should be positive here, got {}", f)
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sortino_ratio_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE s = SORTINO_RATIO([0.05, 0.06, 0.07, -0.01, 0.08], 0.0) SELECT s;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0,
            "SORTINO_RATIO alias should give positive result, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── DRAWDOWN ─────────────────────────────────────────────────────────────────

#[test]
fn test_drawdown_monotonically_increasing_is_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new("QUERY t COMPUTE d = DRAWDOWN([100.0, 110.0, 120.0, 130.0, 140.0]) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            *f == 0.0,
            "monotonically increasing prices should have 0 drawdown, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_max_drawdown_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE d = MAX_DRAWDOWN([100.0, 110.0, 120.0, 130.0, 140.0]) SELECT d;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            *f == 0.0,
            "MAX_DRAWDOWN alias: monotonically increasing → 0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_drawdown_known_drop() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Peak at 200, then drops to 100 → drawdown = (200-100)/200 = 0.5
    let mut p = Parser::new("QUERY t COMPUTE d = DRAWDOWN([100.0, 200.0, 150.0, 100.0]) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => {
            assert!((*f - 0.5).abs() < 1e-9, "expected 0.5 drawdown, got {}", f)
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── BETA_COEFF ───────────────────────────────────────────────────────────────

#[test]
fn test_beta_coeff_identical_series_is_one() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE b = BETA_COEFF([0.01, 0.02, -0.01, 0.03, 0.02], [0.01, 0.02, -0.01, 0.03, 0.02]) SELECT b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "identical series should give beta=1, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_portfolio_beta_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE b = PORTFOLIO_BETA([0.01, 0.02, -0.01, 0.03, 0.02], [0.01, 0.02, -0.01, 0.03, 0.02]) SELECT b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "PORTFOLIO_BETA alias: identical series → 1, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── INFORMATION_RATIO ────────────────────────────────────────────────────────

#[test]
fn test_information_ratio_same_returns_is_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Identical returns → excess returns all 0 → IR = Null (tracking error = 0)
    let mut p = Parser::new("QUERY t COMPUTE ir = INFORMATION_RATIO([0.05, 0.06, 0.04, 0.07], [0.05, 0.06, 0.04, 0.07]) SELECT ir;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // When tracking error is 0, function returns Null
    match r.rows[0].data.get("ir") {
        Some(Value::Null) => {} // expected: TE=0 → Null
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "IR with identical series should be ~0 or Null, got {}",
            f
        ),
        other => panic!("expected Null or near-zero Float, got {:?}", other),
    }
}

#[test]
fn test_ir_ratio_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE ir = IR_RATIO([0.05, 0.07, 0.06, 0.08], [0.04, 0.05, 0.05, 0.06]) SELECT ir;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ir") {
        Some(Value::Float(f)) => assert!(
            f.is_finite(),
            "IR_RATIO should return a finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── VALUE_AT_RISK ─────────────────────────────────────────────────────────────

#[test]
fn test_value_at_risk_returns_positive_float() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE v = VALUE_AT_RISK([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.00, -0.02, 0.04], 0.95) SELECT v;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && f.is_finite(),
            "VaR should be a positive finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_var_p_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE v = VAR_P([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.00, -0.02, 0.04], 0.95) SELECT v;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && f.is_finite(),
            "VAR_P alias should return positive finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── EXPECTED_SHORTFALL ───────────────────────────────────────────────────────

#[test]
fn test_expected_shortfall_gte_var() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // ES should be >= VaR at same confidence level
    let mut p = Parser::new("QUERY t COMPUTE es = EXPECTED_SHORTFALL([0.01, -0.02, 0.03, -0.04, 0.02, -0.03, 0.01, 0.00, -0.05, 0.04, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00, -0.03, 0.02, -0.01, 0.05], 0.9), v = VALUE_AT_RISK([0.01, -0.02, 0.03, -0.04, 0.02, -0.03, 0.01, 0.00, -0.05, 0.04, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00, -0.03, 0.02, -0.01, 0.05], 0.9) SELECT es, v;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let es = match r.rows[0].data.get("es") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for ES, got {:?}", other),
    };
    let v = match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for VaR, got {:?}", other),
    };
    assert!(
        es >= v - 1e-9,
        "Expected Shortfall ({}) should be >= VaR ({}) at same confidence",
        es,
        v
    );
}

#[test]
fn test_cvar_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Use 20 returns and conf=0.9 → cutoff=2, guaranteed non-empty tail
    let mut p = Parser::new("QUERY t COMPUTE es = CVAR([0.01, -0.02, 0.03, -0.04, 0.02, -0.03, 0.01, 0.00, -0.05, 0.04, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00, -0.03, 0.02, -0.01, 0.05], 0.9) SELECT es;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("es") {
        Some(Value::Float(f)) => assert!(
            f.is_finite(),
            "CVaR alias should return finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── HOLDING_PERIOD_RETURN ────────────────────────────────────────────────────

#[test]
fn test_holding_period_return_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // HPR(100, 110) = (110 - 100) / 100 = 0.1
    let mut p = Parser::new("QUERY t COMPUTE h = HOLDING_PERIOD_RETURN(100.0, 110.0) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!((*f - 0.1).abs() < 1e-9, "expected HPR=0.1, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_hpr_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE h = HPR(100.0, 110.0) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.1).abs() < 1e-9,
            "HPR alias: expected 0.1, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_hpr_with_dividend() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // HPR(100, 110, 5) = (110 - 100 + 5) / 100 = 0.15
    let mut p = Parser::new("QUERY t COMPUTE h = HPR(100.0, 110.0, 5.0) SELECT h;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.15).abs() < 1e-9,
            "expected HPR with dividend=0.15, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── ANNUALIZED_RETURN ────────────────────────────────────────────────────────

#[test]
fn test_annualized_return_365_days() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // ANNUALIZED_RETURN(0.1, 365) = (1.1)^(365/365) - 1 = 0.1
    let mut p = Parser::new("QUERY t COMPUTE a = ANNUALIZED_RETURN(0.1, 365.0) SELECT a;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.1).abs() < 1e-9,
            "ANNUALIZED_RETURN(0.1, 365) should be 0.1, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_ann_ret_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE a = ANN_RET(0.1, 365.0) SELECT a;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.1).abs() < 1e-9,
            "ANN_RET alias: expected 0.1, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_annualized_return_half_year() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // HPR of 5% over 182.5 days → annualized ≈ 10.25%
    let mut p = Parser::new("QUERY t COMPUTE a = ANNUALIZED_RETURN(0.05, 182.5) SELECT a;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => {
            let expected = 1.05f64.powf(2.0) - 1.0; // ~0.1025
            assert!(
                (*f - expected).abs() < 0.001,
                "expected ~{}, got {}",
                expected,
                f
            );
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── CALMAR ───────────────────────────────────────────────────────────────────

#[test]
fn test_calmar_ratio_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Prices go 100→150→120→180 over 3 years
    // CAGR = (180/100)^(1/3) - 1 ≈ 21.6%
    // Max drawdown: peak=150, trough=120 → 20%
    // Calmar ≈ 0.216 / 0.20 ≈ 1.08
    let mut p =
        Parser::new("QUERY t COMPUTE c = CALMAR([100.0, 150.0, 120.0, 180.0], 3.0) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && f.is_finite(),
            "CALMAR ratio should be positive finite, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_calmar_ratio_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        "QUERY t COMPUTE c = CALMAR_RATIO([100.0, 150.0, 120.0, 180.0], 3.0) SELECT c;",
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && f.is_finite(),
            "CALMAR_RATIO alias should be positive finite, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── TREYNOR ───────────────────────────────────────────────────────────────────

#[test]
fn test_treynor_ratio_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Portfolio: identical to benchmark → beta=1, so treynor = mean_port - rf
    let mut p = Parser::new("QUERY t COMPUTE t = TREYNOR_RATIO([0.08, 0.10, 0.06, 0.09, 0.07], [0.05, 0.07, 0.04, 0.06, 0.05], 0.02) SELECT t;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("t") {
        Some(Value::Float(f)) => assert!(
            f.is_finite(),
            "TREYNOR_RATIO should return finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_treynor_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE t = TREYNOR([0.08, 0.10, 0.06, 0.09, 0.07], [0.05, 0.07, 0.04, 0.06, 0.05], 0.02) SELECT t;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("t") {
        Some(Value::Float(f)) => assert!(
            f.is_finite(),
            "TREYNOR alias should return finite float, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}
