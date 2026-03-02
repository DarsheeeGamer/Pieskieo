/// Integration tests for PQL built-in financial math functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (dir, db, ex)
}

fn seed(db: &Arc<PieskieoDb>) {
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"x": 1})).unwrap();
}

// ── NPV / IRR / PMT / FV / PV ────────────────────────────────────────────────

#[test]
fn test_npv_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // NPV = 100/1.1 + 100/1.21 + 100/1.331 ≈ 248.69
    let mut p = Parser::new("QUERY t COMPUTE n = NPV(0.1, [100.0, 100.0, 100.0]) SELECT n;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(*f > 248.0 && *f < 249.5, "expected ~248.69, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_net_present_value_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE n = NET_PRESENT_VALUE(0.1, [100.0, 100.0, 100.0]) SELECT n;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(*f > 248.0 && *f < 249.5, "expected ~248.69, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_irr_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // cashflows [-1000, 400, 400, 400] => IRR ≈ 9.7%
    let mut p = Parser::new("QUERY t COMPUTE r = IRR([-1000.0, 400.0, 400.0, 400.0]) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(*f > 0.09 && *f < 0.11, "expected ~9.7% IRR, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_internal_rate_of_return_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE r = INTERNAL_RATE_OF_RETURN([-1000.0, 400.0, 400.0, 400.0]) SELECT r;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(*f > 0.09 && *f < 0.11, "expected ~9.7% IRR, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_pmt_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // PMT(0.01, 12, 1000) — monthly payment on $1000 at 1%/month for 12 months
    let mut p = Parser::new("QUERY t COMPUTE pay = PMT(0.01, 12, 1000.0) SELECT pay;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pay") {
        Some(Value::Float(f)) => {
            // Payment should be negative (outflow), magnitude ~88.85
            assert!(f.abs() > 88.0 && f.abs() < 90.0, "expected ~-88.85, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_loan_payment_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE pay = LOAN_PAYMENT(0.01, 12, 1000.0) SELECT pay;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pay") {
        Some(Value::Float(f)) => assert!(f.abs() > 88.0 && f.abs() < 90.0, "expected ~88.85, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_fv_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // FV(0.05, 10, 0, -1000) => 1000 * 1.05^10 ≈ 1628.89
    let mut p = Parser::new("QUERY t COMPUTE f = FV(0.05, 10, 0.0, -1000.0) SELECT f;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Float(f)) => assert!(f.abs() > 1620.0 && f.abs() < 1640.0, "expected ~1628.89, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_future_value_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE f = FUTURE_VALUE(0.05, 10, 0.0, -1000.0) SELECT f;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Float(f)) => assert!(f.abs() > 1620.0 && f.abs() < 1640.0, "expected ~1628.89, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_pv_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // PV(0.05, 10, -100) => annuity PV ≈ 772.17
    let mut p = Parser::new("QUERY t COMPUTE p = PV(0.05, 10, -100.0) SELECT p;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(f)) => assert!(f.abs() > 770.0 && f.abs() < 775.0, "expected ~772.17, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_present_value_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE p = PRESENT_VALUE(0.05, 10, -100.0) SELECT p;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(f)) => assert!(f.abs() > 770.0 && f.abs() < 775.0, "expected ~772.17, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Compound / Simple interest / CAGR ────────────────────────────────────────

#[test]
fn test_compound_interest_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // COMPOUND_INTEREST(1000, 0.05, 10) => 1000 * 1.05^10 ≈ 1628.89
    let mut p = Parser::new("QUERY t COMPUTE a = COMPOUND_INTEREST(1000.0, 0.05, 10) SELECT a;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(*f > 1628.0 && *f < 1630.0, "expected ~1628.89, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_cagr_amount_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE a = CAGR_AMOUNT(1000.0, 0.05, 10) SELECT a;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(*f > 1628.0 && *f < 1630.0, "expected ~1628.89, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_simple_interest_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // SIMPLE_INTEREST(1000, 0.05, 3) = 1000 * 0.05 * 3 = 150
    let mut p = Parser::new("QUERY t COMPUTE s = SIMPLE_INTEREST(1000.0, 0.05, 3) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!((*f - 150.0).abs() < 0.01, "expected 150.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_simple_int_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE s = SIMPLE_INT(1000.0, 0.05, 3) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!((*f - 150.0).abs() < 0.01, "expected 150.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_cagr_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // CAGR(100, 200, 10) = (200/100)^(1/10) - 1 ≈ 7.18%
    let mut p = Parser::new("QUERY t COMPUTE c = CAGR(100.0, 200.0, 10) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(*f > 0.071 && *f < 0.073, "expected ~7.18%, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_compound_annual_growth_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE c = COMPOUND_ANNUAL_GROWTH(100.0, 200.0, 10) SELECT c;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(*f > 0.071 && *f < 0.073, "expected ~7.18%, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_rule_of_72() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // RULE_OF_72(8) = 72/8 = 9 years
    let mut p = Parser::new("QUERY t COMPUTE d = RULE_OF_72(8.0) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 9.0).abs() < 0.001, "expected 9.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_doubling_time_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE d = DOUBLING_TIME(6.0) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 12.0).abs() < 0.001, "expected 12.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Bond / Annuity ────────────────────────────────────────────────────────────

#[test]
fn test_bond_price_at_par() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // When coupon_rate == ytm, bond price should equal face value
    // BOND_PRICE(1000, 0.05, 0.05, 10) ≈ 1000
    let mut p = Parser::new("QUERY t COMPUTE bp = BOND_PRICE(1000.0, 0.05, 0.05, 10) SELECT bp;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bp") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 1.0, "expected ~1000.0 (at par), got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_present_value_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE bp = BOND_PRESENT_VALUE(1000.0, 0.05, 0.05, 10) SELECT bp;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bp") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 1.0, "expected ~1000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_price_discount() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // When ytm > coupon_rate, bond trades at discount (price < face)
    let mut p = Parser::new("QUERY t COMPUTE bp = BOND_PRICE(1000.0, 0.05, 0.08, 10) SELECT bp;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bp") {
        Some(Value::Float(f)) => assert!(*f < 1000.0, "discount bond price should be < 1000, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_price_premium() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // When ytm < coupon_rate, bond trades at premium (price > face)
    let mut p = Parser::new("QUERY t COMPUTE bp = BOND_PRICE(1000.0, 0.08, 0.05, 10) SELECT bp;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bp") {
        Some(Value::Float(f)) => assert!(*f > 1000.0, "premium bond price should be > 1000, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_yield_approx_at_par() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // When price == face, approx YTM ≈ coupon_rate
    let mut p = Parser::new("QUERY t COMPUTE y = BOND_YIELD_APPROX(1000.0, 0.05, 1000.0, 10) SELECT y;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("y") {
        Some(Value::Float(f)) => assert!((*f - 0.05).abs() < 0.001, "expected ~0.05, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_approx_ytm_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE y = APPROX_YTM(1000.0, 0.05, 1000.0, 10) SELECT y;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("y") {
        Some(Value::Float(f)) => assert!((*f - 0.05).abs() < 0.001, "expected ~0.05, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_yield_approx_discount() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Discount bond (price < face): approx YTM > coupon_rate
    let mut p = Parser::new("QUERY t COMPUTE y = BOND_YIELD_APPROX(1000.0, 0.05, 900.0, 10) SELECT y;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("y") {
        Some(Value::Float(f)) => assert!(*f > 0.05, "discount YTM should be > coupon rate, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_annuity_pv_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // ANNUITY_PV(100, 0.05, 10) = 100 * (1 - 1.05^-10) / 0.05 ≈ 772.17
    let mut p = Parser::new("QUERY t COMPUTE apv = ANNUITY_PV(100.0, 0.05, 10) SELECT apv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("apv") {
        Some(Value::Float(f)) => assert!(*f > 770.0 && *f < 775.0, "expected ~772.17, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_pv_annuity_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE apv = PV_ANNUITY(100.0, 0.05, 10) SELECT apv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("apv") {
        Some(Value::Float(f)) => assert!(*f > 770.0 && *f < 775.0, "expected ~772.17, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_annuity_pv_zero_rate() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // With zero rate, PV = payment * periods
    let mut p = Parser::new("QUERY t COMPUTE apv = ANNUITY_PV(100.0, 0.0, 10) SELECT apv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("apv") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 0.01, "expected 1000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_annuity_fv_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // ANNUITY_FV(100, 0.05, 10) = 100 * (1.05^10 - 1) / 0.05 ≈ 1257.79
    let mut p = Parser::new("QUERY t COMPUTE afv = ANNUITY_FV(100.0, 0.05, 10) SELECT afv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("afv") {
        Some(Value::Float(f)) => assert!(*f > 1255.0 && *f < 1260.0, "expected ~1257.79, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_fv_annuity_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE afv = FV_ANNUITY(100.0, 0.05, 10) SELECT afv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("afv") {
        Some(Value::Float(f)) => assert!(*f > 1255.0 && *f < 1260.0, "expected ~1257.79, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_annuity_fv_zero_rate() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // With zero rate, FV = payment * periods
    let mut p = Parser::new("QUERY t COMPUTE afv = ANNUITY_FV(100.0, 0.0, 10) SELECT afv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("afv") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 0.01, "expected 1000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Risk / Return ─────────────────────────────────────────────────────────────

#[test]
fn test_sharpe_ratio_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // SHARPE_RATIO([0.05, 0.10, 0.03, 0.08, 0.06], 0.02)
    let mut p = Parser::new("QUERY t COMPUTE s = SHARPE_RATIO([0.05, 0.10, 0.03, 0.08, 0.06], 0.02) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Sharpe ratio should be positive for these returns, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sharpe_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE s = SHARPE([0.05, 0.10, 0.03, 0.08, 0.06], 0.02) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Sharpe ratio should be positive, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sortino_ratio_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE s = SORTINO_RATIO([0.05, -0.02, 0.08, -0.01, 0.06], 0.0) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Sortino ratio should be positive, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sortino_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE s = SORTINO([0.05, -0.02, 0.08, -0.01, 0.06], 0.0) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Sortino ratio should be positive, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_max_drawdown_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // prices [100, 120, 90, 110] => peak=120, trough=90 => DD=(120-90)/120=0.25
    let mut p = Parser::new("QUERY t COMPUTE d = MAX_DRAWDOWN([100.0, 120.0, 90.0, 110.0]) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 0.25).abs() < 0.001, "expected 0.25, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_max_dd_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE d = MAX_DD([100.0, 120.0, 90.0, 110.0]) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 0.25).abs() < 0.001, "expected 0.25, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_max_drawdown_monotone_up() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // All prices rising => max drawdown = 0
    let mut p = Parser::new("QUERY t COMPUTE d = MAX_DRAWDOWN([100.0, 110.0, 120.0, 130.0]) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f < 0.001, "no drawdown expected, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_var_parametric_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE v = VAR_PARAMETRIC([0.05, -0.02, 0.08, -0.03, 0.04, -0.01, 0.06], 0.95) SELECT v;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => assert!(f.is_finite(), "VaR should be finite, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_value_at_risk_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE v = VALUE_AT_RISK([0.05, -0.02, 0.08, -0.03, 0.04], 0.95) SELECT v;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => assert!(f.is_finite(), "VaR should be finite, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Tax / Depreciation ────────────────────────────────────────────────────────

#[test]
fn test_straight_line_depr_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // STRAIGHT_LINE_DEPR(10000, 1000, 9) = (10000-1000)/9 = 1000 per period
    let mut p = Parser::new("QUERY t COMPUTE d = STRAIGHT_LINE_DEPR(10000.0, 1000.0, 9) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 0.01, "expected 1000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sl_depreciation_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE d = SL_DEPRECIATION(10000.0, 1000.0, 9) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 0.01, "expected 1000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_straight_line_depr_no_salvage() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // STRAIGHT_LINE_DEPR(5000, 0, 5) = 1000
    let mut p = Parser::new("QUERY t COMPUTE d = STRAIGHT_LINE_DEPR(5000.0, 0.0, 5) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 1000.0).abs() < 0.01, "expected 1000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_double_declining_depr_period0() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // DOUBLE_DECLINING_DEPR(10000, 0, 5, 0): rate=2/5=0.4, no periods skipped
    // depr = 10000 * 0.4 = 4000
    let mut p = Parser::new("QUERY t COMPUTE d = DOUBLE_DECLINING_DEPR(10000.0, 0.0, 5, 0) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 4000.0).abs() < 1.0, "expected ~4000.0 for period 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_ddb_depreciation_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // DDB_DEPRECIATION with period=0 gives first period depreciation
    let mut p = Parser::new("QUERY t COMPUTE d = DDB_DEPRECIATION(10000.0, 0.0, 5, 0) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 4000.0).abs() < 1.0, "expected ~4000.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_double_declining_depr_period1() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // period=1 skips one period first; book after skip = 6000, depr = 6000 * 0.4 = 2400
    let mut p = Parser::new("QUERY t COMPUTE d = DOUBLE_DECLINING_DEPR(10000.0, 0.0, 5, 1) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 2400.0).abs() < 1.0, "expected ~2400.0 for period 1, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_effective_tax_rate_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Simple 2-bracket: 0-10000 at 10%, 10000+ at 25%
    // Income=15000 => tax = 10000*0.10 + 5000*0.25 = 1000+1250 = 2250
    // effective rate = 2250/15000 = 0.15
    let mut p = Parser::new("QUERY t COMPUTE etr = EFFECTIVE_TAX_RATE(15000.0, [[10000.0, 0.10], [100000.0, 0.25]]) SELECT etr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("etr") {
        Some(Value::Float(f)) => assert!((*f - 0.15).abs() < 0.001, "expected 0.15, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_eff_tax_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE etr = EFF_TAX(15000.0, [[10000.0, 0.10], [100000.0, 0.25]]) SELECT etr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("etr") {
        Some(Value::Float(f)) => assert!((*f - 0.15).abs() < 0.001, "expected 0.15, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_effective_tax_rate_single_bracket() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Single bracket: all income at 20%
    let mut p = Parser::new("QUERY t COMPUTE etr = EFFECTIVE_TAX_RATE(10000.0, [[100000.0, 0.20]]) SELECT etr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("etr") {
        Some(Value::Float(f)) => assert!((*f - 0.20).abs() < 0.001, "expected 0.20, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_after_tax_return_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // AFTER_TAX_RETURN(0.10, 0.25) = 0.10 * (1 - 0.25) = 0.075
    let mut p = Parser::new("QUERY t COMPUTE atr = AFTER_TAX_RETURN(0.10, 0.25) SELECT atr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("atr") {
        Some(Value::Float(f)) => assert!((*f - 0.075).abs() < 0.0001, "expected 0.075, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_net_return_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new("QUERY t COMPUTE atr = NET_RETURN(0.10, 0.25) SELECT atr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("atr") {
        Some(Value::Float(f)) => assert!((*f - 0.075).abs() < 0.0001, "expected 0.075, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_after_tax_return_zero_tax() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Zero tax => gross == net
    let mut p = Parser::new("QUERY t COMPUTE atr = AFTER_TAX_RETURN(0.08, 0.0) SELECT atr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("atr") {
        Some(Value::Float(f)) => assert!((*f - 0.08).abs() < 0.0001, "expected 0.08, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn test_npv_negative_rate() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Negative discount rate: should still compute
    let mut p = Parser::new("QUERY t COMPUTE n = NPV(-0.05, [100.0, 100.0]) SELECT n;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(f.is_finite(), "NPV with negative rate should be finite, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_annuity_pv_consistency_with_pv() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // ANNUITY_PV and PV should give consistent results for same inputs
    let mut p = Parser::new("QUERY t COMPUTE a = ANNUITY_PV(100.0, 0.05, 10), b = PV(0.05, 10, -100.0) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = match r.rows[0].data.get("a") { Some(Value::Float(f)) => *f, other => panic!("expected Float for a, got {:?}", other) };
    let b = match r.rows[0].data.get("b") { Some(Value::Float(f)) => *f, other => panic!("expected Float for b, got {:?}", other) };
    assert!((a - b.abs()).abs() < 0.01, "ANNUITY_PV and PV should be consistent: {} vs {}", a, b.abs());
}

#[test]
fn test_cagr_amount_consistency_with_compound_interest() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // CAGR_AMOUNT and COMPOUND_INTEREST should give the same result
    let mut p = Parser::new("QUERY t COMPUTE a = CAGR_AMOUNT(1000.0, 0.07, 5), b = COMPOUND_INTEREST(1000.0, 0.07, 5) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = match r.rows[0].data.get("a") { Some(Value::Float(f)) => *f, other => panic!("expected Float for a, got {:?}", other) };
    let b = match r.rows[0].data.get("b") { Some(Value::Float(f)) => *f, other => panic!("expected Float for b, got {:?}", other) };
    assert!((a - b).abs() < 0.001, "CAGR_AMOUNT and COMPOUND_INTEREST should match: {} vs {}", a, b);
}

#[test]
fn test_rule_of_72_high_rate() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // RULE_OF_72(72) = 1 year
    let mut p = Parser::new("QUERY t COMPUTE d = RULE_OF_72(72.0) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001, "expected 1.0 year, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_loan_payment_consistency_with_pmt() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // LOAN_PAYMENT and PMT should return same result
    let mut p = Parser::new("QUERY t COMPUTE a = LOAN_PAYMENT(0.005, 360, 200000.0), b = PMT(0.005, 360, 200000.0) SELECT a, b;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = match r.rows[0].data.get("a") { Some(Value::Float(f)) => *f, other => panic!("expected Float for a, got {:?}", other) };
    let b = match r.rows[0].data.get("b") { Some(Value::Float(f)) => *f, other => panic!("expected Float for b, got {:?}", other) };
    assert!((a - b).abs() < 0.001, "LOAN_PAYMENT and PMT should match: {} vs {}", a, b);
}

#[test]
fn test_var_parametric_99_conf() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // At 99% confidence, VaR should be larger than at 95%
    let mut p = Parser::new("QUERY t COMPUTE v95 = VAR_PARAMETRIC([0.05, -0.02, 0.08, -0.03, 0.04, -0.01, 0.06], 0.95), v99 = VAR_PARAMETRIC([0.05, -0.02, 0.08, -0.03, 0.04, -0.01, 0.06], 0.99) SELECT v95, v99;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v95 = match r.rows[0].data.get("v95") { Some(Value::Float(f)) => *f, other => panic!("expected Float, got {:?}", other) };
    let v99 = match r.rows[0].data.get("v99") { Some(Value::Float(f)) => *f, other => panic!("expected Float, got {:?}", other) };
    assert!(v99 > v95, "99% VaR ({}) should be greater than 95% VaR ({})", v99, v95);
}
