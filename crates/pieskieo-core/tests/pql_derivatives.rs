/// Integration tests for PQL financial derivatives, options pricing, and portfolio theory.
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

// ── Black-Scholes Call ────────────────────────────────────────────────────────

#[test]
fn test_black_scholes_call_atm() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Classic: S=100, K=100, T=1yr, r=0.05, sigma=0.2 -> call ~10.45
    let mut p = Parser::new(r#"QUERY t COMPUTE c = BLACK_SCHOLES_CALL(100.0, 100.0, 1.0, 0.05, 0.2) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(*f > 9.0 && *f < 12.0, "BS call ATM ~10.45, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_call_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE c = BS_CALL(100.0, 100.0, 1.0, 0.05, 0.2) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(*f > 9.0 && *f < 12.0, "BS_CALL alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_black_scholes_call_itm() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // ITM: S=110, K=100 -> call > intrinsic value 10
    let mut p = Parser::new(r#"QUERY t COMPUTE c = BLACK_SCHOLES_CALL(110.0, 100.0, 1.0, 0.05, 0.2) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(*f > 10.0, "ITM call > 10, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_black_scholes_call_otm() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // OTM: S=90, K=100 -> small call price
    let mut p = Parser::new(r#"QUERY t COMPUTE c = BLACK_SCHOLES_CALL(90.0, 100.0, 1.0, 0.05, 0.2) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(*f > 0.0 && *f < 10.0, "OTM call < 10, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Black-Scholes Put ─────────────────────────────────────────────────────────

#[test]
fn test_black_scholes_put_atm() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // ATM put with r=0 should roughly equal call by put-call parity
    let mut p = Parser::new(r#"QUERY t COMPUTE p = BLACK_SCHOLES_PUT(100.0, 100.0, 1.0, 0.0, 0.2) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(f)) => assert!(*f > 5.0 && *f < 12.0, "ATM put with r=0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_put_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE p = BS_PUT(100.0, 100.0, 1.0, 0.05, 0.2) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "BS_PUT alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_put_call_parity() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // C - P = S - K*e^(-rT) (put-call parity)
    let mut p = Parser::new(r#"QUERY t COMPUTE c = BLACK_SCHOLES_CALL(100.0, 100.0, 1.0, 0.05, 0.2), pu = BLACK_SCHOLES_PUT(100.0, 100.0, 1.0, 0.05, 0.2) SELECT c, pu;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = match r.rows[0].data.get("c") { Some(Value::Float(f)) => *f, _ => panic!("no call") };
    let p_v = match r.rows[0].data.get("pu") { Some(Value::Float(f)) => *f, _ => panic!("no put") };
    // c - p = S - K*exp(-r*T) = 100 - 100*exp(-0.05) ≈ 4.877
    let parity = 100.0 - 100.0 * (-0.05_f64).exp();
    let diff = (c - p_v - parity).abs();
    assert!(diff < 0.1, "Put-call parity violation: c-p={}, expected {}, diff={}", c - p_v, parity, diff);
}

// ── Greeks ────────────────────────────────────────────────────────────────────

#[test]
fn test_bs_delta_call() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Delta of ATM call should be near 0.5
    let mut p = Parser::new(r#"QUERY t COMPUTE d = BS_DELTA(100.0, 100.0, 1.0, 0.05, 0.2, "call") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f > 0.4 && *f < 0.7, "Call delta ~0.5-0.6, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_option_delta_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE d = OPTION_DELTA(100.0, 100.0, 1.0, 0.05, 0.2, "call") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f > 0.4 && *f < 0.7, "OPTION_DELTA alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_delta_put() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Put delta should be negative, ATM near -0.5
    let mut p = Parser::new(r#"QUERY t COMPUTE d = BS_DELTA(100.0, 100.0, 1.0, 0.05, 0.2, "put") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f < 0.0 && *f > -0.7, "Put delta negative, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_gamma() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Gamma is always positive
    let mut p = Parser::new(r#"QUERY t COMPUTE g = BS_GAMMA(100.0, 100.0, 1.0, 0.05, 0.2) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Gamma > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_option_gamma_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE g = OPTION_GAMMA(100.0, 100.0, 1.0, 0.05, 0.2) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "OPTION_GAMMA > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_vega() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Vega is always positive
    let mut p = Parser::new(r#"QUERY t COMPUTE v = BS_VEGA(100.0, 100.0, 1.0, 0.05, 0.2) SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Vega > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_option_vega_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE v = OPTION_VEGA(100.0, 100.0, 1.0, 0.05, 0.2) SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("v") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "OPTION_VEGA alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_theta_call() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Theta for call is typically negative (time decay)
    let mut p = Parser::new(r#"QUERY t COMPUTE th = BS_THETA(100.0, 100.0, 1.0, 0.05, 0.2, "call") SELECT th;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("th") {
        Some(Value::Float(f)) => assert!(*f < 0.0, "Theta call < 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_option_theta_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE th = OPTION_THETA(100.0, 100.0, 1.0, 0.05, 0.2, "call") SELECT th;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("th") {
        Some(Value::Float(f)) => assert!(*f < 0.0, "OPTION_THETA alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_rho_call() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Rho for call is positive
    let mut p = Parser::new(r#"QUERY t COMPUTE rh = BS_RHO(100.0, 100.0, 1.0, 0.05, 0.2, "call") SELECT rh;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rh") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Rho call > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bs_rho_put() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Rho for put is negative
    let mut p = Parser::new(r#"QUERY t COMPUTE rh = BS_RHO(100.0, 100.0, 1.0, 0.05, 0.2, "put") SELECT rh;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rh") {
        Some(Value::Float(f)) => assert!(*f < 0.0, "Rho put < 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_option_rho_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE rh = OPTION_RHO(100.0, 100.0, 1.0, 0.05, 0.2, "call") SELECT rh;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rh") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "OPTION_RHO alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Implied Volatility ────────────────────────────────────────────────────────

#[test]
fn test_implied_volatility_call() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Known price ~ 10.45 from sigma=0.2 call, recover sigma
    let mut p = Parser::new(r#"QUERY t COMPUTE iv = IMPLIED_VOLATILITY(10.45, 100.0, 100.0, 1.0, 0.05, "call") SELECT iv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("iv") {
        Some(Value::Float(f)) => assert!(*f > 0.15 && *f < 0.25, "IV ~0.2, got {}", f),
        other => panic!("expected Float (IV), got {:?}", other),
    }
}

#[test]
fn test_impl_vol_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE iv = IMPL_VOL(10.45, 100.0, 100.0, 1.0, 0.05, "call") SELECT iv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("iv") {
        Some(Value::Float(f)) => assert!(*f > 0.1 && *f < 0.35, "IMPL_VOL alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Portfolio Theory ──────────────────────────────────────────────────────────

#[test]
fn test_portfolio_return() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // weights=[0.6, 0.4], returns=[0.1, 0.05] -> 0.6*0.1 + 0.4*0.05 = 0.08
    let mut p = Parser::new(r#"QUERY t COMPUTE pr = PORTFOLIO_RETURN([0.6, 0.4], [0.1, 0.05]) SELECT pr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pr") {
        Some(Value::Float(f)) => assert!((*f - 0.08).abs() < 1e-9, "Port return = 0.08, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_port_return_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE pr = PORT_RETURN([0.5, 0.5], [0.1, 0.1]) SELECT pr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pr") {
        Some(Value::Float(f)) => assert!((*f - 0.1).abs() < 1e-9, "PORT_RETURN alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_portfolio_variance() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Diagonal cov matrix: [[0.04,0],[0,0.01]], weights=[0.5,0.5]
    // var = 0.25*0.04 + 0.25*0.01 = 0.0125
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = PORTFOLIO_VARIANCE([0.5, 0.5], [[0.04, 0.0], [0.0, 0.01]]) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pv") {
        Some(Value::Float(f)) => assert!((*f - 0.0125).abs() < 1e-9, "Port var = 0.0125, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_port_var_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = PORT_VAR([1.0], [[0.04]]) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pv") {
        Some(Value::Float(f)) => assert!((*f - 0.04).abs() < 1e-9, "PORT_VAR alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_portfolio_std() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Diagonal: [[0.04,0],[0,0.01]], weights=[0.5,0.5], var=0.0125, std=sqrt(0.0125)
    let mut p = Parser::new(r#"QUERY t COMPUTE ps = PORTFOLIO_STD([0.5, 0.5], [[0.04, 0.0], [0.0, 0.01]]) SELECT ps;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ps") {
        Some(Value::Float(f)) => assert!((*f - 0.0125_f64.sqrt()).abs() < 1e-9, "Port std, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_port_std_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE ps = PORT_STD([1.0], [[0.09]]) SELECT ps;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ps") {
        Some(Value::Float(f)) => assert!((*f - 0.3).abs() < 1e-9, "PORT_STD alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_portfolio_sharpe() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // weights=[1], returns=[0.1], cov=[[0.04]], rf=0.02 -> Sharpe=(0.1-0.02)/0.2=0.4
    let mut p = Parser::new(r#"QUERY t COMPUTE sh = PORTFOLIO_SHARPE([1.0], [0.1], [[0.04]], 0.02) SELECT sh;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sh") {
        Some(Value::Float(f)) => assert!((*f - 0.4).abs() < 1e-9, "Sharpe = 0.4, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_port_sharpe_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE sh = PORT_SHARPE([1.0], [0.1], [[0.04]]) SELECT sh;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sh") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "PORT_SHARPE alias > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_equal_weight_portfolio() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE ew = EQUAL_WEIGHT_PORTFOLIO(4) SELECT ew;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ew") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            for w in arr {
                if let Value::Float(f) = w {
                    assert!((*f - 0.25).abs() < 1e-9, "Each weight = 0.25, got {}", f);
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_ew_portfolio_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE ew = EW_PORTFOLIO(3) SELECT ew;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ew") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 3),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_minimum_variance_weights() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Diagonal cov: higher variance asset gets lower weight
    let mut p = Parser::new(r#"QUERY t COMPUTE mv = MINIMUM_VARIANCE_WEIGHTS([[0.04, 0.0], [0.0, 0.01]]) SELECT mv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            let w0 = if let Value::Float(f) = &arr[0] { *f } else { panic!("w0") };
            let w1 = if let Value::Float(f) = &arr[1] { *f } else { panic!("w1") };
            // Asset 0 var=0.04, asset 1 var=0.01 -> w0=1/0.04/(1/0.04+1/0.01)=0.2
            assert!((w0 - 0.2).abs() < 1e-9, "w0=0.2, got {}", w0);
            assert!((w1 - 0.8).abs() < 1e-9, "w1=0.8, got {}", w1);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_min_var_weights_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE mv = MIN_VAR_WEIGHTS([[0.09, 0.0], [0.0, 0.04]]) SELECT mv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mv") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_tracking_error() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Identical returns -> tracking error = 0
    let mut p = Parser::new(r#"QUERY t COMPUTE te = TRACKING_ERROR([0.1, 0.05, 0.08], [0.1, 0.05, 0.08]) SELECT te;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("te") {
        Some(Value::Float(f)) => assert!(*f < 1e-9, "TE = 0 for identical, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_track_error_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE te = TRACK_ERROR([0.1, 0.05], [0.05, 0.05]) SELECT te;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("te") {
        Some(Value::Float(f)) => assert!(*f >= 0.0, "TRACK_ERROR alias >= 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Fixed Income ──────────────────────────────────────────────────────────────

#[test]
fn test_macaulay_duration() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Zero coupon bond: single CF at t=5, YTM=0.05, face=1000 -> duration = 5.0
    let mut p = Parser::new(r#"QUERY t COMPUTE dur = DURATION_MACAULAY([0.0, 0.0, 0.0, 0.0, 1000.0], 0.05, 1000.0) SELECT dur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dur") {
        Some(Value::Float(f)) => assert!((*f - 5.0).abs() < 1e-9, "Zero coupon duration = 5, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_macaulay_duration_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE dur = MACAULAY_DURATION([1000.0], 0.05, 1000.0) SELECT dur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dur") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-9, "Single CF duration = 1, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_modified_duration() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Zero coupon 5yr: Macaulay=5, Modified=5/(1+ytm)=5/1.05
    let mut p = Parser::new(r#"QUERY t COMPUTE mdur = DURATION_MODIFIED([0.0, 0.0, 0.0, 0.0, 1000.0], 0.05) SELECT mdur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mdur") {
        Some(Value::Float(f)) => {
            let expected = 5.0 / 1.05;
            assert!((*f - expected).abs() < 1e-9, "Modified duration = {}, got {}", expected, f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_modified_duration_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE mdur = MODIFIED_DURATION([1000.0], 0.1) SELECT mdur;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mdur") {
        Some(Value::Float(f)) => assert!((*f - 1.0 / 1.1).abs() < 1e-9, "MODIFIED_DURATION alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_convexity() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Convexity is always positive for a bond
    let mut p = Parser::new(r#"QUERY t COMPUTE conv = CONVEXITY([50.0, 50.0, 1050.0], 0.05) SELECT conv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("conv") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "Convexity > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bond_convexity_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE conv = BOND_CONVEXITY([50.0, 50.0, 1050.0], 0.05) SELECT conv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("conv") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "BOND_CONVEXITY alias > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_duration_price_change() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // price=1000, mod_dur=5, delta_y=0.01 -> change = -5*1000*0.01 = -50
    let mut p = Parser::new(r#"QUERY t COMPUTE dpc = DURATION_PRICE_CHANGE(1000.0, 5.0, 0.01) SELECT dpc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dpc") {
        Some(Value::Float(f)) => assert!((*f - (-50.0)).abs() < 1e-9, "DPC = -50, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_duration_dv01_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE dpc = DURATION_DV01(1000.0, 5.0, 0.01) SELECT dpc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dpc") {
        Some(Value::Float(f)) => assert!((*f - (-50.0)).abs() < 1e-9, "DURATION_DV01 alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_current_yield() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // coupon=50, price=1000 -> current yield = 0.05
    let mut p = Parser::new(r#"QUERY t COMPUTE cy = CURRENT_YIELD(50.0, 1000.0) SELECT cy;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cy") {
        Some(Value::Float(f)) => assert!((*f - 0.05).abs() < 1e-9, "CY = 0.05, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_cur_yield_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE cy = CUR_YIELD(60.0, 1200.0) SELECT cy;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cy") {
        Some(Value::Float(f)) => assert!((*f - 0.05).abs() < 1e-9, "CUR_YIELD alias = 0.05, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_accrued_interest() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // face=1000, rate=0.1, 45 days since coupon, 180 days in period
    // = 1000 * 0.1/2 * 45/180 = 1000*0.05*0.25 = 12.5
    let mut p = Parser::new(r#"QUERY t COMPUTE ai = ACCRUED_INTEREST(1000.0, 0.1, 45.0, 180.0) SELECT ai;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ai") {
        Some(Value::Float(f)) => assert!((*f - 12.5).abs() < 1e-9, "Accrued = 12.5, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_accrued_int_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE ai = ACCRUED_INT(1000.0, 0.08, 90.0, 180.0) SELECT ai;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ai") {
        Some(Value::Float(f)) => {
            // 1000 * 0.08/2 * 90/180 = 1000 * 0.04 * 0.5 = 20
            assert!((*f - 20.0).abs() < 1e-9, "ACCRUED_INT alias = 20, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_yield_to_call() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // par bond call in 3 periods, coupon=50, call=1000, price=1000 -> YTC=5%
    let mut p = Parser::new(r#"QUERY t COMPUTE ytc = YIELD_TO_CALL(1000.0, 1000.0, 50.0, 3, 1000.0) SELECT ytc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ytc") {
        Some(Value::Float(f)) => assert!((*f - 0.05).abs() < 0.001, "YTC ~5%, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_ytc_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE ytc = YTC(1000.0, 1000.0, 50.0, 3, 1000.0) SELECT ytc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ytc") {
        Some(Value::Float(f)) => assert!((*f - 0.05).abs() < 0.001, "YTC alias ~5%, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Monte Carlo ───────────────────────────────────────────────────────────────

#[test]
fn test_monte_carlo_normal_count() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE s = MONTE_CARLO_NORMAL(100, 0.0, 1.0, 42) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 100, "100 samples"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_mc_normal_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE s = MC_NORMAL(50, 0.0, 1.0, 42) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 50, "MC_NORMAL alias 50 samples"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_monte_carlo_normal_floats() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE s = MONTE_CARLO_NORMAL(10, 5.0, 1.0, 99) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            for v in arr {
                assert!(matches!(v, Value::Float(_)), "All Float");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_monte_carlo_uniform_count() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE s = MONTE_CARLO_UNIFORM(100, 0.0, 1.0, 42) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 100, "100 uniform samples"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_mc_uniform_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE s = MC_UNIFORM(50, 10.0, 20.0, 42) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 50, "MC_UNIFORM alias 50 samples"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_monte_carlo_uniform_range() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE s = MONTE_CARLO_UNIFORM(200, 5.0, 10.0, 7) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            for v in arr {
                if let Value::Float(f) = v {
                    assert!(*f >= 5.0 && *f <= 10.0, "In [5,10]: {}", f);
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_monte_carlo_pi() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // With many samples pi estimate should be close-ish to pi
    let mut p = Parser::new(r#"QUERY t COMPUTE pi_est = MONTE_CARLO_PI(10000, 1234) SELECT pi_est;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pi_est") {
        Some(Value::Float(f)) => assert!(*f > 2.5 && *f < 3.8, "MC PI estimate in [2.5,3.8], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_mc_pi_estimate_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE pi_est = MC_PI_ESTIMATE(10000, 1234) SELECT pi_est;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pi_est") {
        Some(Value::Float(f)) => assert!(*f > 2.5 && *f < 3.8, "MC_PI_ESTIMATE alias, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_gbm_path_count() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // GBM_PATH(S0, mu, sigma, n_steps, dt, seed) -> n_steps+1 values
    let mut p = Parser::new(r#"QUERY t COMPUTE gbm = GEOMETRIC_BROWNIAN_MOTION(100.0, 0.1, 0.2, 10, 0.1, 42) SELECT gbm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gbm") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 11, "11 points (10 steps + initial)"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_gbm_path_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE gbm = GBM_PATH(100.0, 0.1, 0.2, 5, 0.1, 42) SELECT gbm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gbm") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 6, "GBM_PATH alias 6 points"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_gbm_path_positive_prices() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE gbm = GBM_PATH(50.0, 0.05, 0.3, 20, 0.05, 999) SELECT gbm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gbm") {
        Some(Value::Array(arr)) => {
            for v in arr {
                if let Value::Float(f) = v {
                    assert!(*f > 0.0, "All prices positive: {}", f);
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_gbm_initial_price() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE gbm = GBM_PATH(123.45, 0.1, 0.2, 5, 0.1, 1) SELECT gbm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gbm") {
        Some(Value::Array(arr)) => {
            if let Some(Value::Float(f)) = arr.first() {
                assert!((*f - 123.45).abs() < 1e-9, "First element = S0, got {}", f);
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
