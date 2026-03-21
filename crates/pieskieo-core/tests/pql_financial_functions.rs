/// Integration tests for PQL financial and business math functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_npv() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // NPV at 10% of [-1000, 400, 400, 400] ≈ -6.16
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rate": 0.1}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE n = NPV(rate, [400, 400, 400]) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(
            *f > 800.0 && *f < 1200.0,
            "NPV of [400,400,400] at 10% should be ~994, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_cagr() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // CAGR(1000, 1610, 5) ≈ 10%
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"start_val": 1000.0, "end_val": 1610.51, "years": 5.0}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE c = CAGR(start_val, end_val, years) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => {
            assert!((*f - 0.10).abs() < 0.01, "CAGR should be ~0.10, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_roi() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // ROI(150, 100) = 0.5 (50%)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"gain": 150.0, "cost": 100.0}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE r = ROI(gain, cost) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.001, "ROI should be 0.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_compound_interest() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Compound: 1000 at 10% for 1 year = 1100
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"principal": 1000.0, "rate": 0.10, "t": 1.0}),
    )
    .unwrap();

    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = COMPOUND_INTEREST(principal, rate, t) SELECT a;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!((*f - 1100.0).abs() < 0.01, "should be 1100, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_profit_margin() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Margin: revenue=100, cost=70 → 30%
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"revenue": 100.0, "cost": 70.0}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE m = PROFIT_MARGIN(revenue, cost) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => {
            assert!((*f - 0.3).abs() < 0.001, "margin should be 0.3, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_pmt_pv() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // PMT for 10000 loan at 5% monthly rate for 12 periods
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"rate": 0.05, "n": 12, "pv": 10000.0}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE p = PMT(rate, n, pv) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(f)) => assert!(f.abs() > 0.0, "PMT should be non-zero, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_depreciation_sl() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // SL depreciation: cost=10000, salvage=1000, life=9 → 1000/year
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"cost": 10000.0, "salvage": 1000.0, "life": 9.0}),
    )
    .unwrap();

    let mut p =
        Parser::new(r#"QUERY t COMPUTE d = DEPRECIATION_SL(cost, salvage, life) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f - 1000.0).abs() < 0.01,
            "depreciation should be 1000, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}
