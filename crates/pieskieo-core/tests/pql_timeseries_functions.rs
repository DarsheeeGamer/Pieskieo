/// Integration tests for PQL time-series and forecasting functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_moving_average() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [1, 2, 3, 4, 5] {
        db.put_doc_ns(None, Some("ts"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE ma = MOVING_AVERAGE(val, 3) SELECT ma;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("ma") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "moving average should have 5 values");
            // All values should be floats
            for v in arr {
                assert!(matches!(v, Value::Float(_)), "each moving average value should be a float");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_exponential_ma() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [10, 20, 30, 40, 50] {
        db.put_doc_ns(None, Some("ts"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE ema = EMA(val, 0.5) SELECT ema;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("ema") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            // All values should be floats between 10 and 50
            for v in arr {
                match v {
                    Value::Float(f) => {
                        assert!(*f >= 10.0 && *f <= 50.0, "EMA value {} should be between 10 and 50", f);
                    }
                    other => panic!("expected float, got {:?}", other),
                }
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_linear_trend() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // All same value - slope must be 0 regardless of ordering
    for v in [5, 5, 5, 5, 5] {
        db.put_doc_ns(None, Some("ts"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE slope = LINEAR_TREND(val) SELECT slope;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("slope") {
        Some(Value::Float(f)) => assert!(f.abs() < 0.001, "slope of constant series should be 0.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_delta() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [10, 15, 12, 20] {
        db.put_doc_ns(None, Some("ts"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE d = DELTA(val) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("d") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "delta of 4 values should have 3 diffs");
            // All values should be floats
            for v in arr {
                assert!(matches!(v, Value::Float(_)), "each delta value should be a float");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_normalize_series() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [0, 25, 50, 75, 100] {
        db.put_doc_ns(None, Some("ts"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE n = NORMALIZE_SERIES(val) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("n") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            // min should be 0.0 and max should be 1.0 somewhere in the array
            let has_zero = arr.iter().any(|v| v == &Value::Float(0.0));
            let has_one = arr.iter().any(|v| v == &Value::Float(1.0));
            assert!(has_zero, "normalized array should contain 0.0 (minimum)");
            assert!(has_one, "normalized array should contain 1.0 (maximum)");
            // All values should be in [0, 1]
            for v in arr {
                match v {
                    Value::Float(f) => assert!(*f >= 0.0 && *f <= 1.0, "normalized value {} out of [0,1]", f),
                    other => panic!("expected float, got {:?}", other),
                }
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_detect_outliers() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 99 is a clear outlier among values near 10
    for v in [10, 11, 10, 12, 99, 10] {
        db.put_doc_ns(None, Some("ts"), Uuid::new_v4(), serde_json::json!({"val": v})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE out = DETECT_OUTLIERS(val, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1);
    match r.rows[0].data.get("out") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6);
            // At least one should be true (the 99)
            let any_outlier = arr.iter().any(|v| v == &Value::Bool(true));
            assert!(any_outlier, "should detect at least one outlier");
        }
        other => panic!("expected array, got {:?}", other),
    }
}
