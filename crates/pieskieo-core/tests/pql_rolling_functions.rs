/// Integration tests for PQL rolling window and cumulative aggregate functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_rolling_sum() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [1, 2, 3, 4, 5] {
        db.put_doc_ns(
            None,
            Some("ts"),
            Uuid::new_v4(),
            serde_json::json!({"val": v}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE rs = ROLLING_SUM(val, 3) SELECT rs;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rs") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 5, "rolling sum should have 5 values"),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_cummax_cummin() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [3, 1, 4, 1, 5, 9, 2] {
        db.put_doc_ns(
            None,
            Some("ts"),
            Uuid::new_v4(),
            serde_json::json!({"val": v}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE cmax = CUMMAX(val) COMPUTE cmin = CUMMIN(val) SELECT cmax, cmin;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cmax") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 7);
            // Last cummax should be 9 (the overall max)
            assert_eq!(arr.last(), Some(&Value::Float(9.0)));
        }
        other => panic!("expected array for cmax, got {:?}", other),
    }
}

#[test]
fn test_expanding_mean() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // [10, 20, 30] → cumulative means: [10, 15, 20]
    for v in [10, 20, 30] {
        db.put_doc_ns(
            None,
            Some("ts"),
            Uuid::new_v4(),
            serde_json::json!({"val": v}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE em = EXPANDING_MEAN(val) SELECT em;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("em") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            // Verify the means are all floats and the final cumulative mean = (10+20+30)/3 = 20.0
            let floats: Vec<f64> = arr
                .iter()
                .filter_map(|v| {
                    if let Value::Float(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(floats.len(), 3, "all 3 values should be floats");
            // The last cumulative mean is always (sum of all 3) / 3 = 20.0 regardless of row order
            assert!(
                (floats.last().unwrap() - 20.0).abs() < 0.001,
                "final cumulative mean should be 20.0, got {}",
                floats.last().unwrap()
            );
            // All values should be between the min (10) and max (30) of the series
            assert!(
                floats.iter().all(|&f| f >= 10.0 && f <= 30.0),
                "all cumulative means should be within [10, 30]"
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_zscore_series() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // [0, 0, 0, 0] → all z-scores = 0 (zero variance)
    for _ in 0..4 {
        db.put_doc_ns(
            None,
            Some("ts"),
            Uuid::new_v4(),
            serde_json::json!({"val": 5}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE z = ZSCORE_SERIES(val) SELECT z;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("z") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!(
                arr.iter().all(|v| v == &Value::Float(0.0)),
                "all z-scores should be 0 for constant series"
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_expanding_sum() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [1, 2, 3, 4] {
        db.put_doc_ns(
            None,
            Some("ts"),
            Uuid::new_v4(),
            serde_json::json!({"val": v}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY ts COMPUTE g = 1 GROUP BY g COMPUTE es = EXPANDING_SUM(val) SELECT es;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("es") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            // Last value should be the total sum = 10
            assert_eq!(arr.last(), Some(&Value::Float(10.0)));
        }
        other => panic!("expected array, got {:?}", other),
    }
}
