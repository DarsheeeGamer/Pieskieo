/// Integration tests for PQL streaming and session window processing functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Float(f) => *f as i64,
        _ => i64::MIN,
    }
}

// ── TUMBLING_WINDOW ───────────────────────────────────────────────────────────

#[test]
fn test_tumbling_window_sum() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TUMBLING_WINDOW(arr, 2, "SUM") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 windows for len=6 window_size=2");
            // windows: [1,2]=3, [3,4]=7, [5,6]=11
            assert!(
                (as_f64(&a[0]) - 3.0).abs() < 0.01,
                "window 0 sum: expected 3.0, got {}",
                as_f64(&a[0])
            );
            assert!(
                (as_f64(&a[1]) - 7.0).abs() < 0.01,
                "window 1 sum: expected 7.0, got {}",
                as_f64(&a[1])
            );
            assert!(
                (as_f64(&a[2]) - 11.0).abs() < 0.01,
                "window 2 sum: expected 11.0, got {}",
                as_f64(&a[2])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_tumbling_window_avg() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30, 40]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TUMBLING_WINDOW(arr, 2, "AVG") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            // [10,20] avg=15, [30,40] avg=35
            assert!(
                (as_f64(&a[0]) - 15.0).abs() < 0.01,
                "expected 15.0, got {}",
                as_f64(&a[0])
            );
            assert!(
                (as_f64(&a[1]) - 35.0).abs() < 0.01,
                "expected 35.0, got {}",
                as_f64(&a[1])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_fixed_window_alias() {
    // FIXED_WINDOW is an alias for TUMBLING_WINDOW
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [2, 4, 6]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = FIXED_WINDOW(arr, 3, "COUNT") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1);
            assert_eq!(as_i64(&a[0]), 3, "COUNT of [2,4,6] should be 3");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SLIDING_WINDOW_AGG ────────────────────────────────────────────────────────

#[test]
fn test_sliding_window_agg_sum() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = SLIDING_WINDOW_AGG(arr, 3, "SUM") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // len=5, window=3 => 3 results: [1,2,3]=6, [2,3,4]=9, [3,4,5]=12
            assert_eq!(a.len(), 3, "expected 3 results for sliding window");
            assert!(
                (as_f64(&a[0]) - 6.0).abs() < 0.01,
                "expected 6.0, got {}",
                as_f64(&a[0])
            );
            assert!(
                (as_f64(&a[1]) - 9.0).abs() < 0.01,
                "expected 9.0, got {}",
                as_f64(&a[1])
            );
            assert!(
                (as_f64(&a[2]) - 12.0).abs() < 0.01,
                "expected 12.0, got {}",
                as_f64(&a[2])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_window_agg_alias() {
    // ROLLING_WINDOW_AGG is an alias for SLIDING_WINDOW_AGG
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 1, 3]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = ROLLING_WINDOW_AGG(arr, 2, "MAX") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            // [5,1] max=5, [1,3] max=3
            assert!(
                (as_f64(&a[0]) - 5.0).abs() < 0.01,
                "expected 5.0, got {}",
                as_f64(&a[0])
            );
            assert!(
                (as_f64(&a[1]) - 3.0).abs() < 0.01,
                "expected 3.0, got {}",
                as_f64(&a[1])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SESSION_WINDOW ────────────────────────────────────────────────────────────

#[test]
fn test_session_window_basic() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // [1,2,3,10,11,20,21,22] with gap=2 => 3 sessions
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 10, 11, 20, 21, 22]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SESSION_WINDOW(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(sessions)) => {
            assert_eq!(sessions.len(), 3, "expected 3 sessions, got {}", sessions.len());
            // First session: [1, 2, 3]
            if let Value::Array(s0) = &sessions[0] {
                assert_eq!(s0.len(), 3, "session 0 should have 3 elements");
                assert_eq!(as_i64(&s0[0]), 1);
                assert_eq!(as_i64(&s0[2]), 3);
            } else {
                panic!("session 0 is not an Array");
            }
            // Second session: [10, 11]
            if let Value::Array(s1) = &sessions[1] {
                assert_eq!(s1.len(), 2, "session 1 should have 2 elements");
                assert_eq!(as_i64(&s1[0]), 10);
            } else {
                panic!("session 1 is not an Array");
            }
            // Third session: [20, 21, 22]
            if let Value::Array(s2) = &sessions[2] {
                assert_eq!(s2.len(), 3, "session 2 should have 3 elements");
            } else {
                panic!("session 2 is not an Array");
            }
        }
        other => panic!("expected Array of Arrays, got {:?}", other),
    }
}

#[test]
fn test_session_gaps_alias() {
    // SESSION_GAPS is an alias; single big gap => 2 sessions
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 100, 101]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SESSION_GAPS(arr, 5) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(sessions)) => {
            assert_eq!(sessions.len(), 2, "expected 2 sessions, got {}", sessions.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── EVENT_COUNT_WINDOW ────────────────────────────────────────────────────────

#[test]
fn test_event_count_window() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Array with nulls: [1, null, 3, 4, null]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, null, 3, 4, null]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = EVENT_COUNT_WINDOW(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // len=5, window=3 => 3 results
            assert_eq!(a.len(), 3, "expected 3 results");
            // [1,null,3] => 2 non-null
            assert_eq!(as_i64(&a[0]), 2, "expected 2, got {}", as_i64(&a[0]));
            // [null,3,4] => 2 non-null
            assert_eq!(as_i64(&a[1]), 2, "expected 2, got {}", as_i64(&a[1]));
            // [3,4,null] => 2 non-null
            assert_eq!(as_i64(&a[2]), 2, "expected 2, got {}", as_i64(&a[2]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── STREAK_LENGTH ─────────────────────────────────────────────────────────────

#[test]
fn test_streak_length_with_value() {
    // STREAK_LENGTH([1,1,0,1,1,1], 1) -> 3
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 1, 0, 1, 1, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = STREAK_LENGTH(arr, 1) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            assert_eq!(as_i64(v), 3, "streak of 1s at end should be 3, got {}", as_i64(v));
        }
        None => panic!("expected a value for 'out'"),
    }
}

#[test]
fn test_consecutive_count_alias_no_value() {
    // CONSECUTIVE_COUNT([5,5,5]) with no value arg -> count last-equal tail -> 3
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [2, 5, 5, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CONSECUTIVE_COUNT(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            assert_eq!(as_i64(v), 3, "trailing 5s streak should be 3, got {}", as_i64(v));
        }
        None => panic!("expected a value for 'out'"),
    }
}

// ── CHANGE_POINTS ─────────────────────────────────────────────────────────────

#[test]
fn test_change_points() {
    // [1, 1, 1, 10, 1, 1] with min_change=5 => change at index 3 and 4
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 1, 1, 10, 1, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CHANGE_POINTS(arr, 5) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "expected 2 change points, got {}", a.len());
            assert_eq!(as_i64(&a[0]), 3, "first change point should be index 3");
            assert_eq!(as_i64(&a[1]), 4, "second change point should be index 4");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_detect_change_points_alias() {
    // DETECT_CHANGE_POINTS is an alias for CHANGE_POINTS
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0, 100]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = DETECT_CHANGE_POINTS(arr, 50) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1, "expected 1 change point");
            assert_eq!(as_i64(&a[0]), 1);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── CONSECUTIVE_DIFFERENCES ───────────────────────────────────────────────────

#[test]
fn test_consecutive_differences() {
    // [1,3,6,10] -> [2,3,4]
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 3, 6, 10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CONSECUTIVE_DIFFERENCES(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 differences for 4-element array");
            assert!((as_f64(&a[0]) - 2.0).abs() < 0.01, "expected 2.0, got {}", as_f64(&a[0]));
            assert!((as_f64(&a[1]) - 3.0).abs() < 0.01, "expected 3.0, got {}", as_f64(&a[1]));
            assert!((as_f64(&a[2]) - 4.0).abs() < 0.01, "expected 4.0, got {}", as_f64(&a[2]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_first_differences_alias() {
    // FIRST_DIFFERENCES is alias; test with negative differences
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 7, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FIRST_DIFFERENCES(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            assert!((as_f64(&a[0]) - (-3.0)).abs() < 0.01, "expected -3.0, got {}", as_f64(&a[0]));
            assert!((as_f64(&a[1]) - (-3.0)).abs() < 0.01, "expected -3.0, got {}", as_f64(&a[1]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── WINDOW_RANK ───────────────────────────────────────────────────────────────

#[test]
fn test_window_rank() {
    // [5, 3, 8, 1] with window_size=3
    // i=0: window=[5], rank of 5 among [5] = 1
    // i=1: window=[5,3], rank of 3 among [5,3] = 2 (one value >3)
    // i=2: window=[5,3,8], rank of 8 among [5,3,8] = 1 (none >8)
    // i=3: window=[3,8,1], rank of 1 among [3,8,1] = 3 (two values >1)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 3, 8, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = WINDOW_RANK(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(as_i64(&a[0]), 1, "rank of 5 in [5] should be 1");
            assert_eq!(as_i64(&a[1]), 2, "rank of 3 in [5,3] should be 2");
            assert_eq!(as_i64(&a[2]), 1, "rank of 8 in [5,3,8] should be 1");
            assert_eq!(as_i64(&a[3]), 3, "rank of 1 in [3,8,1] should be 3");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── PEAK_VALLEY ───────────────────────────────────────────────────────────────

#[test]
fn test_peak_valley_peaks() {
    // [1, 3, 2, 5, 4, 6] - peaks at index 1 (3>1,3>2) and 3 (5>2,5>4)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 3, 2, 5, 4, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PEAK_VALLEY(arr, "peaks") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert!(a.len() >= 1, "expected at least 1 peak");
            assert_eq!(as_i64(&a[0]), 1, "first peak should be at index 1");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_peak_valley_valleys() {
    // [5, 1, 4, 0, 3] - valleys at index 1 (1<5,1<4) and index 3 (0<4,0<3)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 1, 4, 0, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PEAK_VALLEY(arr, "valleys") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert!(a.len() >= 1, "expected at least 1 valley");
            assert_eq!(as_i64(&a[0]), 1, "first valley should be at index 1");
            assert_eq!(as_i64(&a[1]), 3, "second valley should be at index 3");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_detect_peaks_alias() {
    // DETECT_PEAKS alias; test "both" mode
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 5, 2, 0, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = DETECT_PEAKS(arr, "both") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // Peak at 1 (5>1,5>2), valley at 3 (0<2,0<4)
            assert!(a.len() >= 2, "expected at least 2 extrema (peaks+valleys), got {}", a.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RATE_OF_CHANGE ────────────────────────────────────────────────────────────

#[test]
fn test_rate_of_change() {
    // [100, 110, 99] -> [10.0, -10.0]
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [100, 110, 99]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = RATE_OF_CHANGE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "expected 2 rate values");
            // (110-100)/100*100 = 10.0
            assert!(
                (as_f64(&a[0]) - 10.0).abs() < 0.01,
                "expected 10.0, got {}",
                as_f64(&a[0])
            );
            // (99-110)/110*100 = -10.0
            let expected = (99.0 - 110.0) / 110.0 * 100.0;
            assert!(
                (as_f64(&a[1]) - expected).abs() < 0.01,
                "expected {:.2}, got {}",
                expected,
                as_f64(&a[1])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_pct_change_alias() {
    // PCT_CHANGE is an alias for RATE_OF_CHANGE
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [50, 100]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PCT_CHANGE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1);
            assert!(
                (as_f64(&a[0]) - 100.0).abs() < 0.01,
                "expected 100.0% increase, got {}",
                as_f64(&a[0])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── EXPONENTIAL_DECAY ─────────────────────────────────────────────────────────

#[test]
fn test_exponential_decay() {
    // arr=[1,0,0], decay=0.5
    // result[2] = 0
    // result[1] = 0 + 0.5 * 0 = 0
    // result[0] = 1 + 0.5 * 0 = 1
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 0, 0]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = EXPONENTIAL_DECAY(arr, 0.5) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert!(
                (as_f64(&a[0]) - 1.0).abs() < 0.01,
                "result[0] should be 1.0, got {}",
                as_f64(&a[0])
            );
            assert!(
                (as_f64(&a[1]) - 0.0).abs() < 0.01,
                "result[1] should be 0.0, got {}",
                as_f64(&a[1])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_exp_decay_alias() {
    // EXP_DECAY alias; verify length is correct
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [4, 3, 2, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = EXP_DECAY(arr, 0.9) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4, "output length should equal input length");
            // All values should be floats
            for v in a {
                assert!(matches!(v, Value::Float(_)), "each value should be Float, got {:?}", v);
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── FILL_FORWARD ──────────────────────────────────────────────────────────────

#[test]
fn test_fill_forward() {
    // [1, null, null, 4, null] -> [1, 1, 1, 4, 4]
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, null, null, 4, null]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FILL_FORWARD(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
            assert_eq!(as_i64(&a[0]), 1);
            assert_eq!(as_i64(&a[1]), 1, "null should be filled with 1");
            assert_eq!(as_i64(&a[2]), 1, "null should be filled with 1");
            assert_eq!(as_i64(&a[3]), 4);
            assert_eq!(as_i64(&a[4]), 4, "trailing null should be filled with 4");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_ffill_alias() {
    // FFILL is an alias for FILL_FORWARD
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [7, null, null]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FFILL(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(as_i64(&a[1]), 7, "null[1] should fill to 7");
            assert_eq!(as_i64(&a[2]), 7, "null[2] should fill to 7");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── FILL_BACKWARD ─────────────────────────────────────────────────────────────

#[test]
fn test_fill_backward() {
    // [null, null, 3, null, 5] -> [3, 3, 3, 5, 5]
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [null, null, 3, null, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FILL_BACKWARD(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
            assert_eq!(as_i64(&a[0]), 3, "leading null[0] should backfill to 3");
            assert_eq!(as_i64(&a[1]), 3, "leading null[1] should backfill to 3");
            assert_eq!(as_i64(&a[2]), 3);
            assert_eq!(as_i64(&a[3]), 5, "null[3] should backfill to 5");
            assert_eq!(as_i64(&a[4]), 5);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_bfill_alias() {
    // BFILL is an alias for FILL_BACKWARD
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [null, 9]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = BFILL(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            assert_eq!(as_i64(&a[0]), 9, "leading null should backfill to 9");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── INTERPOLATE_NULLS ─────────────────────────────────────────────────────────

#[test]
fn test_interpolate_nulls() {
    // [1, null, null, 4] -> [1.0, 2.0, 3.0, 4.0]
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, null, null, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INTERPOLATE_NULLS(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
            assert!((as_f64(&a[0]) - 1.0).abs() < 0.01, "expected 1.0, got {}", as_f64(&a[0]));
            assert!((as_f64(&a[1]) - 2.0).abs() < 0.01, "expected 2.0, got {}", as_f64(&a[1]));
            assert!((as_f64(&a[2]) - 3.0).abs() < 0.01, "expected 3.0, got {}", as_f64(&a[2]));
            assert!((as_f64(&a[3]) - 4.0).abs() < 0.01, "expected 4.0, got {}", as_f64(&a[3]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_linear_fill_alias() {
    // LINEAR_FILL is an alias; test single null in middle
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0, null, 10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = LINEAR_FILL(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            // Midpoint between 0 and 10 = 5
            assert!((as_f64(&a[1]) - 5.0).abs() < 0.01, "expected 5.0, got {}", as_f64(&a[1]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn test_consecutive_differences_single_element_returns_empty() {
    // Array with 1 element should return empty differences
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [42]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CONSECUTIVE_DIFFERENCES(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert!(a.is_empty(), "single-element array should yield 0 differences");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_session_window_no_gaps_single_session() {
    // All values within gap => single session
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SESSION_WINDOW(arr, 10) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(sessions)) => {
            assert_eq!(sessions.len(), 1, "all within gap=10 should be 1 session");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_change_points_no_changes() {
    // Flat array should return no change points
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 5, 5, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CHANGE_POINTS(arr, 1) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert!(a.is_empty(), "flat array should have 0 change points");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
