/// Advanced integration tests for PQL time-series forecasting and anomaly detection functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup(ns: &str, doc: serde_json::Value) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    (dir, db, ex)
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

// ── SIMPLE_MOVING_AVG ────────────────────────────────────────────────────────

#[test]
fn test_simple_moving_avg_basic() {
    let (_dir, _db, ex) = setup("ts_sma", serde_json::json!({"nums": [1, 2, 3, 4, 5]}));
    let mut p = Parser::new("QUERY ts_sma COMPUTE res = SIMPLE_MOVING_AVG(nums, 3) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            // windows(3) over 5 elements → 3 results
            assert_eq!(arr.len(), 3, "SMA window=3 over 5 elements should yield 3 values");
            // [1,2,3]→2, [2,3,4]→3, [3,4,5]→4
            let vals: Vec<f64> = arr.iter().map(as_f64).collect();
            assert!((vals[0] - 2.0).abs() < 1e-9, "first SMA = 2.0, got {}", vals[0]);
            assert!((vals[1] - 3.0).abs() < 1e-9, "second SMA = 3.0, got {}", vals[1]);
            assert!((vals[2] - 4.0).abs() < 1e-9, "third SMA = 4.0, got {}", vals[2]);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_simple_moving_avg_window_one() {
    // Window of 1 should return the original values
    let (_dir, _db, ex) = setup("ts_sma1", serde_json::json!({"nums": [10.0, 20.0, 30.0]}));
    let mut p = Parser::new("QUERY ts_sma1 COMPUTE res = SIMPLE_MOVING_AVG(nums, 1) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            let vals: Vec<f64> = arr.iter().map(as_f64).collect();
            assert!((vals[0] - 10.0).abs() < 1e-9);
            assert!((vals[1] - 20.0).abs() < 1e-9);
            assert!((vals[2] - 30.0).abs() < 1e-9);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── EXPONENTIAL_SMOOTHING / EMA_SMOOTH ──────────────────────────────────────

#[test]
fn test_exponential_smoothing_basic() {
    let (_dir, _db, ex) = setup("ts_ema", serde_json::json!({"nums": [1, 2, 3]}));
    let mut p = Parser::new("QUERY ts_ema COMPUTE res = EXPONENTIAL_SMOOTHING(nums, 0.5) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "EMA should return same length as input");
            // First value equals first input
            assert!((as_f64(&arr[0]) - 1.0).abs() < 1e-9, "first smoothed = 1.0");
            // All values should be floats
            for v in arr {
                assert!(matches!(v, Value::Float(_)), "expected Float in EMA result");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_ema_smooth_alias() {
    let (_dir, _db, ex) = setup("ts_ema2", serde_json::json!({"nums": [10.0, 20.0, 30.0]}));
    let mut p = Parser::new("QUERY ts_ema2 COMPUTE res = EMA_SMOOTH(nums, 0.3) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            // First value = 10.0; second = 0.3*20 + 0.7*10 = 13.0
            assert!((as_f64(&arr[0]) - 10.0).abs() < 1e-9);
            assert!((as_f64(&arr[1]) - 13.0).abs() < 1e-9, "second EMA_SMOOTH = 13.0, got {}", as_f64(&arr[1]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── DOUBLE_EXPONENTIAL / HOLT_SMOOTH ────────────────────────────────────────

#[test]
fn test_double_exponential_basic() {
    let (_dir, _db, ex) = setup("ts_holt", serde_json::json!({"nums": [1, 2, 3, 4, 5]}));
    let mut p = Parser::new("QUERY ts_holt COMPUTE res = DOUBLE_EXPONENTIAL(nums, 0.3, 0.1) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "HOLT_SMOOTH returns same length as input");
            for v in arr {
                assert!(matches!(v, Value::Float(_)), "expected Float values");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_holt_smooth_alias() {
    let (_dir, _db, ex) = setup("ts_holt2", serde_json::json!({"nums": [2, 4, 6, 8, 10]}));
    let mut p = Parser::new("QUERY ts_holt2 COMPUTE res = HOLT_SMOOTH(nums, 0.5, 0.2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "HOLT_SMOOTH alias returns same length as input");
            // Values should be positive and increasing for a linear series
            for v in arr {
                assert!(as_f64(v) > 0.0, "all Holt smoothed values should be positive");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SIMPLE_FORECAST / NAIVE_FORECAST ────────────────────────────────────────

#[test]
fn test_simple_forecast_basic() {
    let (_dir, _db, ex) = setup("ts_sf", serde_json::json!({"nums": [1, 2, 3]}));
    let mut p = Parser::new("QUERY ts_sf COMPUTE res = SIMPLE_FORECAST(nums, 3) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "forecast n=3 should return 3 values");
            // All should equal the last value (3)
            for v in arr {
                assert_eq!(as_f64(v) as i64, 3, "naive forecast should repeat last value");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_naive_forecast_alias() {
    let (_dir, _db, ex) = setup("ts_nf", serde_json::json!({"nums": [10, 20, 30]}));
    let mut p = Parser::new("QUERY ts_nf COMPUTE res = NAIVE_FORECAST(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            for v in arr {
                // last value = 30
                assert_eq!(as_f64(v) as i64, 30, "NAIVE_FORECAST should repeat last value 30");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── LINEAR_FORECAST / TREND_FORECAST ────────────────────────────────────────

#[test]
fn test_linear_forecast_basic() {
    // Perfect linear series [1,2,3,4], slope=1, intercept=1
    // Predictions for indices 4,5 → 5.0, 6.0
    let (_dir, _db, ex) = setup("ts_lf", serde_json::json!({"nums": [1, 2, 3, 4]}));
    let mut p = Parser::new("QUERY ts_lf COMPUTE res = LINEAR_FORECAST(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "LINEAR_FORECAST n=2 should return 2 values");
            let v0 = as_f64(&arr[0]);
            let v1 = as_f64(&arr[1]);
            assert!((v0 - 5.0).abs() < 1e-6, "first forecast = 5.0, got {}", v0);
            assert!((v1 - 6.0).abs() < 1e-6, "second forecast = 6.0, got {}", v1);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_trend_forecast_alias() {
    let (_dir, _db, ex) = setup("ts_tf", serde_json::json!({"nums": [0, 2, 4, 6]}));
    let mut p = Parser::new("QUERY ts_tf COMPUTE res = TREND_FORECAST(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            let v0 = as_f64(&arr[0]);
            let v1 = as_f64(&arr[1]);
            // slope=2, series [0,2,4,6], next would be 8, 10
            assert!((v0 - 8.0).abs() < 1e-6, "TREND_FORECAST first = 8.0, got {}", v0);
            assert!((v1 - 10.0).abs() < 1e-6, "TREND_FORECAST second = 10.0, got {}", v1);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SEASONAL_ADJUST / DESEASONALIZE ─────────────────────────────────────────

#[test]
fn test_seasonal_adjust_basic() {
    // [10,20,10,20] with period=2: seasonal_avg[0]=10, [1]=20
    // deseasonalized: [10/10, 20/20, 10/10, 20/20] = [1, 1, 1, 1]
    let (_dir, _db, ex) = setup("ts_sa", serde_json::json!({"nums": [10, 20, 10, 20]}));
    let mut p = Parser::new("QUERY ts_sa COMPUTE res = SEASONAL_ADJUST(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            for v in arr {
                assert!((as_f64(v) - 1.0).abs() < 1e-9, "deseasonalized value should be 1.0, got {}", as_f64(v));
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_deseasonalize_alias() {
    let (_dir, _db, ex) = setup("ts_ds", serde_json::json!({"nums": [5, 15, 5, 15]}));
    let mut p = Parser::new("QUERY ts_ds COMPUTE res = DESEASONALIZE(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            for v in arr {
                assert!((as_f64(v) - 1.0).abs() < 1e-9, "DESEASONALIZE result should be 1.0, got {}", as_f64(v));
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── Z_SCORE_ANOMALY / ZSCORE_OUTLIER ────────────────────────────────────────

#[test]
fn test_z_score_anomaly_basic() {
    // [1,1,1,100] with threshold=1.2: z-score of 100 is ~1.5 which exceeds 1.2
    let (_dir, _db, ex) = setup("ts_za", serde_json::json!({"nums": [1, 1, 1, 100]}));
    let mut p = Parser::new("QUERY ts_za COMPUTE res = Z_SCORE_ANOMALY(nums, 1.2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            // First three should be false, last should be true
            for v in &arr[..3] {
                assert_eq!(*v, Value::Bool(false), "non-outlier should be false");
            }
            assert_eq!(arr[3], Value::Bool(true), "outlier (100) should be true");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_zscore_outlier_alias() {
    // [1,1,1,1,1,1000]: z-score of 1000 is ~2.04 which exceeds 1.5
    let (_dir, _db, ex) = setup("ts_zo", serde_json::json!({"nums": [1, 1, 1, 1, 1, 1000]}));
    let mut p = Parser::new("QUERY ts_zo COMPUTE res = ZSCORE_OUTLIER(nums, 1.5) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6);
            assert_eq!(arr[5], Value::Bool(true), "last value (1000) should be anomaly");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── IQR_ANOMALY / FENCE_OUTLIER ─────────────────────────────────────────────

#[test]
fn test_iqr_anomaly_basic() {
    // [1,2,3,4,100] — 100 is an outlier by IQR
    let (_dir, _db, ex) = setup("ts_iq", serde_json::json!({"nums": [1, 2, 3, 4, 100]}));
    let mut p = Parser::new("QUERY ts_iq COMPUTE res = IQR_ANOMALY(nums, 1.5) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[4], Value::Bool(true), "100 should be IQR outlier");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_fence_outlier_alias() {
    let (_dir, _db, ex) = setup("ts_fo", serde_json::json!({"nums": [10, 11, 12, 13, 200]}));
    let mut p = Parser::new("QUERY ts_fo COMPUTE res = FENCE_OUTLIER(nums, 1.5) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[4], Value::Bool(true), "200 should be FENCE_OUTLIER");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── CUSUM_DETECT / CUMSUM_CHANGE ─────────────────────────────────────────────

#[test]
fn test_cusum_detect_basic() {
    // [1,1,1,1,10,10,10] — shift happens around index 4
    let (_dir, _db, ex) = setup("ts_cu", serde_json::json!({"nums": [1, 1, 1, 1, 10, 10, 10]}));
    let mut p = Parser::new("QUERY ts_cu COMPUTE res = CUSUM_DETECT(nums, 5.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Integer(idx)) => {
            assert!(*idx >= 4, "CUSUM should detect change at or after index 4, got {}", idx);
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_cumsum_change_alias() {
    let (_dir, _db, ex) = setup("ts_cc", serde_json::json!({"nums": [1, 1, 1, 1, 10, 10, 10]}));
    let mut p = Parser::new("QUERY ts_cc COMPUTE res = CUMSUM_CHANGE(nums, 5.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Integer(idx)) => {
            assert!(*idx >= 4, "CUMSUM_CHANGE alias should detect change at or after index 4");
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_cusum_detect_no_change() {
    // Constant series — no change detected
    let (_dir, _db, ex) = setup("ts_cu2", serde_json::json!({"nums": [5, 5, 5, 5, 5]}));
    let mut p = Parser::new("QUERY ts_cu2 COMPUTE res = CUSUM_DETECT(nums, 10.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Integer(idx)) => {
            assert_eq!(*idx, -1, "CUSUM on constant series should return -1");
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── BOLLINGER_BANDS / BB_BANDS ───────────────────────────────────────────────

#[test]
fn test_bollinger_bands_basic() {
    let nums: Vec<f64> = (1..=20).map(|x| x as f64).collect();
    let (_dir, _db, ex) = setup("ts_bb", serde_json::json!({"nums": nums}));
    let mut p = Parser::new("QUERY ts_bb COMPUTE res = BOLLINGER_BANDS(nums, 10, 2.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("upper"), "BB should have 'upper' key");
            assert!(obj.contains_key("middle"), "BB should have 'middle' key");
            assert!(obj.contains_key("lower"), "BB should have 'lower' key");
            let upper = as_f64(obj.get("upper").unwrap());
            let middle = as_f64(obj.get("middle").unwrap());
            let lower = as_f64(obj.get("lower").unwrap());
            assert!(upper > middle, "upper band > middle");
            assert!(middle > lower, "middle > lower band");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_bb_bands_alias() {
    let nums: Vec<f64> = (1..=15).map(|x| x as f64).collect();
    let (_dir, _db, ex) = setup("ts_bb2", serde_json::json!({"nums": nums}));
    let mut p = Parser::new("QUERY ts_bb2 COMPUTE res = BB_BANDS(nums, 5, 2.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("upper") && obj.contains_key("middle") && obj.contains_key("lower"),
                "BB_BANDS alias should return Object with upper/middle/lower");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GRUBBS_OUTLIER / GRUBBS_STAT ─────────────────────────────────────────────

#[test]
fn test_grubbs_outlier_basic() {
    // [1,2,3,100] — 100 is extreme, Grubbs stat should be > 1
    let (_dir, _db, ex) = setup("ts_gr", serde_json::json!({"nums": [1, 2, 3, 100]}));
    let mut p = Parser::new("QUERY ts_gr COMPUTE res = GRUBBS_OUTLIER(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(g)) => {
            assert!(*g > 1.0, "Grubbs stat for [1,2,3,100] should be > 1.0, got {}", g);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_grubbs_stat_alias() {
    let (_dir, _db, ex) = setup("ts_gs", serde_json::json!({"nums": [10, 11, 12, 500]}));
    let mut p = Parser::new("QUERY ts_gs COMPUTE res = GRUBBS_STAT(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(g)) => {
            assert!(*g > 1.0, "GRUBBS_STAT alias should return value > 1.0 for extreme outlier");
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── MODIFIED_Z_SCORE / MAD_Z_SCORE ───────────────────────────────────────────

#[test]
fn test_modified_z_score_basic() {
    // [1,2,3,100] — last value should have high modified z-score
    let (_dir, _db, ex) = setup("ts_mz", serde_json::json!({"nums": [1, 2, 3, 100]}));
    let mut p = Parser::new("QUERY ts_mz COMPUTE res = MODIFIED_Z_SCORE(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            let last_score = as_f64(&arr[3]).abs();
            let first_score = as_f64(&arr[0]).abs();
            assert!(last_score > first_score, "outlier (100) should have higher modified z-score than normal values");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_mad_z_score_alias() {
    let (_dir, _db, ex) = setup("ts_mad", serde_json::json!({"nums": [5, 6, 7, 8, 500]}));
    let mut p = Parser::new("QUERY ts_mad COMPUTE res = MAD_Z_SCORE(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            let last_score = as_f64(&arr[4]).abs();
            assert!(last_score > 5.0, "MAD_Z_SCORE of 500 in [5,6,7,8,500] should be large, got {}", last_score);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── PEAK_DETECTION / FIND_PEAKS ──────────────────────────────────────────────

#[test]
fn test_peak_detection_basic() {
    // [1,3,2,4,1] — peaks at indices 1 (value 3) and 3 (value 4)
    let (_dir, _db, ex) = setup("ts_pk", serde_json::json!({"nums": [1, 3, 2, 4, 1]}));
    let mut p = Parser::new("QUERY ts_pk COMPUTE res = PEAK_DETECTION(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "should find 2 peaks, got {:?}", arr);
            let indices: Vec<i64> = arr.iter().map(|v| match v { Value::Integer(i) => *i, _ => -1 }).collect();
            assert!(indices.contains(&1), "peak at index 1 not found");
            assert!(indices.contains(&3), "peak at index 3 not found");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_find_peaks_alias() {
    // [0,5,0,5,0] — peaks at indices 1 and 3
    let (_dir, _db, ex) = setup("ts_fp", serde_json::json!({"nums": [0, 5, 0, 5, 0]}));
    let mut p = Parser::new("QUERY ts_fp COMPUTE res = FIND_PEAKS(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "FIND_PEAKS should find 2 peaks");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── VALLEY_DETECTION / FIND_VALLEYS ─────────────────────────────────────────

#[test]
fn test_valley_detection_basic() {
    // [3,1,3,1,3] — valleys at indices 1 and 3
    let (_dir, _db, ex) = setup("ts_vl", serde_json::json!({"nums": [3, 1, 3, 1, 3]}));
    let mut p = Parser::new("QUERY ts_vl COMPUTE res = VALLEY_DETECTION(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "should find 2 valleys, got {:?}", arr);
            let indices: Vec<i64> = arr.iter().map(|v| match v { Value::Integer(i) => *i, _ => -1 }).collect();
            assert!(indices.contains(&1), "valley at index 1 not found");
            assert!(indices.contains(&3), "valley at index 3 not found");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_find_valleys_alias() {
    // [5,0,5,0,5] — valleys at indices 1 and 3
    let (_dir, _db, ex) = setup("ts_fv", serde_json::json!({"nums": [5, 0, 5, 0, 5]}));
    let mut p = Parser::new("QUERY ts_fv COMPUTE res = FIND_VALLEYS(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "FIND_VALLEYS should find 2 valleys");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── AUTOCORRELATION_LAG / ACF_LAG ────────────────────────────────────────────

#[test]
fn test_autocorrelation_lag_basic() {
    // [1,2,1,2,1] at lag=2 — period-2 series should have high positive ACF at lag 2
    let (_dir, _db, ex) = setup("ts_acf", serde_json::json!({"nums": [1, 2, 1, 2, 1]}));
    let mut p = Parser::new("QUERY ts_acf COMPUTE res = AUTOCORRELATION_LAG(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.5, "ACF at lag=2 for [1,2,1,2,1] should be > 0.5, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_acf_lag_alias() {
    let (_dir, _db, ex) = setup("ts_acf2", serde_json::json!({"nums": [1, 2, 1, 2, 1, 2]}));
    let mut p = Parser::new("QUERY ts_acf2 COMPUTE res = ACF_LAG(nums, 2) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.5, "ACF_LAG alias at lag=2 for alternating series should be high, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_autocorrelation_lag_zero() {
    // ACF at lag=0 should be 1.0
    let (_dir, _db, ex) = setup("ts_acf3", serde_json::json!({"nums": [3, 1, 4, 1, 5, 9, 2, 6]}));
    let mut p = Parser::new("QUERY ts_acf3 COMPUTE res = AUTOCORRELATION_LAG(nums, 0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => {
            assert!((*f - 1.0).abs() < 1e-9, "ACF at lag=0 should be 1.0, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Additional edge case tests ───────────────────────────────────────────────

#[test]
fn test_exponential_smoothing_constant_series() {
    // Smoothing a constant series should return the constant
    let (_dir, _db, ex) = setup("ts_ec", serde_json::json!({"nums": [7.0, 7.0, 7.0, 7.0]}));
    let mut p = Parser::new("QUERY ts_ec COMPUTE res = EXPONENTIAL_SMOOTHING(nums, 0.8) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            for v in arr {
                assert!((as_f64(v) - 7.0).abs() < 1e-9, "smoothing constant 7.0 should give 7.0, got {}", as_f64(v));
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_bollinger_bands_constant_series() {
    // Constant series: std=0, upper=middle=lower
    let nums: Vec<f64> = vec![5.0; 10];
    let (_dir, _db, ex) = setup("ts_bbc", serde_json::json!({"nums": nums}));
    let mut p = Parser::new("QUERY ts_bbc COMPUTE res = BOLLINGER_BANDS(nums, 5, 2.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let upper = as_f64(obj.get("upper").unwrap());
            let middle = as_f64(obj.get("middle").unwrap());
            let lower = as_f64(obj.get("lower").unwrap());
            assert!((upper - 5.0).abs() < 1e-9, "upper should be 5.0 for constant series");
            assert!((middle - 5.0).abs() < 1e-9, "middle should be 5.0");
            assert!((lower - 5.0).abs() < 1e-9, "lower should be 5.0");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_linear_forecast_zero_steps() {
    let (_dir, _db, ex) = setup("ts_lf0", serde_json::json!({"nums": [1, 2, 3]}));
    let mut p = Parser::new("QUERY ts_lf0 COMPUTE res = LINEAR_FORECAST(nums, 0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 0, "LINEAR_FORECAST with 0 steps should return empty array");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_z_score_anomaly_no_outliers() {
    // Uniform series: no anomalies
    let (_dir, _db, ex) = setup("ts_zno", serde_json::json!({"nums": [5, 5, 5, 5, 5]}));
    let mut p = Parser::new("QUERY ts_zno COMPUTE res = Z_SCORE_ANOMALY(nums, 2.0) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            for v in arr {
                assert_eq!(*v, Value::Bool(false), "uniform series should have no Z-score anomalies");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_peak_detection_no_peaks() {
    // Monotonically increasing: no interior peaks
    let (_dir, _db, ex) = setup("ts_np", serde_json::json!({"nums": [1, 2, 3, 4, 5]}));
    let mut p = Parser::new("QUERY ts_np COMPUTE res = PEAK_DETECTION(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 0, "monotone increasing series has no peaks");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_valley_detection_no_valleys() {
    // Monotonically decreasing: no interior valleys
    let (_dir, _db, ex) = setup("ts_nv", serde_json::json!({"nums": [5, 4, 3, 2, 1]}));
    let mut p = Parser::new("QUERY ts_nv COMPUTE res = VALLEY_DETECTION(nums) SELECT res;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 0, "monotone decreasing series has no valleys");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
