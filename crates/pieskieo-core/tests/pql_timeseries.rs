/// Integration tests for PQL time-series analysis and forecasting built-in functions.
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

// ── EXPONENTIAL_SMOOTHING ─────────────────────────────────────────────────────

#[test]
fn test_exponential_smoothing() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = EXPONENTIAL_SMOOTHING([1.0, 3.0, 5.0, 7.0, 9.0], 0.5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], Value::Float(1.0));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_exponential_smoothing_alias() {
    let (_d, _db, ex) = setup("ema_alias", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY ema_alias COMPUTE res = EMA_SMOOTH([2.0, 4.0, 6.0], 0.3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((as_f64(&arr[0]) - 2.0).abs() < 1e-9);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── DOUBLE_EXP_SMOOTH / HOLT_LINEAR ──────────────────────────────────────────

#[test]
fn test_double_exp_smooth_basic() {
    let (_d, _db, ex) = setup("des_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY des_ns COMPUTE res = DOUBLE_EXP_SMOOTH([1.0, 2.0, 3.0, 4.0, 5.0], 0.4, 0.3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("level"), "should have level");
            assert!(m.contains_key("trend"), "should have trend");
            assert!(m.contains_key("smoothed"), "should have smoothed");
            match m.get("smoothed") {
                Some(Value::Array(arr)) => assert_eq!(arr.len(), 5),
                other => panic!("smoothed should be array, got {:?}", other),
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_holt_linear_alias() {
    let (_d, _db, ex) = setup("hl_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY hl_ns COMPUTE res = HOLT_LINEAR([10.0, 20.0, 30.0], 0.5, 0.2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("level"));
            // trend should be positive for increasing series
            let trend = as_f64(m.get("trend").unwrap());
            assert!(trend > 0.0, "trend should be positive, got {}", trend);
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── TRIPLE_EXP_SMOOTH / HOLT_WINTERS ─────────────────────────────────────────

#[test]
fn test_triple_exp_smooth_basic() {
    let (_d, _db, ex) = setup("tes_ns", serde_json::json!({"dummy": 1}));
    // Need at least 2 periods of data, period=3 -> need 6+ values
    let mut p = Parser::new(r#"QUERY tes_ns COMPUTE res = TRIPLE_EXP_SMOOTH([1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0], 0.3, 0.1, 0.2, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 9, "should return same length as input");
            for v in arr { assert!(matches!(v, Value::Float(_)), "should be floats"); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_holt_winters_alias() {
    let (_d, _db, ex) = setup("hw_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY hw_ns COMPUTE res = HOLT_WINTERS([1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0], 0.3, 0.1, 0.2, 4) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 8),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── WEIGHTED_MOVING_AVG / WMA ─────────────────────────────────────────────────

#[test]
fn test_weighted_moving_avg_basic() {
    let (_d, _db, ex) = setup("wma_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY wma_ns COMPUTE res = WEIGHTED_MOVING_AVG([1.0, 2.0, 3.0, 4.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            // First element should be Null (window not full yet for w=2 from index 0)
            // index 0: t+1=1 < w=2 -> Null
            assert!(matches!(arr[0], Value::Null), "first WMA should be Null, got {:?}", arr[0]);
            // index 1: t+1=2 >= w=2 -> WMA = (1*1 + 2*2) / 3 = 5/3
            let expected = (1.0 * 1.0 + 2.0 * 2.0) / 3.0;
            assert!((as_f64(&arr[1]) - expected).abs() < 1e-9, "expected {}, got {}", expected, as_f64(&arr[1]));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_wma_alias() {
    let (_d, _db, ex) = setup("wma2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY wma2_ns COMPUTE res = WMA([2.0, 4.0, 6.0, 8.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            // index 2: WMA = (1*2 + 2*4 + 3*6) / 6 = (2+8+18)/6 = 28/6
            let expected = (1.0 * 2.0 + 2.0 * 4.0 + 3.0 * 6.0) / 6.0;
            assert!((as_f64(&arr[2]) - expected).abs() < 1e-9);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── CUMULATIVE_MOVING_AVG / CMA ───────────────────────────────────────────────

#[test]
fn test_cumulative_moving_avg_basic() {
    let (_d, _db, ex) = setup("cma_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY cma_ns COMPUTE res = CUMULATIVE_MOVING_AVG([2.0, 4.0, 6.0, 8.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!((as_f64(&arr[0]) - 2.0).abs() < 1e-9, "CMA[0]=2.0, got {}", as_f64(&arr[0]));
            assert!((as_f64(&arr[1]) - 3.0).abs() < 1e-9, "CMA[1]=3.0, got {}", as_f64(&arr[1]));
            assert!((as_f64(&arr[2]) - 4.0).abs() < 1e-9, "CMA[2]=4.0, got {}", as_f64(&arr[2]));
            assert!((as_f64(&arr[3]) - 5.0).abs() < 1e-9, "CMA[3]=5.0, got {}", as_f64(&arr[3]));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_cma_alias() {
    let (_d, _db, ex) = setup("cma2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY cma2_ns COMPUTE res = CMA([10.0, 20.0, 30.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((as_f64(&arr[2]) - 20.0).abs() < 1e-9, "CMA[2]=(10+20+30)/3=20, got {}", as_f64(&arr[2]));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── TREND_COMPONENT / EXTRACT_TREND ──────────────────────────────────────────

#[test]
fn test_trend_component_basic() {
    let (_d, _db, ex) = setup("tc_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY tc_ns COMPUTE res = TREND_COMPONENT([1.0, 2.0, 3.0, 4.0, 5.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "should return same length as input");
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_extract_trend_alias() {
    let (_d, _db, ex) = setup("et_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY et_ns COMPUTE res = EXTRACT_TREND([10.0, 12.0, 14.0, 16.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 4),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── SEASONAL_INDICES / SEASON_IDX ────────────────────────────────────────────

#[test]
fn test_seasonal_indices_basic() {
    let (_d, _db, ex) = setup("si_ns", serde_json::json!({"dummy": 1}));
    // Simple periodic data with period 4
    let mut p = Parser::new(r#"QUERY si_ns COMPUTE res = SEASONAL_INDICES([1.0,2.0,3.0,4.0,1.0,2.0,3.0,4.0], 4) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4, "should return indices for each period position");
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_season_idx_alias() {
    let (_d, _db, ex) = setup("si2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY si2_ns COMPUTE res = SEASON_IDX([1.0,2.0,1.0,2.0,1.0,2.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── REMOVE_SEASONAL ──────────────────────────────────────────────────────────

#[test]
fn test_remove_seasonal_basic() {
    let (_d, _db, ex) = setup("rs_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY rs_ns COMPUTE res = REMOVE_SEASONAL([2.0, 4.0, 2.0, 4.0, 2.0, 4.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6, "should return same length");
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── STL_SIMPLE / ADDITIVE_DECOMP ─────────────────────────────────────────────

#[test]
fn test_stl_simple_basic() {
    let (_d, _db, ex) = setup("stl_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY stl_ns COMPUTE res = STL_SIMPLE([1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("trend"), "should have trend");
            assert!(m.contains_key("seasonal"), "should have seasonal");
            assert!(m.contains_key("residual"), "should have residual");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_additive_decomp_alias() {
    let (_d, _db, ex) = setup("ad_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY ad_ns COMPUTE res = ADDITIVE_DECOMP([10.0,11.0,12.0,10.0,11.0,12.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("trend"));
            assert!(m.contains_key("seasonal"));
            assert!(m.contains_key("residual"));
            // Each component should be an array of the right length
            if let Some(Value::Array(trend)) = m.get("trend") {
                assert_eq!(trend.len(), 6);
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── HP_FILTER / HODRICK_PRESCOTT ──────────────────────────────────────────────

#[test]
fn test_hp_filter_basic() {
    let (_d, _db, ex) = setup("hp_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY hp_ns COMPUTE res = HP_FILTER([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 100.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("trend"));
            assert!(m.contains_key("cycle"));
            if let Some(Value::Array(trend)) = m.get("trend") {
                assert_eq!(trend.len(), 6);
            }
            if let Some(Value::Array(cycle)) = m.get("cycle") {
                assert_eq!(cycle.len(), 6);
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_hodrick_prescott_alias() {
    let (_d, _db, ex) = setup("hp2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY hp2_ns COMPUTE res = HODRICK_PRESCOTT([5.0, 6.0, 7.0, 8.0, 9.0], 1600.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("trend"));
            assert!(m.contains_key("cycle"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── ACF_ARRAY / AUTOCORR_ARRAY ────────────────────────────────────────────────

#[test]
fn test_acf_array_basic() {
    let (_d, _db, ex) = setup("acf_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY acf_ns COMPUTE res = ACF_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            // lags 0..3 -> 4 values
            assert_eq!(arr.len(), 4, "should return max_lag+1 values");
            // ACF at lag 0 should be 1.0
            assert!((as_f64(&arr[0]) - 1.0).abs() < 1e-9, "ACF[0] should be 1.0");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_autocorr_array_alias() {
    let (_d, _db, ex) = setup("acf2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY acf2_ns COMPUTE res = AUTOCORR_ARRAY([1.0, 2.0, 3.0, 2.0, 1.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((as_f64(&arr[0]) - 1.0).abs() < 1e-9);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── PARTIAL_AUTOCORRELATION / PACF ────────────────────────────────────────────

#[test]
fn test_partial_autocorrelation_basic() {
    let (_d, _db, ex) = setup("pacf_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY pacf_ns COMPUTE res = PARTIAL_AUTOCORRELATION([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4, "PACF should return max_lag+1 values");
            // PACF at lag 0 should be 1.0
            assert!((as_f64(&arr[0]) - 1.0).abs() < 1e-9, "PACF[0] should be 1.0");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_pacf_alias() {
    let (_d, _db, ex) = setup("pacf2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY pacf2_ns COMPUTE res = PACF([2.0, 4.0, 3.0, 5.0, 4.0, 6.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── LJUNG_BOX_TEST / LJUNG_BOX ───────────────────────────────────────────────

#[test]
fn test_ljung_box_test_basic() {
    let (_d, _db, ex) = setup("lb_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY lb_ns COMPUTE res = LJUNG_BOX_TEST([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("q_stat"), "should have q_stat");
            assert!(m.contains_key("p_value_approx"), "should have p_value_approx");
            let q = as_f64(m.get("q_stat").unwrap());
            assert!(q >= 0.0, "Q statistic should be non-negative");
            let p = as_f64(m.get("p_value_approx").unwrap());
            assert!(p >= 0.0 && p <= 1.0, "p-value should be in [0,1], got {}", p);
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_ljung_box_alias() {
    let (_d, _db, ex) = setup("lb2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY lb2_ns COMPUTE res = LJUNG_BOX([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("q_stat"));
            assert!(m.contains_key("p_value_approx"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── DURBIN_WATSON / DW_STAT ───────────────────────────────────────────────────

#[test]
fn test_durbin_watson_no_autocorr() {
    let (_d, _db, ex) = setup("dw_ns", serde_json::json!({"dummy": 1}));
    // Alternating residuals have DW close to 4
    let mut p = Parser::new(r#"QUERY dw_ns COMPUTE res = DURBIN_WATSON([1.0, -1.0, 1.0, -1.0, 1.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(dw)) => {
            assert!(*dw > 3.0, "alternating residuals should give DW > 3, got {}", dw);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_durbin_watson_positive_autocorr() {
    let (_d, _db, ex) = setup("dw2_ns", serde_json::json!({"dummy": 1}));
    // All same sign residuals -> DW near 0
    let mut p = Parser::new(r#"QUERY dw2_ns COMPUTE res = DW_STAT([1.0, 1.1, 1.0, 1.1, 1.0, 1.1]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(dw)) => {
            assert!(*dw < 1.0, "similar residuals should give DW < 1, got {}", dw);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── ADF_TEST / AUGMENTED_DICKEY_FULLER ───────────────────────────────────────

#[test]
fn test_adf_test_basic() {
    let (_d, _db, ex) = setup("adf_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY adf_ns COMPUTE res = ADF_TEST([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("adf_stat"), "should have adf_stat");
            assert!(m.contains_key("is_stationary_approx"), "should have is_stationary_approx");
            // A pure trend is not stationary
            match m.get("is_stationary_approx") {
                Some(Value::Bool(b)) => assert!(!b, "pure trend should not be stationary"),
                other => panic!("expected bool, got {:?}", other),
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_augmented_dickey_fuller_alias() {
    let (_d, _db, ex) = setup("adf2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY adf2_ns COMPUTE res = AUGMENTED_DICKEY_FULLER([1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("adf_stat"));
            assert!(m.contains_key("is_stationary_approx"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── IS_STATIONARY / STATIONARITY_CHECK ───────────────────────────────────────

#[test]
fn test_is_stationary_constant() {
    let (_d, _db, ex) = setup("stat_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY stat_ns COMPUTE res = IS_STATIONARY([5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.1]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Bool(b)) => assert!(*b, "near-constant series should be stationary"),
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_stationarity_check_alias() {
    let (_d, _db, ex) = setup("stat2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY stat2_ns COMPUTE res = STATIONARITY_CHECK([3.0, 3.1, 2.9, 3.0, 3.1, 2.9]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Bool(_)) => { /* result is bool, that's fine */ }
        other => panic!("expected bool, got {:?}", other),
    }
}

// ── NAIVE_PREDICT ─────────────────────────────────────────────────────────────

#[test]
fn test_naive_predict_basic() {
    let (_d, _db, ex) = setup("np_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY np_ns COMPUTE res = NAIVE_PREDICT([1.0, 2.0, 3.0, 4.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "should return 3 forecast values");
            // All values should be the last value (4.0)
            for v in arr {
                assert!((as_f64(v) - 4.0).abs() < 1e-9, "naive forecast should be last value 4.0, got {}", as_f64(v));
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── AR_FORECAST / AR_PREDICT ──────────────────────────────────────────────────

#[test]
fn test_ar_forecast_basic() {
    let (_d, _db, ex) = setup("ar_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY ar_ns COMPUTE res = AR_FORECAST([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "AR(2) should return 3 forecasts");
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_ar_predict_alias() {
    let (_d, _db, ex) = setup("ar2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY ar2_ns COMPUTE res = AR_PREDICT([1.0, 2.0, 3.0, 4.0, 5.0], 1, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── MOVING_AVG_FORECAST / MA_FORECAST ────────────────────────────────────────

#[test]
fn test_moving_avg_forecast_basic() {
    let (_d, _db, ex) = setup("maf_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY maf_ns COMPUTE res = MOVING_AVG_FORECAST([2.0, 4.0, 6.0, 8.0], 2, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "should return 3 forecasts");
            // Mean of last 2 values (6.0, 8.0) = 7.0
            for v in arr {
                assert!((as_f64(v) - 7.0).abs() < 1e-9, "forecast should be mean of last 2, got {}", as_f64(v));
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_ma_forecast_alias() {
    let (_d, _db, ex) = setup("maf2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY maf2_ns COMPUTE res = MA_FORECAST([1.0, 3.0, 5.0, 7.0], 3, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── EXPONENTIAL_FORECAST / EXP_FORECAST ──────────────────────────────────────

#[test]
fn test_exponential_forecast_basic() {
    let (_d, _db, ex) = setup("ef_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY ef_ns COMPUTE res = EXPONENTIAL_FORECAST([1.0, 2.0, 3.0, 4.0, 5.0], 0.5, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "should return 3 forecasts");
            // All should be the same SES value
            let v0 = as_f64(&arr[0]);
            for v in arr {
                assert!((as_f64(v) - v0).abs() < 1e-9, "all ETS(A,N,N) forecasts should be equal");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_exp_forecast_alias() {
    let (_d, _db, ex) = setup("ef2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY ef2_ns COMPUTE res = EXP_FORECAST([10.0, 12.0, 14.0], 0.3, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── HOLT_FORECAST / HOLT_PREDICT ──────────────────────────────────────────────

#[test]
fn test_holt_forecast_basic() {
    let (_d, _db, ex) = setup("hf_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY hf_ns COMPUTE res = HOLT_FORECAST([1.0, 2.0, 3.0, 4.0, 5.0], 0.4, 0.3, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "should return 3 forecasts");
            // For an increasing series, forecasts should be increasing
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_holt_predict_alias() {
    let (_d, _db, ex) = setup("hp3_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY hp3_ns COMPUTE res = HOLT_PREDICT([2.0, 4.0, 6.0, 8.0], 0.5, 0.2, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            let v1 = as_f64(&arr[0]);
            let v2 = as_f64(&arr[1]);
            assert!(v2 > v1, "Holt forecast on increasing series should increase: {} -> {}", v1, v2);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── SEASONAL_NAIVE_FORECAST / SNAIVE ─────────────────────────────────────────

#[test]
fn test_seasonal_naive_forecast_basic() {
    let (_d, _db, ex) = setup("snf_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY snf_ns COMPUTE res = SEASONAL_NAIVE_FORECAST([1.0, 2.0, 3.0, 4.0], 4, 8) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 8, "should return 8 forecasts");
            // Should repeat 1,2,3,4 twice
            assert!((as_f64(&arr[0]) - 1.0).abs() < 1e-9);
            assert!((as_f64(&arr[1]) - 2.0).abs() < 1e-9);
            assert!((as_f64(&arr[2]) - 3.0).abs() < 1e-9);
            assert!((as_f64(&arr[3]) - 4.0).abs() < 1e-9);
            assert!((as_f64(&arr[4]) - 1.0).abs() < 1e-9);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_snaive_alias() {
    let (_d, _db, ex) = setup("snf2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY snf2_ns COMPUTE res = SNAIVE([10.0, 20.0, 30.0], 3, 6) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6);
            assert!((as_f64(&arr[0]) - 10.0).abs() < 1e-9);
            assert!((as_f64(&arr[3]) - 10.0).abs() < 1e-9);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── THETA_FORECAST / THETA_METHOD ────────────────────────────────────────────

#[test]
fn test_theta_forecast_basic() {
    let (_d, _db, ex) = setup("theta_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY theta_ns COMPUTE res = THETA_FORECAST([1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "should return 3 forecasts");
            for v in arr { assert!(matches!(v, Value::Float(_))); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_theta_method_alias() {
    let (_d, _db, ex) = setup("theta2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY theta2_ns COMPUTE res = THETA_METHOD([2.0, 4.0, 6.0, 8.0], 1.5, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── MAPE_SCORE / MEAN_ABSOLUTE_PCT_ERROR ──────────────────────────────────────

#[test]
fn test_mape_score_basic() {
    let (_d, _db, ex) = setup("mape_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY mape_ns COMPUTE res = MAPE_SCORE([100.0, 200.0, 300.0], [110.0, 190.0, 330.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => {
            // MAPE = 100/3 * (10/100 + 10/200 + 30/300) = 100/3 * (0.1+0.05+0.1) = 100/3 * 0.25 ≈ 8.33
            assert!(*v > 0.0, "MAPE should be positive, got {}", v);
            assert!(*v < 100.0, "MAPE should be < 100%, got {}", v);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_mean_absolute_pct_error_alias() {
    let (_d, _db, ex) = setup("mape2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY mape2_ns COMPUTE res = MEAN_ABSOLUTE_PCT_ERROR([10.0, 20.0], [11.0, 22.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => assert!(*v > 0.0),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── SMAPE_SCORE / SYMMETRIC_MAPE ─────────────────────────────────────────────

#[test]
fn test_smape_score_basic() {
    let (_d, _db, ex) = setup("smape_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY smape_ns COMPUTE res = SMAPE_SCORE([100.0, 200.0], [110.0, 180.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => {
            assert!(*v >= 0.0 && *v <= 200.0, "sMAPE in [0,200], got {}", v);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_symmetric_mape_alias() {
    let (_d, _db, ex) = setup("smape2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY smape2_ns COMPUTE res = SYMMETRIC_MAPE([50.0, 60.0, 70.0], [55.0, 57.0, 72.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => assert!(*v >= 0.0),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_smape_perfect_forecast() {
    let (_d, _db, ex) = setup("smape3_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY smape3_ns COMPUTE res = SMAPE_SCORE([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => assert!(v.abs() < 1e-9, "perfect forecast -> sMAPE = 0, got {}", v),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── RMSE_SCORE ────────────────────────────────────────────────────────────────

#[test]
fn test_rmse_score_basic() {
    let (_d, _db, ex) = setup("rmse_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY rmse_ns COMPUTE res = RMSE_SCORE([1.0, 2.0, 3.0], [1.5, 2.5, 3.5]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => {
            // Each error is 0.5, MSE = 0.25, RMSE = 0.5
            assert!((*v - 0.5).abs() < 1e-9, "RMSE should be 0.5, got {}", v);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── FORECAST_BIAS / MEAN_BIAS ─────────────────────────────────────────────────

#[test]
fn test_forecast_bias_positive() {
    let (_d, _db, ex) = setup("fb_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY fb_ns COMPUTE res = FORECAST_BIAS([10.0, 20.0, 30.0], [12.0, 22.0, 32.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => {
            // bias = mean(pred - actual) = mean(2, 2, 2) = 2.0
            assert!((*v - 2.0).abs() < 1e-9, "bias should be 2.0, got {}", v);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_mean_bias_alias() {
    let (_d, _db, ex) = setup("fb2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY fb2_ns COMPUTE res = MEAN_BIAS([5.0, 10.0], [3.0, 8.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => {
            // bias = mean(-2, -2) = -2.0
            assert!((*v - (-2.0)).abs() < 1e-9, "bias should be -2.0, got {}", v);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_forecast_bias_zero() {
    let (_d, _db, ex) = setup("fb3_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY fb3_ns COMPUTE res = FORECAST_BIAS([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => assert!(v.abs() < 1e-9, "unbiased -> 0, got {}", v),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── THEIL_U / THEIL_U_STAT ────────────────────────────────────────────────────

#[test]
fn test_theil_u_perfect_forecast() {
    let (_d, _db, ex) = setup("thu_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY thu_ns COMPUTE res = THEIL_U([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => {
            assert!(v.abs() < 1e-9, "perfect forecast -> Theil U = 0, got {}", v);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_theil_u_stat_alias() {
    let (_d, _db, ex) = setup("thu2_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY thu2_ns COMPUTE res = THEIL_U_STAT([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(v)) => assert!(*v >= 0.0, "Theil U should be non-negative, got {}", v),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Inline array literals in COMPUTE ─────────────────────────────────────────

#[test]
fn test_inline_array_functions() {
    let (_d, _db, ex) = setup("inline_ns", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY inline_ns COMPUTE cma = CMA([10.0, 20.0, 30.0]), wma = WMA([1.0, 2.0, 3.0, 4.0], 2) SELECT cma, wma;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(r.rows[0].data.contains_key("cma"), "should have cma");
    assert!(r.rows[0].data.contains_key("wma"), "should have wma");
}

#[test]
fn test_multiple_forecasts_in_one_query() {
    let (_d, _db, ex) = setup("multi_fc", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY multi_fc COMPUTE naive = NAIVE_PREDICT([1.0,2.0,3.0,4.0,5.0], 2), exp_f = EXP_FORECAST([1.0,2.0,3.0,4.0,5.0], 0.4, 2) SELECT naive, exp_f;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(matches!(r.rows[0].data.get("naive"), Some(Value::Array(_))));
    assert!(matches!(r.rows[0].data.get("exp_f"), Some(Value::Array(_))));
}
