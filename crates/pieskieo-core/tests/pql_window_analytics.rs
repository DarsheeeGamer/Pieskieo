/// Integration tests for PQL window/analytics aggregate functions (array-input).
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    (dir, db, ex)
}

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

// ── PCTILE_RANK ────────────────────────────────────────────────────────────────

#[test]
fn test_pctile_rank_basic() {
    let (_dir, _db, ex) = setup();
    // PCTILE_RANK([1,2,3,4,5], 3) = count(x<=3)/5 = 3/5 = 0.6
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PCTILE_RANK([1.0, 2.0, 3.0, 4.0, 5.0], 3.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 0.6).abs() < 0.001, "expected 0.6, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_pctile_rank_max_value() {
    let (_dir, _db, ex) = setup();
    // PCTILE_RANK([1,2,3,4,5], 5) = 5/5 = 1.0
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PCTILE_RANK([1.0, 2.0, 3.0, 4.0, 5.0], 5.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001, "expected 1.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_pctile_rank_min_value() {
    let (_dir, _db, ex) = setup();
    // PCTILE_RANK([10,20,30], 5) = 0/3 = 0.0
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PCTILE_RANK([10.0, 20.0, 30.0], 5.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 0.001, "expected 0.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── NTILE_ARRAY / NTILE ─────────────────────────────────────────────────────

#[test]
fn test_ntile_array_basic() {
    let (_dir, _db, ex) = setup();
    // NTILE([1,2,3,4,5,6], 3) -> 6 elements into 3 buckets: [1,1,2,2,3,3]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTILE_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6, "expected 6 elements");
            // Buckets should be 1-3
            for v in arr {
                match v {
                    Value::Integer(i) => assert!(*i >= 1 && *i <= 3, "bucket {} out of range 1-3", i),
                    other => panic!("expected Integer, got {:?}", other),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_ntile_alias() {
    let (_dir, _db, ex) = setup();
    // NTILE is an alias for NTILE_ARRAY
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTILE([1.0, 2.0, 3.0, 4.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            // Two buckets; first 2 elements in bucket 1, last 2 in bucket 2
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[3], Value::Integer(2));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── PERCENT_RANK_ARRAY / PCT_RANK_ARR ─────────────────────────────────────────

#[test]
fn test_percent_rank_array_basic() {
    let (_dir, _db, ex) = setup();
    // PCT_RANK_ARR([1,2,3,4,5]) -> [0.0, 0.25, 0.5, 0.75, 1.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PERCENT_RANK_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 0.0).abs() < 0.001, "first should be 0.0");
            assert!((to_f64(&arr[4]) - 1.0).abs() < 0.001, "last should be 1.0");
            assert!((to_f64(&arr[2]) - 0.5).abs() < 0.001, "middle should be 0.5");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_pct_rank_arr_alias() {
    let (_dir, _db, ex) = setup();
    // PCT_RANK_ARR is the short alias
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PCT_RANK_ARR([1.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert!((to_f64(&arr[0]) - 0.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 1.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_percent_rank_array_single() {
    let (_dir, _db, ex) = setup();
    // Single-element array -> [0.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PCT_RANK_ARR([42.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            assert!((to_f64(&arr[0]) - 0.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── CUME_DIST_ARRAY / CUME_DIST ────────────────────────────────────────────────

#[test]
fn test_cume_dist_array_basic() {
    let (_dir, _db, ex) = setup();
    // CUME_DIST([1,2,3,4,5]) -> [0.2, 0.4, 0.6, 0.8, 1.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CUME_DIST_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 0.2).abs() < 0.001, "first should be 0.2, got {}", to_f64(&arr[0]));
            assert!((to_f64(&arr[4]) - 1.0).abs() < 0.001, "last should be 1.0, got {}", to_f64(&arr[4]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_cume_dist_alias() {
    let (_dir, _db, ex) = setup();
    // CUME_DIST is the short alias
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CUME_DIST([1.0, 1.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            // Both 1.0 values have cume_dist = 2/3 ≈ 0.667
            assert!((to_f64(&arr[0]) - 2.0 / 3.0).abs() < 0.01);
            assert!((to_f64(&arr[1]) - 2.0 / 3.0).abs() < 0.01);
            // 2.0 has cume_dist = 3/3 = 1.0
            assert!((to_f64(&arr[2]) - 1.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── DENSE_RANK_ARRAY / DENSE_RANK ──────────────────────────────────────────────

#[test]
fn test_dense_rank_array_basic() {
    let (_dir, _db, ex) = setup();
    // DENSE_RANK([10, 30, 20, 30, 10]) -> [1, 3, 2, 3, 1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DENSE_RANK_ARRAY([10.0, 30.0, 20.0, 30.0, 10.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(3));
            assert_eq!(arr[2], Value::Integer(2));
            assert_eq!(arr[3], Value::Integer(3));
            assert_eq!(arr[4], Value::Integer(1));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_dense_rank_alias() {
    let (_dir, _db, ex) = setup();
    // DENSE_RANK is the alias - no gaps
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DENSE_RANK([1.0, 1.0, 2.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(1));
            assert_eq!(arr[2], Value::Integer(2));
            assert_eq!(arr[3], Value::Integer(3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RANK_ARR ──────────────────────────────────────────────────────────────────

#[test]
fn test_rank_arr_with_gaps() {
    let (_dir, _db, ex) = setup();
    // RANK_ARR([10, 20, 20, 30]) -> [1, 2, 2, 4] (gaps for ties)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RANK_ARR([10.0, 20.0, 20.0, 30.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(2));
            assert_eq!(arr[2], Value::Integer(2));
            assert_eq!(arr[3], Value::Integer(4));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_arr_rank_alias() {
    let (_dir, _db, ex) = setup();
    // ARR_RANK is the alias
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ARR_RANK([5.0, 5.0, 10.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(1));
            assert_eq!(arr[2], Value::Integer(3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── LAG_ARRAY / LAG_ARR ───────────────────────────────────────────────────────

#[test]
fn test_lag_array_basic() {
    let (_dir, _db, ex) = setup();
    // LAG_ARRAY([1,2,3,4,5], 1) -> [null, 1, 2, 3, 4]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LAG_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], Value::Null, "first element should be Null");
            assert!((to_f64(&arr[1]) - 1.0).abs() < 0.001, "arr[1] should be 1.0");
            assert!((to_f64(&arr[4]) - 4.0).abs() < 0.001, "arr[4] should be 4.0");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lag_array_offset2() {
    let (_dir, _db, ex) = setup();
    // LAG_ARRAY([10,20,30,40,50], 2) -> [null, null, 10, 20, 30]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LAG_ARRAY([10.0, 20.0, 30.0, 40.0, 50.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], Value::Null);
            assert_eq!(arr[1], Value::Null);
            assert!((to_f64(&arr[2]) - 10.0).abs() < 0.001);
            assert!((to_f64(&arr[4]) - 30.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lag_arr_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LAG_ARR([100.0, 200.0, 300.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Null);
            assert!((to_f64(&arr[1]) - 100.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── LEAD_ARRAY / LEAD_ARR ─────────────────────────────────────────────────────

#[test]
fn test_lead_array_basic() {
    let (_dir, _db, ex) = setup();
    // LEAD_ARRAY([1,2,3,4,5], 1) -> [2, 3, 4, 5, null]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LEAD_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 2.0).abs() < 0.001, "arr[0] should be 2.0");
            assert!((to_f64(&arr[3]) - 5.0).abs() < 0.001, "arr[3] should be 5.0");
            assert_eq!(arr[4], Value::Null, "last element should be Null");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lead_array_offset2() {
    let (_dir, _db, ex) = setup();
    // LEAD_ARRAY([10,20,30,40,50], 2) -> [30, 40, 50, null, null]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LEAD_ARRAY([10.0, 20.0, 30.0, 40.0, 50.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 30.0).abs() < 0.001);
            assert_eq!(arr[3], Value::Null);
            assert_eq!(arr[4], Value::Null);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lead_arr_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LEAD_ARR([1.0, 2.0, 3.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((to_f64(&arr[0]) - 2.0).abs() < 0.001);
            assert_eq!(arr[2], Value::Null);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── FIRST_VALUE_ARRAY / FIRST_VAL ─────────────────────────────────────────────

#[test]
fn test_first_val_basic() {
    let (_dir, _db, ex) = setup();
    // FIRST_VAL([5,3,8,1]) -> [5,5,5,5]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIRST_VALUE_ARRAY([5.0, 3.0, 8.0, 1.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            for v in arr {
                assert!((to_f64(v) - 5.0).abs() < 0.001, "all should be 5.0, got {}", to_f64(v));
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_first_val_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIRST_VAL([10.0, 20.0, 30.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((to_f64(&arr[2]) - 10.0).abs() < 0.001, "all should be 10.0");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── LAST_VALUE_ARRAY / LAST_VAL ───────────────────────────────────────────────

#[test]
fn test_last_val_basic() {
    let (_dir, _db, ex) = setup();
    // LAST_VAL([5,3,8,1]) -> [1,1,1,1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LAST_VALUE_ARRAY([5.0, 3.0, 8.0, 1.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            for v in arr {
                assert!((to_f64(v) - 1.0).abs() < 0.001, "all should be 1.0, got {}", to_f64(v));
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_last_val_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LAST_VAL([7.0, 8.0, 9.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert!((to_f64(v) - 9.0).abs() < 0.001, "all should be 9.0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── NTH_VALUE_ARRAY / NTH_VAL ─────────────────────────────────────────────────

#[test]
fn test_nth_val_basic() {
    let (_dir, _db, ex) = setup();
    // NTH_VAL([10, 20, 30, 40], 2) -> [20, 20, 20, 20]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTH_VALUE_ARRAY([10.0, 20.0, 30.0, 40.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            for v in arr {
                assert!((to_f64(v) - 20.0).abs() < 0.001, "all should be 20.0, got {}", to_f64(v));
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_nth_val_alias() {
    let (_dir, _db, ex) = setup();
    // NTH_VAL([5, 10, 15], 3) -> [15, 15, 15]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTH_VAL([5.0, 10.0, 15.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert!((to_f64(v) - 15.0).abs() < 0.001, "all should be 15.0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_nth_val_out_of_range() {
    let (_dir, _db, ex) = setup();
    // NTH_VAL([1, 2, 3], 10) -> [Null, Null, Null]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTH_VAL([1.0, 2.0, 3.0], 10) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert_eq!(*v, Value::Null, "all should be Null for out-of-range n");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ROW_NUMBER_ARRAY / ROW_NUM_ARR ────────────────────────────────────────────

#[test]
fn test_row_number_array_basic() {
    let (_dir, _db, ex) = setup();
    // ROW_NUM_ARR([a,b,c,d,e]) -> [1, 2, 3, 4, 5]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROW_NUMBER_ARRAY([10.0, 20.0, 30.0, 40.0, 50.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            for (i, v) in arr.iter().enumerate() {
                assert_eq!(*v, Value::Integer((i + 1) as i64), "row number at {} should be {}", i, i + 1);
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_row_num_arr_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROW_NUM_ARR([5.0, 5.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(2));
            assert_eq!(arr[2], Value::Integer(3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RUN_MIN ────────────────────────────────────────────────────────────────────

#[test]
fn test_run_min_basic() {
    let (_dir, _db, ex) = setup();
    // RUN_MIN([5,3,8,1,4]) -> [5,3,3,1,1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_MIN([5.0, 3.0, 8.0, 1.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 5.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 1.0).abs() < 0.001);
            assert!((to_f64(&arr[4]) - 1.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RUN_MAX ────────────────────────────────────────────────────────────────────

#[test]
fn test_run_max_basic() {
    let (_dir, _db, ex) = setup();
    // RUN_MAX([5,3,8,1,4]) -> [5,5,8,8,8]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_MAX([5.0, 3.0, 8.0, 1.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 5.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 5.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 8.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 8.0).abs() < 0.001);
            assert!((to_f64(&arr[4]) - 8.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RUN_AVG ────────────────────────────────────────────────────────────────────

#[test]
fn test_run_avg_basic() {
    let (_dir, _db, ex) = setup();
    // RUN_AVG([2,4,6,8]) -> [2.0, 3.0, 4.0, 5.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_AVG([2.0, 4.0, 6.0, 8.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!((to_f64(&arr[0]) - 2.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 4.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 5.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RUNNING_SUM / RUN_SUM ─────────────────────────────────────────────────────

#[test]
fn test_running_sum_basic() {
    let (_dir, _db, ex) = setup();
    // RUNNING_SUM([1,2,3,4]) -> [1,3,6,10]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUNNING_SUM([1.0, 2.0, 3.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!((to_f64(&arr[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 6.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 10.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_run_sum_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_SUM([5.0, 5.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((to_f64(&arr[2]) - 15.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RUNNING_COUNT / RUN_COUNT ─────────────────────────────────────────────────

#[test]
fn test_running_count_basic() {
    let (_dir, _db, ex) = setup();
    // RUNNING_COUNT([a,b,c,d]) -> [1,2,3,4]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUNNING_COUNT([1.0, 2.0, 3.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Value::Integer(1));
            assert_eq!(arr[1], Value::Integer(2));
            assert_eq!(arr[2], Value::Integer(3));
            assert_eq!(arr[3], Value::Integer(4));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_run_count_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_COUNT([10.0, 20.0, 30.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[2], Value::Integer(3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ROLLING_WINDOW_AVG / ROLLING_AVG ──────────────────────────────────────────

#[test]
fn test_rolling_avg_basic() {
    let (_dir, _db, ex) = setup();
    // ROLLING_AVG([1,2,3,4,5], 3) -> [1.0, 1.5, 2.0, 3.0, 4.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_WINDOW_AVG([1.0, 2.0, 3.0, 4.0, 5.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 1.0).abs() < 0.001, "arr[0] should be 1.0");
            assert!((to_f64(&arr[1]) - 1.5).abs() < 0.001, "arr[1] should be 1.5");
            assert!((to_f64(&arr[2]) - 2.0).abs() < 0.001, "arr[2] should be 2.0");
            assert!((to_f64(&arr[3]) - 3.0).abs() < 0.001, "arr[3] should be 3.0");
            assert!((to_f64(&arr[4]) - 4.0).abs() < 0.001, "arr[4] should be 4.0");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_avg_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_AVG([10.0, 20.0, 30.0, 40.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!((to_f64(&arr[0]) - 10.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 15.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 25.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 35.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ROLLING_WINDOW_MIN ────────────────────────────────────────────────────────

#[test]
fn test_rolling_window_min_basic() {
    let (_dir, _db, ex) = setup();
    // ROLLING_WINDOW_MIN([3,1,4,1,5,9], 3) -> [3,1,1,1,1,1] (using available elements at edges)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_WINDOW_MIN([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6);
            assert!((to_f64(&arr[0]) - 3.0).abs() < 0.001, "arr[0] should be 3.0");
            assert!((to_f64(&arr[1]) - 1.0).abs() < 0.001, "arr[1] should be 1.0");
            assert!((to_f64(&arr[2]) - 1.0).abs() < 0.001, "arr[2] min(3,1,4)=1");
            assert!((to_f64(&arr[3]) - 1.0).abs() < 0.001, "arr[3] min(1,4,1)=1");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ROLLING_WINDOW_MAX ────────────────────────────────────────────────────────

#[test]
fn test_rolling_window_max_basic() {
    let (_dir, _db, ex) = setup();
    // ROLLING_WINDOW_MAX([3,1,4,1,5,9], 3) -> [3,3,4,4,5,9]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_WINDOW_MAX([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 6);
            assert!((to_f64(&arr[0]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 4.0).abs() < 0.001);
            assert!((to_f64(&arr[5]) - 9.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ROLLING_WINDOW_SUM ────────────────────────────────────────────────────────

#[test]
fn test_rolling_window_sum_basic() {
    let (_dir, _db, ex) = setup();
    // ROLLING_WINDOW_SUM([1,2,3,4,5], 3) -> [1, 3, 6, 9, 12]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_WINDOW_SUM([1.0, 2.0, 3.0, 4.0, 5.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert!((to_f64(&arr[0]) - 1.0).abs() < 0.001, "arr[0] should be 1.0");
            assert!((to_f64(&arr[1]) - 3.0).abs() < 0.001, "arr[1] should be 3.0");
            assert!((to_f64(&arr[2]) - 6.0).abs() < 0.001, "arr[2] should be 6.0");
            assert!((to_f64(&arr[3]) - 9.0).abs() < 0.001, "arr[3] should be 9.0");
            assert!((to_f64(&arr[4]) - 12.0).abs() < 0.001, "arr[4] should be 12.0");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── FIRST_DIFF ────────────────────────────────────────────────────────────────

#[test]
fn test_first_diff_basic() {
    let (_dir, _db, ex) = setup();
    // FIRST_DIFF([10, 15, 12, 20]) -> [null, 5.0, -3.0, 8.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIRST_DIFF([10.0, 15.0, 12.0, 20.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Value::Null, "first element should be Null");
            assert!((to_f64(&arr[1]) - 5.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - (-3.0)).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 8.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_first_diff_single() {
    let (_dir, _db, ex) = setup();
    // FIRST_DIFF([42.0]) -> [null]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIRST_DIFF([42.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], Value::Null);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_PCT_CHANGE ──────────────────────────────────────────────────────────

#[test]
fn test_array_pct_change_basic() {
    let (_dir, _db, ex) = setup();
    // ARRAY_PCT_CHANGE([100, 110, 99]) -> [null, 0.10, -0.10]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ARRAY_PCT_CHANGE([100.0, 110.0, 99.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Null, "first element should be Null");
            assert!((to_f64(&arr[1]) - 0.10).abs() < 0.001, "10% increase expected, got {}", to_f64(&arr[1]));
            assert!((to_f64(&arr[2]) - (-0.10)).abs() < 0.001, "-10% decrease expected, got {}", to_f64(&arr[2]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_pct_change_zero_divisor() {
    let (_dir, _db, ex) = setup();
    // ARRAY_PCT_CHANGE([0, 10]) -> [null, null] (division by zero -> null)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ARRAY_PCT_CHANGE([0.0, 10.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::Null);
            assert_eq!(arr[1], Value::Null, "division by zero should yield Null");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_ZSCORE / Z_SCORES ───────────────────────────────────────────────────

#[test]
fn test_z_scores_basic() {
    let (_dir, _db, ex) = setup();
    // Z_SCORES([2, 4, 4, 4, 5, 5, 7, 9]) should have mean=5, std≈2
    // z-score of 5 should be ~0.0
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ARRAY_ZSCORE([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 8);
            // mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5
            // z-score of first element (2): (2-5)/std < 0
            assert!(to_f64(&arr[0]) < 0.0, "z-score of lowest value should be negative");
            // z-score of last element (9): (9-5)/std > 0
            assert!(to_f64(&arr[7]) > 0.0, "z-score of highest value should be positive");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_z_scores_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = Z_SCORES([1.0, 2.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            // mean=2, std=sqrt(2/3), z[0]=(1-2)/std < 0
            assert!(to_f64(&arr[0]) < 0.0);
            assert!((to_f64(&arr[1])).abs() < 0.001, "middle z-score should be ~0");
            assert!(to_f64(&arr[2]) > 0.0);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_z_scores_constant_array() {
    let (_dir, _db, ex) = setup();
    // Z_SCORES([5, 5, 5]) -> [0.0, 0.0, 0.0] (std=0 case)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = Z_SCORES([5.0, 5.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert!((to_f64(v) - 0.0).abs() < 0.001, "constant array z-scores should be 0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_MINMAX_SCALE / MINMAX_SCALE ─────────────────────────────────────────

#[test]
fn test_minmax_scale_basic() {
    let (_dir, _db, ex) = setup();
    // MINMAX_SCALE([0, 5, 10]) -> [0.0, 0.5, 1.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ARRAY_MINMAX_SCALE([0.0, 5.0, 10.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((to_f64(&arr[0]) - 0.0).abs() < 0.001, "min should scale to 0.0");
            assert!((to_f64(&arr[1]) - 0.5).abs() < 0.001, "mid should scale to 0.5");
            assert!((to_f64(&arr[2]) - 1.0).abs() < 0.001, "max should scale to 1.0");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_minmax_scale_alias() {
    let (_dir, _db, ex) = setup();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MINMAX_SCALE([2.0, 4.0, 6.0, 8.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!((to_f64(&arr[0]) - 0.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 1.0).abs() < 0.001);
            // Mid values should be 1/3 and 2/3
            assert!((to_f64(&arr[1]) - 1.0 / 3.0).abs() < 0.01);
            assert!((to_f64(&arr[2]) - 2.0 / 3.0).abs() < 0.01);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_minmax_scale_constant() {
    let (_dir, _db, ex) = setup();
    // MINMAX_SCALE([7, 7, 7]) -> [0, 0, 0] (range=0 case)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MINMAX_SCALE([7.0, 7.0, 7.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert!((to_f64(v) - 0.0).abs() < 0.001, "constant array should scale to all 0.0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── Edge cases and integration ─────────────────────────────────────────────────

#[test]
fn test_run_min_monotone_increasing() {
    let (_dir, _db, ex) = setup();
    // RUN_MIN on increasing array -> each element equals the first
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_MIN([1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            for v in arr {
                assert!((to_f64(v) - 1.0).abs() < 0.001, "running min of increasing array should always be 1.0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_run_max_monotone_decreasing() {
    let (_dir, _db, ex) = setup();
    // RUN_MAX on decreasing array -> each element equals the first
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_MAX([10.0, 8.0, 6.0, 4.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            for v in arr {
                assert!((to_f64(v) - 10.0).abs() < 0.001, "running max of decreasing array should always be 10.0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_dense_rank_all_same() {
    let (_dir, _db, ex) = setup();
    // DENSE_RANK([5, 5, 5]) -> [1, 1, 1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DENSE_RANK([5.0, 5.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert_eq!(*v, Value::Integer(1), "all same values should have rank 1");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_row_number_array_length_matches() {
    let (_dir, _db, ex) = setup();
    // ROW_NUM_ARR should always return an array with same length as input
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROW_NUM_ARR([99.0, 88.0, 77.0, 66.0, 55.0, 44.0, 33.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 7);
            assert_eq!(arr[6], Value::Integer(7));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lead_array_full_offset() {
    let (_dir, _db, ex) = setup();
    // LEAD_ARRAY([1,2,3], 3) -> [null, null, null] (all offset beyond array)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LEAD_ARRAY([1.0, 2.0, 3.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert_eq!(*v, Value::Null, "all elements beyond offset should be Null");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lag_array_full_offset() {
    let (_dir, _db, ex) = setup();
    // LAG_ARRAY([1,2,3], 5) -> [null, null, null] (all before start)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LAG_ARRAY([1.0, 2.0, 3.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert_eq!(*v, Value::Null, "all elements before start should be Null");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_cume_dist_all_same() {
    let (_dir, _db, ex) = setup();
    // CUME_DIST([3, 3, 3]) -> [1.0, 1.0, 1.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CUME_DIST([3.0, 3.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            for v in arr {
                assert!((to_f64(v) - 1.0).abs() < 0.001, "all same values should have cume_dist=1.0");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_window_sum_window1() {
    let (_dir, _db, ex) = setup();
    // ROLLING_WINDOW_SUM with window=1 -> same as input
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_WINDOW_SUM([3.0, 7.0, 2.0, 8.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
            assert!((to_f64(&arr[0]) - 3.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 7.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 2.0).abs() < 0.001);
            assert!((to_f64(&arr[3]) - 8.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_ntile_array_single_bucket() {
    let (_dir, _db, ex) = setup();
    // NTILE([1,2,3,4,5], 1) -> all in bucket 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NTILE_ARRAY([1.0, 2.0, 3.0, 4.0, 5.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            for v in arr {
                assert_eq!(*v, Value::Integer(1), "all elements should be in bucket 1");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_first_diff_two_elements() {
    let (_dir, _db, ex) = setup();
    // FIRST_DIFF([100, 200]) -> [null, 100.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIRST_DIFF([100.0, 200.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::Null);
            assert!((to_f64(&arr[1]) - 100.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rank_arr_no_ties() {
    let (_dir, _db, ex) = setup();
    // RANK_ARR([1,2,3,4,5]) -> [1,2,3,4,5] (no ties, same as position)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RANK_ARR([1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            for (i, v) in arr.iter().enumerate() {
                assert_eq!(*v, Value::Integer((i + 1) as i64), "rank should equal position+1 for sorted array");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_run_avg_single_element() {
    let (_dir, _db, ex) = setup();
    // RUN_AVG([42.0]) -> [42.0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = RUN_AVG([42.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1);
            assert!((to_f64(&arr[0]) - 42.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_window_min_window_larger_than_array() {
    let (_dir, _db, ex) = setup();
    // ROLLING_WINDOW_MIN([5, 2, 8], 10) -> [5, 2, 2] (available elements used)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ROLLING_WINDOW_MIN([5.0, 2.0, 8.0], 10) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            // At position 0: window from 0..=0 -> min([5]) = 5
            assert!((to_f64(&arr[0]) - 5.0).abs() < 0.001);
            // At position 1: window from 0..=1 -> min([5,2]) = 2
            assert!((to_f64(&arr[1]) - 2.0).abs() < 0.001);
            // At position 2: window from 0..=2 -> min([5,2,8]) = 2
            assert!((to_f64(&arr[2]) - 2.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
