/// Integration tests for PQL advanced array and sequence manipulation functions.
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

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

// ── ARRAY_ROTATE ──────────────────────────────────────────────────────────────

#[test]
fn test_array_rotate_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_ROTATE(arr, 1) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(to_f64(&a[0]), 2.0, "expected 2 at index 0");
            assert_eq!(to_f64(&a[1]), 3.0, "expected 3 at index 1");
            assert_eq!(to_f64(&a[2]), 4.0, "expected 4 at index 2");
            assert_eq!(to_f64(&a[3]), 1.0, "expected 1 at index 3");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rotate_array_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROTATE_ARRAY(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
            assert_eq!(to_f64(&a[0]), 3.0, "expected 3 at index 0 after rotate 2");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_PAD ─────────────────────────────────────────────────────────────────

#[test]
fn test_array_pad_right() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_PAD(arr, 5, 0) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "expected length 5");
            assert_eq!(to_f64(&a[0]), 1.0);
            assert_eq!(to_f64(&a[1]), 2.0);
            assert_eq!(to_f64(&a[2]), 0.0, "padded value should be 0");
            assert_eq!(to_f64(&a[3]), 0.0);
            assert_eq!(to_f64(&a[4]), 0.0);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_pad_left() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_PAD(arr, 4, 0, 'left') SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4, "expected length 4");
            assert_eq!(to_f64(&a[0]), 0.0, "first element should be pad value");
            assert_eq!(to_f64(&a[1]), 0.0);
            assert_eq!(to_f64(&a[2]), 1.0);
            assert_eq!(to_f64(&a[3]), 2.0);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_pad_array_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [7]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PAD_ARRAY(arr, 3, 0) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_CHUNK ───────────────────────────────────────────────────────────────

#[test]
fn test_array_chunk_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5, 6, 7, 8]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_CHUNK(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(chunks)) => {
            assert_eq!(chunks.len(), 3, "expected 3 chunks");
            match &chunks[0] {
                Value::Array(c) => assert_eq!(c.len(), 3, "first chunk should have 3 elements"),
                other => panic!("expected Array chunk, got {:?}", other),
            }
            match &chunks[2] {
                Value::Array(c) => assert_eq!(c.len(), 2, "last chunk should have 2 elements"),
                other => panic!("expected Array chunk, got {:?}", other),
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_chunk_array_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CHUNK_ARRAY(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(chunks)) => {
            assert_eq!(chunks.len(), 2, "expected 2 chunks of size 2");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_INTERLEAVE ──────────────────────────────────────────────────────────

#[test]
fn test_array_interleave_equal_length() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3], "b": [10, 20, 30]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_INTERLEAVE(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 6, "interleaved length should be 6");
            assert_eq!(to_f64(&a[0]), 1.0);
            assert_eq!(to_f64(&a[1]), 10.0);
            assert_eq!(to_f64(&a[2]), 2.0);
            assert_eq!(to_f64(&a[3]), 20.0);
            assert_eq!(to_f64(&a[4]), 3.0);
            assert_eq!(to_f64(&a[5]), 30.0);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_interleave_arrays_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2], "b": [3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INTERLEAVE_ARRAYS(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_DEDUP_CONSECUTIVE ───────────────────────────────────────────────────

#[test]
fn test_array_dedup_consecutive_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 1, 2, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DEDUP_CONSECUTIVE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected [1,2,3] after dedup consecutive");
            assert_eq!(to_f64(&a[0]), 1.0);
            assert_eq!(to_f64(&a[1]), 2.0);
            assert_eq!(to_f64(&a[2]), 3.0);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_dedup_consecutive_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 5, 5, 6, 6]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = DEDUP_CONSECUTIVE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "expected [5,6] after dedup consecutive");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_SYMMETRIC_DIFF ──────────────────────────────────────────────────────

#[test]
fn test_array_symmetric_diff_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3], "b": [2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SYMMETRIC_DIFF(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "symmetric diff of [1,2,3] and [2,3,4] should be [1,4]");
            let vals: Vec<f64> = a.iter().map(|v| to_f64(v)).collect();
            assert!(vals.contains(&1.0), "1 should be in symmetric diff");
            assert!(vals.contains(&4.0), "4 should be in symmetric diff");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_set_sym_diff_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2], "b": [2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_SYM_DIFF(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_COMBINATIONS ────────────────────────────────────────────────────────

#[test]
fn test_array_combinations_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_COMBINATIONS(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(combos)) => {
            // C(3,2) = 3
            assert_eq!(combos.len(), 3, "expected 3 combinations of size 2 from [1,2,3]");
            // Each combination should have 2 elements
            for combo in combos {
                match combo {
                    Value::Array(c) => assert_eq!(c.len(), 2, "each combination should have 2 elements"),
                    other => panic!("expected Array combination, got {:?}", other),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_combinations_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = COMBINATIONS(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(combos)) => {
            // C(4,3) = 4
            assert_eq!(combos.len(), 4, "expected 4 combinations of size 3 from [1,2,3,4]");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_CUMULATIVE_PRODUCT ──────────────────────────────────────────────────

#[test]
fn test_array_cumulative_product_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_CUMULATIVE_PRODUCT(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4, "expected 4 cumulative products");
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001, "expected 1.0 got {}", to_f64(&a[0]));
            assert!((to_f64(&a[1]) - 2.0).abs() < 0.001, "expected 2.0 got {}", to_f64(&a[1]));
            assert!((to_f64(&a[2]) - 6.0).abs() < 0.001, "expected 6.0 got {}", to_f64(&a[2]));
            assert!((to_f64(&a[3]) - 24.0).abs() < 0.001, "expected 24.0 got {}", to_f64(&a[3]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_cumprod_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CUMPROD(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert!((to_f64(a.last().unwrap()) - 24.0).abs() < 0.001, "expected 24 at end");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_WINSORIZE ───────────────────────────────────────────────────────────

#[test]
fn test_array_winsorize_clips_extremes() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_WINSORIZE(arr, 0.1, 0.9) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 10, "winsorize should preserve array length");
            // All values should be in range of the 10th and 90th percentile
            let min_v = a.iter().map(|v| to_f64(v)).fold(f64::INFINITY, f64::min);
            let max_v = a.iter().map(|v| to_f64(v)).fold(f64::NEG_INFINITY, f64::max);
            assert!(min_v >= 1.0, "min should be clipped at lower percentile bound");
            assert!(max_v <= 10.0, "max should be clipped at upper percentile bound");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_winsorize_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = WINSORIZE(arr, 0.2, 0.8) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_SLIDING_MIN ─────────────────────────────────────────────────────────

#[test]
fn test_array_sliding_min_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SLIDING_MIN(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // windows of size 3: [3,1,4]=1, [1,4,1]=1, [4,1,5]=1
            assert_eq!(a.len(), 3, "5 elements - window 3 + 1 = 3 windows");
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001, "window [3,1,4] min = 1");
            assert!((to_f64(&a[1]) - 1.0).abs() < 0.001, "window [1,4,1] min = 1");
            assert!((to_f64(&a[2]) - 1.0).abs() < 0.001, "window [4,1,5] min = 1");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_min_array_version() {
    // Note: This tests the ARRAY_SLIDING_MIN function only; ROLLING_MIN already
    // handles the group-row window case separately.
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 3, 8, 2, 7]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SLIDING_MIN(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // windows: [5,3]=3, [3,8]=3, [8,2]=2, [2,7]=2
            assert_eq!(a.len(), 4);
            assert!((to_f64(&a[0]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 2.0).abs() < 0.001);
            assert!((to_f64(&a[3]) - 2.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_SLIDING_MAX ─────────────────────────────────────────────────────────

#[test]
fn test_array_sliding_max_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SLIDING_MAX(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // windows: [3,1,4]=4, [1,4,1]=4, [4,1,5]=5
            assert_eq!(a.len(), 3);
            assert!((to_f64(&a[0]) - 4.0).abs() < 0.001, "window [3,1,4] max = 4");
            assert!((to_f64(&a[1]) - 4.0).abs() < 0.001, "window [1,4,1] max = 4");
            assert!((to_f64(&a[2]) - 5.0).abs() < 0.001, "window [4,1,5] max = 5");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_sliding_max_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 3, 2, 5, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SLIDING_MAX(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // windows: [1,3]=3, [3,2]=3, [2,5]=5, [5,4]=5
            assert_eq!(a.len(), 4);
            assert!((to_f64(&a[0]) - 3.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── FIRST_DIFFERENCE ──────────────────────────────────────────────────────────

#[test]
fn test_first_difference_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 3, 6, 10]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FIRST_DIFFERENCE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 first differences from 4 elements");
            assert!((to_f64(&a[0]) - 2.0).abs() < 0.001, "3-1 = 2");
            assert!((to_f64(&a[1]) - 3.0).abs() < 0.001, "6-3 = 3");
            assert!((to_f64(&a[2]) - 4.0).abs() < 0.001, "10-6 = 4");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_first_difference_uniform() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 10, 15, 20]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FIRST_DIFFERENCE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            for val in a {
                assert!((to_f64(val) - 5.0).abs() < 0.001, "all diffs should be 5");
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_COVARIANCE ──────────────────────────────────────────────────────────

#[test]
fn test_array_covariance_positive() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3], "b": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_COVARIANCE(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            let cov = to_f64(v);
            assert!(cov > 0.0, "covariance of perfectly correlated arrays should be positive, got {}", cov);
            assert!((cov - 1.0).abs() < 0.001, "sample covariance of [1,2,3] with itself = 1.0, got {}", cov);
        }
        None => panic!("expected covariance value, got None"),
    }
}

#[test]
fn test_array_cov_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_COV(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            let cov = to_f64(v);
            assert!(cov < 0.0, "covariance of negatively correlated arrays should be negative, got {}", cov);
        }
        None => panic!("expected covariance value, got None"),
    }
}

// ── ARRAY_CORRELATION ─────────────────────────────────────────────────────────

#[test]
fn test_array_correlation_perfect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_CORRELATION(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            let corr = to_f64(v);
            assert!((corr - 1.0).abs() < 0.0001, "perfect correlation should be 1.0, got {}", corr);
        }
        None => panic!("expected correlation value, got None"),
    }
}

#[test]
fn test_array_corr_alias_negative() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_CORR(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            let corr = to_f64(v);
            assert!((corr - (-1.0)).abs() < 0.0001, "perfect negative correlation should be -1.0, got {}", corr);
        }
        None => panic!("expected correlation value, got None"),
    }
}

// ── ARRAY_LINREG ──────────────────────────────────────────────────────────────

#[test]
fn test_array_linreg_single_arg() {
    // ARRAY_LINREG([1,2,3,4]) with implicit x = [0,1,2,3]
    // y = x + 1, so slope=1, intercept=1, r²=1
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_LINREG(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(m)) => {
            let slope = to_f64(m.get("slope").unwrap());
            let intercept = to_f64(m.get("intercept").unwrap());
            let r_sq = to_f64(m.get("r_squared").unwrap());
            assert!((slope - 1.0).abs() < 0.001, "slope should be 1.0, got {}", slope);
            assert!((intercept - 1.0).abs() < 0.001, "intercept should be 1.0, got {}", intercept);
            assert!((r_sq - 1.0).abs() < 0.001, "r_squared should be 1.0, got {}", r_sq);
        }
        other => panic!("expected Object with slope/intercept/r_squared, got {:?}", other),
    }
}

#[test]
fn test_array_linear_regression_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [2, 4, 6, 8]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_LINEAR_REGRESSION(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("slope"), "result should contain slope");
            assert!(m.contains_key("intercept"), "result should contain intercept");
            assert!(m.contains_key("r_squared"), "result should contain r_squared");
            let slope = to_f64(m.get("slope").unwrap());
            // y = 2x + 2 (x=[0,1,2,3]), slope=2, intercept=2
            assert!((slope - 2.0).abs() < 0.001, "slope should be 2.0, got {}", slope);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_array_linreg_with_xy() {
    // ARRAY_LINREG(x, y) with a perfect line y = 3x + 5
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"xs": [1, 2, 3, 4], "ys": [8, 11, 14, 17]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_LINREG(xs, ys) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(m)) => {
            let slope = to_f64(m.get("slope").unwrap());
            let intercept = to_f64(m.get("intercept").unwrap());
            let r_sq = to_f64(m.get("r_squared").unwrap());
            assert!((slope - 3.0).abs() < 0.001, "slope should be 3.0, got {}", slope);
            assert!((intercept - 5.0).abs() < 0.001, "intercept should be 5.0, got {}", intercept);
            assert!((r_sq - 1.0).abs() < 0.001, "r_squared should be 1.0, got {}", r_sq);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}
