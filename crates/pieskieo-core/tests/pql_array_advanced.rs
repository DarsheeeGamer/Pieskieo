/// Integration tests for advanced PQL array analytics functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

// ── ARRAY_RUNNING_SUM ─────────────────────────────────────────────────────────

#[test]
fn test_array_running_sum_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RUNNING_SUM(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
            // [1.0, 3.0, 6.0, 10.0]
            assert!(
                (to_f64(&a[0]) - 1.0).abs() < 0.001,
                "expected 1.0 got {}",
                to_f64(&a[0])
            );
            assert!(
                (to_f64(&a[1]) - 3.0).abs() < 0.001,
                "expected 3.0 got {}",
                to_f64(&a[1])
            );
            assert!(
                (to_f64(&a[2]) - 6.0).abs() < 0.001,
                "expected 6.0 got {}",
                to_f64(&a[2])
            );
            assert!(
                (to_f64(&a[3]) - 10.0).abs() < 0.001,
                "expected 10.0 got {}",
                to_f64(&a[3])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_cumsum_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 5, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_CUMSUM(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert!((to_f64(a.last().unwrap()) - 15.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_RUNNING_MAX ─────────────────────────────────────────────────────────

#[test]
fn test_array_running_max() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5, 9]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RUNNING_MAX(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [3, 3, 4, 4, 5, 9]
            assert_eq!(a.len(), 6);
            assert!((to_f64(&a[0]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 4.0).abs() < 0.001);
            assert!((to_f64(&a[3]) - 4.0).abs() < 0.001);
            assert!((to_f64(&a[4]) - 5.0).abs() < 0.001);
            assert!((to_f64(&a[5]) - 9.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_RUNNING_MIN ─────────────────────────────────────────────────────────

#[test]
fn test_array_running_min() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5, 9]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RUNNING_MIN(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [3, 1, 1, 1, 1, 1]
            assert_eq!(a.len(), 6);
            assert!((to_f64(&a[0]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 1.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 1.0).abs() < 0.001);
            assert!((to_f64(&a[5]) - 1.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_PRODUCT ─────────────────────────────────────────────────────────────

#[test]
fn test_array_product() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [2, 3, 4]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_PRODUCT(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => assert!(
            (to_f64(v) - 24.0).abs() < 0.001,
            "expected 24.0, got {}",
            to_f64(v)
        ),
        None => panic!("expected a value"),
    }
}

#[test]
fn test_array_prod_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_PROD(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => assert!((to_f64(v) - 120.0).abs() < 0.001),
        None => panic!("expected a value"),
    }
}

// ── ARRAY_RANK ────────────────────────────────────────────────────────────────

#[test]
fn test_array_rank_dense() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 30, 20, 30, 10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RANK(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // dense rank: 10->1, 20->2, 30->3
            assert_eq!(a.len(), 5);
            assert_eq!(a[0], Value::Integer(1));
            assert_eq!(a[1], Value::Integer(3));
            assert_eq!(a[2], Value::Integer(2));
            assert_eq!(a[3], Value::Integer(3));
            assert_eq!(a[4], Value::Integer(1));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_PERCENTILE ─────────────────────────────────────────────────────────

#[test]
fn test_array_percentile_median() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_PERCENTILE(arr, 0.5) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => assert!(
            (to_f64(v) - 3.0).abs() < 0.001,
            "expected 3.0, got {}",
            to_f64(v)
        ),
        None => panic!("expected a value"),
    }
}

#[test]
fn test_array_quantile_min_max() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30, 40, 50]}),
    )
    .unwrap();
    // 0th percentile = min, 1.0 = max
    let mut p = Parser::new(
        r#"QUERY t COMPUTE lo = ARRAY_QUANTILE(arr, 0.0) COMPUTE hi = ARRAY_QUANTILE(arr, 1.0) SELECT lo, hi;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("lo") {
        Some(v) => assert!((to_f64(v) - 10.0).abs() < 0.001),
        None => panic!("missing lo"),
    }
    match r.rows[0].data.get("hi") {
        Some(v) => assert!((to_f64(v) - 50.0).abs() < 0.001),
        None => panic!("missing hi"),
    }
}

// ── ARRAY_SORT_UNIQUE ─────────────────────────────────────────────────────────

#[test]
fn test_array_sort_unique() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 2, 1, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SORT_UNIQUE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            // [1.0, 2.0, 3.0]
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 2.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 3.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_ZIP_WITH ────────────────────────────────────────────────────────────

#[test]
fn test_array_zip_with_add() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1, 2, 3], "b": [4, 5, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_ZIP_WITH(a, b, "ADD") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!((to_f64(&arr[0]) - 5.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 7.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 9.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_zip_with_mul() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [2, 3, 4], "b": [10, 10, 10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_ZIP_WITH(a, b, "MUL") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(arr)) => {
            assert!((to_f64(&arr[0]) - 20.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 30.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 40.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_SPLIT_CHUNKS ────────────────────────────────────────────────────────

#[test]
fn test_array_split_chunks() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SPLIT_CHUNKS(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(chunks)) => {
            // [[1,2],[3,4],[5]]
            assert_eq!(chunks.len(), 3);
            match &chunks[0] {
                Value::Array(c) => assert_eq!(c.len(), 2),
                other => panic!("expected inner Array, got {:?}", other),
            }
            match &chunks[2] {
                Value::Array(c) => assert_eq!(c.len(), 1),
                other => panic!("expected inner Array, got {:?}", other),
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ROTATE_ARRAY ─────────────────────────────────────────────────────────────

#[test]
fn test_rotate_array_left() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROTATE_ARRAY(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // rotate left by 2: [3,4,5,1,2]
            assert_eq!(a.len(), 5);
            assert_eq!(to_f64(&a[0]) as i64, 3);
            assert_eq!(to_f64(&a[1]) as i64, 4);
            assert_eq!(to_f64(&a[4]) as i64, 2);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_RANGE ──────────────────────────────────────────────────────────────

#[test]
fn test_array_range_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RANGE(0, 5) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [0,1,2,3,4]
            assert_eq!(a.len(), 5);
            assert_eq!(a[0], Value::Integer(0));
            assert_eq!(a[4], Value::Integer(4));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_int_range_with_step() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INT_RANGE(0, 10, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [0,2,4,6,8]
            assert_eq!(a.len(), 5);
            assert_eq!(a[0], Value::Integer(0));
            assert_eq!(a[2], Value::Integer(4));
            assert_eq!(a[4], Value::Integer(8));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_TAKE / ARRAY_DROP ───────────────────────────────────────────────────

#[test]
fn test_array_take() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30, 40, 50]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_TAKE(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(to_f64(&a[0]) as i64, 10);
            assert_eq!(to_f64(&a[2]) as i64, 30);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_head_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_HEAD(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 2),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_drop() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DROP(arr, 2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [3,4,5]
            assert_eq!(a.len(), 3);
            assert_eq!(to_f64(&a[0]) as i64, 3);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_tail_n_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30, 40, 50]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_TAIL_N(arr, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [30,40,50]
            assert_eq!(a.len(), 2);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_EVERY / ARRAY_SOME / ARRAY_NONE ────────────────────────────────────

#[test]
fn test_array_every_all_truthy() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_EVERY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_array_every_with_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 0, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_EVERY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(false)));
}

#[test]
fn test_all_match_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [true, true, true]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ALL_MATCH(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_array_some_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0, 0, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SOME(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_array_some_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0, 0, 0]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SOME(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(false)));
}

#[test]
fn test_array_none_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0, 0, 0]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_NONE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_array_none_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [0, 1, 0]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_NONE(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(false)));
}

#[test]
fn test_none_match_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [false, false]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NONE_MATCH(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

// ── GENERATE_ARRAY alias ──────────────────────────────────────────────────────

#[test]
fn test_generate_array() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = GENERATE_ARRAY(1, 4) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [1,2,3]
            assert_eq!(a.len(), 3);
            assert_eq!(a[0], Value::Integer(1));
            assert_eq!(a[2], Value::Integer(3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ZIP_WITH alias ────────────────────────────────────────────────────────────

#[test]
fn test_zip_with_max_pair() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1, 9, 3], "b": [5, 2, 7]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ZIP_WITH(a, b, "MAX_PAIR") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(arr)) => {
            // [5, 9, 7]
            assert!((to_f64(&arr[0]) - 5.0).abs() < 0.001);
            assert!((to_f64(&arr[1]) - 9.0).abs() < 0.001);
            assert!((to_f64(&arr[2]) - 7.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_CUMMAX_ARR alias ────────────────────────────────────────────────────

#[test]
fn test_array_cummax_arr_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 5, 3, 7, 2]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_CUMMAX_ARR(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            // [1, 5, 5, 7, 7]
            assert_eq!(a.len(), 5);
            assert!((to_f64(&a[1]) - 5.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 5.0).abs() < 0.001);
            assert!((to_f64(&a[3]) - 7.0).abs() < 0.001);
            assert!((to_f64(&a[4]) - 7.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_UNIQUE_SORTED alias ─────────────────────────────────────────────────

#[test]
fn test_array_unique_sorted_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 2, 5, 1, 2]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_UNIQUE_SORTED(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 5.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
