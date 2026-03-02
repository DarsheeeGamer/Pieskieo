/// Integration tests for PQL data transformation, reshaping, and structural manipulation functions.
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

// ── TRANSPOSE_MATRIX ─────────────────────────────────────────────────────────

#[test]
fn test_transpose_matrix_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m": [[1,2,3],[4,5,6]]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TRANSPOSE_MATRIX(m) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 3, "expected 3 cols after transpose");
            match &rows[0] {
                Value::Array(col) => assert_eq!(col.len(), 2, "each col should have 2 rows"),
                _ => panic!("expected inner array"),
            }
            // First column should be [1, 4]
            if let Value::Array(col0) = &rows[0] {
                assert!((to_f64(&col0[0]) - 1.0).abs() < 0.001);
                assert!((to_f64(&col0[1]) - 4.0).abs() < 0.001);
            }
            // Second column should be [2, 5]
            if let Value::Array(col1) = &rows[1] {
                assert!((to_f64(&col1[0]) - 2.0).abs() < 0.001);
                assert!((to_f64(&col1[1]) - 5.0).abs() < 0.001);
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_matrix_transpose_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m": [[1,2],[3,4],[5,6]]}),
    )
    .unwrap();
    // 3x2 matrix transposed to 2x3
    let mut p = Parser::new(r#"QUERY t COMPUTE out = MATRIX_TRANSPOSE(m) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 2, "expected 2 rows after transpose of 3x2");
            match &rows[0] {
                Value::Array(c) => assert_eq!(c.len(), 3),
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── CARTESIAN_PRODUCT ────────────────────────────────────────────────────────

#[test]
fn test_cartesian_product_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1, 2], "b": ["x", "y"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CARTESIAN_PRODUCT(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(pairs)) => {
            assert_eq!(pairs.len(), 4, "2x2 cartesian product should yield 4 pairs");
            // Check first pair is [1, "x"]
            if let Value::Array(p0) = &pairs[0] {
                assert_eq!(p0.len(), 2);
                assert!((to_f64(&p0[0]) - 1.0).abs() < 0.001);
                assert_eq!(p0[1], Value::String("x".to_string()));
            } else {
                panic!("expected inner array pair");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── TALLY / VALUE_COUNTS ─────────────────────────────────────────────────────

#[test]
fn test_tally_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 1, 3, 2, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TALLY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(counts)) => {
            assert_eq!(
                counts.get("1"),
                Some(&Value::Integer(3)),
                "value 1 should appear 3 times"
            );
            assert_eq!(
                counts.get("2"),
                Some(&Value::Integer(2)),
                "value 2 should appear 2 times"
            );
            assert_eq!(
                counts.get("3"),
                Some(&Value::Integer(1)),
                "value 3 should appear 1 time"
            );
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_value_counts_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": ["a", "b", "a"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = VALUE_COUNTS(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(counts)) => {
            assert_eq!(counts.get("a"), Some(&Value::Integer(2)));
            assert_eq!(counts.get("b"), Some(&Value::Integer(1)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── NORMALIZE_ARRAY ───────────────────────────────────────────────────────────

#[test]
fn test_normalize_array_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NORMALIZE_ARRAY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(vals)) => {
            assert_eq!(vals.len(), 5);
            assert!((to_f64(&vals[0]) - 0.0).abs() < 0.001, "min should be 0");
            assert!((to_f64(&vals[2]) - 0.5).abs() < 0.001, "mid should be 0.5");
            assert!((to_f64(&vals[4]) - 1.0).abs() < 0.001, "max should be 1.0");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_normalize_array_constant() {
    // All same values -> all 0.0
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [5, 5, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NORMALIZE_ARRAY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(vals)) => {
            for v in vals {
                assert!(
                    (to_f64(v) - 0.0).abs() < 0.001,
                    "constant array should normalize to all 0.0"
                );
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── ARRAY_GROUP_BY ────────────────────────────────────────────────────────────

#[test]
fn test_array_group_by_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({
            "items": [
                {"cat": "a", "val": 1},
                {"cat": "a", "val": 2},
                {"cat": "b", "val": 3}
            ]
        }),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = ARRAY_GROUP_BY(items, "cat") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(groups)) => {
            assert!(groups.contains_key("a"), "group 'a' should exist");
            assert!(groups.contains_key("b"), "group 'b' should exist");
            if let Some(Value::Array(a_group)) = groups.get("a") {
                assert_eq!(a_group.len(), 2, "group 'a' should have 2 items");
            } else {
                panic!("expected array for group 'a'");
            }
            if let Some(Value::Array(b_group)) = groups.get("b") {
                assert_eq!(b_group.len(), 1, "group 'b' should have 1 item");
            } else {
                panic!("expected array for group 'b'");
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── IDENTITY_MATRIX ───────────────────────────────────────────────────────────

#[test]
fn test_identity_matrix_3x3() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 3}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IDENTITY_MATRIX(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 3);
            for (i, row) in rows.iter().enumerate() {
                if let Value::Array(cols) = row {
                    assert_eq!(cols.len(), 3);
                    for (j, v) in cols.iter().enumerate() {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (to_f64(v) - expected).abs() < 0.001,
                            "expected [{i}][{j}] = {expected}, got {}",
                            to_f64(v)
                        );
                    }
                } else {
                    panic!("expected inner array");
                }
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── RESHAPE ───────────────────────────────────────────────────────────────────

#[test]
fn test_reshape_flat_to_matrix() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = RESHAPE(arr, 2, 3) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 2, "should have 2 rows");
            match &rows[0] {
                Value::Array(c) => {
                    assert_eq!(c.len(), 3, "each row should have 3 cols");
                    assert!((to_f64(&c[0]) - 1.0).abs() < 0.001);
                    assert!((to_f64(&c[1]) - 2.0).abs() < 0.001);
                    assert!((to_f64(&c[2]) - 3.0).abs() < 0.001);
                }
                _ => panic!("expected inner array"),
            }
            match &rows[1] {
                Value::Array(c) => {
                    assert!((to_f64(&c[0]) - 4.0).abs() < 0.001);
                }
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_reshape_flatten() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m": [[1, 2], [3, 4]]}),
    )
    .unwrap();
    // Flatten to 1D with negative dim
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RESHAPE(m, -1) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(flat)) => {
            assert_eq!(flat.len(), 4, "flattened 2x2 should be 4 elements");
            assert!((to_f64(&flat[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&flat[3]) - 4.0).abs() < 0.001);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── MATRIX_TRACE ─────────────────────────────────────────────────────────────

#[test]
fn test_matrix_trace_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m": [[1, 2], [3, 4]]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = MATRIX_TRACE(m) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => {
            assert!(
                (to_f64(v) - 5.0).abs() < 0.001,
                "trace of [[1,2],[3,4]] should be 5.0, got {}",
                to_f64(v)
            );
        }
        None => panic!("no output value"),
    }
}

// ── MATRIX_MULTIPLY ───────────────────────────────────────────────────────────

#[test]
fn test_matrix_multiply_basic() {
    let (db, ex) = setup();
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({
            "a": [[1, 2], [3, 4]],
            "b": [[5, 6], [7, 8]]
        }),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = MATRIX_MULTIPLY(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 2);
            if let Value::Array(row0) = &rows[0] {
                assert!((to_f64(&row0[0]) - 19.0).abs() < 0.001, "expected 19, got {}", to_f64(&row0[0]));
                assert!((to_f64(&row0[1]) - 22.0).abs() < 0.001, "expected 22, got {}", to_f64(&row0[1]));
            } else {
                panic!("expected inner array");
            }
            if let Value::Array(row1) = &rows[1] {
                assert!((to_f64(&row1[0]) - 43.0).abs() < 0.001, "expected 43, got {}", to_f64(&row1[0]));
                assert!((to_f64(&row1[1]) - 50.0).abs() < 0.001, "expected 50, got {}", to_f64(&row1[1]));
            } else {
                panic!("expected inner array");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── ARRAY_PARTITION ───────────────────────────────────────────────────────────

#[test]
fn test_array_partition_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5, 9, 2, 6]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_PARTITION(arr, 4) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(parts)) => {
            assert_eq!(parts.len(), 2, "should produce 2 partitions");
            if let Value::Array(below) = &parts[0] {
                // Elements < 4: [3, 1, 1, 2]
                assert_eq!(below.len(), 4, "4 elements below pivot 4");
                for v in below {
                    assert!(to_f64(v) < 4.0, "element {} should be < 4", to_f64(v));
                }
            } else {
                panic!("expected array for below partition");
            }
            if let Value::Array(above) = &parts[1] {
                // Elements >= 4: [4, 5, 9, 6]
                assert_eq!(above.len(), 4, "4 elements >= pivot 4");
                for v in above {
                    assert!(to_f64(v) >= 4.0, "element {} should be >= 4", to_f64(v));
                }
            } else {
                panic!("expected array for above partition");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── ARRAY_WINDOW_AGG ──────────────────────────────────────────────────────────

#[test]
fn test_array_window_agg_avg() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = ARRAY_WINDOW_AGG(arr, 3, "avg") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(avgs)) => {
            // windows: [1,2,3]=2.0, [2,3,4]=3.0, [3,4,5]=4.0
            assert_eq!(avgs.len(), 3);
            assert!((to_f64(&avgs[0]) - 2.0).abs() < 0.001, "first window avg should be 2.0");
            assert!((to_f64(&avgs[1]) - 3.0).abs() < 0.001, "second window avg should be 3.0");
            assert!((to_f64(&avgs[2]) - 4.0).abs() < 0.001, "third window avg should be 4.0");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_array_window_agg_sum() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30, 40]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = WINDOW_AGG(arr, 2, "sum") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(sums)) => {
            // windows: [10,20]=30, [20,30]=50, [30,40]=70
            assert_eq!(sums.len(), 3);
            assert!((to_f64(&sums[0]) - 30.0).abs() < 0.001);
            assert!((to_f64(&sums[1]) - 50.0).abs() < 0.001);
            assert!((to_f64(&sums[2]) - 70.0).abs() < 0.001);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── RANK_TRANSFORM ────────────────────────────────────────────────────────────

#[test]
fn test_rank_transform_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [10, 30, 20, 30, 10]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = RANK_TRANSFORM(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(ranks)) => {
            assert_eq!(ranks.len(), 5);
            // Dense ranks: 10->1, 20->2, 30->3
            assert_eq!(ranks[0], Value::Integer(1), "10 should have rank 1");
            assert_eq!(ranks[1], Value::Integer(3), "30 should have rank 3");
            assert_eq!(ranks[2], Value::Integer(2), "20 should have rank 2");
            assert_eq!(ranks[3], Value::Integer(3), "30 should have rank 3");
            assert_eq!(ranks[4], Value::Integer(1), "10 should have rank 1");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── STANDARDIZE_ARRAY ────────────────────────────────────────────────────────

#[test]
fn test_standardize_array_basic() {
    let (db, ex) = setup();
    // [2,4,4,4,5,5,7,9] mean=5, std=2
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [2, 4, 4, 4, 5, 5, 7, 9]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = STANDARDIZE_ARRAY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(z_scores)) => {
            assert_eq!(z_scores.len(), 8);
            // Mean should be 5.0, std should be 2.0
            // z-score of 5 should be 0.0
            assert!(
                (to_f64(&z_scores[4]) - 0.0).abs() < 0.001,
                "z-score of mean value should be 0"
            );
            // z-score of 2 should be (2-5)/2 = -1.5
            assert!(
                (to_f64(&z_scores[0]) - (-1.5)).abs() < 0.001,
                "z-score of 2 should be -1.5, got {}",
                to_f64(&z_scores[0])
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── ARRAY_TO_OBJECT ───────────────────────────────────────────────────────────

#[test]
fn test_array_to_object_pairs() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pairs": [["name", "alice"], ["age", 30]]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_TO_OBJECT(pairs) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("name"), "should have 'name' key");
            assert!(obj.contains_key("age"), "should have 'age' key");
            assert_eq!(
                obj.get("name"),
                Some(&Value::String("alice".to_string()))
            );
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_pairs_to_object_two_arrays() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"keys": ["x", "y", "z"], "vals": [10, 20, 30]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE out = PAIRS_TO_OBJECT(keys, vals) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.len(), 3);
            assert_eq!(obj.get("x"), Some(&Value::Integer(10)));
            assert_eq!(obj.get("y"), Some(&Value::Integer(20)));
            assert_eq!(obj.get("z"), Some(&Value::Integer(30)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── OBJECT_TRANSFORM / TRANSFORM_VALUES ──────────────────────────────────────

#[test]
fn test_transform_values_to_upper() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"obj": {"greeting": "hello", "name": "world"}}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE out = TRANSFORM_VALUES(obj, "TO_UPPER") SELECT out;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(obj)) => {
            if let Some(Value::String(s)) = obj.get("greeting") {
                assert_eq!(s, "HELLO");
            } else {
                panic!("expected string for 'greeting'");
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_object_transform_abs() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"obj": {"a": -5, "b": -3}}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE out = OBJECT_TRANSFORM(obj, "ABS") SELECT out;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(obj)) => {
            if let Some(v) = obj.get("a") {
                assert!(to_f64(v) >= 0.0, "ABS should make value non-negative");
                assert!((to_f64(v) - 5.0).abs() < 0.001);
            } else {
                panic!("expected key 'a'");
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── DIAGONAL ─────────────────────────────────────────────────────────────────

#[test]
fn test_diagonal_extract() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m": [[1, 0, 0], [0, 2, 0], [0, 0, 3]]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = DIAGONAL(m) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(diag)) => {
            assert_eq!(diag.len(), 3);
            assert!((to_f64(&diag[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&diag[1]) - 2.0).abs() < 0.001);
            assert!((to_f64(&diag[2]) - 3.0).abs() < 0.001);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── EYE (IDENTITY_MATRIX alias) ───────────────────────────────────────────────

#[test]
fn test_eye_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 2}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = EYE(n) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 2);
            if let Value::Array(r0) = &rows[0] {
                assert!((to_f64(&r0[0]) - 1.0).abs() < 0.001);
                assert!((to_f64(&r0[1]) - 0.0).abs() < 0.001);
            }
            if let Value::Array(r1) = &rows[1] {
                assert!((to_f64(&r1[0]) - 0.0).abs() < 0.001);
                assert!((to_f64(&r1[1]) - 1.0).abs() < 0.001);
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}
