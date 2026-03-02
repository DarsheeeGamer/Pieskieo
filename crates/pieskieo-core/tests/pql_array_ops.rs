/// Integration tests for advanced PQL array/set operations and functional-style array functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
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

fn to_i64(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Float(f) => *f as i64,
        _ => 0,
    }
}

// ── ARRAY_ROTATE ──────────────────────────────────────────────────────────────
// Existing ARRAY_ROTATE: rotate_left(((n%len)+len)%len)
// n=2 on [1,2,3,4,5] → rotate_left(2) → [3,4,5,1,2]

#[test]
fn test_array_rotate_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5], "nv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_ROTATE(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
            assert_eq!(to_i64(&a[0]), 3, "expected 3 at [0]");
            assert_eq!(to_i64(&a[1]), 4, "expected 4 at [1]");
            assert_eq!(to_i64(&a[2]), 5, "expected 5 at [2]");
            assert_eq!(to_i64(&a[3]), 1, "expected 1 at [3]");
            assert_eq!(to_i64(&a[4]), 2, "expected 2 at [4]");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_rotate_negative() {
    // n=-1 on [1,2,3]: ((−1%3)+3)%3 = 2, rotate_left(2) → [3,1,2]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3], "nv": -1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_ROTATE(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            // existing ARRAY_ROTATE: rotate_left(2) → [3,1,2]
            assert_eq!(to_i64(&a[0]), 3, "expected 3 at [0]");
            assert_eq!(to_i64(&a[1]), 1, "expected 1 at [1]");
            assert_eq!(to_i64(&a[2]), 2, "expected 2 at [2]");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rotate_array_alias() {
    // ROTATE_ARRAY is the alias; n=1 on [10,20,30] → rotate_left(1) → [20,30,10]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 30], "nv": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROTATE_ARRAY(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(to_i64(&a[0]), 20);
            assert_eq!(to_i64(&a[1]), 30);
            assert_eq!(to_i64(&a[2]), 10);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_TAKE_WHILE ──────────────────────────────────────────────────────────

#[test]
fn test_array_take_while_basic() {
    // [1,2,3,4,5] take while < 3 → [1,2]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5], "thresh": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_TAKE_WHILE(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "expected 2 elements, got {}", a.len());
            assert_eq!(to_i64(&a[0]), 1);
            assert_eq!(to_i64(&a[1]), 2);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_take_while_lt_alias() {
    // TAKE_WHILE_LT alias: [5,1,2,3] take while < 5 → [](none, 5 is not < 5)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 1, 2, 3], "thresh": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TAKE_WHILE_LT(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 0, "expected empty array, got {:?}", a);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_take_while_all() {
    // all elements < threshold → returns whole array
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3], "thresh": 10})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_TAKE_WHILE(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_DROP_WHILE ──────────────────────────────────────────────────────────

#[test]
fn test_array_drop_while_basic() {
    // [1,2,3,4,5] drop while < 3 → [3,4,5]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5], "thresh": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DROP_WHILE(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 elements, got {}", a.len());
            assert_eq!(to_i64(&a[0]), 3);
            assert_eq!(to_i64(&a[1]), 4);
            assert_eq!(to_i64(&a[2]), 5);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_drop_while_lt_alias() {
    // DROP_WHILE_LT alias: [1,2,6,3] drop while < 5 → [6,3]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 6, 3], "thresh": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = DROP_WHILE_LT(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "expected 2 elements, got {}", a.len());
            assert_eq!(to_i64(&a[0]), 6);
            assert_eq!(to_i64(&a[1]), 3);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_FREQUENCIES ─────────────────────────────────────────────────────────

#[test]
fn test_array_frequencies_integers() {
    // [1,2,2,3,3,3] → {"1":1,"2":2,"3":3}
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 2, 3, 3, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_FREQUENCIES(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(m)) => {
            assert_eq!(to_i64(m.get("1").unwrap()), 1);
            assert_eq!(to_i64(m.get("2").unwrap()), 2);
            assert_eq!(to_i64(m.get("3").unwrap()), 3);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_value_counts_alias() {
    // VALUE_COUNTS is already defined as TALLY alias; test via put_doc
    // Using VALUE_COUNTS([1,1,2]) → {"1":2,"2":1}
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 1, 2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = VALUE_COUNTS(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(m)) => {
            assert_eq!(to_i64(m.get("1").unwrap()), 2, "expected count 2 for key '1'");
            assert_eq!(to_i64(m.get("2").unwrap()), 1, "expected count 1 for key '2'");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── ARRAY_NLARGEST ────────────────────────────────────────────────────────────

#[test]
fn test_array_nlargest_basic() {
    // [3,1,4,1,5,9,2,6] nlargest 3 → [9,6,5]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5, 9, 2, 6], "nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_NLARGEST(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 elements");
            assert_eq!(to_i64(&a[0]), 9);
            assert_eq!(to_i64(&a[1]), 6);
            assert_eq!(to_i64(&a[2]), 5);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_top_n_values_alias() {
    // TOP_N_VALUES([10, 20, 5, 15], 2) → [20, 15]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 5, 15], "nv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TOP_N_VALUES(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            assert_eq!(to_i64(&a[0]), 20);
            assert_eq!(to_i64(&a[1]), 15);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_NSMALLEST ───────────────────────────────────────────────────────────

#[test]
fn test_array_nsmallest_basic() {
    // [3,1,4,1,5,9,2,6] nsmallest 3 → [1,1,2]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [3, 1, 4, 1, 5, 9, 2, 6], "nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_NSMALLEST(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert_eq!(to_i64(&a[0]), 1);
            assert_eq!(to_i64(&a[1]), 1);
            assert_eq!(to_i64(&a[2]), 2);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_bottom_n_values_alias() {
    // BOTTOM_N_VALUES([10, 20, 5, 15], 2) → [5, 10]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [10, 20, 5, 15], "nv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = BOTTOM_N_VALUES(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            assert_eq!(to_i64(&a[0]), 5);
            assert_eq!(to_i64(&a[1]), 10);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SET_UNION / ARRAY_UNION ───────────────────────────────────────────────────

#[test]
fn test_set_union_basic() {
    // SET_UNION([1,2,3],[3,4,5]) → [1,2,3,4,5]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [3, 4, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_UNION(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "expected 5 unique elements, got {:?}", a);
            let vals: Vec<i64> = a.iter().map(to_i64).collect();
            assert!(vals.contains(&1));
            assert!(vals.contains(&2));
            assert!(vals.contains(&3));
            assert!(vals.contains(&4));
            assert!(vals.contains(&5));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_union_alias() {
    // ARRAY_UNION([1,2,3],[3,4,5]) → [1,2,3,4,5]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [3, 4, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_UNION(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "expected 5 unique elements, got {:?}", a);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SET_INTERSECT / ARRAY_INTERSECT ──────────────────────────────────────────

#[test]
fn test_set_intersect_basic() {
    // SET_INTERSECT([1,2,3],[2,3,4]) → [2,3]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_INTERSECT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "expected [2,3], got {:?}", a);
            let vals: Vec<i64> = a.iter().map(to_i64).collect();
            assert!(vals.contains(&2));
            assert!(vals.contains(&3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_intersect_alias() {
    // ARRAY_INTERSECT is an existing alias under SET_INTERSECTION
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_INTERSECT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            let vals: Vec<i64> = a.iter().map(to_i64).collect();
            assert!(vals.contains(&2));
            assert!(vals.contains(&3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SET_DIFFERENCE / ARRAY_DIFFERENCE ────────────────────────────────────────

#[test]
fn test_set_difference_basic() {
    // SET_DIFFERENCE([1,2,3],[2,3]) → [1]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_DIFFERENCE(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1, "expected [1], got {:?}", a);
            assert_eq!(to_i64(&a[0]), 1);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_difference_alias() {
    // ARRAY_DIFFERENCE([1,2,3],[2,3]) → [1]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DIFFERENCE(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1);
            assert_eq!(to_i64(&a[0]), 1);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SET_IS_SUBSET / IS_SUBSET ─────────────────────────────────────────────────

#[test]
fn test_set_is_subset_true() {
    // [1,2] ⊆ [1,2,3] → true
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2], "a2": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_IS_SUBSET(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(*b, "expected true"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

#[test]
fn test_is_subset_alias_false() {
    // IS_SUBSET([1,4],[1,2,3]) → false (4 not in a2)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 4], "a2": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_SUBSET(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(!b, "expected false"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

// ── SET_IS_SUPERSET / IS_SUPERSET ─────────────────────────────────────────────

#[test]
fn test_set_is_superset_true() {
    // [1,2,3] ⊇ [1,2] → true
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [1, 2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_IS_SUPERSET(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(*b, "expected true"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

#[test]
fn test_is_superset_alias_false() {
    // IS_SUPERSET([1,2],[1,2,3]) → false (3 not in a1)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2], "a2": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_SUPERSET(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(!b, "expected false"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

// ── SET_EQUALS / ARRAY_SET_EQ ─────────────────────────────────────────────────

#[test]
fn test_set_equals_true() {
    // [1,2,3] equals [3,1,2] → true (same elements regardless of order)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [3, 1, 2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_EQUALS(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(*b, "expected true"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

#[test]
fn test_set_equals_false() {
    // [1,2] vs [1,3] → false
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2], "a2": [1, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_EQUALS(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(!b, "expected false"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

#[test]
fn test_array_set_eq_alias() {
    // ARRAY_SET_EQ([1,1,2],[2,1]) → true (same unique elements)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 1, 2], "a2": [2, 1]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SET_EQ(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(*b, "expected true"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

// ── ARRAY_FILL / FILL_ARRAY ───────────────────────────────────────────────────

#[test]
fn test_array_fill_integers() {
    // ARRAY_FILL(0, 5) → [0,0,0,0,0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_FILL(0, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
            for v in a { assert_eq!(to_i64(v), 0); }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_fill_array_alias_string() {
    // FILL_ARRAY("x", 3) → ["x","x","x"]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = FILL_ARRAY('x', nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            for v in a {
                match v {
                    Value::String(s) => assert_eq!(s, "x"),
                    other => panic!("expected String, got {:?}", other),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_IOTA ────────────────────────────────────────────────────────────────

#[test]
fn test_array_iota_basic() {
    // ARRAY_IOTA(5) → [0,1,2,3,4]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_IOTA(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
            for (i, v) in a.iter().enumerate() {
                assert_eq!(to_i64(v), i as i64, "expected {} at index {}", i, i);
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_range_existing() {
    // ARRAY_RANGE(start, stop) → integer range (existing function with 2-arg form)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RANGE(0, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "expected 5 elements [0..4]");
            assert_eq!(to_i64(&a[0]), 0);
            assert_eq!(to_i64(&a[4]), 4);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_RANGE_STEP / RANGE_WITH_STEP ───────────────────────────────────────

#[test]
fn test_array_range_step_basic() {
    // ARRAY_RANGE_STEP(0, 10, 3) → [0.0, 3.0, 6.0, 9.0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_RANGE_STEP(0, 10, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4, "expected 4 elements [0,3,6,9], got {:?}", a);
            assert!((to_f64(&a[0]) - 0.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 6.0).abs() < 0.001);
            assert!((to_f64(&a[3]) - 9.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_range_with_step_alias() {
    // RANGE_WITH_STEP(1, 6, 2) → [1.0, 3.0, 5.0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = RANGE_WITH_STEP(1, 6, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected [1,3,5], got {:?}", a);
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 3.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 5.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_MOVING_SUM / ROLLING_SUM_ARRAY ─────────────────────────────────────

#[test]
fn test_array_moving_sum_basic() {
    // ARRAY_MOVING_SUM([1,2,3,4,5], 3) → [6.0, 9.0, 12.0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4, 5], "kv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_MOVING_SUM(arr, kv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 windows, got {:?}", a);
            assert!((to_f64(&a[0]) - 6.0).abs() < 0.001, "window 0: expected 6.0, got {}", to_f64(&a[0]));
            assert!((to_f64(&a[1]) - 9.0).abs() < 0.001, "window 1: expected 9.0, got {}", to_f64(&a[1]));
            assert!((to_f64(&a[2]) - 12.0).abs() < 0.001, "window 2: expected 12.0, got {}", to_f64(&a[2]));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_sum_array_alias() {
    // ROLLING_SUM_ARRAY([2,4,6,8], 2) → [6.0, 10.0, 14.0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [2, 4, 6, 8], "kv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ROLLING_SUM_ARRAY(arr, kv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "expected 3 windows");
            assert!((to_f64(&a[0]) - 6.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 10.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 14.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_DOT_PRODUCT ─────────────────────────────────────────────────────────

#[test]
fn test_array_dot_product_basic() {
    // ARRAY_DOT_PRODUCT([1,2,3],[4,5,6]) → 1*4+2*5+3*6 = 32
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [4, 5, 6]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DOT_PRODUCT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => assert!((to_f64(v) - 32.0).abs() < 0.001, "expected 32.0, got {}", to_f64(v)),
        None => panic!("expected a value"),
    }
}

#[test]
fn test_dot_product_alias() {
    // DOT_PRODUCT is already defined for vectors; test ARRAY_DOT_PRODUCT with floats
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1.0, 0.0], "a2": [0.0, 1.0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DOT_PRODUCT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(v) => assert!((to_f64(v) - 0.0).abs() < 0.001, "expected 0.0, got {}", to_f64(v)),
        None => panic!("expected a value"),
    }
}

// ── ARRAY_OUTER_PRODUCT ───────────────────────────────────────────────────────

#[test]
fn test_array_outer_product_basic() {
    // ARRAY_OUTER_PRODUCT([1,2],[3,4]) → [[3,4],[6,8]]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2], "a2": [3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_OUTER_PRODUCT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(outer)) => {
            assert_eq!(outer.len(), 2, "expected 2 rows");
            match &outer[0] {
                Value::Array(row0) => {
                    assert_eq!(row0.len(), 2);
                    assert!((to_f64(&row0[0]) - 3.0).abs() < 0.001, "expected 3.0");
                    assert!((to_f64(&row0[1]) - 4.0).abs() < 0.001, "expected 4.0");
                }
                other => panic!("expected Array row, got {:?}", other),
            }
            match &outer[1] {
                Value::Array(row1) => {
                    assert_eq!(row1.len(), 2);
                    assert!((to_f64(&row1[0]) - 6.0).abs() < 0.001, "expected 6.0");
                    assert!((to_f64(&row1[1]) - 8.0).abs() < 0.001, "expected 8.0");
                }
                other => panic!("expected Array row, got {:?}", other),
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_outer_product_existing_alias() {
    // OUTER_PRODUCT is an existing alias (VECTOR_OUTER_PRODUCT | OUTER_PRODUCT)
    // test it works with arrays
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2], "a2": [3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = OUTER_PRODUCT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(outer)) => {
            assert_eq!(outer.len(), 2);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── ARRAY_SCAN / CUMULATIVE_APPLY ─────────────────────────────────────────────

#[test]
fn test_array_scan_basic() {
    // ARRAY_SCAN([1,2,3,4]) → [1,3,6,10]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SCAN(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4);
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001, "expected 1.0");
            assert!((to_f64(&a[1]) - 3.0).abs() < 0.001, "expected 3.0");
            assert!((to_f64(&a[2]) - 6.0).abs() < 0.001, "expected 6.0");
            assert!((to_f64(&a[3]) - 10.0).abs() < 0.001, "expected 10.0");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_cumulative_apply_alias() {
    // CUMULATIVE_APPLY([5,5,5]) → [5,10,15]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 5, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CUMULATIVE_APPLY(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert!((to_f64(&a[0]) - 5.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 10.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 15.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── Additional edge-case tests ─────────────────────────────────────────────────

#[test]
fn test_array_rotate_empty() {
    // ARRAY_ROTATE([], 3) → []
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [], "nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_ROTATE(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 0),
        other => panic!("expected empty Array, got {:?}", other),
    }
}

#[test]
fn test_array_take_while_none() {
    // [5,6,7] take while < 3 → []
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 6, 7], "thresh": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_TAKE_WHILE(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 0),
        other => panic!("expected empty Array, got {:?}", other),
    }
}

#[test]
fn test_array_drop_while_none() {
    // [5,6,7] drop while < 3 → [5,6,7] (nothing dropped)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [5, 6, 7], "thresh": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_DROP_WHILE(arr, thresh) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 3),
        other => panic!("expected Array of 3, got {:?}", other),
    }
}

#[test]
fn test_set_union_no_duplicates() {
    // SET_UNION([1,2],[1,2]) → [1,2] (no duplicates)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2], "a2": [1, 2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_UNION(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 2, "expected deduped [1,2], got {:?}", a),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_nlargest_zero() {
    // ARRAY_NLARGEST([1,2,3], 0) → []
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3], "nv": 0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_NLARGEST(arr, nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 0),
        other => panic!("expected empty Array, got {:?}", other),
    }
}

#[test]
fn test_array_iota_zero() {
    // ARRAY_IOTA(0) → []
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"nv": 0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_IOTA(nv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 0),
        other => panic!("expected empty Array, got {:?}", other),
    }
}

#[test]
fn test_array_scan_single_element() {
    // ARRAY_SCAN([42]) → [42.0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [42]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_SCAN(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1);
            assert!((to_f64(&a[0]) - 42.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_array_frequencies_strings() {
    // ARRAY_FREQUENCIES of strings
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": ["a", "b", "a", "c", "b", "a"]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_FREQUENCIES(arr) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(m)) => {
            assert_eq!(to_i64(m.get("a").unwrap()), 3);
            assert_eq!(to_i64(m.get("b").unwrap()), 2);
            assert_eq!(to_i64(m.get("c").unwrap()), 1);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_array_moving_sum_window_one() {
    // ARRAY_MOVING_SUM([1,2,3], 1) → [1.0, 2.0, 3.0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3], "kv": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_MOVING_SUM(arr, kv) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3);
            assert!((to_f64(&a[0]) - 1.0).abs() < 0.001);
            assert!((to_f64(&a[1]) - 2.0).abs() < 0.001);
            assert!((to_f64(&a[2]) - 3.0).abs() < 0.001);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_set_is_subset_empty() {
    // [] ⊆ [1,2,3] → true (empty set is subset of any set)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [], "a2": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SET_IS_SUBSET(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Bool(b)) => assert!(*b, "expected true for empty subset"),
        other => panic!("expected Bool, got {:?}", other),
    }
}

#[test]
fn test_array_outer_product_sizes() {
    // ARRAY_OUTER_PRODUCT([1,2,3],[4,5]) → 3x2 matrix
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a1": [1, 2, 3], "a2": [4, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = ARRAY_OUTER_PRODUCT(a1, a2) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(outer)) => {
            assert_eq!(outer.len(), 3, "expected 3 rows");
            for inner_row in outer {
                match inner_row {
                    Value::Array(inner) => assert_eq!(inner.len(), 2, "each row should have 2 elements"),
                    other => panic!("expected inner Array, got {:?}", other),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
