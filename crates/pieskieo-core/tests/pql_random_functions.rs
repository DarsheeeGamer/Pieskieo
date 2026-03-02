/// Integration tests for PQL random/generation/sampling functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db(ns: &str) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), serde_json::json!({})).unwrap();
    (dir, db, ex)
}

#[test]
fn test_random_int_in_range() {
    let (_dir, _db, ex) = make_db("t");
    let mut p = Parser::new(r#"QUERY t COMPUTE r = RANDOM_INT(1, 100) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Integer(n)) => assert!(*n >= 1 && *n <= 100, "expected 1-100, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_rand_int_alias() {
    let (_dir, _db, ex) = make_db("t2");
    let mut p = Parser::new(r#"QUERY t2 COMPUTE r = RAND_INT(0, 9) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Integer(n)) => assert!(*n >= 0 && *n <= 9, "expected 0-9, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_random_float_in_range() {
    let (_dir, _db, ex) = make_db("t3");
    let mut p = Parser::new(r#"QUERY t3 COMPUTE r = RANDOM_FLOAT(0.0, 1.0) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f < 1.0, "expected [0.0, 1.0), got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_random_bool_returns_bool() {
    let (_dir, _db, ex) = make_db("t4");
    let mut p = Parser::new(r#"QUERY t4 COMPUTE b = RANDOM_BOOL() SELECT b;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("b") {
        Some(Value::Bool(_)) => {}
        other => panic!("expected Bool, got {:?}", other),
    }
}

#[test]
fn test_random_bool_probability_one() {
    // probability 1.0 -> always true
    let (_dir, _db, ex) = make_db("t5");
    let mut p = Parser::new(r#"QUERY t5 COMPUTE b = RANDOM_BOOL(1.0) SELECT b;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("b") {
        Some(Value::Bool(true)) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

#[test]
fn test_random_bool_probability_zero() {
    // probability 0.0 -> always false
    let (_dir, _db, ex) = make_db("t6");
    let mut p = Parser::new(r#"QUERY t6 COMPUTE b = RANDOM_BOOL(0.0) SELECT b;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("b") {
        Some(Value::Bool(false)) => {}
        other => panic!("expected Bool(false), got {:?}", other),
    }
}

#[test]
fn test_generate_uuid_is_valid() {
    let (_dir, _db, ex) = make_db("t7");
    let mut p = Parser::new(r#"QUERY t7 COMPUTE u = GENERATE_UUID() SELECT u;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("u") {
        Some(Value::String(s)) => {
            Uuid::parse_str(s).expect("GENERATE_UUID should return a valid UUID string");
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_uuid_alias_is_valid() {
    // RANDOM_UUID is an existing alias that returns Value::Uuid; verify it is a valid UUID type
    let (_dir, _db, ex) = make_db("t8");
    let mut p = Parser::new(r#"QUERY t8 COMPUTE u = RANDOM_UUID() SELECT u;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("u") {
        Some(Value::String(s)) => {
            Uuid::parse_str(s).expect("RANDOM_UUID should return a valid UUID string");
        }
        Some(Value::Uuid(u)) => {
            // The existing arm returns Value::Uuid directly — also valid
            let _ = u; // non-nil UUID is fine
        }
        other => panic!("expected String or Uuid, got {:?}", other),
    }
}

#[test]
fn test_generate_uuid_unique() {
    // Two calls should produce different UUIDs (with overwhelming probability)
    let (_dir, db, ex) = make_db("t9");
    db.put_doc_ns(None, Some("t9"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    let mut p = Parser::new(r#"QUERY t9 COMPUTE u = GENERATE_UUID() SELECT u;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(result.rows.len(), 2);
    let u1 = match result.rows[0].data.get("u") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    let u2 = match result.rows[1].data.get("u") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    assert_ne!(u1, u2, "UUIDs should be unique across rows");
}

#[test]
fn test_random_string_default_charset() {
    let (_dir, _db, ex) = make_db("t10");
    let mut p = Parser::new(r#"QUERY t10 COMPUTE s = RANDOM_STRING(10) SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 10, "expected length 10, got {}", s.len());
            assert!(s.chars().all(|c| c.is_alphanumeric()), "expected alphanumeric chars, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_string_hex_charset() {
    let (_dir, _db, ex) = make_db("t11");
    let mut p = Parser::new(r#"QUERY t11 COMPUTE s = RANDOM_STRING(8, "hex") SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 8, "expected length 8, got {}", s.len());
            assert!(s.chars().all(|c| "0123456789abcdef".contains(c)), "expected hex chars, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_string_numeric_charset() {
    let (_dir, _db, ex) = make_db("t12");
    let mut p = Parser::new(r#"QUERY t12 COMPUTE s = RANDOM_STRING(6, "numeric") SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 6, "expected length 6, got {}", s.len());
            assert!(s.chars().all(|c| c.is_ascii_digit()), "expected digits only, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_string_alpha_charset() {
    let (_dir, _db, ex) = make_db("t13");
    let mut p = Parser::new(r#"QUERY t13 COMPUTE s = RANDOM_STRING(12, "alpha") SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("s") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 12, "expected length 12, got {}", s.len());
            assert!(s.chars().all(|c| c.is_alphabetic()), "expected alpha chars, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_normal_returns_float() {
    let (_dir, _db, ex) = make_db("t14");
    let mut p = Parser::new(r#"QUERY t14 COMPUTE n = RANDOM_NORMAL(0.0, 1.0) SELECT n;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("n") {
        Some(Value::Float(f)) => {
            // For standard normal, values outside [-10, 10] are astronomically rare
            assert!(f.is_finite(), "expected finite float, got {}", f);
            assert!(*f > -15.0 && *f < 15.0, "value unreasonably far from mean: {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_random_exponential_positive() {
    let (_dir, _db, ex) = make_db("t15");
    let mut p = Parser::new(r#"QUERY t15 COMPUTE e = RANDOM_EXPONENTIAL(1.0) SELECT e;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("e") {
        Some(Value::Float(f)) => {
            assert!(*f >= 0.0, "exponential must be non-negative, got {}", f);
            assert!(f.is_finite(), "expected finite value, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_random_poisson_non_negative() {
    let (_dir, _db, ex) = make_db("t16");
    let mut p = Parser::new(r#"QUERY t16 COMPUTE k = RANDOM_POISSON(5.0) SELECT k;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("k") {
        Some(Value::Integer(k)) => {
            assert!(*k >= 0, "Poisson sample must be non-negative, got {}", k);
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_random_bernoulli_zero_or_one() {
    let (_dir, _db, ex) = make_db("t17");
    let mut p = Parser::new(r#"QUERY t17 COMPUTE b = RANDOM_BERNOULLI(0.5) SELECT b;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("b") {
        Some(Value::Integer(n)) => assert!(*n == 0 || *n == 1, "expected 0 or 1, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_random_bernoulli_always_one() {
    let (_dir, _db, ex) = make_db("t18");
    let mut p = Parser::new(r#"QUERY t18 COMPUTE b = RANDOM_BERNOULLI(1.0) SELECT b;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("b") {
        Some(Value::Integer(1)) => {}
        other => panic!("expected Integer(1), got {:?}", other),
    }
}

#[test]
fn test_random_bernoulli_always_zero() {
    let (_dir, _db, ex) = make_db("t19");
    let mut p = Parser::new(r#"QUERY t19 COMPUTE b = RANDOM_BERNOULLI(0.0) SELECT b;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("b") {
        Some(Value::Integer(0)) => {}
        other => panic!("expected Integer(0), got {:?}", other),
    }
}

#[test]
fn test_random_permutation_contains_all() {
    let (_dir, _db, ex) = make_db("t20");
    let mut p = Parser::new(r#"QUERY t20 COMPUTE perm = RANDOM_PERMUTATION(5) SELECT perm;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("perm") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "expected 5 elements");
            let mut nums: Vec<i64> = arr.iter().map(|v| match v {
                Value::Integer(i) => *i,
                other => panic!("expected Integer, got {:?}", other),
            }).collect();
            nums.sort();
            assert_eq!(nums, vec![0, 1, 2, 3, 4], "expected permutation of 0..5, got {:?}", nums);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_shuffle_preserves_elements() {
    let (_dir, db, ex) = make_db("t21");
    db.put_doc_ns(None, Some("t21"), Uuid::new_v4(), serde_json::json!({"arr": [1, 2, 3, 4, 5]})).unwrap();
    // Use the second doc that has the array field
    let mut p = Parser::new(r#"QUERY t21 WHERE arr != null COMPUTE s = SHUFFLE(arr) SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!result.rows.is_empty(), "expected at least one row");
    match result.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "shuffle should preserve length");
            let mut nums: Vec<i64> = arr.iter().map(|v| match v {
                Value::Integer(i) => *i,
                other => panic!("expected Integer, got {:?}", other),
            }).collect();
            nums.sort();
            assert_eq!(nums, vec![1, 2, 3, 4, 5], "shuffle should preserve elements");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_random_choice_from_array() {
    let (_dir, db, ex) = make_db("t22");
    db.put_doc_ns(None, Some("t22"), Uuid::new_v4(), serde_json::json!({"items": ["a", "b", "c"]})).unwrap();
    let mut p = Parser::new(r#"QUERY t22 WHERE items != null COMPUTE c = RANDOM_CHOICE(items) SELECT c;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!result.rows.is_empty(), "expected at least one row");
    match result.rows[0].data.get("c") {
        Some(Value::String(s)) => {
            assert!(["a", "b", "c"].contains(&s.as_str()), "expected one of a/b/c, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_sample_count() {
    let (_dir, db, ex) = make_db("t23");
    db.put_doc_ns(None, Some("t23"), Uuid::new_v4(), serde_json::json!({"nums": [10, 20, 30, 40, 50]})).unwrap();
    let mut p = Parser::new(r#"QUERY t23 WHERE nums != null COMPUTE s = RANDOM_SAMPLE(nums, 3) SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!result.rows.is_empty(), "expected at least one row");
    match result.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "expected 3 samples, got {}", arr.len());
            // All sampled values should come from the original set
            let valid: Vec<i64> = vec![10, 20, 30, 40, 50];
            for v in arr {
                match v {
                    Value::Integer(i) => assert!(valid.contains(i), "unexpected value {}", i),
                    other => panic!("expected Integer, got {:?}", other),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_random_sample_oversized_n() {
    // n >= len -> return full shuffled copy (len elements)
    let (_dir, db, ex) = make_db("t24");
    db.put_doc_ns(None, Some("t24"), Uuid::new_v4(), serde_json::json!({"nums": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t24 WHERE nums != null COMPUTE s = RANDOM_SAMPLE(nums, 100) SELECT s;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!result.rows.is_empty(), "expected at least one row");
    match result.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "expected full array when n > len, got len={}", arr.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_weighted_choice_picks_from_values() {
    let (_dir, db, ex) = make_db("t25");
    db.put_doc_ns(None, Some("t25"), Uuid::new_v4(), serde_json::json!({"vals": ["x", "y", "z"], "wts": [1.0, 1.0, 1.0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t25 WHERE vals != null COMPUTE c = WEIGHTED_CHOICE(vals, wts) SELECT c;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!result.rows.is_empty(), "expected at least one row");
    match result.rows[0].data.get("c") {
        Some(Value::String(s)) => {
            assert!(["x", "y", "z"].contains(&s.as_str()), "expected x/y/z, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_weighted_choice_skewed() {
    // weight of 1000 on "always" vs 0 on "never"
    let (_dir, db, ex) = make_db("t26");
    db.put_doc_ns(None, Some("t26"), Uuid::new_v4(), serde_json::json!({"vals": ["always", "never"], "wts": [1000.0, 0.0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t26 WHERE vals != null COMPUTE c = WEIGHTED_CHOICE(vals, wts) SELECT c;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!result.rows.is_empty(), "expected at least one row");
    match result.rows[0].data.get("c") {
        Some(Value::String(s)) => {
            assert_eq!(s, "always", "expected 'always' with weight 1000 vs 0, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_seeded_random_deterministic() {
    // Same seed should produce same result
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t27"), Uuid::new_v4(), serde_json::json!({})).unwrap();

    let mut p1 = Parser::new(r#"QUERY t27 COMPUTE r = SEEDED_RANDOM(42, 0.0, 1.0) SELECT r;"#);
    let mut p2 = Parser::new(r#"QUERY t27 COMPUTE r = SEEDED_RANDOM(42, 0.0, 1.0) SELECT r;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();

    let v1 = match r1.rows[0].data.get("r") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {:?}", other),
    };
    let v2 = match r2.rows[0].data.get("r") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {:?}", other),
    };
    assert_eq!(v1, v2, "seeded random must be deterministic");
}

#[test]
fn test_seeded_random_in_range() {
    let (_dir, _db, ex) = make_db("t28");
    let mut p = Parser::new(r#"QUERY t28 COMPUTE r = SEEDED_RANDOM(123, 5.0, 10.0) SELECT r;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Float(f)) => {
            assert!(*f >= 5.0 && *f < 10.0, "expected [5.0, 10.0), got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_seeded_random_different_seeds() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t29"), Uuid::new_v4(), serde_json::json!({})).unwrap();

    let mut p1 = Parser::new(r#"QUERY t29 COMPUTE r = SEEDED_RANDOM(1, 0.0, 1000.0) SELECT r;"#);
    let mut p2 = Parser::new(r#"QUERY t29 COMPUTE r = SEEDED_RANDOM(2, 0.0, 1000.0) SELECT r;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();

    let v1 = match r1.rows[0].data.get("r") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {:?}", other),
    };
    let v2 = match r2.rows[0].data.get("r") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {:?}", other),
    };
    assert_ne!(v1, v2, "different seeds should produce different values");
}
