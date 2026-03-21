/// Integration tests for PQL probabilistic data structure functions.
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

#[test]
fn test_bloom_contains_present() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": ["apple", "banana", "cherry"], "q": "apple"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE found = BLOOM_CONTAINS(items, q, 0.01) SELECT found;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("found"),
        Some(&Value::Bool(true)),
        "apple should be found in the Bloom filter"
    );
}

#[test]
fn test_bloom_contains_absent() {
    // Use a large set to minimize false positives, then query a value that is definitely absent.
    let (db, ex) = setup();
    // Build a set of 100 items, none of which is "zzznothere999"
    let items: Vec<String> = (0..100).map(|i| format!("item{}", i)).collect();
    let json_items = serde_json::Value::Array(
        items
            .iter()
            .map(|s| serde_json::Value::String(s.clone()))
            .collect(),
    );
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": json_items, "q": "zzznothere999"}),
    )
    .unwrap();
    // With n_bits = max(64, 100*10) = 1000, probability of false positive is very low
    let mut p =
        Parser::new(r#"QUERY t COMPUTE found = BLOOM_CONTAINS(items, q, 0.01) SELECT found;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Should be false (absent) given the large bit array
    assert_eq!(
        r.rows[0].data.get("found"),
        Some(&Value::Bool(false)),
        "zzznothere999 should not be in the Bloom filter"
    );
}

#[test]
fn test_bloom_build_returns_object() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": ["a", "b", "c"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bf = BLOOM_BUILD(items, 0.01) SELECT bf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bf") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("bits"), "bloom object should have 'bits'");
            assert!(
                obj.contains_key("n_bits"),
                "bloom object should have 'n_bits'"
            );
            assert!(
                obj.contains_key("n_hashes"),
                "bloom object should have 'n_hashes'"
            );
            // n_hashes should be 3
            assert_eq!(obj.get("n_hashes"), Some(&Value::Integer(3)));
        }
        other => panic!("Expected Object from BLOOM_BUILD, got {:?}", other),
    }
}

#[test]
fn test_hll_count_small() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": ["a", "b", "c", "a", "b", "d"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = HLL_COUNT(items) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Exact distinct count for small array: a, b, c, d = 4
    assert_eq!(
        r.rows[0].data.get("cnt"),
        Some(&Value::Integer(4)),
        "HLL_COUNT should return 4 distinct elements"
    );
}

#[test]
fn test_count_min_sketch_frequency() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"stream": ["apple", "banana", "apple", "cherry", "apple"], "q": "apple"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE freq = COUNT_MIN_SKETCH(stream, q) SELECT freq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("freq") {
        Some(Value::Integer(v)) => {
            assert!(*v >= 3, "apple frequency should be >= 3, got {}", v);
        }
        other => panic!("Expected Integer from COUNT_MIN_SKETCH, got {:?}", other),
    }
}

#[test]
fn test_reservoir_sample_size() {
    let (db, ex) = setup();
    let items: Vec<i64> = (0..100).collect();
    let json_items = serde_json::Value::Array(
        items
            .iter()
            .map(|i| serde_json::Value::Number((*i).into()))
            .collect(),
    );
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": json_items}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE sample = RESERVOIR_SAMPLE(items, 10, 42) SELECT sample;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sample") {
        Some(Value::Array(arr)) => {
            assert_eq!(
                arr.len(),
                10,
                "Reservoir sample should have exactly 10 elements"
            );
        }
        other => panic!("Expected Array from RESERVOIR_SAMPLE, got {:?}", other),
    }
}

#[test]
fn test_minhash_same_set() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({
            "s1": ["apple", "banana", "cherry"],
            "s2": ["apple", "banana", "cherry"]
        }),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sim = MINHASH(s1, s2, 128) SELECT sim;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sim") {
        Some(Value::Float(v)) => {
            assert!(
                (*v - 1.0).abs() < 1e-9,
                "MinHash similarity of identical sets should be 1.0, got {}",
                v
            );
        }
        other => panic!("Expected Float from MINHASH, got {:?}", other),
    }
}

#[test]
fn test_minhash_disjoint_sets() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({
            "s1": ["a", "b", "c"],
            "s2": ["x", "y", "z"]
        }),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sim = MINHASH(s1, s2, 128) SELECT sim;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sim") {
        Some(Value::Float(v)) => {
            // Disjoint sets should have very low similarity
            assert!(
                *v < 0.1,
                "MinHash similarity of disjoint sets should be near 0.0, got {}",
                v
            );
        }
        other => panic!("Expected Float from MINHASH, got {:?}", other),
    }
}

#[test]
fn test_top_k_frequent() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": ["a", "b", "a", "c", "a", "b"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE top = TOP_K_FREQUENT(items, 2) SELECT top;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("top") {
        Some(Value::Array(arr)) => {
            assert_eq!(
                arr.len(),
                2,
                "TOP_K_FREQUENT(items, 2) should return 2 elements"
            );
            // First element should be "a" with count 3
            if let Value::Object(obj) = &arr[0] {
                assert_eq!(obj.get("value"), Some(&Value::String("a".to_string())));
                assert_eq!(obj.get("count"), Some(&Value::Integer(3)));
            } else {
                panic!("Expected Object in result array, got {:?}", arr[0]);
            }
        }
        other => panic!("Expected Array from TOP_K_FREQUENT, got {:?}", other),
    }
}

#[test]
fn test_sketch_percentile() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": [1, 2, 3, 4, 5]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE p50 = SKETCH_PERCENTILE(items, 0.5) SELECT p50;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p50") {
        Some(Value::Float(v)) => {
            assert!(
                (*v - 3.0).abs() < 0.01,
                "50th percentile of [1,2,3,4,5] should be ~3.0, got {}",
                v
            );
        }
        other => panic!("Expected Float from SKETCH_PERCENTILE, got {:?}", other),
    }
}

#[test]
fn test_prob_equal_identical_numbers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v1": 5.0, "v2": 5.0, "tol": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE p = PROB_EQUAL(v1, v2, tol) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(v)) => {
            assert!(
                (*v - 1.0).abs() < 1e-9,
                "PROB_EQUAL of identical values should be 1.0, got {}",
                v
            );
        }
        other => panic!("Expected Float from PROB_EQUAL, got {:?}", other),
    }
}

#[test]
fn test_prob_equal_strings() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello", "s2": "hello", "tol": 0.1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE p = PROB_EQUAL(s1, s2, tol) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Float(v)) => {
            assert!(
                (*v - 1.0).abs() < 1e-9,
                "PROB_EQUAL of identical strings should be 1.0, got {}",
                v
            );
        }
        other => panic!("Expected Float from PROB_EQUAL, got {:?}", other),
    }
}

#[test]
fn test_random_projection_output_dim() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vec": [1.0, 2.0, 3.0, 4.0, 5.0]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE proj = RANDOM_PROJECTION(vec, 3, 42) SELECT proj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("proj") {
        Some(Value::Array(arr)) => {
            assert_eq!(
                arr.len(),
                3,
                "RANDOM_PROJECTION to dim=3 should return 3 elements"
            );
        }
        other => panic!("Expected Array from RANDOM_PROJECTION, got {:?}", other),
    }
}

#[test]
fn test_reservoir_sample_smaller_than_k() {
    // When array length <= k, return all elements
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": [10, 20, 30]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = RESERVOIR_SAMPLE(items, 10, 1) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "Should return all 3 elements when k > len");
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_approx_distinct_alias() {
    // APPROX_DISTINCT is an alias for HLL_COUNT
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": [1, 2, 2, 3, 3, 3]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cnt = APPROX_DISTINCT(items) SELECT cnt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("cnt"),
        Some(&Value::Integer(3)),
        "APPROX_DISTINCT should return 3 distinct values"
    );
}
