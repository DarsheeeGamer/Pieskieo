/// Integration tests for additional PQL vector and embedding utility functions.
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

fn f32_close(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.01
}

// ── VECTOR_SUBTRACT ──────────────────────────────────────────────────────────

#[test]
fn test_vector_subtract() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [5.0, 3.0, 1.0], "b": [1.0, 2.0, 3.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_SUBTRACT(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 4.0), "expected 4.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 1.0), "expected 1.0 got {}", v[1]);
            assert!(f32_close(v[2] as f64, -2.0), "expected -2.0 got {}", v[2]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn test_vec_sub_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [10.0, 5.0], "b": [3.0, 2.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VEC_SUB(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 7.0), "expected 7.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 3.0), "expected 3.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── VECTOR_MULTIPLY (Hadamard) ────────────────────────────────────────────────

#[test]
fn test_vector_multiply_hadamard() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_MULTIPLY(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // [2*5, 3*6, 4*7] = [10, 18, 28]
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 10.0), "expected 10.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 18.0), "expected 18.0 got {}", v[1]);
            assert!(f32_close(v[2] as f64, 28.0), "expected 28.0 got {}", v[2]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn test_vector_hadamard_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_HADAMARD(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 3.0), "expected 3.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 8.0), "expected 8.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── VECTOR_ABS ───────────────────────────────────────────────────────────────

#[test]
fn test_vector_abs() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [-3.0, 4.0, -1.5, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_ABS(v) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 3.0), "expected 3.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 4.0), "expected 4.0 got {}", v[1]);
            assert!(f32_close(v[2] as f64, 1.5), "expected 1.5 got {}", v[2]);
            assert!(f32_close(v[3] as f64, 0.0), "expected 0.0 got {}", v[3]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── Norm functions ────────────────────────────────────────────────────────────

#[test]
fn test_euclidean_norm() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [3.0, 4.0]}),
    )
    .unwrap();
    // sqrt(9 + 16) = 5
    let mut p = Parser::new("QUERY t COMPUTE n = EUCLIDEAN_NORM(v) SELECT n;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 5.0), "expected 5.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_l1_norm() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [-1.0, 2.0, -3.0, 4.0]}),
    )
    .unwrap();
    // |−1| + |2| + |−3| + |4| = 10
    let mut p = Parser::new("QUERY t COMPUTE n = L1_NORM(v) SELECT n;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 10.0), "expected 10.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_manhattan_norm_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [1.0, 1.0, 1.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE n = MANHATTAN_NORM(v) SELECT n;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 3.0), "expected 3.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_linf_norm() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [1.0, -5.0, 3.0, -2.0]}),
    )
    .unwrap();
    // max(|1|, |−5|, |3|, |−2|) = 5
    let mut p = Parser::new("QUERY t COMPUTE n = LINF_NORM(v) SELECT n;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 5.0), "expected 5.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_chebyshev_norm_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [0.0, 7.0, -3.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE n = CHEBYSHEV_NORM(v) SELECT n;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("n") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 7.0), "expected 7.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── VECTOR_MEAN / VECTOR_SUM ──────────────────────────────────────────────────

#[test]
fn test_vector_mean() {
    let (db, ex) = setup();
    // Store three vectors as separate docs, then use a single doc with array of arrays
    // The function takes an Array of Vectors/Arrays.
    // We pass a literal array by using a JSON field containing nested arrays.
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vecs": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE m = VECTOR_MEAN(vecs) SELECT m;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // mean = [3.0, 4.0, 5.0]
    match result.rows[0].data.get("m") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 3.0), "expected 3.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 4.0), "expected 4.0 got {}", v[1]);
            assert!(f32_close(v[2] as f64, 5.0), "expected 5.0 got {}", v[2]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn test_vector_sum_aggregate() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vecs": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE s = VECTOR_SUM(vecs) SELECT s;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // sum = [2.0, 2.0]
    match result.rows[0].data.get("s") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 2.0), "expected 2.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 2.0), "expected 2.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── JACCARD_DISTANCE ──────────────────────────────────────────────────────────

#[test]
fn test_jaccard_distance() {
    let (db, ex) = setup();
    // [1,0,1,1] vs [0,1,1,1]
    // intersection: positions 2,3 (both nonzero) = 2
    // union: positions 0,1,2,3 (at least one nonzero) = 4
    // jaccard distance = 1 - 2/4 = 0.5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 0.0, 1.0, 1.0], "b": [0.0, 1.0, 1.0, 1.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE j = JACCARD_DISTANCE(a, b) SELECT j;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 0.5), "expected 0.5 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_jaccard_identical_vectors() {
    let (db, ex) = setup();
    // identical binary vectors -> distance = 0
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 1.0, 0.0], "b": [1.0, 1.0, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE j = JACCARD_DISTANCE(a, b) SELECT j;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 0.0), "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── CHEBYSHEV_DISTANCE ────────────────────────────────────────────────────────

#[test]
fn test_chebyshev_distance() {
    let (db, ex) = setup();
    // [1,2,3] vs [4,0,3] -> |1-4|=3, |2-0|=2, |3-3|=0 -> max = 3
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [4.0, 0.0, 3.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = CHEBYSHEV_DISTANCE(a, b) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 3.0), "expected 3.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_linf_distance_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [0.0, 0.0], "b": [3.0, 4.0]}),
    )
    .unwrap();
    // max(|0-3|, |0-4|) = 4
    let mut p = Parser::new("QUERY t COMPUTE d = LINF_DISTANCE(a, b) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 4.0), "expected 4.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── MINKOWSKI_DISTANCE ────────────────────────────────────────────────────────

#[test]
fn test_minkowski_distance_p1_equals_manhattan() {
    let (db, ex) = setup();
    // p=1 should equal Manhattan distance
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
    )
    .unwrap();
    // |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
    let mut p = Parser::new("QUERY t COMPUTE d = MINKOWSKI_DISTANCE(a, b, 1) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 9.0), "expected 9.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_minkowski_distance_p2_equals_euclidean() {
    let (db, ex) = setup();
    // p=2 should equal Euclidean distance
    // [0,0] vs [3,4] -> sqrt(9+16) = 5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [0.0, 0.0], "b": [3.0, 4.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = MINKOWSKI_DISTANCE(a, b, 2) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 5.0), "expected 5.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── BRAY_CURTIS_DISSIMILARITY ─────────────────────────────────────────────────

#[test]
fn test_bray_curtis_dissimilarity() {
    let (db, ex) = setup();
    // [1,2,3] vs [4,5,6]
    // numerator = |1-4| + |2-5| + |3-6| = 9
    // denominator = |1+4| + |2+5| + |3+6| = 5 + 7 + 9 = 21
    // result = 9/21 ≈ 0.4286
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = BRAY_CURTIS_DISSIMILARITY(a, b) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 9.0 / 21.0), "expected ~0.4286 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_bray_curtis_alias() {
    let (db, ex) = setup();
    // identical vectors -> dissimilarity = 0
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [2.0, 4.0], "b": [2.0, 4.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = BRAY_CURTIS(a, b) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 0.0), "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── ANGULAR_DISTANCE / COSINE_DISTANCE ───────────────────────────────────────

#[test]
fn test_angular_distance_identical() {
    let (db, ex) = setup();
    // identical vectors -> cosine distance = 0
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = ANGULAR_DISTANCE(a, b) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 0.0), "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_cosine_distance_orthogonal() {
    let (db, ex) = setup();
    // orthogonal vectors -> cosine distance = 1 - 0 = 1
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 0.0], "b": [0.0, 1.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = COSINE_DISTANCE(a, b) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 1.0), "expected 1.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── VECTOR_PROJECTION ────────────────────────────────────────────────────────

#[test]
fn test_vector_projection() {
    let (db, ex) = setup();
    // project [2,3] onto [1,0] (x-axis) -> [2,0]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [2.0, 3.0], "b": [1.0, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_PROJECTION(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 2.0), "expected 2.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 0.0), "expected 0.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn test_vec_proj_alias() {
    let (db, ex) = setup();
    // project [3,4] onto [1,0] -> [3,0]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [3.0, 4.0], "b": [1.0, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VEC_PROJ(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 3.0), "expected 3.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 0.0), "expected 0.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── VECTOR_REJECTION ─────────────────────────────────────────────────────────

#[test]
fn test_vector_rejection() {
    let (db, ex) = setup();
    // reject [2,3] from [1,0] (x-axis)
    // proj = [2,0], rejection = [2,3] - [2,0] = [0,3]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [2.0, 3.0], "b": [1.0, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_REJECTION(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 0.0), "expected 0.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 3.0), "expected 3.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── VECTOR_OUTER_PRODUCT ──────────────────────────────────────────────────────

#[test]
fn test_vector_outer_product() {
    let (db, ex) = setup();
    // [1,2] outer [3,4] = [[1*3, 1*4], [2*3, 2*4]] = [[3,4],[6,8]]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE m = VECTOR_OUTER_PRODUCT(a, b) SELECT m;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("m") {
        Some(Value::Array(rows)) => {
            assert_eq!(rows.len(), 2, "outer product should have 2 rows");
            match &rows[0] {
                Value::Array(r0) => {
                    assert!(
                        f32_close(
                            match &r0[0] {
                                Value::Float(f) => *f,
                                _ => panic!(),
                            },
                            3.0
                        ),
                        "expected 3.0"
                    );
                    assert!(
                        f32_close(
                            match &r0[1] {
                                Value::Float(f) => *f,
                                _ => panic!(),
                            },
                            4.0
                        ),
                        "expected 4.0"
                    );
                }
                _ => panic!("expected Array row"),
            }
            match &rows[1] {
                Value::Array(r1) => {
                    assert!(
                        f32_close(
                            match &r1[0] {
                                Value::Float(f) => *f,
                                _ => panic!(),
                            },
                            6.0
                        ),
                        "expected 6.0"
                    );
                    assert!(
                        f32_close(
                            match &r1[1] {
                                Value::Float(f) => *f,
                                _ => panic!(),
                            },
                            8.0
                        ),
                        "expected 8.0"
                    );
                }
                _ => panic!("expected Array row"),
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── VECTOR_CLAMP ──────────────────────────────────────────────────────────────

#[test]
fn test_vector_clamp() {
    let (db, ex) = setup();
    // clamp [-5, 0, 5, 10] to [-2, 7]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [-5.0, 0.0, 5.0, 10.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_CLAMP(v, -2.0, 7.0) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, -2.0), "expected -2.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 0.0), "expected 0.0 got {}", v[1]);
            assert!(f32_close(v[2] as f64, 5.0), "expected 5.0 got {}", v[2]);
            assert!(f32_close(v[3] as f64, 7.0), "expected 7.0 got {}", v[3]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn test_vec_clamp_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [0.0, 0.5, 1.5, 2.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VEC_CLAMP(v, 0.0, 1.0) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 0.0), "expected 0.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 0.5), "expected 0.5 got {}", v[1]);
            assert!(f32_close(v[2] as f64, 1.0), "expected 1.0 got {}", v[2]);
            assert!(f32_close(v[3] as f64, 1.0), "expected 1.0 got {}", v[3]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── VECTOR_QUANTIZE ───────────────────────────────────────────────────────────

#[test]
fn test_vector_quantize() {
    let (db, ex) = setup();
    // quantize to 4 levels: step = 0.25, round 0.3 -> 0.25, 0.7 -> 0.75, 0.5 -> 0.5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [0.3, 0.7, 0.5, 0.1]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VECTOR_QUANTIZE(v, 4) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            // 0.3 / 0.25 = 1.2 -> round to 1 -> 0.25
            assert!(f32_close(v[0] as f64, 0.25), "expected 0.25 got {}", v[0]);
            // 0.7 / 0.25 = 2.8 -> round to 3 -> 0.75
            assert!(f32_close(v[1] as f64, 0.75), "expected 0.75 got {}", v[1]);
            // 0.5 / 0.25 = 2.0 -> round to 2 -> 0.50
            assert!(f32_close(v[2] as f64, 0.5), "expected 0.5 got {}", v[2]);
            // 0.1 / 0.25 = 0.4 -> round to 0 -> 0.0
            assert!(f32_close(v[3] as f64, 0.0), "expected 0.0 got {}", v[3]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

#[test]
fn test_vec_quantize_alias() {
    let (db, ex) = setup();
    // 2 levels: step = 0.5, so values snap to nearest 0.5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v": [0.0, 0.3, 0.6, 1.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VEC_QUANTIZE(v, 2) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            // 0.0 / 0.5 = 0.0 -> round 0 -> 0.0
            assert!(f32_close(v[0] as f64, 0.0), "expected 0.0 got {}", v[0]);
            // 0.3 / 0.5 = 0.6 -> round 1 -> 0.5
            assert!(f32_close(v[1] as f64, 0.5), "expected 0.5 got {}", v[1]);
            // 0.6 / 0.5 = 1.2 -> round 1 -> 0.5
            assert!(f32_close(v[2] as f64, 0.5), "expected 0.5 got {}", v[2]);
            // 1.0 / 0.5 = 2.0 -> round 2 -> 1.0
            assert!(f32_close(v[3] as f64, 1.0), "expected 1.0 got {}", v[3]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── JACCARD_METRIC alias ──────────────────────────────────────────────────────

#[test]
fn test_jaccard_metric_alias() {
    let (db, ex) = setup();
    // all zeros vs all zeros -> both sets empty, no union -> distance = 0
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE j = JACCARD_METRIC(a, b) SELECT j;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 0.0), "expected 0.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── VEC_MEAN alias ────────────────────────────────────────────────────────────

#[test]
fn test_vec_mean_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vecs": [[0.0, 0.0], [2.0, 4.0]]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE m = VEC_MEAN(vecs) SELECT m;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    // mean = [1.0, 2.0]
    match result.rows[0].data.get("m") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 1.0), "expected 1.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 2.0), "expected 2.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}

// ── LP_DISTANCE alias ─────────────────────────────────────────────────────────

#[test]
fn test_lp_distance_alias() {
    let (db, ex) = setup();
    // same as p=2 test: [0,0] vs [3,4] -> 5.0
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [0.0, 0.0], "b": [3.0, 4.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = LP_DISTANCE(a, b, 2) SELECT d;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f32_close(*f, 5.0), "expected 5.0 got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── VEC_REJ alias ─────────────────────────────────────────────────────────────

#[test]
fn test_vec_rej_alias() {
    let (db, ex) = setup();
    // same geometry as rejection test: [2,3] reject from [1,0] -> [0,3]
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": [2.0, 3.0], "b": [1.0, 0.0]}),
    )
    .unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = VEC_REJ(a, b) SELECT r;");
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("r") {
        Some(Value::Vector(v)) => {
            assert!(f32_close(v[0] as f64, 0.0), "expected 0.0 got {}", v[0]);
            assert!(f32_close(v[1] as f64, 3.0), "expected 3.0 got {}", v[1]);
        }
        other => panic!("expected Vector, got {:?}", other),
    }
}
