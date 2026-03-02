/// Integration tests for PQL vector arithmetic functions.
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

#[test]
fn test_vector_dot_product() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = VECTOR_DOT(a, b) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Float(32.0)));
}

#[test]
fn test_cosine_similarity() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 0.0], "b": [0.0, 1.0]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE sim = COSINE_SIMILARITY(a, b) SELECT sim;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // orthogonal vectors -> similarity = 0
    match r.rows[0].data.get("sim") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-9, "orthogonal cosine sim should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_vector_magnitude() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"v": [3.0, 4.0]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE mag = VECTOR_MAGNITUDE(v) SELECT mag;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // sqrt(9 + 16) = 5
    assert_eq!(r.rows[0].data.get("mag"), Some(&Value::Float(5.0)));
}

#[test]
fn test_vector_normalize() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"v": [3.0, 4.0]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE n = VECTOR_NORMALIZE(v) COMPUTE mag = VECTOR_MAGNITUDE(n) SELECT mag;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mag") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-9, "normalized vector should have magnitude 1, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_euclidean_distance() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [0.0, 0.0], "b": [3.0, 4.0]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE d = EUCLIDEAN_DISTANCE(a, b) SELECT d;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Float(5.0)));
}

#[test]
fn test_vector_add_and_scale() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": [1.0, 2.0], "b": [3.0, 4.0]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE s = VECTOR_ADD(a, b) COMPUTE dim = VECTOR_DIM(s) SELECT dim;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dim"), Some(&Value::Integer(2)));
}
