/// Integration tests for PQL clustering and ML model functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db(ns: &str, doc: serde_json::Value) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    (dir, db, ex)
}

// ── KMEANS_ASSIGN ─────────────────────────────────────────────────────────────

#[test]
fn test_kmeans_assign_near_first_centroid() {
    // Point [0.1, 0.1] is much closer to centroid [0,0] than [10,10]
    let (_dir, _db, ex) = make_db(
        "t",
        serde_json::json!({
            "pt": [0.1, 0.1],
            "centroids": [[0.0, 0.0], [10.0, 10.0]]
        }),
    );
    let mut p = Parser::new(r#"QUERY t COMPUTE c = KMEANS_ASSIGN(pt, centroids) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("c"),
        Some(&Value::Integer(0)),
        "point [0.1,0.1] should be assigned to centroid 0"
    );
}

#[test]
fn test_kmeans_assign_near_second_centroid() {
    // Point [9.9, 9.9] is much closer to centroid [10,10] than [0,0]
    let (_dir, _db, ex) = make_db(
        "t2",
        serde_json::json!({
            "pt": [9.9, 9.9],
            "centroids": [[0.0, 0.0], [10.0, 10.0]]
        }),
    );
    let mut p = Parser::new(r#"QUERY t2 COMPUTE c = CLUSTER_ASSIGN(pt, centroids) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("c"),
        Some(&Value::Integer(1)),
        "point [9.9,9.9] should be assigned to centroid 1"
    );
}

// ── KMEANS_ITERATE ────────────────────────────────────────────────────────────

#[test]
fn test_kmeans_iterate_returns_object_with_correct_keys() {
    // Two clear clusters: [0,0],[0.1,0] and [10,10],[9.9,10]
    let (_dir, _db, ex) = make_db(
        "km",
        serde_json::json!({
            "data": [[0.0,0.0],[0.1,0.0],[10.0,10.0],[9.9,10.0]],
            "k": 2,
            "seed": 1,
            "iters": 20
        }),
    );
    let mut p = Parser::new(
        r#"QUERY km COMPUTE result = KMEANS_ITERATE(data, k, seed, iters) SELECT result;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let obj = match r.rows[0].data.get("result") {
        Some(Value::Object(o)) => o,
        other => panic!("expected Object, got {:?}", other),
    };
    assert!(obj.contains_key("centroids"), "result must contain 'centroids'");
    assert!(obj.contains_key("assignments"), "result must contain 'assignments'");
    assert!(obj.contains_key("inertia"), "result must contain 'inertia'");

    // assignments should be an array of 4 elements
    match obj.get("assignments") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 4, "should have 4 assignments"),
        other => panic!("expected Array for assignments, got {:?}", other),
    }

    // inertia should be a non-negative float
    match obj.get("inertia") {
        Some(Value::Float(f)) => assert!(*f >= 0.0, "inertia must be non-negative"),
        other => panic!("expected Float for inertia, got {:?}", other),
    }
}

#[test]
fn test_kmeans_iterate_two_clusters_well_separated() {
    // With two well-separated clusters, inertia should be small
    let (_dir, _db, ex) = make_db(
        "km2",
        serde_json::json!({
            "data": [
                [0.0, 0.0], [0.0, 0.1], [0.1, 0.0],
                [100.0, 100.0], [100.1, 100.0], [100.0, 100.1]
            ],
            "k": 2,
            "seed": 42,
            "iters": 15
        }),
    );
    let mut p = Parser::new(
        r#"QUERY km2 COMPUTE result = KMEANS_STEP(data, k, seed, iters) SELECT result;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let obj = match r.rows[0].data.get("result") {
        Some(Value::Object(o)) => o,
        other => panic!("expected Object, got {:?}", other),
    };
    // With well-separated data, inertia should be small relative to separation
    match obj.get("inertia") {
        Some(Value::Float(f)) => {
            assert!(*f < 1.0, "inertia should be small for well-separated clusters, got {}", f)
        }
        other => panic!("expected Float for inertia, got {:?}", other),
    }
}

// ── CLUSTER_SILHOUETTE ────────────────────────────────────────────────────────

#[test]
fn test_silhouette_well_separated_clusters() {
    // Two tight clusters far apart should have high silhouette score (close to 1)
    let (_dir, _db, ex) = make_db(
        "sil",
        serde_json::json!({
            "data": [
                [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]
            ],
            "asgn": [0, 0, 0, 1, 1, 1]
        }),
    );
    let mut p = Parser::new(
        r#"QUERY sil COMPUTE s = CLUSTER_SILHOUETTE(data, asgn) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => {
            assert!(
                *f > 0.9,
                "well-separated clusters should have silhouette > 0.9, got {}",
                f
            )
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_silhouette_alias() {
    // SILHOUETTE_SCORE is an alias for CLUSTER_SILHOUETTE
    let (_dir, _db, ex) = make_db(
        "sil2",
        serde_json::json!({
            "data": [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
            "asgn": [0, 0, 1, 1]
        }),
    );
    let mut p =
        Parser::new(r#"QUERY sil2 COMPUTE s = SILHOUETTE_SCORE(data, asgn) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "silhouette should be positive, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── DBSCAN_CLUSTER ────────────────────────────────────────────────────────────

#[test]
fn test_dbscan_two_clusters_and_noise() {
    // Cluster A: points near (0,0), Cluster B: points near (10,10), noise: (5,5)
    let (_dir, _db, ex) = make_db(
        "db",
        serde_json::json!({
            "data": [
                [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],   // cluster A
                [10.0, 10.0], [10.1, 10.0], [10.0, 10.1], // cluster B
                [5.0, 5.0]  // noise
            ],
            "eps": 0.5,
            "minpts": 2
        }),
    );
    let mut p = Parser::new(
        r#"QUERY db COMPUTE labels = DBSCAN_CLUSTER(data, eps, minpts) SELECT labels;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let labels = match r.rows[0].data.get("labels") {
        Some(Value::Array(a)) => a.clone(),
        other => panic!("expected Array, got {:?}", other),
    };
    assert_eq!(labels.len(), 7, "should have 7 labels");

    // The noise point (index 6) should be -1
    assert_eq!(
        labels[6],
        Value::Integer(-1),
        "isolated point should be noise (-1), got {:?}",
        labels[6]
    );

    // Cluster A points (0,1,2) should all share the same label (not -1)
    let la0 = match &labels[0] { Value::Integer(i) => *i, _ => panic!("not integer") };
    let la1 = match &labels[1] { Value::Integer(i) => *i, _ => panic!("not integer") };
    let la2 = match &labels[2] { Value::Integer(i) => *i, _ => panic!("not integer") };
    assert!(la0 >= 0, "cluster A point 0 should not be noise");
    assert_eq!(la0, la1, "cluster A points should share a label");
    assert_eq!(la0, la2, "cluster A points should share a label");

    // Cluster B points (3,4,5) should all share the same label, different from A
    let lb3 = match &labels[3] { Value::Integer(i) => *i, _ => panic!("not integer") };
    let lb4 = match &labels[4] { Value::Integer(i) => *i, _ => panic!("not integer") };
    let lb5 = match &labels[5] { Value::Integer(i) => *i, _ => panic!("not integer") };
    assert!(lb3 >= 0, "cluster B point 3 should not be noise");
    assert_eq!(lb3, lb4, "cluster B points should share a label");
    assert_eq!(lb3, lb5, "cluster B points should share a label");
    assert_ne!(la0, lb3, "cluster A and B should have different labels");
}

// ── ELBOW_SCORE / INERTIA_KMEANS ─────────────────────────────────────────────

#[test]
fn test_elbow_score_decreases_with_k() {
    // Inertia should decrease (or stay same) as k increases
    let (_dir, _db, ex) = make_db(
        "elbow",
        serde_json::json!({
            "data": [
                [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                [5.0, 5.0], [5.1, 5.0],
                [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]
            ],
            "k1": 1,
            "k2": 2,
            "k3": 3,
            "seed": 7
        }),
    );
    let mut p = Parser::new(
        r#"QUERY elbow
           COMPUTE e1 = ELBOW_SCORE(data, k1, seed)
           COMPUTE e2 = ELBOW_SCORE(data, k2, seed)
           COMPUTE e3 = INERTIA_KMEANS(data, k3, seed)
           SELECT e1, e2, e3;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let e1 = match r.rows[0].data.get("e1") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for e1, got {:?}", other),
    };
    let e2 = match r.rows[0].data.get("e2") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for e2, got {:?}", other),
    };
    let e3 = match r.rows[0].data.get("e3") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for e3, got {:?}", other),
    };
    assert!(e1 >= e2, "inertia(k=1) >= inertia(k=2), got {} vs {}", e1, e2);
    assert!(e2 >= e3, "inertia(k=2) >= inertia(k=3), got {} vs {}", e2, e3);
}

// ── KNN_PREDICT ───────────────────────────────────────────────────────────────

#[test]
fn test_knn_predict_classification() {
    // Training: class 0 near (0,0), class 1 near (10,10)
    // Query: (0.2, 0.2) should predict class 0
    let (_dir, _db, ex) = make_db(
        "knn",
        serde_json::json!({
            "train_x": [[0.0,0.0],[0.1,0.0],[0.0,0.1],[10.0,10.0],[9.9,10.0],[10.0,9.9]],
            "train_y": [0, 0, 0, 1, 1, 1],
            "qpt": [0.2, 0.2],
            "k": 3
        }),
    );
    let mut p = Parser::new(
        r#"QUERY knn COMPUTE pred = KNN_PREDICT(train_x, train_y, qpt, k) SELECT pred;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("pred"),
        Some(&Value::Integer(0)),
        "query near (0,0) should predict class 0"
    );
}

#[test]
fn test_knn_predict_regression() {
    // Training: y = x[0] * 2 (approximately)
    // Query at x=5 should predict around 10
    let (_dir, _db, ex) = make_db(
        "knnr",
        serde_json::json!({
            "train_x": [[1.0],[2.0],[3.0],[4.0],[6.0],[7.0]],
            "train_y": [2.0, 4.0, 6.0, 8.0, 12.0, 14.0],
            "qpt": [5.0],
            "k": 2
        }),
    );
    let mut p = Parser::new(
        r#"QUERY knnr COMPUTE pred = K_NEAREST_PREDICT(train_x, train_y, qpt, k) SELECT pred;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pred") {
        Some(Value::Float(f)) => {
            assert!(
                (*f - 10.0).abs() < 2.0,
                "regression near x=5 should predict ~10, got {}",
                f
            )
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── ANOMALY_SCORE ─────────────────────────────────────────────────────────────

#[test]
fn test_anomaly_score_outlier_is_higher() {
    // A point far from the cluster should have a higher anomaly score
    // than a point inside the cluster.
    let (_dir, _db, ex) = make_db(
        "anom",
        serde_json::json!({
            "data": [
                [0.0,0.0],[0.1,0.0],[0.0,0.1],[0.2,0.1],[0.1,0.2],
                [0.3,0.0],[0.0,0.3]
            ],
            "inlier": [0.15, 0.1],
            "outlier": [50.0, 50.0]
        }),
    );
    let mut p = Parser::new(
        r#"QUERY anom
           COMPUTE s_in  = ANOMALY_SCORE(data, inlier)
           COMPUTE s_out = ISOLATION_SCORE(data, outlier)
           SELECT s_in, s_out;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let s_in = match r.rows[0].data.get("s_in") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for s_in, got {:?}", other),
    };
    let s_out = match r.rows[0].data.get("s_out") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for s_out, got {:?}", other),
    };
    assert!(
        s_out > s_in,
        "outlier score ({}) should exceed inlier score ({})",
        s_out,
        s_in
    );
    assert!(
        s_out >= 0.0 && s_out <= 1.0,
        "anomaly score must be in [0,1], got {}",
        s_out
    );
}

// ── PAIRWISE_COSINE ───────────────────────────────────────────────────────────

#[test]
fn test_pairwise_cosine_diagonal_is_one() {
    // cosine_similarity(v, v) == 1 for all non-zero v
    let (_dir, _db, ex) = make_db(
        "cos",
        serde_json::json!({
            "data": [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]
            ]
        }),
    );
    let mut p =
        Parser::new(r#"QUERY cos COMPUTE m = PAIRWISE_COSINE(data) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let matrix = match r.rows[0].data.get("m") {
        Some(Value::Array(a)) => a.clone(),
        other => panic!("expected Array, got {:?}", other),
    };
    assert_eq!(matrix.len(), 3, "3x3 matrix expected");
    for (i, row_val) in matrix.iter().enumerate() {
        let row = match row_val {
            Value::Array(r) => r,
            other => panic!("expected Array row, got {:?}", other),
        };
        assert_eq!(row.len(), 3, "each row should have 3 elements");
        match &row[i] {
            Value::Float(f) => assert!(
                (*f - 1.0).abs() < 1e-9,
                "diagonal M[{i},{i}] should be 1.0, got {}",
                f
            ),
            other => panic!("expected Float at diagonal [{i},{i}], got {:?}", other),
        }
    }
}

#[test]
fn test_pairwise_cosine_orthogonal_vectors() {
    // cosine_similarity([1,0], [0,1]) == 0
    let (_dir, _db, ex) = make_db(
        "cos2",
        serde_json::json!({
            "data": [[1.0, 0.0], [0.0, 1.0]]
        }),
    );
    let mut p =
        Parser::new(r#"QUERY cos2 COMPUTE m = COSINE_MATRIX(data) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let matrix = match r.rows[0].data.get("m") {
        Some(Value::Array(a)) => a.clone(),
        other => panic!("expected Array, got {:?}", other),
    };
    // M[0][1] should be 0 (orthogonal)
    let row0 = match &matrix[0] {
        Value::Array(r) => r,
        other => panic!("expected Array row 0, got {:?}", other),
    };
    match &row0[1] {
        Value::Float(f) => assert!(
            f.abs() < 1e-9,
            "cosine similarity of orthogonal vectors should be 0, got {}",
            f
        ),
        other => panic!("expected Float at [0][1], got {:?}", other),
    }
}
