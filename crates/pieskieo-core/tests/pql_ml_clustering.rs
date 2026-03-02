/// Tests for PQL built-in ML clustering and dimensionality reduction functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    (dir, db, ex)
}

fn run(ex: &Executor, q: &str) -> Vec<std::collections::HashMap<String, Value>> {
    let mut p = Parser::new(q);
    let ast = p.parse().expect("parse error");
    let res = ex.execute(ast).expect("execute error");
    res.rows.into_iter().map(|r| r.data).collect()
}

fn get_float(rows: &[std::collections::HashMap<String, Value>], key: &str) -> f64 {
    match rows[0].get(key) {
        Some(Value::Float(f)) => *f,
        Some(Value::Integer(i)) => *i as f64,
        other => panic!("expected float for {}, got {:?}", key, other),
    }
}

fn get_arr(rows: &[std::collections::HashMap<String, Value>], key: &str) -> Vec<Value> {
    match rows[0].get(key) {
        Some(Value::Array(a)) => a.clone(),
        other => panic!("expected array for {}, got {:?}", key, other),
    }
}

// ── KMEANS_LABELS ─────────────────────────────────────────────────────────────

#[test]
fn test_kmeans_labels_basic() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_LABELS([[0.5, 0.5], [1.0, 1.0], [9.5, 9.5], [10.0, 10.0]], [[0.0, 0.0], [10.0, 10.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 4);
    assert_eq!(arr[0], Value::Integer(0));
    assert_eq!(arr[1], Value::Integer(0));
    assert_eq!(arr[2], Value::Integer(1));
    assert_eq!(arr[3], Value::Integer(1));
}

#[test]
fn test_kmeans_labels_single_centroid() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_LABELS([[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0], Value::Integer(0));
    assert_eq!(arr[1], Value::Integer(0));
}

// ── KMEANS_UPDATE_CENTROIDS ───────────────────────────────────────────────────

#[test]
fn test_kmeans_update_centroids() {
    let (_d, _db, ex) = make_db();
    // Two clusters: points [0,0],[1,0] -> cluster 0; [10,10],[11,10] -> cluster 1
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_UPDATE_CENTROIDS([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]], [0, 0, 1, 1], 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    // centroid 0 should be [0.5, 0.0]
    if let Value::Array(c0) = &arr[0] {
        assert!((match &c0[0] { Value::Float(f) => *f, _ => f64::NAN } - 0.5).abs() < 1e-9);
        assert!((match &c0[1] { Value::Float(f) => *f, _ => f64::NAN } - 0.0).abs() < 1e-9);
    } else { panic!("expected array for centroid 0"); }
}

#[test]
fn test_kmeans_centroids_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_CENTROIDS([[0.0], [2.0], [10.0], [12.0]], [0, 0, 1, 1], 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
}

// ── KMEANS_INERTIA ────────────────────────────────────────────────────────────

#[test]
fn test_kmeans_inertia_zero() {
    let (_d, _db, ex) = make_db();
    // Points exactly at centroids -> inertia = 0
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_INERTIA([[0.0, 0.0], [10.0, 10.0]], [[0.0, 0.0], [10.0, 10.0]]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v.abs() < 1e-9, "inertia should be 0 but got {}", v);
}

#[test]
fn test_kmeans_sse_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_SSE([[1.0, 0.0], [3.0, 0.0]], [[2.0, 0.0]]) SELECT res;"#);
    let v = get_float(&rows, "res");
    // distance from [1,0] to [2,0] = 1, dist from [3,0] to [2,0] = 1, SSE = 1+1 = 2
    assert!((v - 2.0).abs() < 1e-9, "expected 2.0, got {}", v);
}

// ── KMEANS_FIT ────────────────────────────────────────────────────────────────

#[test]
fn test_kmeans_fit_returns_object() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_FIT([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]], 2, 10, 42) SELECT res;"#);
    match rows[0].get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("centroids"), "missing centroids");
            assert!(obj.contains_key("labels"), "missing labels");
            assert!(obj.contains_key("inertia"), "missing inertia");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_kmeans_fit_two_clusters() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_FIT([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0], [10.5, 10.0]], 2, 20, 7) SELECT res;"#);
    if let Some(Value::Object(obj)) = rows[0].get("res") {
        if let Some(Value::Array(labels)) = obj.get("labels") {
            assert_eq!(labels.len(), 4);
            // First two should be in same cluster, last two in other
            assert_eq!(labels[0], labels[1]);
            assert_eq!(labels[2], labels[3]);
            assert_ne!(labels[0], labels[2]);
        } else { panic!("no labels"); }
    } else { panic!("expected object"); }
}

// ── ELBOW_INERTIAS ────────────────────────────────────────────────────────────

#[test]
fn test_elbow_inertias_length() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = ELBOW_INERTIAS([[0.0], [1.0], [10.0], [11.0], [20.0]], 4, 42) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 4, "should have 4 inertias for k=1..4");
}

#[test]
fn test_elbow_inertias_decreasing() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = ELBOW_INERTIAS([[0.0], [1.0], [5.0], [10.0], [11.0]], 3, 42) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    // Inertia generally decreases as k increases
    let v0 = match arr[0] { Value::Float(f) => f, _ => f64::INFINITY };
    let v1 = match arr[1] { Value::Float(f) => f, _ => f64::INFINITY };
    assert!(v0 >= v1, "inertia k=1 ({}) should be >= inertia k=2 ({})", v0, v1);
}

#[test]
fn test_kmeans_elbow_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_ELBOW([[0.0, 0.0], [10.0, 10.0]], 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
}

// ── LINKAGE_SINGLE ────────────────────────────────────────────────────────────

#[test]
fn test_linkage_single_3x3() {
    let (_d, _db, ex) = make_db();
    // 3 points with dist matrix
    let rows = run(&ex, r#"QUERY t COMPUTE res = LINKAGE_SINGLE([[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    // Single linkage of 3 points should produce 2 merge steps
    assert_eq!(arr.len(), 2, "expected 2 merge steps for 3 points");
}

#[test]
fn test_single_linkage_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = SINGLE_LINKAGE([[0.0, 2.0], [2.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 1);
    // The one step should merge at distance 2
    if let Value::Object(obj) = &arr[0] {
        let d = match obj.get("distance") { Some(Value::Float(f)) => *f, _ => panic!("no distance") };
        assert!((d - 2.0).abs() < 1e-9, "expected distance 2.0, got {}", d);
    }
}

// ── LINKAGE_COMPLETE ──────────────────────────────────────────────────────────

#[test]
fn test_linkage_complete_3x3() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = LINKAGE_COMPLETE([[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
}

#[test]
fn test_complete_linkage_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = COMPLETE_LINKAGE([[0.0, 3.0], [3.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 1);
}

// ── LINKAGE_AVERAGE ───────────────────────────────────────────────────────────

#[test]
fn test_linkage_average_3x3() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = LINKAGE_AVERAGE([[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
}

#[test]
fn test_average_linkage_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = AVERAGE_LINKAGE([[0.0, 2.0], [2.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 1);
}

// ── DENDROGRAM_HEIGHT ─────────────────────────────────────────────────────────

#[test]
fn test_dendrogram_height_basic() {
    let (_d, _db, ex) = make_db();
    // Build linkage first, then get height for 2 clusters from 3 points
    let rows = run(&ex, r#"QUERY t COMPUTE lnk = LINKAGE_SINGLE([[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]]) COMPUTE h = DENDROGRAM_HEIGHT(lnk, 2) SELECT h;"#);
    let h = get_float(&rows, "h");
    // The first merge is at distance 1.0 (between the two closest points)
    assert!(h >= 0.0, "height should be non-negative");
}

#[test]
fn test_dendro_height_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE lnk = LINKAGE_COMPLETE([[0.0, 2.0], [2.0, 0.0]]) COMPUTE h = DENDRO_HEIGHT(lnk, 1) SELECT h;"#);
    let h = get_float(&rows, "h");
    assert_eq!(h, 2.0);
}

// ── CUT_TREE ──────────────────────────────────────────────────────────────────

#[test]
fn test_cut_tree_2_clusters() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = CUT_TREE([[0.0, 1.0, 10.0], [1.0, 0.0, 10.0], [10.0, 10.0, 0.0]], 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3);
    // First two should be in same cluster, third different
    assert_eq!(arr[0], arr[1], "first two points should be in same cluster");
    assert_ne!(arr[0], arr[2], "third point should be in different cluster");
}

#[test]
fn test_flat_clusters_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = FLAT_CLUSTERS([[0.0, 1.0], [1.0, 0.0]], 1) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0], arr[1], "both points in 1 cluster");
}

// ── Distance metrics ──────────────────────────────────────────────────────────

#[test]
fn test_l2_dist() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = L2_DIST([3.0, 4.0], [0.0, 0.0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 5.0).abs() < 1e-9, "expected 5.0, got {}", v);
}

#[test]
fn test_l1_dist() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = L1_DIST([3.0, 4.0], [0.0, 0.0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 7.0).abs() < 1e-9, "expected 7.0, got {}", v);
}

#[test]
fn test_cos_dist_orthogonal() {
    let (_d, _db, ex) = make_db();
    // [1,0] and [0,1] are orthogonal, cosine similarity = 0, distance = 1
    let rows = run(&ex, r#"QUERY t COMPUTE res = COS_DIST([1.0, 0.0], [0.0, 1.0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {}", v);
}

#[test]
fn test_cos_dist_identical() {
    let (_d, _db, ex) = make_db();
    // Identical vectors -> cosine distance = 0
    let rows = run(&ex, r#"QUERY t COMPUTE res = COS_DIST([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v.abs() < 1e-9, "expected 0.0, got {}", v);
}

#[test]
fn test_linf_dist() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = LINF_DIST([1.0, 5.0, 3.0], [0.0, 0.0, 0.0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 5.0).abs() < 1e-9, "expected 5.0, got {}", v);
}

#[test]
fn test_lp_dist_p2_equals_l2() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = LP_DIST([3.0, 4.0], [0.0, 0.0], 2) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 5.0).abs() < 1e-9, "expected 5.0, got {}", v);
}

#[test]
fn test_lp_dist_p1_equals_l1() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = LP_DIST([3.0, 4.0], [0.0, 0.0], 1) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 7.0).abs() < 1e-9, "expected 7.0, got {}", v);
}

#[test]
fn test_pairwise_distances_euclidean() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = PAIRWISE_DISTANCES([[0.0, 0.0], [3.0, 4.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    if let Value::Array(row0) = &arr[0] {
        // self-distance should be 0
        let d00 = match &row0[0] { Value::Float(f) => *f, _ => f64::NAN };
        assert!(d00.abs() < 1e-9);
        let d01 = match &row0[1] { Value::Float(f) => *f, _ => f64::NAN };
        assert!((d01 - 5.0).abs() < 1e-9, "expected 5.0, got {}", d01);
    }
}

#[test]
fn test_dist_matrix_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = DIST_MATRIX([[0.0], [3.0], [7.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3);
}

#[test]
fn test_hamming_distance_vec() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = HAMMING_DISTANCE_VEC([1, 2, 3, 4], [1, 5, 3, 7]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert_eq!(v as i64, 2, "positions 1 and 3 differ");
}

#[test]
fn test_hamming_vec_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = HAMMING_VEC([1, 1, 0], [0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert_eq!(v as i64, 2);
}

#[test]
fn test_bray_curtis_dist() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = BRAY_CURTIS_DIST([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v.abs() < 1e-9, "identical vectors -> BC = 0, got {}", v);
}

// ── Dimensionality reduction ──────────────────────────────────────────────────

#[test]
fn test_pca_variance_ratio_sums_to_one() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = PCA_VARIANCE_RATIO([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    let total: f64 = arr.iter().map(|v| match v { Value::Float(f) => *f, _ => 0.0 }).sum();
    assert!((total - 1.0).abs() < 1e-9, "variance ratios should sum to 1, got {}", total);
}

#[test]
fn test_pca_var_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = PCA_VAR([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
}

#[test]
fn test_standardize_data_zero_mean() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = STANDARDIZE_DATA([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    // Column means of standardized data should be 0
    let n = arr.len() as f64;
    let col0_mean: f64 = arr.iter().map(|v| match v {
        Value::Array(r) => match r.first() { Some(Value::Float(f)) => *f, _ => 0.0 },
        _ => 0.0,
    }).sum::<f64>() / n;
    assert!(col0_mean.abs() < 1e-9, "mean of standardized col0 should be 0, got {}", col0_mean);
}

#[test]
fn test_zscore_matrix_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = ZSCORE_MATRIX([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3);
}

#[test]
fn test_normalize_data_range_01() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = NORMALIZE_DATA([[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    // First row should be [0.0, 0.0], last row should be [1.0, 1.0]
    if let Value::Array(r) = &arr[0] {
        let v = match r[0] { Value::Float(f) => f, _ => f64::NAN };
        assert!(v.abs() < 1e-9, "min row should start at 0.0, got {}", v);
    }
    if let Value::Array(r) = &arr[2] {
        let v = match r[0] { Value::Float(f) => f, _ => f64::NAN };
        assert!((v - 1.0).abs() < 1e-9, "max row should end at 1.0, got {}", v);
    }
}

#[test]
fn test_minmax_matrix_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = MINMAX_MATRIX([[2.0], [4.0], [6.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3);
}

#[test]
fn test_column_means() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = COLUMN_MEANS([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    let m0 = match arr[0] { Value::Float(f) => f, _ => f64::NAN };
    let m1 = match arr[1] { Value::Float(f) => f, _ => f64::NAN };
    assert!((m0 - 3.0).abs() < 1e-9, "col0 mean = 3.0, got {}", m0);
    assert!((m1 - 4.0).abs() < 1e-9, "col1 mean = 4.0, got {}", m1);
}

#[test]
fn test_feature_means_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = FEATURE_MEANS([[10.0], [20.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    let m = match arr[0] { Value::Float(f) => f, _ => f64::NAN };
    assert!((m - 15.0).abs() < 1e-9, "expected 15.0, got {}", m);
}

#[test]
fn test_column_stds() {
    let (_d, _db, ex) = make_db();
    // std of [1,3,5] = sqrt(8/3) but actually let's use simple: [0,2,4] -> mean=2, var=8/3
    let rows = run(&ex, r#"QUERY t COMPUTE res = COLUMN_STDS([[0.0], [2.0], [4.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    let s = match arr[0] { Value::Float(f) => f, _ => f64::NAN };
    let expected = (8.0_f64 / 3.0).sqrt();
    assert!((s - expected).abs() < 1e-9, "expected {}, got {}", expected, s);
}

#[test]
fn test_feature_stds_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = FEATURE_STDS([[1.0], [1.0], [1.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    let s = match arr[0] { Value::Float(f) => f, _ => f64::NAN };
    assert!(s.abs() < 1e-9, "constant column -> std=0, got {}", s);
}

#[test]
fn test_covariance_matrix_data() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = COVARIANCE_MATRIX_DATA([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2, "should be 2x2");
    if let Value::Array(row0) = &arr[0] { assert_eq!(row0.len(), 2); }
}

#[test]
fn test_cov_mat_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = COV_MAT([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
}

#[test]
fn test_corr_mat_diagonal_ones() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = CORR_MAT([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    // Diagonal should be 1.0
    if let Value::Array(row0) = &arr[0] {
        let d = match row0[0] { Value::Float(f) => f, _ => f64::NAN };
        assert!((d - 1.0).abs() < 1e-9, "diagonal should be 1.0, got {}", d);
    }
}

// ── Nearest neighbors ─────────────────────────────────────────────────────────

#[test]
fn test_knn_distances_sorted() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KNN_DISTANCES([0.0, 0.0], [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]], 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    let d0 = match arr[0] { Value::Float(f) => f, _ => f64::NAN };
    let d1 = match arr[1] { Value::Float(f) => f, _ => f64::NAN };
    assert!(d0 <= d1, "distances should be sorted: {} <= {}", d0, d1);
    assert!((d0 - 1.0).abs() < 1e-9, "nearest is at dist 1, got {}", d0);
}

#[test]
fn test_k_nearest_dist_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = K_NEAREST_DIST([5.0], [[1.0], [3.0], [4.0], [10.0]], 3) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3);
}

#[test]
fn test_knn_indices_correct() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KNN_INDICES([0.0], [[5.0], [1.0], [2.0], [10.0]], 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    // Nearest is index 1 (dist 1), then index 2 (dist 2)
    assert_eq!(arr[0], Value::Integer(1));
    assert_eq!(arr[1], Value::Integer(2));
}

#[test]
fn test_k_nearest_idx_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = K_NEAREST_IDX([0.0, 0.0], [[3.0, 4.0], [1.0, 0.0]], 1) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr[0], Value::Integer(1));
}

#[test]
fn test_knn_classify_integer_labels() {
    let (_d, _db, ex) = make_db();
    // Query near cluster 0
    let rows = run(&ex, r#"QUERY t COMPUTE res = KNN_CLASSIFY([0.5, 0.5], [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]], [0, 0, 1, 1], 2) SELECT res;"#);
    assert_eq!(rows[0].get("res"), Some(&Value::Integer(0)));
}

#[test]
fn test_radius_neighbors_basic() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = RADIUS_NEIGHBORS([0.0, 0.0], [[1.0, 0.0], [0.5, 0.0], [5.0, 0.0]], 1.0) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    // Only points at distance <= 1.0
    assert_eq!(arr.len(), 2, "only 2 points within radius 1.0");
}

#[test]
fn test_within_radius_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = WITHIN_RADIUS([5.0], [[3.0], [4.0], [6.0], [10.0]], 2.0) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3, "points at dist 2, 1, 1 all within radius 2");
}

// ── Evaluation metrics ────────────────────────────────────────────────────────

#[test]
fn test_silhouette_perfect_clusters() {
    let (_d, _db, ex) = make_db();
    // Two well-separated clusters
    let rows = run(&ex, r#"QUERY t COMPUTE res = SILHOUETTE([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v > 0.5, "well-separated clusters should have high silhouette: {}", v);
}

#[test]
fn test_davies_bouldin_perfect() {
    let (_d, _db, ex) = make_db();
    // Two well-separated clusters -> low DB index
    let rows = run(&ex, r#"QUERY t COMPUTE res = DAVIES_BOULDIN_INDEX([[0.0, 0.0], [0.1, 0.0], [100.0, 100.0], [100.1, 100.0]], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v < 1.0, "well-separated clusters should have low DB index: {}", v);
}

#[test]
fn test_db_index_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = DB_INDEX([[0.0], [1.0], [10.0], [11.0]], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v >= 0.0);
}

#[test]
fn test_calinski_harabasz_basic() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = CALINSKI_HARABASZ([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v > 0.0, "CH score should be positive: {}", v);
}

#[test]
fn test_ch_score_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = CH_SCORE([[0.0], [1.0], [10.0], [11.0]], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v > 0.0);
}

#[test]
fn test_adjusted_rand_index_perfect() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = ADJUSTED_RAND_INDEX([0, 0, 1, 1], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 1.0).abs() < 1e-9, "perfect agreement -> ARI=1, got {}", v);
}

#[test]
fn test_ari_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = ARI([0, 1, 0, 1], [0, 1, 0, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!((v - 1.0).abs() < 1e-9);
}

#[test]
fn test_mutual_info_score_basic() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = MUTUAL_INFO_SCORE([0, 0, 1, 1], [0, 0, 1, 1]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v > 0.0, "MI for identical labelings should be positive: {}", v);
}

#[test]
fn test_mi_score_alias() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = MI_SCORE([0, 1, 0, 1], [1, 0, 1, 0]) SELECT res;"#);
    let v = get_float(&rows, "res");
    assert!(v >= 0.0);
}

// ── Additional edge case tests ─────────────────────────────────────────────────

#[test]
fn test_kmeans_assign_existing_works() {
    // Verify the existing KMEANS_ASSIGN (matrix form) still works
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_ASSIGN([[0.5, 0.5], [9.5, 9.5]], [[0.0, 0.0], [10.0, 10.0]]) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0], Value::Integer(0));
    assert_eq!(arr[1], Value::Integer(1));
}

#[test]
fn test_pairwise_distances_manhattan_metric() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = PAIRWISE_DISTANCES([[0.0, 0.0], [3.0, 4.0]], "manhattan") SELECT res;"#);
    let arr = get_arr(&rows, "res");
    if let Value::Array(r0) = &arr[0] {
        let d = match r0[1] { Value::Float(f) => f, _ => f64::NAN };
        assert!((d - 7.0).abs() < 1e-9, "manhattan dist = 7, got {}", d);
    }
}

#[test]
fn test_pairwise_distances_chebyshev_metric() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = PAIRWISE_DISTANCES([[0.0, 0.0], [3.0, 4.0]], "chebyshev") SELECT res;"#);
    let arr = get_arr(&rows, "res");
    if let Value::Array(r0) = &arr[0] {
        let d = match r0[1] { Value::Float(f) => f, _ => f64::NAN };
        assert!((d - 4.0).abs() < 1e-9, "chebyshev dist = 4, got {}", d);
    }
}

#[test]
fn test_kmeans_fit_default_seed() {
    let (_d, _db, ex) = make_db();
    let rows = run(&ex, r#"QUERY t COMPUTE res = KMEANS_FIT([[0.0, 0.0], [10.0, 10.0]], 2) SELECT res;"#);
    match rows[0].get("res") {
        Some(Value::Object(_)) => {}
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_dbscan_cluster_existing_still_works() {
    let (_d, _db, ex) = make_db();
    // Verify DBSCAN still works (existing function)
    let rows = run(&ex, r#"QUERY t COMPUTE res = DBSCAN_CLUSTER([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]], 1.0, 2) SELECT res;"#);
    let arr = get_arr(&rows, "res");
    assert_eq!(arr.len(), 3);
}
