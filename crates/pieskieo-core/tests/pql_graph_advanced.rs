/// Integration tests for PQL advanced graph algorithm functions.
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

// ── NODE_DEGREE ───────────────────────────────────────────────────────────────

#[test]
fn test_node_degree_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": [1, 2, 3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NODE_DEGREE(nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(3)));
}

#[test]
fn test_node_degree_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": []})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NODE_DEGREE(nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(0)));
}

#[test]
fn test_node_degree_single() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": [42]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NODE_DEGREE(nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(1)));
}

// ── GRAPH_DENSITY_CALC / DENSITY_FROM_COUNTS ─────────────────────────────────

#[test]
fn test_graph_density_calc_basic() {
    let (db, ex) = setup();
    // 2E / (N*(N-1)) = 2*3 / (4*3) = 6/12 = 0.5
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 3, "nv": 4})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_DENSITY_CALC(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.5).abs() < 1e-9, "expected 0.5, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_density_from_counts_alias() {
    let (db, ex) = setup();
    // 2*0 / (3*2) = 0.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 0, "nv": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DENSITY_FROM_COUNTS(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f).abs() < 1e-9, "expected 0.0, got {}", f),
        other => panic!("expected Float(0.0), got {:?}", other),
    }
}

#[test]
fn test_graph_density_calc_single_node() {
    // n <= 1 -> 0.0
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 5, "nv": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_DENSITY_CALC(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(0.0)));
}

// ── GRAPH_TRIANGLE_COUNT / TRIANGLE_COUNT ────────────────────────────────────

#[test]
fn test_graph_triangle_count_basic() {
    let (db, ex) = setup();
    // [1,2,3] and [2,3,4] share 2 and 3 -> 2
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_TRIANGLE_COUNT(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(2)));
}

#[test]
fn test_triangle_count_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TRIANGLE_COUNT(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(2)));
}

#[test]
fn test_graph_triangle_count_no_overlap() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_TRIANGLE_COUNT(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(0)));
}

// ── LOCAL_CLUSTERING_COEFF / LCC ─────────────────────────────────────────────

#[test]
fn test_local_clustering_coeff_basic() {
    let (db, ex) = setup();
    // degree=4, triangles=2 -> 2 / (4*3/2) = 2/6 ≈ 0.333...
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 4, "tri": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOCAL_CLUSTERING_COEFF(deg, tri) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 2.0 / 6.0).abs() < 1e-9, "expected ~0.333, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_lcc_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 4, "tri": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LCC(deg, tri) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 2.0 / 6.0).abs() < 1e-9, "expected ~0.333, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_local_clustering_coeff_single_neighbor() {
    // degree=1 -> 0.0 (no triangles possible)
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 1, "tri": 0})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOCAL_CLUSTERING_COEFF(deg, tri) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(0.0)));
}

// ── MUTUAL_NEIGHBORS / COMMON_FRIENDS_COUNT ──────────────────────────────────

#[test]
fn test_mutual_neighbors_basic() {
    let (db, ex) = setup();
    // [1,2,3] ∩ [2,3,4] = {2,3} -> 2
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MUTUAL_NEIGHBORS(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(2)));
}

#[test]
fn test_common_friends_count_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COMMON_FRIENDS_COUNT(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(2)));
}

#[test]
fn test_mutual_neighbors_disjoint() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MUTUAL_NEIGHBORS(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(0)));
}

// ── JACCARD_GRAPH / LINK_JACCARD ─────────────────────────────────────────────

#[test]
fn test_jaccard_graph_basic() {
    let (db, ex) = setup();
    // [1,2,3] and [2,3,4]: |{2,3}| / |{1,2,3,4}| = 2/4 = 0.5
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JACCARD_GRAPH(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.5).abs() < 1e-9, "expected 0.5, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_link_jaccard_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LINK_JACCARD(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.5).abs() < 1e-9, "expected 0.5, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_jaccard_graph_both_empty() {
    // Both empty -> union=0 -> by convention return 1.0
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [], "nb": []})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JACCARD_GRAPH(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(1.0)));
}

// ── NODE_STRENGTH / WEIGHTED_DEGREE ──────────────────────────────────────────

#[test]
fn test_node_strength_basic() {
    let (db, ex) = setup();
    // 1.5 + 2.5 + 3.0 = 7.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"wts": [1.5, 2.5, 3.0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NODE_STRENGTH(wts) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 7.0).abs() < 1e-9, "expected 7.0, got {}", f),
        other => panic!("expected Float(7.0), got {:?}", other),
    }
}

#[test]
fn test_weighted_degree_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"wts": [1.5, 2.5, 3.0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = WEIGHTED_DEGREE(wts) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 7.0).abs() < 1e-9, "expected 7.0, got {}", f),
        other => panic!("expected Float(7.0), got {:?}", other),
    }
}

#[test]
fn test_node_strength_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"wts": []})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NODE_STRENGTH(wts) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f).abs() < 1e-9, "expected 0.0, got {}", f),
        other => panic!("expected Float(0.0), got {:?}", other),
    }
}

// ── BETWEENNESS_APPROX / APPROX_BETWEENNESS ──────────────────────────────────

#[test]
fn test_betweenness_approx_basic() {
    let (db, ex) = setup();
    // deg=2, total=5 -> 2 / (5*4) = 2/20 = 0.1
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 2, "nv": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BETWEENNESS_APPROX(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.1).abs() < 1e-9, "expected 0.1, got {}", f),
        other => panic!("expected Float(0.1), got {:?}", other),
    }
}

#[test]
fn test_approx_betweenness_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 2, "nv": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = APPROX_BETWEENNESS(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.1).abs() < 1e-9, "expected 0.1, got {}", f),
        other => panic!("expected Float(0.1), got {:?}", other),
    }
}

#[test]
fn test_betweenness_approx_small_graph() {
    // n <= 2 -> 0.0
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 1, "nv": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BETWEENNESS_APPROX(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(0.0)));
}

// ── GRAPH_ASSORTATIV / DEGREE_ASSORTATIV ─────────────────────────────────────

#[test]
fn test_graph_assortativ_perfectly_correlated() {
    let (db, ex) = setup();
    // xs = ys -> Pearson r = 1.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3,4], "nb": [1,2,3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_ASSORTATIV(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 1.0).abs() < 1e-9, "expected 1.0, got {}", f),
        other => panic!("expected Float(1.0), got {:?}", other),
    }
}

#[test]
fn test_degree_assortativ_alias_uncorrelated() {
    let (db, ex) = setup();
    // perfectly negatively correlated -> r = -1.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3,4], "nb": [4,3,2,1]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DEGREE_ASSORTATIV(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - (-1.0)).abs() < 1e-9, "expected -1.0, got {}", f),
        other => panic!("expected Float(-1.0), got {:?}", other),
    }
}

#[test]
fn test_graph_assortativ_too_short() {
    // n < 2 -> 0.0
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [5], "nb": [5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_ASSORTATIV(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(0.0)));
}

// ── TWO_HOP_COUNT ─────────────────────────────────────────────────────────────

#[test]
fn test_two_hop_count_basic() {
    let (db, ex) = setup();
    // [3,4,5] -> sum = 12
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": [3,4,5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TWO_HOP_COUNT(nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(12)));
}

#[test]
fn test_two_hop_count_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": []})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TWO_HOP_COUNT(nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(0)));
}

// ── NEIGHBOR_OVERLAP / OVERLAP_COEFF ─────────────────────────────────────────

#[test]
fn test_neighbor_overlap_full() {
    let (db, ex) = setup();
    // [1,2] ∩ [1,2,3] = {1,2}, min(2,3)=2 -> 2/2=1.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [1,2,3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NEIGHBOR_OVERLAP(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 1.0).abs() < 1e-9, "expected 1.0, got {}", f),
        other => panic!("expected Float(1.0), got {:?}", other),
    }
}

#[test]
fn test_overlap_coeff_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [1,2,3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = OVERLAP_COEFF(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 1.0).abs() < 1e-9, "expected 1.0, got {}", f),
        other => panic!("expected Float(1.0), got {:?}", other),
    }
}

#[test]
fn test_neighbor_overlap_empty() {
    // min_size = 0 -> 0.0
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [], "nb": [1,2]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NEIGHBOR_OVERLAP(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(0.0)));
}

// ── DICE_COEFF / DICE_SIMILARITY ─────────────────────────────────────────────

#[test]
fn test_dice_coeff_basic() {
    let (db, ex) = setup();
    // [1,2] ∩ [2,3] = {2}, 2*1 / (2+2) = 2/4 = 0.5
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [2,3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DICE_COEFF(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.5).abs() < 1e-9, "expected 0.5, got {}", f),
        other => panic!("expected Float(0.5), got {:?}", other),
    }
}

#[test]
fn test_dice_similarity_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [2,3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DICE_SIMILARITY(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 0.5).abs() < 1e-9, "expected 0.5, got {}", f),
        other => panic!("expected Float(0.5), got {:?}", other),
    }
}

#[test]
fn test_dice_coeff_both_empty() {
    // denom = 0 -> 1.0 by convention
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [], "nb": []})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DICE_COEFF(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Float(1.0)));
}

// ── Additional edge case tests ────────────────────────────────────────────────

#[test]
fn test_graph_triangle_count_full_overlap() {
    let (db, ex) = setup();
    // [1,2,3] ∩ [1,2,3] = 3
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2,3], "nb": [1,2,3]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_TRIANGLE_COUNT(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(3)));
}

#[test]
fn test_jaccard_graph_disjoint() {
    let (db, ex) = setup();
    // [1,2] and [3,4]: intersection=0, union=4 -> 0.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"na": [1,2], "nb": [3,4]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JACCARD_GRAPH(na, nb) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f).abs() < 1e-9, "expected 0.0, got {}", f),
        other => panic!("expected Float(0.0), got {:?}", other),
    }
}

#[test]
fn test_node_strength_integer_weights() {
    let (db, ex) = setup();
    // [2, 3, 5] -> 10.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"wts": [2, 3, 5]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NODE_STRENGTH(wts) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 10.0).abs() < 1e-9, "expected 10.0, got {}", f),
        other => panic!("expected Float(10.0), got {:?}", other),
    }
}

#[test]
fn test_two_hop_count_single_element() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": [7]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TWO_HOP_COUNT(nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Integer(7)));
}

#[test]
fn test_local_clustering_coeff_full() {
    let (db, ex) = setup();
    // degree=3, triangles=3 -> 3/(3*2/2)=3/3=1.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 3, "tri": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOCAL_CLUSTERING_COEFF(deg, tri) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 1.0).abs() < 1e-9, "expected 1.0, got {}", f),
        other => panic!("expected Float(1.0), got {:?}", other),
    }
}

#[test]
fn test_graph_density_calc_complete_graph() {
    let (db, ex) = setup();
    // Complete graph K4: edges=6, nodes=4 -> 2*6/(4*3)=12/12=1.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 6, "nv": 4})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_DENSITY_CALC(deg, nv) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((f - 1.0).abs() < 1e-9, "expected 1.0, got {}", f),
        other => panic!("expected Float(1.0), got {:?}", other),
    }
}
