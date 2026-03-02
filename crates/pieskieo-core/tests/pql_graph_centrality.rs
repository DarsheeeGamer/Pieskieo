/// Integration tests for advanced graph centrality and path analysis functions.
use pieskieo_core::{pql::{Executor, Parser, Value}, PieskieoDb};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

// ── GRAPH_BFS_DISTANCE ────────────────────────────────────────────────────────

#[test]
fn test_graph_bfs_distance_same_node() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("nodes"), id, serde_json::json!({"name": "A"})).unwrap();

    let query = format!(
        r#"QUERY nodes COMPUTE d = GRAPH_BFS_DISTANCE("nodes", "{}", "{}") SELECT d;"#,
        id, id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(0)), "same node distance should be 0");
}

#[test]
fn test_graph_bfs_distance_adjacent() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("nodes"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("nodes"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY nodes WHERE name = "A" COMPUTE d = GRAPH_BFS_DISTANCE("nodes", "{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(1)), "adjacent nodes should be distance 1");
}

#[test]
fn test_graph_bfs_distance_two_hops() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("path2"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("path2"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("path2"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY path2 WHERE name = "A" COMPUTE d = GRAPH_BFS_DISTANCE("path2", "{}", "{}") SELECT d;"#,
        a, c
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(2)), "two-hop path should be distance 2");
}

#[test]
fn test_graph_bfs_distance_unreachable() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("unreach"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("unreach"), b, serde_json::json!({"name": "B"})).unwrap();
    // No edges

    let query = format!(
        r#"QUERY unreach WHERE name = "A" COMPUTE d = GRAPH_BFS_DISTANCE("unreach", "{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(-1)), "unreachable should return -1");
}

#[test]
fn test_bfs_distance_alias() {
    // BFS_DISTANCE is already implemented under GRAPH_SHORTEST_PATH_LENGTH,
    // test that GRAPH_BFS_DISTANCE is the new alias
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("alias_bfs"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("alias_bfs"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY alias_bfs WHERE name = "A" COMPUTE d = GRAPH_BFS_DIST("alias_bfs", "{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(1)));
}

// ── NODE_ECCENTRICITY ─────────────────────────────────────────────────────────

#[test]
fn test_node_eccentricity_star_hub() {
    // Star graph: hub connects to 3 leaves, eccentricity of hub = 1
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let hub = Uuid::new_v4();
    let l1 = Uuid::new_v4();
    let l2 = Uuid::new_v4();
    let l3 = Uuid::new_v4();
    db.put_doc_ns(None, Some("star"), hub, serde_json::json!({"role": "hub"})).unwrap();
    db.put_doc_ns(None, Some("star"), l1, serde_json::json!({"role": "leaf"})).unwrap();
    db.put_doc_ns(None, Some("star"), l2, serde_json::json!({"role": "leaf"})).unwrap();
    db.put_doc_ns(None, Some("star"), l3, serde_json::json!({"role": "leaf"})).unwrap();
    db.add_edge(hub, l1, 1.0).unwrap();
    db.add_edge(hub, l2, 1.0).unwrap();
    db.add_edge(hub, l3, 1.0).unwrap();

    let query = format!(
        r#"QUERY star WHERE role = "hub" COMPUTE e = NODE_ECCENTRICITY("star", "{}") SELECT e;"#,
        hub
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("e") {
        Some(Value::Integer(n)) => assert_eq!(*n, 1, "hub eccentricity should be 1, got {}", n),
        other => panic!("expected Integer(1), got {:?}", other),
    }
}

#[test]
fn test_node_eccentricity_leaf_in_path() {
    // Path A-B-C: eccentricity of A = 2 (farthest node is C)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("pathg"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("pathg"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("pathg"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY pathg WHERE name = "A" COMPUTE e = NODE_ECCENTRICITY("pathg", "{}") SELECT e;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("e") {
        Some(Value::Integer(n)) => assert_eq!(*n, 2, "leaf eccentricity in path A-B-C should be 2, got {}", n),
        other => panic!("expected Integer(2), got {:?}", other),
    }
}

#[test]
fn test_node_eccentricity_isolated_returns_null() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("iso"), id, serde_json::json!({"name": "solo"})).unwrap();

    let query = format!(
        r#"QUERY iso COMPUTE e = NODE_ECCENTRICITY("iso", "{}") SELECT e;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("e"), Some(&Value::Null), "isolated node eccentricity should be Null");
}

// ── COMMON_NEIGHBORS / SHARED_NEIGHBORS ──────────────────────────────────────

#[test]
fn test_common_neighbors_shared() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let shared = Uuid::new_v4();
    db.put_doc_ns(None, Some("cn"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("cn"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("cn"), shared, serde_json::json!({"name": "shared"})).unwrap();
    db.add_edge(a, shared, 1.0).unwrap();
    db.add_edge(b, shared, 1.0).unwrap();

    let query = format!(
        r#"QUERY cn WHERE name = "A" COMPUTE c = COMMON_NEIGHBORS("{}", "{}") SELECT c;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("c") {
        Some(Value::Integer(n)) => assert!(*n >= 1, "should have at least 1 common neighbor, got {}", n),
        other => panic!("expected Integer >= 1, got {:?}", other),
    }
}

#[test]
fn test_common_neighbors_unrelated() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("noedge"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("noedge"), b, serde_json::json!({"name": "B"})).unwrap();

    let query = format!(
        r#"QUERY noedge WHERE name = "A" COMPUTE c = COMMON_NEIGHBORS("{}", "{}") SELECT c;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(0)));
}

#[test]
fn test_shared_neighbors_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let shared = Uuid::new_v4();
    db.put_doc_ns(None, Some("sn"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("sn"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("sn"), shared, serde_json::json!({"name": "S"})).unwrap();
    db.add_edge(a, shared, 1.0).unwrap();
    db.add_edge(b, shared, 1.0).unwrap();

    let query = format!(
        r#"QUERY sn WHERE name = "A" COMPUTE c = SHARED_NEIGHBORS("{}", "{}") SELECT c;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("c") {
        Some(Value::Integer(n)) => assert!(*n >= 1, "SHARED_NEIGHBORS alias should work, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── ADAMIC_ADAR / ADAMIC_ADAR_SCORE ──────────────────────────────────────────

#[test]
fn test_adamic_adar_score_alias_positive() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let mid = Uuid::new_v4();
    let x = Uuid::new_v4(); // extra neighbor for mid so degree > 1
    db.put_doc_ns(None, Some("aa"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("aa"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("aa"), mid, serde_json::json!({"name": "mid"})).unwrap();
    db.put_doc_ns(None, Some("aa"), x, serde_json::json!({"name": "X"})).unwrap();
    db.add_edge(a, mid, 1.0).unwrap();
    db.add_edge(b, mid, 1.0).unwrap();
    db.add_edge(mid, x, 1.0).unwrap(); // degree of mid is now > 1

    let query = format!(
        r#"QUERY aa WHERE name = "A" COMPUTE score = ADAMIC_ADAR_SCORE("{}", "{}") SELECT score;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("score") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "ADAMIC_ADAR_SCORE should be positive with a common neighbor of degree > 1, got {}", f),
        other => panic!("expected positive Float, got {:?}", other),
    }
}

#[test]
fn test_adamic_adar_score_no_common_neighbors() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("aa2"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("aa2"), b, serde_json::json!({"name": "B"})).unwrap();

    let query = format!(
        r#"QUERY aa2 WHERE name = "A" COMPUTE score = ADAMIC_ADAR_SCORE("{}", "{}") SELECT score;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("score"), Some(&Value::Float(0.0)));
}

// ── JACCARD_SIMILARITY_GRAPH / GRAPH_JACCARD ─────────────────────────────────

#[test]
fn test_jaccard_similarity_graph_no_neighbors() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("jac"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("jac"), b, serde_json::json!({"name": "B"})).unwrap();

    let query = format!(
        r#"QUERY jac WHERE name = "A" COMPUTE j = JACCARD_SIMILARITY_GRAPH("{}", "{}") SELECT j;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("j"), Some(&Value::Float(0.0)));
}

#[test]
fn test_jaccard_similarity_graph_identical_neighbors() {
    // Two nodes sharing the exact same neighbors → Jaccard = 1.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("jac2"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("jac2"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("jac2"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, c, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    // a and b both have only c as neighbor

    let query = format!(
        r#"QUERY jac2 WHERE name = "A" COMPUTE j = JACCARD_SIMILARITY_GRAPH("{}", "{}") SELECT j;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(*f > 0.0 && *f <= 1.0, "Jaccard with shared neighbor should be in (0,1], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_graph_jaccard_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("jac3"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("jac3"), b, serde_json::json!({"name": "B"})).unwrap();

    let query = format!(
        r#"QUERY jac3 WHERE name = "A" COMPUTE j = GRAPH_JACCARD("{}", "{}") SELECT j;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("j") {
        Some(Value::Float(_)) => {}
        other => panic!("GRAPH_JACCARD alias expected Float, got {:?}", other),
    }
}

// ── KATZ_CENTRALITY / KATZ_SCORE ─────────────────────────────────────────────

#[test]
fn test_katz_centrality_hub_higher_than_leaf() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let hub = Uuid::new_v4();
    let leaf = Uuid::new_v4();
    let l2 = Uuid::new_v4();
    let l3 = Uuid::new_v4();
    db.put_doc_ns(None, Some("katz"), hub, serde_json::json!({"role": "hub"})).unwrap();
    db.put_doc_ns(None, Some("katz"), leaf, serde_json::json!({"role": "leaf"})).unwrap();
    db.put_doc_ns(None, Some("katz"), l2, serde_json::json!({"role": "leaf"})).unwrap();
    db.put_doc_ns(None, Some("katz"), l3, serde_json::json!({"role": "leaf"})).unwrap();
    db.add_edge(hub, leaf, 1.0).unwrap();
    db.add_edge(hub, l2, 1.0).unwrap();
    db.add_edge(hub, l3, 1.0).unwrap();

    let query_hub = format!(
        r#"QUERY katz WHERE role = "hub" COMPUTE k = KATZ_CENTRALITY("{}") SELECT k;"#,
        hub
    );
    let query_leaf = format!(
        r#"QUERY katz WHERE role = "leaf" COMPUTE k = KATZ_CENTRALITY("{}") SELECT k;"#,
        leaf
    );
    let mut p_hub = Parser::new(&query_hub);
    let r_hub = ex.execute(p_hub.parse().unwrap()).unwrap();
    let mut p_leaf = Parser::new(&query_leaf);
    let r_leaf = ex.execute(p_leaf.parse().unwrap()).unwrap();

    assert!(!r_hub.rows.is_empty());
    assert!(!r_leaf.rows.is_empty());

    let hub_score = match r_hub.rows[0].data.get("k") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for hub, got {:?}", other),
    };
    let leaf_score = match r_leaf.rows[0].data.get("k") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for leaf, got {:?}", other),
    };
    assert!(hub_score > leaf_score, "hub Katz centrality {} should exceed leaf {}", hub_score, leaf_score);
}

#[test]
fn test_katz_score_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("ks"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("ks"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY ks WHERE name = "A" COMPUTE k = KATZ_SCORE("{}") SELECT k;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("k") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "KATZ_SCORE alias should return positive value, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── HARMONIC_CENTRALITY / HARMONIC_CENTRAL ───────────────────────────────────

#[test]
fn test_harmonic_centrality_connected_node() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("hc"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("hc"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("hc"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY hc WHERE name = "A" COMPUTE h = HARMONIC_CENTRALITY("{}") SELECT h;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "harmonic centrality should be positive for connected node, got {}", f),
        other => panic!("expected positive Float, got {:?}", other),
    }
}

#[test]
fn test_harmonic_centrality_isolated_node() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("hciso"), id, serde_json::json!({"name": "solo"})).unwrap();

    let query = format!(
        r#"QUERY hciso COMPUTE h = HARMONIC_CENTRALITY("{}") SELECT h;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::Float(0.0)), "isolated node harmonic centrality should be 0.0");
}

#[test]
fn test_harmonic_central_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("hca"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("hca"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY hca WHERE name = "A" COMPUTE h = HARMONIC_CENTRAL("{}") SELECT h;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("h") {
        Some(Value::Float(_)) => {}
        other => panic!("HARMONIC_CENTRAL alias expected Float, got {:?}", other),
    }
}

// ── GRAPH_DFS_REACHABLE / DFS_REACHABLE ──────────────────────────────────────

#[test]
fn test_graph_dfs_reachable_path() {
    // Path A-B-C: from A, 2 nodes reachable (B and C)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("dfs"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("dfs"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("dfs"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY dfs WHERE name = "A" COMPUTE n = GRAPH_DFS_REACHABLE("{}") SELECT n;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("n") {
        Some(Value::Integer(n)) => assert!(*n >= 2, "should reach at least 2 nodes from A in A-B-C, got {}", n),
        other => panic!("expected Integer >= 2, got {:?}", other),
    }
}

#[test]
fn test_graph_dfs_reachable_isolated() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("dfsiso"), id, serde_json::json!({"name": "solo"})).unwrap();

    let query = format!(
        r#"QUERY dfsiso COMPUTE n = GRAPH_DFS_REACHABLE("{}") SELECT n;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(0)), "isolated node should reach 0 other nodes");
}

#[test]
fn test_dfs_reachable_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("dfsa"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("dfsa"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY dfsa WHERE name = "A" COMPUTE n = DFS_REACHABLE("{}") SELECT n;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("n") {
        Some(Value::Integer(n)) => assert!(*n >= 1, "DFS_REACHABLE alias should return >= 1, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── SHORTEST_PATH_LEN / SPL ───────────────────────────────────────────────────

#[test]
fn test_shortest_path_len_same_node() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("spl"), id, serde_json::json!({"name": "A"})).unwrap();

    let query = format!(
        r#"QUERY spl COMPUTE d = SHORTEST_PATH_LEN("{}", "{}") SELECT d;"#,
        id, id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(0)), "same node SPL should be 0");
}

#[test]
fn test_shortest_path_len_connected() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("splc"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("splc"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY splc WHERE name = "A" COMPUTE d = SHORTEST_PATH_LEN("{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(1)), "adjacent nodes SPL should be 1");
}

#[test]
fn test_spl_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("spla"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("spla"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY spla WHERE name = "A" COMPUTE d = SPL("{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(1)), "SPL alias should work");
}

#[test]
fn test_shortest_path_len_unreachable() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("splun"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("splun"), b, serde_json::json!({"name": "B"})).unwrap();

    let query = format!(
        r#"QUERY splun WHERE name = "A" COMPUTE d = SHORTEST_PATH_LEN("{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("d"), Some(&Value::Integer(-1)), "unreachable SPL should return -1");
}

// ── RANDOM_WALK / GRAPH_WALK ──────────────────────────────────────────────────

#[test]
fn test_random_walk_returns_array() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("walk"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("walk"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("walk"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    db.add_edge(c, a, 1.0).unwrap();

    let query = format!(
        r#"QUERY walk WHERE name = "A" COMPUTE w = RANDOM_WALK("{}", 3) SELECT w;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("w") {
        Some(Value::Array(arr)) => {
            assert!(!arr.is_empty(), "walk should contain at least start node");
            assert!(arr.len() <= 4, "walk of 3 steps should have at most 4 elements, got {}", arr.len());
            // First element should be the start node UUID string
            match &arr[0] {
                Value::String(s) => assert_eq!(s, &a.to_string(), "first element should be start node"),
                other => panic!("expected String UUID, got {:?}", other),
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_random_walk_isolated_node() {
    // Isolated node: walk should just be [start]
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("walkiso"), id, serde_json::json!({"name": "solo"})).unwrap();

    let query = format!(
        r#"QUERY walkiso COMPUTE w = RANDOM_WALK("{}", 5) SELECT w;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("w") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1, "isolated node walk should have only start node, got {}", arr.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_graph_walk_alias_length() {
    // GRAPH_WALK alias: walk length <= n_steps + 1
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("gwa"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("gwa"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, a, 1.0).unwrap();

    let n_steps = 4usize;
    let query = format!(
        r#"QUERY gwa WHERE name = "A" COMPUTE w = GRAPH_WALK("{}", {}) SELECT w;"#,
        a, n_steps
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("w") {
        Some(Value::Array(arr)) => {
            assert!(arr.len() <= n_steps + 1, "GRAPH_WALK of {} steps should have at most {} elements, got {}", n_steps, n_steps + 1, arr.len());
            assert!(!arr.is_empty(), "walk should not be empty");
        }
        other => panic!("GRAPH_WALK alias expected Array, got {:?}", other),
    }
}
