/// Integration tests for advanced graph analytics functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_graph_connected_components_count_isolated() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 3 isolated nodes (no edges) → 3 components
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("pts"), a, serde_json::json!({"id": 1})).unwrap();
    db.put_doc_ns(None, Some("pts"), b, serde_json::json!({"id": 2})).unwrap();
    db.put_doc_ns(None, Some("pts"), c, serde_json::json!({"id": 3})).unwrap();

    let mut p = Parser::new(r#"QUERY pts COMPUTE g = 1 GROUP BY g COMPUTE cc = GRAPH_CONNECTED_COMPONENTS_COUNT("pts") SELECT cc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("cc") {
        Some(Value::Integer(n)) => assert_eq!(*n, 3, "should have 3 isolated components, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_graph_connected_components_count_connected() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 3 nodes all connected in a path A-B-C → 1 component
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("conn"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("conn"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("conn"), c, serde_json::json!({"name": "C"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let mut p = Parser::new(r#"QUERY conn COMPUTE g = 1 GROUP BY g COMPUTE cc = GRAPH_CONNECTED_COMPONENTS_COUNT("conn") SELECT cc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("cc") {
        Some(Value::Integer(n)) => assert_eq!(*n, 1, "all connected should be 1 component, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_graph_density_no_edges() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for i in 0..4 {
        db.put_doc_ns(None, Some("g"), Uuid::new_v4(), serde_json::json!({"i": i})).unwrap();
    }

    let mut p = Parser::new(r#"QUERY g COMPUTE g = 1 GROUP BY g COMPUTE d = GRAPH_DENSITY("g") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f <= 1.0, "density should be in [0,1], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_graph_has_path_self() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let node_id = Uuid::new_v4();
    db.put_doc_ns(None, Some("graph"), node_id, serde_json::json!({"name": "self"})).unwrap();

    // A node is reachable from itself
    let query = format!(
        r#"QUERY graph COMPUTE has = GRAPH_HAS_PATH("{}", "{}") SELECT has;"#,
        node_id, node_id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("has"),
        Some(&Value::Bool(true)),
        "a node should have a path to itself"
    );
}

#[test]
fn test_graph_has_path_unreachable() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    db.put_doc_ns(None, Some("graph"), n1, serde_json::json!({"name": "n1"})).unwrap();
    db.put_doc_ns(None, Some("graph"), n2, serde_json::json!({"name": "n2"})).unwrap();
    // No edges added — n1 and n2 are not connected

    let query = format!(
        r#"QUERY graph WHERE name = "n1" COMPUTE has = GRAPH_HAS_PATH("{}", "{}") SELECT has;"#,
        n1, n2
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("has"),
        Some(&Value::Bool(false)),
        "disconnected nodes should not have a path"
    );
}

#[test]
fn test_graph_shortest_path_same_node() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let node_id = Uuid::new_v4();
    db.put_doc_ns(None, Some("graph"), node_id, serde_json::json!({"name": "node"})).unwrap();

    let query = format!(
        r#"QUERY graph COMPUTE d = GRAPH_SHORTEST_PATH_LENGTH("{}", "{}") SELECT d;"#,
        node_id, node_id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Integer(0)),
        "same node should have path length 0"
    );
}

#[test]
fn test_graph_shortest_path_length_connected() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("path_test"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("path_test"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("path_test"), c, serde_json::json!({"name": "C"})).unwrap();
    // A-B-C path
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    // A to C should be distance 2
    let query = format!(
        r#"QUERY path_test WHERE name = "A" COMPUTE d = GRAPH_SHORTEST_PATH_LENGTH("{}", "{}") SELECT d;"#,
        a, c
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Integer(2)),
        "A to C through B should be distance 2"
    );
}

#[test]
fn test_graph_shortest_path_unreachable() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    db.put_doc_ns(None, Some("unreach"), n1, serde_json::json!({"name": "n1"})).unwrap();
    db.put_doc_ns(None, Some("unreach"), n2, serde_json::json!({"name": "n2"})).unwrap();

    let query = format!(
        r#"QUERY unreach WHERE name = "n1" COMPUTE d = GRAPH_SHORTEST_PATH_LENGTH("{}", "{}") SELECT d;"#,
        n1, n2
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Null),
        "unreachable nodes should return Null"
    );
}

#[test]
fn test_graph_common_neighbors_none() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    db.put_doc_ns(None, Some("g"), n1, serde_json::json!({"name": "n1"})).unwrap();
    db.put_doc_ns(None, Some("g"), n2, serde_json::json!({"name": "n2"})).unwrap();

    // No edges → no common neighbors
    let query = format!(
        r#"QUERY g WHERE name = "n1" COMPUTE c = GRAPH_COMMON_NEIGHBORS("{}", "{}") SELECT c;"#,
        n1, n2
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Integer(0)));
}

#[test]
fn test_graph_common_neighbors_with_shared() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let shared = Uuid::new_v4();
    db.put_doc_ns(None, Some("cn"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("cn"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("cn"), shared, serde_json::json!({"name": "shared"})).unwrap();
    // Both A and B connect to shared
    db.add_edge(a, shared, 1.0).unwrap();
    db.add_edge(b, shared, 1.0).unwrap();

    let query = format!(
        r#"QUERY cn WHERE name = "A" COMPUTE c = GRAPH_COMMON_NEIGHBORS("{}", "{}") SELECT c;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("c") {
        Some(Value::Integer(n)) => assert!(*n >= 1, "should have at least 1 common neighbor, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_graph_jaccard_similarity_no_edges() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    db.put_doc_ns(None, Some("jac"), n1, serde_json::json!({"name": "n1"})).unwrap();
    db.put_doc_ns(None, Some("jac"), n2, serde_json::json!({"name": "n2"})).unwrap();

    let query = format!(
        r#"QUERY jac WHERE name = "n1" COMPUTE j = GRAPH_JACCARD_SIMILARITY("{}", "{}") SELECT j;"#,
        n1, n2
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("j"),
        Some(&Value::Float(0.0)),
        "nodes with no neighbors should have Jaccard similarity 0"
    );
}

#[test]
fn test_adamic_adar_no_common() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    db.put_doc_ns(None, Some("aa"), n1, serde_json::json!({"name": "n1"})).unwrap();
    db.put_doc_ns(None, Some("aa"), n2, serde_json::json!({"name": "n2"})).unwrap();

    let query = format!(
        r#"QUERY aa WHERE name = "n1" COMPUTE score = ADAMIC_ADAR("{}", "{}") SELECT score;"#,
        n1, n2
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("score"),
        Some(&Value::Float(0.0)),
        "nodes with no common neighbors should have Adamic-Adar score 0"
    );
}

#[test]
fn test_bfs_distance_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("bfs"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("bfs"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    // BFS_DISTANCE is an alias for GRAPH_SHORTEST_PATH_LENGTH
    let query = format!(
        r#"QUERY bfs WHERE name = "A" COMPUTE d = BFS_DISTANCE("{}", "{}") SELECT d;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Integer(1)),
        "directly connected nodes should have BFS distance 1"
    );
}

#[test]
fn test_is_reachable_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("reach"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("reach"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    // IS_REACHABLE is an alias for GRAPH_HAS_PATH
    let query = format!(
        r#"QUERY reach WHERE name = "A" COMPUTE r = IS_REACHABLE("{}", "{}") SELECT r;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::Bool(true)),
        "connected nodes should be reachable"
    );
}

#[test]
fn test_graph_eccentricity_path() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("ecc"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("ecc"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("ecc"), c, serde_json::json!({"name": "C"})).unwrap();
    // A-B-C path: eccentricity of B is 1, eccentricity of A is 2
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    // Test eccentricity of A (endpoint of path): max distance is 2 (A->B->C)
    let query = format!(
        r#"QUERY ecc WHERE name = "A" COMPUTE ecc = GRAPH_ECCENTRICITY("{}", "ecc") SELECT ecc;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("ecc") {
        Some(Value::Integer(n)) => assert_eq!(*n, 2, "eccentricity of endpoint should be 2, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_network_density_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("nd"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("nd"), b, serde_json::json!({"name": "B"})).unwrap();

    // NETWORK_DENSITY is alias for GRAPH_DENSITY
    let mut p = Parser::new(r#"QUERY nd COMPUTE g = 1 GROUP BY g COMPUTE d = NETWORK_DENSITY("nd") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f <= 1.0, "density should be in [0,1], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}
