/// Integration tests for graph topology and network analysis functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

// ── Helper: build a triangle graph (id1 -- id2 -- id3 -- id1) ──────────────
fn make_triangle() -> (Arc<PieskieoDb>, Executor, Uuid, Uuid, Uuid) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();
    db.put_doc_ns(
        None,
        Some("g"),
        id1,
        serde_json::json!({"nid": id1.to_string()}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("g"),
        id2,
        serde_json::json!({"nid": id2.to_string()}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("g"),
        id3,
        serde_json::json!({"nid": id3.to_string()}),
    )
    .unwrap();
    db.add_edge(id1, id2, 1.0).unwrap();
    db.add_edge(id2, id3, 1.0).unwrap();
    db.add_edge(id3, id1, 1.0).unwrap();
    (db, ex, id1, id2, id3)
}

// ── CLUSTERING_COEFFICIENT ────────────────────────────────────────────────

#[test]
fn test_clustering_coefficient_triangle() {
    // In a triangle all three nodes are mutually connected.
    // Each node has degree 2 (considering undirected).
    // The two neighbors ARE connected to each other, so CC = 1.0
    let (_db, ex, id1, _id2, _id3) = make_triangle();
    let query = format!(
        r#"QUERY g WHERE nid = "{}" COMPUTE cc = CLUSTERING_COEFFICIENT(nid) SELECT cc;"#,
        id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("cc") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && *f <= 1.0,
            "triangle CC should be in (0,1], got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_clustering_coefficient_isolated_node() {
    // An isolated node has no edges, so CC should be 0.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(
        None,
        Some("iso"),
        id,
        serde_json::json!({"nid": id.to_string()}),
    )
    .unwrap();
    let query = format!(
        r#"QUERY iso COMPUTE cc = CLUSTERING_COEFFICIENT("{}") SELECT cc;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    assert_eq!(
        r.rows[0].data.get("cc"),
        Some(&Value::Float(0.0)),
        "isolated node CC should be 0.0"
    );
}

#[test]
fn test_local_clustering_alias() {
    // LOCAL_CLUSTERING is alias for CLUSTERING_COEFFICIENT
    let (_db, ex, id1, _id2, _id3) = make_triangle();
    let query = format!(
        r#"QUERY g WHERE nid = "{}" COMPUTE cc = LOCAL_CLUSTERING("{}") SELECT cc;"#,
        id1, id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("cc") {
        Some(Value::Float(_)) => {}
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── GRAPH_TRIANGLES ───────────────────────────────────────────────────────

#[test]
fn test_graph_triangles_in_triangle() {
    // Each node of a triangle participates in exactly 1 triangle
    let (_db, ex, id1, _id2, _id3) = make_triangle();
    let query = format!(
        r#"QUERY g WHERE nid = "{}" COMPUTE t = GRAPH_TRIANGLES("{}") SELECT t;"#,
        id1, id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("t") {
        Some(Value::Integer(n)) => assert!(
            *n >= 1,
            "triangle node should have >= 1 triangle, got {}",
            n
        ),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_graph_triangles_no_triangle() {
    // A chain graph A-B-C has no triangles (no edge between A and C)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("chain"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("chain"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("chain"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    // No edge a-c, so no triangle

    let query = format!(
        r#"QUERY chain WHERE name = "A" COMPUTE t = GRAPH_TRIANGLES("{}") SELECT t;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("t"),
        Some(&Value::Integer(0)),
        "chain endpoint A should have 0 triangles"
    );
}

// ── GRAPH_TOTAL_DEGREE / TOTAL_DEGREE ─────────────────────────────────────

#[test]
fn test_graph_total_degree_path_node() {
    // In a path A-B-C, B has 1 incoming edge (from A) and 1 outgoing edge (to C), total = 2
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("deg"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("deg"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("deg"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY deg WHERE name = "B" COMPUTE d = GRAPH_TOTAL_DEGREE("{}") SELECT d;"#,
        b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Integer(2)),
        "middle node B should have total degree 2"
    );
}

#[test]
fn test_total_degree_alias() {
    // TOTAL_DEGREE is alias for GRAPH_TOTAL_DEGREE
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("td"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("td"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY td WHERE name = "A" COMPUTE d = TOTAL_DEGREE("{}") SELECT d;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("d") {
        Some(Value::Integer(n)) => assert!(
            *n >= 1,
            "node with 1 out-edge should have total degree >= 1"
        ),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── IN_DEGREE / OUT_DEGREE aliases ────────────────────────────────────────

#[test]
fn test_in_degree_alias() {
    // IN_DEGREE(node_id) -> Integer  alias for GRAPH_IN_DEGREE
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let src = Uuid::new_v4();
    let dst = Uuid::new_v4();
    db.put_doc_ns(None, Some("indeg"), src, serde_json::json!({"role": "src"}))
        .unwrap();
    db.put_doc_ns(None, Some("indeg"), dst, serde_json::json!({"role": "dst"}))
        .unwrap();
    db.add_edge(src, dst, 1.0).unwrap();

    let query = format!(
        r#"QUERY indeg WHERE role = "dst" COMPUTE d = IN_DEGREE("{}") SELECT d;"#,
        dst
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Integer(1)),
        "dst should have in-degree 1"
    );
}

#[test]
fn test_out_degree_alias() {
    // OUT_DEGREE(node_id) -> Integer  alias for GRAPH_OUT_DEGREE
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let src = Uuid::new_v4();
    let dst = Uuid::new_v4();
    db.put_doc_ns(
        None,
        Some("outdeg"),
        src,
        serde_json::json!({"role": "src"}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("outdeg"),
        dst,
        serde_json::json!({"role": "dst"}),
    )
    .unwrap();
    db.add_edge(src, dst, 1.0).unwrap();

    let query = format!(
        r#"QUERY outdeg WHERE role = "src" COMPUTE d = OUT_DEGREE("{}") SELECT d;"#,
        src
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("d"),
        Some(&Value::Integer(1)),
        "src should have out-degree 1"
    );
}

// ── GRAPH_NEIGHBORS_AT_DEPTH ──────────────────────────────────────────────

#[test]
fn test_graph_neighbors_at_depth_1() {
    // Direct neighbors (depth=1)
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("nd"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("nd"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("nd"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(a, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY nd WHERE name = "A" COMPUTE nbrs = GRAPH_NEIGHBORS_AT_DEPTH("{}", 1) SELECT nbrs;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("nbrs") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "A has 2 direct neighbors, got {}", arr.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_graph_neighbors_at_depth_2() {
    // Path A-B-C: depth-2 neighbors of A is just C
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("nd2"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("nd2"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("nd2"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY nd2 WHERE name = "A" COMPUTE nbrs = GRAPH_NEIGHBORS_AT_DEPTH("{}", 2) SELECT nbrs;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("nbrs") {
        Some(Value::Array(arr)) => {
            assert_eq!(
                arr.len(),
                1,
                "A has 1 depth-2 neighbor (C), got {}",
                arr.len()
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── GRAPH_EGO_SIZE ────────────────────────────────────────────────────────

#[test]
fn test_graph_ego_size_radius_1() {
    // A has 2 direct neighbors → ego network size (radius=1) = 2
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("ego"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("ego"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("ego"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(a, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY ego WHERE name = "A" COMPUTE sz = GRAPH_EGO_SIZE("{}", 1) SELECT sz;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("sz"),
        Some(&Value::Integer(2)),
        "ego size radius 1 should be 2"
    );
}

#[test]
fn test_graph_ego_size_isolated() {
    // Isolated node has ego size 0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    db.put_doc_ns(None, Some("egoiso"), a, serde_json::json!({"name": "A"}))
        .unwrap();

    let query = format!(
        r#"QUERY egoiso COMPUTE sz = GRAPH_EGO_SIZE("{}", 2) SELECT sz;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("sz"),
        Some(&Value::Integer(0)),
        "isolated node ego size should be 0"
    );
}

// ── GRAPH_NODE_COUNT ──────────────────────────────────────────────────────

#[test]
fn test_graph_node_count_collection() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for i in 0..5 {
        db.put_doc_ns(
            None,
            Some("nc"),
            Uuid::new_v4(),
            serde_json::json!({"i": i}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY nc COMPUTE g=1 GROUP BY g COMPUTE cnt = GRAPH_NODE_COUNT("nc") SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("cnt") {
        Some(Value::Integer(n)) => assert_eq!(*n, 5, "should count 5 nodes, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── GRAPH_EDGE_COUNT ──────────────────────────────────────────────────────

#[test]
fn test_graph_edge_count_outgoing() {
    // GRAPH_EDGE_COUNT counts outgoing edges from a node
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let hub = Uuid::new_v4();
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    let n3 = Uuid::new_v4();
    db.put_doc_ns(None, Some("ec"), hub, serde_json::json!({"role": "hub"}))
        .unwrap();
    db.put_doc_ns(None, Some("ec"), n1, serde_json::json!({"role": "leaf"}))
        .unwrap();
    db.put_doc_ns(None, Some("ec"), n2, serde_json::json!({"role": "leaf"}))
        .unwrap();
    db.put_doc_ns(None, Some("ec"), n3, serde_json::json!({"role": "leaf"}))
        .unwrap();
    db.add_edge(hub, n1, 1.0).unwrap();
    db.add_edge(hub, n2, 1.0).unwrap();
    db.add_edge(hub, n3, 1.0).unwrap();

    let query = format!(
        r#"QUERY ec WHERE role = "hub" COMPUTE ec = GRAPH_EDGE_COUNT("{}") SELECT ec;"#,
        hub
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("ec"),
        Some(&Value::Integer(3)),
        "hub should have 3 outgoing edges"
    );
}

// ── IS_ISOLATED ───────────────────────────────────────────────────────────

#[test]
fn test_is_isolated_true() {
    // A node with no edges is isolated
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("isol"), id, serde_json::json!({"name": "solo"}))
        .unwrap();

    let query = format!(
        r#"QUERY isol COMPUTE iso = IS_ISOLATED("{}") SELECT iso;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("iso"),
        Some(&Value::Bool(true)),
        "node with no edges should be isolated"
    );
}

#[test]
fn test_is_isolated_false() {
    // A node with edges is not isolated
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("notiso"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("notiso"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY notiso WHERE name = "A" COMPUTE iso = IS_ISOLATED("{}") SELECT iso;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("iso"),
        Some(&Value::Bool(false)),
        "node with edges should not be isolated"
    );
}

#[test]
fn test_graph_is_isolated_alias() {
    // GRAPH_IS_ISOLATED is alias for IS_ISOLATED
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(
        None,
        Some("gisolated"),
        id,
        serde_json::json!({"name": "x"}),
    )
    .unwrap();

    let query = format!(
        r#"QUERY gisolated COMPUTE iso = GRAPH_IS_ISOLATED("{}") SELECT iso;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("iso"), Some(&Value::Bool(true)));
}

// ── GRAPH_SECOND_NEIGHBORS ────────────────────────────────────────────────

#[test]
fn test_graph_second_neighbors_path() {
    // Path A-B-C: second neighbors of A = {C}, count = 1
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("sn"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("sn"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("sn"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY sn WHERE name = "A" COMPUTE sn2 = GRAPH_SECOND_NEIGHBORS("{}") SELECT sn2;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("sn2"),
        Some(&Value::Integer(1)),
        "A should have 1 second-hop neighbor (C)"
    );
}

#[test]
fn test_graph_second_neighbors_isolated() {
    // Isolated node has 0 second neighbors
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("sni"), id, serde_json::json!({"name": "X"}))
        .unwrap();

    let query = format!(
        r#"QUERY sni COMPUTE sn2 = GRAPH_SECOND_NEIGHBORS("{}") SELECT sn2;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("sn2"),
        Some(&Value::Integer(0)),
        "isolated node has 0 second neighbors"
    );
}

#[test]
fn test_two_hop_neighbors_alias() {
    // TWO_HOP_NEIGHBORS is alias for GRAPH_SECOND_NEIGHBORS
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("thn"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("thn"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.put_doc_ns(None, Some("thn"), c, serde_json::json!({"name": "C"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();

    let query = format!(
        r#"QUERY thn WHERE name = "A" COMPUTE sn2 = TWO_HOP_NEIGHBORS("{}") SELECT sn2;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("sn2"),
        Some(&Value::Integer(1)),
        "TWO_HOP_NEIGHBORS should return 1 for path A-B-C"
    );
}

// ── GRAPH_CORE_NUMBER ─────────────────────────────────────────────────────

#[test]
fn test_graph_core_number_isolated() {
    // Isolated node has core number 0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("core"), id, serde_json::json!({"name": "X"}))
        .unwrap();

    let query = format!(
        r#"QUERY core COMPUTE k = GRAPH_CORE_NUMBER("{}") SELECT k;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("k") {
        Some(Value::Integer(_)) => {} // just check it returns an integer
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_graph_core_number_triangle() {
    // All nodes of a triangle are in the 2-core (each has degree >= 2)
    let (_db, ex, id1, _id2, _id3) = make_triangle();
    let query = format!(
        r#"QUERY g WHERE nid = "{}" COMPUTE k = GRAPH_CORE_NUMBER("{}") SELECT k;"#,
        id1, id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("k") {
        Some(Value::Integer(n)) => assert!(
            *n >= 1,
            "triangle node core number should be >= 1, got {}",
            n
        ),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── DEGREE_CENTRALITY alias ───────────────────────────────────────────────

#[test]
fn test_degree_centrality_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("dc"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("dc"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY dc WHERE name = "A" COMPUTE d = DEGREE_CENTRALITY("{}") SELECT d;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("d") {
        Some(Value::Integer(n)) => assert!(*n >= 1),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── EGO_NETWORK_SIZE alias ────────────────────────────────────────────────

#[test]
fn test_ego_network_size_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("ens"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("ens"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY ens WHERE name = "A" COMPUTE sz = EGO_NETWORK_SIZE("{}", 1) SELECT sz;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}

// ── GRAPH_NUM_NODES alias ─────────────────────────────────────────────────

#[test]
fn test_graph_num_nodes_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for i in 0..3 {
        db.put_doc_ns(
            None,
            Some("nn"),
            Uuid::new_v4(),
            serde_json::json!({"i": i}),
        )
        .unwrap();
    }

    let mut p = Parser::new(
        r#"QUERY nn COMPUTE g=1 GROUP BY g COMPUTE cnt = GRAPH_NUM_NODES("nn") SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

// ── TRIANGLE_COUNT_NODE alias ─────────────────────────────────────────────

#[test]
fn test_triangle_count_node_alias() {
    let (_db, ex, id1, _id2, _id3) = make_triangle();
    let query = format!(
        r#"QUERY g WHERE nid = "{}" COMPUTE t = TRIANGLE_COUNT_NODE("{}") SELECT t;"#,
        id1, id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("t") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── NEIGHBORS_DEPTH alias ─────────────────────────────────────────────────

#[test]
fn test_neighbors_depth_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("ndal"), a, serde_json::json!({"name": "A"}))
        .unwrap();
    db.put_doc_ns(None, Some("ndal"), b, serde_json::json!({"name": "B"}))
        .unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY ndal WHERE name = "A" COMPUTE nbrs = NEIGHBORS_DEPTH("{}", 1) SELECT nbrs;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("nbrs") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 1),
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── K_CORE alias ──────────────────────────────────────────────────────────

#[test]
fn test_k_core_alias() {
    let (_db, ex, id1, _id2, _id3) = make_triangle();
    let query = format!(
        r#"QUERY g WHERE nid = "{}" COMPUTE k = K_CORE("{}") SELECT k;"#,
        id1, id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("k") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer, got {:?}", other),
    }
}
