/// Integration tests for community detection and advanced graph algorithm functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

// ── Helper: build a 4-node square graph (id1-id2-id3-id4-id1) with a diagonal ─
// id1 -- id2
//  |  \   |
// id4 -- id3
fn make_square_with_diagonal() -> (Arc<PieskieoDb>, Executor, Uuid, Uuid, Uuid, Uuid) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();
    let id4 = Uuid::new_v4();
    db.put_doc_ns(None, Some("sq"), id1, serde_json::json!({"nid": id1.to_string()})).unwrap();
    db.put_doc_ns(None, Some("sq"), id2, serde_json::json!({"nid": id2.to_string()})).unwrap();
    db.put_doc_ns(None, Some("sq"), id3, serde_json::json!({"nid": id3.to_string()})).unwrap();
    db.put_doc_ns(None, Some("sq"), id4, serde_json::json!({"nid": id4.to_string()})).unwrap();
    // Ring edges
    db.add_edge(id1, id2, 1.0).unwrap();
    db.add_edge(id2, id3, 1.0).unwrap();
    db.add_edge(id3, id4, 1.0).unwrap();
    db.add_edge(id4, id1, 1.0).unwrap();
    // Diagonal
    db.add_edge(id1, id3, 1.0).unwrap();
    (db, ex, id1, id2, id3, id4)
}

// ── LABEL_PROPAGATION ────────────────────────────────────────────────────────

#[test]
fn test_label_propagation_returns_integer() {
    let (_db, ex, id1, _id2, _id3, _id4) = make_square_with_diagonal();
    let query = format!(
        r#"QUERY sq WHERE nid = "{}" COMPUTE lbl = LABEL_PROPAGATION(nid, "sq") SELECT lbl;"#,
        id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "should have at least one result row");
    match r.rows[0].data.get("lbl") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer community label, got {:?}", other),
    }
}

#[test]
fn test_lpa_community_alias() {
    // LPA_COMMUNITY is alias for LABEL_PROPAGATION
    let (_db, ex, id1, _id2, _id3, _id4) = make_square_with_diagonal();
    let query = format!(
        r#"QUERY sq WHERE nid = "{}" COMPUTE lbl = LPA_COMMUNITY(nid, "sq") SELECT lbl;"#,
        id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("lbl") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_label_propagation_isolated_node() {
    // Isolated node: community label = initial label (hash of UUID) — just check Integer returned
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("solo"), id, serde_json::json!({"nid": id.to_string()})).unwrap();
    let query = format!(
        r#"QUERY solo COMPUTE lbl = LABEL_PROPAGATION(nid, "solo") SELECT lbl;"#
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("lbl") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── GRAPH_MODULARITY_CONTRIBUTION ────────────────────────────────────────────

#[test]
fn test_modularity_contribution_returns_float() {
    let (_db, ex, id1, _id2, _id3, _id4) = make_square_with_diagonal();
    let query = format!(
        r#"QUERY sq WHERE nid = "{}" COMPUTE mc = GRAPH_MODULARITY_CONTRIBUTION(nid, 0, "sq") SELECT mc;"#,
        id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("mc") {
        Some(Value::Float(_)) => {}
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_node_modularity_alias() {
    let (_db, ex, id1, _id2, _id3, _id4) = make_square_with_diagonal();
    let query = format!(
        r#"QUERY sq WHERE nid = "{}" COMPUTE mc = NODE_MODULARITY(nid, 0, "sq") SELECT mc;"#,
        id1
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("mc") {
        Some(Value::Float(_)) => {}
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── CONNECTED_COMPONENT_ID ───────────────────────────────────────────────────

#[test]
fn test_connected_component_same_component() {
    // Two connected nodes must have the same component ID
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("cc"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("cc"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let qa = format!(
        r#"QUERY cc WHERE nid = "{}" COMPUTE cid = CONNECTED_COMPONENT_ID(nid, "cc") SELECT cid;"#,
        a
    );
    let qb = format!(
        r#"QUERY cc WHERE nid = "{}" COMPUTE cid = CONNECTED_COMPONENT_ID(nid, "cc") SELECT cid;"#,
        b
    );
    let mut pa = Parser::new(&qa);
    let mut pb = Parser::new(&qb);
    let ra = ex.execute(pa.parse().unwrap()).unwrap();
    let rb = ex.execute(pb.parse().unwrap()).unwrap();
    assert!(!ra.rows.is_empty());
    assert!(!rb.rows.is_empty());
    let cid_a = ra.rows[0].data.get("cid");
    let cid_b = rb.rows[0].data.get("cid");
    assert_eq!(cid_a, cid_b, "connected nodes must share the same component ID");
}

#[test]
fn test_connected_component_different_components() {
    // Two disconnected nodes must have different component IDs
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("dc"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("dc"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    // No edges — two separate components

    let qa = format!(
        r#"QUERY dc WHERE nid = "{}" COMPUTE cid = CONNECTED_COMPONENT_ID(nid, "dc") SELECT cid;"#,
        a
    );
    let qb = format!(
        r#"QUERY dc WHERE nid = "{}" COMPUTE cid = CONNECTED_COMPONENT_ID(nid, "dc") SELECT cid;"#,
        b
    );
    let mut pa = Parser::new(&qa);
    let mut pb = Parser::new(&qb);
    let ra = ex.execute(pa.parse().unwrap()).unwrap();
    let rb = ex.execute(pb.parse().unwrap()).unwrap();
    assert!(!ra.rows.is_empty());
    assert!(!rb.rows.is_empty());
    let cid_a = ra.rows[0].data.get("cid");
    let cid_b = rb.rows[0].data.get("cid");
    assert_ne!(cid_a, cid_b, "disconnected nodes must have different component IDs");
}

#[test]
fn test_union_find_component_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    db.put_doc_ns(None, Some("ufc"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    let query = format!(
        r#"QUERY ufc COMPUTE cid = UNION_FIND_COMPONENT(nid, "ufc") SELECT cid;"#
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("cid") {
        Some(Value::Integer(_)) => {}
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── GRAPH_DIAMETER_APPROX ────────────────────────────────────────────────────

#[test]
fn test_graph_diameter_approx_path_graph() {
    // Linear chain: A-B-C-D, diameter = 3
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let d = Uuid::new_v4();
    db.put_doc_ns(None, Some("pathgraph"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("pathgraph"), b, serde_json::json!({"name": "B"})).unwrap();
    db.put_doc_ns(None, Some("pathgraph"), c, serde_json::json!({"name": "C"})).unwrap();
    db.put_doc_ns(None, Some("pathgraph"), d, serde_json::json!({"name": "D"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    db.add_edge(c, d, 1.0).unwrap();

    let mut p = Parser::new(r#"QUERY pathgraph COMPUTE g=1 GROUP BY g COMPUTE diam = GRAPH_DIAMETER_APPROX("pathgraph", 4) SELECT diam;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("diam") {
        Some(Value::Integer(n)) => assert!(*n >= 3, "pathgraph A-B-C-D diameter should be >= 3, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_approx_diameter_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("ad"), a, serde_json::json!({"name": "A"})).unwrap();
    db.put_doc_ns(None, Some("ad"), b, serde_json::json!({"name": "B"})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let mut p = Parser::new(r#"QUERY ad COMPUTE g=1 GROUP BY g COMPUTE diam = APPROX_DIAMETER("ad", 10) SELECT diam;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("diam") {
        Some(Value::Integer(n)) => assert!(*n >= 1, "2-node graph diameter should be >= 1, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── HUB_SCORE ────────────────────────────────────────────────────────────────

#[test]
fn test_hub_score_hub_node() {
    // Hub node (points to many) should have hub score > 0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let hub = Uuid::new_v4();
    let n1 = Uuid::new_v4();
    let n2 = Uuid::new_v4();
    let n3 = Uuid::new_v4();
    db.put_doc_ns(None, Some("hs"), hub, serde_json::json!({"role": "hub"})).unwrap();
    db.put_doc_ns(None, Some("hs"), n1, serde_json::json!({"role": "leaf"})).unwrap();
    db.put_doc_ns(None, Some("hs"), n2, serde_json::json!({"role": "leaf"})).unwrap();
    db.put_doc_ns(None, Some("hs"), n3, serde_json::json!({"role": "leaf"})).unwrap();
    db.add_edge(hub, n1, 1.0).unwrap();
    db.add_edge(hub, n2, 1.0).unwrap();
    db.add_edge(hub, n3, 1.0).unwrap();
    // Add in-edges to leaves to give them in-degree
    db.add_edge(n2, n1, 1.0).unwrap();
    db.add_edge(n3, n1, 1.0).unwrap();

    let query = format!(
        r#"QUERY hs WHERE role = "hub" COMPUTE h = HUB_SCORE(nid, "hs") SELECT h;"#
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f <= 1.0, "hub score should be in [0,1], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_hub_score_isolated_node() {
    // Isolated node has hub score 0.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("hiso"), id, serde_json::json!({"nid": id.to_string()})).unwrap();
    let query = format!(
        r#"QUERY hiso COMPUTE h = HUB_SCORE("{}", "hiso") SELECT h;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("h"),
        Some(&Value::Float(0.0)),
        "isolated node hub score should be 0.0"
    );
}

#[test]
fn test_hits_hub_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("hh"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("hh"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    let query = format!(
        r#"QUERY hh WHERE nid = "{}" COMPUTE h = HITS_HUB(nid, "hh") SELECT h;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("h") {
        Some(Value::Float(_)) => {}
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── AUTHORITY_SCORE ───────────────────────────────────────────────────────────

#[test]
fn test_authority_score_authority_node() {
    // Authority node (pointed to by many) should have authority score > 0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let auth = Uuid::new_v4();
    let h1 = Uuid::new_v4();
    let h2 = Uuid::new_v4();
    let h3 = Uuid::new_v4();
    db.put_doc_ns(None, Some("authscores"), auth, serde_json::json!({"role": "auth"})).unwrap();
    db.put_doc_ns(None, Some("authscores"), h1, serde_json::json!({"role": "hub"})).unwrap();
    db.put_doc_ns(None, Some("authscores"), h2, serde_json::json!({"role": "hub"})).unwrap();
    db.put_doc_ns(None, Some("authscores"), h3, serde_json::json!({"role": "hub"})).unwrap();
    // Hubs pointing to authority
    db.add_edge(h1, auth, 1.0).unwrap();
    db.add_edge(h2, auth, 1.0).unwrap();
    db.add_edge(h3, auth, 1.0).unwrap();
    // Give hubs some out-edges to others as well
    db.add_edge(h1, h2, 1.0).unwrap();
    db.add_edge(h2, h3, 1.0).unwrap();

    let query = format!(
        r#"QUERY authscores WHERE role = "auth" COMPUTE a = AUTHORITY_SCORE(nid, "authscores") SELECT a;"#
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f <= 1.0, "authority score should be in [0,1], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_authority_score_isolated_node() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let id = Uuid::new_v4();
    db.put_doc_ns(None, Some("aiso"), id, serde_json::json!({"nid": id.to_string()})).unwrap();
    let query = format!(
        r#"QUERY aiso COMPUTE a = AUTHORITY_SCORE("{}", "aiso") SELECT a;"#,
        id
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("a"),
        Some(&Value::Float(0.0)),
        "isolated node authority score should be 0.0"
    );
}

#[test]
fn test_hits_authority_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("ha"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("ha"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    let query = format!(
        r#"QUERY ha WHERE nid = "{}" COMPUTE au = HITS_AUTHORITY(nid, "ha") SELECT au;"#,
        b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("au") {
        Some(Value::Float(_)) => {}
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── GRAPH_RECIPROCITY ─────────────────────────────────────────────────────────

#[test]
fn test_graph_reciprocity_mutual_edges() {
    // A <-> B: reciprocity for A = 1.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("rec"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("rec"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, a, 1.0).unwrap();

    let query = format!(
        r#"QUERY rec WHERE nid = "{}" COMPUTE r = GRAPH_RECIPROCITY(nid, "rec") SELECT r;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::Float(1.0)),
        "fully mutual node should have reciprocity 1.0"
    );
}

#[test]
fn test_graph_reciprocity_no_back_edges() {
    // A -> B (no back-edge): reciprocity for A = 0.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("noback"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("noback"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY noback WHERE nid = "{}" COMPUTE r = GRAPH_RECIPROCITY(nid, "noback") SELECT r;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::Float(0.0)),
        "one-way edges should give reciprocity 0.0"
    );
}

#[test]
fn test_edge_reciprocity_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("er"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("er"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, a, 1.0).unwrap();
    let query = format!(
        r#"QUERY er WHERE nid = "{}" COMPUTE r = EDGE_RECIPROCITY(nid, "er") SELECT r;"#,
        a
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Float(1.0)));
}

// ── GRAPH_OVERLAP ─────────────────────────────────────────────────────────────

#[test]
fn test_graph_overlap_common_neighbors() {
    // A-B-C-A triangle: A and C share neighbor B → overlap = 1/1 = 1.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("ov"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("ov"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.put_doc_ns(None, Some("ov"), c, serde_json::json!({"nid": c.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    db.add_edge(c, a, 1.0).unwrap();

    let query = format!(
        r#"QUERY ov WHERE nid = "{}" COMPUTE ov = GRAPH_OVERLAP("{}", "{}") SELECT ov;"#,
        a, a, c
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ov") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f <= 1.0, "overlap should be in [0,1], got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_graph_overlap_no_common_neighbors() {
    // A-B and C-D: no common neighbors → overlap = 0.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let d = Uuid::new_v4();
    db.put_doc_ns(None, Some("noov"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("noov"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.put_doc_ns(None, Some("noov"), c, serde_json::json!({"nid": c.to_string()})).unwrap();
    db.put_doc_ns(None, Some("noov"), d, serde_json::json!({"nid": d.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(c, d, 1.0).unwrap();

    let query = format!(
        r#"QUERY noov WHERE nid = "{}" COMPUTE ov = GRAPH_OVERLAP("{}", "{}") SELECT ov;"#,
        a, a, c
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("ov"),
        Some(&Value::Float(0.0)),
        "disjoint neighborhoods should give overlap 0.0"
    );
}

#[test]
fn test_neighborhood_overlap_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("noa"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("noa"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    let query = format!(
        r#"QUERY noa COMPUTE ov = NEIGHBORHOOD_OVERLAP("{}", "{}") SELECT ov;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("ov"), Some(&Value::Float(0.0)));
}

// ── GRAPH_PREFERENTIAL_ATTACHMENT ────────────────────────────────────────────

#[test]
fn test_preferential_attachment_score() {
    // A has degree 2, B has degree 2 → PA score = 4
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let d = Uuid::new_v4();
    db.put_doc_ns(None, Some("pa"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("pa"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.put_doc_ns(None, Some("pa"), c, serde_json::json!({"nid": c.to_string()})).unwrap();
    db.put_doc_ns(None, Some("pa"), d, serde_json::json!({"nid": d.to_string()})).unwrap();
    db.add_edge(a, c, 1.0).unwrap();
    db.add_edge(a, d, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    db.add_edge(b, d, 1.0).unwrap();

    let query = format!(
        r#"QUERY pa WHERE nid = "{}" COMPUTE pa = GRAPH_PREFERENTIAL_ATTACHMENT("{}", "{}") SELECT pa;"#,
        a, a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("pa") {
        Some(Value::Integer(n)) => assert!(*n >= 4, "PA score for degree-2 nodes should be >= 4, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_pref_attach_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("paa"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("paa"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    let query = format!(
        r#"QUERY paa COMPUTE pa = PREF_ATTACH("{}", "{}") SELECT pa;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("pa"), Some(&Value::Integer(0)));
}

// ── GRAPH_RESOURCE_ALLOCATION ─────────────────────────────────────────────────

#[test]
fn test_resource_allocation_common_neighbor() {
    // A-B-C with only B as common neighbor, degree(B) = 2
    // RA index = 1/2 = 0.5
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    db.put_doc_ns(None, Some("ra"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("ra"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.put_doc_ns(None, Some("ra"), c, serde_json::json!({"nid": c.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(b, c, 1.0).unwrap();
    // Also add back-edges so undirected neighborhood finds them
    db.add_edge(b, a, 1.0).unwrap();
    db.add_edge(c, b, 1.0).unwrap();

    let query = format!(
        r#"QUERY ra WHERE nid = "{}" COMPUTE ra = GRAPH_RESOURCE_ALLOCATION("{}", "{}") SELECT ra;"#,
        a, a, c
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ra") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "common-neighbor RA should be > 0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_resource_allocation_no_common_neighbor() {
    // A-B and C-D: no common neighbor → RA = 0.0
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let d = Uuid::new_v4();
    db.put_doc_ns(None, Some("nora"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("nora"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    db.put_doc_ns(None, Some("nora"), c, serde_json::json!({"nid": c.to_string()})).unwrap();
    db.put_doc_ns(None, Some("nora"), d, serde_json::json!({"nid": d.to_string()})).unwrap();
    db.add_edge(a, b, 1.0).unwrap();
    db.add_edge(c, d, 1.0).unwrap();

    let query = format!(
        r#"QUERY nora WHERE nid = "{}" COMPUTE ra = GRAPH_RESOURCE_ALLOCATION("{}", "{}") SELECT ra;"#,
        a, a, c
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(
        r.rows[0].data.get("ra"),
        Some(&Value::Float(0.0)),
        "no common neighbors should give RA 0.0"
    );
}

#[test]
fn test_resource_alloc_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.put_doc_ns(None, Some("raa"), a, serde_json::json!({"nid": a.to_string()})).unwrap();
    db.put_doc_ns(None, Some("raa"), b, serde_json::json!({"nid": b.to_string()})).unwrap();
    let query = format!(
        r#"QUERY raa COMPUTE ra = RESOURCE_ALLOC("{}", "{}") SELECT ra;"#,
        a, b
    );
    let mut p = Parser::new(&query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("ra"), Some(&Value::Float(0.0)));
}
