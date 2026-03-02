/// Integration tests for graph metric functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_graph_degree_functions() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    let alice_id = Uuid::new_v4();
    let bob_id = Uuid::new_v4();
    let carol_id = Uuid::new_v4();

    // Store the UUIDs in the document data so they can be referenced in queries
    db.put_doc_ns(None, Some("people"), alice_id,
        serde_json::json!({"name": "Alice", "node_id": alice_id.to_string()})).unwrap();
    db.put_doc_ns(None, Some("people"), bob_id,
        serde_json::json!({"name": "Bob", "node_id": bob_id.to_string()})).unwrap();
    db.put_doc_ns(None, Some("people"), carol_id,
        serde_json::json!({"name": "Carol", "node_id": carol_id.to_string()})).unwrap();

    // Alice -> Bob, Alice -> Carol
    db.add_edge(alice_id, bob_id, 1.0).unwrap();
    db.add_edge(alice_id, carol_id, 1.0).unwrap();

    // Test GRAPH_OUT_DEGREE for Alice (2 outgoing edges)
    let mut params = std::collections::HashMap::new();
    params.insert("alice_name".to_string(), Value::String("Alice".to_string()));
    ex.set_parameters(params);

    let mut p = Parser::new(
        "QUERY people WHERE name = @alice_name COMPUTE deg = GRAPH_OUT_DEGREE(TO_UUID(node_id)) SELECT name, deg;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1, "Expected exactly 1 row for Alice");
    assert_eq!(
        r.rows[0].data.get("deg"),
        Some(&Value::Integer(2)),
        "Expected Alice to have out-degree 2"
    );

    // Test GRAPH_IN_DEGREE for Bob (1 incoming edge from Alice)
    let mut params2 = std::collections::HashMap::new();
    params2.insert("bob_name".to_string(), Value::String("Bob".to_string()));
    ex.set_parameters(params2);

    let mut p2 = Parser::new(
        "QUERY people WHERE name = @bob_name COMPUTE indeg = GRAPH_IN_DEGREE(TO_UUID(node_id)) SELECT name, indeg;"
    );
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(r2.rows.len(), 1, "Expected exactly 1 row for Bob");
    assert_eq!(
        r2.rows[0].data.get("indeg"),
        Some(&Value::Integer(1)),
        "Expected Bob to have in-degree 1"
    );

    // Test GRAPH_DEGREE for Alice (2 out + 0 in = 2 total via neighbors_both)
    let mut params3 = std::collections::HashMap::new();
    params3.insert("alice_name".to_string(), Value::String("Alice".to_string()));
    ex.set_parameters(params3);

    let mut p3 = Parser::new(
        "QUERY people WHERE name = @alice_name COMPUTE total = GRAPH_DEGREE(TO_UUID(node_id)) SELECT name, total;"
    );
    let r3 = ex.execute(p3.parse().unwrap()).unwrap();
    assert_eq!(r3.rows.len(), 1, "Expected exactly 1 row for Alice");
    assert_eq!(
        r3.rows[0].data.get("total"),
        Some(&Value::Integer(2)),
        "Expected Alice to have total degree 2"
    );
}

#[test]
fn test_graph_neighbors_function() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    let alice_id = Uuid::new_v4();
    let bob_id = Uuid::new_v4();

    db.put_doc_ns(None, Some("people"), alice_id,
        serde_json::json!({"name": "Alice", "node_id": alice_id.to_string()})).unwrap();
    db.put_doc_ns(None, Some("people"), bob_id,
        serde_json::json!({"name": "Bob", "node_id": bob_id.to_string()})).unwrap();

    db.add_edge(alice_id, bob_id, 1.0).unwrap();

    let mut params = std::collections::HashMap::new();
    params.insert("alice_name".to_string(), Value::String("Alice".to_string()));
    ex.set_parameters(params);

    let mut p = Parser::new(
        "QUERY people WHERE name = @alice_name COMPUTE nbrs = GRAPH_NEIGHBORS(TO_UUID(node_id)) SELECT name, nbrs;"
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1, "Expected exactly 1 row for Alice");

    if let Some(Value::Array(neighbors)) = r.rows[0].data.get("nbrs") {
        assert!(!neighbors.is_empty(), "Expected at least 1 neighbor");
        assert!(
            neighbors.contains(&Value::Uuid(bob_id)),
            "Expected Bob in Alice's neighbors"
        );
    } else {
        panic!("Expected Array for GRAPH_NEIGHBORS result");
    }
}

#[test]
fn test_graph_basic_traversal() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    let alice_id = Uuid::new_v4();
    let bob_id = Uuid::new_v4();
    let carol_id = Uuid::new_v4();

    db.put_doc_ns(None, Some("people"), alice_id, serde_json::json!({"name": "Alice"})).unwrap();
    db.put_doc_ns(None, Some("people"), bob_id, serde_json::json!({"name": "Bob"})).unwrap();
    db.put_doc_ns(None, Some("people"), carol_id, serde_json::json!({"name": "Carol"})).unwrap();

    // Alice -> Bob, Bob -> Carol
    db.add_edge(alice_id, bob_id, 1.0).unwrap();
    db.add_edge(bob_id, carol_id, 1.0).unwrap();

    // Traverse all nodes at depth 1 to 2. Check that the traversal stats record correctly.
    let r = ex.execute(
        Parser::new("QUERY people TRAVERSE DEPTH 1 TO 2 SELECT name;")
            .parse()
            .unwrap(),
    )
    .unwrap();
    // The graph_traversals stat should be incremented (one traversal operation executed)
    assert_eq!(
        r.stats.graph_traversals, 1,
        "Expected 1 graph traversal recorded in stats"
    );
}
