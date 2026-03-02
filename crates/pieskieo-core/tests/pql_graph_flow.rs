/// Integration tests for PQL advanced graph flow and connectivity algorithm functions.
use pieskieo_core::{pql::{Executor, Parser, Value}, PieskieoDb};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

fn seed(db: &PieskieoDb) {
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
}

fn run(ex: &Executor, query: &str) -> Value {
    let mut p = Parser::new(query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "no rows returned for: {}", query);
    r.rows[0].data.get("res").cloned().unwrap_or(Value::Null)
}

// ── TOPOLOGICAL_SORT ─────────────────────────────────────────────────────────

#[test]
fn test_topological_sort_dag() {
    let (db, ex) = setup();
    seed(&db);
    // DAG: A->B, A->C, B->C
    let v = run(&ex, r#"QUERY t COMPUTE res = TOPOLOGICAL_SORT({"A": ["B", "C"], "B": ["C"], "C": []}) SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3);
            let pos: std::collections::HashMap<String, usize> = arr.iter().enumerate().map(|(i, v)| {
                if let Value::String(s) = v { (s.clone(), i) } else { (String::new(), i) }
            }).collect();
            assert!(pos["A"] < pos["C"], "A must come before C");
            assert!(pos["A"] < pos["B"], "A must come before B");
            assert!(pos["B"] < pos["C"], "B must come before C");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_topological_sort_linear() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = TOPOLOGICAL_SORT({"X": ["Y"], "Y": ["Z"], "Z": []}) SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3);
            let s: Vec<&str> = arr.iter().map(|x| if let Value::String(s) = x { s.as_str() } else { "" }).collect();
            assert_eq!(s, vec!["X", "Y", "Z"]);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_topological_sort_cycle_returns_empty() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = TOPOLOGICAL_SORT({"A": ["B"], "B": ["C"], "C": ["A"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => assert_eq!(arr.len(), 0, "cycle should return empty array"),
        other => panic!("expected empty array, got {:?}", other),
    }
}

#[test]
fn test_topo_sort_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = TOPO_SORT({"A": ["B"], "B": []}) SELECT res;"#);
    match v {
        Value::Array(arr) => assert_eq!(arr.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── TOPOLOGICAL_SORT_DFS ──────────────────────────────────────────────────────

#[test]
fn test_topological_sort_dfs_dag() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = TOPO_SORT_DFS({"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}) SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 4);
            let pos: std::collections::HashMap<String, usize> = arr.iter().enumerate().map(|(i, v)| {
                if let Value::String(s) = v { (s.clone(), i) } else { (String::new(), i) }
            }).collect();
            assert!(pos["A"] < pos["D"], "A before D");
            assert!(pos["B"] < pos["D"], "B before D");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── HAS_CYCLE ─────────────────────────────────────────────────────────────────

#[test]
fn test_has_cycle_true() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = HAS_CYCLE({"A": ["B"], "B": ["C"], "C": ["A"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn test_has_cycle_false_dag() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = HAS_CYCLE({"A": ["B"], "B": ["C"], "C": []}) SELECT res;"#);
    assert_eq!(v, Value::Bool(false));
}

#[test]
fn test_has_cycle_single_node() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = GRAPH_HAS_CYCLE({"A": []}) SELECT res;"#);
    assert_eq!(v, Value::Bool(false));
}

// ── BFS_ORDER ─────────────────────────────────────────────────────────────────

#[test]
fn test_bfs_order_basic() {
    let (db, ex) = setup();
    seed(&db);
    // Star: A connected to B, C, D
    let v = run(&ex, r#"QUERY t COMPUTE res = BFS_ORDER({"A": ["B", "C", "D"], "B": ["A"], "C": ["A"], "D": ["A"]}, "A") SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[0], Value::String("A".to_string()), "A should be first in BFS");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_bfs_traversal_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = BFS_TRAVERSAL({"A": ["B"], "B": ["C"], "C": []}, "A") SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3);
            let s: Vec<&str> = arr.iter().map(|x| if let Value::String(s) = x { s.as_str() } else { "" }).collect();
            assert_eq!(s, vec!["A", "B", "C"]);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── DFS_ORDER ─────────────────────────────────────────────────────────────────

#[test]
fn test_dfs_order_basic() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = DFS_ORDER({"A": ["B", "C"], "B": ["D"], "C": [], "D": []}, "A") SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 4);
            // A must be first
            assert_eq!(arr[0], Value::String("A".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_dfs_traversal_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = DFS_TRAVERSAL({"A": ["B"], "B": ["C"], "C": []}, "A") SELECT res;"#);
    match v {
        Value::Array(arr) => assert_eq!(arr.len(), 3),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── BFS_LAYERS ────────────────────────────────────────────────────────────────

#[test]
fn test_bfs_layers_basic() {
    let (db, ex) = setup();
    seed(&db);
    // Linear chain A-B-C-D
    let v = run(&ex, r#"QUERY t COMPUTE res = BFS_LAYERS({"A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C"]}, "A") SELECT res;"#);
    match v {
        Value::Array(layers) => {
            assert_eq!(layers.len(), 4, "4 BFS levels for a chain of 4");
            if let Value::Array(l0) = &layers[0] { assert_eq!(l0.len(), 1); }
            if let Value::Array(l1) = &layers[1] { assert_eq!(l1.len(), 1); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_bfs_levels_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = BFS_LEVELS({"A": ["B", "C"], "B": [], "C": []}, "A") SELECT res;"#);
    match v {
        Value::Array(layers) => {
            assert_eq!(layers.len(), 2);
            if let Value::Array(l1) = &layers[1] { assert_eq!(l1.len(), 2); }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── IS_CONNECTED ──────────────────────────────────────────────────────────────

#[test]
fn test_is_connected_true() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = IS_CONNECTED({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn test_is_connected_false() {
    let (db, ex) = setup();
    seed(&db);
    // A-B disconnected from C-D
    let v = run(&ex, r#"QUERY t COMPUTE res = IS_GRAPH_CONNECTED({"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(false));
}

// ── CONNECTED_COMPONENTS ──────────────────────────────────────────────────────

#[test]
fn test_connected_components_two() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = CONNECTED_COMPONENTS({"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}) SELECT res;"#);
    match v {
        Value::Array(comps) => assert_eq!(comps.len(), 2, "should have 2 components"),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_weakly_connected_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = WEAKLY_CONNECTED({"A": ["B"], "B": ["A"], "C": []}) SELECT res;"#);
    match v {
        Value::Array(comps) => assert_eq!(comps.len(), 2, "A-B and C are 2 components"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── STRONGLY_CONNECTED_COMPONENTS ─────────────────────────────────────────────

#[test]
fn test_scc_simple() {
    let (db, ex) = setup();
    seed(&db);
    // A->B->C->A cycle, D->A
    let v = run(&ex, r#"QUERY t COMPUTE res = SCC({"A": ["B"], "B": ["C"], "C": ["A"], "D": ["A"]}) SELECT res;"#);
    match v {
        Value::Array(sccs) => {
            // {A,B,C} and {D} are the two SCCs
            assert_eq!(sccs.len(), 2, "should have 2 SCCs, got {:?}", sccs);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_strongly_connected_components_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = STRONGLY_CONNECTED_COMPONENTS({"A": ["B"], "B": ["A"]}) SELECT res;"#);
    match v {
        Value::Array(sccs) => assert_eq!(sccs.len(), 1, "A<->B is one SCC"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── IS_BIPARTITE ──────────────────────────────────────────────────────────────

#[test]
fn test_is_bipartite_true() {
    let (db, ex) = setup();
    seed(&db);
    // Path: A-B-C (bipartite)
    let v = run(&ex, r#"QUERY t COMPUTE res = IS_BIPARTITE({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn test_is_bipartite_false_triangle() {
    let (db, ex) = setup();
    seed(&db);
    // Triangle: odd cycle, not bipartite
    let v = run(&ex, r#"QUERY t COMPUTE res = GRAPH_IS_BIPARTITE({"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(false));
}

// ── IS_EULERIAN / IS_SEMI_EULERIAN ────────────────────────────────────────────

#[test]
fn test_is_eulerian_true() {
    let (db, ex) = setup();
    seed(&db);
    // K3 (triangle) - all vertices degree 2, connected
    let v = run(&ex, r#"QUERY t COMPUTE res = IS_EULERIAN({"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn test_has_euler_circuit_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = HAS_EULER_CIRCUIT({"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn test_is_semi_eulerian_path_exists() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C-D: A and D have odd degree (1), B and C have even degree (2)
    let v = run(&ex, r#"QUERY t COMPUTE res = IS_SEMI_EULERIAN({"A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn test_has_euler_path_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = HAS_EULER_PATH({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    assert_eq!(v, Value::Bool(true));
}

// ── ARTICULATION_POINTS ───────────────────────────────────────────────────────

#[test]
fn test_articulation_points_bridge_graph() {
    let (db, ex) = setup();
    seed(&db);
    // A-B-C with B as articulation point
    let v = run(&ex, r#"QUERY t COMPUTE res = ARTICULATION_POINTS({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert!(arr.iter().any(|x| x == &Value::String("B".to_string())), "B should be AP, got {:?}", arr);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_cut_vertices_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = CUT_VERTICES({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => assert!(!arr.is_empty(), "should find cut vertex B"),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_articulation_points_none_in_triangle() {
    let (db, ex) = setup();
    seed(&db);
    // Triangle: no articulation points
    let v = run(&ex, r#"QUERY t COMPUTE res = ARTICULATION_POINTS({"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => assert!(arr.is_empty(), "triangle should have no APs"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── BRIDGE_EDGES ──────────────────────────────────────────────────────────────

#[test]
fn test_bridge_edges_single_bridge() {
    let (db, ex) = setup();
    seed(&db);
    // Triangle A-B-C with bridge edge D connected only to C
    let v = run(&ex, r#"QUERY t COMPUTE res = BRIDGE_EDGES({"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B", "D"], "D": ["C"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 1, "should find exactly 1 bridge, got {:?}", arr);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_graph_bridges_alias() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C: two bridges
    let v = run(&ex, r#"QUERY t COMPUTE res = GRAPH_BRIDGES({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => assert_eq!(arr.len(), 2, "path of 3 has 2 bridges"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── DIJKSTRA_DIST ─────────────────────────────────────────────────────────────

#[test]
fn test_dijkstra_dist_basic() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = DIJKSTRA_DIST({"A": [["B", 1.0], ["C", 4.0]], "B": [["C", 2.0]], "C": []}, "A") SELECT res;"#);
    match v {
        Value::Object(map) => {
            assert_eq!(map.get("A"), Some(&Value::Float(0.0)));
            assert_eq!(map.get("B"), Some(&Value::Float(1.0)));
            assert_eq!(map.get("C"), Some(&Value::Float(3.0)), "shortest A->C via B is 3");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_shortest_dist_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = SHORTEST_DIST({"A": [["B", 2.0]], "B": [["C", 3.0]], "C": []}, "A") SELECT res;"#);
    match v {
        Value::Object(map) => {
            assert_eq!(map.get("C"), Some(&Value::Float(5.0)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── DIJKSTRA_PATH ─────────────────────────────────────────────────────────────

#[test]
fn test_dijkstra_path_basic() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = DIJKSTRA_PATH({"A": [["B", 1.0], ["C", 10.0]], "B": [["C", 1.0]], "C": []}, "A", "C") SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3, "path A->B->C has 3 nodes");
            assert_eq!(arr[0], Value::String("A".to_string()));
            assert_eq!(arr[2], Value::String("C".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_shortest_path_nodes_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = SHORTEST_PATH_NODES({"A": [["B", 5.0]], "B": []}, "A", "B") SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("A".to_string()));
            assert_eq!(arr[1], Value::String("B".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_dijkstra_path_no_path() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = DIJKSTRA_PATH({"A": [], "B": []}, "A", "B") SELECT res;"#);
    match v {
        Value::Array(arr) => assert!(arr.is_empty(), "no path should return empty array"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── BELLMAN_FORD ──────────────────────────────────────────────────────────────

#[test]
fn test_bellman_ford_basic() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = BELLMAN_FORD({"A": [["B", 1.0], ["C", 4.0]], "B": [["C", 2.0]], "C": []}, "A") SELECT res;"#);
    match v {
        Value::Object(map) => {
            assert_eq!(map.get("A"), Some(&Value::Float(0.0)));
            assert_eq!(map.get("C"), Some(&Value::Float(3.0)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_bellman_ford_dist_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = BELLMAN_FORD_DIST({"A": [["B", 3.0]], "B": []}, "A") SELECT res;"#);
    match v {
        Value::Object(map) => assert_eq!(map.get("B"), Some(&Value::Float(3.0))),
        other => panic!("expected object, got {:?}", other),
    }
}

// ── ALL_PAIRS_SHORTEST ────────────────────────────────────────────────────────

#[test]
fn test_apsp_floyd_basic() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = APSP_FLOYD({"A": [["B", 1.0]], "B": [["C", 2.0]], "C": []}) SELECT res;"#);
    match v {
        Value::Object(outer) => {
            if let Some(Value::Object(from_a)) = outer.get("A") {
                assert_eq!(from_a.get("A"), Some(&Value::Float(0.0)));
                assert_eq!(from_a.get("B"), Some(&Value::Float(1.0)));
                assert_eq!(from_a.get("C"), Some(&Value::Float(3.0)));
            } else {
                panic!("expected object for key A");
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_all_pairs_shortest_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = ALL_PAIRS_SHORTEST({"P": [["Q", 5.0]], "Q": []}) SELECT res;"#);
    match v {
        Value::Object(outer) => {
            if let Some(Value::Object(from_p)) = outer.get("P") {
                assert_eq!(from_p.get("Q"), Some(&Value::Float(5.0)));
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── MAX_FLOW ──────────────────────────────────────────────────────────────────

#[test]
fn test_max_flow_basic() {
    let (db, ex) = setup();
    seed(&db);
    // Simple flow network: S->A cap 3, S->B cap 2, A->T cap 2, B->T cap 3
    let v = run(&ex, r#"QUERY t COMPUTE res = MAX_FLOW({"S": [["A", 3.0], ["B", 2.0]], "A": [["T", 2.0]], "B": [["T", 3.0]], "T": []}, "S", "T") SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 4.0).abs() < 1e-9, "max flow should be 4, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_max_flow_bfs_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = MAX_FLOW_BFS({"S": [["T", 10.0]], "T": []}, "S", "T") SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 10.0).abs() < 1e-9, "direct edge cap 10, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── MIN_CUT ───────────────────────────────────────────────────────────────────

#[test]
fn test_min_cut_equals_max_flow() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = MIN_CUT({"S": [["A", 3.0], ["B", 2.0]], "A": [["T", 2.0]], "B": [["T", 3.0]], "T": []}, "S", "T") SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 4.0).abs() < 1e-9, "min cut = max flow = 4, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_min_cut_value_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = MIN_CUT_VALUE({"S": [["T", 7.0]], "T": []}, "S", "T") SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 7.0).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── FLOW_NETWORK_CAPACITY ─────────────────────────────────────────────────────

#[test]
fn test_flow_network_capacity_basic() {
    let (db, ex) = setup();
    seed(&db);
    // Node S has edges to A(cap=3) and B(cap=2), total outgoing = 5
    let v = run(&ex, r#"QUERY t COMPUTE res = FLOW_NETWORK_CAPACITY({"S": [["A", 3.0], ["B", 2.0]], "A": [], "B": []}, "S") SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 5.0).abs() < 1e-9, "total capacity should be 5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_network_capacity_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = NETWORK_CAPACITY({"X": [["Y", 10.0]], "Y": []}, "X") SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 10.0).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── ECCENTRICITY / DIAMETER / RADIUS / CENTER ─────────────────────────────────

#[test]
fn test_eccentricity_path() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C: A has eccentricity 2, B has eccentricity 1
    let v = run(&ex, r#"QUERY t COMPUTE res = ECCENTRICITY({"A": ["B"], "B": ["A", "C"], "C": ["B"]}, "A") SELECT res;"#);
    assert_eq!(v, Value::Integer(2));
}

#[test]
fn test_node_eccentricity_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = NODE_ECCENTRICITY({"A": ["B"], "B": ["A", "C"], "C": ["B"]}, "B") SELECT res;"#);
    assert_eq!(v, Value::Integer(1));
}

#[test]
fn test_graph_diameter_path() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C-D: diameter is 3
    let v = run(&ex, r#"QUERY t COMPUTE res = GRAPH_DIAMETER({"A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(3));
}

#[test]
fn test_diameter_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = DIAMETER({"A": ["B"], "B": ["A"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(1));
}

#[test]
fn test_graph_radius_path() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C: radius=1 (B is center)
    let v = run(&ex, r#"QUERY t COMPUTE res = GRAPH_RADIUS({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(1));
}

#[test]
fn test_radius_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = RADIUS({"A": ["B"], "B": ["A"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(1));
}

#[test]
fn test_graph_center_nodes_path() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C: center is B (eccentricity 1)
    let v = run(&ex, r#"QUERY t COMPUTE res = GRAPH_CENTER_NODES({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], Value::String("B".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_center_nodes_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = CENTER_NODES({"A": ["B"], "B": ["A"]}) SELECT res;"#);
    match v {
        Value::Array(arr) => assert_eq!(arr.len(), 2, "both nodes are centers for K2"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── MST_KRUSKAL ───────────────────────────────────────────────────────────────

#[test]
fn test_mst_kruskal_basic() {
    let (db, ex) = setup();
    seed(&db);
    // 3-node graph with weights
    let v = run(&ex, r#"QUERY t COMPUTE res = MST_KRUSKAL({"A": [["B", 1.0], ["C", 4.0]], "B": [["A", 1.0], ["C", 2.0]], "C": [["A", 4.0], ["B", 2.0]]}) SELECT res;"#);
    match v {
        Value::Object(obj) => {
            match obj.get("total_weight") {
                Some(Value::Float(f)) => assert!((f - 3.0).abs() < 1e-9, "MST weight should be 3, got {}", f),
                other => panic!("expected float for total_weight, got {:?}", other),
            }
            match obj.get("edges") {
                Some(Value::Array(edges)) => assert_eq!(edges.len(), 2, "MST of 3-node graph has 2 edges"),
                other => panic!("expected array for edges, got {:?}", other),
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_min_spanning_tree_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = MIN_SPANNING_TREE({"A": [["B", 5.0]], "B": [["A", 5.0]]}) SELECT res;"#);
    match v {
        Value::Object(obj) => {
            match obj.get("total_weight") {
                Some(Value::Float(f)) => assert!((f - 5.0).abs() < 1e-9),
                other => panic!("expected float, got {:?}", other),
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── MST_WEIGHT ────────────────────────────────────────────────────────────────

#[test]
fn test_mst_weight_basic() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = MST_WEIGHT({"A": [["B", 1.0], ["C", 4.0]], "B": [["A", 1.0], ["C", 2.0]], "C": [["A", 4.0], ["B", 2.0]]}) SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 3.0).abs() < 1e-9, "MST weight should be 3"),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_minimum_spanning_weight_alias() {
    let (db, ex) = setup();
    seed(&db);
    let v = run(&ex, r#"QUERY t COMPUTE res = MINIMUM_SPANNING_WEIGHT({"A": [["B", 7.0]], "B": [["A", 7.0]]}) SELECT res;"#);
    match v {
        Value::Float(f) => assert!((f - 7.0).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── SPANNING_TREE_COUNT ───────────────────────────────────────────────────────

#[test]
fn test_spanning_tree_count_k3() {
    let (db, ex) = setup();
    seed(&db);
    // K3 (complete graph on 3 nodes): 3 spanning trees
    let v = run(&ex, r#"QUERY t COMPUTE res = SPANNING_TREE_COUNT({"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(3));
}

#[test]
fn test_num_spanning_trees_alias() {
    let (db, ex) = setup();
    seed(&db);
    // Path A-B-C: only 1 spanning tree (the path itself)
    let v = run(&ex, r#"QUERY t COMPUTE res = NUM_SPANNING_TREES({"A": ["B"], "B": ["A", "C"], "C": ["B"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(1));
}

#[test]
fn test_spanning_tree_count_k4() {
    let (db, ex) = setup();
    seed(&db);
    // K4: 16 spanning trees (Cayley's formula: n^(n-2) = 4^2 = 16)
    let v = run(&ex, r#"QUERY t COMPUTE res = SPANNING_TREE_COUNT({"A": ["B","C","D"], "B": ["A","C","D"], "C": ["A","B","D"], "D": ["A","B","C"]}) SELECT res;"#);
    assert_eq!(v, Value::Integer(16));
}
