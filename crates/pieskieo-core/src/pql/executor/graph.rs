use super::{ExecutionStats, QueryResult};
use crate::error::Result;
use crate::graph::Edge;
use crate::pql::ast::{
    Condition, Expression, GraphPattern, PathMode, TraverseDirection, TraverseMode,
};
use std::collections::HashMap;
use uuid::Uuid;

use super::{expressions, source, Executor, Row, Value};

pub(crate) fn execute_add_edge(
    executor: &Executor,
    src: Expression,
    dst: Expression,
    edge_type: Option<Expression>,
    weight: Option<Expression>,
) -> Result<QueryResult> {
    let dummy = Row {
        id: Uuid::nil(),
        data: HashMap::new(),
    };

    let src_id = eval_uuid_for_edge(executor, &src, &dummy, "src")?;
    let dst_id = eval_uuid_for_edge(executor, &dst, &dummy, "dst")?;

    let weight_val = match weight {
        Some(expr) => match expressions::evaluate_expression(executor, &expr, &dummy)? {
            Value::Float(f) => f as f32,
            Value::Integer(i) => i as f32,
            _ => 1.0,
        },
        None => 1.0,
    };

    match edge_type {
        Some(expr) => {
            let et = match expressions::evaluate_expression(executor, &expr, &dummy)? {
                Value::String(s) => s,
                other => format!("{:?}", other),
            };
            executor.db.add_typed_edge(src_id, dst_id, weight_val, et)?;
        }
        None => {
            executor.db.add_edge(src_id, dst_id, weight_val)?;
        }
    }

    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

fn eval_uuid_for_edge(
    executor: &Executor,
    expr: &Expression,
    row: &Row,
    label: &str,
) -> Result<Uuid> {
    match expressions::evaluate_expression(executor, expr, row)? {
        Value::Uuid(u) => Ok(u),
        Value::String(s) => Uuid::parse_str(&s).map_err(|e| {
            crate::error::PieskieoError::Validation(format!("Invalid UUID for {}: {}", label, e))
        }),
        other => Err(crate::error::PieskieoError::Validation(format!(
            "ADD EDGE {} must be a UUID, got {:?}",
            label, other
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_traverse(
    executor: &Executor,
    input: Vec<Row>,
    edge_type: Option<String>,
    edge_filter: Option<Condition>,
    min_depth: usize,
    max_depth: usize,
    direction: TraverseDirection,
    mode: TraverseMode,
) -> Result<Vec<Row>> {
    let mut result_rows = Vec::new();

    for start_row in input {
        let start_id = start_row.id;
        let mut queue = std::collections::VecDeque::new();
        let mut visited: HashMap<Uuid, usize> = HashMap::new();
        queue.push_back((start_id, 0usize));
        visited.insert(start_id, 0);

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let neighbors = get_neighbors(executor, node, direction);
            for edge in neighbors {
                if let Some(ref t) = edge_type {
                    if edge.edge_type.as_deref() != Some(t.as_str()) {
                        continue;
                    }
                }
                let next_depth = depth + 1;
                if next_depth > max_depth {
                    continue;
                }
                if let Some(ref filter) = edge_filter {
                    let edge_row = edge_to_row(&edge);
                    if !expressions::evaluate_condition(executor, filter, &edge_row) {
                        continue;
                    }
                }
                let is_new = match visited.get(&edge.dst) {
                    None => true,
                    Some(prev) => next_depth < *prev && matches!(mode, TraverseMode::Shortest),
                };
                if is_new {
                    visited.insert(edge.dst, next_depth);
                    if next_depth >= min_depth {
                        if let Some(doc) = executor.db.get_doc(&edge.dst) {
                            result_rows.push(source::json_to_row(edge.dst, doc));
                        } else if let Some(row_data) = executor.db.get_row(&edge.dst) {
                            result_rows.push(source::json_to_row(edge.dst, row_data));
                        }
                    }
                    queue.push_back((edge.dst, next_depth));
                }
            }
        }
    }

    Ok(result_rows)
}

pub(crate) fn execute_path(
    executor: &Executor,
    input: Vec<Row>,
    mode: PathMode,
    from: Expression,
    to: Expression,
    max_depth: usize,
    edge_type: Option<String>,
) -> Result<Vec<Row>> {
    let direction = TraverseDirection::Both;
    let edge_filter: Option<Condition> = None;
    let mut output = Vec::new();

    for row in input {
        let from_id = evaluate_uuid_expr(executor, &from, &row)?;
        let to_id = evaluate_uuid_expr(executor, &to, &row)?;
        if from_id.is_none() || to_id.is_none() {
            continue;
        }
        let from_id = from_id.unwrap();
        let to_id = to_id.unwrap();

        let paths: Vec<Vec<Edge>> = match mode {
            PathMode::Shortest => find_shortest_path(
                executor,
                from_id,
                to_id,
                max_depth,
                direction,
                edge_type.as_deref(),
                edge_filter.as_ref(),
            )
            .map(|p| vec![p])
            .unwrap_or_default(),
            PathMode::AllSimple | PathMode::Any => find_all_paths(
                executor,
                from_id,
                to_id,
                max_depth,
                direction,
                edge_type.as_deref(),
                edge_filter.as_ref(),
            ),
        };

        for edges in paths {
            if edges.is_empty() {
                continue;
            }
            let dest = edges.last().map(|e| e.dst).unwrap_or(from_id);
            let mut data = HashMap::new();
            let path_nodes: Vec<Value> = std::iter::once(from_id)
                .chain(edges.iter().map(|e| e.dst))
                .map(Value::Uuid)
                .collect();
            let path_edges: Vec<Value> = edges
                .iter()
                .map(|e| {
                    let mut m = HashMap::new();
                    m.insert("src".to_string(), Value::Uuid(e.src));
                    m.insert("dst".to_string(), Value::Uuid(e.dst));
                    m.insert("weight".to_string(), Value::Float(e.weight as f64));
                    if let Some(t) = &e.edge_type {
                        m.insert("type".to_string(), Value::String(t.clone()));
                    }
                    Value::Object(m)
                })
                .collect();
            let path_cost: f64 = edges.iter().map(|e| e.weight as f64).sum();

            data.insert("_path_nodes".to_string(), Value::Array(path_nodes));
            data.insert("_path_edges".to_string(), Value::Array(path_edges));
            data.insert(
                "_path_length".to_string(),
                Value::Integer(edges.len() as i64),
            );
            data.insert("_path_cost".to_string(), Value::Float(path_cost));

            if let Some(doc) = executor.db.get_doc(&dest) {
                source::merge_json_into_data(&mut data, doc);
            } else if let Some(row_data) = executor.db.get_row(&dest) {
                source::merge_json_into_data(&mut data, row_data);
            }

            output.push(Row { id: dest, data });
        }
    }

    Ok(output)
}

pub(crate) fn execute_match(
    executor: &Executor,
    input: Vec<Row>,
    pattern: GraphPattern,
) -> Result<Vec<Row>> {
    if pattern.edges.is_empty() {
        return Ok(input);
    }
    let mut output = Vec::new();
    for row in input {
        if let Some(first) = pattern.nodes.first() {
            if !node_matches(executor, row.id, first)? {
                continue;
            }
        }
        match_dfs(executor, row.id, &pattern, 0, &mut output)?;
    }
    Ok(output)
}

fn match_dfs(
    executor: &Executor,
    current: Uuid,
    pattern: &GraphPattern,
    edge_idx: usize,
    output: &mut Vec<Row>,
) -> Result<()> {
    if edge_idx >= pattern.edges.len() {
        if let Some(doc) = executor.db.get_any_doc(&current) {
            output.push(source::json_to_row(current, doc));
        } else if let Some(r) = executor.db.get_any_row(&current) {
            output.push(source::json_to_row(current, r));
        }
        return Ok(());
    }

    let edge = &pattern.edges[edge_idx];
    let neighbors = get_neighbors(executor, current, edge.direction);

    for e in neighbors {
        if let Some(ref t) = edge.edge_type {
            if e.edge_type.as_deref() != Some(t.as_str()) {
                continue;
            }
        }
        if let Some(ref cond) = edge.properties {
            let edge_row = edge_to_row(&e);
            if !expressions::evaluate_condition(executor, cond, &edge_row) {
                continue;
            }
        }
        if let Some(node) = pattern.nodes.get(edge_idx + 1) {
            if !node_matches(executor, e.dst, node)? {
                continue;
            }
        }
        match_dfs(executor, e.dst, pattern, edge_idx + 1, output)?;
    }

    Ok(())
}

fn node_matches(
    executor: &Executor,
    id: Uuid,
    node: &crate::pql::ast::NodePattern,
) -> Result<bool> {
    let row = if let Some(doc) = executor.db.get_any_doc(&id) {
        source::json_to_row(id, doc)
    } else if let Some(r) = executor.db.get_any_row(&id) {
        source::json_to_row(id, r)
    } else {
        return Ok(false);
    };

    if !node.labels.is_empty() {
        let label_match = match row.data.get("label") {
            Some(Value::String(s)) => node.labels.iter().any(|l| l == s),
            Some(Value::Array(arr)) => arr.iter().any(|v| {
                if let Value::String(s) = v {
                    node.labels.iter().any(|l| l == s)
                } else {
                    false
                }
            }),
            _ => false,
        };
        if !label_match {
            return Ok(false);
        }
    }

    if let Some(cond) = &node.properties {
        if !expressions::evaluate_condition(executor, cond, &row) {
            return Ok(false);
        }
    }

    Ok(true)
}

fn find_shortest_path(
    executor: &Executor,
    start: Uuid,
    goal: Uuid,
    max_depth: usize,
    direction: TraverseDirection,
    edge_type: Option<&str>,
    edge_filter: Option<&Condition>,
) -> Option<Vec<Edge>> {
    let mut queue = std::collections::VecDeque::new();
    let mut visited: HashMap<Uuid, usize> = HashMap::new();
    let mut parent: HashMap<Uuid, Edge> = HashMap::new();

    queue.push_back(start);
    visited.insert(start, 0);

    while let Some(node) = queue.pop_front() {
        let depth = *visited.get(&node).unwrap_or(&0);
        if depth >= max_depth {
            continue;
        }
        for edge in get_neighbors(executor, node, direction) {
            if let Some(t) = edge_type {
                if edge.edge_type.as_deref() != Some(t) {
                    continue;
                }
            }
            if let Some(filter) = edge_filter {
                let edge_row = edge_to_row(&edge);
                if !expressions::evaluate_condition(executor, filter, &edge_row) {
                    continue;
                }
            }
            if visited.contains_key(&edge.dst) {
                continue;
            }
            visited.insert(edge.dst, depth + 1);
            parent.insert(edge.dst, edge.clone());
            if edge.dst == goal {
                return Some(reconstruct_path(start, goal, &parent));
            }
            queue.push_back(edge.dst);
        }
    }

    None
}

fn reconstruct_path(start: Uuid, goal: Uuid, parent: &HashMap<Uuid, Edge>) -> Vec<Edge> {
    let mut path_edges = Vec::new();
    let mut current = goal;
    while current != start {
        if let Some(edge) = parent.get(&current) {
            path_edges.push(edge.clone());
            current = edge.src;
        } else {
            break;
        }
    }
    path_edges.reverse();
    path_edges
}

fn find_all_paths(
    executor: &Executor,
    start: Uuid,
    goal: Uuid,
    max_depth: usize,
    direction: TraverseDirection,
    edge_type: Option<&str>,
    edge_filter: Option<&Condition>,
) -> Vec<Vec<Edge>> {
    let mut paths = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut stack = Vec::new();
    dfs_paths(
        executor,
        start,
        goal,
        max_depth,
        direction,
        edge_type,
        edge_filter,
        &mut visited,
        &mut stack,
        &mut paths,
    );
    paths
}

#[allow(clippy::too_many_arguments)]
fn dfs_paths(
    executor: &Executor,
    current: Uuid,
    goal: Uuid,
    max_depth: usize,
    direction: TraverseDirection,
    edge_type: Option<&str>,
    edge_filter: Option<&Condition>,
    visited: &mut std::collections::HashSet<Uuid>,
    stack: &mut Vec<Edge>,
    out: &mut Vec<Vec<Edge>>,
) {
    if stack.len() > max_depth {
        return;
    }
    if current == goal && !stack.is_empty() {
        out.push(stack.clone());
        return;
    }
    if !visited.insert(current) {
        return;
    }
    for edge in get_neighbors(executor, current, direction) {
        if let Some(t) = edge_type {
            if edge.edge_type.as_deref() != Some(t) {
                continue;
            }
        }
        if let Some(filter) = edge_filter {
            let edge_row = edge_to_row(&edge);
            if !expressions::evaluate_condition(executor, filter, &edge_row) {
                continue;
            }
        }
        stack.push(edge.clone());
        dfs_paths(
            executor,
            edge.dst,
            goal,
            max_depth,
            direction,
            edge_type,
            edge_filter,
            visited,
            stack,
            out,
        );
        stack.pop();
    }
    visited.remove(&current);
}

fn evaluate_uuid_expr(executor: &Executor, expr: &Expression, row: &Row) -> Result<Option<Uuid>> {
    let value = expressions::evaluate_expression(executor, expr, row)?;
    match value {
        Value::Uuid(u) => Ok(Some(u)),
        Value::String(s) => Ok(Uuid::parse_str(&s).ok()),
        _ => Ok(None),
    }
}

fn get_neighbors(executor: &Executor, node: Uuid, direction: TraverseDirection) -> Vec<Edge> {
    match direction {
        TraverseDirection::Outgoing => executor.db.neighbors(node, usize::MAX),
        TraverseDirection::Incoming => executor.db.neighbors_in(node, usize::MAX),
        TraverseDirection::Both => executor.db.neighbors_both(node, usize::MAX),
    }
}

fn edge_to_row(edge: &Edge) -> Row {
    let mut data = HashMap::new();
    data.insert("src".to_string(), Value::Uuid(edge.src));
    data.insert("dst".to_string(), Value::Uuid(edge.dst));
    data.insert("weight".to_string(), Value::Float(edge.weight as f64));
    if let Some(t) = &edge.edge_type {
        data.insert("type".to_string(), Value::String(t.clone()));
    }
    Row { id: edge.dst, data }
}

pub(crate) fn execute_remove_edge(
    executor: &Executor,
    src: Expression,
    dst: Expression,
) -> Result<QueryResult> {
    let dummy = Row {
        id: Uuid::nil(),
        data: HashMap::new(),
    };
    let src_val = expressions::evaluate_expression(executor, &src, &dummy)?;
    let dst_val = expressions::evaluate_expression(executor, &dst, &dummy)?;
    let src_id = match src_val {
        Value::Uuid(u) => u,
        Value::String(s) => Uuid::parse_str(&s)
            .map_err(|e| crate::error::PieskieoError::Validation(format!("Invalid UUID: {}", e)))?,
        other => {
            return Err(crate::error::PieskieoError::Validation(format!(
                "REMOVE EDGE src must be a UUID, got {:?}",
                other
            )))
        }
    };
    let dst_id = match dst_val {
        Value::Uuid(u) => u,
        Value::String(s) => Uuid::parse_str(&s)
            .map_err(|e| crate::error::PieskieoError::Validation(format!("Invalid UUID: {}", e)))?,
        other => {
            return Err(crate::error::PieskieoError::Validation(format!(
                "REMOVE EDGE dst must be a UUID, got {:?}",
                other
            )))
        }
    };
    executor.db.remove_edge(src_id, dst_id)?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}
