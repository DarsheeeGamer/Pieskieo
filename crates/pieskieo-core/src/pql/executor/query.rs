use crate::error::Result;
use crate::pql::ast::{
    ComparisonOp, Condition, Cte, Expression, Operation, SetOperator, SetOperator as SetOp,
    SourceExpr, Statement,
};

use super::{
    expressions, graph, joins, operations, source, vector, ExecutionStats, Executor, QueryResult,
    Row,
};

pub(super) fn execute_query(
    executor: &Executor,
    with: Vec<Cte>,
    source_expr: SourceExpr,
    ops: Vec<Operation>,
) -> Result<QueryResult> {
    let mut stats = ExecutionStats::default();

    // Pre-compute CTEs and store them in the executor's CTE map
    for cte in with {
        if !cte.recursive {
            let result = executor.execute(*cte.statement)?;
            executor.ctes.write().insert(cte.name, result.rows);
        } else {
            // Recursive CTE: anchor + iterative expansion
            match *cte.statement {
                Statement::SetOperation {
                    op: SetOp::Union,
                    all: true,
                    left,
                    right,
                } => {
                    // Anchor = left side (evaluated without the CTE in scope)
                    let anchor_result = executor.execute(*left)?;
                    let mut accumulated = anchor_result.rows.clone();
                    executor
                        .ctes
                        .write()
                        .insert(cte.name.clone(), accumulated.clone());

                    // Recursive step: keep evaluating right side until no new rows
                    let max_iterations = 1000; // prevent infinite recursion
                    for _ in 0..max_iterations {
                        let recursive_result = executor.execute(*right.clone())?;
                        let new_rows: Vec<Row> = recursive_result
                            .rows
                            .into_iter()
                            .filter(|r| !accumulated.iter().any(|a| a.id == r.id))
                            .collect();
                        if new_rows.is_empty() {
                            break;
                        }
                        accumulated.extend(new_rows);
                        executor
                            .ctes
                            .write()
                            .insert(cte.name.clone(), accumulated.clone());
                    }
                }
                other => {
                    // Non-union recursive CTE: just execute normally
                    let result = executor.execute(other)?;
                    executor.ctes.write().insert(cte.name, result.rows);
                }
            }
        }
    }

    // Fast-path for VectorSearch, Traverse, and FulltextSearch as the first operation.
    // This avoids a full collection scan when we can use a specialized index (HNSW, BM25, Graph).
    let collection_name = match &source_expr {
        SourceExpr::Collection(name) => name.clone(),
        SourceExpr::CollectionAs { name, .. } => name.clone(),
        SourceExpr::Subquery { alias, .. } => alias.clone().unwrap_or_default(),
        SourceExpr::Cte(name) => name.clone(),
        SourceExpr::Values { alias, .. } => alias.clone().unwrap_or_default(),
    };

    let mut ops_iter = ops.into_iter();
    let mut current_rows = match ops_iter.next() {
        Some(Operation::VectorSearch {
            query_vector,
            field,
            top_k,
            threshold,
            metric,
        }) => {
            stats.vector_searches += 1;
            vector::execute_vector_search(
                executor,
                vec![],
                query_vector,
                field,
                top_k,
                threshold,
                metric,
            )?
        }
        Some(Operation::Traverse {
            edge_type,
            edge_filter,
            min_depth,
            max_depth,
            direction,
            mode,
        }) => {
            // Traverse from collection: this means traverse from ALL nodes in the collection
            let start_rows = source::load_source(executor, &source_expr, &mut stats, None)?;
            stats.graph_traversals += 1;
            graph::execute_traverse(
                executor,
                start_rows,
                edge_type,
                edge_filter,
                min_depth,
                max_depth,
                direction,
                mode,
            )?
        }
        Some(Operation::FulltextSearch {
            query,
            field,
            top_k,
        }) => vector::execute_fulltext_search(
            executor,
            &collection_name,
            vec![],
            query,
            field,
            top_k,
        )?,
        Some(
            first_op @ Operation::Filter(Condition::Comparison {
                op: ComparisonOp::Equal,
                left: Expression::FieldAccess(ref path),
                right: Expression::Literal(ref lit),
            }),
        ) => {
            if path.len() == 1 {
                let mut map = std::collections::HashMap::new();
                map.insert(path[0].clone(), source::literal_to_json(lit.clone()));
                source::load_source(executor, &source_expr, &mut stats, Some(&map))?
            } else {
                let start_rows = source::load_source(executor, &source_expr, &mut stats, None)?;
                execute_operation(executor, &collection_name, first_op, start_rows, &mut stats)?
            }
        }
        Some(first_op) => {
            let start_rows = source::load_source(executor, &source_expr, &mut stats, None)?;
            execute_operation(executor, &collection_name, first_op, start_rows, &mut stats)?
        }
        None => source::load_source(executor, &source_expr, &mut stats, None)?,
    };

    for operation in ops_iter {
        current_rows = execute_operation(
            executor,
            &collection_name,
            operation,
            current_rows,
            &mut stats,
        )?;
    }

    // Remove internal group-rows key so it never leaks into query results
    for row in &mut current_rows {
        row.data.remove(super::Executor::GROUP_ROWS_KEY);
    }

    let columns = if !current_rows.is_empty() {
        current_rows[0].data.keys().cloned().collect()
    } else {
        Vec::new()
    };

    Ok(QueryResult {
        rows: current_rows,
        columns,
        stats,
    })
}

pub(crate) fn execute_set_operation(
    executor: &Executor,
    op: SetOperator,
    all: bool,
    left: Statement,
    right: Statement,
) -> Result<QueryResult> {
    let left_result = executor.execute(left)?;
    let right_result = executor.execute(right)?;

    let left_rows = left_result.rows;
    let right_rows = right_result.rows;

    let combined = match op {
        SetOperator::Union => {
            let mut out = left_rows;
            out.extend(right_rows);
            if !all {
                out = operations::execute_distinct(out);
            }
            out
        }
        SetOperator::Intersect => {
            // Keep rows from left that appear in right
            let right_keys: Vec<String> = right_rows.iter().map(row_content_key).collect();
            let mut out: Vec<Row> = left_rows
                .into_iter()
                .filter(|row| right_keys.contains(&row_content_key(row)))
                .collect();
            if !all {
                out = operations::execute_distinct(out);
            }
            out
        }
        SetOperator::Except => {
            // Keep rows from left that do NOT appear in right
            let right_keys: Vec<String> = right_rows.iter().map(row_content_key).collect();
            let mut out: Vec<Row> = left_rows
                .into_iter()
                .filter(|row| !right_keys.contains(&row_content_key(row)))
                .collect();
            if !all {
                out = operations::execute_distinct(out);
            }
            out
        }
    };

    let columns = if !combined.is_empty() {
        combined[0].data.keys().cloned().collect()
    } else {
        Vec::new()
    };

    Ok(QueryResult {
        rows: combined,
        columns,
        stats: ExecutionStats::default(),
    })
}

fn row_content_key(row: &Row) -> String {
    let mut keys: Vec<&String> = row.data.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| format!("{}={}", k, value_key_repr(&row.data[*k])))
        .collect::<Vec<_>>()
        .join(";")
}

fn value_key_repr(val: &super::Value) -> String {
    use super::Value;
    match val {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => format!("{:.15}", f),
        Value::String(s) => format!("\"{}\"", s),
        Value::Uuid(u) => u.to_string(),
        Value::Vector(v) => format!("{:?}", v),
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(value_key_repr).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Object(obj) => {
            let mut ks: Vec<&String> = obj.keys().collect();
            ks.sort();
            let parts: Vec<String> = ks
                .iter()
                .map(|k| format!("{}:{}", k, value_key_repr(&obj[*k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn execute_operation(
    executor: &Executor,
    collection: &str,
    operation: Operation,
    input: Vec<Row>,
    stats: &mut ExecutionStats,
) -> Result<Vec<Row>> {
    match operation {
        Operation::Filter(condition) => {
            let filtered: Vec<Row> = input
                .into_iter()
                .filter(|row| expressions::evaluate_condition(executor, &condition, row))
                .collect();
            stats.rows_filtered += filtered.len();
            Ok(filtered)
        }

        Operation::Distinct => Ok(operations::execute_distinct(input)),

        Operation::VectorSearch {
            query_vector,
            field,
            top_k,
            threshold,
            metric,
        } => {
            stats.vector_searches += 1;
            vector::execute_vector_search(
                executor,
                input,
                query_vector,
                field,
                top_k,
                threshold,
                metric,
            )
        }

        Operation::HybridSearch {
            query,
            field,
            top_k,
            alpha,
        } => {
            stats.vector_searches += 1;
            vector::execute_hybrid_search(executor, collection, input, query, field, top_k, alpha)
        }

        Operation::Traverse {
            edge_type,
            edge_filter,
            min_depth,
            max_depth,
            direction,
            mode,
        } => {
            stats.graph_traversals += 1;
            graph::execute_traverse(
                executor,
                input,
                edge_type,
                edge_filter,
                min_depth,
                max_depth,
                direction,
                mode,
            )
        }

        Operation::Path {
            mode,
            from,
            to,
            max_depth,
            edge_type,
        } => {
            stats.graph_traversals += 1;
            graph::execute_path(executor, input, mode, from, to, max_depth, edge_type)
        }

        Operation::Match { pattern } => graph::execute_match(executor, input, pattern),

        Operation::Join {
            join_type,
            source,
            condition,
        } => joins::execute_join(executor, input, join_type, *source, condition),

        Operation::GroupBy { fields, mode } => {
            operations::execute_group_by(executor, input, fields, mode)
        }

        Operation::Having(condition) => {
            // Filter rows where the group-level aggregates satisfy the condition
            // After GROUP BY + COMPUTE, each row IS a group representative
            Ok(input
                .into_iter()
                .filter(|row| expressions::evaluate_condition(executor, &condition, row))
                .collect())
        }

        Operation::Compute { assignments } => {
            operations::execute_compute(executor, input, assignments)
        }

        Operation::OrderBy { fields } => operations::execute_order_by(executor, input, fields),

        Operation::Limit { count, offset } => {
            let offset = offset.unwrap_or(0);
            Ok(input.into_iter().skip(offset).take(count).collect())
        }

        Operation::Select { fields } => operations::execute_select(executor, input, fields),

        Operation::FulltextSearch {
            query,
            field,
            top_k,
        } => vector::execute_fulltext_search(executor, collection, input, query, field, top_k),

        Operation::Unnest {
            field,
            alias,
            index_field,
            preserve,
        } => operations::execute_unnest(executor, input, field, alias, index_field, preserve),

        Operation::Pivot {
            value_field,
            pivot_field,
            pivot_values,
            aggregate,
        } => operations::execute_pivot(
            executor,
            input,
            value_field,
            pivot_field,
            pivot_values,
            aggregate,
        ),

        Operation::Qualify { condition } => operations::execute_qualify(executor, input, condition),

        Operation::Sample { count } => operations::execute_sample(input, count),
    }
}
