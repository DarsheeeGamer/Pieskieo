use crate::error::Result;
use crate::pql::ast::{Operation, Statement};
use std::collections::HashMap;
use uuid::Uuid;

use super::{ExecutionStats, Executor, QueryResult, Row, Value};

pub(super) fn execute_explain(
    executor: &Executor,
    analyze: bool,
    statement: Statement,
) -> Result<QueryResult> {
    let plan = build_plan(&statement);
    let mut data = HashMap::new();
    data.insert("plan".to_string(), Value::String(plan));

    if analyze {
        let result = executor.execute(statement)?;
        data.insert(
            "analyze".to_string(),
            Value::Object(
                [
                    (
                        "rows_scanned".to_string(),
                        Value::Integer(result.stats.rows_scanned as i64),
                    ),
                    (
                        "rows_filtered".to_string(),
                        Value::Integer(result.stats.rows_filtered as i64),
                    ),
                    (
                        "vector_searches".to_string(),
                        Value::Integer(result.stats.vector_searches as i64),
                    ),
                    (
                        "graph_traversals".to_string(),
                        Value::Integer(result.stats.graph_traversals as i64),
                    ),
                    (
                        "execution_time_ms".to_string(),
                        Value::Integer(result.stats.execution_time_ms as i64),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        );
    }

    Ok(QueryResult {
        rows: vec![Row {
            id: Uuid::nil(),
            data,
        }],
        columns: vec!["plan".to_string(), "analyze".to_string()],
        stats: ExecutionStats::default(),
    })
}

fn build_plan(statement: &Statement) -> String {
    match statement {
        Statement::Query {
            with,
            source,
            operations,
        } => {
            let mut lines = Vec::new();
            if !with.is_empty() {
                lines.push(format!(
                    "WITH {}",
                    with.iter()
                        .map(|c| c.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            lines.push(format!("QUERY SOURCE: {:?}", source));
            for op in operations {
                let (desc, cost) = describe_operation(op);
                lines.push(format!("  OP: {} | est_cost={:.2}", desc, cost));
            }
            lines.join("\n")
        }
        Statement::Insert { target, rows, .. } => {
            format!("INSERT INTO {} rows={}", target, rows.len())
        }
        Statement::Update {
            target,
            assignments,
            ..
        } => {
            format!("UPDATE {} assignments={}", target, assignments.len())
        }
        Statement::Delete { target, .. } => format!("DELETE FROM {}", target),
        Statement::Create(stmt) => format!("CREATE {:?}", stmt),
        Statement::AlterTable { name, operations } => {
            format!("ALTER TABLE {} ops={}", name, operations.len())
        }
        Statement::DropIndex { name, on } => {
            format!(
                "DROP INDEX {} ON {}",
                name,
                on.as_deref().unwrap_or("<unspecified>")
            )
        }
        Statement::DropCollection {
            name,
            is_table,
            cascade,
        } => {
            let kind = if *is_table { "TABLE" } else { "COLLECTION" };
            let suffix = if *cascade { " CASCADE" } else { "" };
            format!("DROP {} {}{}", kind, name, suffix)
        }
        Statement::Explain { .. } => "EXPLAIN".to_string(),
        Statement::CreateView { name, .. } => format!("CREATE VIEW {}", name),
        Statement::DropView { name, .. } => format!("DROP VIEW {}", name),
        Statement::Begin => "BEGIN".to_string(),
        Statement::Commit => "COMMIT".to_string(),
        Statement::Rollback { to } => match to {
            Some(sp) => format!("ROLLBACK TO {}", sp),
            None => "ROLLBACK".to_string(),
        },
        Statement::Savepoint { name } => format!("SAVEPOINT {}", name),
        Statement::ReleaseSavepoint { name } => format!("RELEASE SAVEPOINT {}", name),
        Statement::RemoveEdge { .. } => "REMOVE EDGE".to_string(),
        Statement::Merge { target, .. } => format!("MERGE INTO {}", target),
        Statement::InsertSelect { target, .. } => format!("INSERT INTO {} SELECT ...", target),
        Statement::AddEdge { .. } => "ADD EDGE".to_string(),
        Statement::Truncate { name, .. } => format!("TRUNCATE {}", name),
        Statement::Show(target) => format!("SHOW {:?}", target),
        Statement::CreateSequence { name, .. } => format!("CREATE SEQUENCE {}", name),
        Statement::DropSequence { name, .. } => format!("DROP SEQUENCE {}", name),
        Statement::CopyFrom {
            collection, path, ..
        } => format!("COPY {} FROM '{}'", collection, path),
        Statement::CopyTo {
            collection, path, ..
        } => format!("COPY {} TO '{}'", collection, path),
        Statement::SetOperation {
            op,
            all,
            left,
            right,
        } => {
            let op_name = match op {
                crate::pql::ast::SetOperator::Union => {
                    if *all {
                        "UNION ALL"
                    } else {
                        "UNION"
                    }
                }
                crate::pql::ast::SetOperator::Intersect => {
                    if *all {
                        "INTERSECT ALL"
                    } else {
                        "INTERSECT"
                    }
                }
                crate::pql::ast::SetOperator::Except => {
                    if *all {
                        "EXCEPT ALL"
                    } else {
                        "EXCEPT"
                    }
                }
            };
            format!(
                "{}\n  LEFT: {}\n  RIGHT: {}",
                op_name,
                build_plan(left),
                build_plan(right)
            )
        }
    }
}

fn describe_operation(op: &Operation) -> (String, f64) {
    match op {
        Operation::Filter(_) => ("FILTER".to_string(), 1.0),
        Operation::Distinct => ("DISTINCT".to_string(), 1.0),
        Operation::VectorSearch { top_k, .. } => (
            format!("VECTOR SEARCH top_k={}", top_k),
            *top_k as f64 * 0.5,
        ),
        Operation::HybridSearch { top_k, .. } => (
            format!("HYBRID SEARCH top_k={}", top_k),
            *top_k as f64 * 0.8,
        ),
        Operation::Traverse {
            min_depth,
            max_depth,
            ..
        } => (
            format!("TRAVERSE depth={}..{}", min_depth, max_depth),
            (*max_depth as f64).powi(2),
        ),
        Operation::Path { max_depth, .. } => (
            format!("PATH depth={}", max_depth),
            (*max_depth as f64).powi(2),
        ),
        Operation::Match { .. } => ("MATCH".to_string(), 50.0),
        Operation::Join { join_type, .. } => (format!("JOIN {:?}", join_type), 20.0),
        Operation::GroupBy { fields, mode } => {
            use crate::pql::ast::GroupByMode;
            let mode_str = match mode {
                GroupByMode::Regular => "GROUP BY",
                GroupByMode::Rollup => "GROUP BY ROLLUP",
                GroupByMode::Cube => "GROUP BY CUBE",
            };
            (format!("{} {}", mode_str, fields.len()), 10.0)
        }
        Operation::Having(_) => ("HAVING".to_string(), 1.0),
        Operation::Compute { assignments } => (format!("COMPUTE {}", assignments.len()), 5.0),
        Operation::OrderBy { fields } => (format!("ORDER BY {}", fields.len()), 8.0),
        Operation::Limit { count, .. } => (format!("LIMIT {}", count), 0.1),
        Operation::Select { fields } => (format!("SELECT {}", fields.len()), 0.5),
        Operation::FulltextSearch { top_k, .. } => (
            format!("FULLTEXT SEARCH top_k={}", top_k),
            *top_k as f64 * 0.3,
        ),
        Operation::Unnest { alias, .. } => (
            format!("UNNEST -> {}", alias.as_deref().unwrap_or("value")),
            2.0,
        ),
        Operation::Pivot { aggregate, .. } => (
            format!("PIVOT (aggregate)"),
            5.0,
        ),
        Operation::Qualify { .. } => ("QUALIFY (window filter)".to_string(), 1.0),
        Operation::Sample { count } => (format!("SAMPLE {}", count), 1.0),
    }
}
