use crate::engine::{
    ConstraintDef, ForeignKeyDef, IndexDef, ReferentialAction, SchemaDef, SchemaField,
};
use crate::error::{PieskieoError, Result};
use crate::pql::ast::{
    AlterTableOperation, Constraint, CreateStatement, DataType, PropertyDef,
    ReferentialAction as AstReferentialAction,
};
use std::collections::HashMap;

use super::{source, ExecutionStats, Executor, QueryResult};

pub(super) fn execute_create(executor: &Executor, stmt: CreateStatement) -> Result<QueryResult> {
    match stmt {
        CreateStatement::Collection {
            name,
            fields,
            constraints,
        } => {
            let schema = build_schema_def(fields, constraints, Vec::new());
            executor.db.set_doc_schema(None, Some(&name), schema)?;
        }
        CreateStatement::Table {
            name,
            columns,
            constraints,
        } => {
            let fields = columns
                .into_iter()
                .map(|c| PropertyDef {
                    name: c.name,
                    data_type: c.data_type,
                    required: !c.nullable,
                    unique: c.primary_key || c.unique,
                    default: c.default,
                })
                .collect();
            let schema = build_schema_def(fields, constraints, Vec::new());
            executor.db.set_row_schema(None, Some(&name), schema)?;
        }
        CreateStatement::Index {
            name,
            on,
            fields,
            index_type,
        } => {
            executor
                .db
                .create_index(None, &name, &on, fields, index_type)?;
        }
        CreateStatement::NodeType {
            name,
            properties,
            constraints,
        } => {
            let schema = build_schema_def(properties, constraints, Vec::new());
            executor.db.set_doc_schema(None, Some(&name), schema)?;
        }
        CreateStatement::EdgeType {
            name,
            properties,
            constraints,
            ..
        } => {
            let schema = build_schema_def(properties, constraints, Vec::new());
            executor.db.set_doc_schema(None, Some(&name), schema)?;
        }
    }

    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_alter_table(
    executor: &Executor,
    name: String,
    operations: Vec<AlterTableOperation>,
) -> Result<QueryResult> {
    executor.db.alter_table(None, &name, operations)?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_drop_index(
    executor: &Executor,
    name: String,
    on: Option<String>,
) -> Result<QueryResult> {
    let target =
        on.ok_or_else(|| PieskieoError::Validation("DROP INDEX requires ON <table>".into()))?;
    executor.db.drop_index(None, &target, &name)?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_drop_collection(
    executor: &Executor,
    name: String,
    is_table: bool,
    cascade: bool,
) -> Result<QueryResult> {
    if is_table {
        executor.db.drop_table(None, &name, cascade)?;
    } else {
        executor.db.drop_collection(None, &name, cascade)?;
    }
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_create_view(
    executor: &Executor,
    name: String,
    _if_not_exists: bool,
    query: crate::pql::ast::Statement,
) -> Result<QueryResult> {
    if _if_not_exists && executor.db.get_view(&name).is_some() {
        return Ok(QueryResult {
            rows: Vec::new(),
            columns: vec![],
            stats: ExecutionStats::default(),
        });
    }
    executor.db.create_view(&name, &query)?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_drop_view(
    executor: &Executor,
    name: String,
    if_exists: bool,
) -> Result<QueryResult> {
    if if_exists && executor.db.get_view(&name).is_none() {
        return Ok(QueryResult {
            rows: Vec::new(),
            columns: vec![],
            stats: ExecutionStats::default(),
        });
    }
    executor.db.drop_view(&name)?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_truncate(
    executor: &Executor,
    name: String,
    is_table: bool,
) -> Result<QueryResult> {
    let mut stats = ExecutionStats::default();
    let rows = source::load_source(
        executor,
        &crate::pql::ast::SourceExpr::Collection(name.clone()),
        &mut stats,
        None,
    )?;
    let count = rows.len();
    for row in rows {
        if is_table {
            executor.db.delete_row_ns(None, Some(&name), &row.id)?;
        } else {
            executor.db.delete_doc_ns(None, Some(&name), &row.id)?;
        }
    }
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec!["truncated".to_string()],
        stats: super::ExecutionStats {
            rows_filtered: count,
            ..stats
        },
    })
}

pub(super) fn execute_show(
    executor: &Executor,
    target: crate::pql::ast::ShowTarget,
) -> Result<QueryResult> {
    use crate::pql::ast::ShowTarget;
    match target {
        ShowTarget::Collections => {
            let mut names = executor.db.list_collections(None);
            names.sort();
            let rows = names
                .into_iter()
                .map(|name| {
                    let mut data = std::collections::HashMap::new();
                    data.insert("name".to_string(), super::Value::String(name));
                    super::Row {
                        id: uuid::Uuid::new_v4(),
                        data,
                    }
                })
                .collect();
            Ok(QueryResult {
                rows,
                columns: vec!["name".to_string()],
                stats: ExecutionStats::default(),
            })
        }
        ShowTarget::Tables => {
            let mut names = executor.db.list_tables(None);
            names.sort();
            let rows = names
                .into_iter()
                .map(|name| {
                    let mut data = std::collections::HashMap::new();
                    data.insert("name".to_string(), super::Value::String(name));
                    super::Row {
                        id: uuid::Uuid::new_v4(),
                        data,
                    }
                })
                .collect();
            Ok(QueryResult {
                rows,
                columns: vec!["name".to_string()],
                stats: ExecutionStats::default(),
            })
        }
        ShowTarget::Indexes { on } => {
            let mut names = executor.db.list_indexes(None, &on);
            names.sort();
            let rows = names
                .into_iter()
                .map(|name| {
                    let mut data = std::collections::HashMap::new();
                    data.insert("name".to_string(), super::Value::String(name));
                    super::Row {
                        id: uuid::Uuid::new_v4(),
                        data,
                    }
                })
                .collect();
            Ok(QueryResult {
                rows,
                columns: vec!["name".to_string()],
                stats: ExecutionStats::default(),
            })
        }
        ShowTarget::Schema { of } => {
            let mut fields = executor.db.get_schema_fields(None, &of);
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            let rows = fields
                .into_iter()
                .map(|(fname, ftype)| {
                    let mut data = std::collections::HashMap::new();
                    data.insert("field".to_string(), super::Value::String(fname));
                    data.insert("type".to_string(), super::Value::String(ftype));
                    super::Row {
                        id: uuid::Uuid::new_v4(),
                        data,
                    }
                })
                .collect();
            Ok(QueryResult {
                rows,
                columns: vec!["field".to_string(), "type".to_string()],
                stats: ExecutionStats::default(),
            })
        }
        ShowTarget::Sequences => execute_show_sequences(executor),
        ShowTarget::Views => {
            let guard_result: Vec<(uuid::Uuid, serde_json::Value)> = executor.db.query_docs_ns(
                Some("__system__"),
                Some("__views__"),
                &std::collections::HashMap::new(),
                1000,
                0,
            );
            let mut rows: Vec<super::Row> = guard_result
                .into_iter()
                .filter_map(|(id, json)| {
                    json.get("__view_name__")
                        .and_then(|v| v.as_str())
                        .map(|name| {
                            let mut data = std::collections::HashMap::new();
                            data.insert("name".to_string(), super::Value::String(name.to_string()));
                            super::Row { id, data }
                        })
                })
                .collect();
            rows.sort_by(|a, b| {
                let an = match a.data.get("name") {
                    Some(super::Value::String(s)) => s.as_str(),
                    _ => "",
                };
                let bn = match b.data.get("name") {
                    Some(super::Value::String(s)) => s.as_str(),
                    _ => "",
                };
                an.cmp(bn)
            });
            Ok(QueryResult {
                rows,
                columns: vec!["name".to_string()],
                stats: ExecutionStats::default(),
            })
        }
    }
}

fn build_schema_def(
    props: Vec<PropertyDef>,
    constraints: Vec<Constraint>,
    indexes: Vec<IndexDef>,
) -> SchemaDef {
    let mut fields = HashMap::new();
    for p in props {
        fields.insert(
            p.name,
            SchemaField {
                required: p.required,
                unique: p.unique,
                r#type: Some(data_type_to_string(p.data_type)),
                default: p.default.map(source::literal_to_json),
            },
        );
    }
    let mut constraint_defs = Vec::new();
    for c in constraints {
        if let Some(def) = constraint_to_def(c) {
            constraint_defs.push(def);
        }
    }
    SchemaDef {
        fields,
        constraints: constraint_defs,
        indexes,
    }
}

fn data_type_to_string(data_type: DataType) -> String {
    match data_type {
        DataType::String => "string".to_string(),
        DataType::Integer => "integer".to_string(),
        DataType::Float => "float".to_string(),
        DataType::Boolean => "boolean".to_string(),
        DataType::Date => "date".to_string(),
        DataType::Timestamp => "timestamp".to_string(),
        DataType::Uuid => "uuid".to_string(),
        DataType::Json => "json".to_string(),
        DataType::Vector(dim) => format!("vector({})", dim),
        DataType::Bytes => "bytes".to_string(),
        DataType::Serial => "integer".to_string(),
        DataType::BigSerial => "bigint".to_string(),
    }
}

pub(super) fn execute_create_sequence(
    executor: &Executor,
    name: String,
    if_not_exists: bool,
    start: i64,
    increment: i64,
    min_value: Option<i64>,
    max_value: Option<i64>,
    cycle: bool,
) -> Result<QueryResult> {
    executor.db.create_sequence(
        &name,
        start,
        increment,
        min_value,
        max_value,
        cycle,
        if_not_exists,
    )?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_drop_sequence(
    executor: &Executor,
    name: String,
    if_exists: bool,
) -> Result<QueryResult> {
    executor.db.drop_sequence(&name, if_exists)?;
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_show_sequences(executor: &Executor) -> Result<QueryResult> {
    let mut names = executor.db.list_sequences();
    names.sort();
    let rows = names
        .into_iter()
        .map(|name| {
            let mut data = std::collections::HashMap::new();
            data.insert("name".to_string(), super::Value::String(name));
            super::Row {
                id: uuid::Uuid::new_v4(),
                data,
            }
        })
        .collect();
    Ok(QueryResult {
        rows,
        columns: vec!["name".to_string()],
        stats: ExecutionStats::default(),
    })
}

fn constraint_to_def(c: Constraint) -> Option<ConstraintDef> {
    match c {
        Constraint::Unique { name, fields } => Some(ConstraintDef::Unique {
            name: name.unwrap_or_else(|| format!("unique_{}", fields.join("_"))),
            fields,
        }),
        Constraint::PrimaryKey { name, fields } => Some(ConstraintDef::PrimaryKey {
            name: name.unwrap_or_else(|| format!("pk_{}", fields.join("_"))),
            fields,
        }),
        Constraint::Check { name, condition } => Some(ConstraintDef::Check {
            name: name.unwrap_or_else(|| "check".to_string()),
            condition,
        }),
        Constraint::ForeignKey {
            name,
            fields,
            ref_table,
            ref_fields,
            on_delete,
            on_update,
        } => Some(ConstraintDef::ForeignKey(ForeignKeyDef {
            name: name.unwrap_or_else(|| format!("fk_{}_{}", ref_table, fields.join("_"))),
            columns: fields,
            referenced_table: ref_table,
            referenced_columns: ref_fields,
            on_delete: map_action(on_delete),
            on_update: map_action(on_update),
            enabled: true,
        })),
    }
}

fn map_action(action: AstReferentialAction) -> ReferentialAction {
    match action {
        AstReferentialAction::NoAction => ReferentialAction::NoAction,
        AstReferentialAction::Restrict => ReferentialAction::Restrict,
        AstReferentialAction::Cascade => ReferentialAction::Cascade,
        AstReferentialAction::SetNull => ReferentialAction::SetNull,
        AstReferentialAction::SetDefault => ReferentialAction::SetDefault,
    }
}
