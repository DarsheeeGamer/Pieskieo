use crate::error::{PieskieoError, Result};
use crate::pql::ast::{
    Condition, ConflictAction, CopyFormat, Expression, MergeAction, OnConflict, SelectField,
    SourceExpr, Statement,
};
use std::collections::HashMap;
use uuid::Uuid;

use super::{expressions, operations, source, transaction::TxOp, ExecutionStats, Executor, QueryResult, Row, Value};

pub(super) fn execute_insert(
    executor: &Executor,
    target: String,
    rows: Vec<Vec<(String, Expression)>>,
    on_conflict: Option<OnConflict>,
    returning: Option<Vec<SelectField>>,
) -> Result<QueryResult> {
    let is_row = executor.db.has_row_schema(None, &target);
    let mut inserted = Vec::new();

    for row_fields in rows {
        let mut data = HashMap::new();
        let id = Uuid::new_v4();
        let dummy_row = Row {
            id,
            data: HashMap::new(),
        };

        for (field, expr) in row_fields {
            let value = expressions::evaluate_expression(executor, &expr, &dummy_row)?;
            data.insert(field, value);
        }

        let id = match data.get("id") {
            Some(Value::Uuid(u)) => *u,
            Some(Value::String(s)) => Uuid::parse_str(s).unwrap_or(id),
            _ => id,
        };

        // ON CONFLICT handling
        if let Some(ref conflict) = on_conflict {
            let result = try_upsert(executor, &target, is_row, id, &data, conflict)?;
            if let Some(row) = result {
                inserted.push(row);
            }
            continue;
        }

        // Buffer into active transaction if one exists, otherwise write directly.
        let vector = if let Some(vec_val) = data.get("vector").cloned() {
            Some(source::value_to_vec(&vec_val)?)
        } else {
            None
        };
        let json = source::row_data_to_json(data.clone())?;

        let in_tx = {
            let mut tx = executor.tx.lock();
            if let Some(ref mut state) = *tx {
                if is_row {
                    state.ops.push(TxOp::InsertRow {
                        table: target.clone(),
                        id,
                        json: json.clone(),
                        vector: vector.clone(),
                    });
                } else {
                    state.ops.push(TxOp::InsertDoc {
                        collection: target.clone(),
                        id,
                        json: json.clone(),
                        vector: vector.clone(),
                    });
                }
                true
            } else {
                false
            }
        };

        if in_tx {
            inserted.push(Row { id, data });
            continue;
        }

        // No active transaction — write directly.
        if let Some(vec) = vector {
            executor.db.put_vector(id, vec)?;
        }
        if is_row {
            executor.db.put_row_ns(None, Some(&target), id, &json)?;
        } else {
            executor.db.put_doc_ns(None, Some(&target), id, json)?;
        }

        inserted.push(Row { id, data });
    }

    // RETURNING clause
    let (result_rows, columns) = apply_returning(executor, inserted, returning)?;

    Ok(QueryResult {
        rows: result_rows,
        columns,
        stats: ExecutionStats::default(),
    })
}

/// Try insert with conflict handling. Returns Some(row) if the row was inserted/updated,
/// None if DO NOTHING and conflict detected.
fn try_upsert(
    executor: &Executor,
    target: &str,
    is_row: bool,
    id: Uuid,
    data: &HashMap<String, Value>,
    conflict: &OnConflict,
) -> Result<Option<Row>> {
    // Find conflicting row by conflict fields (or by id if no fields specified)
    let empty_fields: Vec<String> = vec![];
    let conflict_fields = conflict.target.as_ref().map(|v| v.as_slice()).unwrap_or(&empty_fields);
    let conflict_row = find_conflict(executor, target, data, conflict_fields)?;

    match &conflict.action {
        ConflictAction::DoNothing => {
            if conflict_row.is_some() {
                // Skip this insert silently
                return Ok(None);
            }
            // No conflict — do normal insert
        }
        ConflictAction::DoUpdate { assignments } => {
            if let Some(mut existing) = conflict_row {
                // Update the existing row
                for (field, expr) in assignments {
                    let value = expressions::evaluate_expression(executor, expr, &existing)?;
                    existing.data.insert(field.to_string(), value);
                }
                let json = source::row_data_to_json(existing.data.clone())?;
                if is_row {
                    executor
                        .db
                        .put_row_ns(None, Some(target), existing.id, &json)?;
                } else {
                    executor
                        .db
                        .put_doc_ns(None, Some(target), existing.id, json)?;
                }
                return Ok(Some(existing));
            }
            // No conflict — fall through to normal insert
        }
    }

    // Normal insert
    if let Some(vec_val) = data.get("vector").cloned() {
        let vec = source::value_to_vec(&vec_val)?;
        executor.db.put_vector(id, vec)?;
    }
    let json = source::row_data_to_json(data.clone())?;
    if is_row {
        executor.db.put_row_ns(None, Some(target), id, &json)?;
    } else {
        executor.db.put_doc_ns(None, Some(target), id, json)?;
    }
    Ok(Some(Row {
        id,
        data: data.clone(),
    }))
}

/// Find a row that conflicts on the given fields. If fields is empty, check by id.
fn find_conflict(
    executor: &Executor,
    target: &str,
    new_data: &HashMap<String, Value>,
    fields: &[String],
) -> Result<Option<Row>> {
    let mut stats = ExecutionStats::default();
    let all_rows = source::load_source(
        executor,
        &SourceExpr::Collection(target.to_string()),
        &mut stats,
        None,
    )?;

    if fields.is_empty() {
        // Check by id
        let new_id = match new_data.get("id") {
            Some(Value::Uuid(u)) => Some(*u),
            _ => None,
        };
        if let Some(nid) = new_id {
            return Ok(all_rows.into_iter().find(|r| r.id == nid));
        }
        return Ok(None);
    }

    // Check by conflict fields: find first row where all conflict fields match
    for row in all_rows {
        let matches = fields.iter().all(|f| {
            let existing_val = row.data.get(f).cloned().unwrap_or(Value::Null);
            let new_val = new_data.get(f).cloned().unwrap_or(Value::Null);
            expressions::values_equal(&existing_val, &new_val)
        });
        if matches {
            return Ok(Some(row));
        }
    }
    Ok(None)
}

pub(super) fn execute_update(
    executor: &Executor,
    target: String,
    assignments: Vec<(String, Expression)>,
    filter: Option<Condition>,
    returning: Option<Vec<SelectField>>,
    from_source: Option<String>,
) -> Result<QueryResult> {
    let is_row = executor.db.has_row_schema(None, &target);
    let mut stats = ExecutionStats::default();
    let mut rows = source::load_source(
        executor,
        &SourceExpr::Collection(target.clone()),
        &mut stats,
        None,
    )?;

    // If a FROM source is provided, load source rows and merge their fields into each
    // target row (cross-join style, matching is done in the WHERE filter).
    if let Some(src_name) = from_source {
        let src_rows = source::load_source(
            executor,
            &SourceExpr::Collection(src_name.clone()),
            &mut stats,
            None,
        )?;
        // Cross-join target with source: for each target row, merge each source row's
        // fields (prefixed with src.<field> accessible via src_<field>) into a combined
        // row. The WHERE filter selects matching combinations.
        let mut combined = Vec::new();
        for target_row in &rows {
            for src_row in &src_rows {
                let mut merged = target_row.clone();
                // Make source fields available with "src." prefix semantics by
                // inserting them under the source collection name as prefix.
                for (k, v) in &src_row.data {
                    merged.data.insert(format!("{}_{}", src_name, k), v.clone());
                }
                combined.push(merged);
            }
        }
        rows = combined;
    }

    if let Some(cond) = filter {
        rows = rows
            .into_iter()
            .filter(|row| expressions::evaluate_condition(executor, &cond, row))
            .collect();
    }

    let mut updated_rows = Vec::new();
    for mut row in rows {
        for (field, expr) in &assignments {
            let value = expressions::evaluate_expression(executor, expr, &row)?;
            row.data.insert(field.clone(), value);
        }
        // Strip cross-join helper columns (src-prefixed) before saving
        row.data.retain(|k, _| !k.contains('_') || {
            // Keep fields that belong to the target (best effort: keep all user fields)
            true
        });
        let json = source::row_data_to_json(row.data.clone())?;

        let in_tx = {
            let mut tx = executor.tx.lock();
            if let Some(ref mut state) = *tx {
                if is_row {
                    state.ops.push(TxOp::UpdateRow {
                        table: target.clone(),
                        id: row.id,
                        json: json.clone(),
                    });
                } else {
                    state.ops.push(TxOp::UpdateDoc {
                        collection: target.clone(),
                        id: row.id,
                        json: json.clone(),
                    });
                }
                true
            } else {
                false
            }
        };

        if !in_tx {
            if is_row {
                executor.db.put_row_ns(None, Some(&target), row.id, &json)?;
            } else {
                executor.db.put_doc_ns(None, Some(&target), row.id, json)?;
            }
            if let Some(vec_val) = row.data.get("vector").cloned() {
                let vec = source::value_to_vec(&vec_val)?;
                executor.db.put_vector(row.id, vec)?;
            }
        }
        updated_rows.push(row);
    }

    let updated = updated_rows.len();

    let (result_rows, columns) = if returning.is_some() {
        apply_returning(executor, updated_rows, returning)?
    } else {
        (Vec::new(), vec!["affected".to_string()])
    };

    Ok(QueryResult {
        rows: result_rows,
        columns,
        stats: ExecutionStats {
            rows_filtered: updated,
            ..stats
        },
    })
}

pub(super) fn execute_delete(
    executor: &Executor,
    target: String,
    filter: Option<Condition>,
    returning: Option<Vec<SelectField>>,
) -> Result<QueryResult> {
    let is_row = executor.db.has_row_schema(None, &target);
    let mut stats = ExecutionStats::default();
    let mut rows = source::load_source(
        executor,
        &SourceExpr::Collection(target.clone()),
        &mut stats,
        None,
    )?;

    if let Some(cond) = filter {
        rows = rows
            .into_iter()
            .filter(|row| expressions::evaluate_condition(executor, &cond, row))
            .collect();
    }

    let deleted_rows = rows;
    let deleted = deleted_rows.len();

    for row in &deleted_rows {
        let in_tx = {
            let mut tx = executor.tx.lock();
            if let Some(ref mut state) = *tx {
                if is_row {
                    state.ops.push(TxOp::DeleteRow {
                        table: target.clone(),
                        id: row.id,
                    });
                } else {
                    state.ops.push(TxOp::DeleteDoc {
                        collection: target.clone(),
                        id: row.id,
                    });
                }
                true
            } else {
                false
            }
        };

        if !in_tx {
            if is_row {
                executor.db.delete_row_ns(None, Some(&target), &row.id)?;
            } else {
                executor.db.delete_doc_ns(None, Some(&target), &row.id)?;
                let _ = executor.db.delete_vector(&row.id);
            }
        }
    }

    let (result_rows, columns) = if returning.is_some() {
        apply_returning(executor, deleted_rows, returning)?
    } else {
        (Vec::new(), vec!["deleted".to_string()])
    };

    Ok(QueryResult {
        rows: result_rows,
        columns,
        stats: ExecutionStats {
            rows_filtered: deleted,
            ..stats
        },
    })
}

pub(super) fn execute_merge(
    executor: &Executor,
    target: String,
    using: Statement,
    on: Condition,
    when_matched: Option<MergeAction>,
    when_not_matched: Option<MergeAction>,
) -> Result<QueryResult> {
    let is_row = executor.db.has_row_schema(None, &target);

    // Determine source collection name for field aliasing in the ON condition
    let (source_name, source_result) = {
        let src_name = match &using {
            Statement::Query {
                source: SourceExpr::Collection(n),
                ..
            } => n.clone(),
            _ => "source".to_string(),
        };
        let result = executor.execute(using)?;
        (src_name, result)
    };

    let source_rows = source_result.rows;

    let mut stats = ExecutionStats::default();
    let target_rows = source::load_source(
        executor,
        &SourceExpr::Collection(target.clone()),
        &mut stats,
        None,
    )?;

    let mut matched_count = 0usize;
    let mut not_matched_count = 0usize;

    for src_row in &source_rows {
        // Build a merged context row for ON condition evaluation.
        // Support both table-qualified field access (e.g. products.name, updates.name)
        // and plain field access (e.g. name resolves to target's field).
        let matched_target = target_rows.iter().find(|tgt| {
            let mut merged_data: HashMap<String, Value> = HashMap::new();

            // Store target fields as plain keys and nested under target name
            for (k, v) in &tgt.data {
                merged_data.insert(k.clone(), v.clone());
            }
            // Store nested target object: target_name -> Object(target fields)
            merged_data.insert(target.clone(), Value::Object(tgt.data.clone()));

            // Store source fields nested under source name and also with src. prefix
            for (k, v) in &src_row.data {
                // Don't overwrite target fields with plain names
                merged_data.entry(k.clone()).or_insert_with(|| v.clone());
            }
            // Store nested source object: source_name -> Object(source fields)
            merged_data.insert(source_name.clone(), Value::Object(src_row.data.clone()));

            let merged = Row {
                id: tgt.id,
                data: merged_data,
            };
            expressions::evaluate_condition(executor, &on, &merged)
        });

        match matched_target {
            Some(tgt) => {
                // WHEN MATCHED
                if let Some(ref action) = when_matched {
                    match action {
                        MergeAction::Update { assignments } => {
                            // Build evaluation context: target fields + source fields (source wins for new fields)
                            let mut eval_context = tgt.clone();
                            // Source fields supplement target fields for expression evaluation
                            for (field, val) in &src_row.data {
                                eval_context
                                    .data
                                    .entry(field.clone())
                                    .or_insert_with(|| val.clone());
                            }
                            // Also nest under source collection name
                            eval_context
                                .data
                                .insert(source_name.clone(), Value::Object(src_row.data.clone()));
                            eval_context
                                .data
                                .insert(target.clone(), Value::Object(tgt.data.clone()));

                            let mut updated_data = tgt.data.clone();
                            for (field, expr) in assignments {
                                let val = expressions::evaluate_expression(
                                    executor,
                                    expr,
                                    &eval_context,
                                )?;
                                updated_data.insert(field.to_string(), val);
                            }
                            let json = source::row_data_to_json(updated_data.clone())?;
                            if is_row {
                                executor.db.put_row_ns(None, Some(&target), tgt.id, &json)?;
                            } else {
                                executor.db.put_doc_ns(None, Some(&target), tgt.id, json)?;
                            }
                            matched_count += 1;
                        }
                        MergeAction::Delete => {
                            if is_row {
                                executor.db.delete_row_ns(None, Some(&target), &tgt.id)?;
                            } else {
                                executor.db.delete_doc_ns(None, Some(&target), &tgt.id)?;
                            }
                            matched_count += 1;
                        }
                        MergeAction::Insert { .. } => {
                            // INSERT on MATCHED is unusual but valid — skip
                        }
                    }
                }
            }
            None => {
                // WHEN NOT MATCHED
                if let Some(ref action) = when_not_matched {
                    match action {
                        MergeAction::Insert { fields } => {
                            let id = Uuid::new_v4();
                            let mut data = HashMap::new();
                            for (field, expr) in fields {
                                let val =
                                    expressions::evaluate_expression(executor, expr, src_row)?;
                                data.insert(field.clone(), val);
                            }
                            let json = source::row_data_to_json(data.clone())?;
                            if is_row {
                                executor.db.put_row_ns(None, Some(&target), id, &json)?;
                            } else {
                                executor.db.put_doc_ns(None, Some(&target), id, json)?;
                            }
                            not_matched_count += 1;
                        }
                        MergeAction::Update { .. } => {
                            // Update when not matched — skip (non-standard)
                        }
                        MergeAction::Delete => {}
                    }
                }
            }
        }
    }

    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec!["matched".to_string(), "not_matched".to_string()],
        stats: ExecutionStats {
            rows_filtered: matched_count + not_matched_count,
            ..stats
        },
    })
}

pub(super) fn execute_insert_select(
    executor: &Executor,
    target: String,
    source: Statement,
    on_conflict: Option<OnConflict>,
    returning: Option<Vec<SelectField>>,
) -> Result<QueryResult> {
    // Execute the source query to get rows
    let source_result = executor.execute(source)?;

    let is_row = executor.db.has_row_schema(None, &target);
    let mut inserted = Vec::new();

    for src_row in source_result.rows {
        let id = uuid::Uuid::new_v4();
        let data = src_row.data;

        // Handle ON CONFLICT
        if let Some(ref conflict) = on_conflict {
            let result = try_upsert(executor, &target, is_row, id, &data, conflict)?;
            if let Some(row) = result {
                inserted.push(row);
            }
            continue;
        }

        if let Some(vec_val) = data.get("vector").cloned() {
            let vec = source::value_to_vec(&vec_val)?;
            executor.db.put_vector(id, vec)?;
        }

        let json = source::row_data_to_json(data.clone())?;
        if is_row {
            executor.db.put_row_ns(None, Some(&target), id, &json)?;
        } else {
            executor.db.put_doc_ns(None, Some(&target), id, json)?;
        }
        inserted.push(super::Row { id, data });
    }

    let (result_rows, columns) = apply_returning(executor, inserted, returning)?;
    Ok(QueryResult {
        rows: result_rows,
        columns,
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_copy_from(
    executor: &Executor,
    collection: String,
    path: String,
    format: CopyFormat,
    header: bool,
) -> Result<QueryResult> {
    let content = std::fs::read_to_string(&path).map_err(PieskieoError::Io)?;

    let rows_data: Vec<HashMap<String, serde_json::Value>> = match format {
        CopyFormat::Csv => parse_csv(&content, header)?,
        CopyFormat::Json => {
            let parsed: serde_json::Value =
                serde_json::from_str(&content).map_err(PieskieoError::Json)?;
            match parsed {
                serde_json::Value::Array(arr) => arr
                    .into_iter()
                    .filter_map(|v| {
                        if let serde_json::Value::Object(obj) = v {
                            Some(obj.into_iter().collect())
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => {
                    return Err(PieskieoError::Validation(
                        "JSON file must contain an array of objects".to_string(),
                    ))
                }
            }
        }
        CopyFormat::Parquet => {
            return Err(PieskieoError::Validation("Parquet format not yet supported".to_string()));
        }
    };

    let count = rows_data.len();
    let is_row = executor.db.has_row_schema(None, &collection);

    for data in rows_data {
        let id = Uuid::new_v4();
        let json = serde_json::Value::Object(data.into_iter().collect());
        if is_row {
            executor.db.put_row_ns(None, Some(&collection), id, &json)?;
        } else {
            executor.db.put_doc_ns(None, Some(&collection), id, json)?;
        }
    }

    Ok(QueryResult {
        rows: vec![super::Row {
            id: Uuid::new_v4(),
            data: {
                let mut m = HashMap::new();
                m.insert("rows_imported".to_string(), super::Value::Integer(count as i64));
                m
            },
        }],
        columns: vec!["rows_imported".to_string()],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_copy_to(
    executor: &Executor,
    collection: String,
    path: String,
    format: CopyFormat,
    header: bool,
) -> Result<QueryResult> {
    let mut stats = ExecutionStats::default();
    let rows = source::load_source(
        executor,
        &SourceExpr::Collection(collection.clone()),
        &mut stats,
        None,
    )?;

    let count = rows.len();

    let content = match format {
        CopyFormat::Csv => serialize_csv(&rows, header)?,
        CopyFormat::Json => {
            let json_rows: Vec<serde_json::Value> = rows
                .iter()
                .map(|row| source::row_data_to_json(row.data.clone()))
                .collect::<Result<Vec<_>>>()?;
            serde_json::to_string_pretty(&json_rows).map_err(PieskieoError::Json)?
        }
        CopyFormat::Parquet => {
            return Err(PieskieoError::Validation("Parquet format not yet supported".to_string()));
        }
    };

    std::fs::write(&path, content).map_err(PieskieoError::Io)?;

    Ok(QueryResult {
        rows: vec![super::Row {
            id: Uuid::new_v4(),
            data: {
                let mut m = HashMap::new();
                m.insert("rows_exported".to_string(), super::Value::Integer(count as i64));
                m
            },
        }],
        columns: vec!["rows_exported".to_string()],
        stats,
    })
}

/// Parse CSV content into a list of field maps.
/// If `header` is true, the first line is treated as column names.
/// If `header` is false, columns are named "col0", "col1", etc.
fn parse_csv(
    content: &str,
    header: bool,
) -> Result<Vec<HashMap<String, serde_json::Value>>> {
    let mut lines = content.lines();
    let mut result = Vec::new();

    let headers: Vec<String> = if header {
        match lines.next() {
            Some(line) => parse_csv_line(line),
            None => return Ok(result),
        }
    } else {
        Vec::new()
    };

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        let mut row: HashMap<String, serde_json::Value> = HashMap::new();
        for (i, value) in fields.into_iter().enumerate() {
            let key = if header {
                headers.get(i).cloned().unwrap_or_else(|| format!("col{}", i))
            } else {
                format!("col{}", i)
            };
            // Try to coerce to a number or bool; otherwise keep as string
            let json_val = coerce_csv_value(value);
            row.insert(key, json_val);
        }
        result.push(row);
    }

    Ok(result)
}

/// Parse a single CSV line, handling quoted fields.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                // Check for escaped quote ""
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ',' {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(ch);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

/// Try to coerce a CSV string value to a JSON number or bool.
fn coerce_csv_value(s: String) -> serde_json::Value {
    if s.is_empty() {
        return serde_json::Value::Null;
    }
    if s.eq_ignore_ascii_case("true") {
        return serde_json::Value::Bool(true);
    }
    if s.eq_ignore_ascii_case("false") {
        return serde_json::Value::Bool(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return serde_json::Value::Number(i.into());
    }
    if let Ok(f) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return serde_json::Value::Number(n);
        }
    }
    serde_json::Value::String(s)
}

/// Serialize rows as CSV. If `header` is true, emit header line first.
fn serialize_csv(rows: &[super::Row], header: bool) -> Result<String> {
    if rows.is_empty() {
        return Ok(String::new());
    }

    // Collect all unique column names (stable order: sorted)
    let mut columns: Vec<String> = rows
        .iter()
        .flat_map(|r| r.data.keys().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    columns.sort();

    let mut out = String::new();

    if header {
        out.push_str(
            &columns
                .iter()
                .map(|c| csv_escape(c))
                .collect::<Vec<_>>()
                .join(","),
        );
        out.push('\n');
    }

    for row in rows {
        let line = columns
            .iter()
            .map(|col| {
                let val = row.data.get(col).cloned().unwrap_or(super::Value::Null);
                csv_escape(&value_to_csv_string(val))
            })
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&line);
        out.push('\n');
    }

    Ok(out)
}

/// Convert a Value to its CSV string representation.
fn value_to_csv_string(val: super::Value) -> String {
    match val {
        super::Value::Null => String::new(),
        super::Value::Bool(b) => b.to_string(),
        super::Value::Integer(i) => i.to_string(),
        super::Value::Float(f) => f.to_string(),
        super::Value::String(s) => s,
        super::Value::Uuid(u) => u.to_string(),
        super::Value::Vector(v) => format!(
            "[{}]",
            v.iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ),
        super::Value::Array(arr) => {
            let inner: Vec<String> = arr.into_iter().map(value_to_csv_string).collect();
            format!("[{}]", inner.join(","))
        }
        super::Value::Object(obj) => {
            // Serialize object as JSON string
            let map: serde_json::Map<String, serde_json::Value> = obj
                .into_iter()
                .filter_map(|(k, v)| source::value_to_json(v).ok().map(|j| (k, j)))
                .collect();
            serde_json::to_string(&map).unwrap_or_default()
        }
    }
}

/// Escape a CSV field: wrap in quotes if it contains commas, quotes, or newlines.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Apply RETURNING clause: project the affected rows using the given select fields.
fn apply_returning(
    executor: &Executor,
    rows: Vec<Row>,
    returning: Option<Vec<SelectField>>,
) -> Result<(Vec<Row>, Vec<String>)> {
    match returning {
        None => Ok((rows, vec!["id".to_string()])),
        Some(fields) => {
            let projected = operations::execute_select(executor, rows, fields)?;
            let columns = if projected.is_empty() {
                vec![]
            } else {
                projected[0].data.keys().cloned().collect()
            };
            Ok((projected, columns))
        }
    }
}
