use crate::error::Result;
use crate::pql::ast::{Condition, Expression, SelectField, SortOrder};
use std::collections::HashMap;
use uuid::Uuid;

use super::{expressions, source, Executor, Row, Value};

pub(crate) fn execute_group_by(
    executor: &Executor,
    input: Vec<Row>,
    fields: Vec<Expression>,
    mode: crate::pql::ast::GroupByMode,
) -> Result<Vec<Row>> {
    use crate::pql::ast::GroupByMode;
    match mode {
        GroupByMode::Regular => execute_group_by_fields(executor, input, &fields),
        GroupByMode::Rollup => {
            // Generate groups for each prefix: (f0,f1,...,fn), (f0,...,fn-1), ..., ()
            let mut all_rows = Vec::new();
            for len in (0..=fields.len()).rev() {
                let prefix = fields[..len].to_vec();
                let grouped = execute_group_by_fields(executor, input.clone(), &prefix)?;
                for mut row in grouped {
                    for i in len..fields.len() {
                        if let crate::pql::ast::Expression::FieldAccess(path) = &fields[i] {
                            if path.len() == 1 {
                                row.data.entry(path[0].clone()).or_insert(Value::Null);
                            }
                        }
                    }
                    all_rows.push(row);
                }
            }
            Ok(all_rows)
        }
        GroupByMode::Cube => {
            // Generate groups for every subset of fields
            let n = fields.len();
            let mut all_rows = Vec::new();
            for mask in 0u64..(1u64 << n) {
                let subset: Vec<Expression> = (0..n)
                    .filter(|i| (mask >> i) & 1 == 1)
                    .map(|i| fields[i].clone())
                    .collect();
                let grouped = execute_group_by_fields(executor, input.clone(), &subset)?;
                for mut row in grouped {
                    for i in 0..n {
                        if (mask >> i) & 1 == 0 {
                            if let crate::pql::ast::Expression::FieldAccess(path) = &fields[i] {
                                if path.len() == 1 {
                                    row.data.entry(path[0].clone()).or_insert(Value::Null);
                                }
                            }
                        }
                    }
                    all_rows.push(row);
                }
            }
            Ok(all_rows)
        }
    }
}

fn execute_group_by_fields(
    executor: &Executor,
    input: Vec<Row>,
    fields: &[Expression],
) -> Result<Vec<Row>> {
    let mut groups: HashMap<Vec<Value>, Vec<Row>> = HashMap::new();

    for row in input {
        let key: Vec<Value> = fields
            .iter()
            .map(|expr| expressions::evaluate_expression(executor, expr, &row))
            .collect::<Result<Vec<_>>>()?;
        groups.entry(key).or_default().push(row);
    }

    let mut out = Vec::new();
    for (key, rows) in groups {
        let mut data = HashMap::new();
        for (idx, expr) in fields.iter().enumerate() {
            let field_name = source::expression_to_field_name(expr);
            if let Some(val) = key.get(idx) {
                data.insert(field_name, val.clone());
            }
        }
        let group_rows = rows.iter().map(|r| Value::Object(r.data.clone())).collect();
        data.insert(
            Executor::GROUP_ROWS_KEY.to_string(),
            Value::Array(group_rows),
        );
        out.push(Row {
            id: Uuid::nil(),
            data,
        });
    }

    Ok(out)
}

pub(crate) fn execute_compute(
    executor: &Executor,
    input: Vec<Row>,
    assignments: Vec<(String, Expression)>,
) -> Result<Vec<Row>> {
    // Separate window function assignments from regular ones
    let (window_assigns, regular_assigns): (Vec<_>, Vec<_>) = assignments
        .into_iter()
        .partition(|(_, expr)| matches!(expr, Expression::WindowFunction { .. }));

    // First apply window functions (batch over all rows)
    let mut rows = input;
    for (name, expr) in window_assigns {
        if let Expression::WindowFunction {
            func,
            partition_by,
            order_by,
            ..
        } = expr
        {
            // Extract function name and args from the inner func expression
            let (func_name, func_args) = match *func {
                Expression::FunctionCall { name: fn_name, args: fn_args } => (fn_name, fn_args),
                Expression::FieldAccess(parts) => {
                    (parts.last().cloned().unwrap_or_default(), vec![])
                }
                other => (format!("{:?}", other), vec![]),
            };
            rows = apply_window_function_compute(
                &rows,
                &name,
                &func_name,
                &func_args,
                &partition_by,
                &order_by,
                executor,
            )?;
        }
    }

    // Then apply regular assignments row-by-row
    // Note: GROUP_ROWS_KEY is intentionally preserved so subsequent COMPUTE
    // operations in the same pipeline can still access group rows for aggregates.
    // It is cleaned up in execute_query after all operations complete.
    rows.into_iter()
        .map(|mut row| {
            for (field_name, expr) in &regular_assigns {
                let value = expressions::evaluate_expression(executor, expr, &row)?;
                row.data.insert(field_name.clone(), value);
            }
            Ok(row)
        })
        .collect()
}

pub(crate) fn execute_distinct(input: Vec<Row>) -> Vec<Row> {
    let mut seen = std::collections::HashSet::new();
    input
        .into_iter()
        .filter(|row| {
            let key = row_to_dedup_key(row);
            seen.insert(key)
        })
        .collect()
}

fn row_to_dedup_key(row: &Row) -> String {
    let mut keys: Vec<&String> = row.data.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| format!("{}={}", k, value_to_string(&row.data[*k])))
        .collect::<Vec<_>>()
        .join(";")
}

fn value_to_string(val: &Value) -> String {
    match val {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => format!("{:.15}", f),
        Value::String(s) => format!("\"{}\"", s),
        Value::Uuid(u) => u.to_string(),
        Value::Vector(v) => format!("{:?}", v),
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(value_to_string).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Object(obj) => {
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", k, value_to_string(&obj[*k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

/// Computes a window function over all rows and writes the result into each row.
fn apply_window_function_compute(
    rows: &[Row],
    field_name: &str,
    func: &str,
    args: &[Expression],
    partition_by: &[Expression],
    order_by: &[(Expression, SortOrder)],
    executor: &Executor,
) -> Result<Vec<Row>> {
    // Build a mapping from original index to partition key
    let mut partition_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, row) in rows.iter().enumerate() {
        let key = compute_partition_key(executor, partition_by, row);
        partition_map.entry(key).or_default().push(idx);
    }

    // For each partition, sort indices by order_by and compute window function
    let mut results: Vec<Value> = vec![Value::Null; rows.len()];

    for (_part_key, mut indices) in partition_map {
        // Sort indices by order_by
        if !order_by.is_empty() {
            indices.sort_by(|&a, &b| {
                for (expr, ord) in order_by {
                    let va = expressions::evaluate_expression(executor, expr, &rows[a])
                        .unwrap_or(Value::Null);
                    let vb = expressions::evaluate_expression(executor, expr, &rows[b])
                        .unwrap_or(Value::Null);
                    let cmp = expressions::compare_values(&va, &vb);
                    let ordered = match ord {
                        SortOrder::Asc => cmp,
                        SortOrder::Desc => cmp.reverse(),
                    };
                    if ordered != std::cmp::Ordering::Equal {
                        return ordered;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        let n = indices.len();
        let func_upper = func.to_uppercase();

        match func_upper.as_str() {
            "ROW_NUMBER" => {
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    results[orig_idx] = Value::Integer((pos + 1) as i64);
                }
            }

            "RANK" => {
                // Rows with equal order-by values get the same rank; next rank skips
                let mut rank = 1usize;
                let mut i = 0usize;
                while i < n {
                    let current_key = order_key(executor, order_by, &rows[indices[i]]);
                    let mut j = i;
                    while j < n && order_key(executor, order_by, &rows[indices[j]]) == current_key {
                        results[indices[j]] = Value::Integer(rank as i64);
                        j += 1;
                    }
                    rank += j - i;
                    i = j;
                }
            }

            "DENSE_RANK" => {
                let mut rank = 1usize;
                let mut i = 0usize;
                while i < n {
                    let current_key = order_key(executor, order_by, &rows[indices[i]]);
                    let mut j = i;
                    while j < n && order_key(executor, order_by, &rows[indices[j]]) == current_key {
                        results[indices[j]] = Value::Integer(rank as i64);
                        j += 1;
                    }
                    rank += 1;
                    i = j;
                }
            }

            "NTILE" => {
                let buckets = if !args.is_empty() {
                    match expressions::evaluate_expression(executor, &args[0], &rows[indices[0]]) {
                        Ok(Value::Integer(i)) => i.max(1) as usize,
                        _ => 1,
                    }
                } else {
                    1
                };
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    let bucket = (pos * buckets / n) + 1;
                    results[orig_idx] = Value::Integer(bucket as i64);
                }
            }

            "LAG" => {
                let offset = if args.len() > 1 {
                    match expressions::evaluate_expression(executor, &args[1], &rows[indices[0]]) {
                        Ok(Value::Integer(i)) => i.max(0) as usize,
                        _ => 1,
                    }
                } else {
                    1
                };
                let default = if args.len() > 2 {
                    expressions::evaluate_expression(executor, &args[2], &rows[indices[0]])
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                let expr = args.first();
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    let val = if pos >= offset {
                        if let Some(e) = expr {
                            expressions::evaluate_expression(
                                executor,
                                e,
                                &rows[indices[pos - offset]],
                            )
                            .unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        }
                    } else {
                        default.clone()
                    };
                    results[orig_idx] = val;
                }
            }

            "LEAD" => {
                let offset = if args.len() > 1 {
                    match expressions::evaluate_expression(executor, &args[1], &rows[indices[0]]) {
                        Ok(Value::Integer(i)) => i.max(0) as usize,
                        _ => 1,
                    }
                } else {
                    1
                };
                let default = if args.len() > 2 {
                    expressions::evaluate_expression(executor, &args[2], &rows[indices[0]])
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                let expr = args.first();
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    let val = if pos + offset < n {
                        if let Some(e) = expr {
                            expressions::evaluate_expression(
                                executor,
                                e,
                                &rows[indices[pos + offset]],
                            )
                            .unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        }
                    } else {
                        default.clone()
                    };
                    results[orig_idx] = val;
                }
            }

            "FIRST_VALUE" => {
                let expr = args.first();
                let first_val = if let Some(e) = expr {
                    expressions::evaluate_expression(executor, e, &rows[indices[0]])
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                for &orig_idx in &indices {
                    results[orig_idx] = first_val.clone();
                }
            }

            "LAST_VALUE" => {
                let expr = args.first();
                let last_val = if let Some(e) = expr {
                    expressions::evaluate_expression(executor, e, &rows[*indices.last().unwrap()])
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                for &orig_idx in &indices {
                    results[orig_idx] = last_val.clone();
                }
            }

            "NTH_VALUE" => {
                let nth = if args.len() > 1 {
                    match expressions::evaluate_expression(executor, &args[1], &rows[indices[0]]) {
                        Ok(Value::Integer(i)) => (i.max(1) - 1) as usize,
                        _ => 0,
                    }
                } else {
                    0
                };
                let expr = args.first();
                let nth_val = if nth < n {
                    if let Some(e) = expr {
                        expressions::evaluate_expression(executor, e, &rows[indices[nth]])
                            .unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                } else {
                    Value::Null
                };
                for &orig_idx in &indices {
                    results[orig_idx] = nth_val.clone();
                }
            }

            "SUM" => {
                let expr = args.first();
                // Compute cumulative sums within partition in order
                let mut running = 0f64;
                let mut is_int = true;
                let ordered: Vec<usize> = indices.clone();
                for &orig_idx in &ordered {
                    if let Some(e) = expr {
                        match expressions::evaluate_expression(executor, e, &rows[orig_idx])
                            .unwrap_or(Value::Null)
                        {
                            Value::Integer(i) => running += i as f64,
                            Value::Float(f) => {
                                running += f;
                                is_int = false;
                            }
                            _ => {}
                        }
                    }
                }
                // Actually, window SUM is the sum over the whole partition (unbounded frame)
                let total = running;
                for &orig_idx in &indices {
                    results[orig_idx] = if is_int {
                        Value::Integer(total as i64)
                    } else {
                        Value::Float(total)
                    };
                }
            }

            "AVG" => {
                let expr = args.first();
                let mut sum = 0f64;
                let mut cnt = 0usize;
                for &orig_idx in &indices {
                    if let Some(e) = expr {
                        match expressions::evaluate_expression(executor, e, &rows[orig_idx])
                            .unwrap_or(Value::Null)
                        {
                            Value::Integer(i) => {
                                sum += i as f64;
                                cnt += 1;
                            }
                            Value::Float(f) => {
                                sum += f;
                                cnt += 1;
                            }
                            _ => {}
                        }
                    }
                }
                let avg = if cnt > 0 {
                    Value::Float(sum / cnt as f64)
                } else {
                    Value::Null
                };
                for &orig_idx in &indices {
                    results[orig_idx] = avg.clone();
                }
            }

            "COUNT" => {
                let count_val = Value::Integer(indices.len() as i64);
                for &orig_idx in &indices {
                    results[orig_idx] = count_val.clone();
                }
            }

            "MIN" => {
                let expr = args.first();
                let mut min_val: Option<Value> = None;
                for &orig_idx in &indices {
                    if let Some(e) = expr {
                        let v = expressions::evaluate_expression(executor, e, &rows[orig_idx])
                            .unwrap_or(Value::Null);
                        if !matches!(v, Value::Null) {
                            min_val = Some(match min_val {
                                None => v,
                                Some(cur) => {
                                    if expressions::compare_values(&v, &cur)
                                        == std::cmp::Ordering::Less
                                    {
                                        v
                                    } else {
                                        cur
                                    }
                                }
                            });
                        }
                    }
                }
                let min = min_val.unwrap_or(Value::Null);
                for &orig_idx in &indices {
                    results[orig_idx] = min.clone();
                }
            }

            "MAX" => {
                let expr = args.first();
                let mut max_val: Option<Value> = None;
                for &orig_idx in &indices {
                    if let Some(e) = expr {
                        let v = expressions::evaluate_expression(executor, e, &rows[orig_idx])
                            .unwrap_or(Value::Null);
                        if !matches!(v, Value::Null) {
                            max_val = Some(match max_val {
                                None => v,
                                Some(cur) => {
                                    if expressions::compare_values(&v, &cur)
                                        == std::cmp::Ordering::Greater
                                    {
                                        v
                                    } else {
                                        cur
                                    }
                                }
                            });
                        }
                    }
                }
                let max = max_val.unwrap_or(Value::Null);
                for &orig_idx in &indices {
                    results[orig_idx] = max.clone();
                }
            }

            "CUMSUM" | "CUMULATIVE_SUM" => {
                let expr = args.first();
                let mut running = 0f64;
                let mut is_int = true;
                for &orig_idx in &indices {
                    if let Some(e) = expr {
                        match expressions::evaluate_expression(executor, e, &rows[orig_idx])
                            .unwrap_or(Value::Null)
                        {
                            Value::Integer(i) => running += i as f64,
                            Value::Float(f) => {
                                running += f;
                                is_int = false;
                            }
                            _ => {}
                        }
                    }
                    results[orig_idx] = if is_int {
                        Value::Integer(running as i64)
                    } else {
                        Value::Float(running)
                    };
                }
            }

            "CUME_DIST" => {
                // cumulative distribution = position / total rows in partition
                let n_f = n as f64;
                let mut i = 0usize;
                while i < n {
                    let current_key = order_key(executor, order_by, &rows[indices[i]]);
                    let mut j = i;
                    while j < n && order_key(executor, order_by, &rows[indices[j]]) == current_key {
                        j += 1;
                    }
                    let cume = j as f64 / n_f;
                    for k in i..j {
                        results[indices[k]] = Value::Float(cume);
                    }
                    i = j;
                }
            }

            "PERCENT_RANK" => {
                // percent rank = (rank - 1) / (total - 1)
                if n <= 1 {
                    for &orig_idx in &indices {
                        results[orig_idx] = Value::Float(0.0);
                    }
                } else {
                    let mut rank = 1usize;
                    let mut i = 0usize;
                    while i < n {
                        let current_key = order_key(executor, order_by, &rows[indices[i]]);
                        let mut j = i;
                        while j < n
                            && order_key(executor, order_by, &rows[indices[j]]) == current_key
                        {
                            results[indices[j]] = Value::Float((rank - 1) as f64 / (n - 1) as f64);
                            j += 1;
                        }
                        rank += j - i;
                        i = j;
                    }
                }
            }

            "RUNNING_TOTAL" => {
                let expr = args.first();
                let mut running = 0f64;
                for &orig_idx in &indices {
                    if let Some(e) = expr {
                        match expressions::evaluate_expression(executor, e, &rows[orig_idx])
                            .unwrap_or(Value::Null)
                        {
                            Value::Integer(i) => running += i as f64,
                            Value::Float(f) => running += f,
                            _ => {}
                        }
                    }
                    results[orig_idx] = Value::Float(running);
                }
            }

            "MOVING_AVG" | "ROLLING_AVG" | "ROLLING_AVERAGE" => {
                let expr = args.first();
                let window = if args.len() > 1 {
                    match expressions::evaluate_expression(executor, &args[1], &rows[indices[0]]) {
                        Ok(Value::Integer(i)) => i.max(1) as usize,
                        _ => 3,
                    }
                } else {
                    3
                };
                let vals: Vec<f64> = indices.iter().map(|&orig_idx| {
                    if let Some(e) = expr {
                        match expressions::evaluate_expression(executor, e, &rows[orig_idx]).unwrap_or(Value::Null) {
                            Value::Integer(i) => i as f64,
                            Value::Float(f) => f,
                            _ => 0.0,
                        }
                    } else { 0.0 }
                }).collect();
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    let start = pos.saturating_sub(window - 1);
                    let slice = &vals[start..=pos];
                    let avg = slice.iter().sum::<f64>() / slice.len() as f64;
                    results[orig_idx] = Value::Float(avg);
                }
            }

            "MOVING_SUM" | "ROLLING_SUM" => {
                let expr = args.first();
                let window = if args.len() > 1 {
                    match expressions::evaluate_expression(executor, &args[1], &rows[indices[0]]) {
                        Ok(Value::Integer(i)) => i.max(1) as usize,
                        _ => 3,
                    }
                } else {
                    3
                };
                let vals: Vec<f64> = indices.iter().map(|&orig_idx| {
                    if let Some(e) = expr {
                        match expressions::evaluate_expression(executor, e, &rows[orig_idx]).unwrap_or(Value::Null) {
                            Value::Integer(i) => i as f64,
                            Value::Float(f) => f,
                            _ => 0.0,
                        }
                    } else { 0.0 }
                }).collect();
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    let start = pos.saturating_sub(window - 1);
                    let sum: f64 = vals[start..=pos].iter().sum();
                    results[orig_idx] = Value::Float(sum);
                }
            }

            "RUNNING_COUNT" | "CUMULATIVE_COUNT" => {
                for (pos, &orig_idx) in indices.iter().enumerate() {
                    results[orig_idx] = Value::Integer((pos + 1) as i64);
                }
            }

            _ => {
                // Unknown window function — leave as Null
            }
        }
    }

    // Write results back to rows
    let mut out_rows: Vec<Row> = rows.to_vec();
    for (idx, val) in results.into_iter().enumerate() {
        out_rows[idx].data.insert(field_name.to_string(), val);
    }
    Ok(out_rows)
}

fn compute_partition_key(executor: &Executor, partition_by: &[Expression], row: &Row) -> String {
    if partition_by.is_empty() {
        return "__all__".to_string();
    }
    partition_by
        .iter()
        .map(|expr| {
            let val = expressions::evaluate_expression(executor, expr, row).unwrap_or(Value::Null);
            value_to_string(&val)
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn order_key(executor: &Executor, order_by: &[(Expression, SortOrder)], row: &Row) -> Vec<String> {
    order_by
        .iter()
        .map(|(expr, _)| {
            let val = expressions::evaluate_expression(executor, expr, row).unwrap_or(Value::Null);
            value_to_string(&val)
        })
        .collect()
}

pub(crate) fn execute_order_by(
    executor: &Executor,
    mut input: Vec<Row>,
    fields: Vec<(Expression, SortOrder)>,
) -> Result<Vec<Row>> {
    input.sort_by(|a, b| {
        for (expr, order) in &fields {
            let val_a = expressions::evaluate_expression(executor, expr, a).unwrap_or(Value::Null);
            let val_b = expressions::evaluate_expression(executor, expr, b).unwrap_or(Value::Null);
            let cmp = expressions::compare_values(&val_a, &val_b);
            let ordered_cmp = match order {
                SortOrder::Asc => cmp,
                SortOrder::Desc => cmp.reverse(),
            };
            if ordered_cmp != std::cmp::Ordering::Equal {
                return ordered_cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(input)
}

pub(crate) fn execute_select(
    executor: &Executor,
    input: Vec<Row>,
    fields: Vec<SelectField>,
) -> Result<Vec<Row>> {
    if fields.iter().any(|f| matches!(f, SelectField::All)) {
        return Ok(input);
    }

    input
        .into_iter()
        .map(|row| {
            let mut new_data = HashMap::new();
            for field in &fields {
                match field {
                    SelectField::All => new_data.extend(row.data.clone()),
                    SelectField::Field(expr) => {
                        let value = expressions::evaluate_expression(executor, expr, &row)?;
                        let field_name = source::expression_to_field_name(expr);
                        new_data.insert(field_name, value);
                    }
                    SelectField::Aliased { expr, alias } => {
                        let value = expressions::evaluate_expression(executor, expr, &row)?;
                        new_data.insert(alias.clone(), value);
                    }
                }
            }
            Ok(Row {
                id: row.id,
                data: new_data,
            })
        })
        .collect()
}

/// Expands an array field into one row per element (like SQL UNNEST / MongoDB $unwind).
pub(crate) fn execute_unnest(
    executor: &Executor,
    input: Vec<Row>,
    field: crate::pql::ast::Expression,
    alias: Option<String>,
    index_field: Option<String>,
    preserve: bool,
) -> crate::error::Result<Vec<Row>> {
    let field_name = alias
        .clone()
        .unwrap_or_else(|| source::expression_to_field_name(&field));
    let mut out = Vec::new();
    for row in input {
        let val = expressions::evaluate_expression(executor, &field, &row).unwrap_or(Value::Null);
        match val {
            Value::Array(items) => {
                if items.is_empty() && preserve {
                    // Keep the row with null for the unnested field
                    let mut new_data = row.data.clone();
                    new_data.insert(field_name.clone(), Value::Null);
                    if let Some(ref idx_f) = index_field {
                        new_data.insert(idx_f.clone(), Value::Null);
                    }
                    out.push(Row {
                        id: uuid::Uuid::new_v4(),
                        data: new_data,
                    });
                } else {
                    for (idx, item) in items.into_iter().enumerate() {
                        let mut new_data = row.data.clone();
                        new_data.insert(field_name.clone(), item);
                        if let Some(ref idx_f) = index_field {
                            new_data.insert(idx_f.clone(), Value::Integer(idx as i64));
                        }
                        out.push(Row {
                            id: uuid::Uuid::new_v4(),
                            data: new_data,
                        });
                    }
                }
            }
            Value::Null => {
                if preserve {
                    let mut new_data = row.data.clone();
                    new_data.insert(field_name.clone(), Value::Null);
                    if let Some(ref idx_f) = index_field {
                        new_data.insert(idx_f.clone(), Value::Null);
                    }
                    out.push(Row {
                        id: uuid::Uuid::new_v4(),
                        data: new_data,
                    });
                }
                // Otherwise drop the row (MongoDB $unwind default)
            }
            other => {
                // Treat scalar as single-element array
                let mut new_data = row.data.clone();
                new_data.insert(field_name.clone(), other);
                if let Some(ref idx_f) = index_field {
                    new_data.insert(idx_f.clone(), Value::Integer(0));
                }
                out.push(Row {
                    id: uuid::Uuid::new_v4(),
                    data: new_data,
                });
            }
        }
    }
    Ok(out)
}

/// PIVOT value_field ON pivot_field IN ('v1', 'v2', ...) AGGREGATE func
/// Transforms rows into columns based on pivot_field values.
/// Each unique pivot_field value becomes a new column, aggregating value_field.
pub(crate) fn execute_pivot(
    executor: &Executor,
    input: Vec<Row>,
    value_field: Expression,
    pivot_field: Expression,
    pivot_values: Vec<String>,
    aggregate: String,
) -> Result<Vec<Row>> {
    let pivot_field_name = source::expression_to_field_name(&pivot_field);
    let value_field_name = source::expression_to_field_name(&value_field);

    // Group rows: key = all fields except pivot_field and value_field
    // bucket maps: pivot_value → list of aggregated values
    let mut groups: Vec<(Vec<(String, Value)>, HashMap<String, Vec<Value>>)> = Vec::new();
    let mut group_keys: Vec<Vec<(String, Value)>> = Vec::new();

    for row in &input {
        let pv = match expressions::evaluate_expression(executor, &pivot_field, row)? {
            Value::String(s) => s,
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => format!("{}", f),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".to_string(),
            other => format!("{:?}", other),
        };
        let val = expressions::evaluate_expression(executor, &value_field, row)?;

        let mut key_fields: Vec<(String, Value)> = row
            .data
            .iter()
            .filter(|(k, _)| *k != &pivot_field_name && *k != &value_field_name)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        key_fields.sort_by(|a, b| a.0.cmp(&b.0));

        let pos = group_keys.iter().position(|gk| gk == &key_fields);
        match pos {
            Some(idx) => {
                groups[idx].1.entry(pv).or_default().push(val);
            }
            None => {
                let mut bucket: HashMap<String, Vec<Value>> = HashMap::new();
                bucket.insert(pv, vec![val]);
                groups.push((key_fields.clone(), bucket));
                group_keys.push(key_fields);
            }
        }
    }

    let agg_upper = aggregate.to_uppercase();
    let mut out = Vec::new();
    for (key_fields, bucket) in groups {
        let mut data: HashMap<String, Value> = key_fields.into_iter().collect();
        for pv in &pivot_values {
            let values = bucket.get(pv).cloned().unwrap_or_default();
            data.insert(pv.clone(), apply_pivot_aggregate(&agg_upper, values));
        }
        out.push(Row { id: Uuid::new_v4(), data });
    }
    Ok(out)
}

fn apply_pivot_aggregate(func: &str, values: Vec<Value>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    match func {
        "COUNT" => Value::Integer(values.len() as i64),
        "MIN" => values
            .into_iter()
            .min_by(|a, b| expressions::compare_values(a, b))
            .unwrap_or(Value::Null),
        "MAX" => values
            .into_iter()
            .max_by(|a, b| expressions::compare_values(a, b))
            .unwrap_or(Value::Null),
        "AVG" => {
            let nums: Vec<f64> = values
                .iter()
                .filter_map(|v| match v {
                    Value::Integer(i) => Some(*i as f64),
                    Value::Float(f) => Some(*f),
                    _ => None,
                })
                .collect();
            if nums.is_empty() {
                Value::Null
            } else {
                Value::Float(nums.iter().sum::<f64>() / nums.len() as f64)
            }
        }
        _ => {
            // Default: SUM
            let mut sum_i = 0i64;
            let mut sum_f = 0.0f64;
            let mut is_float = false;
            for v in values {
                match v {
                    Value::Integer(i) => sum_i += i,
                    Value::Float(f) => {
                        sum_f += f;
                        is_float = true;
                    }
                    _ => {}
                }
            }
            if is_float {
                Value::Float(sum_f + sum_i as f64)
            } else {
                Value::Integer(sum_i)
            }
        }
    }
}

/// QUALIFY condition — filter rows based on window function results (applied after SELECT/COMPUTE)
pub(crate) fn execute_qualify(
    executor: &Executor,
    input: Vec<Row>,
    condition: Condition,
) -> Result<Vec<Row>> {
    Ok(input
        .into_iter()
        .filter(|row| expressions::evaluate_condition(executor, &condition, row))
        .collect())
}

/// SAMPLE n — return n random rows from input
pub(crate) fn execute_sample(input: Vec<Row>, count: usize) -> Result<Vec<Row>> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut rows = input;
    rows.shuffle(&mut rng);
    Ok(rows.into_iter().take(count).collect())
}
