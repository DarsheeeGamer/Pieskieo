use crate::error::{PieskieoError, Result};
use crate::pql::ast::{Expression, Literal, SourceExpr};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use uuid::Uuid;

use super::{ExecutionStats, Executor, Row, Value};

pub(crate) fn load_source(
    executor: &Executor,
    source: &SourceExpr,
    stats: &mut ExecutionStats,
    index_filter: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<Vec<Row>> {
    match source {
        SourceExpr::Collection(name) | SourceExpr::CollectionAs { name, .. } => {
            // Check if name refers to a view — if so, execute the view's query
            if let Some(view_stmt) = executor.db.get_view(name) {
                let result = executor.execute(view_stmt)?;
                stats.rows_scanned = result.rows.len();
                return Ok(result.rows);
            }
            load_collection(executor, name, stats, index_filter)
        }

        SourceExpr::Cte(name) => {
            let ctes = executor.ctes.read();
            let rows = ctes.get(name.as_str()).cloned().unwrap_or_default();
            stats.rows_scanned = rows.len();
            Ok(rows)
        }

        SourceExpr::Subquery { statement, .. } => {
            let result = executor.execute(*statement.clone())?;
            stats.rows_scanned = result.rows.len();
            Ok(result.rows)
        }

        SourceExpr::Values { rows, alias: _ } => {
            let empty_row = Row {
                id: Uuid::new_v4(),
                data: HashMap::new(),
            };
            let result_rows: Vec<Row> = rows
                .iter()
                .map(|fields| {
                    let data: HashMap<String, Value> = fields
                        .iter()
                        .map(|(k, expr)| {
                            let v =
                                super::expressions::evaluate_expression(executor, expr, &empty_row)
                                    .unwrap_or(Value::Null);
                            (k.clone(), v)
                        })
                        .collect();
                    Row {
                        id: Uuid::new_v4(),
                        data,
                    }
                })
                .collect();
            stats.rows_scanned = result_rows.len();
            Ok(result_rows)
        }
    }
}

fn load_collection(
    executor: &Executor,
    collection: &str,
    stats: &mut ExecutionStats,
    index_filter: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<Vec<Row>> {
    let filter = index_filter.cloned().unwrap_or_default();

    let docs = executor
        .db
        .query_docs_ns(None, Some(collection), &filter, usize::MAX, 0);

    if !docs.is_empty() {
        stats.rows_scanned = docs.len();
        return Ok(docs
            .into_iter()
            .map(|(id, json)| json_to_row(id, json))
            .collect());
    }

    let rows_data = executor
        .db
        .query_rows_ns(None, Some(collection), &filter, usize::MAX, 0);

    stats.rows_scanned = rows_data.len();
    Ok(rows_data
        .into_iter()
        .map(|(id, json)| json_to_row(id, json))
        .collect())
}

pub(crate) fn json_to_row(id: Uuid, json: JsonValue) -> Row {
    let data = json_value_to_hashmap(json);
    Row { id, data }
}

pub(crate) fn json_to_value(json: JsonValue) -> Value {
    match json {
        JsonValue::Null => Value::Null,
        JsonValue::Bool(b) => Value::Bool(b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        JsonValue::String(s) => Value::String(s),
        JsonValue::Array(arr) => Value::Array(arr.into_iter().map(json_to_value).collect()),
        JsonValue::Object(obj) => Value::Object(
            obj.into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect(),
        ),
    }
}

pub(crate) fn merge_json_into_data(data: &mut HashMap<String, Value>, json: JsonValue) {
    if let JsonValue::Object(obj) = json {
        for (k, v) in obj {
            data.insert(k, json_to_value(v));
        }
    }
}

pub(crate) fn merge_rows(left: &Row, right: &Row) -> Row {
    let mut data = left.data.clone();
    for (k, v) in &right.data {
        data.insert(format!("right.{}", k), v.clone());
    }
    Row { id: left.id, data }
}

pub(crate) fn row_with_right_only(right: &Row) -> Row {
    let mut data = HashMap::new();
    for (k, v) in &right.data {
        data.insert(format!("right.{}", k), v.clone());
    }
    Row { id: right.id, data }
}

pub(crate) fn value_to_json(val: Value) -> Result<JsonValue> {
    Ok(match val {
        Value::Null => JsonValue::Null,
        Value::Bool(b) => JsonValue::Bool(b),
        Value::Integer(i) => JsonValue::Number(i.into()),
        Value::Float(f) => {
            JsonValue::Number(serde_json::Number::from_f64(f).unwrap_or_else(|| 0.into()))
        }
        Value::String(s) => JsonValue::String(s),
        Value::Uuid(u) => JsonValue::String(u.to_string()),
        Value::Vector(v) => {
            JsonValue::Array(v.into_iter().map(|f| JsonValue::from(f as f64)).collect())
        }
        Value::Array(arr) => JsonValue::Array(
            arr.into_iter()
                .map(value_to_json)
                .collect::<Result<Vec<_>>>()?,
        ),
        Value::Object(obj) => {
            let mut map = serde_json::Map::new();
            for (k, v) in obj {
                map.insert(k, value_to_json(v)?);
            }
            JsonValue::Object(map)
        }
    })
}

pub(crate) fn row_data_to_json(data: HashMap<String, Value>) -> Result<JsonValue> {
    let mut map = serde_json::Map::new();
    for (k, v) in data {
        map.insert(k, value_to_json(v)?);
    }
    Ok(JsonValue::Object(map))
}

pub(crate) fn expression_to_field_name(expr: &Expression) -> String {
    match expr {
        Expression::FieldAccess(path) => path.join("."),
        Expression::Literal(Literal::String(s)) => s.clone(),
        _ => "field".to_string(),
    }
}

pub(crate) fn value_to_vec(val: &Value) -> Result<Vec<f32>> {
    match val {
        Value::Vector(v) => Ok(v.clone()),
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::Float(f) => Ok(*f as f32),
                Value::Integer(i) => Ok(*i as f32),
                _ => Err(PieskieoError::Validation(
                    "vector array elements must be numeric".into(),
                )),
            })
            .collect(),
        _ => Err(PieskieoError::Validation(
            "vector must be array or vector".into(),
        )),
    }
}

fn json_value_to_hashmap(json: JsonValue) -> HashMap<String, Value> {
    match json {
        JsonValue::Object(map) => map
            .into_iter()
            .map(|(k, v)| (k, json_to_value(v)))
            .collect(),
        _ => HashMap::new(),
    }
}

pub(crate) fn literal_to_json(lit: Literal) -> serde_json::Value {
    match lit {
        Literal::Null => serde_json::Value::Null,
        Literal::Bool(b) => serde_json::Value::Bool(b),
        Literal::Integer(i) => serde_json::Value::Number(i.into()),
        Literal::Float(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Literal::String(s) => serde_json::Value::String(s),
        Literal::Uuid(u) => serde_json::Value::String(u.to_string()),
    }
}

pub(crate) fn tokenize_terms(input: &str) -> Vec<String> {
    input
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}
