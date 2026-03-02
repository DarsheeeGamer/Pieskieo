use crate::error::{PieskieoError, Result};
use crate::pql::ast::{Expression, VectorMetric};
use crate::vector::VectorMetric as EngineVectorMetric;
use std::collections::HashMap;
use uuid::Uuid;

use super::{expressions, source, Executor, Row, Value};

pub(crate) fn execute_vector_search(
    executor: &Executor,
    input: Vec<Row>,
    query_expr: Expression,
    _field: Option<String>,
    top_k: usize,
    threshold: Option<f64>,
    metric: Option<VectorMetric>,
) -> Result<Vec<Row>> {
    let query_value = expressions::evaluate_expression(
        executor,
        &query_expr,
        &Row {
            id: Uuid::nil(),
            data: HashMap::new(),
        },
    )?;

    let query_vec = match query_value {
        Value::Vector(v) => v,
        Value::Array(arr) => arr
            .into_iter()
            .map(|val| match val {
                Value::Float(f) => Ok(f as f32),
                Value::Integer(i) => Ok(i as f32),
                _ => Err(PieskieoError::Validation(
                    "array elements must be numeric".into(),
                )),
            })
            .collect::<Result<Vec<f32>>>()?,
        _ => {
            return Err(PieskieoError::Validation(
                "query vector must be Vector or Array".into(),
            ))
        }
    };

    let engine_metric = to_engine_metric(metric.unwrap_or(VectorMetric::Cosine));
    let results = executor
        .db
        .search_vector_metric(&query_vec, top_k, engine_metric, None)?;

    let filtered_results: Vec<_> = if let Some(thresh) = threshold {
        results
            .into_iter()
            .filter(|r| r.score >= thresh as f32)
            .collect()
    } else {
        results
    };

    if input.is_empty() {
        return Ok(filtered_results
            .into_iter()
            .map(|hit| {
                let mut data = HashMap::new();
                data.insert("id".to_string(), Value::Uuid(hit.id));
                data.insert("score".to_string(), Value::Float(hit.score as f64));
                if let Some(doc) = executor.db.get_doc(&hit.id) {
                    source::merge_json_into_data(&mut data, doc);
                } else if let Some(row) = executor.db.get_row(&hit.id) {
                    source::merge_json_into_data(&mut data, row);
                }
                Row { id: hit.id, data }
            })
            .collect());
    }

    let hit_ids: HashMap<Uuid, f32> = filtered_results
        .into_iter()
        .map(|hit| (hit.id, hit.score))
        .collect();

    let mut output: Vec<Row> = input
        .into_iter()
        .filter_map(|mut row| {
            if let Some(&score) = hit_ids.get(&row.id) {
                row.data
                    .insert("_vector_score".to_string(), Value::Float(score as f64));
                Some(row)
            } else {
                None
            }
        })
        .collect();

    output.sort_by(|a, b| {
        let sa = match a.data.get("_vector_score") {
            Some(Value::Float(f)) => *f,
            _ => 0.0,
        };
        let sb = match b.data.get("_vector_score") {
            Some(Value::Float(f)) => *f,
            _ => 0.0,
        };
        sb.partial_cmp(&sa).unwrap()
    });

    Ok(output)
}

pub(crate) fn execute_hybrid_search(
    executor: &Executor,
    collection: &str,
    input: Vec<Row>,
    query_expr: Expression,
    _field: Option<String>,
    top_k: usize,
    alpha: f64,
) -> Result<Vec<Row>> {
    let vector_weight = alpha;
    let keyword_weight = 1.0 - alpha;

    let input_ids: std::collections::HashSet<Uuid> = input.iter().map(|row| row.id).collect();

    // Vector search (if query evaluates to a vector)
    let mut vector_scores: HashMap<Uuid, f64> = HashMap::new();
    if vector_weight > 0.0 {
        if let Ok(query_vec) = evaluate_vector_expr(executor, query_expr.clone()) {
            let engine_metric = to_engine_metric(VectorMetric::Cosine);
            if let Ok(vector_hits) =
                executor
                    .db
                    .search_vector_metric(&query_vec, top_k * 5, engine_metric, None)
            {
                for hit in vector_hits {
                    if input_ids.is_empty() || input_ids.contains(&hit.id) {
                        vector_scores.insert(hit.id, hit.score as f64);
                    }
                }
            }
        }
    }

    // Keyword search
    let keywords = evaluate_keywords_expr(executor, query_expr)?;
    let query_terms = source::tokenize_terms(&keywords);
    let query_str = query_terms.join(" ");

    let is_row = executor.db.has_row_schema(None, collection);
    let keyword_scores_vec: Vec<(Uuid, f64)> = if is_row {
        executor.db.bm25_scores_row(None, collection, &query_str)
    } else {
        executor.db.bm25_scores_doc(None, collection, &query_str)
    };
    let keyword_scores: HashMap<Uuid, f64> = keyword_scores_vec
        .into_iter()
        .filter(|(id, _)| input_ids.is_empty() || input_ids.contains(id))
        .collect();

    let vec_max = vector_scores.values().cloned().fold(0.0_f64, f64::max);
    let kw_max = keyword_scores.values().cloned().fold(0.0_f64, f64::max);

    // All candidate IDs: union of vector hits and keyword hits (filtered by input)
    let all_ids: std::collections::HashSet<Uuid> = if input_ids.is_empty() {
        vector_scores
            .keys()
            .cloned()
            .chain(keyword_scores.keys().cloned())
            .collect()
    } else {
        input_ids.clone()
    };

    let mut combined: Vec<(Uuid, f64)> = all_ids
        .iter()
        .filter_map(|id| {
            let v = vector_scores.get(id).cloned().unwrap_or(0.0);
            let k = keyword_scores.get(id).cloned().unwrap_or(0.0);
            let v_norm = if vec_max > 0.0 { v / vec_max } else { 0.0 };
            let k_norm = if kw_max > 0.0 { k / kw_max } else { 0.0 };
            let score = vector_weight * v_norm + keyword_weight * k_norm;
            if score > 0.0 {
                Some((*id, score))
            } else {
                None
            }
        })
        .collect();

    combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    combined.truncate(top_k);

    let mut output = Vec::with_capacity(combined.len());
    for (id, score) in combined {
        let mut data = HashMap::new();
        data.insert("_hybrid_score".to_string(), Value::Float(score));
        if let Some(vs) = vector_scores.get(&id) {
            data.insert("_vector_score".to_string(), Value::Float(*vs));
        }
        if let Some(ks) = keyword_scores.get(&id) {
            data.insert("_keyword_score".to_string(), Value::Float(*ks));
        }
        if let Some(doc) = executor.db.get_doc(&id) {
            source::merge_json_into_data(&mut data, doc);
        } else if let Some(row) = executor.db.get_row(&id) {
            source::merge_json_into_data(&mut data, row);
        }
        output.push(Row { id, data });
    }

    Ok(output)
}

fn evaluate_vector_expr(executor: &Executor, expr: Expression) -> Result<Vec<f32>> {
    let query_value = expressions::evaluate_expression(
        executor,
        &expr,
        &Row {
            id: Uuid::nil(),
            data: HashMap::new(),
        },
    )?;
    match query_value {
        Value::Vector(v) => Ok(v),
        Value::Array(arr) => arr
            .into_iter()
            .map(|val| match val {
                Value::Float(f) => Ok(f as f32),
                Value::Integer(i) => Ok(i as f32),
                _ => Err(PieskieoError::Validation(
                    "array elements must be numeric".into(),
                )),
            })
            .collect::<Result<Vec<f32>>>(),
        _ => Err(PieskieoError::Validation(
            "query vector must be Vector or Array".into(),
        )),
    }
}

fn evaluate_keywords_expr(executor: &Executor, expr: Expression) -> Result<String> {
    let value = expressions::evaluate_expression(
        executor,
        &expr,
        &Row {
            id: Uuid::nil(),
            data: HashMap::new(),
        },
    )?;
    match value {
        Value::String(s) => Ok(s),
        Value::Array(items) => {
            let parts: Vec<String> = items
                .into_iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s),
                    Value::Integer(i) => Some(i.to_string()),
                    Value::Float(f) => Some(f.to_string()),
                    Value::Bool(b) => Some(b.to_string()),
                    Value::Uuid(u) => Some(u.to_string()),
                    _ => None,
                })
                .collect();
            Ok(parts.join(" "))
        }
        Value::Integer(i) => Ok(i.to_string()),
        Value::Float(f) => Ok(f.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Uuid(u) => Ok(u.to_string()),
        _ => Ok(String::new()),
    }
}

pub(crate) fn execute_fulltext_search(
    executor: &Executor,
    collection: &str,
    input: Vec<Row>,
    query: Expression,
    field: Option<String>,
    top_k: usize,
) -> Result<Vec<Row>> {
    let query_val = expressions::evaluate_expression(
        executor,
        &query,
        &Row {
            id: Uuid::nil(),
            data: HashMap::new(),
        },
    )?;
    let query_str = match query_val {
        Value::String(s) => s,
        other => format!("{:?}", other),
    };
    let terms = source::tokenize_terms(&query_str);
    let limit = top_k;

    // Use BM25 index if available (try doc first, then row)
    let is_row = executor.db.has_row_schema(None, collection);
    let query_str = terms.join(" ");
    let scores: std::collections::HashMap<Uuid, f64> = if is_row {
        executor
            .db
            .bm25_scores_row(None, collection, &query_str)
            .into_iter()
            .collect()
    } else {
        executor
            .db
            .bm25_scores_doc(None, collection, &query_str)
            .into_iter()
            .collect()
    };

    // If no BM25 scores (no fulltext index), fall back to linear scan with contains check
    if scores.is_empty() && !input.is_empty() {
        let q_lower = query_str.to_lowercase();
        let mut matched: Vec<Row> = input
            .into_iter()
            .filter(|row| {
                // Check all string fields (or the specific field if given)
                if let Some(ref fname) = field {
                    if let Some(val) = row.data.get(fname) {
                        match val {
                            Value::String(s) => return s.to_lowercase().contains(&q_lower),
                            _ => return false,
                        }
                    }
                    return false;
                }
                row.data.values().any(|v| {
                    if let Value::String(s) = v {
                        s.to_lowercase().contains(&q_lower)
                    } else {
                        false
                    }
                })
            })
            .collect();
        matched.truncate(limit);
        return Ok(matched);
    }

    // Sort by BM25 score descending, keep top_k
    let mut scored: Vec<(Uuid, f64)> = scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    // Look up the actual rows for the top-k IDs
    let mut result = Vec::new();
    for (id, score) in scored {
        let row_opt = if is_row {
            executor.db.get_row_ns(None, Some(collection), &id)
        } else {
            executor.db.get_doc_ns(None, Some(collection), &id)
        };
        if let Some(json) = row_opt {
            let mut row = source::json_to_row(id, json);
            row.data
                .insert("__bm25_score__".to_string(), Value::Float(score));
            // Filter by specific field if requested — already scored, just include
            result.push(row);
        }
    }

    Ok(result)
}

fn to_engine_metric(metric: VectorMetric) -> EngineVectorMetric {
    match metric {
        VectorMetric::L2 => EngineVectorMetric::L2,
        VectorMetric::Cosine => EngineVectorMetric::Cosine,
        VectorMetric::Dot => EngineVectorMetric::Dot,
        VectorMetric::Hamming => EngineVectorMetric::L2, // fallback: use L2 for Hamming
    }
}
