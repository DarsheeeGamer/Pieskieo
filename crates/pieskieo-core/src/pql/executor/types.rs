use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub rows: Vec<Row>,
    pub columns: Vec<String>,
    pub stats: ExecutionStats,
}

/// Single result row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub id: Uuid,
    pub data: HashMap<String, Value>,
}

/// Unified value type for query results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Uuid(Uuid),
    Vector(Vec<f32>),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Null => 0u8.hash(state),
            Value::Bool(b) => {
                1u8.hash(state);
                b.hash(state);
            }
            Value::Integer(i) => {
                2u8.hash(state);
                i.hash(state);
            }
            Value::Float(f) => {
                3u8.hash(state);
                f.to_bits().hash(state);
            }
            Value::String(s) => {
                4u8.hash(state);
                s.hash(state);
            }
            Value::Uuid(u) => {
                5u8.hash(state);
                u.hash(state);
            }
            Value::Vector(v) => {
                6u8.hash(state);
                for f in v {
                    f.to_bits().hash(state);
                }
            }
            Value::Array(arr) => {
                7u8.hash(state);
                arr.hash(state);
            }
            Value::Object(obj) => {
                8u8.hash(state);
                let mut keys: Vec<_> = obj.keys().collect();
                keys.sort();
                for key in keys {
                    key.hash(state);
                    obj[key].hash(state);
                }
            }
        }
    }
}

/// Execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub rows_scanned: usize,
    pub rows_filtered: usize,
    pub vector_searches: usize,
    pub graph_traversals: usize,
    pub execution_time_ms: u64,
}

/// Graph metrics cache — keyed by algorithm+params string for deduplication within a query
#[derive(Debug, Clone, Default)]
pub struct GraphMetricsCache {
    pub pagerank: HashMap<String, HashMap<Uuid, f64>>,
    pub betweenness: HashMap<String, HashMap<Uuid, f64>>,
    pub closeness: HashMap<String, HashMap<Uuid, f64>>,
    pub components: HashMap<String, HashMap<Uuid, usize>>,
    pub louvain: HashMap<String, HashMap<Uuid, usize>>,
}
