mod ddl;
mod dml;
mod explain;
mod expressions;
mod graph;
mod joins;
mod operations;
mod query;
mod source;
mod transaction;
mod types;
mod vector;

pub use types::{ExecutionStats, GraphMetricsCache, QueryResult, Row, Value};

use crate::engine::PieskieoDb;
use crate::error::Result;
use crate::pql::ast::Statement;
use parking_lot::Mutex;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// PQL executor — dispatches statements to specialized modules.
pub struct Executor {
    pub(crate) db: Arc<PieskieoDb>,
    pub(crate) params: Arc<RwLock<HashMap<String, Value>>>,
    pub(crate) graph_cache: Arc<RwLock<GraphMetricsCache>>,
    pub(crate) ctes: Arc<RwLock<HashMap<String, Vec<Row>>>>,
    pub(crate) tx: Arc<Mutex<Option<transaction::TxState>>>,
}

impl Executor {
    pub(crate) const GROUP_ROWS_KEY: &'static str = "_group_rows";

    pub fn new(db: Arc<PieskieoDb>) -> Self {
        Self {
            db,
            params: Arc::new(RwLock::new(HashMap::new())),
            graph_cache: Arc::new(RwLock::new(GraphMetricsCache::default())),
            ctes: Arc::new(RwLock::new(HashMap::new())),
            tx: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_parameters(&self, params: HashMap<String, Value>) {
        *self.params.write() = params;
    }

    pub fn execute(&self, stmt: Statement) -> Result<QueryResult> {
        let start = std::time::Instant::now();

        let mut result = match stmt {
            Statement::Query {
                with,
                source,
                operations,
            } => query::execute_query(self, with, source, operations)?,
            Statement::Insert {
                target,
                rows,
                on_conflict,
                returning,
            } => dml::execute_insert(self, target, rows, on_conflict, returning)?,
            Statement::Update {
                target,
                assignments,
                filter,
                returning,
                from_source,
            } => dml::execute_update(self, target, assignments, filter, returning, from_source)?,
            Statement::Delete {
                target,
                filter,
                returning,
            } => dml::execute_delete(self, target, filter, returning)?,
            Statement::Create(stmt) => ddl::execute_create(self, stmt)?,
            Statement::AlterTable { name, operations } => {
                ddl::execute_alter_table(self, name, operations)?
            }
            Statement::DropIndex { name, on } => ddl::execute_drop_index(self, name, on)?,
            Statement::DropCollection {
                name,
                is_table,
                cascade,
            } => ddl::execute_drop_collection(self, name, is_table, cascade)?,
            Statement::Explain { analyze, statement } => {
                explain::execute_explain(self, analyze, *statement)?
            }
            Statement::SetOperation {
                op,
                all,
                left,
                right,
            } => query::execute_set_operation(self, op, all, *left, *right)?,
            Statement::CreateView {
                name,
                if_not_exists,
                query,
            } => ddl::execute_create_view(self, name, if_not_exists, *query)?,
            Statement::DropView { name, if_exists } => {
                ddl::execute_drop_view(self, name, if_exists)?
            }
            Statement::Begin => transaction::execute_begin(self)?,
            Statement::Commit => transaction::execute_commit(self)?,
            Statement::Rollback { to } => transaction::execute_rollback(self, to)?,
            Statement::Savepoint { name } => transaction::execute_savepoint(self, name)?,
            Statement::ReleaseSavepoint { name } => {
                transaction::execute_release_savepoint(self, name)?
            }
            Statement::RemoveEdge { src, dst } => graph::execute_remove_edge(self, src, dst)?,
            Statement::Merge {
                target,
                using,
                on,
                when_matched,
                when_not_matched,
            } => dml::execute_merge(self, target, *using, on, when_matched, when_not_matched)?,
            Statement::InsertSelect {
                target,
                source,
                on_conflict,
                returning,
            } => dml::execute_insert_select(self, target, *source, on_conflict, returning)?,
            Statement::AddEdge {
                src,
                dst,
                edge_type,
                weight,
            } => graph::execute_add_edge(self, src, dst, edge_type, weight)?,
            Statement::Truncate { name, is_table } => ddl::execute_truncate(self, name, is_table)?,
            Statement::Show(target) => ddl::execute_show(self, target)?,
            Statement::CreateSequence {
                name,
                if_not_exists,
                start,
                increment,
                min_value,
                max_value,
                cycle,
            } => ddl::execute_create_sequence(
                self,
                name,
                if_not_exists,
                start,
                increment,
                min_value,
                max_value,
                cycle,
            )?,
            Statement::DropSequence { name, if_exists } => {
                ddl::execute_drop_sequence(self, name, if_exists)?
            }
            Statement::CopyFrom {
                collection,
                path,
                format,
                header,
            } => dml::execute_copy_from(self, collection, path, format, header)?,
            Statement::CopyTo {
                collection,
                path,
                format,
                header,
            } => dml::execute_copy_to(self, collection, path, format, header)?,
        };

        result.stats.execution_time_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }
}
