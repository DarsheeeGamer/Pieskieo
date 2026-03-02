//! PQL transaction support: BEGIN / COMMIT / ROLLBACK / SAVEPOINT
//!
//! Implements optimistic buffered transactions:
//! - Reads are always READ COMMITTED (see the live committed state)
//! - Writes are buffered in TxState and applied atomically at COMMIT
//! - ROLLBACK discards the buffer entirely
//! - SAVEPOINT marks a position; ROLLBACK TO discards ops back to that mark

use crate::error::{PieskieoError, Result};
use serde_json::Value as JsonValue;
use uuid::Uuid;

use super::{ExecutionStats, Executor, QueryResult};

/// A single buffered write operation within a transaction.
#[allow(dead_code)]
pub(crate) enum TxOp {
    InsertDoc {
        collection: String,
        id: Uuid,
        json: JsonValue,
        vector: Option<Vec<f32>>,
    },
    InsertRow {
        table: String,
        id: Uuid,
        json: JsonValue,
        vector: Option<Vec<f32>>,
    },
    UpdateDoc {
        collection: String,
        id: Uuid,
        json: JsonValue,
    },
    UpdateRow {
        table: String,
        id: Uuid,
        json: JsonValue,
    },
    DeleteDoc {
        collection: String,
        id: Uuid,
    },
    DeleteRow {
        table: String,
        id: Uuid,
    },
}

/// State held during an active transaction.
pub(crate) struct TxState {
    /// All buffered operations, in order.
    pub(crate) ops: Vec<TxOp>,
    /// Savepoints: (name, ops.len() at savepoint time)
    pub(crate) savepoints: Vec<(String, usize)>,
}

impl TxState {
    pub fn new() -> Self {
        TxState {
            ops: Vec::new(),
            savepoints: Vec::new(),
        }
    }
}

pub(super) fn execute_begin(executor: &Executor) -> Result<QueryResult> {
    let mut tx = executor.tx.lock();
    if tx.is_some() {
        return Err(PieskieoError::Validation(
            "Transaction already in progress. Use COMMIT or ROLLBACK first.".into(),
        ));
    }
    *tx = Some(TxState::new());
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_commit(executor: &Executor) -> Result<QueryResult> {
    let ops = {
        let mut tx = executor.tx.lock();
        let state = tx
            .take()
            .ok_or_else(|| PieskieoError::Validation("No active transaction to COMMIT.".into()))?;
        state.ops
    };

    // Apply all buffered operations sequentially
    for op in ops {
        apply_tx_op(executor, op)?;
    }

    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_rollback(executor: &Executor, to: Option<String>) -> Result<QueryResult> {
    let mut tx = executor.tx.lock();
    if let Some(ref mut state) = *tx {
        if let Some(sp_name) = to {
            // ROLLBACK TO SAVEPOINT — truncate ops to savepoint position
            let pos = state
                .savepoints
                .iter()
                .rev()
                .find(|(name, _)| *name == sp_name)
                .map(|(_, pos)| *pos)
                .ok_or_else(|| {
                    PieskieoError::Validation(format!("Savepoint '{}' not found.", sp_name))
                })?;
            state.ops.truncate(pos);
            // Remove savepoints after this one
            state.savepoints.retain(|(_, p)| *p <= pos);
        } else {
            // Full rollback
            *tx = None;
        }
    } else {
        return Err(PieskieoError::Validation(
            "No active transaction to ROLLBACK.".into(),
        ));
    }

    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_savepoint(executor: &Executor, name: String) -> Result<QueryResult> {
    let mut tx = executor.tx.lock();
    let state = tx
        .as_mut()
        .ok_or_else(|| PieskieoError::Validation("No active transaction for SAVEPOINT.".into()))?;
    let pos = state.ops.len();
    // Replace if same name already exists
    state.savepoints.retain(|(n, _)| n != &name);
    state.savepoints.push((name, pos));
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

pub(super) fn execute_release_savepoint(executor: &Executor, name: String) -> Result<QueryResult> {
    let mut tx = executor.tx.lock();
    let state = tx.as_mut().ok_or_else(|| {
        PieskieoError::Validation("No active transaction for RELEASE SAVEPOINT.".into())
    })?;
    let before = state.savepoints.len();
    state.savepoints.retain(|(n, _)| n != &name);
    if state.savepoints.len() == before {
        return Err(PieskieoError::Validation(format!(
            "Savepoint '{}' not found.",
            name
        )));
    }
    Ok(QueryResult {
        rows: Vec::new(),
        columns: vec![],
        stats: ExecutionStats::default(),
    })
}

fn apply_tx_op(executor: &Executor, op: TxOp) -> Result<()> {
    match op {
        TxOp::InsertDoc {
            collection,
            id,
            json,
            vector,
        } => {
            executor.db.put_doc_ns(None, Some(&collection), id, json)?;
            if let Some(v) = vector {
                executor.db.put_vector(id, v)?;
            }
        }
        TxOp::InsertRow {
            table,
            id,
            json,
            vector,
        } => {
            executor.db.put_row_ns(None, Some(&table), id, &json)?;
            if let Some(v) = vector {
                executor.db.put_vector(id, v)?;
            }
        }
        TxOp::UpdateDoc {
            collection,
            id,
            json,
        } => {
            executor.db.put_doc_ns(None, Some(&collection), id, json)?;
        }
        TxOp::UpdateRow { table, id, json } => {
            executor.db.put_row_ns(None, Some(&table), id, &json)?;
        }
        TxOp::DeleteDoc { collection, id } => {
            executor.db.delete_doc_ns(None, Some(&collection), &id)?;
            let _ = executor.db.delete_vector(&id);
        }
        TxOp::DeleteRow { table, id } => {
            executor.db.delete_row_ns(None, Some(&table), &id)?;
        }
    }
    Ok(())
}
