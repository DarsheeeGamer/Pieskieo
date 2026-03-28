# MongoDB Feature: findAndModify (PRODUCTION-GRADE)

**Status**: 🔴 Not Started
**Priority**: High
**Dependencies**: Update Operators, Upsert
**Estimated Effort**: 2-3 weeks

---

## Overview

`findAndModify` (and its wrapper methods `findOneAndUpdate`, `findOneAndReplace`, `findOneAndDelete`) atomically modifies and returns a single document. By default, it returns the document as it was *before* the modification, but can be configured to return the *updated* document. This is critical for implementing queues, state machines, and atomic counters where the caller needs to know exactly which document was changed and what its state was.

---

## Operations Supported

### 1. findOneAndUpdate
Updates a single document and returns it.

```javascript
// Return document BEFORE update (default)
db.tasks.findOneAndUpdate(
    { status: "pending" },
    { $set: { status: "processing", start_time: new Date() } },
    { sort: { priority: -1 } }
);

// Return document AFTER update
db.tasks.findOneAndUpdate(
    { status: "pending" },
    { $set: { status: "processing", start_time: new Date() } },
    { returnNewDocument: true, sort: { priority: -1 } }
);
```

### 2. findOneAndReplace
Replaces a single document entirely and returns it.

```javascript
db.configs.findOneAndReplace(
    { _id: "server_config" },
    { max_connections: 500, timeout_ms: 3000 },
    { returnNewDocument: true }
);
```

### 3. findOneAndDelete
Deletes a single document and returns the deleted document.

```javascript
db.queue.findOneAndDelete(
    { status: "processed" },
    { sort: { processed_at: 1 } }
);
```

---

## Key Features & Semantics

1. **Atomicity:** The find and the modify operations must happen as a single, isolated atomic step. No other transaction can modify the document between the find and the modify.
2. **Sorting:** Crucial for picking the *right* document to modify (e.g., pulling the highest priority task from a queue).
3. **Projection:** Return only specific fields of the document (reduces network overhead).
4. **Upsert:** If no document matches, an upsert can insert a new document and return it (or null, depending on `returnNewDocument`).
5. **returnNewDocument (returnDocument: "after"):** Determines whether the pre-modification or post-modification document is returned.

---

## Implementation Plan

### Phase 1: API and Types

**File:** `crates/pieskieo-core/src/mongodb/find_and_modify.rs`

```rust
use serde_json::Value as JsonValue;
use crate::error::Result;

#[derive(Debug, Clone, Default)]
pub struct FindAndModifyOptions {
    pub sort: Option<JsonValue>,
    pub projection: Option<JsonValue>,
    pub return_new_document: bool, // false = return old, true = return new
    pub upsert: bool,
}

pub enum ModifyOperation {
    Update(JsonValue),  // Update operators ($set, $inc, etc.)
    Replace(JsonValue), // Full replacement document
    Delete,
}

pub struct FindAndModifyExecutor {
    db: Arc<PieskieoDb>,
    lock_manager: Arc<LockManager>,
}
```

### Phase 2: Core Execution Logic

The operation must be atomic. We will use a select-for-update lock mechanism to ensure no other operation touches the document while we are modifying it.

```rust
impl FindAndModifyExecutor {
    pub async fn execute(
        &self,
        collection: &str,
        query: &JsonValue,
        operation: ModifyOperation,
        options: &FindAndModifyOptions,
    ) -> Result<Option<JsonValue>> {
        // Start a transaction or acquire a write lock on the collection/partition
        let mut txn = self.db.begin_transaction().await?;

        // 1. Find the document (with FOR UPDATE semantics to lock it)
        // We must apply the sort if provided to ensure we get the correct single document.
        let find_options = FindOptions {
            sort: options.sort.clone(),
            limit: Some(1),
            lock_for_update: true, // CRITICAL: Lock the row
            ..Default::default()
        };

        let mut results = self.db.find_internal(&mut txn, collection, query, &find_options).await?;
        let original_doc = results.pop();

        match original_doc {
            Some(doc) => {
                // Document found. Apply the modification.
                let modified_doc = self.apply_modification(&mut txn, collection, &doc, &operation).await?;

                // Commit transaction
                self.db.commit_transaction(txn).await?;

                // Determine which version to return
                let doc_to_return = if options.return_new_document {
                    // For Delete, returnNewDocument is ignored (or error), always return original
                    if matches!(operation, ModifyOperation::Delete) {
                        doc
                    } else {
                        modified_doc
                    }
                } else {
                    doc
                };

                // Apply projection before returning
                Ok(Some(self.apply_projection(doc_to_return, &options.projection)?))
            }
            None => {
                // Document not found. Check for upsert.
                if options.upsert {
                    match operation {
                        ModifyOperation::Update(update_doc) | ModifyOperation::Replace(update_doc) => {
                            let new_doc = self.perform_upsert_insert(&mut txn, collection, query, &update_doc).await?;
                            self.db.commit_transaction(txn).await?;

                            if options.return_new_document {
                                Ok(Some(self.apply_projection(new_doc, &options.projection)?))
                            } else {
                                // MongoDB returns null when upserting and returnNewDocument is false
                                Ok(None)
                            }
                        }
                        ModifyOperation::Delete => {
                            // Cannot upsert on delete
                            self.db.commit_transaction(txn).await?;
                            Ok(None)
                        }
                    }
                } else {
                    // No document found, no upsert
                    self.db.commit_transaction(txn).await?;
                    Ok(None)
                }
            }
        }
    }

    async fn apply_modification(
        &self,
        txn: &mut Transaction,
        collection: &str,
        original_doc: &JsonValue,
        operation: &ModifyOperation,
    ) -> Result<JsonValue> {
        let doc_id = extract_id(original_doc)?;

        match operation {
            ModifyOperation::Update(update_ops) => {
                let mut updated_doc = original_doc.clone();
                // Apply update operators ($set, $inc, etc.)
                self.db.update_executor.apply_operators(&mut updated_doc, update_ops)?;
                // Save back to DB
                self.db.update_document_internal(txn, collection, &doc_id, &updated_doc).await?;
                Ok(updated_doc)
            }
            ModifyOperation::Replace(replacement_doc) => {
                let mut new_doc = replacement_doc.clone();
                // Ensure _id is preserved
                new_doc["_id"] = original_doc["_id"].clone();
                self.db.replace_document_internal(txn, collection, &doc_id, &new_doc).await?;
                Ok(new_doc)
            }
            ModifyOperation::Delete => {
                self.db.delete_document_internal(txn, collection, &doc_id).await?;
                Ok(original_doc.clone())
            }
        }
    }

    // Additional helper methods omitted for brevity (apply_projection, perform_upsert_insert, etc.)
}
```

### Phase 3: Optimizations

1. **Index Pushdown for Sort & Limit:**
   When a `sort` is provided, `findAndModify` is effectively a "Top-1" query. If an index matches the sort criteria, we can find and lock the target document in `O(log N)` time without scanning.

2. **Fast-Path Single Document Match:**
   If the query includes the `_id` field (or a unique key), we skip the sorting phase entirely and use a direct point-lookup and row-lock, which is extremely fast.

### Phase 4: Distributed Execution

In a sharded environment, `findAndModify` requires care:
1. **Targeted Shard:** If the query includes the shard key, the entire operation (find, lock, modify) is routed to the single shard holding the data. This is efficient and fully atomic.
2. **Scatter-Gather (Avoid if possible):** If the query *doesn't* include the shard key and includes a `sort`, the coordinator must query all shards for their top candidate, determine the global top candidate, and then send the modify command to that specific shard. This requires distributed locking or a two-phase protocol to ensure the document wasn't modified between the find and the update.

---

## Test Cases

### Test 1: findOneAndUpdate - Return Old vs New
```javascript
// Setup
db.inventory.insertOne({ _id: 1, item: "apple", qty: 10 });

// Return Old (Default)
let old_doc = db.inventory.findOneAndUpdate(
    { _id: 1 },
    { $inc: { qty: 5 } }
);
// Assert: old_doc.qty == 10

// Return New
let new_doc = db.inventory.findOneAndUpdate(
    { _id: 1 },
    { $inc: { qty: 5 } },
    { returnNewDocument: true }
);
// Assert: new_doc.qty == 20
```

### Test 2: findOneAndDelete with Sort (Queue Processing)
```javascript
// Setup
db.queue.insertMany([
    { _id: 1, priority: 1, task: "low" },
    { _id: 2, priority: 10, task: "high" },
    { _id: 3, priority: 5, task: "medium" }
]);

// Pop highest priority task
let task = db.queue.findOneAndDelete(
    {}, // match all
    { sort: { priority: -1 } }
);
// Assert: task._id == 2 (highest priority)
// Assert: db.queue.count() == 2
```

### Test 3: Upsert Behavior
```javascript
// Attempt to update non-existent document with upsert
let result = db.configs.findOneAndUpdate(
    { _id: "missing" },
    { $set: { val: 42 } },
    { upsert: true, returnNewDocument: true }
);
// Assert: result is not null, result.val == 42
// Assert: Document was inserted into collection
```

---

## Metrics to Track
- `pieskieo_find_and_modify_total` - Counter by operation (update, replace, delete)
- `pieskieo_find_and_modify_latency_ms` - Histogram for latency
- `pieskieo_find_and_modify_upserts` - Counter for triggered upserts

---

**Created**: 2026-02-08
**Author**: Implementation Team
**Review Status**: Production-Ready
