# MongoDB Feature: Update Operators

**Status**: 🔴 Not Started
**Priority**: CRITICAL
**Dependencies**: JSONB Operators (PostgreSQL compatibility)
**Estimated Effort**: 3-4 weeks

---

## Overview

MongoDB's update operators (`$set`, `$inc`, `$mul`, `$rename`, `$setOnInsert`, etc.) allow fine-grained, in-place modifications of document fields without requiring the client to fetch, modify, and replace the entire document. This is critical for performance and atomic operations.

## Supported Operators

### 1. Field Update Operators
- `$set`: Sets the value of a field.
- `$unset`: Removes the specified field.
- `$inc`: Increments the value of the field by the specified amount.
- `$mul`: Multiplies the value of the field by the specified amount.
- `$rename`: Renames a field.
- `$min` / `$max`: Updates the value if the specified value is less/greater than the current value.
- `$currentDate`: Sets the field to the current date (Date or Timestamp).

### 2. Array Update Operators
- `$push`: Appends a specified value to an array.
- `$pull`: Removes all array elements that match a specified query.
- `$pop`: Removes the first or last element of an array.
- `$addToSet`: Adds elements to an array only if they do not already exist in the set.

---

## Implementation Plan

### Phase 1: AST Representation

We need to parse the update document into a structured AST.

```rust
// crates/pieskieo-core/src/pql/ast.rs

#[derive(Clone, Debug, PartialEq)]
pub enum UpdateOperator {
    Set(HashMap<String, Expr>),
    Unset(Vec<String>),
    Inc(HashMap<String, Expr>),
    Mul(HashMap<String, Expr>),
    Rename(HashMap<String, String>),
    Min(HashMap<String, Expr>),
    Max(HashMap<String, Expr>),
    CurrentDate(HashMap<String, DateType>),
    Push(HashMap<String, PushModifiers>),
    Pull(HashMap<String, Expr>), // Expr is the query condition
    Pop(HashMap<String, i32>), // 1 or -1
    AddToSet(HashMap<String, Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DateType {
    Timestamp,
    Date,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PushModifiers {
    pub each: Option<Vec<Expr>>,
    pub slice: Option<i32>,
    pub sort: Option<HashMap<String, i32>>,
    pub position: Option<i32>,
}
```

### Phase 2: Translation to SQL (JSONB functions)

Pieskieo executes updates via the core SQL engine. Therefore, we translate MongoDB update operators into PostgreSQL `jsonb` manipulation functions (`jsonb_set`, `jsonb_insert`, `jsonb_set_lax`, or our custom equivalents).

**Example MongoDB Update:**
```javascript
db.users.update(
  { _id: 1 },
  {
    $set: { "profile.name": "Alice" },
    $inc: { "login_count": 1 },
    $push: { "tags": "active" }
  }
)
```

**Translated to SQL:**
```sql
UPDATE users
SET document = jsonb_set(
                 jsonb_set(
                   jsonb_set(
                     document,
                     '{profile,name}', '"Alice"'::jsonb
                   ),
                   '{login_count}', (COALESCE((document->>'login_count')::int, 0) + 1)::text::jsonb
                 ),
                 '{tags}', jsonb_insert(
                             COALESCE(document->'tags', '[]'::jsonb),
                             '{-1}', '"active"'::jsonb, true
                           )
               )
WHERE _id = 1;
```

**Note**: To avoid massive nested `jsonb_set` calls which can be slow and hard to read, we should implement a custom, highly optimized internal function: `pieskieo_jsonb_update(document, update_ast)`.

```rust
// crates/pieskieo-core/src/functions/jsonb_update.rs

pub fn pieskieo_jsonb_update(doc: &mut Value, ops: &[UpdateOperator]) -> Result<()> {
    // This function modifies the JSON value *in place* for performance.
    for op in ops {
        match op {
            UpdateOperator::Set(fields) => {
                for (path, expr) in fields {
                    let new_val = eval_expr(expr)?;
                    doc.set_path(path, new_val)?; // Custom method to walk path and set
                }
            }
            UpdateOperator::Inc(fields) => {
                for (path, expr) in fields {
                    let inc_val = eval_expr_as_number(expr)?;
                    let current = doc.get_path_as_number(path).unwrap_or(0.0);
                    doc.set_path(path, Value::Number(current + inc_val))?;
                }
            }
            UpdateOperator::Push(fields) => {
                // Implement $push logic, creating arrays if they don't exist
            }
            // ... handle other operators
        }
    }
    Ok(())
}
```

### Phase 3: Path Traversal and Creation

A key feature of `$set` and `$push` is that they create intermediate objects if they don't exist.

```javascript
// If doc is {}
// {$set: {"a.b.c": 1}}
// Results in {a: {b: {c: 1}}}
```

```rust
impl Jsonb {
    // Helper to traverse and create missing objects
    pub fn set_path(&mut self, path: &str, new_value: Value) -> Result<()> {
        let segments: Vec<&str> = path.split('.').collect();
        let mut current = &mut self.0;

        for (i, segment) in segments.iter().enumerate() {
            if i == segments.len() - 1 {
                // Leaf node
                if let Value::Object(map) = current {
                    map.insert((*segment).to_string(), new_value);
                } else if let Value::Array(arr) = current {
                     // Array index assignment (e.g. "tags.0")
                     if let Ok(idx) = segment.parse::<usize>() {
                         if idx < arr.len() { arr[idx] = new_value; }
                     }
                }
            } else {
                // Internal node
                match current {
                    Value::Object(map) => {
                        // Create missing object if necessary
                        if !map.contains_key(*segment) {
                            map.insert((*segment).to_string(), Value::Object(Map::new()));
                        }
                        current = map.get_mut(*segment).unwrap();
                    }
                    _ => return Err(PieskieoError::InvalidPath("Cannot traverse through non-object".into())),
                }
            }
        }
        Ok(())
    }
}
```

### Phase 4: Atomic Execution

Because multiple operators might target the same fields, or operators like `$inc` depend on the *current* state of the document, the update must be evaluated atomically within the engine (during the row modification phase).

```rust
// crates/pieskieo-core/src/engine/update.rs

impl UnifiedExecutor {
    pub async fn execute_update(&self, stmt: UpdateStatement) -> Result<u64> {
        let mut rows_updated = 0;

        // Scan matching rows
        let mut stream = self.scan_table_with_filter(&stmt.table, &stmt.where_clause).await?;

        while let Some(row) = stream.next().await {
            let mut row = row?;

            // Apply update operators in-place
            let mut doc = row.get_json_document()?;
            pieskieo_jsonb_update(&mut doc, &stmt.update_operators)?;

            row.set_json_document(doc)?;

            // Write back to WAL/Storage
            self.storage.update_row(&stmt.table, row.id, row).await?;
            rows_updated += 1;
        }

        Ok(rows_updated)
    }
}
```

---

## Test Cases

### Test 1: $set with nested paths
```javascript
db.users.insert({ _id: 1 });
db.users.update({ _id: 1 }, { $set: { "a.b": 2, "c": 3 } });
// Result: { _id: 1, a: { b: 2 }, c: 3 }
```

### Test 2: $inc and $mul
```javascript
db.stats.insert({ _id: 1, views: 10, score: 5.0 });
db.stats.update({ _id: 1 }, { $inc: { views: 5 }, $mul: { score: 2.0 } });
// Result: { _id: 1, views: 15, score: 10.0 }
```

### Test 3: $push with modifiers
```javascript
db.scores.insert({ _id: 1, quizzes: [8, 9] });
db.scores.update(
   { _id: 1 },
   { $push: { quizzes: { $each: [10, 12, 11], $sort: -1, $slice: 3 } } }
);
// Expected: [8, 9, 10, 12, 11] -> sorted [-1] -> [12, 11, 10, 9, 8] -> sliced [3] -> [12, 11, 10]
// Result: { _id: 1, quizzes: [12, 11, 10] }
```

---

## Performance Targets

- **In-place updates**: Avoid full document deserialization/serialization where possible. If the storage format supports delta updates, only write the modified fields to the WAL.
- Latency per document update should be < 50µs for simple `$set` operations.

## Metrics to Track

- `pieskieo_update_operators_applied`
- `pieskieo_update_document_deserializations` (to track inefficiency)

**Created**: 2026-02-08
**Author**: Implementation Team
