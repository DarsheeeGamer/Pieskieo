# PostgreSQL Feature: GIN Indexes (Generalized Inverted Index)

**Status**: 🔴 Not Started  
**Priority**: High  
**Dependencies**: B-tree Indexes, JSON/JSONB Types
**Estimated Effort**: 3-4 weeks

---

## Overview

GIN (Generalized Inverted Index) is designed for handling cases where the items to be indexed are composite values, and the queries need to search for element values that appear within the composite items. They are critical for accelerating searches over arrays, JSONB, and full-text search documents.

## Target Functionality

### 1. Array Indexing
- Fast containment queries (`@>`, `<@`, `&&`, `=`)
- Fast element search (`ANY`)

### 2. JSONB Indexing
- Fast key existence checks (`?`, `?|`, `?&`)
- Fast path/value queries (`@>`)

### 3. Full-Text Search
- Fast text matching (`@@`)

---

## Implementation Plan

### Phase 1: GIN Data Structure

The GIN index structure consists of:
1. **Entry Tree**: A B-tree of all unique elements (entries) found in the indexed column.
2. **Posting Lists/Trees**: Associated with each entry in the Entry Tree. It contains a list (or B-tree) of `ItemPointer`s (row IDs) where the entry occurs.

```rust
// crates/pieskieo-core/src/index/gin.rs

pub struct GinIndex {
    name: String,
    table: String,
    column: String,
    // Entry Tree maps a specific value (e.g., array element or JSON key/value)
    // to a list of row IDs (Posting List)
    entry_tree: BTreeMap<IndexValue, PostingList>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexValue {
    String(String),
    Integer(i64),
    Float(ordered_float::OrderedFloat<f64>),
    Boolean(bool),
    // For JSONB, we might index paths and values together
    JsonPathValue { path: String, value: Box<IndexValue> },
}

pub struct PostingList {
    // Row IDs where the entry is found.
    // Optimized as an ordered list for fast intersection/union.
    row_ids: Vec<RowId>,
    // If posting list gets too large, it can become a B-tree (Posting Tree)
}
```

### Phase 2: Index Extraction Strategy

For each indexed data type, we need an extraction strategy to pull entries from a row.

#### Array Extraction

```rust
impl GinIndex {
    fn extract_array_entries(&self, array: &Value) -> Result<Vec<IndexValue>> {
        if let Value::Array(arr) = array {
            let mut entries = Vec::new();
            for item in arr {
                entries.push(IndexValue::from_value(item)?);
            }
            // Remove duplicates within the same row to save space
            entries.sort();
            entries.dedup();
            Ok(entries)
        } else {
            Err(PieskieoError::InvalidType("Expected Array".into()))
        }
    }
}
```

#### JSONB Extraction

Two main strategies for JSONB (PostgreSQL supports `jsonb_ops` and `jsonb_path_ops`):

1. **Default (`jsonb_ops`)**: Index every key, value, and array element.
2. **Path Ops (`jsonb_path_ops`)**: Index the hash of the full path to a value.

```rust
impl GinIndex {
    fn extract_jsonb_entries(&self, json: &Value) -> Result<Vec<IndexValue>> {
        let mut entries = Vec::new();
        self.extract_jsonb_recursive(json, "", &mut entries)?;
        entries.sort();
        entries.dedup();
        Ok(entries)
    }

    fn extract_jsonb_recursive(&self, json: &Value, current_path: &str, entries: &mut Vec<IndexValue>) -> Result<()> {
        match json {
            Value::Object(obj) => {
                for (k, v) in obj {
                    // Index the key itself
                    entries.push(IndexValue::String(k.clone()));
                    // Recurse
                    let new_path = if current_path.is_empty() { k.clone() } else { format!("{}.{}", current_path, k) };
                    self.extract_jsonb_recursive(v, &new_path, entries)?;
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    self.extract_jsonb_recursive(item, current_path, entries)?;
                }
            }
            // Index the leaf value
            _ => entries.push(IndexValue::from_value(json)?),
        }
        Ok(())
    }
}
```

### Phase 3: Query Execution

The query execution involves finding the posting lists for query elements and performing set operations.

```rust
impl GinIndex {
    pub fn search(&self, query: &GinQuery) -> Result<Vec<RowId>> {
        match query {
            GinQuery::ContainsAll(entries) => {
                // @> operator (Array or JSONB)
                // Need to find rows that have ALL the specified entries
                let mut result = None;
                for entry in entries {
                    if let Some(posting_list) = self.entry_tree.get(entry) {
                        match result {
                            None => result = Some(posting_list.row_ids.clone()),
                            Some(mut current) => {
                                // Intersect current with posting_list
                                current = self.intersect(&current, &posting_list.row_ids);
                                result = Some(current);
                            }
                        }
                    } else {
                        // Entry not found, so no row contains ALL entries
                        return Ok(Vec::new());
                    }
                }
                Ok(result.unwrap_or_default())
            }
            GinQuery::ContainsAny(entries) => {
                // && operator or JSONB ?| operator
                // Need to find rows that have ANY of the specified entries
                let mut result = Vec::new();
                for entry in entries {
                    if let Some(posting_list) = self.entry_tree.get(entry) {
                        // Union current with posting_list
                        result = self.union(&result, &posting_list.row_ids);
                    }
                }
                Ok(result)
            }
            // ... handling other operators
        }
    }

    fn intersect(&self, list1: &[RowId], list2: &[RowId]) -> Vec<RowId> {
        // Fast intersection of two sorted lists
        let mut i = 0;
        let mut j = 0;
        let mut result = Vec::new();
        while i < list1.len() && j < list2.len() {
            if list1[i] == list2[j] {
                result.push(list1[i]);
                i += 1;
                j += 1;
            } else if list1[i] < list2[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        result
    }

    fn union(&self, list1: &[RowId], list2: &[RowId]) -> Vec<RowId> {
        // Fast union of two sorted lists
        let mut i = 0;
        let mut j = 0;
        let mut result = Vec::new();
        while i < list1.len() && j < list2.len() {
            if list1[i] == list2[j] {
                result.push(list1[i]);
                i += 1;
                j += 1;
            } else if list1[i] < list2[j] {
                result.push(list1[i]);
                i += 1;
            } else {
                result.push(list2[j]);
                j += 1;
            }
        }
        while i < list1.len() {
            result.push(list1[i]);
            i += 1;
        }
        while j < list2.len() {
            result.push(list2[j]);
            j += 1;
        }
        result
    }
}
```

### Phase 4: Concurrency and Updates (Fast Update)

Updating a GIN index entry by entry for every row modification is very slow because a single row can generate dozens of index entries.

**Optimization: Fast Update (Pending List)**

PostgreSQL uses a "pending list" to batch GIN updates.

```rust
pub struct GinIndexWithFastUpdate {
    // The main index structure
    main_index: GinIndex,
    // Unsorted list of new entries (RowId, Extracted Entries)
    pending_list: RwLock<Vec<(RowId, Vec<IndexValue>)>>,
    // Configuration
    fast_update_limit_bytes: usize,
}

impl GinIndexWithFastUpdate {
    pub fn insert(&self, row_id: RowId, value: &Value) -> Result<()> {
        let entries = self.main_index.extract_entries(value)?;
        
        let mut pending = self.pending_list.write();
        pending.push((row_id, entries));
        
        // If pending list is too large, flush to main index
        if self.estimate_pending_size(&pending) > self.fast_update_limit_bytes {
            self.flush_pending_list(&mut pending)?;
        }
        Ok(())
    }
    
    fn flush_pending_list(&self, pending: &mut Vec<(RowId, Vec<IndexValue>)>) -> Result<()> {
        // 1. Group by IndexValue
        let mut grouped = BTreeMap::new();
        for (row_id, entries) in pending.drain(..) {
            for entry in entries {
                grouped.entry(entry).or_insert_with(Vec::new).push(row_id);
            }
        }
        
        // 2. Insert into main index in batch
        for (entry, mut row_ids) in grouped {
            row_ids.sort();
            row_ids.dedup();
            self.main_index.batch_insert(entry, row_ids)?;
        }
        Ok(())
    }

    pub fn search(&self, query: &GinQuery) -> Result<Vec<RowId>> {
        // Must search BOTH main index and pending list
        let main_results = self.main_index.search(query)?;
        
        let pending = self.pending_list.read();
        let pending_results = self.search_pending(&pending, query)?;
        
        Ok(self.main_index.union(&main_results, &pending_results))
    }
}
```

---

## Test Cases

### Test 1: Array Contains (`@>`)
```sql
CREATE TABLE docs (id INT, tags TEXT[]);
INSERT INTO docs VALUES (1, '{"tech", "database", "rust"}'), (2, '{"tech", "web"}'), (3, '{"database"}');
CREATE INDEX idx_tags ON docs USING gin (tags);

-- Should use index and return row 1
EXPLAIN SELECT * FROM docs WHERE tags @> '{"tech", "rust"}';
```

### Test 2: JSONB Path Exists (`?`)
```sql
CREATE TABLE logs (id INT, data JSONB);
INSERT INTO logs VALUES (1, '{"user": "alice", "action": "login"}');
INSERT INTO logs VALUES (2, '{"user": "bob", "error": "auth_failed"}');
CREATE INDEX idx_data ON logs USING gin (data);

-- Should use index and return row 2
EXPLAIN SELECT * FROM logs WHERE data ? 'error';
```

### Test 3: JSONB Contains (`@>`)
```sql
-- Should use index and return row 1
EXPLAIN SELECT * FROM logs WHERE data @> '{"user": "alice"}';
```

---

## Performance Targets

- **Insertion**: Fast Update must batch at least 10,000 items before flushing.
- **Search Latency**: < 5ms for highly selective queries on 1M rows.
- **Memory Overhead**: Posting lists must be tightly packed (e.g., delta encoded or using roaring bitmaps).

## Metrics to Track

- `pieskieo_gin_pending_list_size_bytes`
- `pieskieo_gin_flush_duration_ms`
- `pieskieo_gin_index_size_bytes`
- `pieskieo_gin_search_duration_ms`

**Created**: 2026-02-08
**Author**: Implementation Team
