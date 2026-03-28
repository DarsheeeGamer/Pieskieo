# MongoDB Feature: $facet Aggregation Stage

**Status**: 🔴 Not Started
**Priority**: CRITICAL
**Dependencies**: CTEs (PostgreSQL compatibility), Execution Engine Parallelism
**Estimated Effort**: 2-3 weeks

---

## Overview

The `$facet` stage allows the execution of multiple sub-pipelines within a single stage on the same set of input documents. Each sub-pipeline outputs its results as an array of documents within the single output document.

## Supported Syntax

```json
{
  $facet: {
    <outputField1>: [ <stage1>, <stage2>, ... ],
    <outputField2>: [ <stage1>, <stage2>, ... ],
    ...
  }
}
```

---

## Implementation Plan

### Phase 1: AST Representation

The `$facet` stage simply holds a map of string names to a list of other aggregation stages.

```rust
// crates/pieskieo-core/src/pql/ast.rs

#[derive(Clone, Debug, PartialEq)]
pub enum AggregationStage {
    Match(Expr),
    // ...
    Facet(HashMap<String, Vec<AggregationStage>>),
}
```

### Phase 2: Execution Engine Integration (The "Fork" Model)

A `$facet` stage is essentially a "fork" in the execution plan. The input documents must be fed into multiple, independent pipelines.

**Challenge**: We cannot simply clone the entire input dataset in memory if it's large.

**Solution**: The execution engine should support reading from a shared, buffered stream, or materialize the input into a temporary structure (like a CTE result) that can be scanned multiple times concurrently.

```rust
// crates/pieskieo-core/src/engine/executor.rs

impl UnifiedExecutor {
    pub async fn execute_facet(
        &self,
        input_stream: BoxStream<'static, Result<Document>>,
        facets: HashMap<String, Vec<AggregationStage>>,
    ) -> Result<Document> {

        // 1. Materialize the input stream.
        // If the stream is small, keep in memory.
        // If large, spill to a temporary table/file (similar to CTE materialization).
        let materialized_input = self.materialize_stream(input_stream).await?;

        // 2. Spawn concurrent tasks for each sub-pipeline.
        let mut handles = HashMap::new();
        for (name, pipeline) in facets {
            let engine = self.clone();
            // Create a new stream reader for the materialized data
            let stream = materialized_input.create_reader();

            let handle = tokio::spawn(async move {
                // Execute the pipeline and collect all results into a JSON array
                let results = engine.execute_pipeline_stream(stream, pipeline).await?;
                let mut array = Vec::new();
                while let Some(doc) = results.next().await {
                    array.push(Value::Object(doc?));
                }
                Ok::<Value, PieskieoError>(Value::Array(array))
            });
            handles.insert(name, handle);
        }

        // 3. Collect results and construct the single output document.
        let mut output_doc = Map::new();
        for (name, handle) in handles {
            let sub_result = handle.await.map_err(|e| PieskieoError::Internal(e.to_string()))??;
            output_doc.insert(name, sub_result);
        }

        Ok(output_doc)
    }
}
```

### Phase 3: SQL Translation (Using CTEs)

If we want to translate a full MongoDB pipeline ending in `$facet` to SQL, it maps perfectly to Common Table Expressions (CTEs).

**MongoDB Pipeline:**
```javascript
db.products.aggregate([
  { $match: { category: "Electronics" } },
  {
    $facet: {
      "price_ranges": [
        { $bucket: { groupBy: "$price", boundaries: [0, 100, 500, 1000] } }
      ],
      "top_brands": [
        { $group: { _id: "$brand", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 3 }
      ]
    }
  }
]);
```

**Translates to SQL:**
```sql
WITH input_docs AS (
  SELECT * FROM products WHERE category = 'Electronics'
),
price_ranges AS (
  -- $bucket logic (case statements) on input_docs
  SELECT bucket_id, count(*) FROM input_docs GROUP BY bucket_id
),
top_brands AS (
  SELECT brand, count(*) as count FROM input_docs GROUP BY brand ORDER BY count DESC LIMIT 3
)
SELECT
  (SELECT jsonb_agg(row_to_json(pr.*)) FROM price_ranges pr) AS price_ranges,
  (SELECT jsonb_agg(row_to_json(tb.*)) FROM top_brands tb) AS top_brands;
```

This translation leverages our existing CTE and subquery optimizer logic, ensuring high performance without duplicating the input data physically.

---

## Test Cases

### Test 1: Empty Input Stream
```javascript
// Setup: Empty collection
db.empty.aggregate([
  {
    $facet: {
      "a": [{ $match: { x: 1 } }],
      "b": [{ $group: { _id: null, count: { $sum: 1 } } }] // Group still runs on empty input
    }
  }
]);

// Expected output
// [{ "a": [], "b": [{ "_id": null, "count": 0 }] }]
```

### Test 2: Multiple Complex Pipelines
```javascript
// Setup
db.inventory.insert([
  { _id: 1, type: "food", item: "apple", qty: 100, price: 2.5 },
  { _id: 2, type: "food", item: "pie", qty: 50, price: 10 },
  { _id: 3, type: "electronics", item: "phone", qty: 10, price: 500 },
  { _id: 4, type: "electronics", item: "charger", qty: 200, price: 20 }
]);

// Query
db.inventory.aggregate([
  {
    $facet: {
      "categorizedByPrice": [
        { $match: { price: { $exists: 1 } } },
        {
          $bucket: {
            groupBy: "$price",
            boundaries: [  0, 10, 50, 1000 ],
            default: "Other",
            output: { count: { $sum: 1 }, items: { $push: "$item" } }
          }
        }
      ],
      "categorizedByType": [
        { $group: { _id: "$type", count: { $sum: 1 } } },
        { $sort: { count: -1 } }
      ]
    }
  }
]);

// Expected output includes an array for "categorizedByPrice" and an array for "categorizedByType".
```

---

## Performance Targets

- The input documents up to the `$facet` stage must be evaluated **exactly once**.
- The sub-pipelines within the `$facet` stage should execute **concurrently** using `tokio::spawn`.
- Memory usage must be bounded. Large `$facet` inputs must spill to disk (temp table).

## Metrics to Track

- `pieskieo_facet_stages_executed`
- `pieskieo_facet_materialization_bytes` (monitor if it exceeds thresholds)
- `pieskieo_facet_concurrent_pipelines`

**Created**: 2026-02-08
**Author**: Implementation Team
