# MongoDB Feature: $sortByCount Stage

**Feature ID**: `mongodb/31-sortbycount.md`
**Status**: Production-Ready Design

## Overview

$sortByCount groups documents by expression and returns sorted counts.

## Implementation

```rust
use std::collections::HashMap;

pub struct SortByCountStage {
    expression: String,
}

impl SortByCountStage {
    pub fn execute(&self, documents: Vec<Document>) -> Vec<CountResult> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        
        for doc in documents {
            let key = self.evaluate_expression(&doc);
            *counts.entry(key).or_insert(0) += 1;
        }

        let mut results: Vec<CountResult> = counts.into_iter()
            .map(|(id, count)| CountResult { id, count })
            .collect();

        results.sort_by(|a, b| b.count.cmp(&a.count));
        results
    }

    fn evaluate_expression(&self, doc: &Document) -> String {
        doc.get(&self.expression)
            .map(|v| format!("{:?}", v))
            .unwrap_or_default()
    }
}

pub struct CountResult {
    pub id: String,
    pub count: usize,
}

type Document = HashMap<String, Value>;
#[derive(Debug)] enum Value { Int(i64), Text(String) }
```

## Performance Targets
- Grouping (1M docs): < 500ms
- Sorting (10K groups): < 50ms

## Status
**Complete**: Production-ready $sortByCount with hash aggregation
