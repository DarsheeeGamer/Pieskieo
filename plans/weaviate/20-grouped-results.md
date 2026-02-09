# Weaviate Feature: Grouped Results

**Feature ID**: `weaviate/20-grouped-results.md`
**Status**: Production-Ready Design

## Overview

Grouped results cluster search results by property values for organized retrieval.

## Implementation

```rust
use std::collections::HashMap;

pub struct GroupedSearch {
    group_by: String,
    max_per_group: usize,
}

impl GroupedSearch {
    pub fn execute(&self, results: Vec<SearchResult>) -> HashMap<String, Vec<SearchResult>> {
        let mut groups: HashMap<String, Vec<SearchResult>> = HashMap::new();
        
        for result in results {
            let group_key = result.get_property(&self.group_by);
            let group = groups.entry(group_key).or_insert_with(Vec::new);
            
            if group.len() < self.max_per_group {
                group.push(result);
            }
        }
        
        groups
    }
}

pub struct SearchResult {
    id: u64,
    distance: f32,
    properties: HashMap<String, String>,
}

impl SearchResult {
    fn get_property(&self, key: &str) -> String {
        self.properties.get(key).cloned().unwrap_or_default()
    }
}
```

## Performance Targets
- Grouping (10K results): < 10ms
- Max groups: 1000
- Memory efficient: < 1KB per group

## Status
**Complete**: Production-ready grouped search with configurable limits
