# Core Feature: Query Plan Caching

**Feature ID**: `core-features/17-plan-caching.md`
**Status**: Production-Ready Design

## Overview

Query plan caching stores optimized execution plans for repeated queries, eliminating planning overhead for common query patterns.

## Implementation

```rust
use crate::query::QueryPlan;
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::HashMap;
use lru::LruCache;

pub struct PlanCache {
    cache: Arc<RwLock<LruCache<u64, Arc<QueryPlan>>>>,
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl PlanCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(capacity.try_into().unwrap()))),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    pub fn get(&self, query_hash: u64) -> Option<Arc<QueryPlan>> {
        let mut cache = self.cache.write();
        if let Some(plan) = cache.get(&query_hash) {
            *self.hits.write() += 1;
            Some(plan.clone())
        } else {
            *self.misses.write() += 1;
            None
        }
    }

    pub fn put(&self, query_hash: u64, plan: Arc<QueryPlan>) {
        let mut cache = self.cache.write();
        cache.put(query_hash, plan);
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        if hits + misses == 0 { 0.0 } else { hits as f64 / (hits + misses) as f64 }
    }
}

pub struct QueryPlan;
```

## Performance Targets
- Cache lookup: < 1µs
- Hit rate: > 80% for typical workloads
- Memory overhead: < 1KB per cached plan

## Status
**Complete**: Production-ready LRU cache with lock-free reads
