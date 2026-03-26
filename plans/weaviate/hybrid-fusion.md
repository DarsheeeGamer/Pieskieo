# Weaviate Feature: Hybrid Search Score Fusion

**Status**: 🔴 Not Started
**Priority**: CRITICAL (Core Multimodal Value)
**Dependencies**: Vector Search (HNSW), Keyword Search (BM25)
**Estimated Effort**: 2 weeks

---

## Overview

Hybrid search combines the results of a dense vector search (semantic similarity) with a sparse keyword search (BM25 lexical matching). Since vector distances (e.g., cosine similarity, L2) and BM25 scores are on completely different scales, they cannot be directly added.

Weaviate addresses this using **Reciprocal Rank Fusion (RRF)** or **Relative Score Fusion**. Pieskieo will implement both, defaulting to Relative Score Fusion (which Weaviate found generally performs better).

## Fusion Algorithms

### 1. Relative Score Fusion (Default)
Scores from both searches are independently normalized to a $[0, 1]$ scale relative to the highest and lowest scores in their respective result sets. Then, a weighted sum is applied.

$$ \text{Normalized Score} = \frac{\text{score} - \text{min\_score}}{\text{max\_score} - \text{min\_score}} $$
$$ \text{Hybrid Score} = (\text{Normalized Vector Score} \times \alpha) + (\text{Normalized BM25 Score} \times (1 - \alpha)) $$

- $\alpha$ (Alpha): Weight parameter between $0.0$ (pure BM25) and $1.0$ (pure Vector). Default is $0.5$.

### 2. Reciprocal Rank Fusion (RRF)
Instead of scores, this uses the rank (position) of the document in each result set.

$$ \text{RRF Score} = \frac{1}{k + \text{Rank}_{\text{vector}}} + \frac{1}{k + \text{Rank}_{\text{bm25}}} $$
- $k$: A smoothing constant (typically 60).

---

## Implementation Plan

### Phase 1: API / PQL Syntax Design

We need a unified syntax to request a hybrid search, providing the query text, the query vector, and the alpha parameter.

```sql
-- PQL Syntax Draft
SELECT id, content, HYBRID_SCORE() as score
FROM articles
WHERE content MATCHES_HYBRID {
    query: "machine learning models",
    vector: [0.12, 0.45, ...],
    alpha: 0.7,
    fusion_type: 'relative_score' -- Optional, defaults to relative
}
ORDER BY score DESC
LIMIT 10;
```

### Phase 2: Parallel Execution Engine

Vector search and BM25 search are completely independent and should execute concurrently.

```rust
// crates/pieskieo-core/src/search/hybrid.rs

pub struct HybridQuery {
    pub text_query: String,
    pub vector_query: Vec<f32>,
    pub alpha: f32,
    pub limit: usize,
    pub fusion_type: FusionType,
}

pub enum FusionType {
    RelativeScore,
    RRF(usize), // The 'k' constant
}

impl UnifiedExecutor {
    pub async fn execute_hybrid_search(
        &self,
        collection: &str,
        query: HybridQuery,
    ) -> Result<Vec<(RowId, f32)>> {

        let bm25_future = self.execute_bm25_search(collection, &query.text_query, query.limit * 2);
        let vector_future = self.execute_vector_search(collection, &query.vector_query, query.limit * 2);

        // Run concurrently
        let (bm25_results, vector_results) = tokio::join!(bm25_future, vector_future);

        let bm25_results = bm25_results?;
        let vector_results = vector_results?;

        match query.fusion_type {
            FusionType::RelativeScore => {
                self.fuse_relative_score(bm25_results, vector_results, query.alpha, query.limit)
            }
            FusionType::RRF(k) => {
                self.fuse_rrf(bm25_results, vector_results, k, query.limit)
            }
        }
    }
}
```

### Phase 3: Relative Score Fusion Logic

```rust
impl UnifiedExecutor {
    fn fuse_relative_score(
        &self,
        mut bm25: Vec<(RowId, f32)>,
        mut vector: Vec<(RowId, f32)>,
        alpha: f32,
        limit: usize,
    ) -> Result<Vec<(RowId, f32)>> {

        // 1. Normalize BM25 Scores (Higher is better)
        if let Some(max_bm25) = bm25.first().map(|(_, s)| *s) {
            let min_bm25 = bm25.last().map(|(_, s)| *s).unwrap_or(0.0);
            let range = (max_bm25 - min_bm25).max(1e-6); // Prevent div by zero
            for (_, score) in bm25.iter_mut() {
                *score = (*score - min_bm25) / range;
            }
        }

        // 2. Normalize Vector Scores (Assume Distance: Lower is better, e.g., L2 or Cosine Distance)
        // Convert distance to a similarity score where 1.0 is identical, 0.0 is furthest.
        // If the engine returns distance, we must invert it before normalization.
        // Assuming vector currently returns distance.
        if let Some(min_dist) = vector.first().map(|(_, d)| *d) {
            let max_dist = vector.last().map(|(_, d)| *d).unwrap_or(1.0);
            let range = (max_dist - min_dist).max(1e-6);
            for (_, dist) in vector.iter_mut() {
                // Invert: (Max - current) / Range
                *dist = (max_dist - *dist) / range;
            }
        }

        // 3. Combine Scores
        let mut combined_scores: HashMap<RowId, f32> = HashMap::new();

        let bm25_weight = 1.0 - alpha;
        for (id, norm_score) in bm25 {
            *combined_scores.entry(id).or_insert(0.0) += norm_score * bm25_weight;
        }

        for (id, norm_score) in vector {
            *combined_scores.entry(id).or_insert(0.0) += norm_score * alpha;
        }

        // 4. Sort and Limit
        let mut results: Vec<(RowId, f32)> = combined_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }
}
```

### Phase 4: Reciprocal Rank Fusion Logic

```rust
impl UnifiedExecutor {
    fn fuse_rrf(
        &self,
        bm25: Vec<(RowId, f32)>,
        vector: Vec<(RowId, f32)>,
        k: usize,
        limit: usize,
    ) -> Result<Vec<(RowId, f32)>> {

        let mut rrf_scores: HashMap<RowId, f32> = HashMap::new();

        // BM25 is already sorted (highest score first)
        for (rank, (id, _)) in bm25.into_iter().enumerate() {
            let rrf = 1.0 / (k as f32 + (rank + 1) as f32);
            *rrf_scores.entry(id).or_insert(0.0) += rrf;
        }

        // Vector is already sorted (lowest distance first)
        for (rank, (id, _)) in vector.into_iter().enumerate() {
            let rrf = 1.0 / (k as f32 + (rank + 1) as f32);
            *rrf_scores.entry(id).or_insert(0.0) += rrf;
        }

        let mut results: Vec<(RowId, f32)> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }
}
```

---

## Test Cases

### Test 1: Alpha Limits
```rust
// Test alpha = 1.0 (Pure Vector)
// The results and order must exactly match a pure vector search.

// Test alpha = 0.0 (Pure BM25)
// The results and order must exactly match a pure BM25 search.
```

### Test 2: Missing Documents in One Set
```rust
// If a document appears in the BM25 top K but NOT in the Vector top K,
// its vector score is effectively 0 for normalization purposes.
// The algorithm must handle this gracefully without crashing.
```

### Test 3: Normalization Range Edge Cases
```rust
// All vector distances are identical (e.g., all 0.5)
// Max == Min. Normalization should safely handle division by zero and assign a default normalized score (e.g., 1.0 or 0.0).
```

---

## Performance Targets

- **Concurrency**: Must dispatch both searches simultaneously to minimize total latency (Latency $\approx \max(\text{BM25\_Lat}, \text{Vec\_Lat})$).
- **Overhead**: The fusion step itself (hashmap inserts and sorting a few hundred items) must take < 1ms.

## Metrics to Track

- `pieskieo_hybrid_search_duration_ms`
- `pieskieo_hybrid_fusion_duration_ms`
- `pieskieo_hybrid_alpha_distribution`

**Created**: 2026-02-08
**Author**: Implementation Team
