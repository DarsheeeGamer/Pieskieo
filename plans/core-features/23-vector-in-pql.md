# Feature Plan: Vector Operations in PQL (Extended Examples)

**Feature ID**: core-features-023  
**Status**: ✅ Complete - Comprehensive PQL vector operation examples and patterns

---

## Overview

Provides **extended examples and patterns** for vector operations in PQL, demonstrating **real-world use cases** across AI search, recommendations, semantic analysis, and hybrid queries.

### Extended PQL Examples

```pql
-- 1. Semantic search with metadata filtering
QUERY documents
WHERE created_at > @last_month AND status = "published"
SIMILAR TO embed(@user_query) TOP 20 THRESHOLD 0.7
COMPUTE relevance = VECTOR_SCORE() * (1.0 + view_count / 10000.0)
ORDER BY relevance DESC
SELECT id, title, summary, relevance;

-- 2. Multi-vector search (title + content embeddings)
QUERY articles
SIMILAR TO embed_title(@query) IN title_embedding TOP 50
SIMILAR TO embed_content(@query) IN content_embedding TOP 50
COMPUTE combined_score = 
  0.3 * VECTOR_SCORE(title_embedding) + 
  0.7 * VECTOR_SCORE(content_embedding)
WHERE combined_score > 0.6
ORDER BY combined_score DESC
LIMIT 10;

-- 3. Recommendation with negative examples
QUERY products
SIMILAR TO @user_preference_vector TOP 100
NOT SIMILAR TO @disliked_items_centroid THRESHOLD 0.8
WHERE category IN @preferred_categories AND price <= @max_price
COMPUTE recommendation_score = 
  VECTOR_SCORE() * (1.0 - purchase_probability) * popularity
SELECT id, name, price, recommendation_score;

-- 4. Clustering with vector similarity
QUERY embeddings
SIMILAR TO CENTROID(cluster_id = @target_cluster) TOP 1000
GROUP BY cluster_id
COMPUTE
  avg_similarity = AVG(VECTOR_SCORE()),
  cluster_size = COUNT(),
  cluster_center = CENTROID(embedding)
SELECT cluster_id, avg_similarity, cluster_size;

-- 5. Time-weighted semantic search
QUERY memories
SIMILAR TO embed(@search_query) TOP 50
COMPUTE recency_weight = 1.0 / (1.0 + DAYS_SINCE(created_at) / 30.0),
        final_score = VECTOR_SCORE() * recency_weight
WHERE final_score > 0.5
ORDER BY final_score DESC
SELECT id, content, created_at, final_score;

-- 6. Multi-modal search (text + image embeddings)
QUERY products
SIMILAR TO @text_embedding IN description_embedding TOP 100
OR SIMILAR TO @image_embedding IN product_image_embedding TOP 100
COMPUTE multimodal_score = 
  MAX(VECTOR_SCORE(description_embedding), 
      VECTOR_SCORE(product_image_embedding))
WHERE stock_quantity > 0
ORDER BY multimodal_score DESC
LIMIT 20;

-- 7. Hybrid BM25 + vector search
QUERY articles
WHERE MATCH(title, content) AGAINST (@keywords)
SIMILAR TO embed(@query) TOP 100
COMPUTE hybrid_score = 
  0.4 * BM25_SCORE() + 
  0.6 * VECTOR_SCORE()
ORDER BY hybrid_score DESC
LIMIT 10;

-- 8. Batch vector insertion with auto-embedding
QUERY documents
CREATE NODES [
  { id: "doc1", content: "Machine learning tutorial", auto_embed: true },
  { id: "doc2", content: "Deep learning basics", auto_embed: true },
  { id: "doc3", content: "Neural networks explained", auto_embed: true }
]
WITH embedding_model = "text-embedding-3-large";
-- Automatically generates embeddings for content field

-- 9. Vector similarity join
QUERY users AS u1
SIMILAR TO u1.preference_vector IN users.preference_vector TOP 10
AS u2
WHERE u1.id != u2.id AND u2.active = true
SELECT u1.id AS user_id, 
       u2.id AS similar_user_id, 
       VECTOR_SCORE() AS similarity,
       u2.name;

-- 10. Anomaly detection with vector distance
QUERY events
COMPUTE distance_from_normal = VECTOR_DISTANCE(
  event_embedding,
  @normal_behavior_centroid
)
WHERE distance_from_normal > 2.5  -- 2.5 standard deviations
ORDER BY distance_from_normal DESC
SELECT id, event_type, timestamp, distance_from_normal;
```

---

## Common Vector Patterns

### Pattern 1: Semantic Search with Filters
```pql
QUERY {collection}
WHERE {metadata_filters}
SIMILAR TO {query_vector} TOP {k} THRESHOLD {min_score}
COMPUTE {derived_metrics}
ORDER BY {ranking_formula}
```

### Pattern 2: Re-ranking
```pql
QUERY {collection}
SIMILAR TO {query_vector} TOP {large_k}  -- Fetch candidates
COMPUTE rerank_score = {complex_formula}  -- Re-rank
ORDER BY rerank_score DESC
LIMIT {final_k};
```

### Pattern 3: Multi-Vector Fusion
```pql
QUERY {collection}
SIMILAR TO {vec1} IN {field1} TOP {k1}
SIMILAR TO {vec2} IN {field2} TOP {k2}
COMPUTE fusion_score = {weighted_combination}
```

### Pattern 4: Vector + Graph Traversal
```pql
QUERY {entities}
SIMILAR TO {query_vector} TOP {k}
TRAVERSE {edge_type} DEPTH {min} TO {max}
WHERE {edge_filters}
COMPUTE path_score = VECTOR_SCORE() * GRAPH_RELEVANCE()
```

---

## Vector Functions Reference

| Function | Description | Example |
|----------|-------------|---------|
| `VECTOR_SCORE()` | Similarity score from SIMILAR TO | `VECTOR_SCORE()` |
| `VECTOR_DISTANCE(v1, v2)` | Euclidean distance | `VECTOR_DISTANCE(a, b)` |
| `COSINE_SIMILARITY(v1, v2)` | Cosine similarity | `COSINE_SIMILARITY(a, b)` |
| `DOT_PRODUCT(v1, v2)` | Dot product | `DOT_PRODUCT(a, b)` |
| `CENTROID(field)` | Average vector | `CENTROID(embedding)` |
| `embed(text)` | Generate embedding | `embed("search query")` |
| `embed_batch(texts)` | Batch embeddings | `embed_batch(ARRAY[t1, t2])` |

---

**Status**: ✅ Complete  
Comprehensive PQL vector operation examples demonstrating real-world semantic search, recommendations, and hybrid query patterns.
