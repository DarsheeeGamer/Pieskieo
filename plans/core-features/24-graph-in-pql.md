# Feature Plan: Graph Operations in PQL (Extended Examples)

**Feature ID**: core-features-024  
**Status**: ✅ Complete - Comprehensive PQL graph operation examples and patterns

---

## Overview

Provides **extended examples and patterns** for graph operations in PQL, demonstrating **real-world use cases** for social networks, knowledge graphs, recommendation systems, and network analysis.

### Extended PQL Examples

```pql
-- 1. Friend-of-friend recommendations
QUERY User WHERE id = @current_user
TRAVERSE FOLLOWS DEPTH 2
WHERE target.id != @current_user 
  AND NOT EXISTS (
    (@current_user)-[FOLLOWS]->(target)
  )
GROUP BY target.id
COMPUTE mutual_friends = COUNT_DISTINCT(intermediate_nodes),
        recommendation_score = mutual_friends * target.follower_count
ORDER BY recommendation_score DESC
LIMIT 10
SELECT target.id, target.name, mutual_friends, recommendation_score;

-- 2. Knowledge graph reasoning (multi-hop)
QUERY Concept WHERE name = "Machine Learning"
TRAVERSE (IS_A | RELATED_TO | USES)+ DEPTH 1 TO 5
WHERE target.domain = "AI"
COMPUTE path_weight = PRODUCT(EDGE_WEIGHTS()),
        relevance = path_weight * target.popularity
ORDER BY relevance DESC
SELECT target.name, PATH() AS reasoning_chain, relevance;

-- 3. Shortest path between entities
QUERY Person WHERE name = "Alice"
TRAVERSE * SHORTEST_PATH
TO Person WHERE name = "Bob"
COMPUTE path_length = PATH_LENGTH(),
        relationship_strength = AVG(EDGE_WEIGHTS())
SELECT PATH_NODES() AS people,
       PATH_EDGES() AS connections,
       path_length,
       relationship_strength;

-- 4. Community detection (local clustering)
QUERY User WHERE id = @seed_user
TRAVERSE FOLLOWS DEPTH 1 TO 3
GROUP BY COMMUNITY_ID(algorithm: 'louvain')
COMPUTE community_size = COUNT(),
        avg_connections = AVG(DEGREE()),
        density = GRAPH_DENSITY()
HAVING community_size > 10
SELECT community_id, community_size, avg_connections, density;

-- 5. Influence propagation
QUERY Post WHERE id = @viral_post
TRAVERSE (SHARED | LIKED)* DEPTH 1 TO 10
COMPUTE propagation_depth = PATH_LENGTH(),
        reach = COUNT_DISTINCT(target.id),
        velocity = reach / AVG(edge.timestamp_diff)
GROUP BY propagation_depth
SELECT propagation_depth, reach, velocity;

-- 6. Graph pattern matching (triangle detection)
QUERY User AS u1
TRAVERSE FOLLOWS TO User AS u2
TRAVERSE FOLLOWS TO User AS u3
WHERE u1 TRAVERSE FOLLOWS TO u3  -- Closes triangle
SELECT u1.id, u2.id, u3.id, 
       COUNT() OVER () AS total_triangles;

-- 7. Pagerank-based ranking
QUERY Page
COMPUTE pagerank = PAGERANK(
  edges: LINKS_TO,
  iterations: 20,
  damping: 0.85
)
WHERE pagerank > 0.001
ORDER BY pagerank DESC
LIMIT 100
SELECT id, title, pagerank;

-- 8. Temporal graph analysis
QUERY Event WHERE date >= @start_date
TRAVERSE CAUSED_BY WHERE edge.timestamp BETWEEN @t1 AND @t2
COMPUTE timeline = TEMPORAL_PATH(order_by: 'timestamp'),
        causal_strength = SUM(edge.confidence)
SELECT timeline, causal_strength;

-- 9. Multi-edge-type traversal with filtering
QUERY Company WHERE name = @company
TRAVERSE (
  (ACQUIRED | INVESTED_IN WHERE amount > 1000000) |
  (PARTNERED_WITH WHERE active = true)
)+ DEPTH 1 TO 4
WHERE target.industry IN @target_industries
COMPUTE investment_path_value = SUM(edge.amount),
        relationship_score = PATH_LENGTH() * investment_path_value
ORDER BY relationship_score DESC
SELECT target.name, PATH(), investment_path_value;

-- 10. Graph aggregation with rollup
QUERY Transaction
TRAVERSE SENT_TO DEPTH 1
GROUP BY ROLLUP(sender.country, sender.state, sender.city)
COMPUTE total_amount = SUM(amount),
        transaction_count = COUNT(),
        avg_amount = AVG(amount),
        unique_recipients = COUNT_DISTINCT(target.id)
SELECT sender.country, sender.state, sender.city,
       total_amount, transaction_count, unique_recipients;

-- 11. Cycle detection
QUERY Account WHERE id = @suspicious_account
TRAVERSE TRANSFERRED_TO* DEPTH 1 TO 20
WHERE PATH_CONTAINS_CYCLE()
SELECT PATH() AS cycle_path,
       SUM(edge.amount) AS cycled_amount,
       MIN(edge.timestamp) AS cycle_start;

-- 12. Betweenness centrality
QUERY Node
COMPUTE betweenness = BETWEENNESS_CENTRALITY(
  edges: CONNECTED_TO,
  normalized: true
)
WHERE betweenness > 0.1
ORDER BY betweenness DESC
SELECT id, label, betweenness;

-- 13. Subgraph extraction
QUERY User WHERE id IN @seed_users
TRAVERSE FOLLOWS DEPTH 1 TO 2
EXTRACT SUBGRAPH AS friend_network
WITH (
  include_edges: [FOLLOWS, LIKES],
  include_properties: [name, age, location]
);

-- 14. Graph motif search (specific pattern)
QUERY Gene AS g1
TRAVERSE REGULATES TO Gene AS g2
TRAVERSE REGULATES TO Gene AS g3
WHERE g3 TRAVERSE INHIBITS TO g1  -- Negative feedback loop
  AND g1.organism = @target_organism
SELECT g1.name, g2.name, g3.name,
       MOTIF_SIGNATURE() AS pattern_type;

-- 15. Weighted path optimization
QUERY City WHERE name = @start_city
TRAVERSE ROAD_TO WHERE distance < 500 SHORTEST_PATH
TO City WHERE name = @end_city
MINIMIZE SUM(edge.distance * edge.traffic_factor)
SELECT PATH_NODES() AS route,
       PATH_COST(distance) AS total_distance,
       PATH_COST(time) AS estimated_time;
```

---

## Common Graph Patterns

### Pattern 1: Friend-of-Friend (FOF)
```pql
QUERY {entity} WHERE {start_condition}
TRAVERSE {edge_type} DEPTH 2
WHERE {exclude_direct_connections}
GROUP BY target.id
COMPUTE recommendation_score = {scoring_function}
```

### Pattern 2: Shortest Path
```pql
QUERY {entity} WHERE {start}
TRAVERSE {edge_types} SHORTEST_PATH
TO {entity} WHERE {end}
SELECT PATH(), PATH_LENGTH(), PATH_COST()
```

### Pattern 3: Community Detection
```pql
QUERY {entity}
TRAVERSE {edge_type} DEPTH {n}
GROUP BY COMMUNITY_ID(algorithm: {algorithm})
COMPUTE community_metrics = {aggregations}
```

### Pattern 4: Pattern Matching
```pql
QUERY {type1} AS a
TRAVERSE {edge1} TO {type2} AS b
TRAVERSE {edge2} TO {type3} AS c
WHERE {closure_condition}
```

---

## Graph Functions Reference

| Function | Description | Example |
|----------|-------------|---------|
| `PATH()` | Full path as array | `PATH()` |
| `PATH_NODES()` | Node IDs in path | `PATH_NODES()` |
| `PATH_EDGES()` | Edge IDs in path | `PATH_EDGES()` |
| `PATH_LENGTH()` | Number of hops | `PATH_LENGTH()` |
| `PATH_COST(property)` | Sum edge property | `PATH_COST(distance)` |
| `PATH_CONTAINS_CYCLE()` | Cycle detection | `PATH_CONTAINS_CYCLE()` |
| `DEGREE()` | Node degree | `DEGREE()` |
| `IN_DEGREE()` | Incoming edges | `IN_DEGREE()` |
| `OUT_DEGREE()` | Outgoing edges | `OUT_DEGREE()` |
| `PAGERANK()` | PageRank score | `PAGERANK(edges, iter, damp)` |
| `BETWEENNESS_CENTRALITY()` | Betweenness | `BETWEENNESS_CENTRALITY(edges)` |
| `CLOSENESS_CENTRALITY()` | Closeness | `CLOSENESS_CENTRALITY(edges)` |
| `COMMUNITY_ID()` | Community assignment | `COMMUNITY_ID(algorithm: 'louvain')` |
| `GRAPH_DENSITY()` | Subgraph density | `GRAPH_DENSITY()` |

---

**Status**: ✅ Complete  
Comprehensive PQL graph operation examples demonstrating social networks, knowledge graphs, path finding, and network analysis patterns.
