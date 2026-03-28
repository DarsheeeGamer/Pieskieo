# PostgreSQL Feature: Cost-Based Optimizer (CBO)

**Status**: 🔴 Not Started
**Priority**: CRITICAL
**Dependencies**: Statistics Collection (ANALYZE)
**Estimated Effort**: 4-6 weeks

---

## Overview

A Cost-Based Optimizer (CBO) evaluates different execution strategies (execution plans) for a SQL query and chooses the one with the lowest estimated "cost". The cost represents the anticipated resource consumption (I/O, CPU) to execute the plan. This is a massive upgrade over a rule-based optimizer.

## Core Components

1. **Plan Enumeration**: Generate alternative ways to execute the query (e.g., different join orders, index vs. sequential scan).
2. **Cardinality Estimation**: Estimate how many rows will be output by each node in the plan, using statistics.
3. **Cost Model**: Assign a numerical cost to each operation based on cardinality and system parameters.

---

## Implementation Plan

### Phase 1: Statistics Representation

The optimizer needs accurate data about the tables.

```rust
// crates/pieskieo-core/src/optimizer/stats.rs

pub struct ColumnStats {
    // Number of distinct values
    pub ndistinct: i64,
    // Fraction of null values
    pub null_frac: f64,
    // Average width in bytes
    pub avg_width: i32,
    // Most Common Values (MCVs) and their frequencies
    pub most_common_vals: Vec<Value>,
    pub most_common_freqs: Vec<f64>,
    // Histogram bounds for range queries
    pub histogram_bounds: Vec<Value>,
    // Correlation between physical row order and column value order
    pub correlation: f64,
}

pub struct TableStats {
    pub reltuples: f64, // Total estimated rows
    pub relpages: i32,  // Total estimated disk pages
    pub column_stats: HashMap<String, ColumnStats>,
}
```

### Phase 2: Cost Model Parameters

PostgreSQL defines several cost variables that we need to implement.

```rust
// crates/pieskieo-core/src/optimizer/cost.rs

pub struct CostParameters {
    // CPU cost to process a tuple
    pub cpu_tuple_cost: f64,        // e.g., 0.01
    // CPU cost to evaluate an index entry
    pub cpu_index_tuple_cost: f64,  // e.g., 0.005
    // CPU cost to evaluate an operator/function
    pub cpu_operator_cost: f64,     // e.g., 0.0025
    // Cost to read a sequential page
    pub seq_page_cost: f64,         // e.g., 1.0
    // Cost to read a random page
    pub random_page_cost: f64,      // e.g., 4.0 (higher for HDDs, lower for SSDs)
}

pub struct PathCost {
    pub startup_cost: f64, // Cost before first row is returned
    pub total_cost: f64,   // Total cost to return all rows
    pub rows: f64,         // Estimated output rows
}
```

### Phase 3: Cardinality Estimation

Estimate the selectivity of predicates (WHERE clauses).

```rust
// crates/pieskieo-core/src/optimizer/cardinality.rs

impl Optimizer {
    fn estimate_selectivity(&self, expr: &Expr, stats: &TableStats) -> f64 {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                if let (Expr::Identifier(col), Expr::Value(val)) = (&**left, &**right) {
                    if let Some(col_stats) = stats.column_stats.get(col) {
                        return self.estimate_op_selectivity(col_stats, op, val);
                    }
                }
                // Default selectivity if unknown
                0.33
            }
            Expr::And(left, right) => {
                // Assuming independence (P(A and B) = P(A) * P(B))
                self.estimate_selectivity(left, stats) * self.estimate_selectivity(right, stats)
            }
            Expr::Or(left, right) => {
                let s1 = self.estimate_selectivity(left, stats);
                let s2 = self.estimate_selectivity(right, stats);
                // P(A or B) = P(A) + P(B) - P(A and B)
                s1 + s2 - (s1 * s2)
            }
            // ...
        }
    }

    fn estimate_op_selectivity(&self, stats: &ColumnStats, op: &BinaryOperator, val: &Value) -> f64 {
        match op {
            BinaryOperator::Eq => {
                // 1. Check MCVs
                if let Some(idx) = stats.most_common_vals.iter().position(|v| v == val) {
                    return stats.most_common_freqs[idx];
                }
                // 2. Uniform distribution assumption for the rest
                if stats.ndistinct > 0 {
                    let mcv_freq_sum: f64 = stats.most_common_freqs.iter().sum();
                    (1.0 - mcv_freq_sum) / (stats.ndistinct as f64 - stats.most_common_vals.len() as f64)
                } else {
                    1.0 / stats.ndistinct.abs() as f64
                }
            }
            BinaryOperator::Gt | BinaryOperator::Lt => {
                // Use histogram bounds to estimate range selectivity
                self.estimate_range_selectivity(stats, op, val)
            }
            // ...
        }
    }
}
```

### Phase 4: Path Generation and Costing

Generate physical access paths for a single relation.

```rust
// crates/pieskieo-core/src/optimizer/paths.rs

pub enum AccessPath {
    SeqScan { table: String, filter: Option<Expr> },
    IndexScan { table: String, index: String, index_cond: Expr, filter: Option<Expr> },
    BitmapHeapScan { table: String, bitmap_path: Box<AccessPath> },
}

impl Optimizer {
    fn build_access_paths(&self, rel: &RelationInfo) -> Vec<(AccessPath, PathCost)> {
        let mut paths = Vec::new();

        // 1. Seq Scan Path
        let seq_cost = self.cost_seqscan(rel);
        paths.push((AccessPath::SeqScan { ... }, seq_cost));

        // 2. Index Paths
        for index in &rel.indexes {
            if let Some(index_cond) = self.match_index_clauses(rel, index) {
                let idx_cost = self.cost_indexscan(rel, index, &index_cond);
                paths.push((AccessPath::IndexScan { ... }, idx_cost));
            }
        }

        // Keep the cheapest path (or multiple if they have different sort orders)
        self.add_path_to_rel(rel, paths);
    }

    fn cost_seqscan(&self, rel: &RelationInfo) -> PathCost {
        let startup_cost = 0.0;
        let run_cost = self.params.cpu_tuple_cost * rel.stats.reltuples
                     + self.params.seq_page_cost * rel.stats.relpages as f64;

        let selectivity = self.estimate_selectivity(&rel.baserestrictinfo, &rel.stats);
        let rows = (rel.stats.reltuples * selectivity).round();

        PathCost { startup_cost, total_cost: startup_cost + run_cost, rows }
    }
}
```

### Phase 5: Join Planning (Dynamic Programming)

For joins involving $N$ relations, we must find the optimal join order and join method (Nested Loop, Hash, Merge).

```rust
// crates/pieskieo-core/src/optimizer/join.rs

impl Optimizer {
    fn make_join_rel(&self, outer: &RelOptInfo, inner: &RelOptInfo, join_clauses: &[Expr]) -> Option<RelOptInfo> {
        // Estimate join cardinality
        let join_rows = self.estimate_join_cardinality(outer, inner, join_clauses);

        let mut joinrel = RelOptInfo::new(outer.relids.union(&inner.relids), join_rows);

        // 1. Consider Nested Loop Join
        let nl_cost = self.cost_nestedloop(outer, inner, join_clauses);
        joinrel.add_path(JoinPath::NestedLoop { ... }, nl_cost);

        // 2. Consider Hash Join (if equality conditions exist)
        if self.has_hashable_clauses(join_clauses) {
            let hash_cost = self.cost_hashjoin(outer, inner, join_clauses);
            joinrel.add_path(JoinPath::HashJoin { ... }, hash_cost);
        }

        // 3. Consider Merge Join (if paths are sorted or can be sorted)
        if self.has_mergeable_clauses(join_clauses) {
            let merge_cost = self.cost_mergejoin(outer, inner, join_clauses);
            joinrel.add_path(JoinPath::MergeJoin { ... }, merge_cost);
        }

        Some(joinrel)
    }

    fn standard_join_search(&self, initial_rels: Vec<RelOptInfo>) -> RelOptInfo {
        // Dynamic Programming: Level n builds on level n-1
        // Example for 3 relations (A, B, C):
        // Level 1: {A}, {B}, {C}
        // Level 2: {A,B}, {A,C}, {B,C}
        // Level 3: {A,B,C}

        let mut levels = vec![initial_rels];
        let n = levels[0].len();

        for level in 2..=n {
            let mut current_level_rels = Vec::new();

            // Try joining relations from level K with level (level-K)
            for k in 1..level {
                for outer in &levels[k - 1] {
                    for inner in &levels[level - k - 1] {
                        if !outer.relids.is_disjoint(&inner.relids) { continue; }

                        // Check if they have a valid join clause (avoid cross joins if possible)
                        if let Some(join_clauses) = self.find_join_clauses(outer, inner) {
                            if let Some(new_rel) = self.make_join_rel(outer, inner, &join_clauses) {
                                current_level_rels.push(new_rel);
                            }
                        }
                    }
                }
            }
            // Group and prune redundant relations (keep only the cheapest paths for exactly the same relid set)
            levels.push(self.prune_join_rels(current_level_rels));
        }

        levels.last().unwrap().into_iter().min_by_key(|rel| rel.cheapest_total_cost()).unwrap().clone()
    }
}
```

### Phase 6: Subquery Optimization (Integration)

Integrate with the decorrelation techniques from `01-subqueries.md`.
The optimizer must cost the decorrelated plan (e.g., using a Semi-Join) against evaluating the subquery per row (Nested Loop).

---

## Test Cases

### Test 1: Index Selection
```sql
CREATE TABLE users (id INT, age INT, country TEXT);
CREATE INDEX idx_age ON users(age);
-- Insert 1M rows, age uniformly distributed 1-100.
ANALYZE users;

-- Cost model should pick Index Scan
EXPLAIN SELECT * FROM users WHERE age = 25;

-- Cost model should pick Seq Scan (selectivity too high)
EXPLAIN SELECT * FROM users WHERE age > 10;
```

### Test 2: Join Order Selection
```sql
CREATE TABLE a (id INT); CREATE TABLE b (id INT, a_id INT); CREATE TABLE c (id INT, b_id INT);
-- a has 10 rows, b has 10,000 rows, c has 1,000,000 rows
ANALYZE a; ANALYZE b; ANALYZE c;

-- Optimizer should order joins to minimize intermediate rows: (a JOIN b) JOIN c
EXPLAIN SELECT * FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id WHERE a.id = 1;
```

---

## Performance Targets

- **Planning Time**: < 2ms for queries with up to 5 joins. < 10ms for up to 12 joins.
- **Join Search Strategy**: Use dynamic programming for up to 12 tables. Fall back to Genetic Query Optimizer (GEQO) for >12 tables to prevent exponential planning time.

## Metrics to Track

- `pieskieo_optimizer_planning_time_ms`
- `pieskieo_optimizer_plan_cache_hits`
- `pieskieo_optimizer_geqo_invocations`

**Created**: 2026-02-08
**Author**: Implementation Team
