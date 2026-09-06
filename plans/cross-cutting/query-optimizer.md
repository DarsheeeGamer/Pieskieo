# Pieskieo Feature: Query Optimizer Architecture

**Status**: 🔴 Not Started
**Priority**: High
**Dependencies**: Parser Architecture
**Estimated Effort**: 4-5 weeks

## Overview

The query optimizer is the bridge between the logical AST and the execution engine. For a multimodal database like Pieskieo, the optimizer is exceptionally critical; it must harmonize operations across distinct storage models—relational tables, JSON documents, graph data, and vectors. It is responsible for selecting the most efficient physical plan by leveraging index information, filter pushdowns, and cross-model optimizations.

## Architecture Highlights

- **Logical Plan Transformation**: Applies rule-based rewrites to the AST (e.g., predicate pushdown, constant folding, and query simplification).
- **Physical Plan Selection**: Maps logical operators to efficient physical execution operators (e.g., deciding between an index scan and a sequential scan).
- **Multimodal Optimization**: Intelligently orders operations involving multiple storage engines, such as prioritizing a highly selective vector similarity search before joining with a massive relational table.
- **Join Planning**: Analyzes table relationships to determine the optimal join execution order and algorithm.

## Implementation Plan

### Phase 1: Logical Plan Optimization

Implement a robust rule-based transformation engine:
- Convert the AST into an internal Logical Plan representation.
- **Predicate Pushdown**: Move filter clauses as close to the data sources as possible.
- **Projection Pruning**: Eliminate unused columns/fields early in the query lifecycle.
- **Subquery Decorrelation**: Rewrite correlated subqueries into efficient joins (integrates with Subquery plan).

### Phase 2: Index Utilization and Physical Mapping

Develop the physical plan generator:
- **Index Selection**: Assess available B-Tree, HNSW, or Full-Text indexes and select the optimal scan methodology.
- **Model-Specific Selection**: Differentiate execution paths based on target models (e.g., use optimized Cypher traversal for graph nodes vs. sequential scan for documents).

### Phase 3: Multimodal Join Optimization

Implement strategy for cross-model queries:
- **Join Ordering**: Establish heuristics or deterministic rules to order joins, significantly minimizing intermediate result sizes.
- **Cross-Model Execution Planning**: Outline the strategy for bridging execution boundaries (e.g., when a vector query result is used as a filter in a graph traversal).

### Phase 4: Plan Caching

Enhance performance for repeated queries:
- **Query Fingerprinting**: Generate a hash/signature for logical plans.
- **Cache Management**: Store pre-compiled physical plans for high-frequency queries to bypass parsing and optimization overhead.

## Testing and Validation

- **Unit Testing**: Validate that logic transformations (like predicate pushdown) yield the correctly altered logical plan.
- **Optimization Scenarios**: Construct complex multi-model queries and assert that the generated physical plan matches the expected optimized outcome.
- **Benchmark Comparisons**: Compare execution times of complex queries with and without optimization enabled to empirically prove its efficacy.

## Performance Metrics

- Optimization phase must introduce minimal overhead (target < 2ms for complex queries).
- Optimized physical plans should result in significantly lower resource utilization compared to naive execution models.

---
**Created**: 2026-02-08
**Author**: Implementation Team
