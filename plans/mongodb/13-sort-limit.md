# MongoDB Feature: Sort, Limit, and Skip

**Status**: 🔴 Not Started
**Priority**: Medium
**Dependencies**: MongoDB Update Operators, Match Stage
**Estimated Effort**: 1-2 weeks

---

## Overview

The `$sort`, `$limit`, and `$skip` stages provide control over the order and number of documents returned from an aggregation pipeline or standard query.

---

## Architecture & Design

### Data Structures
- PQL AST will need extensions for Sort, Limit, and Skip specific syntax.
- BSON to native type mappings must be thoroughly defined to handle complex structures.

### Execution Flow
1. **Parser**: Parse `Sort, Limit, and Skip` clauses and construct the AST nodes.
2. **Optimizer**: Check for index applicability and re-order operations if possible.
3. **Execution**: Stream documents through the execution nodes evaluating `Sort, Limit, and Skip`.

---

## Implementation Details

### Parser Updates
- Add support for `Sort, Limit, and Skip` within the PQL unified syntax.
- Ensure backwards compatibility with MongoDB's wire protocol when receiving these queries.

### Optimizer Rules
- Predicate pushdown for `Sort, Limit, and Skip` operations.
- Index selection using compound indexes if applicable.

### Edge Cases
- Handling of nulls and missing fields.
- Type mismatches in comparison.
- Nested array bounds.

---

## Distributed Systems Support

### Sharding Considerations
- Ensure queries utilizing `Sort, Limit, and Skip` route effectively to the appropriate shards.
- Support scatter-gather patterns if the query cannot be targeted.

### Fault Tolerance
- Consistent results regardless of node failure.
- Graceful degradation for `Sort, Limit, and Skip` execution.

---

## Testing Strategy

- **Unit tests**: Exhaustive tests covering all variants of `Sort, Limit, and Skip`.
- **Integration tests**: End-to-end queries including index hits.
- **Performance tests**: Large dataset evaluations (10k+ documents).
