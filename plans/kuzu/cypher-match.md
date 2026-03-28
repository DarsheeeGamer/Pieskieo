# Kùzu Feature: Cypher MATCH Patterns

**Status**: 🔴 Not Started
**Priority**: CRITICAL (Graph Query Foundation)
**Dependencies**: Graph Storage Engine
**Estimated Effort**: 4-5 weeks

---

## Overview

The `MATCH` clause is the heart of Cypher. It allows users to specify patterns of nodes and relationships to find in the graph. Implementing `MATCH` requires a parser for graph patterns, a plan to execute those patterns (often via joins or specialized graph traversal operators), and integration into the broader Unified Query Language.

## Core Syntax Supported

1. **Node Patterns**: `(a)`, `(a:Person)`, `(a:Person {name: 'Alice'})`
2. **Relationship Patterns**: `-[r]-`, `-[r:KNOWS]->`, `<-[r:LIKES]-`
3. **Path Patterns**: `(a)-[r]->(b)`, `(a)-[:KNOWS]->(b)-[:LIKES]->(c)`
4. **Variable Binding**: Binding nodes and relationships to variables (`a`, `r`, `b`) for use in subsequent clauses (e.g., `WHERE`, `RETURN`).

---

## Implementation Plan

### Phase 1: Cypher Pattern AST

Extend our Unified Query AST to represent graph patterns.

```rust
// crates/pieskieo-core/src/pql/ast.rs

#[derive(Clone, Debug, PartialEq)]
pub enum GraphPattern {
    Node(NodePattern),
    Path(Vec<PatternElement>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum PatternElement {
    Node(NodePattern),
    Relationship(RelationshipPattern),
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub labels: Vec<String>,
    pub properties: Option<HashMap<String, Expr>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelationshipPattern {
    pub variable: Option<String>,
    pub types: Vec<String>,
    pub direction: Direction,
    pub properties: Option<HashMap<String, Expr>>,
    // Variable length path parameters (e.g., *1..3)
    pub length: Option<Range<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Direction {
    Outgoing, // ->
    Incoming, // <-
    Both,     // -
}
```

### Phase 2: Translation to Relational Algebra (Joins)

Since Pieskieo is fundamentally a relational engine at its core (with extensions), a `MATCH` clause can be translated into a series of `JOIN`s between node tables and edge tables.

*Assuming we store graphs as:*
- **Nodes**: A table `nodes` (id, label, properties) OR typed tables (e.g., `Person`).
- **Edges**: A table `edges` (src_id, dst_id, type, properties).

**Example Cypher:**
```cypher
MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person)
RETURN a.name, b.name
```

**Translated to SQL/Relational Plan:**
```sql
SELECT a.properties->>'name', b.properties->>'name'
FROM Person a
JOIN edges r ON a.id = r.src_id AND r.type = 'KNOWS'
JOIN Person b ON r.dst_id = b.id;
```

```rust
// crates/pieskieo-core/src/engine/cypher_translate.rs

impl CypherTranslator {
    pub fn translate_match(&self, pattern: &GraphPattern) -> Result<SelectStatement> {
        match pattern {
            GraphPattern::Path(elements) => self.translate_path(elements),
            GraphPattern::Node(node) => self.translate_single_node(node),
        }
    }

    fn translate_path(&self, elements: &[PatternElement]) -> Result<SelectStatement> {
        let mut from_clauses = Vec::new();
        let mut current_node_alias = None;

        for (i, element) in elements.iter().enumerate() {
            match element {
                PatternElement::Node(n) => {
                    let alias = n.variable.clone().unwrap_or_else(|| format!("__node_{}", i));
                    let table_source = self.build_node_table_source(n, &alias);

                    if i == 0 {
                        from_clauses.push(table_source);
                    } else {
                        // We must join this node to the preceding relationship
                        // This logic handled when processing the relationship
                    }
                    current_node_alias = Some(alias);
                }
                PatternElement::Relationship(r) => {
                    let prev_alias = current_node_alias.clone().unwrap();
                    let rel_alias = r.variable.clone().unwrap_or_else(|| format!("__rel_{}", i));

                    // Look ahead for the next node
                    if let Some(PatternElement::Node(next_node)) = elements.get(i + 1) {
                        let next_alias = next_node.variable.clone().unwrap_or_else(|| format!("__node_{}", i+1));

                        // Construct the joins
                        let rel_join = self.build_relationship_join(r, &prev_alias, &rel_alias, &next_alias);
                        from_clauses.push(rel_join);
                        // We push the next node join as well
                    } else {
                        return Err(PieskieoError::SyntaxError("Relationship pattern must end with a node".into()));
                    }
                }
            }
        }

        // Construct final select statement
        Ok(SelectStatement {
            from: from_clauses,
            ..Default::default()
        })
    }
}
```

### Phase 3: Worst-Case Optimal Joins (WCOJ) Preparation

Translating paths to binary joins (e.g., A join B, then result join C) works fine for simple chains. However, for cyclic queries (e.g., triangles: A knows B, B knows C, C knows A), standard binary joins are extremely slow (often producing huge intermediate results).

Kùzu excels here by using **Worst-Case Optimal Joins (WCOJ)** (like the Leapfrog Triejoin).

*Our Phase 1 implementation will use standard binary joins.* However, the query planner must be designed to recognize cyclic patterns and eventually route them to a specialized WCOJ physical operator.

```rust
// crates/pieskieo-core/src/optimizer/graph.rs

impl Optimizer {
    fn plan_graph_match(&self, match_ast: &GraphPattern) -> ExecutionPlan {
        if self.detect_cycles(match_ast) {
            // Future feature: route to WCOJ executor
            ExecutionPlan::WorstCaseOptimalJoin { ... }
        } else {
            // Acyclic path: use standard hash/merge joins
            ExecutionPlan::NestedBinaryJoins { ... }
        }
    }
}
```

### Phase 4: Variable Length Paths (Recursion)

A pattern like `(a)-[:KNOWS*1..3]->(b)` cannot be mapped to a fixed number of SQL joins. It requires recursion.

We will translate this into a **Recursive CTE** (see `02-ctes.md`).

```rust
// Translation logic for variable length paths:
// 1. Base case: a -[:KNOWS]-> b (depth 1)
// 2. Recursive step: JOIN depth n-1 result with edges table to produce depth n
// 3. Union results up to max depth (3)
```

---

## Test Cases

### Test 1: Single Node Match
```cypher
MATCH (p:Person {name: 'Alice'})
RETURN p.age;
```
Translates to: `SELECT age FROM Person WHERE name = 'Alice'`

### Test 2: Simple Path Match
```cypher
MATCH (a:Person)-[r:KNOWS]->(b:Person)
WHERE a.age > 30
RETURN a.name, b.name;
```
Translates to Joins. Validates direction (`src_id` -> `dst_id`).

### Test 3: Undirected Path Match
```cypher
MATCH (a)-[r:KNOWS]-(b)
RETURN a.id, b.id;
```
Translates to a join where `(a.id = r.src_id AND b.id = r.dst_id) OR (a.id = r.dst_id AND b.id = r.src_id)`.

### Test 4: Variable Binding
```cypher
MATCH (a)-[r]->(b)
WHERE r.weight > 0.5
RETURN a.id;
```
Validates that relationship properties can be filtered in the `WHERE` clause.

---

## Performance Targets

- **Compilation Time**: Translating a complex MATCH to Joins should take < 1ms.
- **Execution**: The planner must use Hash Joins for path traversals over large sets.
- **Indexes**: The optimizer must recognize `(a:Person {name: 'Alice'})` and use an index on `name` before traversing edges.

## Metrics to Track

- `pieskieo_cypher_matches_executed`
- `pieskieo_cypher_translation_duration_ms`

**Created**: 2026-02-08
**Author**: Implementation Team
