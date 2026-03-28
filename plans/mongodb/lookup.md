# MongoDB Feature: $lookup Aggregation Stage

**Status**: 🔴 Not Started
**Priority**: CRITICAL
**Dependencies**: Relational Joins (PostgreSQL compatibility), JSON/JSONB Types
**Estimated Effort**: 3-4 weeks

---

## Overview

The `$lookup` stage performs a left outer join to an unsharded collection in the same database to filter in documents from the "joined" collection for processing. The stage adds a new array field to each input document. The new array field contains the matching documents from the "joined" collection.

## Supported Syntaxes

MongoDB supports two main syntaxes for `$lookup`:

### 1. Equality Match (Single Field)
```json
{
   $lookup: {
     from: <collection to join>,
     localField: <field from the input documents>,
     foreignField: <field from the documents of the "from" collection>,
     as: <output array field>
   }
}
```

### 2. Join Conditions and Uncorrelated Sub-queries (Pipeline)
```json
{
   $lookup: {
     from: <joined collection>,
     let: { <var_1>: <expression>, ..., <var_n>: <expression> },
     pipeline: [ <pipeline to run on joined collection> ],
     as: <output array field>
   }
}
```

---

## Implementation Plan

### Phase 1: AST and Configuration Representation

We need to map the `$lookup` syntax to our unified AST. Since Pieskieo's underlying engine is SQL-based (supporting `LEFT JOIN LATERAL`), we will translate `$lookup` into a `LATERAL` left join that aggregates results into an array.

```rust
// crates/pieskieo-core/src/pql/ast.rs

#[derive(Clone, Debug, PartialEq)]
pub enum AggregationStage {
    Match(Expr),
    Project(Expr),
    // ...
    Lookup(LookupSpec),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LookupSpec {
    pub from: String,
    pub as_field: String,
    pub condition: LookupCondition,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LookupCondition {
    // Basic syntax
    Equality {
        local_field: String,
        foreign_field: String,
    },
    // Advanced syntax with 'let' and 'pipeline'
    Pipeline {
        let_vars: HashMap<String, Expr>,
        pipeline: Vec<AggregationStage>,
    },
}
```

### Phase 2: Translation to SQL AST (Relational Mapping)

A `$lookup` stage is essentially:
1.  A `LEFT JOIN` against the `from` collection.
2.  If it's an equality match, the condition is `local_field = foreign_field`.
3.  If it's a pipeline match, the condition evaluates the `let` variables and executes the sub-pipeline (a `LATERAL` join).
4.  The output is aggregated into a JSON array (`jsonb_agg` or similar) under the `as` field.

```rust
// crates/pieskieo-core/src/engine/mongo_translate.rs

impl MongoTranslator {
    pub fn translate_lookup(&self, spec: &LookupSpec, input_query: SelectStatement) -> Result<SelectStatement> {
        // Assume 'input_query' represents the pipeline up to this point.
        // Let's call the input table/subquery 'local_docs'

        // 1. Determine the join condition
        let join_cond = match &spec.condition {
            LookupCondition::Equality { local_field, foreign_field } => {
                // local_docs.local_field = from_coll.foreign_field
                Expr::BinaryOp {
                    left: Box::new(Expr::Identifier(format!("local_docs.{}", local_field))),
                    op: BinaryOperator::Eq,
                    right: Box::new(Expr::Identifier(format!("{}.{}", spec.from, foreign_field))),
                }
            }
            LookupCondition::Pipeline { let_vars, pipeline } => {
                // Pipeline lookup requires a LATERAL join.
                // We'll translate the sub-pipeline into a subquery, injecting the 'let' vars into its context.
                // For simplicity here, assume `build_subquery` handles the translation.
                self.build_lateral_subquery(spec.from.clone(), let_vars, pipeline)?
            }
        };

        // 2. Build the LATERAL LEFT JOIN
        let joined_table = if let LookupCondition::Pipeline { .. } = spec.condition {
            // LATERAL subquery as table source
            TableSource::LateralSubquery {
                query: Box::new(join_cond), // join_cond is the translated subquery here
                alias: spec.from.clone(),
            }
        } else {
            // Standard table
            TableSource::Table {
                name: spec.from.clone(),
                alias: Some(spec.from.clone()),
            }
        };

        let join = TableSource::Join {
            left: Box::new(TableSource::Subquery {
                query: Box::new(input_query.clone()),
                alias: "local_docs".into(),
            }),
            right: Box::new(joined_table),
            join_type: JoinType::Left,
            on: match &spec.condition {
                 LookupCondition::Equality { .. } => Some(join_cond),
                 _ => None, // LATERAL subqueries don't need ON clauses if they correlate internally
            },
        };

        // 3. Aggregate the matched rows into a JSON array
        // SELECT local_docs.*, jsonb_agg(from_coll.*) as as_field
        // GROUP BY local_docs.* (simplified for concept)

        let mut projections = vec![Projection::Wildcard(Some("local_docs".into()))];
        projections.push(Projection::Function {
            name: "jsonb_agg".into(),
            args: vec![Expr::Identifier(format!("{}.*", spec.from))],
            alias: Some(spec.as_field.clone()),
        });

        Ok(SelectStatement {
            projections,
            from: vec![join],
            group_by: vec![Expr::Identifier("local_docs.id".into())], // Assuming _id is primary key
            ..Default::default()
        })
    }
}
```

### Phase 3: Array Field Matching (The MongoDB Quirk)

If the `localField` is an array, `$lookup` behaves differently than a standard SQL join. It matches any element in the `localField` array with the `foreignField`.

```rust
// In translation for Equality:
// if typeof(local_docs.local_field) == array
// then from_coll.foreign_field IN (local_docs.local_field)
```

PostgreSQL uses `<@` or `@>` for array containment. We need our optimizer to recognize this pattern and rewrite the condition efficiently (e.g., using a GIN index on `foreign_field`).

### Phase 4: Correlated Sub-pipelines (LATERAL Execution)

For pipeline lookups:

```json
{
  $lookup: {
    from: "orders",
    let: { customer_id: "$_id" },
    pipeline: [
      { $match: { $expr: { $eq: ["$cust_id", "$$customer_id"] } } },
      { $limit: 5 }
    ],
    as: "recent_orders"
  }
}
```

This translates closely to our PostgreSQL Correlated Subquery plan (`01-subqueries.md`). The execution engine binds the `let` variables to the current row context before executing the `pipeline` (subquery) for that row.

---

## Test Cases

### Test 1: Basic Equality
```javascript
// Setup
db.users.insert({ _id: 1, name: "Alice" });
db.orders.insert([{ _id: 101, user_id: 1, total: 50 }, { _id: 102, user_id: 1, total: 150 }]);

// Query
db.users.aggregate([
  {
    $lookup: {
      from: "orders",
      localField: "_id",
      foreignField: "user_id",
      as: "user_orders"
    }
  }
]);

// Expected output for Alice
// { _id: 1, name: "Alice", user_orders: [{ _id: 101, user_id: 1, total: 50 }, { _id: 102, user_id: 1, total: 150 }] }
```

### Test 2: Local Field is an Array
```javascript
// Setup
db.classes.insert({ _id: 1, title: "Math", students: [10, 20] });
db.students.insert([{ _id: 10, name: "Bob" }, { _id: 20, name: "Charlie" }]);

// Query
db.classes.aggregate([
  {
    $lookup: {
      from: "students",
      localField: "students",
      foreignField: "_id",
      as: "enrolled_students"
    }
  }
]);

// Expected
// { _id: 1, title: "Math", students: [10, 20], enrolled_students: [{_id:10, name:"Bob"}, {_id:20, name:"Charlie"}] }
```

### Test 3: Uncorrelated Pipeline
```javascript
db.users.aggregate([
  {
    $lookup: {
      from: "orders",
      pipeline: [ { $match: { total: { $gt: 100 } } } ], // No 'let', just joins all large orders to everyone
      as: "large_orders"
    }
  }
]);
```

### Test 4: Correlated Pipeline with Let
```javascript
// Get top 2 orders per user
db.users.aggregate([
  {
    $lookup: {
      from: "orders",
      let: { uid: "$_id" },
      pipeline: [
        { $match: { $expr: { $eq: ["$user_id", "$$uid"] } } },
        { $sort: { total: -1 } },
        { $limit: 2 }
      ],
      as: "top_orders"
    }
  }
]);
```

---

## Performance Targets

- **Basic Equality**: Should be planned as a Hash Left Join or Index Scan if indexes are present. Performance identical to SQL `LEFT JOIN`.
- **Array Matches**: Must leverage GIN indexes if available.
- **Correlated Pipelines**: Should attempt decorrelation (see `01-subqueries.md`) if possible, otherwise execute as a fast nested loop.

## Metrics to Track

- `pieskieo_lookup_stages_executed`
- `pieskieo_lookup_decorrelated`
- `pieskieo_lookup_execution_duration_ms`

**Created**: 2026-02-08
**Author**: Implementation Team
