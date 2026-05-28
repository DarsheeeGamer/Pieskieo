# Execution Engine

## Design
The execution engine receives a logical plan and transforms it into an optimized physical plan.
It must handle various types of storage structures, including relational tables, vector spaces, and graph topologies.

## Implementation details
- Iterator-based execution model (Volcano model).
- Cross-model integration points in memory structure for fast relational and graph joins.
