# Parser Architecture

## Design
The parser architecture is responsible for taking the raw query string and generating a unified abstract syntax tree (AST).
It must be able to support generic SQL operations like SELECT, JOIN, as well as specific extensions for graph matching (e.g. MATCH clauses) and vector similarity searches.

## Implementation details
- Uses the `sqlparser` crate.
- We will add extensions for specific functions or custom operators (e.g. `<->` for vector distance).
