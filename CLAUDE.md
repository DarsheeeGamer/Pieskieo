# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## MISSION

**Pieskieo is the LAST database anyone will ever need to install.**

We are building a **single, unified, production-grade database** that completely replaces:
- **PostgreSQL** (relational + SQL)
- **MongoDB** (documents + aggregations)
- **Weaviate** (vector search + hybrid)
- **LanceDB** (columnar analytics)
- **Kùzu/Neo4j** (graph + Cypher)

With **ZERO network hops**, **ONE query language**, and **FULL feature parity** with all five.

---

## CRITICAL RULES — READ BEFORE TOUCHING ANY CODE

### 1. NO COMPROMISES EVER

**FORBIDDEN phrases and approaches:**
- "Initial version" / "Later version" / "For now" / "Initially"
- "Known limitations (can be addressed later)"
- "Simple algorithm first, then improve"
- "Single-node first, distributed later"
- "MVP approach" / "Can be added in follow-up"
- `TODO:` comments in production code
- Placeholder implementations
- Partial feature implementations

**Required mindset:**
- Production-ready from commit 1
- Best-in-class algorithms from day 1
- All optimizations included upfront
- Complete implementations only

### 2. COMPLETE FEATURE PARITY MANDATE

We are building **100% feature parity** with ALL of:
- **PostgreSQL**: ALL SQL features, indexes, transactions, partitioning, full-text, JSON
- **MongoDB**: ALL aggregation stages, update operators, change streams, indexes
- **Weaviate**: ALL hybrid search, multi-vector, quantization, reranking
- **LanceDB**: ALL columnar features, time-travel, Arrow, predicate pushdown
- **Kùzu/Neo4j**: ALL Cypher features, graph algorithms, WCOJ, CSR storage

Not "most commonly used features" — **EVERYTHING**.

### 3. PLANNING BEFORE IMPLEMENTATION

All 157 features must be planned in `plans/` before code is written:
- Each plan: 4000–6000 tokens of detail
- Full Rust implementation specs
- All optimizations specified
- All edge cases handled
- Complete test scenarios

Before writing ANY code:
1. Check if a feature plan exists in `plans/`
2. If not, **create the plan first**
3. Then implement exactly as planned

### 4. ALGORITHM SELECTION STANDARDS

Always use state-of-the-art algorithms:

| Component | Don't use | Use instead |
|-----------|-----------|-------------|
| Spatial Index | Basic R-tree | R*-tree with bulk loading |
| Graph Joins | Hash joins | Worst-Case Optimal Joins (WCOJ) |
| Vector Search | Flat index | HNSW + IVF-PQ hybrid |
| Deadlock Detection | Timeout-based | Wait-for graph with cycle detection |
| Query Optimizer | Rule-based | Cost-based with cardinality estimation |
| Storage | Row-only | Hybrid (columnar + row + vector + graph) |
| Compression | Single algorithm | Adaptive (LZ4/Zstd based on data) |
| Lock Manager | Coarse locks | Fine-grained + lock-free where possible |

### 5. PRODUCTION-GRADE REQUIREMENTS

Every feature must include from day 1:
- SIMD/vectorization on hot paths (`std::arch`)
- Lock-free data structures where possible (`crossbeam`, atomics)
- Memory pooling and zero-copy
- Adaptive compression
- Full ACID guarantees — no eventual consistency shortcuts
- WAL with fsync guarantees + crash recovery tested
- Prometheus metrics for all operations
- Structured logging (`tracing`)
- Query explain plans with cost estimates

### 6. TESTING STANDARDS

- **Unit tests**: 90%+ coverage minimum
- **Integration tests**: cross-component interactions
- **Stress tests**: 10k+ concurrent operations
- **Chaos tests**: network failures, crashes, disk failures
- **Benchmarks**: must meet or beat competitor databases

Performance targets (p99):
| Operation | Target |
|-----------|--------|
| Point query | < 1ms |
| Range scan (1000 rows) | < 10ms |
| Vector search (top 10) | < 5ms |
| Graph traversal (3 hops) | < 20ms |
| Complex JOIN (3 tables) | < 50ms |
| Aggregation (1M rows) | < 100ms |

---

## Commands

```bash
# Build all crates
cargo build --release

# Build specific crates
cargo build -p pieskieo-core --release
cargo build -p pieskieo-server --release
cargo build -p pieskieo-server --release --features tls

# Run all tests
cargo test -p pieskieo-core

# Run a single test by name
cargo test -p pieskieo-core -- <test_name>

# Lint / format (required before commits)
cargo fmt
cargo clippy --all-targets --all-features

# Run the server
PIESKIEO_DATA=./data PIESKIEO_LISTEN=0.0.0.0:8000 cargo run -p pieskieo-server --release

# Run the CLI (network-only REPL)
cargo run -p pieskieo-cli -- --connect pieskieo@localhost --port 8000 -W

# Run benchmarks
cargo run -p pieskieo-core --bin bench --release -- <n> <dim> [ef_c] [ef_s]
cargo run -p pieskieo-server --bin load --release -- <url> <dim> <n_vec> <searches>
```

---

## Workspace Structure

Three crates in `crates/`:
- **`pieskieo-core`**: Storage engine library — collections, indexes, WAL, HNSW vectors, graph, PQL stack. No external pieskieo dependencies.
- **`pieskieo-server`**: Axum HTTP API server with auth, rate limiting, sharding, audit logging.
- **`pieskieo-cli`**: Network-only psql-style REPL client.

---

## Architecture

### Core Engine (`pieskieo-core/src/`)

- **`engine.rs`**: `PieskieoDb` struct — row/doc storage, all secondary index types (equality, BTree, hash, fulltext, BM25), schema definitions, multi-shard coordination. Cost-based planner (`RUST_LOG=planner=debug`).
- **`vector.rs`**: HNSW ANN index. L2/Cosine/Dot metrics, metadata filtering, snapshot persistence. Tuned via `PIESKIEO_EF_CONSTRUCTION`, `PIESKIEO_EF_SEARCH`, `PIESKIEO_LINK_K`.
- **`graph.rs`**: Edge store with adjacency lists (in/out), typed edges, BFS/DFS, PageRank, centrality, community detection.
- **`wal.rs`**: Write-Ahead Log — bincode serialization, fsync durability, replay.

### PQL Stack (`pieskieo-core/src/pql/`)

Pipeline: `lexer.rs` → `parser.rs` → `ast.rs` → `executor/`

- **`lexer.rs`**: Tokenizes PQL text.
- **`parser.rs`**: Recursive-descent parser building AST.
- **`ast.rs`**: All AST types (Statement, Operation, Expression, Condition).
- **`executor/mod.rs`**: `Executor` struct + `execute()` dispatch. Clean entry point, no business logic.
- **`executor/types.rs`**: `Value`, `Row`, `QueryResult`, `ExecutionStats`, `GraphMetricsCache`.
- **`executor/expressions.rs`**: All expression/condition evaluation, function execution, graph metric functions.
- **`executor/source.rs`**: Source loading, JSON↔Value conversion utilities.
- **`executor/query.rs`**: `execute_query`, `execute_operation` dispatch.
- **`executor/operations.rs`**: GROUP BY, COMPUTE, ORDER BY, SELECT.
- **`executor/vector.rs`**: VECTOR SEARCH, HYBRID SEARCH.
- **`executor/graph.rs`**: TRAVERSE, PATH, MATCH with DFS helpers.
- **`executor/joins.rs`**: All JOIN types (Inner, Left, Right, Full, Cross).
- **`executor/dml.rs`**: INSERT, UPDATE, DELETE.
- **`executor/ddl.rs`**: CREATE, ALTER TABLE, DROP INDEX, DROP COLLECTION.
- **`executor/explain.rs`**: EXPLAIN ANALYZE with cost estimates.
- **`integration_tests.rs`**: End-to-end PQL tests.

### Server (`pieskieo-server/src/lib.rs`)

Axum routes for health, CRUD, vector, graph, metrics, WAL replication. Argon2id auth (Basic + Bearer), per-IP rate limiting, audit logging, transparent intra-process sharding.

---

## Key Data Structures

**`Value`** (`executor/types.rs`): `Null | Bool | Integer(i64) | Float(f64) | String | Uuid | Vector(Vec<f32>) | Array | Object`

**`Collections`** (`engine.rs`): Nested HashMaps: namespace → table/collection → id → JSON value. Parallel index maps per index type.

**`VectorIndex`** (`vector.rs`): `Arc<RwLock<HashMap<Uuid, Vec<f32>>>>` + optional HNSW graph + UUID↔index bidirectional maps + metadata.

**`GraphStore`** (`graph.rs`): `Arc<RwLock<HashMap<Uuid, Vec<Edge>>>>` for `adj_out` and `adj_in`.

**`GraphMetricsCache`** (`executor/types.rs`): `HashMap<String, HashMap<Uuid, f64>>` keyed by algorithm+params string for deduplication within a query.

---

## Visibility Model (executor modules)

- Private `fn` — only called within same file
- `pub(crate)` — called by sibling modules within `executor/`
- `pub(super)` — called only from `executor/mod.rs` dispatcher
- `pub` — top-level public API

---

## Code Style

- **Rust Edition**: 2021, minimum 1.92.0
- **Concurrency**: `parking_lot::{RwLock, Mutex}` over `std::sync`; `Arc` for shared state
- **Async**: `tokio` throughout; `#[tokio::test]` for async tests
- **Errors**: `thiserror` for custom types; `?` for propagation; no `.unwrap()` outside tests
- **Imports**: crate-local first, then external crates, then `std`

---

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|---|---|---|
| `PIESKIEO_DATA` | `./data` | Data directory |
| `PIESKIEO_LISTEN` | `0.0.0.0:8000` | Bind address |
| `PIESKIEO_SHARD_TOTAL` | `1` | Number of shards |
| `PIESKIEO_TOKEN` | — | Bearer token for admin auth |
| `PIESKIEO_USERS` | — | JSON array of `{user, pass, role}` |
| `PIESKIEO_AUTH_USER` / `PIESKIEO_AUTH_PASSWORD` | — | Single admin credentials |
| `PIESKIEO_TLS_CERT` / `PIESKIEO_TLS_KEY` | — | PEM paths for TLS (`--features tls`) |
| `PIESKIEO_EF_CONSTRUCTION` | `200` | HNSW build quality |
| `PIESKIEO_EF_SEARCH` | `50` | HNSW search beam width |
| `PIESKIEO_LINK_K` | — | Auto-mesh KNN links per vector insert |
| `PIESKIEO_RATE_MAX` | `300` | Max requests per rate window |
| `PIESKIEO_RATE_WINDOW_SECS` | `60` | Rate limit window |
| `PIESKIEO_AUDIT_MAX_MB` | — | Audit log rotation size |
| `RUST_LOG` | — | `planner=debug` traces index selection |
