# GEMINI.md

This file provides foundational mandates and expert procedural guidance for Gemini CLI when working on the **Pieskieo** codebase.

---

## 🚀 Mission
**Pieskieo is the LAST database anyone will ever need to install.**
A unified, production-grade database replacing PostgreSQL (relational), MongoDB (document), Weaviate (vector), LanceDB (columnar), and Neo4j (graph) with a single binary, zero network hops, and a unified query language (PQL).

---

## 📜 Core Mandates (MUST READ)

### 1. Absolute Feature Parity
We aim for **100% feature parity** with the target databases. No "partial implementations" or "MVP-only" features. Every optimization, edge case, and niche feature must be included.

### 2. Planning First
**NEVER** implement a feature without a corresponding plan in `plans/`.
- Plans must be 4000–6000 tokens of detail.
- If a plan is missing for a requested feature, **create the plan first**.
- Follow the plan exactly during implementation.

### 3. Production-Ready from Day 1
- **Forbidden:** "Initial version", "later version", "TODO", "placeholder", "simple algorithm first".
- **Required:** SIMD/vectorization, lock-free structures, memory pooling, zero-copy, adaptive compression, FULL ACID, WAL with fsync, Prometheus metrics, structured logging.

### 4. Technical Excellence
Always use state-of-the-art algorithms:
- **Vector Search:** HNSW + IVF-PQ hybrid.
- **Graph Joins:** Worst-Case Optimal Joins (WCOJ).
- **Deadlock Detection:** Wait-for graph with cycle detection.
- **Optimizer:** Cost-based with cardinality estimation.

---

## 🛠 Project Structure

- **`crates/pieskieo-core`**: The heart of the engine. Storage, WAL, HNSW, Graph, PQL stack (lexer, parser, AST, executor).
- **`crates/pieskieo-server`**: Axum HTTP API. Handles auth (Argon2id), rate limiting, sharding, and sharded fan-out.
- **`crates/pieskieo-cli`**: Psql-style REPL client for network-only interaction.
- **`plans/`**: Comprehensive specifications for every feature, categorized by target database (PG, Mongo, etc.).

---

## 💻 Development Workflow

### Building & Running
```bash
# Build all
cargo build --release

# Run Server (Development)
PIESKIEO_DATA=./data PIESKIEO_LISTEN=0.0.0.0:8000 cargo run -p pieskieo-server --release

# Run Server (with TLS)
cargo run -p pieskieo-server --release --features tls

# Run CLI
cargo run -p pieskieo-cli -- --connect pieskieo@localhost --port 8000 -W
```

### Testing & Validation
- **Unit/Integration Tests:** `cargo test -p pieskieo-core`
- **Linting:** `cargo clippy --all-targets --all-features`
- **Formatting:** `cargo fmt`
- **Coverage:** 90%+ coverage is the minimum standard.
- **Benchmarks:**
  - Core: `cargo run -p pieskieo-core --bin bench --release -- <n> <dim> [ef_c] [ef_s]`
  - HTTP Load: `cargo run -p pieskieo-server --bin load --release -- <url> <dim> <n_vec> <searches>`

---

## 🔧 Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PIESKIEO_DATA` | `./data` | Data directory |
| `PIESKIEO_LISTEN` | `0.0.0.0:8000` | Bind address |
| `PIESKIEO_SHARD_TOTAL` | `1` | Number of shards |
| `PIESKIEO_USERS` | — | JSON array of `{user, pass, role}` |
| `PIESKIEO_TLS_CERT` / `PIESKIEO_TLS_KEY` | — | Paths for TLS (requires `--features tls`) |
| `PIESKIEO_EF_SEARCH` | `50` | HNSW search beam width |
| `PIESKIEO_LINK_K` | — | Auto-mesh KNN links per vector insert |

---

## 🎨 Coding Standards
- **Rust Edition:** 2021 (Min v1.92.0).
- **Concurrency:** `parking_lot` for locks, `Arc` for shared state.
- **Async:** `tokio` everywhere.
- **Error Handling:** `thiserror` for library errors, `anyhow` for applications. **NO UNWRAPS.**
- **Logging:** `tracing` for structured logs. `RUST_LOG=planner=debug` for query plan tracing.
- **Style:** Imports should be crate-local, then external, then `std`.
