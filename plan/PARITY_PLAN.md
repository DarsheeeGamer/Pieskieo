# Pieskieo Parity Roadmap (Postgres / MongoDB / Weaviate / LanceDB / Kùzu)

Status legend: ☐ not started · ◐ in progress · ☐✚ planned after dependency · ☑ done/landed

## 1) PostgreSQL‑like
- ☐ MVCC with snapshot/serializable; savepoints; prepared/extended protocol.
- ☐ Constraints: FK/unique/check; sequences/serial; ALTER TABLE migrations.
- ☐ SQL surface: CTEs, window functions, subqueries everywhere, scalar/builtin funcs.
- ☐ Indexes: GIN/GiST/BRIN style, partial indexes; ANALYZE stats + cost-based planner V2 with column stats.
- ☐ WAL streaming + PITR; logical replication slots; follower reads.
- ◐ Cost-based planner V1 (basic) — extend for projections/joins/filters with stats.

## 2) MongoDB‑like
- ☐ Aggregation pipeline ($match/$project/$group/$unwind/$lookup/$facet/$bucket).
- ☐ Update operators on nested arrays ($set/$unset/$inc/$push/$pull/$addToSet, arrayFilters, positional).
- ☐ Change streams/CDC; capped collections; TTL indexes; collation.
- ☐ Transactions across shards; schema validation.

## 3) Weaviate‑like (Vector/Hybrid)
- ☐ Hybrid vector+BM25 search with score fusion; rerankers.
- ☐ Per-class HNSW tuning; background rebuild; multi-tenant isolation.
- ☐ Backups/import-export; replication factor & consistency levels.
- ☐ Object references with filters; vector filters on references.

## 4) LanceDB‑like (columnar + vector)
- ☐ Columnar storage with append/compaction; snapshot/time-travel.
- ☐ Predicate pushdown & column pruning in vector/tabular mixed queries.
- ☐ Concurrent writers semantics; embedded/in-process mode.

## 5) Kùzu‑like (property graph)
- ☐ Cypher-like MATCH/OPTIONAL MATCH, patterns, var-length paths.
- ☐ Node/edge typed schemas, property constraints, property indexes.
- ☐ Graph algos: shortest path variants, centrality, community detection.
- ☐ Bulk LOAD CSV; graph+vector combined execution.

## 6) Cross-cutting
- ◐ Auth/RBAC: default admin present; TODO: OIDC/LDAP, password policy, lockout tunables.
- ☑ Logging: default log dir /PieskieoLogs (or C:\\PieskieoLogs); override PIESKIEO_LOG_DIR.
- ☐ Metrics/tracing: Prometheus/exporters; slow query log.
- ☐ Backup/restore & PITR; online reshard verification; follower reads.
- ☐ CLI/SDK: multiline REPL polish; completion/help; Python SDK full parity tests.

## 7) Immediate fixes (blocking)
- ◐ Projection support: broaden SELECT projection parsing to allow auth probe queries without “projection item not supported”.
- ☐ Auth bootstrap: ensure default admin applies even when auth_users.json exists but empty; add CLI auth check integration test.

## Milestones (proposed)
- M1: Fix projection error; harden auth bootstrap; tag v2.0.2 binaries.
- M2: Cost-based planner V2 with stats + projection/aggregation coverage; SQL joins/filters correctness tests.
- M3: Mongo aggregation & update operators; change streams.
- M4: Hybrid search + HNSW tunables + backups/import-export.
- M5: Graph Cypher subset + property constraints + shortest paths.

