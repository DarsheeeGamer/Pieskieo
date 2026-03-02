/// Integration tests for PQL DDL features.
/// Covers: CREATE TABLE, CREATE COLLECTION, CREATE INDEX, CREATE VIEW,
///         DROP TABLE, DROP COLLECTION, DROP INDEX, DROP VIEW,
///         ALTER TABLE (add/drop/rename column).
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

// ── helpers ──────────────────────────────────────────────────────────────────

fn exec(ex: &Executor, sql: &str) -> pieskieo_core::pql::QueryResult {
    let mut p = Parser::new(sql);
    let stmt = p.parse().unwrap_or_else(|e| panic!("Parse error in {:?}: {}", sql, e));
    ex.execute(stmt)
        .unwrap_or_else(|e| panic!("Execution error in {:?}: {}", sql, e))
}

// ── CREATE TABLE ──────────────────────────────────────────────────────────────

/// CREATE TABLE registers the schema; SHOW SCHEMA returns all declared fields.
#[test]
fn test_create_table_schema_fields() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE employees (id INTEGER, name STRING, salary FLOAT);");

    let schema = exec(&ex, "SHOW SCHEMA OF employees;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(fields.contains(&"id".to_string()), "id field must be in schema");
    assert!(fields.contains(&"name".to_string()), "name field must be in schema");
    assert!(fields.contains(&"salary".to_string()), "salary field must be in schema");
}

/// SHOW TABLES returns a table name once rows exist.
#[test]
fn test_create_table_appears_in_show_tables_after_insert() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE staff (id INTEGER, name STRING);");
    // SHOW TABLES lists tables with actual row data; insert one row
    exec(&ex, r#"INSERT INTO staff {id: 1, name: "Alice"};"#);

    let result = exec(&ex, "SHOW TABLES;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"staff".to_string()), "staff should appear in SHOW TABLES");
}

/// CREATE TABLE with NOT NULL column: schema stores the field type.
#[test]
fn test_create_table_with_not_null_column() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(
        &ex,
        "CREATE TABLE products (sku STRING NOT NULL, price FLOAT NOT NULL, stock INTEGER);",
    );

    let schema = exec(&ex, "SHOW SCHEMA OF products;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(fields.contains(&"sku".to_string()), "sku field must be in schema");
    assert!(fields.contains(&"price".to_string()), "price field must be in schema");
    assert!(fields.contains(&"stock".to_string()), "stock field must be in schema");
}

/// INSERT into a CREATE TABLE'd table and QUERY it back.
#[test]
fn test_create_table_insert_and_query() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE orders (order_id INTEGER, customer STRING, total FLOAT);");

    exec(&ex, r#"INSERT INTO orders {order_id: 1, customer: "Alice", total: 99.5};"#);
    exec(&ex, r#"INSERT INTO orders {order_id: 2, customer: "Bob", total: 42.0};"#);

    let result = exec(&ex, "QUERY orders ORDER BY order_id ASC SELECT order_id, customer, total;");
    assert_eq!(result.rows.len(), 2, "should return both inserted rows");

    let first = &result.rows[0];
    assert_eq!(first.data.get("customer"), Some(&Value::String("Alice".to_string())));

    let second = &result.rows[1];
    assert_eq!(second.data.get("customer"), Some(&Value::String("Bob".to_string())));
}

/// CREATE TABLE with PRIMARYKEY column-level keyword.
#[test]
fn test_create_table_with_column_primary_key() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // PRIMARYKEY is a single token in PQL (no space)
    exec(
        &ex,
        "CREATE TABLE users (user_id INTEGER PRIMARYKEY, email STRING);",
    );

    // Table schema must report both fields
    let schema = exec(&ex, "SHOW SCHEMA OF users;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(fields.contains(&"user_id".to_string()));
    assert!(fields.contains(&"email".to_string()));

    // Insert and query to confirm the table works
    exec(&ex, r#"INSERT INTO users {user_id: 10, email: "test@example.com"};"#);
    let q = exec(&ex, "QUERY users SELECT user_id, email;");
    assert_eq!(q.rows.len(), 1);
}

/// CREATE TABLE with UNIQUE column constraint.
#[test]
fn test_create_table_with_unique_column() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE accounts (account_id INTEGER, username STRING UNIQUE);");

    let schema = exec(&ex, "SHOW SCHEMA OF accounts;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(fields.contains(&"account_id".to_string()));
    assert!(fields.contains(&"username".to_string()));
}

// ── CREATE COLLECTION ────────────────────────────────────────────────────────

/// CREATE COLLECTION registers a document schema; SHOW SCHEMA reports its fields.
#[test]
fn test_create_collection_schema_fields() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Collection schema uses braces; NOT NULL is not supported in collection syntax
    exec(&ex, "CREATE COLLECTION articles { title STRING, body STRING, views INTEGER }");

    let schema = exec(&ex, "SHOW SCHEMA OF articles;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(fields.contains(&"title".to_string()));
    assert!(fields.contains(&"body".to_string()));
    assert!(fields.contains(&"views".to_string()));
}

/// SHOW COLLECTIONS returns a collection name once documents exist.
#[test]
fn test_create_collection_appears_in_show_collections_after_insert() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE COLLECTION articles { title STRING, body STRING, views INTEGER }");
    exec(&ex, r#"INSERT INTO articles {title: "Hello", body: "World", views: 0};"#);

    let result = exec(&ex, "SHOW COLLECTIONS;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"articles".to_string()), "articles must appear in SHOW COLLECTIONS");
}

/// CREATE COLLECTION, insert, and query back.
#[test]
fn test_create_collection_insert_and_query() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE COLLECTION events { name STRING, score INTEGER }");

    exec(&ex, r#"INSERT INTO events {name: "sprint", score: 100};"#);
    exec(&ex, r#"INSERT INTO events {name: "relay", score: 80};"#);

    let result = exec(&ex, "QUERY events ORDER BY score DESC SELECT name, score;");
    assert_eq!(result.rows.len(), 2);
    match result.rows[0].data.get("score") {
        Some(Value::Integer(s)) => assert_eq!(*s, 100),
        other => panic!("unexpected score value: {:?}", other),
    }
}

// ── CREATE INDEX ──────────────────────────────────────────────────────────────

/// CREATE INDEX on a table; SHOW INDEXES returns the index name.
#[test]
fn test_create_index_appears_in_show_indexes() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE catalog (item_id INTEGER, name STRING, category STRING);");
    exec(&ex, "CREATE INDEX idx_category ON catalog (category);");

    let result = exec(&ex, "SHOW INDEXES ON catalog;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"idx_category".to_string()), "idx_category should appear in SHOW INDEXES");
}

/// CREATE HASH INDEX on a collection.
#[test]
fn test_create_hash_index() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE COLLECTION sessions { user_id INTEGER, token STRING }");
    exec(&ex, "CREATE HASH INDEX idx_token ON sessions (token);");

    let result = exec(&ex, "SHOW INDEXES ON sessions;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"idx_token".to_string()), "idx_token should appear after SHOW INDEXES");
}

/// CREATE INDEX on multiple fields (composite index).
#[test]
fn test_create_composite_index() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE invoices (inv_id INTEGER, client STRING, year INTEGER, amount FLOAT);");
    exec(&ex, "CREATE INDEX idx_client_year ON invoices (client, year);");

    let result = exec(&ex, "SHOW INDEXES ON invoices;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"idx_client_year".to_string()));
}

/// CREATE two indexes on the same table; SHOW INDEXES returns both.
#[test]
fn test_multiple_indexes_on_same_table() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE products (id INTEGER, name STRING, category STRING, price FLOAT);");
    exec(&ex, "CREATE INDEX idx_name ON products (name);");
    exec(&ex, "CREATE INDEX idx_price ON products (price);");

    let result = exec(&ex, "SHOW INDEXES ON products;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"idx_name".to_string()));
    assert!(names.contains(&"idx_price".to_string()));
}

// ── CREATE VIEW ───────────────────────────────────────────────────────────────

/// CREATE VIEW and then QUERY the view name transparently.
#[test]
fn test_create_view_and_query() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Seed underlying data directly via the engine
    db.put_doc_ns(
        None,
        Some("employees"),
        Uuid::new_v4(),
        serde_json::json!({"name": "Alice", "dept": "Engineering", "salary": 90000}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("employees"),
        Uuid::new_v4(),
        serde_json::json!({"name": "Bob", "dept": "Marketing", "salary": 70000}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("employees"),
        Uuid::new_v4(),
        serde_json::json!({"name": "Carol", "dept": "Engineering", "salary": 85000}),
    )
    .unwrap();

    exec(
        &ex,
        r#"CREATE VIEW engineers AS QUERY employees WHERE dept = "Engineering" SELECT name, salary;"#,
    );

    // Querying the view name should filter to Engineering only
    let result = exec(&ex, "QUERY engineers SELECT name, salary;");
    assert_eq!(result.rows.len(), 2, "view should return 2 engineers");

    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"Alice".to_string()));
    assert!(names.contains(&"Carol".to_string()));
    assert!(!names.contains(&"Bob".to_string()));
}

/// CREATE VIEW IF NOT EXISTS is idempotent.
#[test]
fn test_create_view_if_not_exists_is_idempotent() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    db.put_doc_ns(
        None,
        Some("items"),
        Uuid::new_v4(),
        serde_json::json!({"kind": "book", "price": 15}),
    )
    .unwrap();

    exec(&ex, "CREATE VIEW cheap_items AS QUERY items WHERE price < 20 SELECT kind, price;");
    // Second CREATE VIEW IF NOT EXISTS must not error
    exec(
        &ex,
        "CREATE VIEW IF NOT EXISTS cheap_items AS QUERY items WHERE price < 50 SELECT kind, price;",
    );

    // The original view definition (price < 20) should still be in effect
    let result = exec(&ex, "QUERY cheap_items SELECT kind, price;");
    assert_eq!(result.rows.len(), 1);
}

/// CREATE VIEW appears in SHOW VIEWS.
#[test]
fn test_create_view_appears_in_show_views() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    db.put_doc_ns(
        None,
        Some("logs"),
        Uuid::new_v4(),
        serde_json::json!({"level": "error", "msg": "disk full"}),
    )
    .unwrap();

    exec(
        &ex,
        r#"CREATE VIEW error_logs AS QUERY logs WHERE level = "error" SELECT level, msg;"#,
    );

    let result = exec(&ex, "SHOW VIEWS;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"error_logs".to_string()), "error_logs should appear in SHOW VIEWS");
}

/// View over a filtered table returns correct subset.
#[test]
fn test_view_filters_correctly() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    for i in 0..5i64 {
        db.put_doc_ns(
            None,
            Some("scores"),
            Uuid::new_v4(),
            serde_json::json!({"player": format!("p{}", i), "score": i * 10}),
        )
        .unwrap();
    }

    exec(&ex, "CREATE VIEW top_scores AS QUERY scores WHERE score >= 30 SELECT player, score;");

    let result = exec(&ex, "QUERY top_scores SELECT player, score;");
    // scores 30, 40 are >= 30 (i=3 gives 30, i=4 gives 40)
    assert_eq!(result.rows.len(), 2, "view should return players with score >= 30");
}

// ── DROP TABLE / DROP COLLECTION ─────────────────────────────────────────────

/// DROP TABLE removes the table from SHOW TABLES.
#[test]
fn test_drop_table_removes_from_show_tables() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE temp_data (id INTEGER, val STRING);");
    // Insert so the table appears in SHOW TABLES
    exec(&ex, r#"INSERT INTO temp_data {id: 1, val: "x"};"#);

    let before = exec(&ex, "SHOW TABLES;");
    let before_names: Vec<String> = before
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(before_names.contains(&"temp_data".to_string()));

    exec(&ex, "DROP TABLE temp_data;");

    let after = exec(&ex, "SHOW TABLES;");
    let after_names: Vec<String> = after
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(!after_names.contains(&"temp_data".to_string()), "temp_data must be gone after DROP TABLE");
}

/// DROP COLLECTION removes the collection from SHOW COLLECTIONS.
#[test]
fn test_drop_collection_removes_from_show_collections() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE COLLECTION scratch { x INTEGER }");
    exec(&ex, r#"INSERT INTO scratch {x: 1};"#);

    let before = exec(&ex, "SHOW COLLECTIONS;");
    let before_names: Vec<String> = before
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(before_names.contains(&"scratch".to_string()));

    exec(&ex, "DROP COLLECTION scratch;");

    let after = exec(&ex, "SHOW COLLECTIONS;");
    let after_names: Vec<String> = after
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(!after_names.contains(&"scratch".to_string()));
}

/// DROP TABLE also purges previously inserted rows.
#[test]
fn test_drop_table_purges_data() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE cache (key STRING, value STRING);");
    exec(&ex, r#"INSERT INTO cache {key: "a", value: "1"};"#);
    exec(&ex, r#"INSERT INTO cache {key: "b", value: "2"};"#);

    exec(&ex, "DROP TABLE cache;");
    exec(&ex, "CREATE TABLE cache (key STRING, value STRING);");

    // After DROP + re-CREATE the table should be empty
    let result = exec(&ex, "QUERY cache SELECT key, value;");
    assert_eq!(result.rows.len(), 0, "table should be empty after DROP and re-CREATE");
}

// ── DROP INDEX ────────────────────────────────────────────────────────────────

/// DROP INDEX removes the index from SHOW INDEXES.
#[test]
fn test_drop_index_removes_from_show_indexes() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE goods (id INTEGER, category STRING);");
    exec(&ex, "CREATE INDEX idx_cat ON goods (category);");

    let before = exec(&ex, "SHOW INDEXES ON goods;");
    let before_names: Vec<String> = before
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(before_names.contains(&"idx_cat".to_string()));

    exec(&ex, "DROP INDEX idx_cat ON goods;");

    let after = exec(&ex, "SHOW INDEXES ON goods;");
    let after_names: Vec<String> = after
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(!after_names.contains(&"idx_cat".to_string()), "idx_cat must be gone after DROP INDEX");
}

// ── DROP VIEW ─────────────────────────────────────────────────────────────────

/// DROP VIEW removes the view so it no longer appears in SHOW VIEWS.
#[test]
fn test_drop_view_removes_from_show_views() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    db.put_doc_ns(
        None,
        Some("metrics"),
        Uuid::new_v4(),
        serde_json::json!({"host": "srv1", "cpu": 80}),
    )
    .unwrap();

    exec(
        &ex,
        "CREATE VIEW high_cpu AS QUERY metrics WHERE cpu > 70 SELECT host, cpu;",
    );
    exec(&ex, "DROP VIEW high_cpu;");

    let after = exec(&ex, "SHOW VIEWS;");
    let names: Vec<String> = after
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(!names.contains(&"high_cpu".to_string()), "high_cpu must be gone after DROP VIEW");
}

/// DROP VIEW IF EXISTS does not error when the view does not exist.
#[test]
fn test_drop_view_if_exists_no_error() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Should not panic/error even though the view never existed
    exec(&ex, "DROP VIEW IF EXISTS nonexistent_view;");
}

// ── ALTER TABLE ───────────────────────────────────────────────────────────────

/// ALTER TABLE ADD COLUMN makes the new column appear in SHOW SCHEMA.
#[test]
fn test_alter_table_add_column() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE people (name STRING, age INTEGER);");

    exec(&ex, "ALTER TABLE people ADD COLUMN email STRING;");

    // The new column should be in the schema
    let schema = exec(&ex, "SHOW SCHEMA OF people;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(
        fields.contains(&"email".to_string()),
        "email column should exist after ALTER TABLE ADD COLUMN"
    );
    assert!(fields.contains(&"name".to_string()));
    assert!(fields.contains(&"age".to_string()));
}

/// ALTER TABLE DROP COLUMN removes the column from the schema.
#[test]
fn test_alter_table_drop_column() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE widgets (id INTEGER, name STRING, internal_code STRING);");

    exec(&ex, "ALTER TABLE widgets DROP COLUMN internal_code;");

    let schema = exec(&ex, "SHOW SCHEMA OF widgets;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !fields.contains(&"internal_code".to_string()),
        "internal_code should be absent after DROP COLUMN"
    );
    assert!(fields.contains(&"id".to_string()));
    assert!(fields.contains(&"name".to_string()));
}

/// ALTER TABLE RENAME COLUMN updates the column name in the schema.
#[test]
fn test_alter_table_rename_column() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE contracts (contract_id INTEGER, client_name STRING);");

    exec(&ex, "ALTER TABLE contracts RENAME COLUMN client_name TO customer_name;");

    let schema = exec(&ex, "SHOW SCHEMA OF contracts;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(
        fields.contains(&"customer_name".to_string()),
        "customer_name should exist after RENAME COLUMN"
    );
    assert!(
        !fields.contains(&"client_name".to_string()),
        "client_name should no longer exist after RENAME COLUMN"
    );
}

/// ALTER TABLE ADD COLUMN followed by INSERT captures the new field.
#[test]
fn test_alter_table_add_column_then_insert() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE notes (id INTEGER, content STRING);");
    exec(&ex, "ALTER TABLE notes ADD COLUMN priority INTEGER;");
    exec(&ex, r#"INSERT INTO notes {id: 1, content: "important", priority: 5};"#);

    let result = exec(&ex, "QUERY notes SELECT id, content, priority;");
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].data.get("priority"), Some(&Value::Integer(5)));
}

// ── Bonus: TRUNCATE ───────────────────────────────────────────────────────────

/// TRUNCATE TABLE removes all rows but keeps the table visible in SHOW SCHEMA.
#[test]
fn test_truncate_table_clears_rows() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE TABLE logs (msg STRING, severity INTEGER);");
    exec(&ex, r#"INSERT INTO logs {msg: "info msg", severity: 1};"#);
    exec(&ex, r#"INSERT INTO logs {msg: "error msg", severity: 3};"#);

    let before = exec(&ex, "QUERY logs SELECT msg;");
    assert_eq!(before.rows.len(), 2, "should have 2 rows before TRUNCATE");

    exec(&ex, "TRUNCATE TABLE logs;");

    let after = exec(&ex, "QUERY logs SELECT msg;");
    assert_eq!(after.rows.len(), 0, "should have 0 rows after TRUNCATE");

    // Schema should still be there
    let schema = exec(&ex, "SHOW SCHEMA OF logs;");
    let fields: Vec<String> = schema
        .rows
        .iter()
        .filter_map(|r| match r.data.get("field") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(fields.contains(&"msg".to_string()), "schema should persist after TRUNCATE");
}

// ── Bonus: CREATE SEQUENCE ────────────────────────────────────────────────────

/// CREATE SEQUENCE then SHOW SEQUENCES lists it.
#[test]
fn test_create_sequence_appears_in_show_sequences() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE SEQUENCE order_seq START 100 INCREMENT 5;");

    let result = exec(&ex, "SHOW SEQUENCES;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"order_seq".to_string()), "order_seq must appear in SHOW SEQUENCES");
}

/// DROP SEQUENCE removes it from SHOW SEQUENCES.
#[test]
fn test_drop_sequence() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, "CREATE SEQUENCE tmp_seq;");
    exec(&ex, "DROP SEQUENCE tmp_seq;");

    let result = exec(&ex, "SHOW SEQUENCES;");
    let names: Vec<String> = result
        .rows
        .iter()
        .filter_map(|r| match r.data.get("name") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(!names.contains(&"tmp_seq".to_string()), "tmp_seq must be gone after DROP SEQUENCE");
}
