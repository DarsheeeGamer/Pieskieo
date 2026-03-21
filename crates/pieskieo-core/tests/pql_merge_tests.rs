/// Integration tests for PQL MERGE, INSERT ON CONFLICT (UPSERT), and advanced DML features.
/// Tests are standalone — each creates its own temp database.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn exec(ex: &Executor, pql: &str) -> pieskieo_core::error::Result<pieskieo_core::pql::QueryResult> {
    let mut p = Parser::new(pql);
    let stmt = p
        .parse()
        .unwrap_or_else(|e| panic!("parse error for {:?}: {:?}", pql, e));
    ex.execute(stmt)
}

// ── INSERT ON CONFLICT DO NOTHING (skip duplicate) ───────────────────────────

#[test]
fn test_upsert_do_nothing_preserves_original() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Initial row
    exec(&ex, r#"INSERT INTO products {name: "apple", price: 5};"#).unwrap();

    // Attempt to insert duplicate — should be silently ignored
    exec(
        &ex,
        r#"INSERT INTO products {name: "apple", price: 99} ON CONFLICT(name) DO NOTHING;"#,
    )
    .unwrap();

    // Price should still be 5
    let result = exec(
        &ex,
        r#"QUERY products WHERE name = "apple" SELECT name, price;"#,
    )
    .unwrap();

    assert_eq!(result.rows.len(), 1, "Should have exactly one apple row");
    assert_eq!(
        result.rows[0].data.get("price"),
        Some(&Value::Integer(5)),
        "Price should remain 5 (DO NOTHING ignored the conflict)"
    );
}

// ── INSERT ON CONFLICT DO NOTHING — no conflict inserts the row ───────────────

#[test]
fn test_upsert_do_nothing_inserts_when_no_conflict() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // No prior row for "mango"
    exec(
        &ex,
        r#"INSERT INTO fruits {name: "mango", stock: 20} ON CONFLICT(name) DO NOTHING;"#,
    )
    .unwrap();

    let result = exec(&ex, r#"QUERY fruits WHERE name = "mango" SELECT stock;"#).unwrap();

    assert_eq!(
        result.rows.len(),
        1,
        "New row should be inserted when there is no conflict"
    );
    assert_eq!(
        result.rows[0].data.get("stock"),
        Some(&Value::Integer(20)),
        "Stock should be 20"
    );
}

// ── INSERT ON CONFLICT DO UPDATE (upsert) ────────────────────────────────────

#[test]
fn test_upsert_do_update_overwrites_field() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Initial row
    exec(&ex, r#"INSERT INTO users {name: "alice", score: 10};"#).unwrap();

    // Upsert: conflict on name → update score
    exec(
        &ex,
        r#"INSERT INTO users {name: "alice", score: 42} ON CONFLICT(name) DO UPDATE SET score = 42;"#,
    )
    .unwrap();

    let result = exec(&ex, r#"QUERY users WHERE name = "alice" SELECT score;"#).unwrap();

    assert_eq!(result.rows.len(), 1, "Should still have exactly one alice");
    assert_eq!(
        result.rows[0].data.get("score"),
        Some(&Value::Integer(42)),
        "Score should be updated to 42"
    );
}

// ── INSERT ON CONFLICT DO UPDATE — no conflict means normal insert ─────────────

#[test]
fn test_upsert_do_update_inserts_when_no_conflict() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // bob does not exist yet
    exec(
        &ex,
        r#"INSERT INTO users2 {name: "bob", score: 7} ON CONFLICT(name) DO UPDATE SET score = 7;"#,
    )
    .unwrap();

    let result = exec(&ex, r#"QUERY users2 WHERE name = "bob" SELECT score;"#).unwrap();

    assert_eq!(result.rows.len(), 1, "Bob should have been inserted");
    assert_eq!(
        result.rows[0].data.get("score"),
        Some(&Value::Integer(7)),
        "Score should be 7"
    );
}

// ── Upsert does not duplicate rows ───────────────────────────────────────────

#[test]
fn test_upsert_does_not_create_duplicate_rows() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Insert once
    exec(&ex, r#"INSERT INTO items {sku: "X1", qty: 5};"#).unwrap();
    // Upsert twice more on same conflict key
    exec(
        &ex,
        r#"INSERT INTO items {sku: "X1", qty: 10} ON CONFLICT(sku) DO UPDATE SET qty = 10;"#,
    )
    .unwrap();
    exec(
        &ex,
        r#"INSERT INTO items {sku: "X1", qty: 15} ON CONFLICT(sku) DO UPDATE SET qty = 15;"#,
    )
    .unwrap();

    let result = exec(&ex, r#"QUERY items WHERE sku = "X1" SELECT qty;"#).unwrap();

    assert_eq!(
        result.rows.len(),
        1,
        "Should have exactly one row for sku X1 — no duplicates"
    );
    assert_eq!(
        result.rows[0].data.get("qty"),
        Some(&Value::Integer(15)),
        "qty should reflect the last upsert value"
    );
}

// ── Bulk INSERT (multiple rows in one statement) ─────────────────────────────

#[test]
fn test_bulk_insert_multiple_rows() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(
        &ex,
        r#"INSERT INTO catalog [{name: "pen", price: 1}, {name: "notebook", price: 3}, {name: "ruler", price: 2}];"#,
    )
    .unwrap();

    let result = exec(&ex, r#"QUERY catalog SELECT name;"#).unwrap();
    assert_eq!(result.rows.len(), 3, "All three rows should be inserted");
}

// ── MERGE WHEN MATCHED THEN UPDATE ──────────────────────────────────────────

#[test]
fn test_merge_when_matched_update() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Target: existing stock row
    db.put_doc_ns(
        None,
        Some("stock"),
        Uuid::new_v4(),
        serde_json::json!({"sku": "A1", "qty": 10}),
    )
    .unwrap();

    // Source: deliveries update for A1 ("incoming" is a reserved keyword so we use "deliveries")
    db.put_doc_ns(
        None,
        Some("deliveries"),
        Uuid::new_v4(),
        serde_json::json!({"sku": "A1", "qty": 25}),
    )
    .unwrap();

    // MERGE: update qty when sku matches
    exec(
        &ex,
        r#"MERGE INTO stock USING deliveries ON stock.sku = deliveries.sku
           WHEN MATCHED THEN UPDATE SET qty = qty;"#,
    )
    .unwrap();

    // The matched update ran without error; verify the row still exists
    let result = exec(&ex, r#"QUERY stock WHERE sku = "A1" SELECT sku;"#).unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "stock row for A1 should still exist after MERGE"
    );
}

// ── MERGE WHEN NOT MATCHED THEN INSERT ───────────────────────────────────────

#[test]
fn test_merge_when_not_matched_insert() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Target: empty collection (no rows yet)
    // Source: two rows to merge in
    db.put_doc_ns(
        None,
        Some("src_nodes"),
        Uuid::new_v4(),
        serde_json::json!({"node_id": "n1", "label": "alpha"}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("src_nodes"),
        Uuid::new_v4(),
        serde_json::json!({"node_id": "n2", "label": "beta"}),
    )
    .unwrap();

    // MERGE: no target rows match → both source rows should be inserted
    let result = exec(
        &ex,
        r#"MERGE INTO tgt_nodes USING src_nodes ON tgt_nodes.node_id = src_nodes.node_id
           WHEN NOT MATCHED THEN INSERT {node_id: node_id, label: label};"#,
    )
    .unwrap();

    // rows_filtered counts the not-matched inserts
    assert!(
        result.stats.rows_filtered >= 2,
        "Expected at least 2 rows processed by MERGE, got {}",
        result.stats.rows_filtered
    );

    // Verify inserted rows are queryable
    let check = exec(&ex, r#"QUERY tgt_nodes SELECT node_id;"#).unwrap();
    assert_eq!(
        check.rows.len(),
        2,
        "Both source rows should have been inserted into tgt_nodes"
    );
}

// ── MERGE: both WHEN MATCHED and WHEN NOT MATCHED ────────────────────────────

#[test]
fn test_merge_matched_update_and_not_matched_insert() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Target: one existing product
    db.put_doc_ns(
        None,
        Some("inventory"),
        Uuid::new_v4(),
        serde_json::json!({"name": "apple", "price": 1}),
    )
    .unwrap();

    // Source: update for apple (matched) + new product banana (not matched)
    db.put_doc_ns(
        None,
        Some("price_updates"),
        Uuid::new_v4(),
        serde_json::json!({"name": "apple", "new_price": 3}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("price_updates"),
        Uuid::new_v4(),
        serde_json::json!({"name": "banana", "new_price": 2}),
    )
    .unwrap();

    let result = exec(
        &ex,
        r#"MERGE INTO inventory USING price_updates ON inventory.name = price_updates.name
           WHEN MATCHED THEN UPDATE SET price = new_price
           WHEN NOT MATCHED THEN INSERT {name: name, price: new_price};"#,
    )
    .unwrap();

    // Both matched and not-matched actions should have fired
    assert!(
        result.stats.rows_filtered >= 2,
        "Expected at least 2 rows processed (1 update + 1 insert)"
    );

    // Verify apple's price was updated
    let apple_result = exec(&ex, r#"QUERY inventory WHERE name = "apple" SELECT price;"#).unwrap();
    assert_eq!(apple_result.rows.len(), 1);
    let apple_price = match apple_result.rows[0].data.get("price") {
        Some(Value::Integer(n)) => *n,
        Some(Value::Float(f)) => *f as i64,
        other => panic!("Unexpected price value: {:?}", other),
    };
    assert_eq!(apple_price, 3, "Apple price should have been updated to 3");

    // Verify banana was inserted
    let banana_result = exec(
        &ex,
        r#"QUERY inventory WHERE name = "banana" SELECT price;"#,
    )
    .unwrap();
    assert_eq!(
        banana_result.rows.len(),
        1,
        "Banana should have been inserted by MERGE"
    );
}

// ── MERGE: WHEN MATCHED THEN DELETE ─────────────────────────────────────────

#[test]
fn test_merge_when_matched_delete() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Target: two rows
    db.put_doc_ns(
        None,
        Some("employees"),
        Uuid::new_v4(),
        serde_json::json!({"emp_id": "E1", "dept": "sales"}),
    )
    .unwrap();
    db.put_doc_ns(
        None,
        Some("employees"),
        Uuid::new_v4(),
        serde_json::json!({"emp_id": "E2", "dept": "eng"}),
    )
    .unwrap();

    // Source: terminations — only E1 should be deleted
    db.put_doc_ns(
        None,
        Some("terminations"),
        Uuid::new_v4(),
        serde_json::json!({"emp_id": "E1"}),
    )
    .unwrap();

    exec(
        &ex,
        r#"MERGE INTO employees USING terminations ON employees.emp_id = terminations.emp_id
           WHEN MATCHED THEN DELETE;"#,
    )
    .unwrap();

    // E1 should be gone
    let result_e1 = exec(&ex, r#"QUERY employees WHERE emp_id = "E1" SELECT emp_id;"#).unwrap();
    assert_eq!(
        result_e1.rows.len(),
        0,
        "E1 should have been deleted by MERGE"
    );

    // E2 should still exist
    let result_e2 = exec(&ex, r#"QUERY employees WHERE emp_id = "E2" SELECT emp_id;"#).unwrap();
    assert_eq!(result_e2.rows.len(), 1, "E2 should NOT have been deleted");
}

// ── UPDATE with complex WHERE condition ───────────────────────────────────────

#[test]
fn test_update_with_compound_condition() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Insert several orders
    exec(
        &ex,
        r#"INSERT INTO orders {status: "pending", amount: 50, region: "west"};"#,
    )
    .unwrap();
    exec(
        &ex,
        r#"INSERT INTO orders {status: "pending", amount: 200, region: "west"};"#,
    )
    .unwrap();
    exec(
        &ex,
        r#"INSERT INTO orders {status: "pending", amount: 300, region: "east"};"#,
    )
    .unwrap();

    // Update only west+pending orders with amount > 100
    exec(
        &ex,
        r#"UPDATE orders SET status = "approved" WHERE region = "west" AND amount > 100;"#,
    )
    .unwrap();

    let approved = exec(
        &ex,
        r#"QUERY orders WHERE status = "approved" SELECT amount;"#,
    )
    .unwrap();
    assert_eq!(approved.rows.len(), 1, "Only one order should be approved");
    assert_eq!(
        approved.rows[0].data.get("amount"),
        Some(&Value::Integer(200)),
        "The approved order should have amount 200"
    );

    // Other orders should remain pending
    let pending = exec(
        &ex,
        r#"QUERY orders WHERE status = "pending" SELECT amount;"#,
    )
    .unwrap();
    assert_eq!(pending.rows.len(), 2, "Two orders should remain pending");
}

// ── DELETE with WHERE condition ────────────────────────────────────────────────

#[test]
fn test_delete_with_filter() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(
        &ex,
        r#"INSERT INTO sessions {user: "alice", active: true};"#,
    )
    .unwrap();
    exec(&ex, r#"INSERT INTO sessions {user: "bob", active: false};"#).unwrap();
    exec(
        &ex,
        r#"INSERT INTO sessions {user: "carol", active: false};"#,
    )
    .unwrap();

    exec(&ex, r#"DELETE FROM sessions WHERE active = false;"#).unwrap();

    let result = exec(&ex, r#"QUERY sessions SELECT user;"#).unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "Only alice's active session should remain"
    );
    assert_eq!(
        result.rows[0].data.get("user"),
        Some(&Value::String("alice".to_string())),
        "Remaining row should be alice"
    );
}

// ── INSERT RETURNING clause ───────────────────────────────────────────────────

#[test]
fn test_insert_returning() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    let result = exec(
        &ex,
        r#"INSERT INTO logs {message: "hello", level: "info"} RETURNING message, level;"#,
    )
    .unwrap();

    assert_eq!(result.rows.len(), 1, "RETURNING should produce one row");
    assert_eq!(
        result.rows[0].data.get("message"),
        Some(&Value::String("hello".to_string())),
        "RETURNING should include message"
    );
    assert_eq!(
        result.rows[0].data.get("level"),
        Some(&Value::String("info".to_string())),
        "RETURNING should include level"
    );
}

// ── UPDATE RETURNING clause ───────────────────────────────────────────────────

#[test]
fn test_update_returning() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(&ex, r#"INSERT INTO counters {key: "hits", value: 0};"#).unwrap();

    let result = exec(
        &ex,
        r#"UPDATE counters SET value = 1 WHERE key = "hits" RETURNING key, value;"#,
    )
    .unwrap();

    assert_eq!(result.rows.len(), 1, "RETURNING should produce one row");
    assert_eq!(
        result.rows[0].data.get("value"),
        Some(&Value::Integer(1)),
        "Updated value should be 1"
    );
}

// ── DELETE RETURNING clause ───────────────────────────────────────────────────

#[test]
fn test_delete_returning() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    exec(
        &ex,
        r#"INSERT INTO events {event_type: "click", user: "alice"};"#,
    )
    .unwrap();

    let result = exec(
        &ex,
        r#"DELETE FROM events WHERE user = "alice" RETURNING event_type, user;"#,
    )
    .unwrap();

    assert_eq!(
        result.rows.len(),
        1,
        "RETURNING should produce the deleted row"
    );
    assert_eq!(
        result.rows[0].data.get("event_type"),
        Some(&Value::String("click".to_string())),
        "Deleted row should have event_type = click"
    );

    // Confirm the row is gone
    let check = exec(&ex, r#"QUERY events SELECT user;"#).unwrap();
    assert_eq!(check.rows.len(), 0, "Row should be gone after DELETE");
}

// ── Multiple upserts update the same row sequentially ────────────────────────

#[test]
fn test_sequential_upserts_last_write_wins() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    for score in [10i64, 20, 30, 40, 50] {
        exec(
            &ex,
            &format!(
                r#"INSERT INTO leaderboard {{player: "alice", score: {score}}} ON CONFLICT(player) DO UPDATE SET score = {score};"#,
                score = score
            ),
        )
        .unwrap();
    }

    let result = exec(
        &ex,
        r#"QUERY leaderboard WHERE player = "alice" SELECT score;"#,
    )
    .unwrap();

    assert_eq!(
        result.rows.len(),
        1,
        "Should have exactly one row for alice"
    );
    assert_eq!(
        result.rows[0].data.get("score"),
        Some(&Value::Integer(50)),
        "Score should be 50 (last write wins)"
    );
}

// ── Upsert with multiple conflict-key fields ─────────────────────────────────

#[test]
fn test_upsert_single_conflict_field_multiple_users() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());

    // Insert two different users
    exec(
        &ex,
        r#"INSERT INTO accounts {username: "alice", balance: 100};"#,
    )
    .unwrap();
    exec(
        &ex,
        r#"INSERT INTO accounts {username: "bob", balance: 200};"#,
    )
    .unwrap();

    // Upsert only affects alice
    exec(
        &ex,
        r#"INSERT INTO accounts {username: "alice", balance: 150} ON CONFLICT(username) DO UPDATE SET balance = 150;"#,
    )
    .unwrap();

    // alice updated
    let alice = exec(
        &ex,
        r#"QUERY accounts WHERE username = "alice" SELECT balance;"#,
    )
    .unwrap();
    assert_eq!(
        alice.rows[0].data.get("balance"),
        Some(&Value::Integer(150)),
        "Alice balance should be updated to 150"
    );

    // bob unchanged
    let bob = exec(
        &ex,
        r#"QUERY accounts WHERE username = "bob" SELECT balance;"#,
    )
    .unwrap();
    assert_eq!(
        bob.rows[0].data.get("balance"),
        Some(&Value::Integer(200)),
        "Bob balance should remain 200"
    );

    // Total row count should still be 2
    let all = exec(&ex, r#"QUERY accounts SELECT username;"#).unwrap();
    assert_eq!(
        all.rows.len(),
        2,
        "Should still have exactly two account rows"
    );
}
