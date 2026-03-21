/// Integration tests for PQL JSON/JSONB path query and type conversion functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

// ── JSON_EXISTS ───────────────────────────────────────────────────────────────

#[test]
fn test_json_exists_present() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"profile": {"city": "London"}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("profile", profile) COMPUTE ok = JSON_EXISTS(base, "profile") SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_json_exists_missing() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": 1}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a) COMPUTE missing = JSON_EXISTS(base, "b") SELECT missing;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("missing"), Some(&Value::Bool(false)));
}

// ── JSON_VALUE ────────────────────────────────────────────────────────────────

#[test]
fn test_json_value_scalar() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"score": 42}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("score", score) COMPUTE v = JSON_VALUE(base, "score") SELECT v;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(42)));
}

#[test]
fn test_json_value_missing_returns_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"x": 1}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x) COMPUTE v = JSON_VALUE(base, "z") SELECT v;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Null));
}

// ── JSONB_SET ─────────────────────────────────────────────────────────────────

#[test]
fn test_jsonb_set_inserts_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": 1}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a) COMPUTE updated = JSONB_SET(base, "b", 99) COMPUTE sz = JSON_SIZE(updated) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_jsonb_set_updates_existing_key() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"val": 10}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("val", val) COMPUTE updated = JSONB_SET(base, "val", 999) COMPUTE v = JSON_VALUE(updated, "val") SELECT v;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(999)));
}

// ── JSONB_INSERT ──────────────────────────────────────────────────────────────

#[test]
fn test_jsonb_insert_adds_key() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"x": 5}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x) COMPUTE inserted = JSONB_INSERT(base, "y", 10) COMPUTE sz = JSON_SIZE(inserted) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

// ── JSONB_DELETE ──────────────────────────────────────────────────────────────

#[test]
fn test_jsonb_delete_removes_key() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE trimmed = JSONB_DELETE(base, "b") COMPUTE sz = JSON_SIZE(trimmed) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}

// ── JSONB_STRIP_NULLS ─────────────────────────────────────────────────────────

#[test]
fn test_jsonb_strip_nulls_removes_null_values() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"k": "hello"}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("k", k, "n", null) COMPUTE stripped = JSONB_STRIP_NULLS(base) COMPUTE sz = JSON_SIZE(stripped) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}

// ── JSONB_TYPEOF ──────────────────────────────────────────────────────────────

#[test]
fn test_jsonb_typeof_number() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 42}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE t = JSONB_TYPEOF(n) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("t"),
        Some(&Value::String("number".to_string()))
    );
}

#[test]
fn test_jsonb_typeof_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "hello"}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE t = JSONB_TYPEOF(s) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("t"),
        Some(&Value::String("string".to_string()))
    );
}

#[test]
fn test_jsonb_typeof_array() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3]}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE t = JSONB_TYPEOF(arr) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("t"),
        Some(&Value::String("array".to_string()))
    );
}

#[test]
fn test_jsonb_typeof_object() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"obj": {"k": "v"}}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE t = JSONB_TYPEOF(obj) SELECT t;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("t"),
        Some(&Value::String("object".to_string()))
    );
}

// ── JSONB_ARRAY_LENGTH ────────────────────────────────────────────────────────

#[test]
fn test_jsonb_array_length() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": [10, 20, 30, 40]}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE len = JSONB_ARRAY_LENGTH(items) SELECT len;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("len"), Some(&Value::Integer(4)));
}

// ── JSON_EACH / JSONB_EACH ────────────────────────────────────────────────────

#[test]
fn test_json_each_returns_array_of_kv_objects() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE pairs = JSON_EACH(base) COMPUTE sz = ARRAY_COUNT(pairs) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

#[test]
fn test_jsonb_each_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 10, "y": 20, "z": 30}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x, "y", y, "z", z) COMPUTE pairs = JSONB_EACH(base) COMPUTE cnt = ARRAY_COUNT(pairs) SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

// ── JSON_EACH_TEXT / JSONB_EACH_TEXT ──────────────────────────────────────────

#[test]
fn test_json_each_text_values_are_strings() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("n", n) COMPUTE pairs = JSON_EACH_TEXT(base) COMPUTE cnt = ARRAY_COUNT(pairs) SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(1)));
}

// ── JSON_ARRAY_ELEMENTS / JSONB_ARRAY_ELEMENTS ────────────────────────────────

#[test]
fn test_json_array_elements_returns_array() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vals": [7, 8, 9]}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE elems = JSON_ARRAY_ELEMENTS(vals) COMPUTE cnt = ARRAY_COUNT(elems) SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

#[test]
fn test_jsonb_array_elements_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nums": [1, 2]}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE elems = JSONB_ARRAY_ELEMENTS(nums) COMPUTE cnt = ARRAY_COUNT(elems) SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(2)));
}

// ── JSON_ARRAY_ELEMENTS_TEXT ──────────────────────────────────────────────────

#[test]
fn test_json_array_elements_text_coerces_to_strings() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nums": [1, 2, 3]}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE strs = JSON_ARRAY_ELEMENTS_TEXT(nums) COMPUTE cnt = ARRAY_COUNT(strs) SELECT cnt;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

// ── ROW_TO_JSON ───────────────────────────────────────────────────────────────

#[test]
fn test_row_to_json_converts_arg_to_value() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE result = ROW_TO_JSON(base) COMPUTE sz = JSON_SIZE(result) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}

// ── JSON_CONTAINS / JSONB_CONTAINS ────────────────────────────────────────────

#[test]
fn test_json_contains_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE container = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE contained = JSON_BUILD_OBJECT("a", a) COMPUTE ok = JSON_CONTAINS(container, contained) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_json_contains_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": 1}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE container = JSON_BUILD_OBJECT("a", a) COMPUTE contained = JSON_BUILD_OBJECT("b", 999) COMPUTE ok = JSON_CONTAINS(container, contained) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_jsonb_contains_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"x": 10, "y": 20}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE container = JSON_BUILD_OBJECT("x", x, "y", y) COMPUTE contained = JSON_BUILD_OBJECT("x", x) COMPUTE ok = JSONB_CONTAINS(container, contained) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── JSON_OVERLAPS / JSONB_OVERLAPS ────────────────────────────────────────────

#[test]
fn test_json_overlaps_shared_keys() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE obj1 = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE obj2 = JSON_BUILD_OBJECT("b", b) COMPUTE overlap = JSON_OVERLAPS(obj1, obj2) SELECT overlap;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("overlap"), Some(&Value::Bool(true)));
}

#[test]
fn test_json_overlaps_no_shared_keys() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": 1}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE obj1 = JSON_BUILD_OBJECT("a", a) COMPUTE obj2 = JSON_BUILD_OBJECT("z", 99) COMPUTE overlap = JSON_OVERLAPS(obj1, obj2) SELECT overlap;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("overlap"), Some(&Value::Bool(false)));
}

// ── JSON_PATH_QUERY_FIRST ─────────────────────────────────────────────────────

#[test]
fn test_json_path_query_first_returns_value() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"meta": {"version": 3}}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("meta", meta) COMPUTE v = JSON_PATH_QUERY_FIRST(base, "meta") SELECT v;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Should return the object at meta key
    assert!(r.rows[0].data.get("v").is_some());
}

// ── JSON_VALID / IS_JSON ──────────────────────────────────────────────────────

#[test]
fn test_json_valid_object() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": 1}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a) COMPUTE ok = JSON_VALID(base) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_json_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 42}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_JSON(n) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── JSON_LENGTH ───────────────────────────────────────────────────────────────

#[test]
fn test_json_length_object() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 1, "b": 2, "c": 3}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b, "c", c) COMPUTE len = JSON_LENGTH(base) SELECT len;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("len"), Some(&Value::Integer(3)));
}

#[test]
fn test_json_length_array() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"items": [1, 2, 3, 4, 5]}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE len = JSON_LENGTH(items) SELECT len;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("len"), Some(&Value::Integer(5)));
}

#[test]
fn test_json_length_scalar() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 99}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE len = JSON_LENGTH(n) SELECT len;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("len"), Some(&Value::Integer(1)));
}

// ── JSONB_PRETTY ─────────────────────────────────────────────────────────────

#[test]
fn test_jsonb_pretty_returns_string() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"key": "value"}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("key", key) COMPUTE pretty = JSONB_PRETTY(base) SELECT pretty;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pretty") {
        Some(Value::String(s)) => {
            // Should contain the key name
            assert!(
                s.contains("key"),
                "Expected 'key' in pretty output, got: {}",
                s
            );
        }
        other => panic!("Expected String from JSONB_PRETTY, got {:?}", other),
    }
}

// ── JSON_NORMALIZE ────────────────────────────────────────────────────────────

#[test]
fn test_json_normalize_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"x": 7}))
        .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("x", x) COMPUTE norm = JSON_NORMALIZE(base) COMPUTE sz = JSON_SIZE(norm) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(1)));
}

// ── JSON_POPULATE_RECORD / JSON_TO_RECORD ─────────────────────────────────────

#[test]
fn test_json_populate_record_passthrough() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": 10, "b": 20}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE base = JSON_BUILD_OBJECT("a", a, "b", b) COMPUTE rec = JSON_POPULATE_RECORD(base) COMPUTE sz = JSON_SIZE(rec) SELECT sz;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sz"), Some(&Value::Integer(2)));
}
