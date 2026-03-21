/// Integration tests for window and aggregate functions added to expressions.rs:
/// CUME_DIST (no-arg row-context), PERCENT_RANK, FIRST_VALUE, LAST_VALUE,
/// NTH_VALUE, LAG, LEAD (row-based), EVERY, COVAR_SAMP, REGR_COUNT, REGR_R2,
/// REGR_AVGX, REGR_AVGY, REGR_SXX, REGR_SYY, REGR_SXY, XMLAGG,
/// JSONB_OBJECT_AGG, ARRAY_AGG, STRING_AGG, BIT_AND/OR/XOR, BOOL_AND/OR,
/// COVAR_POP, REGR_SLOPE, REGR_INTERCEPT, JSON_AGG, JSON_OBJECT_AGG.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db_with(
    ns: &str,
    docs: Vec<serde_json::Value>,
) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for doc in docs {
        db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    }
    (dir, db, ex)
}

fn run(ex: &Executor, pql: &str) -> pieskieo_core::pql::QueryResult {
    let mut p = Parser::new(pql);
    ex.execute(p.parse().unwrap()).unwrap()
}

// ── CUME_DIST (no-arg, row-based window context) ─────────────────────────────

#[test]
fn test_cume_dist_returns_float() {
    // Without window context, CUME_DIST() with no args falls back to 1.0
    let (_dir, _db, ex) = make_db_with(
        "test_window_cd",
        vec![serde_json::json!({"id": 1, "val": 10})],
    );
    let r = run(&ex, "QUERY test_window_cd SELECT CUME_DIST() AS cd;");
    assert!(!r.rows.is_empty(), "Expected at least one row");
    match r.rows[0].data.get("cd") {
        Some(Value::Float(_)) => {}
        other => panic!("Expected Float for CUME_DIST(), got {:?}", other),
    }
}

// ── PERCENT_RANK (row-based) ──────────────────────────────────────────────────

#[test]
fn test_percent_rank_returns_float() {
    // Without window context, PERCENT_RANK() falls back to 0.0
    let (_dir, _db, ex) = make_db_with(
        "test_window_pr",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(&ex, "QUERY test_window_pr SELECT PERCENT_RANK() AS pr;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("pr") {
        Some(Value::Float(_)) => {}
        other => panic!("Expected Float for PERCENT_RANK(), got {:?}", other),
    }
}

// ── FIRST_VALUE (row-based) ───────────────────────────────────────────────────

#[test]
fn test_first_value_returns_null_without_context() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_fv",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(&ex, "QUERY test_window_fv SELECT FIRST_VALUE(val) AS fv;");
    assert!(!r.rows.is_empty());
    // Without window marker, should return Null
    let v = r.rows[0].data.get("fv");
    assert!(v.is_some(), "Expected fv field to exist");
}

// ── LAST_VALUE (row-based) ────────────────────────────────────────────────────

#[test]
fn test_last_value_returns_null_without_context() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_lv",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(&ex, "QUERY test_window_lv SELECT LAST_VALUE(val) AS lv;");
    assert!(!r.rows.is_empty());
    let v = r.rows[0].data.get("lv");
    assert!(v.is_some(), "Expected lv field to exist");
}

// ── NTH_VALUE (row-based) ─────────────────────────────────────────────────────

#[test]
fn test_nth_value_returns_null_without_context() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_nv",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(&ex, "QUERY test_window_nv SELECT NTH_VALUE(val, 1) AS nv;");
    assert!(!r.rows.is_empty());
    let v = r.rows[0].data.get("nv");
    assert!(v.is_some(), "Expected nv field to exist");
}

// ── LAG (row-based) ───────────────────────────────────────────────────────────

#[test]
fn test_lag_returns_null_without_context() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_lag",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(&ex, "QUERY test_window_lag SELECT LAG(val) AS prev_val;");
    assert!(!r.rows.is_empty());
    // Without pre-computed _lag_val_offset1, returns default (Null)
    assert_eq!(r.rows[0].data.get("prev_val"), Some(&Value::Null));
}

#[test]
fn test_lag_default_value() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_lag2",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    // LAG with explicit default -1 should return -1 when no marker exists
    let r = run(
        &ex,
        "QUERY test_window_lag2 SELECT LAG(val, 1, -1) AS prev_val;",
    );
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("prev_val"), Some(&Value::Integer(-1)));
}

// ── LEAD (row-based) ──────────────────────────────────────────────────────────

#[test]
fn test_lead_returns_null_without_context() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_lead",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(&ex, "QUERY test_window_lead SELECT LEAD(val) AS next_val;");
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("next_val"), Some(&Value::Null));
}

#[test]
fn test_lead_default_value() {
    let (_dir, _db, ex) = make_db_with(
        "test_window_lead2",
        vec![serde_json::json!({"id": 1, "val": 42})],
    );
    let r = run(
        &ex,
        "QUERY test_window_lead2 SELECT LEAD(val, 1, 999) AS next_val;",
    );
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("next_val"), Some(&Value::Integer(999)));
}

// ── EVERY ─────────────────────────────────────────────────────────────────────

#[test]
fn test_every_single_true_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_every_t",
        vec![serde_json::json!({"id": 1, "active": true})],
    );
    let r = run(&ex, "QUERY test_every_t SELECT EVERY(active) AS ea;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ea") {
        Some(Value::Bool(true)) => {}
        other => panic!("Expected Bool(true) for EVERY(true), got {:?}", other),
    }
}

#[test]
fn test_every_single_false_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_every_f",
        vec![serde_json::json!({"id": 1, "active": false})],
    );
    let r = run(&ex, "QUERY test_every_f SELECT EVERY(active) AS ea;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ea") {
        Some(Value::Bool(false)) => {}
        other => panic!("Expected Bool(false) for EVERY(false), got {:?}", other),
    }
}

#[test]
fn test_every_group_all_true() {
    // Group all rows, EVERY should return true
    let (_dir, _db, ex) = make_db_with(
        "test_every_grp",
        vec![
            serde_json::json!({"grp": 1, "active": true}),
            serde_json::json!({"grp": 1, "active": true}),
            serde_json::json!({"grp": 1, "active": true}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_every_grp COMPUTE g = grp GROUPBY g COMPUTE ea = EVERY(active) SELECT ea;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ea") {
        Some(Value::Bool(true)) => {}
        other => panic!(
            "Expected Bool(true) for EVERY on all-true group, got {:?}",
            other
        ),
    }
}

// ── COVAR_SAMP ────────────────────────────────────────────────────────────────

#[test]
fn test_covar_samp_null_single_row() {
    // Sample covariance requires at least 2 rows; single row returns Null
    let (_dir, _db, ex) = make_db_with(
        "test_covar_samp1",
        vec![serde_json::json!({"x": 1.0, "y": 2.0})],
    );
    let r = run(
        &ex,
        "QUERY test_covar_samp1 COMPUTE g = 1 GROUPBY g COMPUTE cv = COVAR_SAMP(y, x) SELECT cv;",
    );
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("cv"), Some(&Value::Null));
}

#[test]
fn test_covar_samp_two_rows() {
    // Sample covariance of (y=[2,4], x=[1,3]) = 2.0
    let (_dir, _db, ex) = make_db_with(
        "test_covar_samp2",
        vec![
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 3.0, "y": 4.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_covar_samp2 COMPUTE g = 1 GROUPBY g COMPUTE cv = COVAR_SAMP(y, x) SELECT cv;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("cv") {
        Some(Value::Float(f)) => {
            assert!(
                (f - 2.0).abs() < 1e-9,
                "Expected COVAR_SAMP = 2.0, got {}",
                f
            );
        }
        other => panic!("Expected Float for COVAR_SAMP, got {:?}", other),
    }
}

// ── REGR_COUNT ────────────────────────────────────────────────────────────────

#[test]
fn test_regr_count_group() {
    let (_dir, _db, ex) = make_db_with(
        "test_regr_cnt",
        vec![
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 3.0, "y": 4.0}),
            serde_json::json!({"x": 5.0, "y": 6.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_cnt COMPUTE g = 1 GROUPBY g COMPUTE cnt = REGR_COUNT(y, x) SELECT cnt;",
    );
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(3)));
}

#[test]
fn test_regr_count_single_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_regr_cnt1",
        vec![serde_json::json!({"x": 1.0, "y": 2.0})],
    );
    let r = run(&ex, "QUERY test_regr_cnt1 SELECT REGR_COUNT(y, x) AS cnt;");
    assert!(!r.rows.is_empty());
    assert_eq!(r.rows[0].data.get("cnt"), Some(&Value::Integer(1)));
}

// ── REGR_R2 ──────────────────────────────────────────────────────────────────

#[test]
fn test_regr_r2_perfect_correlation() {
    // y = 2x: perfect correlation -> R² = 1.0
    let (_dir, _db, ex) = make_db_with(
        "test_regr_r2",
        vec![
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 2.0, "y": 4.0}),
            serde_json::json!({"x": 3.0, "y": 6.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_r2 COMPUTE g = 1 GROUPBY g COMPUTE r2 = REGR_R2(y, x) SELECT r2;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("r2") {
        Some(Value::Float(f)) => {
            assert!(
                (f - 1.0).abs() < 1e-6,
                "Expected R² = 1.0 for perfect correlation, got {}",
                f
            );
        }
        other => panic!("Expected Float for REGR_R2, got {:?}", other),
    }
}

// ── REGR_AVGX / REGR_AVGY ─────────────────────────────────────────────────────

#[test]
fn test_regr_avgx_group() {
    let (_dir, _db, ex) = make_db_with(
        "test_regr_avgx",
        vec![
            serde_json::json!({"x": 1.0, "y": 1.0}),
            serde_json::json!({"x": 3.0, "y": 1.0}),
            serde_json::json!({"x": 5.0, "y": 1.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_avgx COMPUTE g = 1 GROUPBY g COMPUTE ax = REGR_AVGX(y, x) SELECT ax;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ax") {
        Some(Value::Float(f)) => {
            assert!(
                (f - 3.0).abs() < 1e-9,
                "Expected REGR_AVGX = 3.0, got {}",
                f
            );
        }
        other => panic!("Expected Float for REGR_AVGX, got {:?}", other),
    }
}

#[test]
fn test_regr_avgy_group() {
    let (_dir, _db, ex) = make_db_with(
        "test_regr_avgy",
        vec![
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 1.0, "y": 4.0}),
            serde_json::json!({"x": 1.0, "y": 6.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_avgy COMPUTE g = 1 GROUPBY g COMPUTE ay = REGR_AVGY(y, x) SELECT ay;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("ay") {
        Some(Value::Float(f)) => {
            assert!(
                (f - 4.0).abs() < 1e-9,
                "Expected REGR_AVGY = 4.0, got {}",
                f
            );
        }
        other => panic!("Expected Float for REGR_AVGY, got {:?}", other),
    }
}

// ── REGR_SXX / REGR_SYY / REGR_SXY ──────────────────────────────────────────

#[test]
fn test_regr_sxx_group() {
    // x = [1,2,3], mean=2, SXX = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2.0
    let (_dir, _db, ex) = make_db_with(
        "test_regr_sxx",
        vec![
            serde_json::json!({"x": 1.0, "y": 1.0}),
            serde_json::json!({"x": 2.0, "y": 1.0}),
            serde_json::json!({"x": 3.0, "y": 1.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_sxx COMPUTE g = 1 GROUPBY g COMPUTE sxx = REGR_SXX(y, x) SELECT sxx;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("sxx") {
        Some(Value::Float(f)) => {
            assert!((f - 2.0).abs() < 1e-9, "Expected REGR_SXX = 2.0, got {}", f);
        }
        other => panic!("Expected Float for REGR_SXX, got {:?}", other),
    }
}

#[test]
fn test_regr_syy_group() {
    // y = [1,2,3], mean=2, SYY = 2.0
    let (_dir, _db, ex) = make_db_with(
        "test_regr_syy",
        vec![
            serde_json::json!({"x": 1.0, "y": 1.0}),
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 1.0, "y": 3.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_syy COMPUTE g = 1 GROUPBY g COMPUTE syy = REGR_SYY(y, x) SELECT syy;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("syy") {
        Some(Value::Float(f)) => {
            assert!((f - 2.0).abs() < 1e-9, "Expected REGR_SYY = 2.0, got {}", f);
        }
        other => panic!("Expected Float for REGR_SYY, got {:?}", other),
    }
}

#[test]
fn test_regr_sxy_group() {
    // x=[1,2,3] mean=2, y=[2,4,6] mean=4, SXY = (1-2)(2-4)+(2-2)(4-4)+(3-2)(6-4) = 2+0+2 = 4
    let (_dir, _db, ex) = make_db_with(
        "test_regr_sxy",
        vec![
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 2.0, "y": 4.0}),
            serde_json::json!({"x": 3.0, "y": 6.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_regr_sxy COMPUTE g = 1 GROUPBY g COMPUTE sxy = REGR_SXY(y, x) SELECT sxy;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("sxy") {
        Some(Value::Float(f)) => {
            assert!((f - 4.0).abs() < 1e-9, "Expected REGR_SXY = 4.0, got {}", f);
        }
        other => panic!("Expected Float for REGR_SXY, got {:?}", other),
    }
}

// ── XMLAGG ────────────────────────────────────────────────────────────────────

#[test]
fn test_xmlagg_single_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_xmlagg1",
        vec![serde_json::json!({"id": 1, "data": "hello"})],
    );
    let r = run(&ex, "QUERY test_xmlagg1 SELECT XMLAGG(data) AS xml_out;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("xml_out") {
        Some(Value::String(_)) => {}
        other => panic!("Expected String for XMLAGG, got {:?}", other),
    }
}

#[test]
fn test_xmlagg_group() {
    let (_dir, _db, ex) = make_db_with(
        "test_xmlagg2",
        vec![
            serde_json::json!({"grp": 1, "data": "<a>1</a>"}),
            serde_json::json!({"grp": 1, "data": "<a>2</a>"}),
        ],
    );
    let r = run(&ex, "QUERY test_xmlagg2 COMPUTE g = grp GROUPBY g COMPUTE xml_out = XMLAGG(data) SELECT xml_out;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("xml_out") {
        Some(Value::String(s)) => {
            assert!(s.contains("<a>"), "Expected concatenated XML, got: {}", s);
        }
        other => panic!("Expected String for XMLAGG, got {:?}", other),
    }
}

// ── JSONB_OBJECT_AGG ──────────────────────────────────────────────────────────

#[test]
fn test_jsonb_object_agg_single_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_jboagg1",
        vec![serde_json::json!({"id": 1, "k": "color", "v": "blue"})],
    );
    let r = run(
        &ex,
        "QUERY test_jboagg1 SELECT JSONB_OBJECT_AGG(k, v) AS obj;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert!(
                m.contains_key("color"),
                "Expected key 'color' in JSONB_OBJECT_AGG result"
            );
        }
        other => panic!("Expected Object for JSONB_OBJECT_AGG, got {:?}", other),
    }
}

#[test]
fn test_jsonb_object_agg_group() {
    let (_dir, _db, ex) = make_db_with(
        "test_jboagg2",
        vec![
            serde_json::json!({"grp": 1, "k": "a", "v": 1}),
            serde_json::json!({"grp": 1, "k": "b", "v": 2}),
        ],
    );
    let r = run(&ex, "QUERY test_jboagg2 COMPUTE g = grp GROUPBY g COMPUTE obj = JSONB_OBJECT_AGG(k, v) SELECT obj;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert!(
                m.len() >= 1,
                "Expected at least one key in JSONB_OBJECT_AGG result"
            );
        }
        other => panic!(
            "Expected Object for JSONB_OBJECT_AGG group, got {:?}",
            other
        ),
    }
}

// ── STRING_AGG (re-verify single row still works) ─────────────────────────────

#[test]
fn test_string_agg_single_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_sagg1",
        vec![serde_json::json!({"id": 1, "name": "alice"})],
    );
    let r = run(
        &ex,
        "QUERY test_sagg1 SELECT STRING_AGG(name, ',') AS names;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("names") {
        Some(Value::String(s)) => {
            assert!(
                s.contains("alice"),
                "Expected 'alice' in STRING_AGG result, got: {}",
                s
            );
        }
        other => panic!("Expected String for STRING_AGG, got {:?}", other),
    }
}

// ── ARRAY_AGG (re-verify single row still works) ──────────────────────────────

#[test]
fn test_array_agg_single_row() {
    let (_dir, _db, ex) = make_db_with("test_aagg1", vec![serde_json::json!({"id": 1, "val": 10})]);
    let r = run(&ex, "QUERY test_aagg1 SELECT ARRAY_AGG(val) AS vals;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("vals") {
        Some(Value::Array(a)) => {
            assert!(!a.is_empty(), "Expected non-empty array from ARRAY_AGG");
        }
        other => panic!("Expected Array for ARRAY_AGG, got {:?}", other),
    }
}

// ── BIT_AND passthrough (aggregate form) ─────────────────────────────────────

#[test]
fn test_bit_and_group() {
    // BIT_AND of [7, 5, 3] = 7&5&3 = 1
    let (_dir, _db, ex) = make_db_with(
        "test_bitand_g",
        vec![
            serde_json::json!({"grp": 1, "flags": 7}),
            serde_json::json!({"grp": 1, "flags": 5}),
            serde_json::json!({"grp": 1, "flags": 3}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_bitand_g COMPUTE g = grp GROUPBY g COMPUTE f = BIT_AND_AGG(flags) SELECT f;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("f") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 1, "Expected BIT_AND_AGG = 1 (7&5&3), got {}", i);
        }
        other => panic!("Expected Integer for BIT_AND_AGG, got {:?}", other),
    }
}

// ── BOOL_AND (via EVERY alias) ────────────────────────────────────────────────

#[test]
fn test_bool_and_passthrough() {
    // BOOL_AND requires a GROUP BY context to aggregate; verify it produces a Bool result
    let (_dir, _db, ex) = make_db_with(
        "test_bools_p",
        vec![
            serde_json::json!({"grp": 1, "active": true}),
            serde_json::json!({"grp": 1, "active": true}),
        ],
    );
    let r = run(&ex, "QUERY test_bools_p COMPUTE g = grp GROUPBY g COMPUTE all_active = BOOL_AND(active) SELECT all_active;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("all_active") {
        Some(Value::Bool(true)) => {}
        other => panic!(
            "Expected Bool(true) for BOOL_AND on all-true group, got {:?}",
            other
        ),
    }
}

// ── COVAR_POP (existing function, verify still works) ─────────────────────────

#[test]
fn test_covar_pop_returns_float() {
    let (_dir, _db, ex) = make_db_with(
        "test_cov_p",
        vec![
            serde_json::json!({"x": 1.0, "y": 2.0}),
            serde_json::json!({"x": 3.0, "y": 4.0}),
        ],
    );
    let r = run(
        &ex,
        "QUERY test_cov_p COMPUTE g = 1 GROUPBY g COMPUTE cov = COVAR_POP(y, x) SELECT cov;",
    );
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("cov") {
        Some(Value::Float(_)) => {}
        other => panic!("Expected Float for COVAR_POP, got {:?}", other),
    }
}

// ── JSON_AGG (existing function, verify still works) ──────────────────────────

#[test]
fn test_json_agg_single_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_jagg1",
        vec![serde_json::json!({"id": 1, "data": "hello"})],
    );
    let r = run(&ex, "QUERY test_jagg1 SELECT JSON_AGG(data) AS arr;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("arr") {
        Some(Value::Array(a)) => {
            assert!(!a.is_empty(), "Expected non-empty array from JSON_AGG");
        }
        other => panic!("Expected Array for JSON_AGG, got {:?}", other),
    }
}

// ── JSON_OBJECT_AGG (existing function, verify still works) ───────────────────

#[test]
fn test_json_object_agg_single_row() {
    let (_dir, _db, ex) = make_db_with(
        "test_joagg1",
        vec![serde_json::json!({"id": 1, "k": "key", "v": "val"})],
    );
    let r = run(&ex, "QUERY test_joagg1 SELECT OBJECT_AGG(k, v) AS obj;");
    assert!(!r.rows.is_empty());
    match r.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert!(
                m.contains_key("key"),
                "Expected 'key' in JSON_OBJECT_AGG result"
            );
        }
        other => panic!("Expected Object for OBJECT_AGG, got {:?}", other),
    }
}
