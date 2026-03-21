/// Integration tests for PQL statistical testing and hypothesis testing functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (dir, db, ex)
}

fn get_float(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => panic!("Expected Float/Integer, got {:?}", v),
    }
}

fn get_obj_float(obj: &std::collections::HashMap<String, Value>, key: &str) -> f64 {
    get_float(obj.get(key).unwrap_or_else(|| {
        panic!(
            "Key '{}' not in obj {:?}",
            key,
            obj.keys().collect::<Vec<_>>()
        )
    }))
}

// ── PEARSON_CORR ──────────────────────────────────────────────────────────────

#[test]
fn test_pearson_perfect_positive() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = PEARSON_CORR([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f - 1.0).abs() < 0.001,
        "perfect correlation should be 1.0, got {}",
        f
    );
}

#[test]
fn test_pearson_perfect_negative() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = PEARSON_CORR([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 6.0, 4.0, 2.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f + 1.0).abs() < 0.001,
        "perfect negative correlation should be -1.0, got {}",
        f
    );
}

#[test]
fn test_pearson_coef_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = PEARSON_COEF([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f - 1.0).abs() < 0.001,
        "PEARSON_COEF alias should work, got {}",
        f
    );
}

#[test]
fn test_pearson_zero_correlation() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Orthogonal-ish data
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = PEARSON_CORR([1.0, 2.0, 3.0, 4.0], [1.0, -1.0, 1.0, -1.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(f.abs() < 0.1, "near-zero correlation, got {}", f);
}

// ── SPEARMAN_RHO ──────────────────────────────────────────────────────────────

#[test]
fn test_spearman_perfect_positive() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = SPEARMAN_RHO([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f - 1.0).abs() < 0.001,
        "perfect Spearman should be 1.0, got {}",
        f
    );
}

#[test]
fn test_spearman_rank_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = SPEARMAN_RANK([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f + 1.0).abs() < 0.001,
        "SPEARMAN_RANK alias perfect negative, got {}",
        f
    );
}

// ── KENDALL_TAU ───────────────────────────────────────────────────────────────

#[test]
fn test_kendall_tau_perfect_concordant() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = KENDALL_TAU([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f - 1.0).abs() < 0.001,
        "all concordant should be 1.0, got {}",
        f
    );
}

#[test]
fn test_kendall_correlation_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = KENDALL_CORRELATION([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        (f + 1.0).abs() < 0.001,
        "KENDALL_CORRELATION all discordant should be -1.0, got {}",
        f
    );
}

// ── POINT_BISERIAL ────────────────────────────────────────────────────────────

#[test]
fn test_point_biserial_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE r = POINT_BISERIAL([0.0, 0.0, 1.0, 1.0], [1.0, 2.0, 8.0, 9.0]) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("r").unwrap());
    assert!(
        f > 0.9,
        "binary 0,0,1,1 with low,low,high,high should be ~1, got {}",
        f
    );
}

#[test]
fn test_point_biserial_r_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE r = POINT_BISERIAL_R([0.0, 1.0], [1.0, 2.0]) SELECT r;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(v) => {
            let _ = get_float(v);
        } // should return a number
        None => panic!("Expected result"),
    }
}

// ── T_TEST_ONE_SAMPLE ─────────────────────────────────────────────────────────

#[test]
fn test_t_test_one_sample_zero_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // mean=3, mu=3 => t_stat~0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = T_TEST_ONE_SAMPLE([1.0, 2.0, 3.0, 4.0, 5.0], 3.0) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let t = get_obj_float(obj, "t_stat");
            assert!(t.abs() < 0.001, "mean==mu => t~0, got {}", t);
            assert!(obj.contains_key("df"), "should have df");
            assert!(
                obj.contains_key("p_value_approx"),
                "should have p_value_approx"
            );
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_t_test_one_sample_large_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = T_TEST_ONE_SAMPLE([10.0, 11.0, 12.0, 13.0, 14.0], 0.0) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let t = get_obj_float(obj, "t_stat");
            assert!(
                t > 5.0,
                "large deviation from mu=0 should give t>5, got {}",
                t
            );
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_t_one_sample_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE res = T_ONE_SAMPLE([1.0, 2.0, 3.0], 2.0) SELECT res;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("t_stat"), "T_ONE_SAMPLE alias should work");
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── T_TEST_TWO_SAMPLE ─────────────────────────────────────────────────────────

#[test]
fn test_t_test_two_sample_same_distributions() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = T_TEST_TWO_SAMPLE([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let t = get_obj_float(obj, "t_stat");
            assert!(t.abs() < 0.001, "same distributions => t~0, got {}", t);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_t_test_two_sample_different_means() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = T_TEST_TWO_SAMPLE([1.0, 2.0, 3.0], [100.0, 101.0, 102.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let t = get_obj_float(obj, "t_stat");
            assert!(t.abs() > 5.0, "very different means => |t|>5, got {}", t);
            let df = get_obj_float(obj, "df");
            assert!(df > 0.0, "df should be positive, got {}", df);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── T_TEST_PAIRED ─────────────────────────────────────────────────────────────

#[test]
fn test_t_test_paired_no_difference() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Identical arrays => t=0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = T_TEST_PAIRED([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Null) => {} // acceptable when std_d=0
        Some(Value::Object(obj)) => {
            let t = get_obj_float(obj, "t_stat");
            assert!(t.abs() < 0.001, "identical pairs => t~0, got {}", t);
        }
        other => panic!("Expected Null or Object, got {:?}", other),
    }
}

#[test]
fn test_t_test_paired_consistent_increase() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = PAIRED_T_TEST([5.0, 6.0, 7.0, 8.0, 9.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let t = get_obj_float(obj, "t_stat");
            assert!(t > 5.0, "consistent +4 shift => t>5, got {}", t);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── Z_TEST_ONE_SAMPLE ─────────────────────────────────────────────────────────

#[test]
fn test_z_test_zero_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE z = Z_TEST_ONE_SAMPLE([3.0, 3.0, 3.0, 3.0], 3.0, 1.0) SELECT z;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("z").unwrap());
    assert!(f.abs() < 0.001, "mean==mu => z~0, got {}", f);
}

#[test]
fn test_z_test_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE z = Z_TEST([5.0, 5.0, 5.0], 0.0, 1.0) SELECT z;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("z").unwrap());
    assert!(f > 5.0, "Z_TEST alias: mean=5 vs mu=0 => z>5, got {}", f);
}

// ── F_TEST_VARIANCE ───────────────────────────────────────────────────────────

#[test]
fn test_f_test_equal_variance() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE f = F_TEST_VARIANCE([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT f;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("f").unwrap());
    assert!((f - 1.0).abs() < 0.001, "equal variances => F=1, got {}", f);
}

#[test]
fn test_f_test_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE f = F_TEST([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) SELECT f;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("f").unwrap());
    assert!(
        f > 0.0,
        "F_TEST alias should return positive value, got {}",
        f
    );
}

#[test]
fn test_f_test_higher_var_first() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // var([10,20,30]) = 100, var([1,2,3]) = 1 => F~100
    let mut p =
        Parser::new(r#"QUERY t COMPUTE f = F_TEST([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]) SELECT f;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("f").unwrap());
    assert!((f - 100.0).abs() < 1.0, "F should be ~100, got {}", f);
}

// ── CHI_SQUARED_GOF ───────────────────────────────────────────────────────────

#[test]
fn test_chi_squared_gof_perfect_fit() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = CHI_SQUARED_GOF([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let chi2 = get_obj_float(obj, "chi2_stat");
            assert!(chi2.abs() < 1e-10, "perfect fit => chi2=0, got {}", chi2);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_chi_squared_gof_bad_fit() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = CHI_SQ_TEST([50.0, 5.0, 5.0], [20.0, 20.0, 20.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let chi2 = get_obj_float(obj, "chi2_stat");
            assert!(chi2 > 1.0, "bad fit should give chi2>1, got {}", chi2);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── CHI_SQUARED_INDEPENDENCE ──────────────────────────────────────────────────

#[test]
fn test_chi_squared_independence_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // 2x2 table with some association
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = CHI_SQUARED_INDEPENDENCE([[10.0, 20.0], [20.0, 10.0]]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("chi2_stat"), "should have chi2_stat");
            assert!(obj.contains_key("df"), "should have df");
            let df = obj.get("df").unwrap();
            assert_eq!(df, &Value::Integer(1), "2x2 table: df=1, got {:?}", df);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_chi_sq_independence_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = CHI_SQ_INDEPENDENCE([[10.0, 10.0], [10.0, 10.0]]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let chi2 = get_obj_float(obj, "chi2_stat");
            assert!(chi2.abs() < 0.001, "uniform table => chi2~0, got {}", chi2);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── MANN_WHITNEY ──────────────────────────────────────────────────────────────

#[test]
fn test_mann_whitney_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE res = MANN_WHITNEY([1.0, 2.0], [3.0, 4.0]) SELECT res;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let u = obj.get("u_stat").unwrap();
            assert_eq!(u, &Value::Integer(0), "all g1<g2 => U=0, got {:?}", u);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_mann_whitney_test_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = MANN_WHITNEY_TEST([10.0, 20.0], [1.0, 2.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let u = obj.get("u_stat").unwrap();
            assert_eq!(u, &Value::Integer(4), "all g1>g2 => U=4, got {:?}", u);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── WILCOXON_SIGNED_RANK ──────────────────────────────────────────────────────

#[test]
fn test_wilcoxon_identical_arrays() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = WILCOXON_SIGNED_RANK([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let w = get_obj_float(obj, "w_stat");
            assert!(w == 0.0, "identical => w=0, got {}", w);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_wilcoxon_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = WILCOXON([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let w = get_obj_float(obj, "w_stat");
            assert!(
                w > 0.0,
                "WILCOXON alias: all positive diffs => W>0, got {}",
                w
            );
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── KRUSKAL_WALLIS ────────────────────────────────────────────────────────────

#[test]
fn test_kruskal_wallis_same_groups() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = KRUSKAL_WALLIS([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let h = get_obj_float(obj, "h_stat");
            assert!(h.abs() < 0.1, "same groups => H~0, got {}", h);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_kruskal_h_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = KRUSKAL_H([[1.0, 2.0], [10.0, 11.0], [20.0, 21.0]]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let h = get_obj_float(obj, "h_stat");
            assert!(h > 1.0, "KRUSKAL_H: different groups => H>1, got {}", h);
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── RUNS_TEST ─────────────────────────────────────────────────────────────────

#[test]
fn test_runs_test_alternating() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Alternating above/below median => many runs
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = RUNS_TEST([1.0, 10.0, 1.0, 10.0, 1.0, 10.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let runs = obj.get("runs").unwrap();
            match runs {
                Value::Integer(r) => {
                    assert!(*r >= 2, "alternating should give multiple runs, got {}", r)
                }
                _ => panic!("Expected Integer for runs, got {:?}", runs),
            }
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_wald_wolfowitz_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = WALD_WOLFOWITZ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(
                obj.contains_key("runs"),
                "WALD_WOLFOWITZ alias should return runs"
            );
            assert!(
                obj.contains_key("expected_runs"),
                "should have expected_runs"
            );
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── LINEAR_REGRESSION ─────────────────────────────────────────────────────────

#[test]
fn test_linear_regression_perfect_fit() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // y = 2x + 1
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = LINEAR_REGRESSION([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 5.0, 7.0, 9.0, 11.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            let slope = get_obj_float(obj, "slope");
            let intercept = get_obj_float(obj, "intercept");
            let r2 = get_obj_float(obj, "r_squared");
            assert!(
                (slope - 2.0).abs() < 0.001,
                "slope should be 2, got {}",
                slope
            );
            assert!(
                (intercept - 1.0).abs() < 0.001,
                "intercept should be 1, got {}",
                intercept
            );
            assert!(
                (r2 - 1.0).abs() < 0.001,
                "R2 should be 1 for perfect fit, got {}",
                r2
            );
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_linreg_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = LINREG([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("slope"), "LINREG alias should work");
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── PREDICT_LINEAR ────────────────────────────────────────────────────────────

#[test]
fn test_predict_linear_on_line() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // y = 3x + 2, predict at x=10 => 32
    let mut p = Parser::new(
        r#"QUERY t COMPUTE yhat = PREDICT_LINEAR([0.0, 1.0, 2.0, 3.0], [2.0, 5.0, 8.0, 11.0], 10.0) SELECT yhat;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("yhat").unwrap());
    assert!(
        (f - 32.0).abs() < 0.001,
        "predict at x=10 for y=3x+2 should be 32, got {}",
        f
    );
}

#[test]
fn test_linreg_predict_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE yhat = LINREG_PREDICT([0.0, 1.0, 2.0], [0.0, 2.0, 4.0], 5.0) SELECT yhat;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("yhat").unwrap());
    assert!(
        (f - 10.0).abs() < 0.001,
        "LINREG_PREDICT: y=2x, predict at 5 => 10, got {}",
        f
    );
}

// ── REGRESSION_RESIDUALS ──────────────────────────────────────────────────────

#[test]
fn test_residuals_zero_for_perfect_fit() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = REGRESSION_RESIDUALS([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "should have 3 residuals");
            for r in arr {
                let f = get_float(r);
                assert!(
                    f.abs() < 0.001,
                    "residuals should be ~0 for perfect fit, got {}",
                    f
                );
            }
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_residuals_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = RESIDUALS([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4, "RESIDUALS alias should return 4 residuals");
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

// ── POLYNOMIAL_REGRESSION ─────────────────────────────────────────────────────

#[test]
fn test_poly_reg_linear() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // y = 2x + 1 => coeffs should be [1, 2]
    let mut p = Parser::new(
        r#"QUERY t COMPUTE coeffs = POLYNOMIAL_REGRESSION([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], 1) SELECT coeffs;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("coeffs") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "degree 1 should give 2 coefficients");
            let a0 = get_float(&arr[0]);
            let a1 = get_float(&arr[1]);
            assert!((a0 - 1.0).abs() < 0.001, "a0 should be 1, got {}", a0);
            assert!((a1 - 2.0).abs() < 0.001, "a1 should be 2, got {}", a1);
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_poly_reg_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE coeffs = POLY_REG([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 4.0, 9.0, 16.0], 2) SELECT coeffs;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("coeffs") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "POLY_REG degree 2 should give 3 coefficients");
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

// ── MOVING_REGRESSION ─────────────────────────────────────────────────────────

#[test]
fn test_rolling_linreg_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // y=2x, window=3 => every window gives slope=2
    let mut p = Parser::new(
        r#"QUERY t COMPUTE slopes = MOVING_REGRESSION([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 4.0, 6.0, 8.0], 3) SELECT slopes;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("slopes") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "5 points, window=3 => 3 slopes");
            for s in arr {
                let f = get_float(s);
                assert!(
                    (f - 2.0).abs() < 0.001,
                    "each window slope should be 2, got {}",
                    f
                );
            }
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_rolling_linreg_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE slopes = ROLLING_LINREG([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], 2) SELECT slopes;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("slopes") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "ROLLING_LINREG: 4 pts window=2 => 3 slopes");
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

// ── ANDERSON_DARLING ──────────────────────────────────────────────────────────

#[test]
fn test_anderson_darling_returns_float() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a2 = ANDERSON_DARLING([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]) SELECT a2;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("a2").unwrap());
    assert!(
        f.is_finite(),
        "Anderson-Darling should return finite value, got {}",
        f
    );
}

#[test]
fn test_ad_test_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a2 = AD_TEST([0.0, 1.0, 2.0, 3.0, 4.0]) SELECT a2;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("a2") {
        Some(v) => {
            let _ = get_float(v);
        }
        None => panic!("AD_TEST alias should return a value"),
    }
}

// ── SHAPIRO_WILK_APPROX ───────────────────────────────────────────────────────

#[test]
fn test_shapiro_wilk_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE w = SHAPIRO_WILK_APPROX([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) SELECT w;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("w").unwrap());
    assert!(
        f >= 0.0 && f <= 1.0,
        "Shapiro W should be in [0,1], got {}",
        f
    );
}

#[test]
fn test_shapiro_test_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE w = SHAPIRO_TEST([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]) SELECT w;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("w").unwrap());
    assert!(
        f >= 0.0 && f <= 1.0,
        "SHAPIRO_TEST alias: W in [0,1], got {}",
        f
    );
}

// ── JARQUE_BERA ───────────────────────────────────────────────────────────────

#[test]
fn test_jarque_bera_normal_data() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Roughly normal data => small JB stat
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = JARQUE_BERA([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("jb_stat"), "should have jb_stat");
            assert!(obj.contains_key("skewness"), "should have skewness");
            assert!(obj.contains_key("kurtosis"), "should have kurtosis");
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_jb_test_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE res = JB_TEST([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) SELECT res;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("jb_stat"), "JB_TEST alias should work");
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

// ── LILLIEFORS_TEST ───────────────────────────────────────────────────────────

#[test]
fn test_lilliefors_test_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE d = LILLIEFORS_TEST([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) SELECT d;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("d").unwrap());
    assert!(
        f >= 0.0 && f <= 1.0,
        "Lilliefors D should be in [0,1], got {}",
        f
    );
}

#[test]
fn test_ks_normality_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE d = KS_NORMALITY([1.0, 2.0, 3.0, 4.0, 5.0]) SELECT d;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("d").unwrap());
    assert!(
        f >= 0.0 && f <= 1.0,
        "KS_NORMALITY alias: D in [0,1], got {}",
        f
    );
}

// ── COHENS_D ──────────────────────────────────────────────────────────────────

#[test]
fn test_cohens_d_zero_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE d = COHENS_D_EFFECT([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT d;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("d").unwrap());
    assert!(f.abs() < 0.001, "same distributions => d=0, got {}", f);
}

#[test]
fn test_cohens_d_large_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE d = COHEN_D([1.0, 2.0, 3.0], [10.0, 11.0, 12.0]) SELECT d;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("d").unwrap());
    assert!(f.abs() > 3.0, "large gap => |d|>3, got {}", f);
}

// ── HEDGES_G ──────────────────────────────────────────────────────────────────

#[test]
fn test_hedges_g_small_sample_correction() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE g = HEDGES_G([10.0, 11.0, 12.0], [1.0, 2.0, 3.0]) SELECT g;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("g").unwrap());
    assert!(
        f > 0.0,
        "Hedges g should be positive when grp1 > grp2, got {}",
        f
    );
}

#[test]
fn test_effect_size_g_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE g = EFFECT_SIZE_G([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]) SELECT g;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("g").unwrap());
    assert!(
        f.abs() < 0.001,
        "EFFECT_SIZE_G: same distributions => g~0, got {}",
        f
    );
}

// ── ETA_SQUARED ───────────────────────────────────────────────────────────────

#[test]
fn test_eta_squared_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE eta = ETA_SQUARED(50.0, 100.0) SELECT eta;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("eta").unwrap());
    assert!(
        (f - 0.5).abs() < 0.001,
        "SS_between=50, SS_total=100 => eta^2=0.5, got {}",
        f
    );
}

#[test]
fn test_eta_sq_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE eta = ETA_SQ(20.0, 100.0) SELECT eta;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("eta").unwrap());
    assert!((f - 0.2).abs() < 0.001, "ETA_SQ: 20/100=0.2, got {}", f);
}

// ── OMEGA_SQUARED ─────────────────────────────────────────────────────────────

#[test]
fn test_omega_squared_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE omega = OMEGA_SQUARED(5.0, 2.0, 30.0) SELECT omega;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("omega").unwrap());
    assert!(f > 0.0 && f < 1.0, "omega^2 should be in (0,1), got {}", f);
}

#[test]
fn test_omega_sq_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE omega = OMEGA_SQ(10.0, 3.0, 60.0) SELECT omega;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("omega").unwrap());
    assert!(
        f > 0.0 && f < 1.0,
        "OMEGA_SQ alias: value in (0,1), got {}",
        f
    );
}

// ── COHENS_H ──────────────────────────────────────────────────────────────────

#[test]
fn test_cohens_h_zero_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = COHENS_H(0.5, 0.5) SELECT h;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("h").unwrap());
    assert!(f.abs() < 0.001, "same proportions => h=0, got {}", f);
}

#[test]
fn test_cohens_h_prop_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = COHENS_H_PROP(0.8, 0.2) SELECT h;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(res.rows[0].data.get("h").unwrap());
    assert!(
        f > 0.5,
        "COHENS_H_PROP: 0.8 vs 0.2 should give h>0.5, got {}",
        f
    );
}
