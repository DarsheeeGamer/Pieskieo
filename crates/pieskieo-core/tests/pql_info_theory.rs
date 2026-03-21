/// Integration tests for PQL information theory and entropy functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db(ns: &str, doc: serde_json::Value) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    (dir, db, ex)
}

// ── Shannon Entropy ───────────────────────────────────────────────────────────

#[test]
fn test_shannon_entropy_uniform() {
    // Uniform distribution over 4 outcomes -> entropy = 2 bits
    let (_dir, _db, ex) = make_db("t", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t COMPUTE h = SHANNON_ENTROPY([0.25, 0.25, 0.25, 0.25]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 2.0).abs() < 0.001,
            "uniform entropy should be 2 bits, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_shannon_entropy_certain() {
    // Deterministic distribution -> entropy = 0
    let (_dir, _db, ex) = make_db("t2", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t2 COMPUTE h = SHANNON_ENTROPY([1.0, 0.0, 0.0]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(f.abs() < 0.001, "certain entropy should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_h_entropy_alias() {
    // H_ENTROPY is alias for SHANNON_ENTROPY
    let (_dir, _db, ex) = make_db("t3", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t3 COMPUTE h = H_ENTROPY([0.5, 0.5]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "binary entropy should be 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_entropy_from_counts() {
    // ENTROPY_FROM_COUNTS normalizes automatically
    let (_dir, _db, ex) = make_db("t4", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t4 COMPUTE h = ENTROPY_FROM_COUNTS([10, 10, 10, 10]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 2.0).abs() < 0.001,
            "uniform counts entropy should be 2 bits, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_entropy_counts_alias() {
    let (_dir, _db, ex) = make_db("t5", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t5 COMPUTE h = ENTROPY_COUNTS([50, 50]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "equal counts should give 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_entropy_nats() {
    // H_NATS for uniform 2-outcome -> ln(2) nats
    let (_dir, _db, ex) = make_db("t6", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t6 COMPUTE h = ENTROPY_NATS([0.5, 0.5]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ln2 = 2.0_f64.ln();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - ln2).abs() < 0.001,
            "binary entropy should be ln(2) nats, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_h_nats_alias() {
    let (_dir, _db, ex) = make_db("t7", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t7 COMPUTE h = H_NATS([0.25, 0.25, 0.25, 0.25]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let expected = 4.0_f64.ln(); // ln(4) nats for uniform 4-outcome
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - expected).abs() < 0.001,
            "expected ln(4) nats, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_normalized_entropy() {
    // Uniform over 4 -> normalized entropy = 1.0
    let (_dir, _db, ex) = make_db("t8", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t8 COMPUTE h = NORMALIZED_ENTROPY([0.25, 0.25, 0.25, 0.25]) SELECT h;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "normalized entropy of uniform should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_h_norm_alias() {
    let (_dir, _db, ex) = make_db("t9", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t9 COMPUTE h = H_NORM([1.0, 0.0]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "degenerate norm entropy should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Joint and Conditional Entropy ─────────────────────────────────────────────

#[test]
fn test_joint_entropy_independent() {
    // For independent X,Y uniform binary: H(X,Y) = H(X) + H(Y) = 2 bits
    let (_dir, _db, ex) = make_db("t10", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t10 COMPUTE h = JOINT_ENTROPY([[0.25, 0.25], [0.25, 0.25]]) SELECT h;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 2.0).abs() < 0.001,
            "joint entropy of independent uniform should be 2 bits, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_h_joint_alias() {
    let (_dir, _db, ex) = make_db("t11", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t11 COMPUTE h = H_JOINT([[0.5, 0.0], [0.0, 0.5]]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "perfectly correlated -> H(X,Y) = 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_conditional_entropy_perfect_dependence() {
    // Perfect dependence: H(Y|X) = 0
    let (_dir, _db, ex) = make_db("t12", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t12 COMPUTE h = CONDITIONAL_ENTROPY([[0.5, 0.0], [0.0, 0.5]]) SELECT h;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "conditional entropy with perfect dependence should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_h_cond_alias() {
    // Independent: H(Y|X) = H(Y) = 1 bit
    let (_dir, _db, ex) = make_db("t13", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t13 COMPUTE h = H_COND([[0.25, 0.25], [0.25, 0.25]]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "conditional entropy of independent should be 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_mutual_information_independent() {
    // Independent X,Y: I(X;Y) = 0
    let (_dir, _db, ex) = make_db("t14", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t14 COMPUTE mi = MUTUAL_INFORMATION([[0.25, 0.25], [0.25, 0.25]]) SELECT mi;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mi") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "MI of independent variables should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_mutual_info_alias_perfect() {
    // Perfect dependence: I(X;Y) = H(X) = 1 bit
    let (_dir, _db, ex) = make_db("t15", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t15 COMPUTE mi = MUTUAL_INFO([[0.5, 0.0], [0.0, 0.5]]) SELECT mi;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mi") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "MI of perfectly dependent should be 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_information_gain_basic() {
    // Pure parent [5,5], pure children [[5,0],[0,5]] -> IG = 1 bit
    let (_dir, _db, ex) = make_db("t16", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t16 COMPUTE ig = INFORMATION_GAIN([5, 5], [[5, 0], [0, 5]]) SELECT ig;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ig") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "IG after perfect split should be 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_info_gain_alias_no_gain() {
    // Same distribution in all children -> zero gain
    let (_dir, _db, ex) = make_db("t17", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t17 COMPUTE ig = INFO_GAIN([4, 4], [[2, 2], [2, 2]]) SELECT ig;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ig") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "IG with no improvement should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── KL Divergence ─────────────────────────────────────────────────────────────

#[test]
fn test_kl_divergence_identical() {
    let (_dir, _db, ex) = make_db("t18", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t18 COMPUTE d = KL_DIVERGENCE([0.5, 0.5], [0.5, 0.5]) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "KL of identical distributions should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_kl_div_alias() {
    // KL([1,0],[0.5,0.5]) should be large (infinite in theory -> 1e18)
    let (_dir, _db, ex) = make_db("t19", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t19 COMPUTE d = KL_DIV([1.0, 0.0], [0.5, 0.5]) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            *f > 0.5,
            "KL of [1,0]||[0.5,0.5] should be positive, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_kl_divergence_known_value() {
    // KL([0.5,0.5]||[0.25,0.75]) = 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.75)
    // = 0.5*1 + 0.5*log2(2/3) ~ 0.5 - 0.5*0.585 = 0.5 - 0.292 = 0.208
    let (_dir, _db, ex) = make_db("t20", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t20 COMPUTE d = KL_DIVERGENCE([0.5, 0.5], [0.25, 0.75]) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.208).abs() < 0.01,
            "KL divergence should be ~0.208, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── JS Divergence ─────────────────────────────────────────────────────────────

#[test]
fn test_js_divergence_identical() {
    let (_dir, _db, ex) = make_db("t21", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t21 COMPUTE d = JS_DIVERGENCE([0.5, 0.5], [0.5, 0.5]) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f.abs() < 0.001, "JS of identical should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_js_div_alias_max() {
    // JS([1,0],[0,1]) = 1 bit (maximum for JS in bits)
    let (_dir, _db, ex) = make_db("t22", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t22 COMPUTE d = JS_DIV([1.0, 0.0], [0.0, 1.0]) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "JS of opposite distributions should be 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_js_divergence_symmetric() {
    // JS(P||Q) = JS(Q||P)
    let (_dir, _db, ex) = make_db("t23", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t23 COMPUTE d1 = JS_DIVERGENCE([0.7, 0.3], [0.2, 0.8]) COMPUTE d2 = JS_DIVERGENCE([0.2, 0.8], [0.7, 0.3]) SELECT d1, d2;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let d1 = match r.rows[0].data.get("d1") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected float for d1, got {:?}", other),
    };
    let d2 = match r.rows[0].data.get("d2") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected float for d2, got {:?}", other),
    };
    assert!(
        (d1 - d2).abs() < 0.001,
        "JS divergence should be symmetric, got {} vs {}",
        d1,
        d2
    );
}

// ── H_CROSS ───────────────────────────────────────────────────────────────────

#[test]
fn test_h_cross_identical() {
    // Cross-entropy of distribution with itself = entropy
    let (_dir, _db, ex) = make_db("t24", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t24 COMPUTE ce = H_CROSS([0.5, 0.5], [0.5, 0.5]) COMPUTE h = SHANNON_ENTROPY([0.5, 0.5]) SELECT ce, h;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ce = match r.rows[0].data.get("ce") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected float for ce, got {:?}", other),
    };
    let h = match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected float for h, got {:?}", other),
    };
    assert!(
        (ce - h).abs() < 0.01,
        "H_CROSS(p,p) should equal H(p), got ce={} h={}",
        ce,
        h
    );
}

// ── Total Variation Distance ──────────────────────────────────────────────────

#[test]
fn test_tv_dist_identical() {
    let (_dir, _db, ex) = make_db("t25", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t25 COMPUTE tv = TOTAL_VARIATION_DIST([0.3, 0.7], [0.3, 0.7]) SELECT tv;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tv") {
        Some(Value::Float(f)) => assert!(f.abs() < 0.001, "TV of identical should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_tv_dist_alias_max() {
    // TV([1,0],[0,1]) = 1
    let (_dir, _db, ex) = make_db("t26", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t26 COMPUTE tv = TV_DIST([1.0, 0.0], [0.0, 1.0]) SELECT tv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tv") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "TV of opposite distributions should be 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Hellinger Distance ────────────────────────────────────────────────────────

#[test]
fn test_hellinger_identical() {
    let (_dir, _db, ex) = make_db("t27", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t27 COMPUTE h = HELLINGER_DISTANCE([0.5, 0.5], [0.5, 0.5]) SELECT h;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => {
            assert!(
                f.abs() < 0.001,
                "Hellinger of identical should be 0, got {}",
                f
            )
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_hellinger_alias() {
    // HELLINGER([1,0],[0,1]) = 1
    let (_dir, _db, ex) = make_db("t28", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t28 COMPUTE h = HELLINGER([1.0, 0.0], [0.0, 1.0]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "Hellinger of opposite should be 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Bhattacharyya ─────────────────────────────────────────────────────────────

#[test]
fn test_bhattacharyya_coeff_identical() {
    // BC of distribution with itself = 1.0
    let (_dir, _db, ex) = make_db("t29", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t29 COMPUTE bc = BHATTACHARYYA_COEFF([0.5, 0.5], [0.5, 0.5]) SELECT bc;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "Bhattacharyya coeff of identical should be 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_bhattacharyya_alias() {
    let (_dir, _db, ex) = make_db("t30", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t30 COMPUTE bc = BHATTACHARYYA([0.25, 0.75], [0.75, 0.25]) SELECT bc;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bc") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && *f <= 1.0,
            "Bhattacharyya coeff should be in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_bhattacharyya_distance_zero() {
    // Distance for identical = 0 (since -ln(1) = 0)
    let (_dir, _db, ex) = make_db("t31", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t31 COMPUTE bd = BHATTACHARYYA_DISTANCE([0.5, 0.5], [0.5, 0.5]) SELECT bd;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bd") {
        Some(Value::Float(f)) => {
            assert!(
                f.abs() < 0.001,
                "Bhattacharyya distance for identical should be ~0, got {}",
                f
            )
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_bhatt_dist_alias() {
    let (_dir, _db, ex) = make_db("t32", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t32 COMPUTE bd = BHATT_DIST([0.3, 0.7], [0.6, 0.4]) SELECT bd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bd") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0,
            "Bhattacharyya distance should be >= 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Huffman / Coding ──────────────────────────────────────────────────────────

#[test]
fn test_huffman_bound_uniform() {
    // Same as entropy
    let (_dir, _db, ex) = make_db("t33", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t33 COMPUTE hb = HUFFMAN_BOUND([25, 25, 25, 25]) SELECT hb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hb") {
        Some(Value::Float(f)) => assert!(
            (*f - 2.0).abs() < 0.001,
            "Huffman bound for uniform should be 2 bits, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_huffman_entropy_alias() {
    let (_dir, _db, ex) = make_db("t34", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t34 COMPUTE hb = HUFFMAN_ENTROPY([1, 1]) SELECT hb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hb") {
        Some(Value::Float(f)) => {
            assert!(
                (*f - 1.0).abs() < 0.001,
                "Huffman bound for binary should be 1 bit, got {}",
                f
            )
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── LZ Complexity ─────────────────────────────────────────────────────────────

#[test]
fn test_lz_complexity_repeated() {
    // Highly repetitive string "aaaaaa" -> low complexity
    let (_dir, _db, ex) = make_db("t35", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t35 COMPUTE c = LEMPEL_ZIV_COMPLEXITY("aaaaaa") SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Integer(i)) => assert!(
            *i <= 3,
            "repeated string should have low LZ complexity, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_lz_complexity_alias() {
    let (_dir, _db, ex) = make_db("t36", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t36 COMPUTE c = LZ_COMPLEXITY("abcdef") SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Integer(i)) => assert!(*i > 0, "LZ complexity should be positive, got {}", i),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── RLE / Kolmogorov ──────────────────────────────────────────────────────────

#[test]
fn test_kolmogorov_compress_repetitive() {
    // "aaaaaaa" compresses to ratio < 1
    let (_dir, _db, ex) = make_db("t37", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t37 COMPUTE r = KOLMOGOROV_APPROX("aaaaaaaaaa") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            *f < 1.0,
            "repetitive string should compress, ratio got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_kolmogorov_compress_alias() {
    let (_dir, _db, ex) = make_db("t38", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t38 COMPUTE r = KOLMOGOROV_COMPRESS("abcde") SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.0, "compression ratio should be positive, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Redundancy ────────────────────────────────────────────────────────────────

#[test]
fn test_redundancy_uniform_is_zero() {
    // Uniform -> max entropy -> 0 redundancy
    let (_dir, _db, ex) = make_db("t39", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t39 COMPUTE r = REDUNDANCY([0.25, 0.25, 0.25, 0.25]) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => {
            assert!(f.abs() < 0.001, "uniform redundancy should be 0, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_info_redundancy_alias_degenerate() {
    // Degenerate distribution -> 1.0 redundancy
    let (_dir, _db, ex) = make_db("t40", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t40 COMPUTE r = INFO_REDUNDANCY([1.0, 0.0, 0.0, 0.0]) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "degenerate redundancy should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Channel Capacity ──────────────────────────────────────────────────────────

#[test]
fn test_bsc_capacity_fair() {
    // p=0.5 -> completely noisy channel -> capacity = 0
    let (_dir, _db, ex) = make_db("t41", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t41 COMPUTE c = BSC_CAPACITY(0.5) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "BSC capacity at p=0.5 should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_capacity_binary_channel_perfect() {
    // p=0 -> perfect channel -> capacity = 1 bit
    let (_dir, _db, ex) = make_db("t42", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t42 COMPUTE c = CAPACITY_BINARY_CHANNEL(0.0) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "BSC capacity at p=0 should be 1 bit, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_awgn_capacity() {
    // SNR=0 -> C = log2(1) = 0, SNR=3 -> C = log2(4) = 2
    let (_dir, _db, ex) = make_db("t43", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t43 COMPUTE c0 = AWGN_CAPACITY(0) COMPUTE c3 = AWGN_CAPACITY(3) SELECT c0, c3;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c0 = match r.rows[0].data.get("c0") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected float for c0, got {:?}", other),
    };
    let c3 = match r.rows[0].data.get("c3") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected float for c3, got {:?}", other),
    };
    assert!(
        c0.abs() < 0.001,
        "AWGN capacity at SNR=0 should be 0, got {}",
        c0
    );
    assert!(
        (c3 - 2.0).abs() < 0.001,
        "AWGN capacity at SNR=3 should be 2 bits, got {}",
        c3
    );
}

#[test]
fn test_channel_capacity_awgn_alias() {
    let (_dir, _db, ex) = make_db("t44", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t44 COMPUTE c = CHANNEL_CAPACITY_AWGN(15) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            *f > 3.9 && *f < 4.1,
            "AWGN at SNR=15 should be ~4 bits, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Gini Impurity ─────────────────────────────────────────────────────────────

#[test]
fn test_gini_impurity_uniform() {
    // Uniform 2-class -> G = 0.5
    let (_dir, _db, ex) = make_db("t45", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t45 COMPUTE g = GINI_IMPURITY([0.5, 0.5]) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.5).abs() < 0.001,
            "Gini of uniform binary should be 0.5, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_gini_alias_pure() {
    // Pure class -> G = 0
    let (_dir, _db, ex) = make_db("t46", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t46 COMPUTE g = GINI([1.0, 0.0]) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(f.abs() < 0.001, "pure Gini should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_gini_from_counts() {
    let (_dir, _db, ex) = make_db("t47", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t47 COMPUTE g = GINI_FROM_COUNTS([10, 10]) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.5).abs() < 0.001,
            "Gini from equal counts should be 0.5, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_gini_counts_alias() {
    let (_dir, _db, ex) = make_db("t48", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t48 COMPUTE g = GINI_COUNTS([100, 0]) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => {
            assert!(f.abs() < 0.001, "pure GINI_COUNTS should be 0, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Weighted Gini ─────────────────────────────────────────────────────────────

#[test]
fn test_weighted_gini_pure_children() {
    // Pure children -> weighted Gini = 0
    let (_dir, _db, ex) = make_db("t49", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t49 COMPUTE wg = WEIGHTED_GINI([[10, 0], [0, 10]]) SELECT wg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("wg") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "weighted Gini with pure children should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_split_gini_alias() {
    let (_dir, _db, ex) = make_db("t50", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t50 COMPUTE wg = SPLIT_GINI([[5, 5], [5, 5]]) SELECT wg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("wg") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.5).abs() < 0.001,
            "SPLIT_GINI of uniform children should be 0.5, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Gain Ratio ────────────────────────────────────────────────────────────────

#[test]
fn test_gain_ratio_perfect_split() {
    // Perfect split: gain = 1 bit, split_info = 1 bit -> gain_ratio = 1.0
    let (_dir, _db, ex) = make_db("t51", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t51 COMPUTE gr = GAIN_RATIO([5, 5], [[5, 0], [0, 5]]) SELECT gr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gr") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 0.001,
            "gain ratio for perfect split should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_information_gain_ratio_alias() {
    let (_dir, _db, ex) = make_db("t52", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t52 COMPUTE gr = INFORMATION_GAIN_RATIO([4, 4], [[2, 2], [2, 2]]) SELECT gr;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gr") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "gain ratio with no gain should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Variance Reduction ────────────────────────────────────────────────────────

#[test]
fn test_variance_reduction_no_split() {
    // Same group -> reduction = 0
    let (_dir, _db, ex) = make_db("t53", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t53 COMPUTE vr = VARIANCE_REDUCTION([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0]) SELECT vr;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("vr") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "variance reduction with single group should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_var_reduction_alias_perfect() {
    // Groups [1,1] and [5,5]: parent var > 0, within-group var = 0 -> large reduction
    let (_dir, _db, ex) = make_db("t54", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t54 COMPUTE vr = VAR_REDUCTION([1.0, 1.0, 5.0, 5.0], [0, 0, 1, 1]) SELECT vr;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("vr") {
        Some(Value::Float(f)) => assert!(
            *f > 3.0,
            "variance reduction with perfect split should be large, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── Coding Theory ─────────────────────────────────────────────────────────────

#[test]
fn test_hamming_bit_zero() {
    // Same number -> distance = 0
    let (_dir, _db, ex) = make_db("t55", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t55 COMPUTE d = HAMMING_DISTANCE_BINARY(12, 12) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 0, "Hamming distance of same numbers should be 0")
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_hamming_bit_alias() {
    // 0b0000 vs 0b1111 -> distance = 4
    let (_dir, _db, ex) = make_db("t56", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t56 COMPUTE d = HAMMING_BIT(0, 15) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 4, "Hamming distance of 0 and 15 should be 4, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_parity_bit_even() {
    // 6 = 0b110 -> 2 ones -> even parity = 0
    let (_dir, _db, ex) = make_db("t57", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t57 COMPUTE p = PARITY_BIT(6) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Integer(i)) => assert_eq!(*i, 0, "parity of 6 (0b110) should be 0, got {}", i),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_even_parity_alias() {
    // 7 = 0b111 -> 3 ones -> odd -> even parity = 1
    let (_dir, _db, ex) = make_db("t58", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t58 COMPUTE p = EVEN_PARITY(7) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("p") {
        Some(Value::Integer(i)) => assert_eq!(*i, 1, "parity of 7 (0b111) should be 1, got {}", i),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_hamming_encode_length() {
    // Hamming(7,4): 4 bits -> 7-bit codeword
    let (_dir, _db, ex) = make_db("t59", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t59 COMPUTE cw = HAMMING_CODE_ENCODE([1, 0, 1, 1]) SELECT cw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cw") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            7,
            "Hamming(7,4) should produce 7 bits, got {}",
            a.len()
        ),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_hamming_encode_alias() {
    let (_dir, _db, ex) = make_db("t60", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t60 COMPUTE cw = HAMMING_ENCODE([0, 0, 0, 0]) SELECT cw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cw") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 7, "Hamming(7,4) should produce 7 bits");
            // all-zero data -> all-zero codeword
            for bit in a {
                match bit {
                    Value::Integer(i) => {
                        assert_eq!(*i, 0, "all-zero data should encode to all-zero codeword")
                    }
                    _ => panic!("expected integer bit"),
                }
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_hamming_detect_valid() {
    // Encode then detect: should detect as valid
    let (_dir, _db, ex) = make_db("t61", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t61 COMPUTE cw = HAMMING_CODE_ENCODE([1, 0, 1, 1]) COMPUTE ok = HAMMING_CODE_DETECT(cw) SELECT ok;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ok") {
        Some(Value::Bool(b)) => assert!(*b, "freshly encoded codeword should be detected as valid"),
        other => panic!("expected boolean, got {:?}", other),
    }
}

#[test]
fn test_hamming_detect_alias() {
    // Known valid Hamming(7,4) codeword for data [1,0,1,1]
    let (_dir, _db, ex) = make_db("t62", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t62 COMPUTE ok = HAMMING_DETECT([0, 0, 0, 0, 0, 0, 0]) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ok") {
        Some(Value::Bool(b)) => assert!(*b, "all-zero codeword should be valid"),
        other => panic!("expected boolean, got {:?}", other),
    }
}

// ── Repetition Code ───────────────────────────────────────────────────────────

#[test]
fn test_rep_encode_basic() {
    // [1, 0] with n=3 -> [1, 1, 1, 0, 0, 0]
    let (_dir, _db, ex) = make_db("t63", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t63 COMPUTE enc = REPETITION_CODE_ENCODE([1, 0], 3) SELECT enc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("enc") {
        Some(Value::Array(a)) => {
            assert_eq!(
                a.len(),
                6,
                "REP(3) of 2 bits should give 6 bits, got {}",
                a.len()
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_rep_encode_alias() {
    let (_dir, _db, ex) = make_db("t64", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t64 COMPUTE enc = REP_ENCODE([1], 5) SELECT enc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("enc") {
        Some(Value::Array(a)) => {
            assert_eq!(
                a.len(),
                5,
                "REP(5) of 1 bit should give 5 bits, got {}",
                a.len()
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_rep_decode_majority() {
    // [1, 1, 0] -> majority 1
    let (_dir, _db, ex) = make_db("t65", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t65 COMPUTE dec = REPETITION_CODE_DECODE([1, 1, 0], 3) SELECT dec;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dec") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 1, "decode of 3-bit group should give 1 bit");
            match &a[0] {
                Value::Integer(i) => {
                    assert_eq!(*i, 1, "majority of [1,1,0] should be 1, got {}", i)
                }
                _ => panic!("expected integer"),
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_rep_decode_alias() {
    let (_dir, _db, ex) = make_db("t66", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t66 COMPUTE dec = REP_DECODE([0, 0, 1, 1, 1, 0], 3) SELECT dec;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dec") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2, "decode of two 3-bit groups should give 2 bits");
            // [0,0,1] -> 0, [1,1,0] -> 1
            match (&a[0], &a[1]) {
                (Value::Integer(b0), Value::Integer(b1)) => {
                    assert_eq!(*b0, 0, "first group [0,0,1] -> 0, got {}", b0);
                    assert_eq!(*b1, 1, "second group [1,1,0] -> 1, got {}", b1);
                }
                _ => panic!("expected integers"),
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── Extra edge-case tests ─────────────────────────────────────────────────────

#[test]
fn test_entropy_single_element() {
    // Single element -> entropy = 0
    let (_dir, _db, ex) = make_db("t67", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t67 COMPUTE h = SHANNON_ENTROPY([1.0]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 0.001,
            "single element entropy should be 0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_kl_divergence_non_negative() {
    // KL divergence is always >= 0
    let (_dir, _db, ex) = make_db("t68", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t68 COMPUTE d = KL_DIVERGENCE([0.3, 0.4, 0.3], [0.1, 0.8, 0.1]) SELECT d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f >= 0.0, "KL divergence should be >= 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_hellinger_bounded() {
    // Hellinger distance in [0, 1]
    let (_dir, _db, ex) = make_db("t69", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(
        r#"QUERY t69 COMPUTE h = HELLINGER_DISTANCE([0.1, 0.4, 0.5], [0.5, 0.3, 0.2]) SELECT h;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "Hellinger distance should be in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_gini_three_class_uniform() {
    // Uniform 3-class -> G = 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
    let (_dir, _db, ex) = make_db("t70", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t70 COMPUTE g = GINI_IMPURITY([1.0, 1.0, 1.0]) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let expected = 2.0 / 3.0;
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(
            (*f - expected).abs() < 0.001,
            "Gini of uniform 3-class should be 2/3, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_bsc_capacity_range() {
    // BSC capacity always in [0, 1]
    let (_dir, _db, ex) = make_db("t71", serde_json::json!({"dummy": 1}));
    let mut p = Parser::new(r#"QUERY t71 COMPUTE c = BSC_CAPACITY(0.1) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "BSC capacity should be in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_entropy_8bit_max() {
    // Uniform 8-outcome -> 3 bits
    let (_dir, _db, ex) = make_db("t72", serde_json::json!({"dummy": 1}));
    let mut p =
        Parser::new(r#"QUERY t72 COMPUTE h = SHANNON_ENTROPY([1, 1, 1, 1, 1, 1, 1, 1]) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 3.0).abs() < 0.001,
            "uniform 8-outcome should be 3 bits, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}
