/// Integration tests for PQL statistical hypothesis testing functions.
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

// ── CHI_SQUARED_STAT ──────────────────────────────────────────────────────────

#[test]
fn test_chi_squared_stat_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE s = CHI_SQUARED_STAT([10.0, 20.0, 30.0], [15.0, 15.0, 30.0]) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.0, "chi-squared stat should be positive, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_chi_squared_stat_perfect_fit() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // When observed == expected, stat should be 0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE s = CHI_SQUARED_STAT([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "perfect fit should give 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_chi_squared_stat_misfit_greater_than_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Large difference from expected => large statistic
    let mut p = Parser::new(
        r#"QUERY t COMPUTE s = CHI_SQUARED_STAT([50.0, 5.0, 5.0], [20.0, 20.0, 20.0]) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 1.0, "misfit should produce stat > 1, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_chi_sq_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = CHI_SQ([10.0, 20.0], [15.0, 15.0]) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0,
            "CHI_SQ alias should produce positive stat, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── T_STAT ────────────────────────────────────────────────────────────────────

#[test]
fn test_t_stat_mean_equals_mu() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Values centered at 0, mu=0 => t should be ~0
    let mut p = Parser::new(r#"QUERY t COMPUTE s = T_STAT([-1.0, 0.0, 1.0], 0.0) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => {
            assert!(f.abs() < 1e-10, "mean equals mu, t should be ~0, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_t_stat_large_deviation_from_mu() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Values all at 100, mu=0 => large t
    let mut p = Parser::new(r#"QUERY t COMPUTE s = T_STAT([100.0, 100.0, 100.0], 0.0) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Null) => { /* all same value means std=0, returns Null */ }
        Some(Value::Float(f)) => assert!(
            f.abs() > 1.0,
            "large deviation should give large t, got {}",
            f
        ),
        other => panic!("expected float or null, got {:?}", other),
    }
}

#[test]
fn test_t_stat_nonzero_mu() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE s = T_STAT([1.0, 2.0, 3.0, 4.0, 5.0], 3.0) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "mean=3.0, mu=3.0 => t~0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_one_sample_t_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = ONE_SAMPLE_T([1.0, 2.0, 3.0], 0.0) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "ONE_SAMPLE_T alias should work, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── WELCH_T_STAT ──────────────────────────────────────────────────────────────

#[test]
fn test_welch_t_stat_same_distributions() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Same distributions => t ~0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE s = WELCH_T_STAT([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "same distributions => t~0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_welch_t_stat_different_means() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Very different means => large |t|
    let mut p = Parser::new(
        r#"QUERY t COMPUTE s = WELCH_T_STAT([1.0, 2.0, 3.0], [100.0, 101.0, 102.0]) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(f.abs() > 1.0, "different means => |t|>1, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_two_sample_t_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE s = TWO_SAMPLE_T([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) SELECT s;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => {
            assert!(f.abs() > 0.0, "TWO_SAMPLE_T alias should work, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── MANN_WHITNEY_U ────────────────────────────────────────────────────────────

#[test]
fn test_mann_whitney_u_no_wins() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // g1=[1,2] vs g2=[3,4]: every g1 element is less than every g2 element => U=0
    let mut p =
        Parser::new(r#"QUERY t COMPUTE u = MANN_WHITNEY_U([1.0, 2.0], [3.0, 4.0]) SELECT u;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("u") {
        Some(Value::Integer(i)) => assert_eq!(*i, 0, "g1=[1,2] vs g2=[3,4] => U=0, got {}", i),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_mann_whitney_u_all_wins() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // g1=[10,20] vs g2=[1,2]: g1 wins all comparisons => U = 2*2 = 4
    let mut p =
        Parser::new(r#"QUERY t COMPUTE u = MANN_WHITNEY_U([10.0, 20.0], [1.0, 2.0]) SELECT u;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("u") {
        Some(Value::Integer(i)) => assert_eq!(*i, 4, "g1=[10,20] vs g2=[1,2] => U=4, got {}", i),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_wilcoxon_u_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE u = WILCOXON_U([1.0, 2.0], [3.0, 4.0]) SELECT u;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("u") {
        Some(Value::Integer(i)) => assert_eq!(*i, 0, "WILCOXON_U alias: g1<g2 => U=0, got {}", i),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── BINOMIAL_TEST_P ───────────────────────────────────────────────────────────

#[test]
fn test_binomial_test_p_center() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // k = n/2, p = 0.5 => p-value should be near 1 (not significant)
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = BINOMIAL_TEST_P(50, 100, 0.5) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pv") {
        Some(Value::Float(f)) => assert!(
            *f > 0.5,
            "center case should give p-value near 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_binomial_test_p_extreme() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // k = 100 (all successes), p = 0.5, n = 100 => very significant => small p-value
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = BINOMIAL_TEST_P(100, 100, 0.5) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pv") {
        Some(Value::Float(f)) => assert!(
            *f < 0.01,
            "extreme case should give small p-value, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_exact_binom_p_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pv = EXACT_BINOM_P(5, 10, 0.5) SELECT pv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pv") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "EXACT_BINOM_P should return valid p-value [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── F_STAT ────────────────────────────────────────────────────────────────────

#[test]
fn test_f_stat_same_means() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // All groups have same mean => F should be ~0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE f = F_STAT([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]) SELECT f;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "same means => F~0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_f_stat_different_means() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Very different group means => F > 1
    let mut p = Parser::new(
        r#"QUERY t COMPUTE f = F_STAT([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]) SELECT f;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Float(f)) => assert!(*f > 1.0, "different means => F>1, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_anova_f_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE f = ANOVA_F([[1.0, 2.0], [5.0, 6.0]]) SELECT f;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("f") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0,
            "ANOVA_F alias should work with different means, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── KOLMOGOROV_SMIRNOV ────────────────────────────────────────────────────────

#[test]
fn test_ks_stat_identical_distributions() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Identical distributions => KS = 0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE k = KS_STAT([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) SELECT k;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("k") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-10,
            "identical distributions => KS=0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_ks_stat_different_distributions() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Very different distributions => KS > 0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE k = KS_STAT([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) SELECT k;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("k") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "different distributions => KS>0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_kolmogorov_smirnov_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE k = KOLMOGOROV_SMIRNOV([1.0, 2.0], [5.0, 6.0]) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("k") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.0, "KOLMOGOROV_SMIRNOV alias should work, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── FISHER_EXACT_ODDS ─────────────────────────────────────────────────────────

#[test]
fn test_fisher_exact_odds_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // [a=6, b=2, c=2, d=6] => OR = (6*6)/(2*2) = 36/4 = 9.0
    let mut p =
        Parser::new(r#"QUERY t COMPUTE o = FISHER_EXACT_ODDS([6.0, 2.0, 2.0, 6.0]) SELECT o;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("o") {
        Some(Value::Float(f)) => assert!((*f - 9.0).abs() < 1e-10, "OR should be 9.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_fisher_exact_odds_unit_ratio() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // [a=1, b=1, c=1, d=1] => OR = (1*1)/(1*1) = 1.0
    let mut p =
        Parser::new(r#"QUERY t COMPUTE o = FISHER_EXACT_ODDS([1.0, 1.0, 1.0, 1.0]) SELECT o;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("o") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-10, "OR should be 1.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_fisher_odds_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE o = FISHER_ODDS([4.0, 2.0, 1.0, 8.0]) SELECT o;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("o") {
        // OR = (4*8)/(2*1) = 32/2 = 16.0
        Some(Value::Float(f)) => assert!(
            (*f - 16.0).abs() < 1e-10,
            "FISHER_ODDS alias: OR should be 16.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── EFFECT_SIZE_D ─────────────────────────────────────────────────────────────

#[test]
fn test_effect_size_d_same_group() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Identical groups => d = 0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE d = EFFECT_SIZE_D([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) SELECT d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "same groups => d=0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_effect_size_d_different_means() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // Different means => d != 0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE d = EFFECT_SIZE_D([1.0, 2.0, 3.0], [10.0, 11.0, 12.0]) SELECT d;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f.abs() > 1.0, "different means => |d|>1, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_cohens_d_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE d = COHENS_D([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(f.abs() > 0.0, "COHENS_D alias should work, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── POWER_ANALYSIS ────────────────────────────────────────────────────────────

#[test]
fn test_power_analysis_medium_effect() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // effect_size=0.5, alpha=0.05, power=0.8 => ~63 per group (Cohen's rule of thumb)
    let mut p = Parser::new(r#"QUERY t COMPUTE n = POWER_ANALYSIS(0.5, 0.05, 0.8) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Integer(i)) => assert!(
            *i > 10 && *i < 500,
            "medium effect: sample size should be reasonable, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_power_analysis_large_effect_smaller_n() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    // large effect_size=1.0 => fewer samples needed than effect_size=0.5
    let mut p_large =
        Parser::new(r#"QUERY t COMPUTE n = POWER_ANALYSIS(1.0, 0.05, 0.8) SELECT n;"#);
    let mut p_medium =
        Parser::new(r#"QUERY t COMPUTE n = POWER_ANALYSIS(0.5, 0.05, 0.8) SELECT n;"#);
    let r_large = ex.execute(p_large.parse().unwrap()).unwrap();
    let r_medium = ex.execute(p_medium.parse().unwrap()).unwrap();
    let n_large = match r_large.rows[0].data.get("n") {
        Some(Value::Integer(i)) => *i,
        other => panic!("expected integer for large effect, got {:?}", other),
    };
    let n_medium = match r_medium.rows[0].data.get("n") {
        Some(Value::Integer(i)) => *i,
        other => panic!("expected integer for medium effect, got {:?}", other),
    };
    assert!(
        n_large < n_medium,
        "larger effect should need fewer samples: {} < {}",
        n_large,
        n_medium
    );
}

#[test]
fn test_sample_size_est_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = SAMPLE_SIZE_EST(0.5, 0.05, 0.8) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::Integer(i)) => assert!(
            *i > 0,
            "SAMPLE_SIZE_EST alias should return positive integer, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}
