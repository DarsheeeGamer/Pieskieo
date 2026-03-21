/// Integration tests for PQL fuzzy string matching and string distance metric functions.
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

// ── 1. LEVENSHTEIN("hello", "hello") → 0 ──────────────────────────────────
#[test]
fn test_levenshtein_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello", "s2": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = LEVENSHTEIN(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 0,
            "LEVENSHTEIN of identical strings should be 0, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 2. LEVENSHTEIN("", "abc") → 3 ─────────────────────────────────────────
#[test]
fn test_levenshtein_empty_vs_abc() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "", "s2": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = LEVENSHTEIN(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 3, "LEVENSHTEIN('', 'abc') should be 3, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 3. LEVENSHTEIN("kitten", "sitting") → 3 ───────────────────────────────
#[test]
fn test_levenshtein_kitten_sitting() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "kitten", "s2": "sitting"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = LEVENSHTEIN(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "LEVENSHTEIN('kitten','sitting') should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 4. EDIT_DISTANCE alias ─────────────────────────────────────────────────
#[test]
fn test_edit_distance_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "kitten", "s2": "sitting"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = EDIT_DISTANCE(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "EDIT_DISTANCE('kitten','sitting') should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 5. DAMERAU_LEVENSHTEIN("ca", "abc") <= LEVENSHTEIN("ca", "abc") ────────
#[test]
fn test_damerau_levenshtein_lte_levenshtein() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "ca", "s2": "abc"}),
    )
    .unwrap();
    let mut p_dl = Parser::new(r#"QUERY t COMPUTE d = DAMERAU_LEVENSHTEIN(s1, s2) SELECT d;"#);
    let mut p_lv = Parser::new(r#"QUERY t COMPUTE d = LEVENSHTEIN(s1, s2) SELECT d;"#);
    let r_dl = ex.execute(p_dl.parse().unwrap()).unwrap();
    let r_lv = ex.execute(p_lv.parse().unwrap()).unwrap();
    let dl = match r_dl.rows[0].data.get("d") {
        Some(Value::Integer(i)) => *i,
        other => panic!("expected integer, got {:?}", other),
    };
    let lv = match r_lv.rows[0].data.get("d") {
        Some(Value::Integer(i)) => *i,
        other => panic!("expected integer, got {:?}", other),
    };
    assert!(
        dl <= lv,
        "DAMERAU_LEVENSHTEIN should be <= LEVENSHTEIN, got dl={} lv={}",
        dl,
        lv
    );
}

// ── 6. DAMERAU_DIST alias ─────────────────────────────────────────────────
#[test]
fn test_damerau_dist_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "ab", "s2": "ba"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DAMERAU_DIST(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 1,
            "DAMERAU_DIST('ab','ba') should be 1 (one transposition), got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 7. JARO("martha", "marhta") → ≈0.944 ─────────────────────────────────
#[test]
fn test_jaro_martha_marhta() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "martha", "s2": "marhta"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.944).abs() < 0.01,
            "JARO('martha','marhta') should be ≈0.944, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 8. JARO_SIMILARITY alias (maps to JARO_WINKLER in existing code) ──────
#[test]
fn test_jaro_similarity_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "martha", "s2": "marhta"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO_SIMILARITY(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "JARO_SIMILARITY should return value in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 9. JARO("", "") → 1.0 ─────────────────────────────────────────────────
#[test]
fn test_jaro_both_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "", "s2": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "JARO('','') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 10. JARO_WINKLER("john", "john") → 1.0 ────────────────────────────────
#[test]
fn test_jaro_winkler_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "john", "s2": "john"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO_WINKLER(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "JARO_WINKLER('john','john') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 11. JARO_WINKLER("john", "jane") → between 0 and 1 ───────────────────
#[test]
fn test_jaro_winkler_partial() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "john", "s2": "jane"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO_WINKLER(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && *f < 1.0,
            "JARO_WINKLER('john','jane') should be between 0 and 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 12. JARO_WINKLER_SIM alias ─────────────────────────────────────────────
#[test]
fn test_jaro_winkler_sim_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "john", "s2": "john"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO_WINKLER_SIM(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "JARO_WINKLER_SIM('john','john') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 13. NORMALIZED_EDIT_DIST("abc", "abc") → 0.0 ─────────────────────────
#[test]
fn test_normalized_edit_dist_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc", "s2": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = NORMALIZED_EDIT_DIST(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f).abs() < 1e-9,
            "NORMALIZED_EDIT_DIST of identical strings should be 0.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 14. NORMALIZED_EDIT_DIST("abc", "") → 1.0 ────────────────────────────
#[test]
fn test_normalized_edit_dist_empty() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc", "s2": ""}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = NORMALIZED_EDIT_DIST(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "NORMALIZED_EDIT_DIST('abc','') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 15. NORMALIZED_LEVENSHTEIN alias (existing function) ──────────────────
#[test]
fn test_normalized_levenshtein_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc", "s2": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = NORMALIZED_LEVENSHTEIN(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f).abs() < 1e-9,
            "NORMALIZED_LEVENSHTEIN of identical strings should be 0.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 16. LCS_LENGTH("abcde", "ace") → 3 ───────────────────────────────────
#[test]
fn test_lcs_length_abcde_ace() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abcde", "s2": "ace"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LCS_LENGTH(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 3, "LCS_LENGTH('abcde','ace') should be 3, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 17. LCS_LENGTH("abc", "abc") → 3 ─────────────────────────────────────
#[test]
fn test_lcs_length_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc", "s2": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LCS_LENGTH(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 3, "LCS_LENGTH('abc','abc') should be 3, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 18. LONGEST_COMMON_SUBSEQ alias ───────────────────────────────────────
#[test]
fn test_longest_common_subseq_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abcde", "s2": "ace"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LONGEST_COMMON_SUBSEQ(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "LONGEST_COMMON_SUBSEQ('abcde','ace') should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 19. LONGEST_COMMON_SUBSTR("abcdef", "bcd") → 3 ───────────────────────
#[test]
fn test_longest_common_substr_bcd() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abcdef", "s2": "bcd"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LONGEST_COMMON_SUBSTR(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "LONGEST_COMMON_SUBSTR('abcdef','bcd') should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 20. LONGEST_COMMON_SUBSTR("abcdef", "xyz") → 0 ───────────────────────
#[test]
fn test_longest_common_substr_no_match() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abcdef", "s2": "xyz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LONGEST_COMMON_SUBSTR(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 0,
            "LONGEST_COMMON_SUBSTR('abcdef','xyz') should be 0, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 21. LCSUBSTR_LENGTH alias ─────────────────────────────────────────────
#[test]
fn test_lcsubstr_length_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abcdef", "s2": "bcd"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LCSUBSTR_LENGTH(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "LCSUBSTR_LENGTH('abcdef','bcd') should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

// ── 22. FUZZY_SCORE("hello", "hello") → 1.0 ──────────────────────────────
#[test]
fn test_fuzzy_score_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello", "s2": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = FUZZY_SCORE(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "FUZZY_SCORE('hello','hello') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 23. FUZZY_SCORE("hello", "world") → between 0 and 1 ──────────────────
#[test]
fn test_fuzzy_score_different() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello", "s2": "world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = FUZZY_SCORE(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && *f < 1.0,
            "FUZZY_SCORE('hello','world') should be between 0 and 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 24. FUZZY_MATCH_SCORE alias ───────────────────────────────────────────
#[test]
fn test_fuzzy_match_score_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello", "s2": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = FUZZY_MATCH_SCORE(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "FUZZY_MATCH_SCORE('hello','hello') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 25. TOKEN_SORT_RATIO("foo bar", "bar foo") → 1.0 ─────────────────────
#[test]
fn test_token_sort_ratio_reordered() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "foo bar", "s2": "bar foo"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = TOKEN_SORT_RATIO(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "TOKEN_SORT_RATIO('foo bar','bar foo') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 26. TOKEN_SORT_RATIO("abc def", "xyz uvw") → low score ───────────────
#[test]
fn test_token_sort_ratio_different() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc def", "s2": "xyz uvw"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = TOKEN_SORT_RATIO(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            *f < 0.5,
            "TOKEN_SORT_RATIO of very different strings should be low, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 27. SORTED_TOKEN_SIMILARITY alias ────────────────────────────────────
#[test]
fn test_sorted_token_similarity_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "foo bar", "s2": "bar foo"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = SORTED_TOKEN_SIMILARITY(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "SORTED_TOKEN_SIMILARITY('foo bar','bar foo') should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── 28. CONTAINS_FUZZY("hello world", "wrold", 2) → true ─────────────────
#[test]
fn test_contains_fuzzy_typo() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"hay": "hello world", "ndl": "wrold", "thresh": 2}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONTAINS_FUZZY(hay, ndl, thresh) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(
            *b,
            "CONTAINS_FUZZY('hello world','wrold',2) should be true (typo within 2 edits)"
        ),
        other => panic!("expected bool, got {:?}", other),
    }
}

// ── 29. CONTAINS_FUZZY("hello world", "xyz", 0) → false ──────────────────
#[test]
fn test_contains_fuzzy_no_match() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"hay": "hello world", "ndl": "xyz", "thresh": 0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONTAINS_FUZZY(hay, ndl, thresh) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => {
            assert!(!*b, "CONTAINS_FUZZY('hello world','xyz',0) should be false")
        }
        other => panic!("expected bool, got {:?}", other),
    }
}

// ── 30. FUZZY_CONTAINS alias ──────────────────────────────────────────────
#[test]
fn test_fuzzy_contains_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"hay": "hello world", "ndl": "world", "thresh": 0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FUZZY_CONTAINS(hay, ndl, thresh) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(
            *b,
            "FUZZY_CONTAINS('hello world','world',0) should be true (exact match)"
        ),
        other => panic!("expected bool, got {:?}", other),
    }
}

// ── Extra tests to ensure robustness ──────────────────────────────────────

#[test]
fn test_levenshtein_abc_xyz() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc", "s2": "xyz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = LEVENSHTEIN(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 3, "LEVENSHTEIN('abc','xyz') should be 3, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_damerau_dist_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello", "s2": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DAMERAU_DIST(s1, s2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 0,
            "DAMERAU_DIST of identical strings should be 0, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_jaro_winkler_sim_partial() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "john", "s2": "jane"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO_WINKLER_SIM(s1, s2) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "JARO_WINKLER_SIM should be in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_lcs_length_no_common() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "aaa", "s2": "bbb"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LCS_LENGTH(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 0, "LCS_LENGTH('aaa','bbb') should be 0, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_fuzzy_score_case_insensitive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "Hello", "s2": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = FUZZY_SCORE(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "FUZZY_SCORE should be case-insensitive, 'Hello'/'hello' should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_token_sort_ratio_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "hello world", "s2": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = TOKEN_SORT_RATIO(s1, s2) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "TOKEN_SORT_RATIO of identical strings should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_contains_fuzzy_exact_match() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"hay": "hello world", "ndl": "hello", "thresh": 0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CONTAINS_FUZZY(hay, ndl, thresh) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(*b, "CONTAINS_FUZZY with exact match should return true"),
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_longest_common_substr_full_overlap() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "abc", "s2": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE l = LONGEST_COMMON_SUBSTR(s1, s2) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "LONGEST_COMMON_SUBSTR('abc','abc') should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}
