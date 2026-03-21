/// Integration tests for PQL fuzzy matching and data quality functions.
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

#[test]
fn test_jaro() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "MARTHA", "b": "MARHTA"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO(a, b) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(*f > 0.9, "JARO similarity should be high, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_jaro_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "hello", "b": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO(a, b) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "identical strings should give 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_jaro_distance_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "CRATE", "b": "TRACE"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = JARO_DISTANCE(a, b) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("j") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "JARO_DISTANCE should be in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_similarity_ratio() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "kitten", "b": "kitten"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = SIMILARITY_RATIO(a, b) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "identical strings: similarity should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_similarity_ratio_partial() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "kitten", "b": "sitting"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = SIMILARITY_RATIO(a, b) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && *f < 1.0,
            "partial similarity expected between 0 and 1, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_str_similarity_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "abc", "b": "xyz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = STR_SIMILARITY(a, b) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f <= 1.0,
            "STR_SIMILARITY should be in [0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_damerau_levenshtein_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "hello", "b": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DAMERAU_LEVENSHTEIN(a, b) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 0, "identical strings: distance should be 0, got {}", i)
        }
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_damerau_levenshtein_transposition() {
    let (db, ex) = setup();
    // "ab" -> "ba" is 1 transposition, Damerau-Levenshtein = 1
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "ab", "b": "ba"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DAMERAU_LEVENSHTEIN(a, b) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 1,
            "one transposition: DL distance should be 1, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_dl_distance_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "cat", "b": "dog"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = DL_DISTANCE(a, b) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert!(
            *i > 0,
            "different strings should have DL_DISTANCE > 0, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_normalize_str() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "  Hello, World!  "}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = NORMALIZE_STR(s) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::String(s)) => {
            assert_eq!(
                s, "hello world",
                "NORMALIZE_STR should lowercase and remove punctuation, got '{}'",
                s
            );
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_normalize_str_collapse_spaces() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "foo   bar   baz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = NORMALIZE_STR(s) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(Value::String(s)) => {
            assert_eq!(
                s, "foo bar baz",
                "NORMALIZE_STR should collapse spaces, got '{}'",
                s
            );
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_ngram_similarity_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "abcde", "b": "abcde"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = NGRAM_SIMILARITY(a, b, 2) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "identical strings: ngram similarity should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_ngram_similarity_different() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "abcde", "b": "vwxyz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = NGRAM_JACCARD(a, b, 2) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f < 0.5,
            "very different strings: ngram similarity should be low, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_is_similar_to_true() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "hello", "b": "hello"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_SIMILAR_TO(a, b, 0.9) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(*b, "identical strings should be similar at threshold 0.9"),
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_is_similar_to_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "hello", "b": "world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_SIMILAR_TO(a, b, 0.99) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(
            !b,
            "very different strings should not be similar at threshold 0.99"
        ),
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_fuzzy_equal_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "kitten", "b": "sitten"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FUZZY_EQUAL(a, b, 0.8) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(_)) => {} // just check it runs
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_levenshtein_normalized_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "test", "b": "test"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = LEVENSHTEIN_NORMALIZED(a, b) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(
            (*f).abs() < 1e-9,
            "identical strings: normalized distance should be 0.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_levenshtein_normalized_different() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "abc", "b": "xyz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = NORMALIZED_LEVENSHTEIN(a, b) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => {
            assert!(
                *f > 0.0 && *f <= 1.0,
                "different strings: normalized distance should be in (0,1], got {}",
                f
            );
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_metaphone_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "Smith"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = METAPHONE(s) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::String(s)) => assert!(
            !s.is_empty(),
            "METAPHONE should return non-empty string for 'Smith', got '{}'",
            s
        ),
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_metaphone_code_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s": "Knight"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = METAPHONE_CODE(s) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::String(s)) => assert!(
            !s.is_empty(),
            "METAPHONE_CODE should return non-empty string, got '{}'",
            s
        ),
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_fuzzy_contains_exact() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"h": "hello world", "n": "world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FUZZY_CONTAINS(h, n, 0) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(*b, "exact match should be found with 0 errors"),
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_fuzzy_contains_with_errors() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"h": "hello world", "n": "wrold"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = APPROX_CONTAINS(h, n, 2) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Bool(b)) => assert!(
            *b,
            "fuzzy match with 2 errors should find 'wrold' in 'hello world'"
        ),
        other => panic!("expected bool, got {:?}", other),
    }
}

#[test]
fn test_string_overlap_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "abcde", "b": "abcde"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = STRING_OVERLAP(a, b, 2) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "identical strings: overlap should be 1.0, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_overlap_coefficient_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "abcde", "b": "cdefg"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = OVERLAP_COEFFICIENT(a, b, 2) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0 && *f <= 1.0,
            "partial overlap should be in (0,1], got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_most_similar() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"target": "hello", "candidates": ["world", "hello!", "hi"]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE idx = MOST_SIMILAR(target, candidates) SELECT idx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("idx") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 1,
            "most similar to 'hello' should be index 1 ('hello!'), got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_best_match_index_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"target": "cat", "candidates": ["dog", "bat", "cat"]}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE idx = BEST_MATCH_INDEX(target, candidates) SELECT idx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("idx") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 2,
            "most similar to 'cat' should be index 2 ('cat'), got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_approximate_distinct() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": ["a", "b", "a", "c", "b", "a"]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = APPROXIMATE_DISTINCT(arr) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "distinct count of ['a','b','a','c','b','a'] should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}

#[test]
fn test_approx_count_distinct_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"arr": [1, 2, 3, 2, 1]}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = APPROX_COUNT_DISTINCT(arr) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 3,
            "distinct count of [1,2,3,2,1] should be 3, got {}",
            i
        ),
        other => panic!("expected integer, got {:?}", other),
    }
}
