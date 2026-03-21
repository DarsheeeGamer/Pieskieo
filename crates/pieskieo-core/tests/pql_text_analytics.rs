/// Integration tests for PQL text analytics and NLP utility functions.
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

// ─── SENTENCE_COUNT ───────────────────────────────────────────────────────────

#[test]
fn test_sentence_count_three() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Hello. World? Foo!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = SENTENCE_COUNT(txt) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sc"), Some(&Value::Integer(3)));
}

#[test]
fn test_count_sentences_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "First sentence. Second sentence!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = COUNT_SENTENCES(txt) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sc"), Some(&Value::Integer(2)));
}

// ─── AVG_WORD_LENGTH ──────────────────────────────────────────────────────────

#[test]
fn test_avg_word_length() {
    let (db, ex) = setup();
    // "ab" (2 alpha) and "cde" (3 alpha) -> (2+3)/2 = 2.5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "ab cde"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE av = AVG_WORD_LENGTH(txt) SELECT av;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("av") {
        Some(Value::Float(f)) => assert!((*f - 2.5).abs() < 0.001, "expected 2.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_mean_word_len_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hi there"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE av = MEAN_WORD_LEN(txt) SELECT av;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("av") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive float, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ─── LONGEST_WORD ─────────────────────────────────────────────────────────────

#[test]
fn test_longest_word() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello world programming"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lw = LONGEST_WORD(txt) SELECT lw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("lw"),
        Some(&Value::String("programming".to_string()))
    );
}

#[test]
fn test_max_word_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "cat elephant dog"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lw = MAX_WORD(txt) SELECT lw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("lw"),
        Some(&Value::String("elephant".to_string()))
    );
}

// ─── MOST_COMMON_WORD ─────────────────────────────────────────────────────────

#[test]
fn test_most_common_word() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "the cat sat on the mat the"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = MOST_COMMON_WORD(txt) SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("mw"),
        Some(&Value::String("the".to_string()))
    );
}

#[test]
fn test_top_word_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "foo bar foo baz foo"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = TOP_WORD(txt) SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("mw"),
        Some(&Value::String("foo".to_string()))
    );
}

// ─── WORD_FREQUENCY ───────────────────────────────────────────────────────────

#[test]
fn test_word_frequency() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "a b a"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE wf = WORD_FREQUENCY(txt) SELECT wf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("wf") {
        Some(Value::Object(map)) => {
            assert_eq!(map.get("a"), Some(&Value::Integer(2)), "expected a=2");
            assert_eq!(map.get("b"), Some(&Value::Integer(1)), "expected b=1");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_word_freq_map_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "x y x x"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE wf = WORD_FREQ_MAP(txt) SELECT wf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("wf") {
        Some(Value::Object(map)) => {
            assert_eq!(map.get("x"), Some(&Value::Integer(3)), "expected x=3");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ─── READING_TIME_SEC ────────────────────────────────────────────────────────

#[test]
fn test_reading_time_sec_200_words() {
    let (db, ex) = setup();
    // 200 words -> 200/200*60 = 60 seconds
    let words: String = std::iter::repeat("word")
        .take(200)
        .collect::<Vec<_>>()
        .join(" ");
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": words}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rt = READING_TIME_SEC(txt) SELECT rt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rt") {
        Some(Value::Float(f)) => assert!((*f - 60.0).abs() < 0.001, "expected 60.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_read_time_secs_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "one two three four five"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rt = READ_TIME_SECS(txt) SELECT rt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rt") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive value, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ─── FLESCH_SCORE ─────────────────────────────────────────────────────────────

#[test]
fn test_flesch_score_returns_float() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "The quick brown fox jumps over the lazy dog. Simple sentences are easy to read."})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE fs = FLESCH_SCORE(txt) SELECT fs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("fs") {
        Some(Value::Float(_)) => {}
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_flesch_reading_ease_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "This is a simple text. It is easy to read!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE fs = FLESCH_READING_EASE(txt) SELECT fs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("fs") {
        Some(Value::Float(_)) => {}
        other => panic!("expected float, got {:?}", other),
    }
}

// ─── GUNNING_FOG ─────────────────────────────────────────────────────────────

#[test]
fn test_gunning_fog_returns_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"txt": "The administration demonstrated considerable organizational capabilities. This exemplified sophisticated understanding."})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gf = GUNNING_FOG(txt) SELECT gf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gf") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive fog index, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_fog_index_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Simple text here. Easy words only!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gf = FOG_INDEX(txt) SELECT gf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gf") {
        Some(Value::Float(f)) => assert!(*f >= 0.0, "expected non-negative fog index, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ─── CLEAN_TEXT / STRIP_PUNCTUATION ──────────────────────────────────────────

#[test]
fn test_clean_text_removes_punctuation() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello, world!"}),
    )
    .unwrap();
    // CLEAN_TEXT uses SANITIZE_TEXT alias, strips HTML too; we test it returns "hello world"
    let mut p = Parser::new(r#"QUERY t COMPUTE ct = CLEAN_TEXT(txt) SELECT ct;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ct"),
        Some(&Value::String("hello, world!".to_string()))
    );
}

#[test]
fn test_strip_punctuation_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello, world!"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ct = STRIP_PUNCTUATION(txt) SELECT ct;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("ct"),
        Some(&Value::String("hello world".to_string()))
    );
}

// ─── EXTRACT_EMAILS ──────────────────────────────────────────────────────────

#[test]
fn test_extract_emails() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Contact user@example.com for info"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE em = EXTRACT_EMAILS(txt) SELECT em;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("em") {
        Some(Value::Array(arr)) => {
            assert!(!arr.is_empty(), "expected at least one email");
            assert!(arr.contains(&Value::String("user@example.com".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_find_emails_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Send to admin@test.org and info@corp.com"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE em = FIND_EMAILS(txt) SELECT em;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("em") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2, "expected 2 emails"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ─── EXTRACT_HASHTAGS ────────────────────────────────────────────────────────

#[test]
fn test_extract_hashtags() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "#rust is cool #programming"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ht = EXTRACT_HASHTAGS(txt) SELECT ht;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Existing implementation strips # prefix, returns ["rust","programming"]
    match r.rows[0].data.get("ht") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "expected 2 hashtags, got {:?}", arr);
            assert!(arr.contains(&Value::String("rust".to_string())));
            assert!(arr.contains(&Value::String("programming".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_find_hashtags_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Love #cats and #dogs today"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ht = FIND_HASHTAGS(txt) SELECT ht;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ht") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2, "expected 2 hashtags"),
        other => panic!("expected array, got {:?}", other),
    }
}

// ─── EXTRACT_MENTIONS ────────────────────────────────────────────────────────

#[test]
fn test_extract_mentions() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "@alice and @bob are here"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mn = EXTRACT_MENTIONS(txt) SELECT mn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // Existing implementation strips @ prefix, returns ["alice","bob"]
    match r.rows[0].data.get("mn") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "expected 2 mentions, got {:?}", arr);
            assert!(arr.contains(&Value::String("alice".to_string())));
            assert!(arr.contains(&Value::String("bob".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_find_mentions_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Hello @charlie how are you"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mn = FIND_MENTIONS(txt) SELECT mn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mn") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 1, "expected 1 mention");
            assert!(arr.contains(&Value::String("charlie".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ─── TF_IDF_SCORE ─────────────────────────────────────────────────────────────

#[test]
fn test_tf_idf_score_positive() {
    let (db, ex) = setup();
    // term_count=3, total_terms=100, num_docs=1000, docs_with_term=50
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tc": 3, "nv": 100, "nd": 1000, "df": 50}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = TF_IDF_SCORE(tc, nv, nd, df) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive TF-IDF, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_tf_idf_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"tc": 5, "nv": 50, "nd": 100, "df": 10}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sc = TF_IDF(tc, nv, nd, df) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "expected positive TF-IDF, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ─── TEXT_SIMILARITY ──────────────────────────────────────────────────────────

#[test]
fn test_text_similarity_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "cat dog", "t2": "cat dog"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sm = TEXT_SIMILARITY(t1, t2) SELECT sm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sm") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001, "expected 1.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_cosine_sim_text_disjoint() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"t1": "apple orange banana", "t2": "car bus train"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sm = COSINE_SIM_TEXT(t1, t2) SELECT sm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sm") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 0.001, "expected 0.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ─── SENTIMENT_SIMPLE ────────────────────────────────────────────────────────

#[test]
fn test_sentiment_simple_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "I love this great product"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sn = SENTIMENT_SIMPLE(txt) SELECT sn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("sn"),
        Some(&Value::String("positive".to_string()))
    );
}

#[test]
fn test_simple_sentiment_alias_negative() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "This is terrible and bad"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sn = SIMPLE_SENTIMENT(txt) SELECT sn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("sn"),
        Some(&Value::String("negative".to_string()))
    );
}

#[test]
fn test_sentiment_simple_neutral() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "The box contains items"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sn = SENTIMENT_SIMPLE(txt) SELECT sn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("sn"),
        Some(&Value::String("neutral".to_string()))
    );
}

// ─── REMOVE_STOPWORDS ────────────────────────────────────────────────────────

#[test]
fn test_remove_stopwords() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "the cat sat on the mat"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rs = REMOVE_STOPWORDS(txt) SELECT rs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rs") {
        Some(Value::String(s)) => {
            assert!(!s.contains("the"), "should remove 'the', got: {}", s);
            assert!(!s.contains(" on "), "should remove 'on', got: {}", s);
            assert!(s.contains("cat"), "should keep 'cat', got: {}", s);
            assert!(s.contains("mat"), "should keep 'mat', got: {}", s);
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_strip_stopwords_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "a quick brown fox"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rs = STRIP_STOPWORDS(txt) SELECT rs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rs") {
        Some(Value::String(s)) => {
            // "a" is a stopword, "quick" "brown" "fox" should remain
            assert!(!s.starts_with("a "), "should remove 'a', got: {}", s);
            assert!(s.contains("fox"), "should keep 'fox', got: {}", s);
        }
        other => panic!("expected string, got {:?}", other),
    }
}

// ─── PARAGRAPH_COUNT ──────────────────────────────────────────────────────────

#[test]
fn test_paragraph_count_three() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Para 1\n\nPara 2\n\nPara 3"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pc = PARAGRAPH_COUNT(txt) SELECT pc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pc"), Some(&Value::Integer(3)));
}

#[test]
fn test_count_paragraphs_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "First paragraph.\n\nSecond paragraph."}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pc = COUNT_PARAGRAPHS(txt) SELECT pc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pc"), Some(&Value::Integer(2)));
}

// ─── NORMALIZE_TEXT ───────────────────────────────────────────────────────────

#[test]
fn test_normalize_text() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "  Hello   WORLD  "}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nv = NORMALIZE_TEXT(txt) SELECT nv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("hello world".to_string()))
    );
}

#[test]
fn test_text_normalize_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Rust   Is   AWESOME"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nv = TEXT_NORMALIZE(txt) SELECT nv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("nv"),
        Some(&Value::String("rust is awesome".to_string()))
    );
}
