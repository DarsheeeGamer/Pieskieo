/// Integration tests for PQL text analysis, classification, and heuristic NLP functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_sentiment_score_positive() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "This is great and wonderful and awesome"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = SENTIMENT_SCORE(txt) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f > 0.0,
            "positive text should have positive score, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_sentiment_score_negative() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "This is terrible and horrible and awful"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE s = POLARITY_SCORE(txt) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            *f < 0.0,
            "negative text should have negative score, got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_language_detect_english() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "the quick brown fox and the dog are running"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lang = LANGUAGE_DETECT(txt) SELECT lang;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("lang"),
        Some(&Value::String("en".to_string()))
    );
}

#[test]
fn test_language_detect_french() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "le chat et la souris sont dans le jardin"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lang = DETECT_LANGUAGE(txt) SELECT lang;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("lang"),
        Some(&Value::String("fr".to_string()))
    );
}

#[test]
fn test_extract_emails() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Contact us at hello@example.com or support@foo.org for help"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE emails = EXTRACT_EMAILS(txt) SELECT emails;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("emails") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "expected 2 emails, got: {:?}", arr);
            assert!(arr.contains(&Value::String("hello@example.com".to_string())));
            assert!(arr.contains(&Value::String("support@foo.org".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_extract_urls() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "Visit https://www.example.com and http://foo.org/page for more"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE urls = EXTRACT_URLS(txt) SELECT urls;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("urls") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "expected 2 URLs, got: {:?}", arr);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_extract_numbers() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "The price is 42.50 and quantity is 3"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nums = EXTRACT_NUMBERS(txt) SELECT nums;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nums") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "expected 2 numbers, got: {:?}", arr);
            assert!(arr.contains(&Value::Float(42.5)));
            assert!(arr.contains(&Value::Float(3.0)));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_extract_hashtags() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "#hello world #foo_bar check #rust2024"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tags = EXTRACT_HASHTAGS(txt) SELECT tags;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tags") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "expected 3 hashtags, got: {:?}", arr);
            assert!(arr.contains(&Value::String("hello".to_string())));
            assert!(arr.contains(&Value::String("foo_bar".to_string())));
            assert!(arr.contains(&Value::String("rust2024".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_extract_mentions() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Hey @alice and @bob_smith, how are you?"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mentions = EXTRACT_MENTIONS(txt) SELECT mentions;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mentions") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2, "expected 2 mentions, got: {:?}", arr);
            assert!(arr.contains(&Value::String("alice".to_string())));
            assert!(arr.contains(&Value::String("bob_smith".to_string())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_extract_phone_numbers() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Call us at (555) 123-4567 or 800-555-1234"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE phones = EXTRACT_PHONE_NUMBERS(txt) SELECT phones;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("phones") {
        Some(Value::Array(arr)) => {
            assert!(
                !arr.is_empty(),
                "expected at least one phone number, got: {:?}",
                arr
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_remove_stopwords() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "the quick brown fox is jumping over a lazy dog"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE filtered = REMOVE_STOPWORDS(txt) SELECT filtered;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("filtered") {
        Some(Value::String(s)) => {
            assert!(
                !s.contains("the "),
                "stopword 'the' should be removed, got: {}",
                s
            );
            assert!(
                !s.contains(" is "),
                "stopword 'is' should be removed, got: {}",
                s
            );
            assert!(
                s.contains("quick"),
                "content word 'quick' should remain, got: {}",
                s
            );
            assert!(
                s.contains("fox"),
                "content word 'fox' should remain, got: {}",
                s
            );
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_keyword_extract() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"txt": "rust programming rust language rust code programming language code"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kws = KEYWORD_EXTRACT(txt, 3) SELECT kws;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kws") {
        Some(Value::Array(arr)) => {
            assert!(
                arr.len() <= 3,
                "expected at most 3 keywords, got: {:?}",
                arr
            );
            assert!(!arr.is_empty(), "should return some keywords");
            // "rust" appears 3 times, should be top keyword
            assert!(
                arr.contains(&Value::String("rust".to_string())),
                "rust should be a top keyword, got: {:?}",
                arr
            );
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_abbreviate() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Natural Language Processing"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE abbr = ABBREVIATE(txt) SELECT abbr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("abbr"),
        Some(&Value::String("NLP".to_string()))
    );
}

#[test]
fn test_expand_contractions() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "I don't know what he's doing"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE expanded = EXPAND_CONTRACTIONS(txt) SELECT expanded;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("expanded") {
        Some(Value::String(s)) => {
            assert!(
                s.contains("do not"),
                "don't should expand to 'do not', got: {}",
                s
            );
            assert!(
                s.contains("he is"),
                "he's should expand to 'he is', got: {}",
                s
            );
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_clean_text() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "<p>Hello  <b>world</b>!   This is   clean.</p>"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cleaned = CLEAN_TEXT(txt) SELECT cleaned;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cleaned") {
        Some(Value::String(s)) => {
            assert!(!s.contains('<'), "HTML tags should be removed, got: {}", s);
            assert!(!s.contains('>'), "HTML tags should be removed, got: {}", s);
            assert!(
                !s.contains("  "),
                "Multiple spaces should be normalized, got: {}",
                s
            );
            assert!(s.contains("Hello"), "content should remain, got: {}", s);
            assert!(s.contains("world"), "content should remain, got: {}", s);
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_count_tokens() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 10 words * 1.3 = 13 tokens
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "one two three four five six seven eight nine ten"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tokens = COUNT_TOKENS(txt) SELECT tokens;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("tokens"), Some(&Value::Integer(13)));
}

#[test]
fn test_find_hashtags_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "#openai #gpt4"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tags = FIND_HASHTAGS(txt) SELECT tags;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tags") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2, "expected 2 hashtags, got: {:?}", arr),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_initialism_alias() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "Application Programming Interface"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE abbr = INITIALISM(txt) SELECT abbr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("abbr"),
        Some(&Value::String("API".to_string()))
    );
}
