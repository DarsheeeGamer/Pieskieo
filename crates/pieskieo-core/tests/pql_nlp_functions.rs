/// Integration tests for PQL text/NLP analytics functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_ngrams() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "the quick brown fox"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE ng = NGRAMS(text, 2) SELECT ng;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ng") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "bigrams of 4 words = 3, got: {:?}", arr);
            assert_eq!(&arr[0], &Value::String("the quick".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_term_freq() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "cat cat dog cat dog"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE tf = TERM_FREQ(text, "cat") SELECT tf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tf") {
        Some(Value::Float(f)) => assert!((*f - 0.6).abs() < 0.01, "cat freq = 3/5 = 0.6, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_unique_words() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "apple orange apple banana"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE uw = UNIQUE_WORDS(text) SELECT uw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("uw"), Some(&Value::Integer(3)));
}

#[test]
fn test_text_entropy() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Uniform distribution has max entropy
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "aabbccdd"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE e = TEXT_ENTROPY(text) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("e") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "entropy should be positive, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_sentence_count() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "Hello world. How are you? I am fine!"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE sc = SENTENCE_COUNT(text) SELECT sc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("sc"), Some(&Value::Integer(3)));
}

#[test]
fn test_word_frequency_map() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "cat dog cat"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE m = WORD_FREQUENCY_MAP(text) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("cat"), Some(&Value::Integer(2)));
            assert_eq!(obj.get("dog"), Some(&Value::Integer(1)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_lexical_diversity() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 2 unique words out of 4 total
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"text": "cat dog cat dog"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE ld = LEXICAL_DIVERSITY(text) SELECT ld;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ld") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.01, "diversity = 2/4 = 0.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}
