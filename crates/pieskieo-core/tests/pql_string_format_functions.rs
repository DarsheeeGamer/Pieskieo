/// Integration tests for PQL string format/case/encoding functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
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
fn test_camel_to_snake() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "helloWorldFoo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CAMEL_TO_SNAKE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("hello_world_foo".to_string())));
}

#[test]
fn test_camel_to_snake_single_word() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CAMEL_TO_SNAKE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("hello".to_string())));
}

#[test]
fn test_snake_to_camel() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello_world_foo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SNAKE_TO_CAMEL(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("helloWorldFoo".to_string())));
}

#[test]
fn test_to_pascal_case() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello_world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TO_PASCAL_CASE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("HelloWorld".to_string())));
}

#[test]
fn test_to_pascal_case_with_spaces() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello world foo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TO_PASCAL_CASE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("HelloWorldFoo".to_string())));
}

#[test]
fn test_to_kebab_case() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello_World"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TO_KEBAB_CASE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("hello-world".to_string())));
}

#[test]
fn test_to_screaming_snake() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello_world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TO_SCREAMING_SNAKE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("HELLO_WORLD".to_string())));
}

#[test]
fn test_is_palindrome_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "racecar"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_PALINDROME(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_palindrome_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_PALINDROME(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_palindrome_case_insensitive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "A man a plan a canal Panama"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = IS_PALINDROME(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Bool(true)));
}

#[test]
fn test_syllable_count() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "education"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SYLLABLE_COUNT(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // "ed-u-ca-tion" has 4 vowel groups: e, u, a, io -> 4, no trailing silent e -> 4
    match r.rows[0].data.get("out") {
        Some(Value::Integer(n)) => assert!(*n >= 1 && *n <= 6, "Unexpected syllable count: {}", n),
        other => panic!("Expected Integer, got {:?}", other),
    }
}

#[test]
fn test_syllable_count_single() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "strength"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SYLLABLE_COUNT(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // "strength" has 1 vowel group: e -> 1
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::Integer(1)));
}

#[test]
fn test_char_frequency() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = CHAR_FREQUENCY(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Object(map)) => {
            assert_eq!(map.get("h"), Some(&Value::Integer(1)));
            assert_eq!(map.get("e"), Some(&Value::Integer(1)));
            assert_eq!(map.get("l"), Some(&Value::Integer(2)));
            assert_eq!(map.get("o"), Some(&Value::Integer(1)));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

#[test]
fn test_squeeze() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "heeellllo  world"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SQUEEZE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("helo world".to_string())));
}

#[test]
fn test_interleave() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": "abc", "b": "123"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INTERLEAVE(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("a1b2c3".to_string())));
}

#[test]
fn test_interleave_unequal_length() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"a": "abcde", "b": "12"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = INTERLEAVE(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("a1b2cde".to_string())));
}

#[test]
fn test_pad_center() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hi"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = PAD_CENTER(s, 10, " ") SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 10, "Expected length 10, got {}", s.len());
            assert!(s.contains("hi"), "Expected 'hi' in result: {:?}", s);
            assert!(s.trim() == "hi", "Expected only 'hi' after trim, got {:?}", s.trim());
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_wrap_text() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello world foo bar"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = WRAP_TEXT(s, 10) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::String(s)) => {
            let lines: Vec<&str> = s.split('\n').collect();
            assert!(lines.len() > 1, "Expected multiple lines, got: {:?}", s);
            for line in &lines {
                assert!(line.len() <= 10, "Line too long ({}): {:?}", line.len(), line);
            }
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_expand_tabs() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "a\tb"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = EXPAND_TABS(s, 4) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // "a" is at col 0, tab at col 1 -> expand to 4 spaces from col 1 -> 3 spaces
    match r.rows[0].data.get("out") {
        Some(Value::String(s)) => {
            assert!(!s.contains('\t'), "Expected no tabs in result: {:?}", s);
            assert!(s.starts_with('a'), "Expected result to start with 'a': {:?}", s);
            assert!(s.ends_with('b'), "Expected result to end with 'b': {:?}", s);
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_normalize_whitespace() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello   world\t\nfoo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = NORMALIZE_WHITESPACE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("hello world foo".to_string())));
}

#[test]
fn test_title_case_smart() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "the lord of the rings"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TITLE_CASE_SMART(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("The Lord of the Rings".to_string())));
}

#[test]
fn test_title_case_smart_first_word_always_capitalized() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "a tale of two cities"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = TITLE_CASE_SMART(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("A Tale of Two Cities".to_string())));
}

#[test]
fn test_screaming_snake_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "hello_world_foo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SCREAMING_SNAKE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("HELLO_WORLD_FOO".to_string())));
}

#[test]
fn test_squeeze_no_duplicates() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "abc"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = SQUEEZE(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("abc".to_string())));
}

#[test]
fn test_collapse_spaces_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"s": "  leading and  trailing  "})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE out = COLLAPSE_SPACES(s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("out"), Some(&Value::String("leading and trailing".to_string())));
}
