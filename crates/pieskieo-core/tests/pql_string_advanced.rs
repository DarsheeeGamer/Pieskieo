/// Integration tests for PQL advanced string similarity, phonetic, and NLP functions.
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

// ── SOUNDEX ─────────────────────────────────────────────────────────────────

#[test]
fn test_soundex_robert() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "Robert"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = SOUNDEX(nm) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::String(s)) => assert_eq!(s, "R163"),
        other => panic!("expected R163, got {:?}", other),
    }
}

#[test]
fn test_soundex_code_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "Robert"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = SOUNDEX_CODE(nm) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::String(s)) => assert_eq!(s, "R163"),
        other => panic!("expected R163 from SOUNDEX_CODE, got {:?}", other),
    }
}

#[test]
fn test_soundex_rupert_same_as_robert() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "Rupert"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = SOUNDEX(nm) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::String(s)) => assert_eq!(
            s, "R163",
            "Rupert and Robert should share Soundex code R163"
        ),
        other => panic!("expected R163, got {:?}", other),
    }
}

// ── NYSIIS ───────────────────────────────────────────────────────────────────

#[test]
fn test_nysiis_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "Smith"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = NYSIIS(nm) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::String(s)) => assert!(
            !s.is_empty(),
            "NYSIIS should return non-empty string, got {:?}",
            s
        ),
        other => panic!("expected string from NYSIIS, got {:?}", other),
    }
}

#[test]
fn test_nysiis_code_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "Smith"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = NYSIIS_CODE(nm) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::String(s)) => assert!(!s.is_empty(), "NYSIIS_CODE alias should work"),
        other => panic!("expected string from NYSIIS_CODE, got {:?}", other),
    }
}

// ── TOKENIZE ─────────────────────────────────────────────────────────────────

#[test]
fn test_tokenize_words() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE toks = TOKENIZE(txt) SELECT toks;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("toks") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("hello".to_string()));
            assert_eq!(arr[1], Value::String("world".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_str_tokenize_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE toks = STR_TOKENIZE(txt) SELECT toks;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("toks") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── BIGRAMS ───────────────────────────────────────────────────────────────────

#[test]
fn test_bigrams_abc() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bg = BIGRAMS(txt) SELECT bg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bg") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("ab".to_string()));
            assert_eq!(arr[1], Value::String("bc".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_str_bigrams_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bg = STR_BIGRAMS(txt) SELECT bg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bg") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array from STR_BIGRAMS, got {:?}", other),
    }
}

// ── TRIGRAMS ──────────────────────────────────────────────────────────────────

#[test]
fn test_trigrams_abcd() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abcd"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tg = TRIGRAMS(txt) SELECT tg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tg") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("abc".to_string()));
            assert_eq!(arr[1], Value::String("bcd".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_str_trigrams_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abcd"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tg = STR_TRIGRAMS(txt) SELECT tg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("tg") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array from STR_TRIGRAMS, got {:?}", other),
    }
}

// ── STR_NGRAMS ────────────────────────────────────────────────────────────────

#[test]
fn test_str_ngrams_3() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abcde"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ng = STR_NGRAMS(txt, 3) SELECT ng;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ng") {
        Some(Value::Array(arr)) => {
            // "abcde" with n=3: "abc", "bcd", "cde" → 3 trigrams
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::String("abc".to_string()));
            assert_eq!(arr[1], Value::String("bcd".to_string()));
            assert_eq!(arr[2], Value::String("cde".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── COMMON_PREFIX ─────────────────────────────────────────────────────────────

#[test]
fn test_common_prefix_flower_flow() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "flower", "b": "flow"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pfx = COMMON_PREFIX(a, b) SELECT pfx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pfx") {
        Some(Value::String(s)) => assert_eq!(s, "flow"),
        other => panic!("expected 'flow', got {:?}", other),
    }
}

#[test]
fn test_longest_prefix_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "flower", "b": "flow"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pfx = LONGEST_PREFIX(a, b) SELECT pfx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pfx") {
        Some(Value::String(s)) => assert_eq!(s, "flow"),
        other => panic!("expected 'flow' from LONGEST_PREFIX, got {:?}", other),
    }
}

// ── COMMON_SUFFIX ─────────────────────────────────────────────────────────────

#[test]
fn test_common_suffix_testing_ring() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "testing", "b": "ring"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sfx = COMMON_SUFFIX(a, b) SELECT sfx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sfx") {
        Some(Value::String(s)) => assert_eq!(s, "ing"),
        other => panic!("expected 'ing', got {:?}", other),
    }
}

#[test]
fn test_longest_suffix_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "testing", "b": "ring"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sfx = LONGEST_SUFFIX(a, b) SELECT sfx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sfx") {
        Some(Value::String(s)) => assert_eq!(s, "ing"),
        other => panic!("expected 'ing' from LONGEST_SUFFIX, got {:?}", other),
    }
}

// ── REGEX_MATCH_ALL ───────────────────────────────────────────────────────────

#[test]
fn test_regex_match_all_words() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ms = REGEX_MATCH_ALL(txt, "\\w+") SELECT ms;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ms") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], Value::String("hello".to_string()));
            assert_eq!(arr[1], Value::String("world".to_string()));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_regex_find_all_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ms = REGEX_FIND_ALL(txt, "\\w+") SELECT ms;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ms") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 2),
        other => panic!("expected array from REGEX_FIND_ALL, got {:?}", other),
    }
}

// ── REGEX_REPLACE_ALL ─────────────────────────────────────────────────────────

#[test]
fn test_regex_replace_all() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "hello world"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE rep = REGEX_REPLACE_ALL(txt, "o", "0") SELECT rep;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rep") {
        Some(Value::String(s)) => assert_eq!(s, "hell0 w0rld"),
        other => panic!("expected 'hell0 w0rld', got {:?}", other),
    }
}

// ── STR_ROTATE / CAESAR_CIPHER ────────────────────────────────────────────────

#[test]
fn test_str_rotate_abc_by_1() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rot = STR_ROTATE(txt, 1) SELECT rot;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rot") {
        Some(Value::String(s)) => assert_eq!(s, "bcd"),
        other => panic!("expected 'bcd', got {:?}", other),
    }
}

#[test]
fn test_caesar_cipher_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "abc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rot = CAESAR_CIPHER(txt, 1) SELECT rot;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rot") {
        Some(Value::String(s)) => assert_eq!(s, "bcd"),
        other => panic!("expected 'bcd' from CAESAR_CIPHER, got {:?}", other),
    }
}

#[test]
fn test_str_rotate_wraps_z_to_a() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"txt": "xyz"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rot = STR_ROTATE(txt, 3) SELECT rot;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rot") {
        Some(Value::String(s)) => assert_eq!(s, "abc"),
        other => panic!("expected 'abc' (wrap-around), got {:?}", other),
    }
}

// ── ANAGRAM_OF ────────────────────────────────────────────────────────────────

#[test]
fn test_anagram_of_listen_silent() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "listen", "b": "silent"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE anag = ANAGRAM_OF(a, b) SELECT anag;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("anag") {
        Some(Value::Bool(b)) => assert!(*b, "listen and silent should be anagrams"),
        other => panic!("expected true, got {:?}", other),
    }
}

#[test]
fn test_anagram_of_hello_world_false() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "hello", "b": "world"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE anag = ANAGRAM_OF(a, b) SELECT anag;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("anag") {
        Some(Value::Bool(b)) => assert!(!*b, "hello and world should not be anagrams"),
        other => panic!("expected false, got {:?}", other),
    }
}

#[test]
fn test_is_anagram_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"a": "listen", "b": "silent"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE anag = IS_ANAGRAM(a, b) SELECT anag;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("anag") {
        Some(Value::Bool(b)) => assert!(*b, "IS_ANAGRAM alias should return true for anagrams"),
        other => panic!("expected true from IS_ANAGRAM, got {:?}", other),
    }
}

// ── ABBREVIATE_NAME ───────────────────────────────────────────────────────────

#[test]
fn test_abbreviate_name_john_smith() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "John Smith"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE abbr = ABBREVIATE_NAME(nm) SELECT abbr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("abbr") {
        Some(Value::String(s)) => assert_eq!(s, "J. Smith"),
        other => panic!("expected 'J. Smith', got {:?}", other),
    }
}

#[test]
fn test_name_abbrev_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"nm": "John Smith"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE abbr = NAME_ABBREV(nm) SELECT abbr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("abbr") {
        Some(Value::String(s)) => assert_eq!(s, "J. Smith"),
        other => panic!("expected 'J. Smith' from NAME_ABBREV, got {:?}", other),
    }
}

// ── PLURALIZE ─────────────────────────────────────────────────────────────────

#[test]
fn test_pluralize_cat() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"wd": "cat"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pl = PLURALIZE(wd) SELECT pl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pl") {
        Some(Value::String(s)) => assert_eq!(s, "cats"),
        other => panic!("expected 'cats', got {:?}", other),
    }
}

#[test]
fn test_pluralize_city() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"wd": "city"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pl = PLURALIZE(wd) SELECT pl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pl") {
        Some(Value::String(s)) => assert_eq!(s, "cities"),
        other => panic!("expected 'cities', got {:?}", other),
    }
}

#[test]
fn test_simple_plural_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"wd": "cat"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pl = SIMPLE_PLURAL(wd) SELECT pl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pl") {
        Some(Value::String(s)) => assert_eq!(s, "cats"),
        other => panic!("expected 'cats' from SIMPLE_PLURAL, got {:?}", other),
    }
}

// ── NTH_SUFFIX ────────────────────────────────────────────────────────────────

#[test]
fn test_nth_suffix_1_st() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 1}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "st"),
        other => panic!("expected 'st', got {:?}", other),
    }
}

#[test]
fn test_nth_suffix_2_nd() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 2}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "nd"),
        other => panic!("expected 'nd', got {:?}", other),
    }
}

#[test]
fn test_nth_suffix_3_rd() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 3}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "rd"),
        other => panic!("expected 'rd', got {:?}", other),
    }
}

#[test]
fn test_nth_suffix_4_th() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 4}))
        .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "th"),
        other => panic!("expected 'th', got {:?}", other),
    }
}

#[test]
fn test_nth_suffix_11_th() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 11}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "th", "11 should use 'th' not 'st'"),
        other => panic!("expected 'th', got {:?}", other),
    }
}

#[test]
fn test_nth_suffix_12_th() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 12}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "th", "12 should use 'th' not 'nd'"),
        other => panic!("expected 'th', got {:?}", other),
    }
}

#[test]
fn test_nth_suffix_21_st() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 21}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE suf = NTH_SUFFIX(n) SELECT suf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("suf") {
        Some(Value::String(s)) => assert_eq!(s, "st"),
        other => panic!("expected 'st', got {:?}", other),
    }
}
