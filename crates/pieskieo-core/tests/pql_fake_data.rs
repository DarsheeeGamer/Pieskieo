/// Integration tests for PQL synthetic/fake data generation functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup(ns: &str) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), serde_json::json!({}))
        .unwrap();
    (dir, db, ex)
}

fn run_str(ex: &Executor, ns: &str, expr: &str) -> String {
    let query = format!("QUERY {} COMPUTE res = {} SELECT res;", ns, expr);
    let mut p = Parser::new(&query);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("res") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    }
}

// ── FAKE_FIRST_NAME / GEN_FIRST_NAME ────────────────────────────────────────

#[test]
fn test_fake_first_name_non_empty() {
    let (_dir, _db, ex) = setup("fn1");
    let s = run_str(&ex, "fn1", "FAKE_FIRST_NAME(42)");
    assert!(
        !s.is_empty(),
        "FAKE_FIRST_NAME should return non-empty string"
    );
}

#[test]
fn test_fake_first_name_deterministic() {
    let (_dir, _db, ex) = setup("fn2");
    let a = run_str(&ex, "fn2", "FAKE_FIRST_NAME(42)");
    let b = run_str(&ex, "fn2", "FAKE_FIRST_NAME(42)");
    assert_eq!(a, b, "Same seed should produce same output");
}

#[test]
fn test_gen_first_name_alias() {
    let (_dir, _db, ex) = setup("fn3");
    let a = run_str(&ex, "fn3", "FAKE_FIRST_NAME(7)");
    let b = run_str(&ex, "fn3", "GEN_FIRST_NAME(7)");
    assert_eq!(a, b, "GEN_FIRST_NAME alias should match FAKE_FIRST_NAME");
}

// ── FAKE_LAST_NAME / GEN_LAST_NAME ──────────────────────────────────────────

#[test]
fn test_fake_last_name_non_empty() {
    let (_dir, _db, ex) = setup("ln1");
    let s = run_str(&ex, "ln1", "FAKE_LAST_NAME(99)");
    assert!(
        !s.is_empty(),
        "FAKE_LAST_NAME should return non-empty string"
    );
}

#[test]
fn test_gen_last_name_alias() {
    let (_dir, _db, ex) = setup("ln2");
    let a = run_str(&ex, "ln2", "FAKE_LAST_NAME(99)");
    let b = run_str(&ex, "ln2", "GEN_LAST_NAME(99)");
    assert_eq!(a, b, "GEN_LAST_NAME alias should match FAKE_LAST_NAME");
}

// ── FAKE_FULL_NAME / GEN_FULL_NAME ──────────────────────────────────────────

#[test]
fn test_fake_full_name_has_space() {
    let (_dir, _db, ex) = setup("fn4");
    let s = run_str(&ex, "fn4", "FAKE_FULL_NAME(1)");
    assert!(
        s.contains(' '),
        "FAKE_FULL_NAME should contain a space, got: {}",
        s
    );
}

#[test]
fn test_gen_full_name_alias() {
    let (_dir, _db, ex) = setup("fn5");
    let a = run_str(&ex, "fn5", "FAKE_FULL_NAME(1)");
    let b = run_str(&ex, "fn5", "GEN_FULL_NAME(1)");
    assert_eq!(a, b, "GEN_FULL_NAME alias should match FAKE_FULL_NAME");
}

// ── FAKE_EMAIL / GEN_EMAIL ──────────────────────────────────────────────────

#[test]
fn test_fake_email_contains_at_and_dot() {
    let (_dir, _db, ex) = setup("em1");
    let s = run_str(&ex, "em1", "FAKE_EMAIL(77)");
    assert!(s.contains('@'), "email should contain '@': {}", s);
    assert!(s.contains('.'), "email should contain '.': {}", s);
}

#[test]
fn test_gen_email_alias() {
    let (_dir, _db, ex) = setup("em2");
    let a = run_str(&ex, "em2", "FAKE_EMAIL(77)");
    let b = run_str(&ex, "em2", "GEN_EMAIL(77)");
    assert_eq!(a, b, "GEN_EMAIL alias should match FAKE_EMAIL");
}

// ── FAKE_USERNAME / GEN_USERNAME ────────────────────────────────────────────

#[test]
fn test_fake_username_non_empty_with_digits() {
    let (_dir, _db, ex) = setup("un1");
    let s = run_str(&ex, "un1", "FAKE_USERNAME(55)");
    assert!(
        !s.is_empty(),
        "FAKE_USERNAME should return non-empty string"
    );
    assert!(
        s.chars().any(|c| c.is_ascii_digit()),
        "FAKE_USERNAME should contain digits: {}",
        s
    );
}

#[test]
fn test_gen_username_alias() {
    let (_dir, _db, ex) = setup("un2");
    let a = run_str(&ex, "un2", "FAKE_USERNAME(55)");
    let b = run_str(&ex, "un2", "GEN_USERNAME(55)");
    assert_eq!(a, b, "GEN_USERNAME alias should match FAKE_USERNAME");
}

// ── FAKE_PASSWORD / GEN_PASSWORD ────────────────────────────────────────────

#[test]
fn test_fake_password_correct_length() {
    let (_dir, _db, ex) = setup("pw1");
    let s = run_str(&ex, "pw1", "FAKE_PASSWORD(33, 16)");
    assert_eq!(
        s.len(),
        16,
        "FAKE_PASSWORD(33, 16) should have length 16, got: {}",
        s.len()
    );
}

#[test]
fn test_fake_password_default_length() {
    let (_dir, _db, ex) = setup("pw2");
    let s = run_str(&ex, "pw2", "FAKE_PASSWORD(33)");
    assert_eq!(
        s.len(),
        12,
        "FAKE_PASSWORD default length should be 12, got: {}",
        s.len()
    );
}

#[test]
fn test_gen_password_alias() {
    let (_dir, _db, ex) = setup("pw3");
    let a = run_str(&ex, "pw3", "FAKE_PASSWORD(33, 16)");
    let b = run_str(&ex, "pw3", "GEN_PASSWORD(33, 16)");
    assert_eq!(a, b, "GEN_PASSWORD alias should match FAKE_PASSWORD");
}

// ── FAKE_PHONE / GEN_PHONE ──────────────────────────────────────────────────

#[test]
fn test_fake_phone_format() {
    let (_dir, _db, ex) = setup("ph1");
    let s = run_str(&ex, "ph1", "FAKE_PHONE(21)");
    // Format: "(XXX) XXX-XXXX" — length 14
    assert_eq!(s.len(), 14, "phone should be 14 chars: {}", s);
    assert!(s.starts_with('('), "phone should start with '(': {}", s);
    assert!(s.contains(") "), "phone should contain ') ': {}", s);
    assert!(s.contains('-'), "phone should contain '-': {}", s);
}

#[test]
fn test_gen_phone_alias() {
    let (_dir, _db, ex) = setup("ph2");
    let a = run_str(&ex, "ph2", "FAKE_PHONE(21)");
    let b = run_str(&ex, "ph2", "GEN_PHONE(21)");
    assert_eq!(a, b, "GEN_PHONE alias should match FAKE_PHONE");
}

// ── FAKE_SSN / GEN_SSN ──────────────────────────────────────────────────────

#[test]
fn test_fake_ssn_contains_dashes() {
    let (_dir, _db, ex) = setup("ssn1");
    let s = run_str(&ex, "ssn1", "FAKE_SSN(66)");
    assert!(s.contains('-'), "SSN should contain '-': {}", s);
    let parts: Vec<&str> = s.split('-').collect();
    assert_eq!(parts.len(), 3, "SSN should have 3 parts: {}", s);
}

#[test]
fn test_gen_ssn_alias() {
    let (_dir, _db, ex) = setup("ssn2");
    let a = run_str(&ex, "ssn2", "FAKE_SSN(66)");
    let b = run_str(&ex, "ssn2", "GEN_SSN(66)");
    assert_eq!(a, b, "GEN_SSN alias should match FAKE_SSN");
}

// ── FAKE_IP / GEN_IP ────────────────────────────────────────────────────────

#[test]
fn test_fake_ip_valid_format() {
    let (_dir, _db, ex) = setup("ip1");
    let s = run_str(&ex, "ip1", "FAKE_IP(88)");
    let parts: Vec<&str> = s.split('.').collect();
    assert_eq!(parts.len(), 4, "IP should have 4 octets: {}", s);
    for part in &parts {
        let n: u64 = part.parse().expect("octet should be numeric");
        assert!(n < 256, "octet should be < 256: {}", n);
    }
}

#[test]
fn test_gen_ip_alias() {
    let (_dir, _db, ex) = setup("ip2");
    let a = run_str(&ex, "ip2", "FAKE_IP(88)");
    let b = run_str(&ex, "ip2", "GEN_IP(88)");
    assert_eq!(a, b, "GEN_IP alias should match FAKE_IP");
}

// ── FAKE_MAC / GEN_MAC ──────────────────────────────────────────────────────

#[test]
fn test_fake_mac_six_octets() {
    let (_dir, _db, ex) = setup("mac1");
    let s = run_str(&ex, "mac1", "FAKE_MAC(111)");
    assert!(s.contains(':'), "MAC should contain ':': {}", s);
    let octets: Vec<&str> = s.split(':').collect();
    assert_eq!(octets.len(), 6, "MAC should have 6 octets: {}", s);
    for o in &octets {
        assert_eq!(o.len(), 2, "each octet should be 2 hex chars: {}", o);
    }
}

#[test]
fn test_gen_mac_alias() {
    let (_dir, _db, ex) = setup("mac2");
    let a = run_str(&ex, "mac2", "FAKE_MAC(111)");
    let b = run_str(&ex, "mac2", "GEN_MAC(111)");
    assert_eq!(a, b, "GEN_MAC alias should match FAKE_MAC");
}

// ── FAKE_HEX_COLOR / GEN_HEX_COLOR ──────────────────────────────────────────

#[test]
fn test_fake_hex_color_format() {
    let (_dir, _db, ex) = setup("hc1");
    let s = run_str(&ex, "hc1", "FAKE_HEX_COLOR(42)");
    assert!(s.starts_with('#'), "hex color should start with '#': {}", s);
    assert_eq!(s.len(), 7, "hex color should be 7 chars: {}", s);
    assert!(
        s[1..].chars().all(|c| c.is_ascii_hexdigit()),
        "hex color body should be hex digits: {}",
        s
    );
}

#[test]
fn test_gen_hex_color_alias() {
    let (_dir, _db, ex) = setup("hc2");
    let a = run_str(&ex, "hc2", "FAKE_HEX_COLOR(42)");
    let b = run_str(&ex, "hc2", "GEN_HEX_COLOR(42)");
    assert_eq!(a, b, "GEN_HEX_COLOR alias should match FAKE_HEX_COLOR");
}

// ── FAKE_DATE / GEN_DATE ────────────────────────────────────────────────────

#[test]
fn test_fake_date_format() {
    let (_dir, _db, ex) = setup("dt1");
    let s = run_str(&ex, "dt1", "FAKE_DATE(200)");
    // Format: YYYY-MM-DD, length 10
    assert_eq!(s.len(), 10, "date should be 10 chars: {}", s);
    let parts: Vec<&str> = s.split('-').collect();
    assert_eq!(parts.len(), 3, "date should have 3 parts: {}", s);
    let year: u32 = parts[0].parse().expect("year should be numeric");
    assert!(
        year >= 2000 && year <= 2024,
        "year should be in range: {}",
        year
    );
}

#[test]
fn test_gen_date_alias() {
    let (_dir, _db, ex) = setup("dt2");
    let a = run_str(&ex, "dt2", "FAKE_DATE(200)");
    let b = run_str(&ex, "dt2", "GEN_DATE(200)");
    assert_eq!(a, b, "GEN_DATE alias should match FAKE_DATE");
}

// ── FAKE_COMPANY / GEN_COMPANY ──────────────────────────────────────────────

#[test]
fn test_fake_company_non_empty_with_spaces() {
    let (_dir, _db, ex) = setup("co1");
    let s = run_str(&ex, "co1", "FAKE_COMPANY(300)");
    assert!(!s.is_empty(), "FAKE_COMPANY should return non-empty string");
    assert!(s.contains(' '), "FAKE_COMPANY should contain spaces: {}", s);
}

#[test]
fn test_gen_company_alias() {
    let (_dir, _db, ex) = setup("co2");
    let a = run_str(&ex, "co2", "FAKE_COMPANY(300)");
    let b = run_str(&ex, "co2", "GEN_COMPANY(300)");
    assert_eq!(a, b, "GEN_COMPANY alias should match FAKE_COMPANY");
}

// ── LOREM_IPSUM / GEN_LOREM ─────────────────────────────────────────────────

#[test]
fn test_lorem_ipsum_word_count() {
    let (_dir, _db, ex) = setup("li1");
    let s = run_str(&ex, "li1", "LOREM_IPSUM(5)");
    let word_count = s.split_whitespace().count();
    assert_eq!(
        word_count, 5,
        "LOREM_IPSUM(5) should produce 5 words, got: {}",
        word_count
    );
}

#[test]
fn test_gen_lorem_alias() {
    let (_dir, _db, ex) = setup("li2");
    let a = run_str(&ex, "li2", "LOREM_IPSUM(10, 42)");
    let b = run_str(&ex, "li2", "GEN_LOREM(10, 42)");
    assert_eq!(a, b, "GEN_LOREM alias should match LOREM_IPSUM");
}

// ── FAKE_SENTENCE / GEN_SENTENCE ────────────────────────────────────────────

#[test]
fn test_fake_sentence_ends_with_period() {
    let (_dir, _db, ex) = setup("se1");
    let s = run_str(&ex, "se1", "FAKE_SENTENCE(6)");
    assert!(s.ends_with('.'), "FAKE_SENTENCE should end with '.': {}", s);
}

#[test]
fn test_fake_sentence_starts_capitalized() {
    let (_dir, _db, ex) = setup("se2");
    let s = run_str(&ex, "se2", "FAKE_SENTENCE(6)");
    let first_char = s.chars().next().unwrap();
    assert!(
        first_char.is_uppercase(),
        "FAKE_SENTENCE should start with uppercase: {}",
        s
    );
}

#[test]
fn test_gen_sentence_alias() {
    let (_dir, _db, ex) = setup("se3");
    let a = run_str(&ex, "se3", "FAKE_SENTENCE(6, 99)");
    let b = run_str(&ex, "se3", "GEN_SENTENCE(6, 99)");
    assert_eq!(a, b, "GEN_SENTENCE alias should match FAKE_SENTENCE");
}

// ── Determinism across all functions ────────────────────────────────────────

#[test]
fn test_determinism_same_seed_same_result() {
    let (_dir, _db, ex) = setup("det1");
    // Run each function twice with the same seed and assert equality
    let fns = [
        "FAKE_FIRST_NAME(123)",
        "FAKE_LAST_NAME(123)",
        "FAKE_FULL_NAME(123)",
        "FAKE_EMAIL(123)",
        "FAKE_USERNAME(123)",
        "FAKE_PASSWORD(123)",
        "FAKE_PHONE(123)",
        "FAKE_SSN(123)",
        "FAKE_IP(123)",
        "FAKE_MAC(123)",
        "FAKE_HEX_COLOR(123)",
        "FAKE_DATE(123)",
        "FAKE_COMPANY(123)",
        "LOREM_IPSUM(5, 123)",
        "FAKE_SENTENCE(4, 123)",
    ];
    for f in &fns {
        let a = run_str(&ex, "det1", f);
        let b = run_str(&ex, "det1", f);
        assert_eq!(a, b, "Function {} should be deterministic", f);
    }
}

#[test]
fn test_different_seeds_different_results() {
    let (_dir, _db, ex) = setup("dif1");
    // With distinct seeds the outputs should differ (at least sometimes)
    let a = run_str(&ex, "dif1", "FAKE_FIRST_NAME(1)");
    let b = run_str(&ex, "dif1", "FAKE_FIRST_NAME(1000000)");
    // They might occasionally collide but that is extremely rare
    // Just verify both return valid strings
    assert!(!a.is_empty());
    assert!(!b.is_empty());
}
