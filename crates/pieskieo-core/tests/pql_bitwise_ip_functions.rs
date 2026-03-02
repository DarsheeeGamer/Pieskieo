/// Integration tests for PQL bitwise and IP address functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_popcount() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 255 = 0b11111111 → 8 set bits; 0 → 0 bits; 7 = 0b111 → 3 bits
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": 255, "b": 0, "c": 7})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE pa = POPCOUNT(a) COMPUTE pb = POPCOUNT(b) COMPUTE pc = POPCOUNT(c) SELECT pa, pb, pc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("pa"), Some(&Value::Integer(8)));
    assert_eq!(r.rows[0].data.get("pb"), Some(&Value::Integer(0)));
    assert_eq!(r.rows[0].data.get("pc"), Some(&Value::Integer(3)));
}

#[test]
fn test_bit_get_set_clear() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // 5 = 0b101: bit 0 = 1, bit 1 = 0, bit 2 = 1
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 5, "pos0": 0, "pos1": 1})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE b0 = BIT_GET(n, pos0) COMPUTE b1 = BIT_GET(n, pos1) COMPUTE set1 = BIT_SET(n, pos1) COMPUTE clr0 = BIT_CLEAR(n, pos0) SELECT b0, b1, set1, clr0;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("b0"), Some(&Value::Integer(1)));
    assert_eq!(r.rows[0].data.get("b1"), Some(&Value::Integer(0)));
    assert_eq!(r.rows[0].data.get("set1"), Some(&Value::Integer(7)));   // 5 | 2 = 7
    assert_eq!(r.rows[0].data.get("clr0"), Some(&Value::Integer(4)));   // 5 & ~1 = 4
}

#[test]
fn test_ip_to_int_and_back() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.1"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE n = IP_TO_INT(ip) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ip_int = match r.rows[0].data.get("n") {
        Some(Value::Integer(i)) => *i,
        other => panic!("expected integer, got {:?}", other),
    };
    assert!(ip_int > 0, "IP integer should be positive");

    // Round trip: int back to IP
    db.put_doc_ns(None, Some("t2"), Uuid::new_v4(), serde_json::json!({"n": ip_int})).unwrap();
    let mut p2 = Parser::new(r#"QUERY t2 COMPUTE ip = INT_TO_IP(n) SELECT ip;"#);
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(r2.rows[0].data.get("ip"), Some(&Value::String("192.168.1.1".to_string())));
}

#[test]
fn test_is_ipv4_ipv6() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"v4": "10.0.0.1", "v6": "::1", "bad": "hello"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE a = IS_IPV4(v4) COMPUTE b = IS_IPV4(bad) COMPUTE c = IS_IPV6(v6) SELECT a, b, c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Bool(true)));
}

#[test]
fn test_ip_in_cidr() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip1": "192.168.1.100", "ip2": "10.0.0.1", "cidr": "192.168.1.0/24"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE in_net = IP_IN_CIDR(ip1, cidr) COMPUTE out_net = IP_IN_CIDR(ip2, cidr) SELECT in_net, out_net;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("in_net"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("out_net"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_private_ip() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"private": "192.168.1.1", "public": "8.8.8.8", "rfc172": "172.16.5.1"})).unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE a = IS_PRIVATE_IP(private) COMPUTE b = IS_PRIVATE_IP(public) COMPUTE c = IS_PRIVATE_IP(rfc172) SELECT a, b, c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::Bool(true)));
}
