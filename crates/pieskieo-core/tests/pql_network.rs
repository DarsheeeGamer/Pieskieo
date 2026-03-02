/// Tests for the new PQL built-in network/IP address functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (dir, db, ex)
}

// ── IS_VALID_IPV4 / IS_IPV4 ──────────────────────────────────────────────────

#[test]
fn test_is_valid_ipv4() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.1.1", "bad": "999.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_IPV4(addr) COMPUTE bad = IS_VALID_IPV4(bad) SELECT ok, bad;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("bad"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_is_ipv4_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "10.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_IPV4(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_valid_ipv4_invalid_octet() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "256.1.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_IPV4(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

// ── IS_VALID_IPV6 / IS_IPV6 ──────────────────────────────────────────────────

#[test]
fn test_is_valid_ipv6() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "2001:0db8:0000:0000:0000:0000:0000:0001"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_IPV6(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_ipv6_compressed() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "2001:db8::1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_IPV6(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── IS_VALID_IP / IS_IP ───────────────────────────────────────────────────────

#[test]
fn test_is_valid_ip_v4() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "8.8.8.8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_valid_ip_v6() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "::1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_ip_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "not-an-ip"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

// ── IP_OCTETS / IPV4_OCTETS ───────────────────────────────────────────────────

#[test]
fn test_ip_octets() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.1.5"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_OCTETS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(192),
            Value::Integer(168),
            Value::Integer(1),
            Value::Integer(5),
        ]))
    );
    drop(dir);
}

#[test]
fn test_ipv4_octets_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "10.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IPV4_OCTETS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("r"),
        Some(&Value::Array(vec![
            Value::Integer(10),
            Value::Integer(0),
            Value::Integer(0),
            Value::Integer(1),
        ]))
    );
    drop(dir);
}

#[test]
fn test_ip_octets_invalid_returns_null() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "not-an-ip"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_OCTETS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
    drop(dir);
}

// ── IP_ADDR_VERSION ───────────────────────────────────────────────────────────

#[test]
fn test_ip_addr_version_v4() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_ADDR_VERSION(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(4)));
    drop(dir);
}

#[test]
fn test_ip_addr_version_v6() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "::1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_ADDR_VERSION(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(6)));
    drop(dir);
}

#[test]
fn test_ip_addr_version_invalid_returns_null() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "garbage"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_ADDR_VERSION(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
    drop(dir);
}

// ── IP_IN_SUBNET / CIDR_INCLUDES ─────────────────────────────────────────────

#[test]
fn test_ip_in_subnet_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.1.100", "net": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_IN_SUBNET(addr, net) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_ip_in_subnet_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "10.0.0.1", "net": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IP_IN_SUBNET(addr, net) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_cidr_includes_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "10.1.2.3", "net": "10.0.0.0/8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CIDR_INCLUDES(addr, net) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── SUBNET_NETWORK ────────────────────────────────────────────────────────────

#[test]
fn test_subnet_network() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBNET_NETWORK(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("192.168.1.0".to_string())));
    drop(dir);
}

#[test]
fn test_subnet_network_slash16() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "172.16.5.42/16"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBNET_NETWORK(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("172.16.0.0".to_string())));
    drop(dir);
}

// ── SUBNET_BROADCAST ──────────────────────────────────────────────────────────

#[test]
fn test_subnet_broadcast() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBNET_BROADCAST(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("192.168.1.255".to_string())));
    drop(dir);
}

// ── CIDR_MASK / SUBNET_MASK ───────────────────────────────────────────────────

#[test]
fn test_cidr_mask_24() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CIDR_MASK(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("255.255.255.0".to_string())));
    drop(dir);
}

#[test]
fn test_subnet_mask_alias_16() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "172.16.0.0/16"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBNET_MASK(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("255.255.0.0".to_string())));
    drop(dir);
}

#[test]
fn test_cidr_mask_8() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "10.0.0.0/8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = CIDR_MASK(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("255.0.0.0".to_string())));
    drop(dir);
}

// ── SUBNET_HOSTS ──────────────────────────────────────────────────────────────

#[test]
fn test_subnet_hosts_24() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBNET_HOSTS(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(254)));
    drop(dir);
}

#[test]
fn test_subnet_hosts_32() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "10.0.0.1/32"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = SUBNET_HOSTS(cidr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // /32 has 1 total address, 0 usable hosts (total - 2 clamped to 0)
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(0)));
    drop(dir);
}

// ── IS_LOOPBACK_IP ────────────────────────────────────────────────────────────

#[test]
fn test_is_loopback_ip_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "127.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_LOOPBACK_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_loopback_ip_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_LOOPBACK_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_is_loopback_ip_ipv6_loopback() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "::1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_LOOPBACK_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── IS_MULTICAST / IS_MULTICAST_IP ────────────────────────────────────────────

#[test]
fn test_is_multicast_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "224.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_MULTICAST(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_multicast_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_MULTICAST(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

// ── IS_LINK_LOCAL_IP / IS_APIPA ───────────────────────────────────────────────

#[test]
fn test_is_link_local_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "169.254.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_LINK_LOCAL_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_apipa_alias_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "169.254.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_APIPA(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_link_local_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_LINK_LOCAL_IP(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

// ── IPV4_CLASS ────────────────────────────────────────────────────────────────

#[test]
fn test_ipv4_class_a() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "10.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IPV4_CLASS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("A".to_string())));
    drop(dir);
}

#[test]
fn test_ipv4_class_b() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "172.16.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IPV4_CLASS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("B".to_string())));
    drop(dir);
}

#[test]
fn test_ipv4_class_c() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "192.168.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IPV4_CLASS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("C".to_string())));
    drop(dir);
}

#[test]
fn test_ipv4_class_d() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "224.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IPV4_CLASS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("D".to_string())));
    drop(dir);
}

#[test]
fn test_ipv4_class_e() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"addr": "240.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IPV4_CLASS(addr) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("E".to_string())));
    drop(dir);
}

// ── EXTRACT_SCHEME ────────────────────────────────────────────────────────────

#[test]
fn test_extract_scheme_https() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com/foo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_SCHEME(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("https".to_string())));
    drop(dir);
}

#[test]
fn test_extract_scheme_http() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "http://example.com"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_SCHEME(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("http".to_string())));
    drop(dir);
}

#[test]
fn test_extract_scheme_no_scheme_returns_null() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "example.com"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_SCHEME(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
    drop(dir);
}

// ── URL_DOMAIN ────────────────────────────────────────────────────────────────

#[test]
fn test_url_domain() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://www.example.com/page"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_DOMAIN(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("www.example.com".to_string())));
    drop(dir);
}

#[test]
fn test_url_domain_strips_port() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "http://example.com:8080/api"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_DOMAIN(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("example.com".to_string())));
    drop(dir);
}

// ── EXTRACT_PATH ──────────────────────────────────────────────────────────────

#[test]
fn test_extract_path() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com/foo/bar?q=1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_PATH(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("/foo/bar".to_string())));
    drop(dir);
}

#[test]
fn test_extract_path_no_path() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_PATH(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("/".to_string())));
    drop(dir);
}

// ── EXTRACT_QUERY_STRING ──────────────────────────────────────────────────────

#[test]
fn test_extract_query_string() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com?foo=1&bar=2"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_QUERY_STRING(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("foo=1&bar=2".to_string())));
    drop(dir);
}

#[test]
fn test_extract_query_string_none_returns_null() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com/page"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = EXTRACT_QUERY_STRING(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
    drop(dir);
}

// ── HOSTNAME_FROM_URL ─────────────────────────────────────────────────────────

#[test]
fn test_hostname_from_url() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://api.example.com/v1/foo"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HOSTNAME_FROM_URL(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("api.example.com".to_string())));
    drop(dir);
}

#[test]
fn test_hostname_from_url_strips_port() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "http://localhost:3000/app"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = HOSTNAME_FROM_URL(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("localhost".to_string())));
    drop(dir);
}

// ── PORT_FROM_URL / URL_PORT ───────────────────────────────────────────────────

#[test]
fn test_port_from_url() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "http://example.com:8080/path"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PORT_FROM_URL(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(8080)));
    drop(dir);
}

#[test]
fn test_url_port_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com:443/"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = URL_PORT(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(443)));
    drop(dir);
}

#[test]
fn test_port_from_url_no_port_returns_null() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"url": "https://example.com/path"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = PORT_FROM_URL(url) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
    drop(dir);
}

// ── IS_VALID_MAC / IS_MAC_ADDR ────────────────────────────────────────────────

#[test]
fn test_is_valid_mac_colon() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "AA:BB:CC:DD:EE:FF"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_MAC(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_mac_addr_hyphen() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "AA-BB-CC-DD-EE-FF"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_MAC_ADDR(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_valid_mac_lowercase() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "aa:bb:cc:dd:ee:ff"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_MAC(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_valid_mac_invalid() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "AA:BB:CC:DD:EE"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = IS_VALID_MAC(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Bool(false)));
    drop(dir);
}

// ── MAC_TO_INT / MAC_ADDR_TO_INT ──────────────────────────────────────────────

#[test]
fn test_mac_to_int() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "00:00:00:00:00:01"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = MAC_TO_INT(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(1)));
    drop(dir);
}

#[test]
fn test_mac_addr_to_int_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "00:00:00:00:01:00"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = MAC_ADDR_TO_INT(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Integer(256)));
    drop(dir);
}

// ── NORMALIZE_MAC / FORMAT_MAC ────────────────────────────────────────────────

#[test]
fn test_normalize_mac_from_lowercase() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "aa:bb:cc:dd:ee:ff"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = NORMALIZE_MAC(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("AA:BB:CC:DD:EE:FF".to_string())));
    drop(dir);
}

#[test]
fn test_format_mac_from_hyphens() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "AA-BB-CC-DD-EE-FF"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = FORMAT_MAC(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("AA:BB:CC:DD:EE:FF".to_string())));
    drop(dir);
}

#[test]
fn test_normalize_mac_invalid_returns_null() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "not-a-mac"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = NORMALIZE_MAC(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::Null));
    drop(dir);
}

// ── MAC_OUI ───────────────────────────────────────────────────────────────────

#[test]
fn test_mac_oui() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "AA:BB:CC:DD:EE:FF"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = MAC_OUI(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("AA:BB:CC".to_string())));
    drop(dir);
}

#[test]
fn test_mac_oui_lowercase_normalized() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "aa:bb:cc:dd:ee:ff"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = MAC_OUI(mac) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("r"), Some(&Value::String("AA:BB:CC".to_string())));
    drop(dir);
}
