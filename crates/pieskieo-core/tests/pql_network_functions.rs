/// Integration tests for PQL network and IP address built-in functions.
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

// ── IP_TO_INT / IPV4_TO_INT ──────────────────────────────────────────────────

#[test]
fn test_ip_to_int() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = IP_TO_INT(ip) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // 192*2^24 + 168*2^16 + 1*2^8 + 1 = 3232235777
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(3232235777)));
    drop(dir);
}

#[test]
fn test_ipv4_to_int_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = IPV4_TO_INT(ip) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(3232235777)));
    drop(dir);
}

// ── INT_TO_IP / INT_TO_IPV4 ──────────────────────────────────────────────────

#[test]
fn test_int_to_ip() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 3232235777_i64})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ip = INT_TO_IP(n) SELECT ip;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ip"), Some(&Value::String("192.168.1.1".to_string())));
    drop(dir);
}

#[test]
fn test_int_to_ipv4_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"n": 3232235777_i64})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ip = INT_TO_IPV4(n) SELECT ip;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ip"), Some(&Value::String("192.168.1.1".to_string())));
    drop(dir);
}

#[test]
fn test_ip_round_trip() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "10.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE back = INT_TO_IP(IP_TO_INT(ip)) SELECT back;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("back"), Some(&Value::String("10.0.0.1".to_string())));
    drop(dir);
}

// ── CIDR_CONTAINS / IP_IN_CIDR ───────────────────────────────────────────────

#[test]
fn test_cidr_contains_true() {
    let (dir, db, ex) = setup();
    // Existing implementation: IP_IN_CIDR(ip, cidr) — ip is first argument
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24", "ip": "192.168.1.5"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = CIDR_CONTAINS(ip, cidr) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_cidr_contains_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24", "ip": "192.168.2.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = CIDR_CONTAINS(ip, cidr) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_ip_in_cidr_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "10.0.0.0/8", "ip": "10.1.2.3"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IP_IN_CIDR(ip, cidr) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── IP_VER ───────────────────────────────────────────────────────────────────

#[test]
fn test_ip_ver_ipv4() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = IP_VER(ip) SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(4)));
    drop(dir);
}

#[test]
fn test_ip_ver_ipv6() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "::1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = IP_VER(ip) SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("v"), Some(&Value::Integer(6)));
    drop(dir);
}

// ── IS_PRIVATE_IP / IS_RFC1918 ───────────────────────────────────────────────

#[test]
fn test_is_private_ip_10() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "10.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_PRIVATE_IP(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_private_ip_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "8.8.8.8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_PRIVATE_IP(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_is_rfc1918_alias_172() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "172.20.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_RFC1918(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_private_ip_192_168() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_PRIVATE_IP(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── CIDR_NETWORK / CIDR_NET_ADDR ─────────────────────────────────────────────

#[test]
fn test_cidr_network() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.100/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE net = CIDR_NETWORK(cidr) SELECT net;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("net"), Some(&Value::String("192.168.1.0".to_string())));
    drop(dir);
}

#[test]
fn test_cidr_net_addr_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "10.0.0.0/8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE net = CIDR_NET_ADDR(cidr) SELECT net;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("net"), Some(&Value::String("10.0.0.0".to_string())));
    drop(dir);
}

// ── CIDR_BROADCAST / CIDR_BCAST ──────────────────────────────────────────────

#[test]
fn test_cidr_broadcast() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bcast = CIDR_BROADCAST(cidr) SELECT bcast;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("bcast"), Some(&Value::String("192.168.1.255".to_string())));
    drop(dir);
}

#[test]
fn test_cidr_bcast_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "172.16.0.0/12"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bcast = CIDR_BCAST(cidr) SELECT bcast;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("bcast"), Some(&Value::String("172.31.255.255".to_string())));
    drop(dir);
}

// ── CIDR_HOST_COUNT / CIDR_HOSTS ─────────────────────────────────────────────

#[test]
fn test_cidr_host_count_24() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.1.0/24"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hosts = CIDR_HOST_COUNT(cidr) SELECT hosts;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("hosts"), Some(&Value::Integer(254)));
    drop(dir);
}

#[test]
fn test_cidr_hosts_alias_8() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "10.0.0.0/8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hosts = CIDR_HOSTS(cidr) SELECT hosts;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // /8: 2^24 - 2 = 16777214
    assert_eq!(r.rows[0].data.get("hosts"), Some(&Value::Integer(16777214)));
    drop(dir);
}

// ── CIDR_PREFIX_LEN / SUBNET_BITS ────────────────────────────────────────────

#[test]
fn test_cidr_prefix_len() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "10.0.0.0/8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE plen = CIDR_PREFIX_LEN(cidr) SELECT plen;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("plen"), Some(&Value::Integer(8)));
    drop(dir);
}

#[test]
fn test_subnet_bits_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"cidr": "192.168.0.0/16"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE plen = SUBNET_BITS(cidr) SELECT plen;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("plen"), Some(&Value::Integer(16)));
    drop(dir);
}

// ── IS_LOOPBACK / IS_LOCALHOST ───────────────────────────────────────────────

#[test]
fn test_is_loopback_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "127.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_LOOPBACK(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_loopback_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "8.8.8.8"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_LOOPBACK(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_is_localhost_alias_ipv6() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "::1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_LOCALHOST(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── IS_MULTICAST_IP / IS_MCAST ───────────────────────────────────────────────

#[test]
fn test_is_multicast_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "224.0.0.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_MULTICAST_IP(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_multicast_false() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_MULTICAST_IP(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_is_mcast_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "239.255.255.255"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_MCAST(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── IS_VALID_IPV4 / VALID_IPV4 ───────────────────────────────────────────────

#[test]
fn test_is_valid_ipv4_true() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.1"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_VALID_IPV4(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

#[test]
fn test_is_valid_ipv4_false_out_of_range() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "999.999.999.999"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = IS_VALID_IPV4(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(false)));
    drop(dir);
}

#[test]
fn test_valid_ipv4_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "0.0.0.0"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE result = VALID_IPV4(ip) SELECT result;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("result"), Some(&Value::Bool(true)));
    drop(dir);
}

// ── MAC_VENDOR_PREFIX / OUI_PREFIX ───────────────────────────────────────────

#[test]
fn test_mac_vendor_prefix() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "aa:bb:cc:dd:ee:ff"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE oui = MAC_VENDOR_PREFIX(mac) SELECT oui;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("oui"), Some(&Value::String("AA:BB:CC".to_string())));
    drop(dir);
}

#[test]
fn test_oui_prefix_alias_with_dashes() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"mac": "00-1A-2B-3C-4D-5E"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE oui = OUI_PREFIX(mac) SELECT oui;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("oui"), Some(&Value::String("00:1A:2B".to_string())));
    drop(dir);
}

// ── IP_ANONYMIZE / MASK_IP ───────────────────────────────────────────────────

#[test]
fn test_ip_anonymize() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "192.168.1.100"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE anon = IP_ANONYMIZE(ip) SELECT anon;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("anon"), Some(&Value::String("192.168.1.0".to_string())));
    drop(dir);
}

#[test]
fn test_mask_ip_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"ip": "10.20.30.40"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE anon = MASK_IP(ip) SELECT anon;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("anon"), Some(&Value::String("10.20.30.0".to_string())));
    drop(dir);
}

// ── PORT_NAME / WELL_KNOWN_PORT ──────────────────────────────────────────────

#[test]
fn test_port_name_http() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"port": 80})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE name = PORT_NAME(port) SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("name"), Some(&Value::String("http".to_string())));
    drop(dir);
}

#[test]
fn test_port_name_https() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"port": 443})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE name = PORT_NAME(port) SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("name"), Some(&Value::String("https".to_string())));
    drop(dir);
}

#[test]
fn test_port_name_ssh() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"port": 22})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE name = PORT_NAME(port) SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("name"), Some(&Value::String("ssh".to_string())));
    drop(dir);
}

#[test]
fn test_well_known_port_alias() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"port": 53})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE name = WELL_KNOWN_PORT(port) SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("name"), Some(&Value::String("dns".to_string())));
    drop(dir);
}

#[test]
fn test_port_name_unknown() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"port": 9999})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE name = PORT_NAME(port) SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("name"), Some(&Value::String("unknown".to_string())));
    drop(dir);
}

#[test]
fn test_port_name_mysql() {
    let (dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"port": 3306})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE name = PORT_NAME(port) SELECT name;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("name"), Some(&Value::String("mysql".to_string())));
    drop(dir);
}
