/// Integration tests for PQL UUID, ID generation, Base62, and Slug functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db(ns: &str) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), serde_json::json!({})).unwrap();
    (dir, db, ex)
}

// ── IS_VALID_UUID / UUID_VALID ──────────────────────────────────────────────

#[test]
fn test_is_valid_uuid_valid() {
    let (_dir, _db, ex) = make_db("t1");
    let mut p = Parser::new(r#"QUERY t1 COMPUTE sv = IS_VALID_UUID("550e8400-e29b-41d4-a716-446655440000") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::Bool(true)) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

#[test]
fn test_is_valid_uuid_invalid_string() {
    let (_dir, _db, ex) = make_db("t2");
    let mut p = Parser::new(r#"QUERY t2 COMPUTE sv = IS_VALID_UUID("not-a-uuid") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::Bool(false)) => {}
        other => panic!("expected Bool(false), got {:?}", other),
    }
}

#[test]
fn test_is_valid_uuid_empty_string() {
    let (_dir, _db, ex) = make_db("t3");
    let mut p = Parser::new(r#"QUERY t3 COMPUTE sv = IS_VALID_UUID("") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::Bool(false)) => {}
        other => panic!("expected Bool(false), got {:?}", other),
    }
}

#[test]
fn test_uuid_valid_alias() {
    let (_dir, _db, ex) = make_db("t4");
    let mut p = Parser::new(r#"QUERY t4 COMPUTE sv = UUID_VALID("550e8400-e29b-41d4-a716-446655440000") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::Bool(true)) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

// ── UUID_VERSION / UUID_VER ─────────────────────────────────────────────────

#[test]
fn test_uuid_version_v4() {
    let (_dir, _db, ex) = make_db("t5");
    // UUID v4 has '4' at position 14 (0-indexed)
    let mut p = Parser::new(r#"QUERY t5 COMPUTE nv = UUID_VERSION("550e8400-e29b-41d4-a716-446655440000") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 4, "expected version 4, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_uuid_ver_alias() {
    let (_dir, _db, ex) = make_db("t6");
    let mut p = Parser::new(r#"QUERY t6 COMPUTE nv = UUID_VER("550e8400-e29b-41d4-a716-446655440000") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 4, "expected version 4, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── UUID_TO_HEX / UUID_STRIP_DASHES ────────────────────────────────────────

#[test]
fn test_uuid_to_hex() {
    let (_dir, _db, ex) = make_db("t7");
    let mut p = Parser::new(r#"QUERY t7 COMPUTE sv = UUID_TO_HEX("550e8400-e29b-41d4-a716-446655440000") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "550e8400e29b41d4a716446655440000", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_uuid_strip_dashes_alias() {
    let (_dir, _db, ex) = make_db("t8");
    let mut p = Parser::new(r#"QUERY t8 COMPUTE sv = UUID_STRIP_DASHES("550e8400-e29b-41d4-a716-446655440000") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "550e8400e29b41d4a716446655440000", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

// ── HEX_TO_UUID / FORMAT_UUID ───────────────────────────────────────────────

#[test]
fn test_hex_to_uuid() {
    let (_dir, _db, ex) = make_db("t9");
    let mut p = Parser::new(r#"QUERY t9 COMPUTE uuid_str = HEX_TO_UUID("550e8400e29b41d4a716446655440000") SELECT uuid_str;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("uuid_str") {
        Some(Value::String(s)) => assert_eq!(s, "550e8400-e29b-41d4-a716-446655440000", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_format_uuid_alias() {
    let (_dir, _db, ex) = make_db("t10");
    let mut p = Parser::new(r#"QUERY t10 COMPUTE uuid_str = FORMAT_UUID("550e8400e29b41d4a716446655440000") SELECT uuid_str;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("uuid_str") {
        Some(Value::String(s)) => assert_eq!(s, "550e8400-e29b-41d4-a716-446655440000", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_uuid_to_hex_then_hex_to_uuid_round_trip() {
    // Round-trip: UUID_TO_HEX produces a hex, HEX_TO_UUID restores it.
    // Test both directions using stored field values to avoid nested literal parsing issues.
    let original = "550e8400-e29b-41d4-a716-446655440000";
    let expected_hex = "550e8400e29b41d4a716446655440000";

    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t11"), Uuid::new_v4(),
        serde_json::json!({"uuid_str": original, "hex_val": expected_hex})).unwrap();

    // Step 1: UUID_TO_HEX of the stored UUID string
    let mut p1 = Parser::new(r#"QUERY t11 WHERE uuid_str != null COMPUTE sv = UUID_TO_HEX(uuid_str) SELECT sv;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let hex_result = match r1.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String from UUID_TO_HEX, got {:?}", other),
    };
    assert_eq!(hex_result, expected_hex, "UUID_TO_HEX produced unexpected hex");

    // Step 2: HEX_TO_UUID of the stored hex string
    let mut p2 = Parser::new(r#"QUERY t11 WHERE hex_val != null COMPUTE sv = HEX_TO_UUID(hex_val) SELECT sv;"#);
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    let uuid_result = match r2.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String from HEX_TO_UUID, got {:?}", other),
    };
    assert_eq!(uuid_result, original, "HEX_TO_UUID round-trip failed, got {}", uuid_result);
}

// ── UUID_NAMESPACE / UUID_NS ────────────────────────────────────────────────

#[test]
fn test_uuid_namespace_dns() {
    let (_dir, _db, ex) = make_db("t12");
    let mut p = Parser::new(r#"QUERY t12 COMPUTE uuid_str = UUID_NAMESPACE("dns") SELECT uuid_str;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("uuid_str") {
        Some(Value::String(s)) => assert_eq!(s, "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_uuid_namespace_url() {
    let (_dir, _db, ex) = make_db("t13");
    let mut p = Parser::new(r#"QUERY t13 COMPUTE uuid_str = UUID_NAMESPACE("url") SELECT uuid_str;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("uuid_str") {
        Some(Value::String(s)) => assert_eq!(s, "6ba7b811-9dad-11d1-80b4-00c04fd430c8", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_uuid_ns_alias() {
    let (_dir, _db, ex) = make_db("t14");
    let mut p = Parser::new(r#"QUERY t14 COMPUTE uuid_str = UUID_NS("dns") SELECT uuid_str;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("uuid_str") {
        Some(Value::String(s)) => assert_eq!(s, "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

// ── RANDOM_ID / GEN_RANDOM_ID ───────────────────────────────────────────────

#[test]
fn test_random_id_length_with_seed() {
    let (_dir, _db, ex) = make_db("t15");
    let mut p = Parser::new(r#"QUERY t15 COMPUTE sv = RANDOM_ID(8, 42) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 8, "expected length 8, got {}", s.len());
            assert!(s.chars().all(|c| c.is_alphanumeric()), "expected alphanumeric, got {}", s);
        }
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_random_id_same_seed_deterministic() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t16"), Uuid::new_v4(), serde_json::json!({})).unwrap();

    let mut p1 = Parser::new(r#"QUERY t16 COMPUTE sv = RANDOM_ID(8, 42) SELECT sv;"#);
    let mut p2 = Parser::new(r#"QUERY t16 COMPUTE sv = RANDOM_ID(8, 42) SELECT sv;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();

    let v1 = match r1.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    let v2 = match r2.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    assert_eq!(v1, v2, "same seed must produce same result");
}

#[test]
fn test_random_id_different_seeds_differ() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t17"), Uuid::new_v4(), serde_json::json!({})).unwrap();

    let mut p1 = Parser::new(r#"QUERY t17 COMPUTE sv = RANDOM_ID(8, 42) SELECT sv;"#);
    let mut p2 = Parser::new(r#"QUERY t17 COMPUTE sv = RANDOM_ID(8, 99) SELECT sv;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();

    let v1 = match r1.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    let v2 = match r2.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    assert_ne!(v1, v2, "different seeds should produce different results");
}

#[test]
fn test_gen_random_id_alias() {
    let (_dir, _db, ex) = make_db("t18");
    let mut p = Parser::new(r#"QUERY t18 COMPUTE sv = GEN_RANDOM_ID(8, 42) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s.len(), 8, "expected length 8, got {}", s.len()),
        other => panic!("expected String, got {:?}", other),
    }
}

// ── SHORT_UUID / UUID_SHORT ─────────────────────────────────────────────────

#[test]
fn test_short_uuid_length() {
    let (_dir, _db, ex) = make_db("t19");
    let mut p = Parser::new(r#"QUERY t19 COMPUTE sv = SHORT_UUID(42) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s.len(), 12, "expected length 12, got {}", s.len()),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_uuid_short_alias() {
    let (_dir, _db, ex) = make_db("t20");
    let mut p = Parser::new(r#"QUERY t20 COMPUTE sv = UUID_SHORT(42) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s.len(), 12, "expected length 12, got {}", s.len()),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_short_uuid_deterministic() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t21"), Uuid::new_v4(), serde_json::json!({})).unwrap();

    let mut p1 = Parser::new(r#"QUERY t21 COMPUTE sv = SHORT_UUID(42) SELECT sv;"#);
    let mut p2 = Parser::new(r#"QUERY t21 COMPUTE sv = SHORT_UUID(42) SELECT sv;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();

    let v1 = match r1.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    let v2 = match r2.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };
    assert_eq!(v1, v2, "same seed must produce same result");
}

// ── BASE62_ENCODE / TO_BASE62 ───────────────────────────────────────────────

#[test]
fn test_base62_encode_zero() {
    let (_dir, _db, ex) = make_db("t22");
    let mut p = Parser::new(r#"QUERY t22 COMPUTE sv = BASE62_ENCODE(0) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "0", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_base62_encode_61() {
    let (_dir, _db, ex) = make_db("t23");
    let mut p = Parser::new(r#"QUERY t23 COMPUTE sv = BASE62_ENCODE(61) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "Z", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_base62_encode_62() {
    let (_dir, _db, ex) = make_db("t24");
    let mut p = Parser::new(r#"QUERY t24 COMPUTE sv = BASE62_ENCODE(62) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "10", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_to_base62_alias() {
    let (_dir, _db, ex) = make_db("t25");
    let mut p = Parser::new(r#"QUERY t25 COMPUTE sv = TO_BASE62(62) SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "10", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

// ── BASE62_DECODE / FROM_BASE62 ─────────────────────────────────────────────

#[test]
fn test_base62_decode_zero() {
    let (_dir, _db, ex) = make_db("t26");
    let mut p = Parser::new(r#"QUERY t26 COMPUTE nv = BASE62_DECODE("0") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 0, "got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_base62_decode_z() {
    let (_dir, _db, ex) = make_db("t27");
    let mut p = Parser::new(r#"QUERY t27 COMPUTE nv = BASE62_DECODE("Z") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 61, "got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_base62_decode_10() {
    let (_dir, _db, ex) = make_db("t28");
    let mut p = Parser::new(r#"QUERY t28 COMPUTE nv = BASE62_DECODE("10") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 62, "got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_from_base62_alias() {
    let (_dir, _db, ex) = make_db("t29");
    let mut p = Parser::new(r#"QUERY t29 COMPUTE nv = FROM_BASE62("10") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 62, "got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_base62_round_trip() {
    // Round-trip: BASE62_ENCODE then BASE62_DECODE must recover the original integer.
    // Uses a stored field to avoid nested literal call parsing issues.
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t30"), Uuid::new_v4(),
        serde_json::json!({"num": 12345, "encoded": "3d7"})).unwrap();

    // Verify BASE62_ENCODE(12345)
    let mut p1 = Parser::new(r#"QUERY t30 WHERE num != null COMPUTE sv = BASE62_ENCODE(num) SELECT sv;"#);
    let r1 = ex.execute(p1.parse().unwrap()).unwrap();
    let encoded = match r1.rows[0].data.get("sv") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };

    // Verify BASE62_DECODE of the encoded value restores 12345
    let mut p2 = Parser::new(r#"QUERY t30 WHERE encoded != null COMPUTE nv = BASE62_DECODE(encoded) SELECT nv;"#);
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    let decoded = match r2.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => *n,
        other => panic!("expected Integer, got {:?}", other),
    };
    // BASE62_ENCODE(12345): 12345 / 62 = 199 r 7, 199 / 62 = 3 r 13 ('d'), 3 / 62 = 0 r 3
    // chars[3]='3', chars[13]='d', chars[7]='7' -> "3d7"
    assert_eq!(&encoded, "3d7", "BASE62_ENCODE(12345) expected '3d7', got {}", encoded);
    assert_eq!(decoded, 12345, "BASE62_DECODE round-trip failed, got {}", decoded);
}

// ── SLUG_FROM_STRING / GEN_SLUG ─────────────────────────────────────────────

#[test]
fn test_slug_from_string_hello_world() {
    let (_dir, _db, ex) = make_db("t31");
    let mut p = Parser::new(r#"QUERY t31 COMPUTE sv = SLUG_FROM_STRING("Hello World!") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "hello-world", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_slug_from_string_multiple_spaces() {
    let (_dir, _db, ex) = make_db("t32");
    let mut p = Parser::new(r#"QUERY t32 COMPUTE sv = SLUG_FROM_STRING("  foo  bar  ") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "foo-bar", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

#[test]
fn test_gen_slug_alias() {
    let (_dir, _db, ex) = make_db("t33");
    let mut p = Parser::new(r#"QUERY t33 COMPUTE sv = GEN_SLUG("Hello World!") SELECT sv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("sv") {
        Some(Value::String(s)) => assert_eq!(s, "hello-world", "got {}", s),
        other => panic!("expected String, got {:?}", other),
    }
}

// ── UUID_COMPARE / COMPARE_UUIDS ────────────────────────────────────────────

#[test]
fn test_uuid_compare_less_than() {
    let (_dir, _db, ex) = make_db("t34");
    let mut p = Parser::new(r#"QUERY t34 COMPUTE nv = UUID_COMPARE("a", "b") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, -1, "expected -1, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_uuid_compare_greater_than() {
    let (_dir, _db, ex) = make_db("t35");
    let mut p = Parser::new(r#"QUERY t35 COMPUTE nv = UUID_COMPARE("b", "a") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 1, "expected 1, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_uuid_compare_equal() {
    let (_dir, _db, ex) = make_db("t36");
    let mut p = Parser::new(r#"QUERY t36 COMPUTE nv = UUID_COMPARE("a", "a") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, 0, "expected 0, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_compare_uuids_alias() {
    let (_dir, _db, ex) = make_db("t37");
    let mut p = Parser::new(r#"QUERY t37 COMPUTE nv = COMPARE_UUIDS("a", "b") SELECT nv;"#);
    let result = ex.execute(p.parse().unwrap()).unwrap();
    match result.rows[0].data.get("nv") {
        Some(Value::Integer(n)) => assert_eq!(*n, -1, "expected -1, got {}", n),
        other => panic!("expected Integer, got {:?}", other),
    }
}
