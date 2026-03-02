/// Integration tests for PQL bitwise and bit-manipulation functions.
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

// 1. BIT_AND(12, 10) → 8  (1100 & 1010 = 1000)
#[test]
fn test_bit_and_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 12, "bv": 10})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_AND(av, bv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 2. BITWISE_AND alias
#[test]
fn test_bitwise_and_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 12, "bv": 10})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BITWISE_AND(av, bv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 3. BIT_OR(12, 10) → 14
#[test]
fn test_bit_or_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 12, "bv": 10})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_OR(av, bv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(14)));
}

// 4. BITWISE_OR alias
#[test]
fn test_bitwise_or_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 12, "bv": 10})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BITWISE_OR(av, bv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(14)));
}

// 5. BIT_XOR(12, 10) → 6
#[test]
fn test_bit_xor_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 12, "bv": 10})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_XOR(av, bv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(6)));
}

// 6. BITWISE_XOR alias
#[test]
fn test_bitwise_xor_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 12, "bv": 10})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BITWISE_XOR(av, bv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(6)));
}

// 7. BIT_NOT(0) → -1 (all bits set)
#[test]
fn test_bit_not_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 0})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_NOT(av) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(-1)));
}

// 8. BITWISE_NOT alias
#[test]
fn test_bitwise_not_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 0})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BITWISE_NOT(av) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(-1)));
}

// 9. BIT_SHIFT_LEFT(1, 3) → 8
#[test]
fn test_bit_shift_left_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 1, "nv": 3})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_SHIFT_LEFT(av, nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 10. SHL alias
#[test]
fn test_shl_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 1, "nv": 3})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = SHL(av, nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 11. BIT_SHIFT_RIGHT(16, 2) → 4
#[test]
fn test_bit_shift_right_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 16, "nv": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_SHIFT_RIGHT(av, nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(4)));
}

// 12. SHR alias
#[test]
fn test_shr_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": 16, "nv": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = SHR(av, nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(4)));
}

// 13. POPCOUNT(7) → 3  (111 has 3 ones)
#[test]
fn test_popcount_seven() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 7})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = POPCOUNT(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// 14. BIT_COUNT alias
#[test]
fn test_bit_count_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 7})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_COUNT(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// 15. POPCOUNT(255) → 8
#[test]
fn test_popcount_255() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 255})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = POPCOUNT(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 16. LEADING_ZEROS(1) → 63
#[test]
fn test_leading_zeros_one() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = LEADING_ZEROS(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(63)));
}

// 17. CLZ alias
#[test]
fn test_clz_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = CLZ(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(63)));
}

// 18. TRAILING_ZEROS(8) → 3  (1000 has 3 trailing zeros)
#[test]
fn test_trailing_zeros_eight() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 8})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = TRAILING_ZEROS(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// 19. CTZ alias
#[test]
fn test_ctz_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 8})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = CTZ(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// 20. BIT_LENGTH(1) → 1
#[test]
fn test_bit_length_one() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_LENGTH(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(1)));
}

// 21. BIT_LENGTH(255) → 8
#[test]
fn test_bit_length_255() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 255})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_LENGTH(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 22. HIGHEST_BIT_POS alias
#[test]
fn test_highest_bit_pos_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 255})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = HIGHEST_BIT_POS(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 23. GET_BIT(8, 3) → 1  (bit 3 of 1000 = 1)
#[test]
fn test_get_bit_set() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 8, "pos": 3})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = GET_BIT(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(1)));
}

// 24. GET_BIT(8, 0) → 0
#[test]
fn test_get_bit_clear() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 8, "pos": 0})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = GET_BIT(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(0)));
}

// 25. BIT_GET alias
#[test]
fn test_bit_get_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 8, "pos": 3})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_GET(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(1)));
}

// 26. SET_BIT(0, 2) → 4
#[test]
fn test_set_bit_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 0, "pos": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = SET_BIT(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(4)));
}

// 27. BIT_SET alias
#[test]
fn test_bit_set_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 0, "pos": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_SET(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(4)));
}

// 28. CLEAR_BIT(7, 1) → 5  (111 → 101)
#[test]
fn test_clear_bit_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 7, "pos": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = CLEAR_BIT(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(5)));
}

// 29. BIT_CLEAR alias
#[test]
fn test_bit_clear_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 7, "pos": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BIT_CLEAR(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(5)));
}

// 30. FLIP_BIT(5, 1) → 7  (101 → 111)
#[test]
fn test_flip_bit_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 5, "pos": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = FLIP_BIT(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(7)));
}

// 31. TOGGLE_BIT alias
#[test]
fn test_toggle_bit_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 5, "pos": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = TOGGLE_BIT(nv, pos) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(7)));
}

// 32. INT_TO_BITS(5) → [0,0,0,0,0,1,0,1]  (8-bit)
#[test]
fn test_int_to_bits_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = INT_TO_BITS(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let expected = Value::Array(vec![
        Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0),
        Value::Integer(0), Value::Integer(1), Value::Integer(0), Value::Integer(1),
    ]);
    assert_eq!(res.rows[0].data.get("r"), Some(&expected));
}

// 33. TO_BIT_ARRAY alias with width 4: [0,1,0,1]
#[test]
fn test_to_bit_array_alias_with_width() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 5, "wv": 4})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = TO_BIT_ARRAY(nv, wv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let expected = Value::Array(vec![
        Value::Integer(0), Value::Integer(1), Value::Integer(0), Value::Integer(1),
    ]);
    assert_eq!(res.rows[0].data.get("r"), Some(&expected));
}

// 34. BITS_TO_INT([1,0,1]) → 5
#[test]
fn test_bits_to_int_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": [1, 0, 1]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BITS_TO_INT(av) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(5)));
}

// 35. FROM_BIT_ARRAY alias
#[test]
fn test_from_bit_array_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"av": [1, 0, 1]})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = FROM_BIT_ARRAY(av) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(5)));
}

// 36. GRAY_CODE(0) → 0
#[test]
fn test_gray_code_zero() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 0})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = GRAY_CODE(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(0)));
}

// 37. GRAY_CODE(1) → 1
#[test]
fn test_gray_code_one() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = GRAY_CODE(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(1)));
}

// 38. GRAY_CODE(2) → 3  (10 → 11)
#[test]
fn test_gray_code_two() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = GRAY_CODE(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// 39. INT_TO_GRAY alias
#[test]
fn test_int_to_gray_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 2})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = INT_TO_GRAY(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(3)));
}

// 40. GRAY_TO_INT(3) → 2
#[test]
fn test_gray_to_int_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 3})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = GRAY_TO_INT(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(2)));
}

// 41. FROM_GRAY_CODE alias
#[test]
fn test_from_gray_code_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 3})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = FROM_GRAY_CODE(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(2)));
}

// 42. IS_POWER_OF_TWO(8) → true
#[test]
fn test_is_power_of_two_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 8})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_POWER_OF_TWO(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

// 43. IS_POWER_OF_TWO(6) → false
#[test]
fn test_is_power_of_two_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 6})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_POWER_OF_TWO(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Bool(false)));
}

// 44. IS_POW2 alias
#[test]
fn test_is_pow2_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 16})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = IS_POW2(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Bool(true)));
}

// 45. NEXT_POWER_OF_TWO(5) → 8
#[test]
fn test_next_power_of_two_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = NEXT_POWER_OF_TWO(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 46. NEXT_POW2 alias
#[test]
fn test_next_pow2_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 5})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = NEXT_POW2(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(res.rows[0].data.get("r"), Some(&Value::Integer(8)));
}

// 47. BYTE_SWAP: check returns integer type
#[test]
fn test_byte_swap_returns_integer() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 1})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BYTE_SWAP(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let val = res.rows[0].data.get("r");
    assert!(matches!(val, Some(Value::Integer(_))), "BYTE_SWAP should return Integer, got {:?}", val);
    // 1i64 byte-swapped: 0x0000000000000001 → 0x0100000000000000
    assert_eq!(val, Some(&Value::Integer(1i64.swap_bytes())));
}

// 48. BSWAP alias
#[test]
fn test_bswap_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"nv": 256})).unwrap();
    let mut p = Parser::new("QUERY t COMPUTE r = BSWAP(nv) SELECT r;");
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let val = res.rows[0].data.get("r");
    assert!(matches!(val, Some(Value::Integer(_))), "BSWAP should return Integer, got {:?}", val);
    assert_eq!(val, Some(&Value::Integer(256i64.swap_bytes())));
}
