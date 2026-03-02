/// Integration tests for PQL built-in chemistry and periodic table functions.
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

fn get_float(r: &pieskieo_core::pql::QueryResult, key: &str) -> f64 {
    match r.rows[0].data.get(key) {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for key '{}', got {:?}", key, other),
    }
}

fn get_int(r: &pieskieo_core::pql::QueryResult, key: &str) -> i64 {
    match r.rows[0].data.get(key) {
        Some(Value::Integer(i)) => *i,
        other => panic!("expected Integer for key '{}', got {:?}", key, other),
    }
}

fn get_string(r: &pieskieo_core::pql::QueryResult, key: &str) -> String {
    match r.rows[0].data.get(key) {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String for key '{}', got {:?}", key, other),
    }
}

fn get_bool(r: &pieskieo_core::pql::QueryResult, key: &str) -> bool {
    match r.rows[0].data.get(key) {
        Some(Value::Bool(b)) => *b,
        other => panic!("expected Boolean for key '{}', got {:?}", key, other),
    }
}

fn is_null(r: &pieskieo_core::pql::QueryResult, key: &str) -> bool {
    matches!(r.rows[0].data.get(key), Some(Value::Null))
}

// ── Element Lookup ────────────────────────────────────────────────────────────

#[test]
fn test_element_symbol_hydrogen() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sym = ELEMENT_SYMBOL(znum) SELECT sym;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "sym"), "H");
}

#[test]
fn test_element_symbol_iron() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 26})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sym = ELEMENT_SYMBOL(znum) SELECT sym;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "sym"), "Fe");
}

#[test]
fn test_elem_symbol_alias_krypton() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 36})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sym = ELEM_SYMBOL(znum) SELECT sym;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "sym"), "Kr");
}

#[test]
fn test_element_symbol_out_of_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 999})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sym = ELEMENT_SYMBOL(znum) SELECT sym;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(is_null(&r, "sym"), "out-of-range z should return Null");
}

#[test]
fn test_element_name_carbon() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 6})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nm = ELEMENT_NAME(znum) SELECT nm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "nm"), "Carbon");
}

#[test]
fn test_elem_name_alias_gold() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 29})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nm = ELEM_NAME(znum) SELECT nm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "nm"), "Copper");
}

#[test]
fn test_atomic_mass_by_number() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = ELEMENT_ATOMIC_MASS(znum) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let m = get_float(&r, "m");
    assert!((m - 1.008).abs() < 0.001, "H mass should be ~1.008, got {}", m);
}

#[test]
fn test_atomic_mass_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = ATOMIC_MASS("Fe") SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let m = get_float(&r, "m");
    assert!((m - 55.845).abs() < 0.01, "Fe mass should be ~55.845, got {}", m);
}

#[test]
fn test_element_period_hydrogen() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE per = ELEMENT_PERIOD(znum) SELECT per;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "per"), 1);
}

#[test]
fn test_element_period_iron() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 26})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE per = ELEM_PERIOD(znum) SELECT per;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "per"), 4);
}

#[test]
fn test_element_group_sodium() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 11})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE grp = ELEMENT_GROUP(znum) SELECT grp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "grp"), 1);
}

#[test]
fn test_element_group_chlorine() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 17})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE grp = ELEM_GROUP(znum) SELECT grp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "grp"), 17);
}

#[test]
fn test_element_category_iron() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 26})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cat = ELEMENT_CATEGORY(znum) SELECT cat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "cat"), "transition metal");
}

#[test]
fn test_element_category_neon() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 10})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cat = ELEM_CATEGORY(znum) SELECT cat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "cat"), "noble gas");
}

#[test]
fn test_atomic_number_fe() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE zn = ATOMIC_NUMBER("Fe") SELECT zn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "zn"), 26);
}

#[test]
fn test_elem_z_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE zn = ELEM_Z("H") SELECT zn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "zn"), 1);
}

#[test]
fn test_atomic_number_unknown() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE zn = ATOMIC_NUMBER("Xx") SELECT zn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(is_null(&r, "zn"), "unknown symbol should return Null");
}

#[test]
fn test_element_info_carbon() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 6})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE info = ELEMENT_INFO(znum) SELECT info;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("info") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("symbol"), Some(&Value::String("C".to_string())));
            assert_eq!(obj.get("name"), Some(&Value::String("Carbon".to_string())));
            assert_eq!(obj.get("atomic_number"), Some(&Value::Integer(6)));
            assert_eq!(obj.get("period"), Some(&Value::Integer(2)));
            assert_eq!(obj.get("group"), Some(&Value::Integer(14)));
            assert_eq!(obj.get("category"), Some(&Value::String("nonmetal".to_string())));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_element_alias_by_symbol() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE info = ELEMENT("Na") SELECT info;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("info") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("symbol"), Some(&Value::String("Na".to_string())));
            assert_eq!(obj.get("atomic_number"), Some(&Value::Integer(11)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── Formula Parsing ───────────────────────────────────────────────────────────

#[test]
fn test_molecular_weight_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = MOLECULAR_WEIGHT("H2O") SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mw = get_float(&r, "mw");
    // H2O = 2*1.008 + 15.999 = 18.015
    assert!((mw - 18.015).abs() < 0.1, "H2O mw should be ~18.015, got {}", mw);
}

#[test]
fn test_molecular_weight_co2() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = MOL_WEIGHT("CO2") SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mw = get_float(&r, "mw");
    // CO2 = 12.011 + 2*15.999 = 44.009
    assert!((mw - 44.009).abs() < 0.1, "CO2 mw should be ~44.009, got {}", mw);
}

#[test]
fn test_molecular_weight_glucose() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = MOLECULAR_WEIGHT("C6H12O6") SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mw = get_float(&r, "mw");
    // C6H12O6 = 6*12.011 + 12*1.008 + 6*15.999 = 72.066 + 12.096 + 95.994 = 180.156
    assert!((mw - 180.156).abs() < 0.5, "C6H12O6 mw should be ~180.156, got {}", mw);
}

#[test]
fn test_molecular_weight_nacl() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = MOLECULAR_WEIGHT("NaCl") SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mw = get_float(&r, "mw");
    // NaCl = 22.990 + 35.453 = 58.443
    assert!((mw - 58.443).abs() < 0.1, "NaCl mw should be ~58.443, got {}", mw);
}

#[test]
fn test_formula_elements_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE elems = FORMULA_ELEMENTS("H2O") SELECT elems;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("elems") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("H"), Some(&Value::Integer(2)));
            assert_eq!(obj.get("O"), Some(&Value::Integer(1)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_parse_formula_alias_glucose() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE elems = PARSE_FORMULA("C6H12O6") SELECT elems;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("elems") {
        Some(Value::Object(obj)) => {
            assert_eq!(obj.get("C"), Some(&Value::Integer(6)));
            assert_eq!(obj.get("H"), Some(&Value::Integer(12)));
            assert_eq!(obj.get("O"), Some(&Value::Integer(6)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_formula_atom_count_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = FORMULA_ATOM_COUNT("H2O") SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "n"), 3);
}

#[test]
fn test_total_atoms_alias_glucose() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = TOTAL_ATOMS("C6H12O6") SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "n"), 24);
}

#[test]
fn test_is_valid_formula_true() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = IS_VALID_FORMULA("H2O") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(get_bool(&r, "v"));
}

#[test]
fn test_valid_formula_alias_false() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = VALID_FORMULA("Xy3Z") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!get_bool(&r, "v"));
}

#[test]
fn test_empirical_formula_glucose() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ef = EMPIRICAL_FORMULA("C6H12O6") SELECT ef;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "ef"), "CH2O");
}

#[test]
fn test_simplify_formula_alias_h2o2() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ef = SIMPLIFY_FORMULA("H2O2") SELECT ef;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // H2O2 -> HO (GCD=2)
    assert_eq!(get_string(&r, "ef"), "HO");
}

// ── Stoichiometry ─────────────────────────────────────────────────────────────

#[test]
fn test_moles_to_grams_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE g = MOLES_TO_GRAMS(2.0, "H2O") SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let g = get_float(&r, "g");
    // 2 mol * 18.015 g/mol ≈ 36.03
    assert!((g - 36.03).abs() < 0.5, "2 mol H2O should be ~36.03g, got {}", g);
}

#[test]
fn test_mol_to_g_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE g = MOL_TO_G(1.0, "NaCl") SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let g = get_float(&r, "g");
    // 1 mol NaCl ≈ 58.443g
    assert!((g - 58.443).abs() < 0.5, "1 mol NaCl should be ~58.443g, got {}", g);
}

#[test]
fn test_grams_to_moles_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mol = GRAMS_TO_MOLES(18.015, "H2O") SELECT mol;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mol = get_float(&r, "mol");
    assert!((mol - 1.0).abs() < 0.01, "18.015g H2O should be ~1 mol, got {}", mol);
}

#[test]
fn test_g_to_mol_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mol = G_TO_MOL(58.443, "NaCl") SELECT mol;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mol = get_float(&r, "mol");
    assert!((mol - 1.0).abs() < 0.02, "58.443g NaCl should be ~1 mol, got {}", mol);
}

#[test]
fn test_molarity() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = MOLARITY(2.0, 4.0) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = get_float(&r, "c");
    assert!((c - 0.5).abs() < 1e-9, "2 mol / 4 L should be 0.5 M, got {}", c);
}

#[test]
fn test_concentration_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = CONCENTRATION(0.5, 0.25) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = get_float(&r, "c");
    assert!((c - 2.0).abs() < 1e-9, "0.5 mol / 0.25 L should be 2 M, got {}", c);
}

#[test]
fn test_dilution_volume() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // C1*V1 = C2*V2 => 1.0 * 1.0 / 0.5 = 2.0
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = DILUTION_VOLUME(1.0, 1.0, 0.5) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 2.0).abs() < 1e-9, "dilution volume should be 2.0, got {}", v2);
}

#[test]
fn test_dilute_to_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = DILUTE_TO(2.0, 0.5, 1.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 1.0).abs() < 1e-9, "dilute_to result should be 1.0, got {}", v2);
}

#[test]
fn test_percent_composition_hydrogen_in_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pct = PERCENT_COMPOSITION("H", "H2O") SELECT pct;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pct = get_float(&r, "pct");
    // 2*1.008 / 18.015 * 100 ≈ 11.19%
    assert!((pct - 11.19).abs() < 0.5, "H in H2O should be ~11.19%, got {}", pct);
}

#[test]
fn test_mass_percent_alias_oxygen_in_water() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pct = MASS_PERCENT("O", "H2O") SELECT pct;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pct = get_float(&r, "pct");
    // 15.999 / 18.015 * 100 ≈ 88.81%
    assert!((pct - 88.81).abs() < 0.5, "O in H2O should be ~88.81%, got {}", pct);
}

#[test]
fn test_limiting_reagent_a_limits() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // amount_a=1, coeff_a=2 -> ratio=0.5; amount_b=3, coeff_b=1 -> ratio=3 -> A limits
    let mut p = Parser::new(r#"QUERY t COMPUTE lr = LIMITING_REAGENT(1.0, 2.0, 3.0, 1.0) SELECT lr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "lr"), "A");
}

#[test]
fn test_stoich_ratio_alias_b_limits() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // amount_a=10, coeff_a=1 -> ratio=10; amount_b=2, coeff_b=3 -> ratio≈0.667 -> B limits
    let mut p = Parser::new(r#"QUERY t COMPUTE lr = STOICH_RATIO(10.0, 1.0, 2.0, 3.0) SELECT lr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "lr"), "B");
}

// ── Physical Chemistry ────────────────────────────────────────────────────────

#[test]
fn test_boyle_law_volume() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // P1=2, V1=3, P2=6 -> V2 = 2*3/6 = 1
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = BOYLE_LAW_VOLUME(2.0, 3.0, 6.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 1.0).abs() < 1e-9, "Boyle V2 should be 1.0, got {}", v2);
}

#[test]
fn test_boyle_v2_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // P1=1, V1=10, P2=2 -> V2 = 5
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = BOYLE_V2(1.0, 10.0, 2.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 5.0).abs() < 1e-9, "BOYLE_V2 should be 5.0, got {}", v2);
}

#[test]
fn test_charles_law_volume() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // V1=2, T1=300, T2=600 -> V2 = 4
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = CHARLES_LAW_VOLUME(2.0, 300.0, 600.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 4.0).abs() < 1e-9, "Charles V2 should be 4.0, got {}", v2);
}

#[test]
fn test_charles_v2_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // V1=10, T1=500, T2=250 -> V2 = 5
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = CHARLES_V2(10.0, 500.0, 250.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 5.0).abs() < 1e-9, "CHARLES_V2 should be 5.0, got {}", v2);
}

#[test]
fn test_combined_gas_law_solve_v2() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // P1=1, V1=2, T1=300, P2=2, V2=null, T2=300 -> V2 = P1*V1*T2/(T1*P2) = 1*2*300/(300*2) = 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COMBINED_GAS_LAW(1.0, 2.0, 300.0, 2.0, NULL, 300.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(v2)) = obj.get("V2") {
                assert!((v2 - 1.0).abs() < 1e-6, "COMBINED_GAS V2 should be 1.0, got {}", v2);
            } else {
                panic!("expected V2 float in result object, got {:?}", obj);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_henderson_hasselbalch() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // At equal concentrations: pH = pKa
    let mut p = Parser::new(r#"QUERY t COMPUTE ph = HENDERSON_HASSELBALCH(4.74, 1.0, 1.0) SELECT ph;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ph = get_float(&r, "ph");
    assert!((ph - 4.74).abs() < 0.001, "H-H at equal concs should equal pKa=4.74, got {}", ph);
}

#[test]
fn test_henderson_hasselbalch_eq_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // pKa=4.74, [A-]=10*[HA] -> pH = 4.74 + log10(10) = 5.74
    let mut p = Parser::new(r#"QUERY t COMPUTE ph = HENDERSON_HASSELBALCH_EQ(4.74, 1.0, 10.0) SELECT ph;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ph = get_float(&r, "ph");
    assert!((ph - 5.74).abs() < 0.01, "H-H with 10:1 base:acid should be 5.74, got {}", ph);
}

#[test]
fn test_ph_from_concentration() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // [H+] = 1e-7 -> pH = 7
    let mut p = Parser::new(r#"QUERY t COMPUTE ph = PH_FROM_CONCENTRATION(0.0000001) SELECT ph;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ph = get_float(&r, "ph");
    assert!((ph - 7.0).abs() < 0.001, "pH from 1e-7 should be 7.0, got {}", ph);
}

#[test]
fn test_ph_from_conc_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // [H+] = 0.01 -> pH = 2
    let mut p = Parser::new(r#"QUERY t COMPUTE ph = PH_FROM_CONC(0.01) SELECT ph;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ph = get_float(&r, "ph");
    assert!((ph - 2.0).abs() < 0.001, "pH from 0.01 should be 2.0, got {}", ph);
}

#[test]
fn test_concentration_from_ph() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // pH=7 -> [H+] = 1e-7
    let mut p = Parser::new(r#"QUERY t COMPUTE hc = CONCENTRATION_FROM_PH(7.0) SELECT hc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let hc = get_float(&r, "hc");
    assert!((hc - 1e-7).abs() < 1e-9, "[H+] from pH 7 should be 1e-7, got {}", hc);
}

#[test]
fn test_h_from_ph_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // pH=2 -> [H+] = 0.01
    let mut p = Parser::new(r#"QUERY t COMPUTE hc = H_FROM_PH(2.0) SELECT hc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let hc = get_float(&r, "hc");
    assert!((hc - 0.01).abs() < 1e-6, "[H+] from pH 2 should be 0.01, got {}", hc);
}

#[test]
fn test_buffer_capacity() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // At pH=pKa: beta = 0.576 * total_conc
    let mut p = Parser::new(r#"QUERY t COMPUTE beta = BUFFER_CAPACITY(4.74, 1.0) SELECT beta;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let beta = get_float(&r, "beta");
    assert!((beta - 0.576).abs() < 0.001, "buffer capacity at pKa should be 0.576, got {}", beta);
}

#[test]
fn test_buffering_capacity_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE beta = BUFFERING_CAPACITY(7.4, 0.1) SELECT beta;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let beta = get_float(&r, "beta");
    assert!((beta - 0.0576).abs() < 0.001, "buffer capacity should be 0.0576, got {}", beta);
}

#[test]
fn test_activation_energy_arrhenius() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Ea = -R * ln(k2/k1) / (1/T2 - 1/T1)
    // k1=1, k2=10, T1=300, T2=400
    // Ea = -8.314 * ln(10) / (1/400 - 1/300) = -8.314*2.3026 / (-8.333e-4) ≈ 22941 J/mol
    let mut p = Parser::new(r#"QUERY t COMPUTE ea = ACTIVATION_ENERGY(1.0, 10.0, 300.0, 400.0) SELECT ea;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ea = get_float(&r, "ea");
    assert!(ea > 20000.0 && ea < 30000.0, "Ea should be ~22941 J/mol, got {}", ea);
}

#[test]
fn test_arrhenius_ea_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ea = ARRHENIUS_EA(1.0, 2.0, 298.0, 308.0) SELECT ea;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ea = get_float(&r, "ea");
    assert!(ea > 0.0, "activation energy should be positive, got {}", ea);
}

#[test]
fn test_rate_constant_arrhenius() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // k = A * exp(-Ea/(R*T))
    // A=1e13, Ea=50000, T=298 -> k = 1e13 * exp(-50000/(8.314*298))
    let mut p = Parser::new(r#"QUERY t COMPUTE k = RATE_CONSTANT_ARRHENIUS(1000000000000.0, 50000.0, 298.0) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k = get_float(&r, "k");
    assert!(k > 0.0, "rate constant should be positive, got {}", k);
}

#[test]
fn test_arrhenius_k_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // A=1, Ea=0 -> k = 1 regardless of T
    let mut p = Parser::new(r#"QUERY t COMPUTE k = ARRHENIUS_K(1.0, 0.0, 300.0) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k = get_float(&r, "k");
    assert!((k - 1.0).abs() < 1e-9, "k with Ea=0 and A=1 should be 1.0, got {}", k);
}

// ── Additional edge cases ─────────────────────────────────────────────────────

#[test]
fn test_element_symbol_helium() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE sym = ELEMENT_SYMBOL(znum) SELECT sym;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "sym"), "He");
}

#[test]
fn test_element_name_krypton() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 36})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nm = ELEMENT_NAME(znum) SELECT nm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "nm"), "Krypton");
}

#[test]
fn test_atomic_mass_by_symbol_string() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = ATOMIC_MASS("C") SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let m = get_float(&r, "m");
    assert!((m - 12.011).abs() < 0.001, "C mass should be 12.011, got {}", m);
}

#[test]
fn test_element_group_noble_gas_18() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 18})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE grp = ELEMENT_GROUP(znum) SELECT grp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "grp"), 18, "Argon should be in group 18");
}

#[test]
fn test_is_valid_formula_nacl() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v = IS_VALID_FORMULA("NaCl") SELECT v;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(get_bool(&r, "v"), "NaCl should be valid");
}

#[test]
fn test_formula_atom_count_co2() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = FORMULA_ATOM_COUNT("CO2") SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "n"), 3);
}

#[test]
fn test_empirical_formula_simple() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // NaCl - already empirical (GCD=1)
    let mut p = Parser::new(r#"QUERY t COMPUTE ef = EMPIRICAL_FORMULA("NaCl") SELECT ef;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ef = get_string(&r, "ef");
    assert!(!ef.is_empty(), "NaCl empirical formula should not be empty, got '{}'", ef);
}

#[test]
fn test_ph_roundtrip() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // pH=4 -> [H+] = 0.0001 -> pH back to 4
    let mut p = Parser::new(r#"QUERY t COMPUTE ph2 = PH_FROM_CONCENTRATION(H_FROM_PH(4.0)) SELECT ph2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ph2 = get_float(&r, "ph2");
    assert!((ph2 - 4.0).abs() < 0.001, "pH roundtrip should give 4.0, got {}", ph2);
}

#[test]
fn test_molarity_division_by_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = MOLARITY(1.0, 0.0) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(is_null(&r, "c"), "MOLARITY with 0 liters should return Null");
}

#[test]
fn test_boyle_division_by_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = BOYLE_LAW_VOLUME(1.0, 1.0, 0.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(is_null(&r, "v2"), "BOYLE_LAW_VOLUME with P2=0 should return Null");
}

#[test]
fn test_element_period_2_carbon() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 6})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE per = ELEMENT_PERIOD(znum) SELECT per;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "per"), 2, "Carbon should be in period 2");
}

#[test]
fn test_element_category_alkali_metal() {
    let (_dir, db, ex) = setup();
    // Li = 3 = alkali metal
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 3})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cat = ELEMENT_CATEGORY(znum) SELECT cat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "cat"), "alkali metal");
}

#[test]
fn test_element_category_metalloid_boron() {
    let (_dir, db, ex) = setup();
    // B = 5 = metalloid
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"znum": 5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cat = ELEMENT_CATEGORY(znum) SELECT cat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_string(&r, "cat"), "metalloid");
}

#[test]
fn test_molecular_weight_co_molecule() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mw = MOLECULAR_WEIGHT("CO") SELECT mw;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mw = get_float(&r, "mw");
    // CO = 12.011 + 15.999 = 28.01
    assert!((mw - 28.010).abs() < 0.1, "CO mw should be ~28.010, got {}", mw);
}

#[test]
fn test_charles_law_proportional() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // V1/T1 = V2/T2 -> V2 = V1*T2/T1 = 3.0 * 400/300 = 4.0
    let mut p = Parser::new(r#"QUERY t COMPUTE v2 = CHARLES_LAW_VOLUME(3.0, 300.0, 400.0) SELECT v2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v2 = get_float(&r, "v2");
    assert!((v2 - 4.0).abs() < 1e-9, "Charles V2 should be 4.0, got {}", v2);
}
