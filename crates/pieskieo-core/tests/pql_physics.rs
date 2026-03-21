/// Integration tests for PQL built-in physics and engineering functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
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

// ── Temperature conversions ───────────────────────────────────────────────────

#[test]
fn test_celsius_to_fahrenheit() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 100.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE f = CELSIUS_TO_FAHRENHEIT(temp) SELECT f;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(&r, "f");
    assert!((f - 212.0).abs() < 0.01, "100C should be 212F, got {}", f);
}

#[test]
fn test_c_to_f_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 0.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE f = C_TO_F(temp) SELECT f;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(&r, "f");
    assert!((f - 32.0).abs() < 0.01, "0C should be 32F, got {}", f);
}

#[test]
fn test_fahrenheit_to_celsius() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 212.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = FAHRENHEIT_TO_CELSIUS(temp) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = get_float(&r, "c");
    assert!((c - 100.0).abs() < 0.01, "212F should be 100C, got {}", c);
}

#[test]
fn test_f_to_c_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 32.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = F_TO_C(temp) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = get_float(&r, "c");
    assert!((c - 0.0).abs() < 0.01, "32F should be 0C, got {}", c);
}

#[test]
fn test_celsius_to_kelvin() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 0.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE k = CELSIUS_TO_KELVIN(temp) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k = get_float(&r, "k");
    assert!((k - 273.15).abs() < 0.01, "0C should be 273.15K, got {}", k);
}

#[test]
fn test_c_to_k_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 100.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE k = C_TO_K(temp) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k = get_float(&r, "k");
    assert!(
        (k - 373.15).abs() < 0.01,
        "100C should be 373.15K, got {}",
        k
    );
}

#[test]
fn test_kelvin_to_celsius() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 273.15}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = KELVIN_TO_CELSIUS(temp) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = get_float(&r, "c");
    assert!((c - 0.0).abs() < 0.01, "273.15K should be 0C, got {}", c);
}

#[test]
fn test_k_to_c_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 373.15}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = K_TO_C(temp) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let c = get_float(&r, "c");
    assert!(
        (c - 100.0).abs() < 0.01,
        "373.15K should be 100C, got {}",
        c
    );
}

#[test]
fn test_fahrenheit_to_kelvin() {
    let (_dir, db, ex) = setup();
    // 32F = 273.15K
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 32.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE k = FAHRENHEIT_TO_KELVIN(temp) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k = get_float(&r, "k");
    assert!(
        (k - 273.15).abs() < 0.01,
        "32F should be 273.15K, got {}",
        k
    );
}

#[test]
fn test_f_to_k_alias() {
    let (_dir, db, ex) = setup();
    // 212F = 373.15K
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 212.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE k = F_TO_K(temp) SELECT k;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k = get_float(&r, "k");
    assert!(
        (k - 373.15).abs() < 0.01,
        "212F should be 373.15K, got {}",
        k
    );
}

// ── Mass conversions ──────────────────────────────────────────────────────────

#[test]
fn test_kg_to_lbs() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lbs = KG_TO_LBS(mass) SELECT lbs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lbs = get_float(&r, "lbs");
    assert!(
        (lbs - 2.20462).abs() < 0.001,
        "1kg should be ~2.20462 lbs, got {}",
        lbs
    );
}

#[test]
fn test_kilograms_to_pounds_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lbs = KILOGRAMS_TO_POUNDS(mass) SELECT lbs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lbs = get_float(&r, "lbs");
    assert!(
        (lbs - 22.0462).abs() < 0.001,
        "10kg should be ~22.0462 lbs, got {}",
        lbs
    );
}

#[test]
fn test_lbs_to_kg() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 2.20462}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kg = LBS_TO_KG(mass) SELECT kg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let kg = get_float(&r, "kg");
    assert!(
        (kg - 1.0).abs() < 0.001,
        "2.20462 lbs should be ~1 kg, got {}",
        kg
    );
}

#[test]
fn test_pounds_to_kilograms_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 100.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kg = POUNDS_TO_KILOGRAMS(mass) SELECT kg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let kg = get_float(&r, "kg");
    assert!(
        (kg - 45.3592).abs() < 0.01,
        "100 lbs should be ~45.36 kg, got {}",
        kg
    );
}

// ── Distance conversions ──────────────────────────────────────────────────────

#[test]
fn test_km_to_miles() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dist": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mi = KM_TO_MILES(dist) SELECT mi;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mi = get_float(&r, "mi");
    assert!(
        (mi - 0.621371).abs() < 0.0001,
        "1km should be ~0.621371 mi, got {}",
        mi
    );
}

#[test]
fn test_km_to_mi_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dist": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mi = KM_TO_MI(dist) SELECT mi;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mi = get_float(&r, "mi");
    assert!(
        (mi - 6.21371).abs() < 0.001,
        "10km should be ~6.214 mi, got {}",
        mi
    );
}

#[test]
fn test_miles_to_km() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dist": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE km = MILES_TO_KM(dist) SELECT km;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let km = get_float(&r, "km");
    assert!(
        (km - 1.60934).abs() < 0.001,
        "1 mile should be ~1.609 km, got {}",
        km
    );
}

#[test]
fn test_mi_to_km_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dist": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE km = MI_TO_KM(dist) SELECT km;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let km = get_float(&r, "km");
    assert!(
        (km - 8.04672).abs() < 0.01,
        "5 miles should be ~8.047 km, got {}",
        km
    );
}

#[test]
fn test_meters_to_feet() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"len": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ft = METERS_TO_FEET(len) SELECT ft;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ft = get_float(&r, "ft");
    assert!(
        (ft - 3.28084).abs() < 0.0001,
        "1m should be ~3.28084 ft, got {}",
        ft
    );
}

#[test]
fn test_feet_to_meters() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"len": 3.28084}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = FEET_TO_METERS(len) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let m = get_float(&r, "m");
    assert!(
        (m - 1.0).abs() < 0.001,
        "3.28084 ft should be ~1m, got {}",
        m
    );
}

// ── Volume conversions ────────────────────────────────────────────────────────

#[test]
fn test_liters_to_gallons() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vol": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gal = LITERS_TO_GALLONS(vol) SELECT gal;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let gal = get_float(&r, "gal");
    assert!(
        (gal - 0.264172).abs() < 0.0001,
        "1L should be ~0.264172 gal, got {}",
        gal
    );
}

#[test]
fn test_gallons_to_liters() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vol": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lit = GALLONS_TO_LITERS(vol) SELECT lit;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lit = get_float(&r, "lit");
    assert!(
        (lit - 3.78541).abs() < 0.01,
        "1 gal should be ~3.785 L, got {}",
        lit
    );
}

// ── Energy conversions ────────────────────────────────────────────────────────

#[test]
fn test_joules_to_calories() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"energy": 4.184}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cal = JOULES_TO_CALORIES(energy) SELECT cal;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cal = get_float(&r, "cal");
    assert!(
        (cal - 1.0).abs() < 0.001,
        "4.184 J should be ~1 cal, got {}",
        cal
    );
}

#[test]
fn test_calories_to_joules() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"energy": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = CALORIES_TO_JOULES(energy) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let j = get_float(&r, "j");
    assert!(
        (j - 4.184).abs() < 0.001,
        "1 cal should be 4.184 J, got {}",
        j
    );
}

// ── Power conversions ─────────────────────────────────────────────────────────

#[test]
fn test_watts_to_hp() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pwr": 745.7}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hp = WATTS_TO_HP(pwr) SELECT hp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let hp = get_float(&r, "hp");
    assert!(
        (hp - 1.0).abs() < 0.001,
        "745.7 W should be ~1 HP, got {}",
        hp
    );
}

#[test]
fn test_hp_to_watts() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pwr": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = HP_TO_WATTS(pwr) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let w = get_float(&r, "w");
    assert!((w - 745.7).abs() < 0.1, "1 HP should be 745.7 W, got {}", w);
}

// ── Pressure conversions ──────────────────────────────────────────────────────

#[test]
fn test_pa_to_psi() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pres": 6894.76}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE psi = PA_TO_PSI(pres) SELECT psi;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let psi = get_float(&r, "psi");
    assert!(
        (psi - 1.0).abs() < 0.001,
        "6894.76 Pa should be ~1 PSI, got {}",
        psi
    );
}

#[test]
fn test_psi_to_pa() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pres": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pa = PSI_TO_PA(pres) SELECT pa;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pa = get_float(&r, "pa");
    assert!(
        (pa - 6894.76).abs() < 0.1,
        "1 PSI should be 6894.76 Pa, got {}",
        pa
    );
}

// ── Speed conversions ─────────────────────────────────────────────────────────

#[test]
fn test_knots_to_mps() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"spd": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mps = KNOTS_TO_MPS(spd) SELECT mps;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mps = get_float(&r, "mps");
    assert!(
        (mps - 0.514444).abs() < 0.0001,
        "1 knot should be ~0.514444 m/s, got {}",
        mps
    );
}

#[test]
fn test_mps_to_knots() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"spd": 0.514444}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kn = MPS_TO_KNOTS(spd) SELECT kn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let kn = get_float(&r, "kn");
    assert!(
        (kn - 1.0).abs() < 0.001,
        "0.514444 m/s should be ~1 knot, got {}",
        kn
    );
}

// ── Mechanics ─────────────────────────────────────────────────────────────────

#[test]
fn test_kinetic_energy() {
    let (_dir, db, ex) = setup();
    // KE = 0.5 * 2 * 10^2 = 100 J
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 2.0, "vel": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ke = KINETIC_ENERGY(mass, vel) SELECT ke;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ke = get_float(&r, "ke");
    assert!(
        (ke - 100.0).abs() < 0.001,
        "KE(2, 10) should be 100 J, got {}",
        ke
    );
}

#[test]
fn test_ke_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 1.0, "vel": 4.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ke = KE(mass, vel) SELECT ke;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ke = get_float(&r, "ke");
    assert!(
        (ke - 8.0).abs() < 0.001,
        "KE(1, 4) should be 8 J, got {}",
        ke
    );
}

#[test]
fn test_potential_energy() {
    let (_dir, db, ex) = setup();
    // PE = 10 * 9.81 * 5 = 490.5 J
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 10.0, "ht": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pe = POTENTIAL_ENERGY(mass, ht) SELECT pe;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pe = get_float(&r, "pe");
    assert!(
        (pe - 490.5).abs() < 0.1,
        "PE(10, 5) should be ~490.5 J, got {}",
        pe
    );
}

#[test]
fn test_momentum() {
    let (_dir, db, ex) = setup();
    // p = 5 * 20 = 100 kg*m/s
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 5.0, "vel": 20.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mom = MOMENTUM(mass, vel) SELECT mom;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mom = get_float(&r, "mom");
    assert!(
        (mom - 100.0).abs() < 0.001,
        "MOMENTUM(5, 20) should be 100 kg*m/s, got {}",
        mom
    );
}

#[test]
fn test_linear_momentum_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 3.0, "vel": 7.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mom = LINEAR_MOMENTUM(mass, vel) SELECT mom;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mom = get_float(&r, "mom");
    assert!(
        (mom - 21.0).abs() < 0.001,
        "LINEAR_MOMENTUM(3, 7) should be 21, got {}",
        mom
    );
}

#[test]
fn test_force_newton() {
    let (_dir, db, ex) = setup();
    // F = 5 * 9.81 = 49.05 N
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 5.0, "accel": 9.81}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE force = FORCE_NEWTON(mass, accel) SELECT force;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let force = get_float(&r, "force");
    assert!(
        (force - 49.05).abs() < 0.01,
        "FORCE_NEWTON(5, 9.81) should be ~49.05 N, got {}",
        force
    );
}

#[test]
fn test_f_ma_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 10.0, "accel": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE force = F_MA(mass, accel) SELECT force;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let force = get_float(&r, "force");
    assert!(
        (force - 50.0).abs() < 0.001,
        "F_MA(10, 5) should be 50 N, got {}",
        force
    );
}

#[test]
fn test_power_watts() {
    let (_dir, db, ex) = setup();
    // P = 100 * 5 = 500 W
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"force": 100.0, "vel": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pwr = POWER_WATTS(force, vel) SELECT pwr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pwr = get_float(&r, "pwr");
    assert!(
        (pwr - 500.0).abs() < 0.001,
        "POWER_WATTS(100, 5) should be 500 W, got {}",
        pwr
    );
}

#[test]
fn test_work_joules() {
    let (_dir, db, ex) = setup();
    // W = 50 * 10 = 500 J
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"force": 50.0, "dist": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE wrk = WORK_JOULES(force, dist) SELECT wrk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let wrk = get_float(&r, "wrk");
    assert!(
        (wrk - 500.0).abs() < 0.001,
        "WORK_JOULES(50, 10) should be 500 J, got {}",
        wrk
    );
}

#[test]
fn test_gravitational_force() {
    let (_dir, db, ex) = setup();
    // F = G * 1e10 * 1e10 / 1^2 = 6.674e-11 * 1e20 = 6.674e9 N
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m1": 1.0e10, "m2": 1.0e10, "dist": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gf = GRAVITATIONAL_FORCE(m1, m2, dist) SELECT gf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let gf = get_float(&r, "gf");
    let expected = 6.674e-11 * 1.0e10 * 1.0e10 / 1.0_f64;
    assert!(
        (gf - expected).abs() / expected < 0.001,
        "GRAVITATIONAL_FORCE(1e10, 1e10, 1) should be ~{:.3e} N, got {:.3e}",
        expected,
        gf
    );
}

#[test]
fn test_escape_velocity_earth_approx() {
    let (_dir, db, ex) = setup();
    // Earth: M = 5.972e24 kg, R = 6.371e6 m -> ve ≈ 11.186 km/s = 11186 m/s
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 5.972e24, "radius": 6.371e6}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ve = ESCAPE_VELOCITY(mass, radius) SELECT ve;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ve = get_float(&r, "ve");
    assert!(
        ve > 11000.0 && ve < 11300.0,
        "Earth escape velocity should be ~11186 m/s, got {}",
        ve
    );
}

#[test]
fn test_projectile_range_45_deg() {
    let (_dir, db, ex) = setup();
    // Max range at 45 deg: v0^2 / g = 100 / 9.81 ≈ 10.194 m
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v0": 10.0, "ang": 45.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rng = PROJECTILE_RANGE(v0, ang) SELECT rng;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let rng = get_float(&r, "rng");
    assert!(
        (rng - 10.194).abs() < 0.01,
        "PROJ_RANGE(10, 45) should be ~10.194 m, got {}",
        rng
    );
}

#[test]
fn test_projectile_max_height() {
    let (_dir, db, ex) = setup();
    // H = (v0*sin(90))^2 / (2g) = 100 / (2*9.81) ≈ 5.097 m (90 deg -> all vertical)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v0": 10.0, "ang": 90.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ht = PROJECTILE_MAX_HEIGHT(v0, ang) SELECT ht;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ht = get_float(&r, "ht");
    assert!(
        (ht - 5.097).abs() < 0.01,
        "PROJ_HEIGHT(10, 90) should be ~5.097 m, got {}",
        ht
    );
}

// ── Electricity ───────────────────────────────────────────────────────────────

#[test]
fn test_ohm_voltage() {
    let (_dir, db, ex) = setup();
    // V = 2A * 10 ohm = 20V
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"curr": 2.0, "res": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE volt = OHM_VOLTAGE(curr, res) SELECT volt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let volt = get_float(&r, "volt");
    assert!(
        (volt - 20.0).abs() < 0.001,
        "OHM_VOLTAGE(2, 10) should be 20 V, got {}",
        volt
    );
}

#[test]
fn test_v_ir_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"curr": 5.0, "res": 4.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE volt = V_IR(curr, res) SELECT volt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let volt = get_float(&r, "volt");
    assert!(
        (volt - 20.0).abs() < 0.001,
        "V_IR(5, 4) should be 20 V, got {}",
        volt
    );
}

#[test]
fn test_ohm_current() {
    let (_dir, db, ex) = setup();
    // I = 12V / 6 ohm = 2A
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"volt": 12.0, "res": 6.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE curr = OHM_CURRENT(volt, res) SELECT curr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let curr = get_float(&r, "curr");
    assert!(
        (curr - 2.0).abs() < 0.001,
        "OHM_CURRENT(12, 6) should be 2 A, got {}",
        curr
    );
}

#[test]
fn test_ohm_resistance() {
    let (_dir, db, ex) = setup();
    // R = 24V / 3A = 8 ohm
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"volt": 24.0, "curr": 3.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = OHM_RESISTANCE(volt, curr) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let res = get_float(&r, "res");
    assert!(
        (res - 8.0).abs() < 0.001,
        "OHM_RESISTANCE(24, 3) should be 8 ohm, got {}",
        res
    );
}

#[test]
fn test_power_electric() {
    let (_dir, db, ex) = setup();
    // P = 12V * 5A = 60 W
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"volt": 12.0, "curr": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pwr = POWER_ELECTRIC(volt, curr) SELECT pwr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pwr = get_float(&r, "pwr");
    assert!(
        (pwr - 60.0).abs() < 0.001,
        "POWER_ELECTRIC(12, 5) should be 60 W, got {}",
        pwr
    );
}

#[test]
fn test_electric_power_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"volt": 120.0, "curr": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pwr = ELECTRIC_POWER(volt, curr) SELECT pwr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pwr = get_float(&r, "pwr");
    assert!(
        (pwr - 1200.0).abs() < 0.001,
        "ELECTRIC_POWER(120, 10) should be 1200 W, got {}",
        pwr
    );
}

#[test]
fn test_capacitor_energy() {
    let (_dir, db, ex) = setup();
    // E = 0.5 * 0.01 * 100^2 = 50 J
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"cap": 0.01, "volt": 100.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE en = CAPACITOR_ENERGY(cap, volt) SELECT en;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let en = get_float(&r, "en");
    assert!(
        (en - 50.0).abs() < 0.001,
        "CAP_ENERGY(0.01, 100) should be 50 J, got {}",
        en
    );
}

#[test]
fn test_inductor_energy() {
    let (_dir, db, ex) = setup();
    // E = 0.5 * 2 * 4^2 = 16 J
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ind": 2.0, "curr": 4.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE en = INDUCTOR_ENERGY(ind, curr) SELECT en;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let en = get_float(&r, "en");
    assert!(
        (en - 16.0).abs() < 0.001,
        "IND_ENERGY(2, 4) should be 16 J, got {}",
        en
    );
}

#[test]
fn test_rc_time_constant() {
    let (_dir, db, ex) = setup();
    // tau = 1000 * 0.001 = 1.0 s
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"res": 1000.0, "cap": 0.001}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tau = RC_TIME_CONSTANT(res, cap) SELECT tau;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let tau = get_float(&r, "tau");
    assert!(
        (tau - 1.0).abs() < 0.001,
        "RC_TAU(1000, 0.001) should be 1.0 s, got {}",
        tau
    );
}

#[test]
fn test_decibels_to_ratio() {
    let (_dir, db, ex) = setup();
    // 20 dB -> 10^(20/20) = 10
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"db": 20.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ratio = DECIBELS_TO_RATIO(db) SELECT ratio;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ratio = get_float(&r, "ratio");
    assert!(
        (ratio - 10.0).abs() < 0.001,
        "DB_TO_RATIO(20) should be 10, got {}",
        ratio
    );
}

#[test]
fn test_ratio_to_decibels() {
    let (_dir, db, ex) = setup();
    // 10 -> 20 * log10(10) = 20 dB
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ratio": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE db = RATIO_TO_DECIBELS(ratio) SELECT db;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let db = get_float(&r, "db");
    assert!(
        (db - 20.0).abs() < 0.001,
        "RATIO_TO_DB(10) should be 20 dB, got {}",
        db
    );
}

// ── Optics and waves ──────────────────────────────────────────────────────────

#[test]
fn test_wavelength_to_freq() {
    let (_dir, db, ex) = setup();
    // lambda = 1m -> f = 3e8 Hz
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"wl": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE freq = WAVELENGTH_TO_FREQ(wl) SELECT freq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let freq = get_float(&r, "freq");
    assert!(
        (freq - 3.0e8).abs() < 1.0,
        "WAVE_FREQ(1) should be 3e8 Hz, got {}",
        freq
    );
}

#[test]
fn test_freq_to_wavelength() {
    let (_dir, db, ex) = setup();
    // f = 3e8 -> lambda = 1m
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 3.0e8}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE wl = FREQ_TO_WAVELENGTH(freq) SELECT wl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let wl = get_float(&r, "wl");
    assert!(
        (wl - 1.0).abs() < 0.001,
        "FREQ_WAVE(3e8) should be 1 m, got {}",
        wl
    );
}

#[test]
fn test_snell_refraction() {
    let (_dir, db, ex) = setup();
    // air (n1=1) to glass (n2=1.5), angle_in=30 deg
    // sin(theta2) = 1*sin(30)/1.5 = 0.5/1.5 = 0.333 -> theta2 = 19.47 deg
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ang": 30.0, "n1": 1.0, "n2": 1.5}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ang2 = SNELL_REFRACTION(ang, n1, n2) SELECT ang2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ang2 = get_float(&r, "ang2");
    assert!(
        (ang2 - 19.47).abs() < 0.1,
        "SNELL(30, 1, 1.5) should be ~19.47 deg, got {}",
        ang2
    );
}

#[test]
fn test_doppler_freq_approaching() {
    let (_dir, db, ex) = setup();
    // source moving toward stationary observer at 34.3 m/s (10% of sound speed)
    // f_obs = 1000 * (343 + 0) / (343 - 34.3) = 1000 * 343/308.7 ≈ 1111.1 Hz
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 1000.0, "vsrc": 34.3, "vobs": 0.0}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE fobs = DOPPLER_FREQ(freq, vsrc, vobs) SELECT fobs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let fobs = get_float(&r, "fobs");
    assert!(
        fobs > 1100.0 && fobs < 1120.0,
        "DOPPLER approaching should be ~1111 Hz, got {}",
        fobs
    );
}

#[test]
fn test_photon_energy() {
    let (_dir, db, ex) = setup();
    // E = h * f = 6.626e-34 * 6e14 (visible light ~600nm) ≈ 3.976e-19 J
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 6.0e14}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE en = PHOTON_ENERGY(freq) SELECT en;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let en = get_float(&r, "en");
    let expected = 6.626e-34 * 6.0e14;
    assert!(
        (en - expected).abs() < 1e-21,
        "PHOTON_E(6e14) should be ~{:.3e} J, got {:.3e}",
        expected,
        en
    );
}

#[test]
fn test_refractive_index() {
    let (_dir, db, ex) = setup();
    // n = 3e8 / 2e8 = 1.5
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"cmed": 2.0e8}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE idx = REFRACTIVE_INDEX(cmed) SELECT idx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let idx = get_float(&r, "idx");
    assert!(
        (idx - 1.5).abs() < 0.001,
        "N_INDEX(2e8) should be 1.5, got {}",
        idx
    );
}

// ── Thermodynamics ────────────────────────────────────────────────────────────

#[test]
fn test_heat_transfer() {
    let (_dir, db, ex) = setup();
    // Q = 2 * 4184 * 10 = 83680 J (water, 10C rise)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 2.0, "sheat": 4184.0, "dt": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE q = HEAT_TRANSFER(mass, sheat, dt) SELECT q;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let q = get_float(&r, "q");
    assert!(
        (q - 83680.0).abs() < 1.0,
        "HEAT_Q(2, 4184, 10) should be 83680 J, got {}",
        q
    );
}

#[test]
fn test_heat_q_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 1.0, "sheat": 1000.0, "dt": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE q = HEAT_Q(mass, sheat, dt) SELECT q;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let q = get_float(&r, "q");
    assert!(
        (q - 5000.0).abs() < 0.001,
        "HEAT_Q(1, 1000, 5) should be 5000 J, got {}",
        q
    );
}

#[test]
fn test_thermal_expansion() {
    let (_dir, db, ex) = setup();
    // dL = 10 * 12e-6 * 100 = 0.012 m (steel rod)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"len": 10.0, "alpha": 12.0e-6, "dt": 100.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dl = THERMAL_EXPANSION(len, alpha, dt) SELECT dl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let dl = get_float(&r, "dl");
    assert!(
        (dl - 0.012).abs() < 0.00001,
        "THERMAL_EXP(10, 12e-6, 100) should be 0.012 m, got {}",
        dl
    );
}

#[test]
fn test_ideal_gas_pressure() {
    let (_dir, db, ex) = setup();
    // P = 1 * 8.314 * 273.15 / 0.02271 ≈ 101325 Pa (standard atmosphere approximation)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1.0, "temp": 273.15, "vol": 0.022414}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE pres = IDEAL_GAS_PRESSURE(n, temp, vol) SELECT pres;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pres = get_float(&r, "pres");
    // At STP, 1 mol in 22.414 L -> ~101.3 kPa
    assert!(
        pres > 100000.0 && pres < 102000.0,
        "GAS_P(1, 273.15, 0.022414) should be ~101325 Pa, got {}",
        pres
    );
}

#[test]
fn test_gas_p_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 2.0, "temp": 300.0, "vol": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pres = GAS_P(n, temp, vol) SELECT pres;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pres = get_float(&r, "pres");
    // P = 2 * 8.314 * 300 / 1 = 4988.4
    assert!(
        (pres - 4988.4).abs() < 1.0,
        "GAS_P(2, 300, 1) should be ~4988.4 Pa, got {}",
        pres
    );
}

#[test]
fn test_ideal_gas_volume() {
    let (_dir, db, ex) = setup();
    // V = 1 * 8.314 * 273.15 / 101325 ≈ 0.022414 m^3
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1.0, "temp": 273.15, "pres": 101325.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE vol = IDEAL_GAS_VOLUME(n, temp, pres) SELECT vol;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let vol = get_float(&r, "vol");
    assert!(
        (vol - 0.022414).abs() < 0.0001,
        "GAS_V(1, 273.15, 101325) should be ~0.022414 m^3, got {}",
        vol
    );
}

#[test]
fn test_carnot_efficiency() {
    let (_dir, db, ex) = setup();
    // eta = 1 - 300/600 = 0.5 (50%)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"thot": 600.0, "tcold": 300.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE eta = CARNOT_EFFICIENCY(thot, tcold) SELECT eta;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let eta = get_float(&r, "eta");
    assert!(
        (eta - 0.5).abs() < 0.001,
        "CARNOT(600, 300) should be 0.5, got {}",
        eta
    );
}

#[test]
fn test_carnot_alias() {
    let (_dir, db, ex) = setup();
    // eta = 1 - 200/1000 = 0.8 (80%)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"thot": 1000.0, "tcold": 200.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE eta = CARNOT(thot, tcold) SELECT eta;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let eta = get_float(&r, "eta");
    assert!(
        (eta - 0.8).abs() < 0.001,
        "CARNOT(1000, 200) should be 0.8, got {}",
        eta
    );
}

// ── Alias coverage round-trip checks ─────────────────────────────────────────

#[test]
fn test_m_to_ft_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"len": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ft = M_TO_FT(len) SELECT ft;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ft = get_float(&r, "ft");
    assert!(
        (ft - 3.28084).abs() < 0.0001,
        "M_TO_FT(1) should be 3.28084, got {}",
        ft
    );
}

#[test]
fn test_ft_to_m_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"len": 3.28084}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = FT_TO_M(len) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let m = get_float(&r, "m");
    assert!(
        (m - 1.0).abs() < 0.001,
        "FT_TO_M(3.28084) should be ~1, got {}",
        m
    );
}

#[test]
fn test_l_to_gal_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vol": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gal = L_TO_GAL(vol) SELECT gal;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let gal = get_float(&r, "gal");
    assert!(
        (gal - 2.64172).abs() < 0.001,
        "L_TO_GAL(10) should be ~2.64172, got {}",
        gal
    );
}

#[test]
fn test_gal_to_l_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"vol": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lit = GAL_TO_L(vol) SELECT lit;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lit = get_float(&r, "lit");
    assert!(
        (lit - 3.78541).abs() < 0.01,
        "GAL_TO_L(1) should be ~3.785 L, got {}",
        lit
    );
}

#[test]
fn test_j_to_cal_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"energy": 4.184}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cal = J_TO_CAL(energy) SELECT cal;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cal = get_float(&r, "cal");
    assert!(
        (cal - 1.0).abs() < 0.001,
        "J_TO_CAL(4.184) should be 1, got {}",
        cal
    );
}

#[test]
fn test_cal_to_j_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"energy": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE j = CAL_TO_J(energy) SELECT j;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let j = get_float(&r, "j");
    assert!(
        (j - 4.184).abs() < 0.001,
        "CAL_TO_J(1) should be 4.184, got {}",
        j
    );
}

#[test]
fn test_w_to_hp_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pwr": 7457.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hp = W_TO_HP(pwr) SELECT hp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let hp = get_float(&r, "hp");
    assert!(
        (hp - 10.0).abs() < 0.01,
        "W_TO_HP(7457) should be ~10 HP, got {}",
        hp
    );
}

#[test]
fn test_hp_to_w_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pwr": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE w = HP_TO_W(pwr) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let w = get_float(&r, "w");
    assert!(
        (w - 7457.0).abs() < 0.5,
        "HP_TO_W(10) should be 7457 W, got {}",
        w
    );
}

#[test]
fn test_pascals_to_psi_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pres": 6894.76}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE psi = PASCALS_TO_PSI(pres) SELECT psi;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let psi = get_float(&r, "psi");
    assert!(
        (psi - 1.0).abs() < 0.001,
        "PASCALS_TO_PSI(6894.76) should be 1, got {}",
        psi
    );
}

#[test]
fn test_psi_to_pascals_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"pres": 14.696}),
    )
    .unwrap();
    // 14.696 PSI = 1 atm = ~101325 Pa
    let mut p = Parser::new(r#"QUERY t COMPUTE pa = PSI_TO_PASCALS(pres) SELECT pa;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pa = get_float(&r, "pa");
    assert!(
        pa > 101000.0 && pa < 102000.0,
        "PSI_TO_PASCALS(14.696) should be ~101325 Pa, got {}",
        pa
    );
}

#[test]
fn test_knots_to_ms_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"spd": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE mps = KNOTS_TO_MS(spd) SELECT mps;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mps = get_float(&r, "mps");
    assert!(
        (mps - 5.14444).abs() < 0.001,
        "KNOTS_TO_MS(10) should be ~5.14444 m/s, got {}",
        mps
    );
}

#[test]
fn test_ms_to_knots_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"spd": 5.14444}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kn = MS_TO_KNOTS(spd) SELECT kn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let kn = get_float(&r, "kn");
    assert!(
        (kn - 10.0).abs() < 0.01,
        "MS_TO_KNOTS(5.14444) should be ~10 knots, got {}",
        kn
    );
}

#[test]
fn test_power_w_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"force": 200.0, "vel": 3.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pwr = POWER_W(force, vel) SELECT pwr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pwr = get_float(&r, "pwr");
    assert!(
        (pwr - 600.0).abs() < 0.001,
        "POWER_W(200, 3) should be 600 W, got {}",
        pwr
    );
}

#[test]
fn test_work_j_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"force": 25.0, "dist": 8.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE wrk = WORK_J(force, dist) SELECT wrk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let wrk = get_float(&r, "wrk");
    assert!(
        (wrk - 200.0).abs() < 0.001,
        "WORK_J(25, 8) should be 200 J, got {}",
        wrk
    );
}

#[test]
fn test_grav_force_alias() {
    let (_dir, db, ex) = setup();
    // F = G * 1e10 * 1e10 / 1^2 = 6.674e-11 * 1e20 = 6.674e9 N
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"m1": 1.0e10, "m2": 1.0e10, "dist": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gf = GRAV_FORCE(m1, m2, dist) SELECT gf;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let gf = get_float(&r, "gf");
    let expected = 6.674e-11 * 1.0e10 * 1.0e10 / 1.0_f64;
    assert!(
        (gf - expected).abs() / expected < 0.001,
        "GRAV_FORCE should be ~{:.3e} N, got {:.3e}",
        expected,
        gf
    );
}

#[test]
fn test_ve_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 5.972e24, "radius": 6.371e6}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ve = VE(mass, radius) SELECT ve;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ve = get_float(&r, "ve");
    assert!(
        ve > 11000.0 && ve < 11300.0,
        "VE alias for escape velocity should be ~11186 m/s, got {}",
        ve
    );
}

#[test]
fn test_proj_range_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v0": 10.0, "ang": 45.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rng = PROJ_RANGE(v0, ang) SELECT rng;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let rng = get_float(&r, "rng");
    assert!(
        (rng - 10.194).abs() < 0.01,
        "PROJ_RANGE(10, 45) should be ~10.194 m, got {}",
        rng
    );
}

#[test]
fn test_proj_height_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"v0": 10.0, "ang": 90.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ht = PROJ_HEIGHT(v0, ang) SELECT ht;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ht = get_float(&r, "ht");
    assert!(
        (ht - 5.097).abs() < 0.01,
        "PROJ_HEIGHT(10, 90) should be ~5.097 m, got {}",
        ht
    );
}

#[test]
fn test_i_vr_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"volt": 120.0, "res": 60.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE curr = I_VR(volt, res) SELECT curr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let curr = get_float(&r, "curr");
    assert!(
        (curr - 2.0).abs() < 0.001,
        "I_VR(120, 60) should be 2 A, got {}",
        curr
    );
}

#[test]
fn test_r_vi_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"volt": 9.0, "curr": 3.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = R_VI(volt, curr) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let res = get_float(&r, "res");
    assert!(
        (res - 3.0).abs() < 0.001,
        "R_VI(9, 3) should be 3 ohm, got {}",
        res
    );
}

#[test]
fn test_cap_energy_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"cap": 0.002, "volt": 10.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE en = CAP_ENERGY(cap, volt) SELECT en;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let en = get_float(&r, "en");
    // E = 0.5 * 0.002 * 100 = 0.1 J
    assert!(
        (en - 0.1).abs() < 0.001,
        "CAP_ENERGY(0.002, 10) should be 0.1 J, got {}",
        en
    );
}

#[test]
fn test_ind_energy_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ind": 0.5, "curr": 2.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE en = IND_ENERGY(ind, curr) SELECT en;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let en = get_float(&r, "en");
    // E = 0.5 * 0.5 * 4 = 1 J
    assert!(
        (en - 1.0).abs() < 0.001,
        "IND_ENERGY(0.5, 2) should be 1 J, got {}",
        en
    );
}

#[test]
fn test_rc_tau_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"res": 470.0, "cap": 0.0001}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tau = RC_TAU(res, cap) SELECT tau;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let tau = get_float(&r, "tau");
    assert!(
        (tau - 0.047).abs() < 0.001,
        "RC_TAU(470, 0.0001) should be 0.047 s, got {}",
        tau
    );
}

#[test]
fn test_db_to_ratio_alias() {
    let (_dir, db, ex) = setup();
    // 0 dB -> ratio 1.0
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"db": 0.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ratio = DB_TO_RATIO(db) SELECT ratio;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ratio = get_float(&r, "ratio");
    assert!(
        (ratio - 1.0).abs() < 0.001,
        "DB_TO_RATIO(0) should be 1, got {}",
        ratio
    );
}

#[test]
fn test_ratio_to_db_alias() {
    let (_dir, db, ex) = setup();
    // ratio 1 -> 0 dB
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ratio": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE db = RATIO_TO_DB(ratio) SELECT db;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let db = get_float(&r, "db");
    assert!(
        (db - 0.0).abs() < 0.001,
        "RATIO_TO_DB(1) should be 0 dB, got {}",
        db
    );
}

#[test]
fn test_wave_freq_alias() {
    let (_dir, db, ex) = setup();
    // lambda = 0.5m -> 6e8 Hz
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"wl": 0.5}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE freq = WAVE_FREQ(wl) SELECT freq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let freq = get_float(&r, "freq");
    assert!(
        (freq - 6.0e8).abs() < 1.0,
        "WAVE_FREQ(0.5) should be 6e8 Hz, got {}",
        freq
    );
}

#[test]
fn test_freq_wave_alias() {
    let (_dir, db, ex) = setup();
    // f = 1.5e8 -> lambda = 2 m
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 1.5e8}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE wl = FREQ_WAVE(freq) SELECT wl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let wl = get_float(&r, "wl");
    assert!(
        (wl - 2.0).abs() < 0.001,
        "FREQ_WAVE(1.5e8) should be 2 m, got {}",
        wl
    );
}

#[test]
fn test_snell_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"ang": 30.0, "n1": 1.0, "n2": 1.5}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ang2 = SNELL(ang, n1, n2) SELECT ang2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ang2 = get_float(&r, "ang2");
    assert!(
        (ang2 - 19.47).abs() < 0.1,
        "SNELL(30, 1, 1.5) should be ~19.47 deg, got {}",
        ang2
    );
}

#[test]
fn test_doppler_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 1000.0, "vsrc": 34.3, "vobs": 0.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE fobs = DOPPLER(freq, vsrc, vobs) SELECT fobs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let fobs = get_float(&r, "fobs");
    assert!(
        fobs > 1100.0 && fobs < 1120.0,
        "DOPPLER alias should be ~1111 Hz, got {}",
        fobs
    );
}

#[test]
fn test_photon_e_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 6.0e14}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE en = PHOTON_E(freq) SELECT en;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let en = get_float(&r, "en");
    let expected = 6.626e-34 * 6.0e14;
    assert!(
        (en - expected).abs() < 1e-21,
        "PHOTON_E alias should be ~{:.3e}, got {:.3e}",
        expected,
        en
    );
}

#[test]
fn test_n_index_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"cmed": 2.0e8}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE idx = N_INDEX(cmed) SELECT idx;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let idx = get_float(&r, "idx");
    assert!(
        (idx - 1.5).abs() < 0.001,
        "N_INDEX(2e8) should be 1.5, got {}",
        idx
    );
}

#[test]
fn test_thermal_exp_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"len": 10.0, "alpha": 12.0e-6, "dt": 100.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dl = THERMAL_EXP(len, alpha, dt) SELECT dl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let dl = get_float(&r, "dl");
    assert!(
        (dl - 0.012).abs() < 0.00001,
        "THERMAL_EXP alias should be 0.012 m, got {}",
        dl
    );
}

#[test]
fn test_gas_v_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1.0, "temp": 273.15, "pres": 101325.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE vol = GAS_V(n, temp, pres) SELECT vol;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let vol = get_float(&r, "vol");
    assert!(
        (vol - 0.022414).abs() < 0.0001,
        "GAS_V alias should be ~0.022414 m^3, got {}",
        vol
    );
}

#[test]
fn test_pe_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 10.0, "ht": 5.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pe = PE(mass, ht) SELECT pe;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pe = get_float(&r, "pe");
    assert!(
        (pe - 490.5).abs() < 0.1,
        "PE alias should be ~490.5 J, got {}",
        pe
    );
}

#[test]
fn test_linear_momentum_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"mass": 0.145, "vel": 40.0}),
    )
    .unwrap();
    // Baseball: 145g at 40 m/s -> 5.8 kg*m/s
    let mut p = Parser::new(r#"QUERY t COMPUTE mom = MOMENTUM(mass, vel) SELECT mom;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mom = get_float(&r, "mom");
    assert!(
        (mom - 5.8).abs() < 0.01,
        "MOMENTUM(0.145, 40) should be 5.8 kg*m/s, got {}",
        mom
    );
}

#[test]
fn test_temperature_round_trip_c_f() {
    let (_dir, db, ex) = setup();
    // Convert 37C -> F -> back to C
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": 37.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE f = CELSIUS_TO_FAHRENHEIT(temp) SELECT f;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let f = get_float(&r, "f");
    assert!((f - 98.6).abs() < 0.01, "37C should be 98.6F, got {}", f);
    // now back
    let (_dir2, db2, ex2) = setup();
    db2.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"temp": f}),
    )
    .unwrap();
    let mut p2 = Parser::new(r#"QUERY t COMPUTE c = FAHRENHEIT_TO_CELSIUS(temp) SELECT c;"#);
    let r2 = ex2.execute(p2.parse().unwrap()).unwrap();
    let c = get_float(&r2, "c");
    assert!(
        (c - 37.0).abs() < 0.01,
        "98.6F should be back to 37C, got {}",
        c
    );
}

#[test]
fn test_doppler_stationary_source_moving_observer() {
    let (_dir, db, ex) = setup();
    // Observer moving toward source at 34.3 m/s, source stationary
    // f_obs = 1000 * (343 + 34.3) / (343 - 0) = 1000 * 377.3 / 343 ≈ 1100 Hz
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"freq": 1000.0, "vsrc": 0.0, "vobs": 34.3}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE fobs = DOPPLER_FREQ(freq, vsrc, vobs) SELECT fobs;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let fobs = get_float(&r, "fobs");
    assert!(
        fobs > 1090.0 && fobs < 1110.0,
        "Doppler with moving observer should be ~1100 Hz, got {}",
        fobs
    );
}

#[test]
fn test_carnot_efficiency_perfect_hot() {
    let (_dir, db, ex) = setup();
    // Infinite hot reservoir approximation: T_cold very small fraction of T_hot
    // eta = 1 - 1/1000 = 0.999
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"thot": 1000.0, "tcold": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE eta = CARNOT_EFFICIENCY(thot, tcold) SELECT eta;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let eta = get_float(&r, "eta");
    assert!(
        (eta - 0.999).abs() < 0.001,
        "CARNOT(1000, 1) should be 0.999, got {}",
        eta
    );
}

#[test]
fn test_ideal_gas_pressure_high_temp() {
    let (_dir, db, ex) = setup();
    // P = 1 * 8.314 * 1000 / 1 = 8314 Pa
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"n": 1.0, "temp": 1000.0, "vol": 1.0}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pres = GAS_P(n, temp, vol) SELECT pres;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let pres = get_float(&r, "pres");
    assert!(
        (pres - 8314.0).abs() < 1.0,
        "GAS_P(1, 1000, 1) should be 8314 Pa, got {}",
        pres
    );
}
