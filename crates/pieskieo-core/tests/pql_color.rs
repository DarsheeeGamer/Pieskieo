/// Integration tests for the new PQL built-in color and visualization functions.
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

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

// ── RGB_TO_HSV (individual args via RGB_TO_HSB alias) ────────────────────────

#[test]
fn test_rgb_to_hsv() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Red: RGB(255, 0, 0) -> HSV(0, 1, 1)
    let mut p = Parser::new(r#"QUERY t COMPUTE hsv = RGB_TO_HSB(255, 0, 0) SELECT hsv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsv") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("h") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 1.0,
                    "h should be ~0, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("s") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "s should be 1.0, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("v") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "v should be 1.0, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_rgb_to_hsv_black() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Black: RGB(0, 0, 0) -> HSV(0, 0, 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE hsv = RGB_TO_HSB(0, 0, 0) SELECT hsv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsv") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("s") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 0.01,
                    "s should be 0, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("v") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 0.01,
                    "v should be 0, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── HSV_TO_RGB ────────────────────────────────────────────────────────────────

#[test]
fn test_hsv_to_rgb_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // HSV(0, 1, 1) -> RGB(255, 0, 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = HSV_TO_RGB(0.0, 1.0, 1.0) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(255)), "r should be 255");
            assert_eq!(m.get("g"), Some(&Value::Integer(0)), "g should be 0");
            assert_eq!(m.get("b"), Some(&Value::Integer(0)), "b should be 0");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_hsb_to_rgb_white() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // HSV(0, 0, 1) -> RGB(255, 255, 255)
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = HSB_TO_RGB(0.0, 0.0, 1.0) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(255)), "r should be 255");
            assert_eq!(m.get("g"), Some(&Value::Integer(255)), "g should be 255");
            assert_eq!(m.get("b"), Some(&Value::Integer(255)), "b should be 255");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── HSV_TO_HEX / HEX_TO_HSV ──────────────────────────────────────────────────

#[test]
fn test_hsv_to_hex() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // HSV(0, 1, 1) -> #FF0000
    let mut p = Parser::new(r#"QUERY t COMPUTE hex = HSV_TO_HEX(0.0, 1.0, 1.0) SELECT hex;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("hex"),
        Some(&Value::String("#FF0000".to_string()))
    );
}

#[test]
fn test_hex_to_hsv_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"col": "#FF0000"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsv = HEX_TO_HSV(col) SELECT hsv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsv") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("s") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "s should be 1.0, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("v") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "v should be 1.0, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_hex_to_hsb_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"col": "#0000FF"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsv = HEX_TO_HSB(col) SELECT hsv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsv") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("h") {
                assert!(
                    (to_f64(v) - 240.0).abs() < 2.0,
                    "h should be ~240, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── RGB_TO_CMYK ───────────────────────────────────────────────────────────────

#[test]
fn test_rgb_to_cmyk_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Red RGB(255,0,0) -> CMYK(0, 1, 1, 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE cmyk = RGB_TO_CMYK(255, 0, 0) SELECT cmyk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cmyk") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("c") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 0.01,
                    "c should be 0, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("m") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "m should be 1, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("y") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "y should be 1, got {}",
                    to_f64(v)
                );
            }
            if let Some(v) = m.get("k") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 0.01,
                    "k should be 0, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_rgb_to_cmyk_black() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Black RGB(0,0,0) -> CMYK(0, 0, 0, 1)
    let mut p = Parser::new(r#"QUERY t COMPUTE cmyk = RGB_TO_CMYK(0, 0, 0) SELECT cmyk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cmyk") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("k") {
                assert!(
                    (to_f64(v) - 1.0).abs() < 0.01,
                    "k should be 1, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── CMYK_TO_RGB ───────────────────────────────────────────────────────────────

#[test]
fn test_cmyk_to_rgb_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // CMYK(0, 1, 1, 0) -> RGB(255, 0, 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = CMYK_TO_RGB(0.0, 1.0, 1.0, 0.0) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(255)));
            assert_eq!(m.get("g"), Some(&Value::Integer(0)));
            assert_eq!(m.get("b"), Some(&Value::Integer(0)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── HEX_TO_CMYK / CMYK_TO_HEX ────────────────────────────────────────────────

#[test]
fn test_hex_to_cmyk_white() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"col": "#FFFFFF"}),
    )
    .unwrap();
    // White -> CMYK(0, 0, 0, 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE cmyk = HEX_TO_CMYK(col) SELECT cmyk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cmyk") {
        Some(Value::Object(m)) => {
            for key in ["c", "m", "y", "k"] {
                if let Some(v) = m.get(key) {
                    assert!(
                        (to_f64(v) - 0.0).abs() < 0.01,
                        "{} should be 0 for white, got {}",
                        key,
                        to_f64(v)
                    );
                }
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_cmyk_to_hex_black() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // CMYK(0, 0, 0, 1) -> #000000
    let mut p = Parser::new(r#"QUERY t COMPUTE hex = CMYK_TO_HEX(0.0, 0.0, 0.0, 1.0) SELECT hex;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("hex"),
        Some(&Value::String("#000000".to_string()))
    );
}

// ── RGB_TO_XYZ ────────────────────────────────────────────────────────────────

#[test]
fn test_rgb_to_xyz_white() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // White: RGB(255,255,255) -> XYZ roughly (95, 100, 108)
    let mut p = Parser::new(r#"QUERY t COMPUTE xyz = RGB_TO_XYZ(255, 255, 255) SELECT xyz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("xyz") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("y") {
                assert!(
                    (to_f64(v) - 100.0).abs() < 5.0,
                    "Y should be ~100 for white, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_rgb_to_ciexyz_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE xyz = RGB_TO_CIEXYZ(0, 0, 0) SELECT xyz;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("xyz") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("y") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 0.01,
                    "Y should be 0 for black, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── XYZ_TO_LAB ────────────────────────────────────────────────────────────────

#[test]
fn test_xyz_to_lab_white() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // D65 white point: XYZ(95.047, 100, 108.883) -> LAB(100, 0, 0)
    let mut p =
        Parser::new(r#"QUERY t COMPUTE lab = XYZ_TO_LAB(95.047, 100.0, 108.883) SELECT lab;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("lab") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("L") {
                assert!(
                    (to_f64(v) - 100.0).abs() < 1.0,
                    "L should be ~100, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── RGB_TO_LAB ────────────────────────────────────────────────────────────────

#[test]
fn test_rgb_to_lab_black() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Black: RGB(0,0,0) -> LAB(0, 0, 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE lab = RGB_TO_LAB(0, 0, 0) SELECT lab;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("lab") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("L") {
                assert!(
                    (to_f64(v) - 0.0).abs() < 1.0,
                    "L should be ~0 for black, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_rgb_to_cielab_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lab = RGB_TO_CIELAB(255, 255, 255) SELECT lab;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("lab") {
        Some(Value::Object(m)) => {
            if let Some(v) = m.get("L") {
                assert!(
                    to_f64(v) > 90.0,
                    "L should be >90 for white, got {}",
                    to_f64(v)
                );
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── COLOR_DELTA_E ─────────────────────────────────────────────────────────────

#[test]
fn test_color_delta_e_same_color() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Same color -> delta_E = 0
    let mut p =
        Parser::new(r#"QUERY t COMPUTE de = COLOR_DELTA_E(255, 0, 0, 255, 0, 0) SELECT de;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("de") {
        assert!(
            to_f64(v).abs() < 0.001,
            "delta_E same color should be 0, got {}",
            to_f64(v)
        );
    }
}

#[test]
fn test_color_delta_e_different_colors() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Black vs white -> large delta_E
    let mut p = Parser::new(r#"QUERY t COMPUTE de = DELTA_E(0, 0, 0, 255, 255, 255) SELECT de;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("de") {
        assert!(
            to_f64(v) > 50.0,
            "delta_E black/white should be large, got {}",
            to_f64(v)
        );
    }
}

// ── SPLIT_COMPLEMENTARY ───────────────────────────────────────────────────────

#[test]
fn test_split_complementary_returns_3() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE cols = SPLIT_COMPLEMENTARY(255, 0, 0) SELECT cols;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cols") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "split complementary should return 3 colors");
            // First element should be the original
            assert_eq!(arr[0], Value::String("#FF0000".to_string()));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_split_comp_colors_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cols = SPLIT_COMP_COLORS(0, 255, 0) SELECT cols;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cols") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SQUARE_COLORS ─────────────────────────────────────────────────────────────

#[test]
fn test_square_colors_returns_4() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cols = SQUARE_COLORS(255, 0, 0) SELECT cols;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cols") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4, "square colors should return 4 colors");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_tetradic_colors_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE cols = TETRADIC_COLORS(100, 150, 200) SELECT cols;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cols") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── MONOCHROMATIC_PALETTE ─────────────────────────────────────────────────────

#[test]
fn test_monochromatic_palette_n5() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE pal = MONOCHROMATIC_PALETTE(200, 100, 50, 5) SELECT pal;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pal") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5, "monochromatic palette should return 5 colors");
            for item in arr {
                match item {
                    Value::String(s) => {
                        assert!(s.starts_with('#'), "each color should start with #");
                        assert_eq!(s.len(), 7, "each color should be 7 chars");
                    }
                    _ => panic!("expected String in palette"),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_mono_palette_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pal = MONO_PALETTE(100, 200, 50, 3) SELECT pal;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pal") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── GRADIENT_COLORS ───────────────────────────────────────────────────────────

#[test]
fn test_gradient_colors_endpoints() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r##"QUERY t COMPUTE grad = GRADIENT_COLORS("#FF0000", "#0000FF", 5) SELECT grad;"##,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("grad") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0], Value::String("#FF0000".to_string()));
            assert_eq!(arr[4], Value::String("#0000FF".to_string()));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_color_gradient_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(
        r##"QUERY t COMPUTE grad = COLOR_GRADIENT("#000000", "#FFFFFF", 3) SELECT grad;"##,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("grad") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            // Middle should be gray
            if let Value::String(s) = &arr[1] {
                assert_eq!(
                    s, "#808080",
                    "middle of black-white gradient should be gray"
                );
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── RANDOM_COLOR ──────────────────────────────────────────────────────────────

#[test]
fn test_random_color_seeded_deterministic() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE col = RANDOM_COLOR(42) SELECT col;"#);
    let r1 = ex.execute(p.parse().unwrap()).unwrap();
    let mut p2 = Parser::new(r#"QUERY t COMPUTE col = RANDOM_COLOR(42) SELECT col;"#);
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(
        r1.rows[0].data.get("col"),
        r2.rows[0].data.get("col"),
        "same seed should give same color"
    );
    // Verify it's a hex string
    if let Some(Value::String(s)) = r1.rows[0].data.get("col") {
        assert!(s.starts_with('#'));
        assert_eq!(s.len(), 7);
    }
}

#[test]
fn test_rand_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE col = RAND_COLOR(99) SELECT col;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::String(s)) = r.rows[0].data.get("col") {
        assert!(s.starts_with('#'));
        assert_eq!(s.len(), 7);
    } else {
        panic!("expected String");
    }
}

// ── RANDOM_PASTEL ─────────────────────────────────────────────────────────────

#[test]
fn test_random_pastel_seeded() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE col = RANDOM_PASTEL(10) SELECT col;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::String(s)) = r.rows[0].data.get("col") {
        assert!(s.starts_with('#'));
        assert_eq!(s.len(), 7);
    } else {
        panic!("expected String");
    }
}

#[test]
fn test_pastel_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE col = PASTEL_COLOR(77) SELECT col;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::String(s)) = r.rows[0].data.get("col") {
        assert!(s.starts_with('#'));
    } else {
        panic!("expected String");
    }
}

// ── RANDOM_DARK ───────────────────────────────────────────────────────────────

#[test]
fn test_random_dark_seeded() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE col = RANDOM_DARK(55) SELECT col;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::String(s)) = r.rows[0].data.get("col") {
        assert!(s.starts_with('#'));
        assert_eq!(s.len(), 7);
    } else {
        panic!("expected String");
    }
}

#[test]
fn test_dark_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE col = DARK_COLOR(33) SELECT col;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(Value::String(s)) = r.rows[0].data.get("col") {
        assert!(s.starts_with('#'));
    } else {
        panic!("expected String");
    }
}

// ── COLOR_TEMP_K ──────────────────────────────────────────────────────────────

#[test]
fn test_color_temp_k_warm() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Very red: high r, low b -> low temperature (warm)
    let mut p = Parser::new(r#"QUERY t COMPUTE temp = COLOR_TEMP_K(255, 128, 0) SELECT temp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("temp") {
        assert!(
            to_f64(v) < 3000.0,
            "warm color should have low temp, got {}",
            to_f64(v)
        );
    }
}

#[test]
fn test_color_temp_k_cool() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Very blue: low r, high b -> high temperature (cool)
    let mut p = Parser::new(r#"QUERY t COMPUTE temp = COLOR_TEMP_K(0, 128, 255) SELECT temp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("temp") {
        assert!(
            to_f64(v) > 6500.0,
            "cool color should have high temp, got {}",
            to_f64(v)
        );
    }
}

// ── IS_WARM_COLOR ─────────────────────────────────────────────────────────────

#[test]
fn test_is_warm_color_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE warm = IS_WARM_COLOR(255, 0, 0) SELECT warm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("warm"),
        Some(&Value::Bool(true)),
        "red should be warm"
    );
}

#[test]
fn test_is_warm_alias_blue() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE warm = IS_WARM(0, 0, 255) SELECT warm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("warm"),
        Some(&Value::Bool(false)),
        "blue should not be warm"
    );
}

// ── IS_COOL_COLOR ─────────────────────────────────────────────────────────────

#[test]
fn test_is_cool_color_blue() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cool = IS_COOL_COLOR(0, 0, 255) SELECT cool;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("cool"),
        Some(&Value::Bool(true)),
        "blue should be cool"
    );
}

#[test]
fn test_is_cool_alias_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cool = IS_COOL(255, 0, 0) SELECT cool;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("cool"),
        Some(&Value::Bool(false)),
        "red should not be cool"
    );
}

// ── COLOR_BRIGHTNESS ──────────────────────────────────────────────────────────

#[test]
fn test_color_brightness_white() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // White -> brightness ~255
    let mut p = Parser::new(r#"QUERY t COMPUTE br = COLOR_BRIGHTNESS(255, 255, 255) SELECT br;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("br") {
        assert!(
            (to_f64(v) - 255.0).abs() < 1.0,
            "white brightness should be ~255, got {}",
            to_f64(v)
        );
    }
}

#[test]
fn test_color_brightness_black() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE br = PERCEIVED_BRIGHTNESS(0, 0, 0) SELECT br;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("br") {
        assert!(
            to_f64(v).abs() < 0.01,
            "black brightness should be 0, got {}",
            to_f64(v)
        );
    }
}

// ── COLOR_SATURATION_LEVEL ────────────────────────────────────────────────────

#[test]
fn test_color_saturation_level_gray() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Gray: RGB(128, 128, 128) -> saturation 0
    let mut p =
        Parser::new(r#"QUERY t COMPUTE sat = COLOR_SATURATION_LEVEL(128, 128, 128) SELECT sat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("sat") {
        assert!(
            to_f64(v).abs() < 0.01,
            "gray saturation should be 0, got {}",
            to_f64(v)
        );
    }
}

#[test]
fn test_saturation_amount_alias_red() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"dummy": 1}),
    )
    .unwrap();
    // Red: RGB(255, 0, 0) -> saturation 1
    let mut p = Parser::new(r#"QUERY t COMPUTE sat = SATURATION_AMOUNT(255, 0, 0) SELECT sat;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Some(v) = r.rows[0].data.get("sat") {
        assert!(
            (to_f64(v) - 1.0).abs() < 0.01,
            "red saturation should be 1.0, got {}",
            to_f64(v)
        );
    }
}
