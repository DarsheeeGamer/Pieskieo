/// Integration tests for PQL built-in color space conversion and manipulation functions.
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

fn to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

// ── RGB_TO_HEX ───────────────────────────────────────────────────────────────

#[test]
fn test_rgb_to_hex_red() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = RGB_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("#FF0000".to_string())));
}

#[test]
fn test_rgb_to_hex_green() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 255, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = RGB_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("#00FF00".to_string())));
}

#[test]
fn test_rgb_to_hex_blue() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = RGB_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("#0000FF".to_string())));
}

#[test]
fn test_color_to_hex_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 165, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = COLOR_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("#FFA500".to_string())));
}

#[test]
fn test_rgb_to_hex_black() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = RGB_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("#000000".to_string())));
}

#[test]
fn test_rgb_to_hex_white() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 255, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = RGB_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("h"), Some(&Value::String("#FFFFFF".to_string())));
}

// ── HEX_TO_RGB ───────────────────────────────────────────────────────────────

#[test]
fn test_hex_to_rgb_red() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"hex": "#FF0000"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = HEX_TO_RGB(hex) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(255)));
            assert_eq!(m.get("g"), Some(&Value::Integer(0)));
            assert_eq!(m.get("b"), Some(&Value::Integer(0)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_hex_to_rgb_without_hash() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"hex": "00FF00"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = HEX_TO_RGB(hex) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(0)));
            assert_eq!(m.get("g"), Some(&Value::Integer(255)));
            assert_eq!(m.get("b"), Some(&Value::Integer(0)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_hex_to_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"hex": "#0000FF"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE obj = HEX_TO_COLOR(hex) SELECT obj;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(0)));
            assert_eq!(m.get("g"), Some(&Value::Integer(0)));
            assert_eq!(m.get("b"), Some(&Value::Integer(255)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── RGB_TO_HSL ───────────────────────────────────────────────────────────────

#[test]
fn test_rgb_to_hsl_red() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsl = RGB_TO_HSL(c) SELECT hsl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsl") {
        Some(Value::Object(m)) => {
            let h = to_f64(m.get("h").unwrap());
            let s = to_f64(m.get("s").unwrap());
            let l = to_f64(m.get("l").unwrap());
            assert!((h - 0.0).abs() < 1.0, "hue should be ~0, got {}", h);
            assert!((s - 1.0).abs() < 0.01, "saturation should be ~1.0, got {}", s);
            assert!((l - 0.5).abs() < 0.01, "lightness should be ~0.5, got {}", l);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_color_to_hsl_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 255, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsl = COLOR_TO_HSL(c) SELECT hsl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsl") {
        Some(Value::Object(m)) => {
            let h = to_f64(m.get("h").unwrap());
            assert!((h - 120.0).abs() < 1.0, "hue for green should be ~120, got {}", h);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_rgb_to_hsl_white() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 255, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsl = RGB_TO_HSL(c) SELECT hsl;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsl") {
        Some(Value::Object(m)) => {
            let s = to_f64(m.get("s").unwrap());
            let l = to_f64(m.get("l").unwrap());
            assert!((s - 0.0).abs() < 0.01, "saturation for white should be 0, got {}", s);
            assert!((l - 1.0).abs() < 0.01, "lightness for white should be 1.0, got {}", l);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── HSL_TO_RGB ───────────────────────────────────────────────────────────────

#[test]
fn test_hsl_to_rgb_red() {
    // h=0, s=1.0, l=0.5 → red [255, 0, 0]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"h": 0.0, "s": 1.0, "l": 0.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = HSL_TO_RGB(h, s, l) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Object(m)) => {
            let rv = to_f64(m.get("r").unwrap());
            let gv = to_f64(m.get("g").unwrap());
            let bv = to_f64(m.get("b").unwrap());
            assert!((rv - 255.0).abs() < 2.0, "r should be ~255, got {}", rv);
            assert!((gv - 0.0).abs() < 2.0, "g should be ~0, got {}", gv);
            assert!((bv - 0.0).abs() < 2.0, "b should be ~0, got {}", bv);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_hsl_to_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"h": 240.0, "s": 1.0, "l": 0.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = HSL_TO_COLOR(h, s, l) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Object(m)) => {
            let bv = to_f64(m.get("b").unwrap());
            assert!((bv - 255.0).abs() < 2.0, "b should be ~255 for blue, got {}", bv);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── RGB_TO_HSV ───────────────────────────────────────────────────────────────

#[test]
fn test_rgb_to_hsv_red() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsv = RGB_TO_HSV(c) SELECT hsv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsv") {
        Some(Value::Object(m)) => {
            let h = to_f64(m.get("h").unwrap());
            let s = to_f64(m.get("s").unwrap());
            let v = to_f64(m.get("v").unwrap());
            assert!((h - 0.0).abs() < 1.0, "hue should be ~0, got {}", h);
            assert!((s - 1.0).abs() < 0.01, "saturation should be ~1.0, got {}", s);
            assert!((v - 1.0).abs() < 0.01, "value should be ~1.0, got {}", v);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_color_to_hsv_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hsv = COLOR_TO_HSV(c) SELECT hsv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hsv") {
        Some(Value::Object(m)) => {
            let v = to_f64(m.get("v").unwrap());
            assert!((v - 0.0).abs() < 0.01, "value for black should be 0, got {}", v);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── LUMINANCE ────────────────────────────────────────────────────────────────

#[test]
fn test_luminance_black() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lum = LUMINANCE(c) SELECT lum;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lum = to_f64(r.rows[0].data.get("lum").unwrap());
    assert!((lum - 0.0).abs() < 0.0001, "luminance of black should be 0, got {}", lum);
}

#[test]
fn test_luminance_white() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 255, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lum = LUMINANCE(c) SELECT lum;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lum = to_f64(r.rows[0].data.get("lum").unwrap());
    assert!((lum - 1.0).abs() < 0.001, "luminance of white should be ~1.0, got {}", lum);
}

#[test]
fn test_relative_luminance_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lum = RELATIVE_LUMINANCE(c) SELECT lum;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let lum = to_f64(r.rows[0].data.get("lum").unwrap());
    // Red luminance is ~0.2126
    assert!(lum > 0.2 && lum < 0.22, "luminance of red should be ~0.2126, got {}", lum);
}

// ── CONTRAST_RATIO ───────────────────────────────────────────────────────────

#[test]
fn test_contrast_ratio_black_white() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": [0, 0, 0], "b_col": [255, 255, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cr = CONTRAST_RATIO(a, b_col) SELECT cr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cr = to_f64(r.rows[0].data.get("cr").unwrap());
    assert!((cr - 21.0).abs() < 0.1, "contrast ratio black/white should be ~21.0, got {}", cr);
}

#[test]
fn test_color_contrast_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": [0, 0, 0], "b_col": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE cr = COLOR_CONTRAST(a, b_col) SELECT cr;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cr = to_f64(r.rows[0].data.get("cr").unwrap());
    assert!((cr - 1.0).abs() < 0.01, "contrast ratio same colors should be 1.0, got {}", cr);
}

// ── DARKEN ───────────────────────────────────────────────────────────────────

#[test]
fn test_darken_red() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0], "amt": 0.1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dc = DARKEN(c, amt) SELECT dc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dc") {
        Some(Value::Array(arr)) => {
            let rv = to_f64(&arr[0]);
            // Red channel should be less than 255 after darkening
            assert!(rv < 255.0, "darkened red R channel should be < 255, got {}", rv);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_darken_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [200, 100, 50], "amt": 0.2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dc = DARKEN_COLOR(c, amt) SELECT dc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dc") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "result should have 3 channels");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── LIGHTEN ──────────────────────────────────────────────────────────────────

#[test]
fn test_lighten_black() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0], "amt": 0.1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lc = LIGHTEN(c, amt) SELECT lc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("lc") {
        Some(Value::Array(arr)) => {
            let rv = to_f64(&arr[0]);
            let gv = to_f64(&arr[1]);
            let bv = to_f64(&arr[2]);
            // Lightened black should have non-zero values
            assert!(rv > 0.0 || gv > 0.0 || bv > 0.0, "lightened black should be non-zero");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_lighten_color_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [100, 100, 100], "amt": 0.2})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE lc = LIGHTEN_COLOR(c, amt) SELECT lc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("lc") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "result should have 3 channels");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── INVERT_COLOR ─────────────────────────────────────────────────────────────

#[test]
fn test_invert_color_red() {
    // Invert [255, 0, 0] → [0, 255, 255]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE inv = INVERT_COLOR(c) SELECT inv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Integer(0));
            assert_eq!(arr[1], Value::Integer(255));
            assert_eq!(arr[2], Value::Integer(255));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_color_invert_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE inv = COLOR_INVERT(c) SELECT inv;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("inv") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr[0], Value::Integer(255));
            assert_eq!(arr[1], Value::Integer(255));
            assert_eq!(arr[2], Value::Integer(255));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── BLEND_COLORS ─────────────────────────────────────────────────────────────

#[test]
fn test_blend_colors_half() {
    // Blend red [255,0,0] and blue [0,0,255] at 0.5 → middle purple ~[128,0,128]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": [255, 0, 0], "b_col": [0, 0, 255], "ratio": 0.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE blended = BLEND_COLORS(a, b_col, ratio) SELECT blended;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("blended") {
        Some(Value::Array(arr)) => {
            let rv = to_f64(&arr[0]);
            let gv = to_f64(&arr[1]);
            let bv = to_f64(&arr[2]);
            assert!((rv - 128.0).abs() < 2.0, "blended R should be ~128, got {}", rv);
            assert!((gv - 0.0).abs() < 2.0, "blended G should be ~0, got {}", gv);
            assert!((bv - 128.0).abs() < 2.0, "blended B should be ~128, got {}", bv);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_mix_colors_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"a": [100, 200, 50], "b_col": [50, 100, 150], "ratio": 0.5})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE blended = MIX_COLORS(a, b_col, ratio) SELECT blended;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("blended") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "blend result should have 3 channels");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── GRAYSCALE ────────────────────────────────────────────────────────────────

#[test]
fn test_grayscale_red() {
    // GRAYSCALE([255, 0, 0]) = round(0.299*255) = round(76.245) = 76
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE g = GRAYSCALE(c) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Integer(v)) => {
            assert!(*v >= 75 && *v <= 77, "grayscale of red should be ~76, got {}", v);
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_to_grayscale_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 255, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE g = TO_GRAYSCALE(c) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Integer(v)) => {
            assert_eq!(*v, 255, "grayscale of white should be 255");
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_grayscale_black() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE g = GRAYSCALE(c) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Integer(v)) => {
            assert_eq!(*v, 0, "grayscale of black should be 0");
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── IS_DARK_COLOR ────────────────────────────────────────────────────────────

#[test]
fn test_is_dark_color_black() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dark = IS_DARK_COLOR(c) SELECT dark;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dark"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_dark_color_white() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 255, 255]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dark = IS_DARK_COLOR(c) SELECT dark;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dark"), Some(&Value::Bool(false)));
}

#[test]
fn test_color_is_dark_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [50, 50, 50]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dark = COLOR_IS_DARK(c) SELECT dark;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("dark"), Some(&Value::Bool(true)));
}

// ── COMPLEMENTARY_COLOR ──────────────────────────────────────────────────────

#[test]
fn test_complementary_color_red() {
    // Complement of red [255,0,0] should be cyan [0,255,255]
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [255, 0, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE comp = COMPLEMENTARY_COLOR(c) SELECT comp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("comp") {
        Some(Value::Array(arr)) => {
            let rv = to_f64(&arr[0]);
            let gv = to_f64(&arr[1]);
            let bv = to_f64(&arr[2]);
            assert!(rv < 5.0, "complementary of red: R should be ~0, got {}", rv);
            assert!(gv > 250.0, "complementary of red: G should be ~255, got {}", gv);
            assert!(bv > 250.0, "complementary of red: B should be ~255, got {}", bv);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_complement_hue_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [0, 255, 0]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE comp = COMPLEMENT_HUE(c) SELECT comp;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("comp") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "complement result should have 3 channels");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── COLOR_TEMPERATURE ─────────────────────────────────────────────────────────

#[test]
fn test_color_temperature_6500k() {
    // 6500K daylight should return an array of 3 RGB values
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"k": 6500})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = COLOR_TEMPERATURE(k) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "COLOR_TEMPERATURE should return 3 channels");
            for ch in arr {
                let v = to_f64(ch);
                assert!(v >= 0.0 && v <= 255.0, "channel value should be in [0,255], got {}", v);
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_kelvin_to_rgb_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"k": 3000})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = KELVIN_TO_RGB(k) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "KELVIN_TO_RGB should return 3 channels");
            // 3000K is warm (orange-ish), so R should be 255
            assert_eq!(arr[0], Value::Integer(255), "3000K should have R=255");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_color_temperature_high_kelvin() {
    // Very high Kelvin (10000K) should produce bluish light
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"k": 10000})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rgb = COLOR_TEMPERATURE(k) SELECT rgb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rgb") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "result should have 3 channels");
            // High kelvin should have B=255
            assert_eq!(arr[2], Value::Integer(255), "10000K should have B=255");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── Round-trip tests ──────────────────────────────────────────────────────────

#[test]
fn test_hex_rgb_hex_roundtrip() {
    // HEX → RGB → HEX should return original
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"c": [128, 64, 192]})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = RGB_TO_HEX(c) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let hex = match r.rows[0].data.get("h") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String, got {:?}", other),
    };

    // Now decode it back
    let (db2, ex2) = setup();
    db2.put_doc_ns(None, Some("t2"), Uuid::new_v4(), serde_json::json!({"hex": hex})).unwrap();
    let mut p2 = Parser::new(r#"QUERY t2 COMPUTE obj = HEX_TO_RGB(hex) SELECT obj;"#);
    let r2 = ex2.execute(p2.parse().unwrap()).unwrap();
    match r2.rows[0].data.get("obj") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("r"), Some(&Value::Integer(128)));
            assert_eq!(m.get("g"), Some(&Value::Integer(64)));
            assert_eq!(m.get("b"), Some(&Value::Integer(192)));
        }
        other => panic!("expected Object, got {:?}", other),
    }
}
