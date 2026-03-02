/// Integration tests for extended PQL geospatial built-in functions.
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

// ── GEO_BEARING / INITIAL_BEARING ────────────────────────────────────────────

#[test]
fn test_geo_bearing_london_to_nyc() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Bearing from London to New York (roughly westward ~288 degrees)
    let mut p = Parser::new(r#"QUERY t COMPUTE b = GEO_BEARING(51.5074, -0.1278, 40.7128, -74.006) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(*f > 250.0 && *f < 310.0,
            "bearing London->NYC should be ~288 degrees, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_initial_bearing_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE b = INITIAL_BEARING(0.0, 0.0, 0.0, 1.0) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!((*f - 90.0).abs() < 1.0, "due east bearing should be ~90, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geo_bearing_due_north() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE b = GEO_BEARING(0.0, 0.0, 1.0, 0.0) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(f.abs() < 1.0 || (*f - 360.0).abs() < 1.0,
            "due north should be ~0 or 360, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geo_bearing_result_in_0_360_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE b = GEO_BEARING(40.7128, -74.006, 51.5074, -0.1278) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(*f >= 0.0 && *f < 360.0, "bearing should be in [0,360), got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── GEO_DISTANCE_METERS / GEO_DIST_M ─────────────────────────────────────────

#[test]
fn test_geo_distance_meters_nyc_london() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // NYC to London ~5570 km = ~5570000 m
    let mut p = Parser::new(r#"QUERY t COMPUTE d = GEO_DISTANCE_METERS(40.7128, -74.006, 51.5074, -0.1278) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f > 5_000_000.0 && *f < 6_000_000.0,
            "NYC-London should be ~5570000m, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geo_dist_m_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = GEO_DIST_M(0.0, 0.0, 0.0, 1.0) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f > 100_000.0 && *f < 120_000.0,
            "1 degree lon at equator should be ~111km = ~111000m, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geo_distance_meters_is_km_times_1000() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // GEO_DISTANCE_METERS should equal GEO_DISTANCE_KM * 1000
    let mut p = Parser::new(r#"QUERY t COMPUTE dm = GEO_DISTANCE_METERS(10.0, 20.0, 11.0, 21.0) COMPUTE dk = GEO_DISTANCE_KM(10.0, 20.0, 11.0, 21.0) SELECT dm, dk;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let dm = match r.rows[0].data.get("dm") { Some(Value::Float(f)) => *f, _ => panic!("expected dm float") };
    let dk = match r.rows[0].data.get("dk") { Some(Value::Float(f)) => *f, _ => panic!("expected dk float") };
    assert!((dm - dk * 1000.0).abs() < 0.001, "meters should be km*1000, dm={}, dk*1000={}", dm, dk * 1000.0);
}

// ── GEO_DIST_KM ───────────────────────────────────────────────────────────────

#[test]
fn test_geo_dist_km_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Same as GEO_DISTANCE_KM
    let mut p = Parser::new(r#"QUERY t COMPUTE d = GEO_DIST_KM(40.7128, -74.006, 51.5074, -0.1278) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!(*f > 5000.0 && *f < 6000.0,
            "NYC-London should be ~5570 km, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── GEO_CENTER ────────────────────────────────────────────────────────────────

#[test]
fn test_geo_center_same_point() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = GEO_CENTER(10.0, 20.0, 10.0, 20.0) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 10.0).abs() < 1e-5, "center lat should be 10.0, got {}", lat);
            } else { panic!("lat missing"); }
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 20.0).abs() < 1e-5, "center lon should be 20.0, got {}", lon);
            } else { panic!("lon missing"); }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geo_center_has_lat_lon_keys() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE m = GEO_CENTER(0.0, 0.0, 0.0, 90.0) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "should have lat");
            assert!(obj.contains_key("lon"), "should have lon");
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 45.0).abs() < 1.0, "center lon of (0,0)-(0,90) should be ~45, got {}", lon);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEO_MOVE ──────────────────────────────────────────────────────────────────

#[test]
fn test_geo_move_north_increases_lat() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Travel north (bearing=0) from (0,0) for 100km
    let mut p = Parser::new(r#"QUERY t COMPUTE dest = GEO_MOVE(0.0, 0.0, 0.0, 100.0) SELECT dest;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dest") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!(*lat > 0.5, "traveling north 100km should increase lat, got {}", lat);
            } else { panic!("lat missing"); }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geo_move_returns_lat_lon() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dest = GEO_MOVE(10.0, 20.0, 90.0, 111.32) SELECT dest;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dest") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "destination should have lat");
            assert!(obj.contains_key("lon"), "destination should have lon");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEOHASH_ENCODE / GEOHASH ──────────────────────────────────────────────────

#[test]
fn test_geohash_encode_precision_11() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = GEOHASH_ENCODE(57.64911, 10.40744, 11) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => assert_eq!(s, "u4pruydqqvj",
            "geohash mismatch, got {}", s),
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_geohash_encode_precision_5() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = GEOHASH_ENCODE(51.5074, -0.1278, 5) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => {
            assert_eq!(s.len(), 5, "geohash at precision 5 should be 5 chars, got '{}'", s);
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_geohash_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE h = GEOHASH(57.64911, 10.40744, 11) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => assert_eq!(s, "u4pruydqqvj",
            "GEOHASH alias mismatch, got {}", s),
        other => panic!("expected string, got {:?}", other),
    }
}

// ── GEOHASH_DECODE / GEOHASH_TO_POINT ────────────────────────────────────────

#[test]
fn test_geohash_decode_round_trip() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Encode then decode: lat/lon should be within error bounds
    let mut p = Parser::new(r#"QUERY t COMPUTE decoded = GEOHASH_DECODE("u4pruydqqvj") SELECT decoded;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("decoded") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "decoded should have lat");
            assert!(obj.contains_key("lon"), "decoded should have lon");
            assert!(obj.contains_key("lat_err"), "decoded should have lat_err");
            assert!(obj.contains_key("lon_err"), "decoded should have lon_err");
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 57.64911).abs() < 0.001,
                    "decoded lat should be ~57.64911, got {}", lat);
            }
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 10.40744).abs() < 0.001,
                    "decoded lon should be ~10.40744, got {}", lon);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geohash_to_point_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pt = GEOHASH_TO_POINT("u4pruydqqvj") SELECT pt;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pt") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "should have lat");
            assert!(obj.contains_key("lon"), "should have lon");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geohash_decode_error_bounds_are_small_for_high_precision() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE decoded = GEOHASH_DECODE("u4pruydqqvj") SELECT decoded;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("decoded") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat_err)) = obj.get("lat_err") {
                assert!(*lat_err < 0.001, "lat_err should be very small for precision-11, got {}", lat_err);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEOHASH_NEIGHBORS / GEOHASH_ADJACENT ─────────────────────────────────────

#[test]
fn test_geohash_neighbors_has_all_directions() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nb = GEOHASH_NEIGHBORS("u4pruy") SELECT nb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nb") {
        Some(Value::Object(obj)) => {
            for dir in &["n", "ne", "e", "se", "s", "sw", "w", "nw"] {
                assert!(obj.contains_key(*dir), "neighbors should have key '{}'", dir);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geohash_adjacent_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nb = GEOHASH_ADJACENT("u4pruy") SELECT nb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nb") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("n"), "neighbors should have 'n'");
            assert!(obj.contains_key("s"), "neighbors should have 's'");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geohash_neighbors_values_are_strings() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE nb = GEOHASH_NEIGHBORS("u4pruy") SELECT nb;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("nb") {
        Some(Value::Object(obj)) => {
            for dir in &["n", "ne", "e", "se", "s", "sw", "w", "nw"] {
                match obj.get(*dir) {
                    Some(Value::String(s)) => assert_eq!(s.len(), 6,
                        "neighbor '{}' should be a 6-char geohash, got '{}'", dir, s),
                    other => panic!("neighbor '{}' should be String, got {:?}", dir, other),
                }
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEOHASH_PRECISION_TO_KM / GEOHASH_PREC_KM ────────────────────────────────

#[test]
fn test_geohash_precision_to_km_level_1() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE km = GEOHASH_PRECISION_TO_KM(1) SELECT km;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("km") {
        Some(Value::Float(f)) => assert!((*f - 2500.0).abs() < 1.0, "precision 1 should be 2500km, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geohash_precision_to_km_level_5() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE km = GEOHASH_PRECISION_TO_KM(5) SELECT km;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("km") {
        Some(Value::Float(f)) => assert!((*f - 2.4).abs() < 0.01, "precision 5 should be 2.4km, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geohash_prec_km_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE km = GEOHASH_PREC_KM(9) SELECT km;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("km") {
        Some(Value::Float(f)) => assert!((*f - 0.0024).abs() < 0.0001, "precision 9 should be 0.0024km, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geohash_precision_decreases_as_level_increases() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE k3 = GEOHASH_PRECISION_TO_KM(3) COMPUTE k6 = GEOHASH_PRECISION_TO_KM(6) SELECT k3, k6;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let k3 = match r.rows[0].data.get("k3") { Some(Value::Float(f)) => *f, _ => panic!("expected k3") };
    let k6 = match r.rows[0].data.get("k6") { Some(Value::Float(f)) => *f, _ => panic!("expected k6") };
    assert!(k3 > k6, "higher precision = smaller error; k3={} should > k6={}", k3, k6);
}

// ── GEO_POLYGON_AREA / POLY_AREA_KM2 ─────────────────────────────────────────

#[test]
fn test_geo_polygon_area_equator_degree_square() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // 1-degree square at equator: ~12392 km²
    let mut p = Parser::new(r#"QUERY t COMPUTE area = GEO_POLYGON_AREA([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT area;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("area") {
        Some(Value::Float(f)) => assert!(*f > 10000.0 && *f < 15000.0,
            "1-deg square at equator ~12392 km², got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_poly_area_km2_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE area = POLY_AREA_KM2([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT area;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("area") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "area should be positive, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geo_polygon_area_fewer_than_3_points_returns_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE area = GEO_POLYGON_AREA([[0.0, 0.0], [1.0, 0.0]]) SELECT area;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("area") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-10, "2-point polygon should have 0 area, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── GEO_POINT_IN_POLYGON / POINT_IN_POLY ─────────────────────────────────────

#[test]
fn test_geo_point_in_polygon_inside() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Point (0.5, 0.5) inside the unit square
    let mut p = Parser::new(r#"QUERY t COMPUTE inside = GEO_POINT_IN_POLYGON(0.5, 0.5, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT inside;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("inside"), Some(&Value::Bool(true)), "point should be inside");
}

#[test]
fn test_geo_point_in_polygon_outside() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Point (2.0, 2.0) outside the unit square
    let mut p = Parser::new(r#"QUERY t COMPUTE inside = GEO_POINT_IN_POLYGON(2.0, 2.0, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT inside;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("inside"), Some(&Value::Bool(false)), "point should be outside");
}

#[test]
fn test_point_in_poly_alias_inside() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE inside = POINT_IN_POLY(0.5, 0.5, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT inside;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("inside"), Some(&Value::Bool(true)), "POINT_IN_POLY alias should return true");
}

// ── GEO_CIRCLE_BBOX / GEO_RADIUS_BBOX ────────────────────────────────────────

#[test]
fn test_geo_circle_bbox_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Bounding box for a 100km circle around (0,0)
    let mut p = Parser::new(r#"QUERY t COMPUTE bbox = GEO_CIRCLE_BBOX(0.0, 0.0, 100.0) SELECT bbox;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("min_lat"), "should have min_lat");
            assert!(obj.contains_key("max_lat"), "should have max_lat");
            assert!(obj.contains_key("min_lon"), "should have min_lon");
            assert!(obj.contains_key("max_lon"), "should have max_lon");
            if let Some(Value::Float(min_lat)) = obj.get("min_lat") {
                assert!(*min_lat < 0.0, "min_lat should be negative for center at equator");
            }
            if let Some(Value::Float(max_lat)) = obj.get("max_lat") {
                assert!(*max_lat > 0.0, "max_lat should be positive for center at equator");
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geo_radius_bbox_alias_symmetric_at_equator() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bbox = GEO_RADIUS_BBOX(0.0, 0.0, 50.0) SELECT bbox;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            let min_lat = match obj.get("min_lat") { Some(Value::Float(f)) => *f, _ => panic!("no min_lat") };
            let max_lat = match obj.get("max_lat") { Some(Value::Float(f)) => *f, _ => panic!("no max_lat") };
            assert!((min_lat + max_lat).abs() < 1e-6,
                "bbox at equator should be symmetric: min_lat={} max_lat={}", min_lat, max_lat);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEO_POINTS_BBOX / POINTS_BOUNDING_BOX ────────────────────────────────────

#[test]
fn test_geo_points_bbox_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bbox = GEO_POINTS_BBOX([[10.0, 20.0], [30.0, 40.0], [20.0, 50.0]]) SELECT bbox;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("min_lat"), "should have min_lat");
            assert!(obj.contains_key("max_lat"), "should have max_lat");
            assert!(obj.contains_key("min_lon"), "should have min_lon");
            assert!(obj.contains_key("max_lon"), "should have max_lon");
            if let Some(Value::Float(v)) = obj.get("min_lat") {
                assert!((*v - 10.0).abs() < 1e-10, "min_lat should be 10, got {}", v);
            }
            if let Some(Value::Float(v)) = obj.get("max_lat") {
                assert!((*v - 30.0).abs() < 1e-10, "max_lat should be 30, got {}", v);
            }
            if let Some(Value::Float(v)) = obj.get("min_lon") {
                assert!((*v - 20.0).abs() < 1e-10, "min_lon should be 20, got {}", v);
            }
            if let Some(Value::Float(v)) = obj.get("max_lon") {
                assert!((*v - 50.0).abs() < 1e-10, "max_lon should be 50, got {}", v);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_points_bounding_box_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE bbox = POINTS_BOUNDING_BOX([[10.0, 20.0], [30.0, 40.0]]) SELECT bbox;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("min_lat"), "POINTS_BOUNDING_BOX should have min_lat");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── DEG_TO_RAD / DEGREES_TO_RADIANS ──────────────────────────────────────────

#[test]
fn test_deg_to_rad_180() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DEG_TO_RAD(180.0) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => {
            let pi = std::f64::consts::PI;
            assert!((*f - pi).abs() < 1e-10, "180 deg should be pi rad, got {}", f);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_degrees_to_radians_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DEGREES_TO_RADIANS(90.0) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => {
            let half_pi = std::f64::consts::PI / 2.0;
            assert!((*f - half_pi).abs() < 1e-10, "90 deg should be pi/2 rad, got {}", f);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ── RAD_TO_DEG / RADIANS_TO_DEGREES ──────────────────────────────────────────

#[test]
fn test_rad_to_deg_pi() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // PI rad = 180 degrees
    let mut p = Parser::new(r#"QUERY t COMPUTE d = RAD_TO_DEG(3.141592653589793) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 180.0).abs() < 1e-8, "pi rad should be 180 deg, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_radians_to_degrees_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE d = RADIANS_TO_DEGREES(1.5707963267948966) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => assert!((*f - 90.0).abs() < 1e-8, "pi/2 rad should be 90 deg, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_rad_deg_round_trip() {
    let (_dir, db, ex) = setup();
    // 45 degrees in radians = PI/4 = 0.7853981633974483
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"deg": 45.0, "rad": 0.7853981633974483})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE r = DEG_TO_RAD(deg) COMPUTE back = RAD_TO_DEG(rad) SELECT r, back;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let rad_val = match r.rows[0].data.get("r") { Some(Value::Float(f)) => *f, _ => panic!("expected r float") };
    let back_val = match r.rows[0].data.get("back") { Some(Value::Float(f)) => *f, _ => panic!("expected back float") };
    assert!((rad_val - 0.7853981633974483).abs() < 1e-10, "45 deg should be pi/4 rad, got {}", rad_val);
    assert!((back_val - 45.0).abs() < 1e-8, "pi/4 rad should be 45 deg, got {}", back_val);
}

// ── DMS_TO_DECIMAL / DMS_TO_DD ────────────────────────────────────────────────

#[test]
fn test_dms_to_decimal_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // (40, 26, 46.302) -> 40.446195
    let mut p = Parser::new(r#"QUERY t COMPUTE dd = DMS_TO_DECIMAL(40.0, 26.0, 46.302) SELECT dd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dd") {
        Some(Value::Float(f)) => assert!((*f - 40.446195).abs() < 0.0001,
            "DMS should be ~40.446195, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_dms_to_dd_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dd = DMS_TO_DD(40.0, 26.0, 46.302) SELECT dd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dd") {
        Some(Value::Float(f)) => assert!((*f - 40.446195).abs() < 0.0001,
            "DMS_TO_DD alias should be ~40.446195, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── DECIMAL_TO_DMS / DD_TO_DMS ────────────────────────────────────────────────

#[test]
fn test_decimal_to_dms_basic() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dms = DECIMAL_TO_DMS(40.446195) SELECT dms;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dms") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("degrees"), "should have degrees");
            assert!(obj.contains_key("minutes"), "should have minutes");
            assert!(obj.contains_key("seconds"), "should have seconds");
            if let Some(Value::Float(d)) = obj.get("degrees") {
                assert!((*d - 40.0).abs() < 1e-6, "degrees should be 40, got {}", d);
            }
            if let Some(Value::Float(m)) = obj.get("minutes") {
                assert!((*m - 26.0).abs() < 1e-4, "minutes should be ~26, got {}", m);
            }
            if let Some(Value::Float(s)) = obj.get("seconds") {
                assert!((*s - 46.302).abs() < 0.01, "seconds should be ~46.302, got {}", s);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_dd_to_dms_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE dms = DD_TO_DMS(40.446195) SELECT dms;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dms") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("degrees"), "DD_TO_DMS alias should have degrees");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_dms_round_trip() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Convert 51.5074 -> DMS -> DD should give back ~51.5074
    let mut p = Parser::new(r#"QUERY t COMPUTE dms = DECIMAL_TO_DMS(51.5074) SELECT dms;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dms") {
        Some(Value::Object(obj)) => {
            let d = match obj.get("degrees") { Some(Value::Float(f)) => *f, _ => panic!("no degrees") };
            let m = match obj.get("minutes") { Some(Value::Float(f)) => *f, _ => panic!("no minutes") };
            let s = match obj.get("seconds") { Some(Value::Float(f)) => *f, _ => panic!("no seconds") };
            let back = d + m / 60.0 + s / 3600.0;
            assert!((back - 51.5074).abs() < 0.0001,
                "round-trip should give 51.5074, got {}", back);
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── METERS_TO_DEGREES_LAT / M_TO_DEG_LAT ─────────────────────────────────────

#[test]
fn test_meters_to_degrees_lat_111320() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // 111320 m -> 1 degree lat
    let mut p = Parser::new(r#"QUERY t COMPUTE deg = METERS_TO_DEGREES_LAT(111320.0) SELECT deg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("deg") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-6, "111320m should be 1 deg lat, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_m_to_deg_lat_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE deg = M_TO_DEG_LAT(55660.0) SELECT deg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("deg") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.01, "55660m should be ~0.5 deg lat, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── METERS_TO_DEGREES_LON / M_TO_DEG_LON ─────────────────────────────────────

#[test]
fn test_meters_to_degrees_lon_at_equator() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // At equator: 111320m -> 1 degree lon
    let mut p = Parser::new(r#"QUERY t COMPUTE deg = METERS_TO_DEGREES_LON(111320.0, 0.0) SELECT deg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("deg") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 1e-6, "111320m at equator should be 1 deg lon, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_m_to_deg_lon_alias_at_60deg_lat() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // At 60 degrees lat, 1 degree lon = 111320 * cos(60) = 55660 m
    let mut p = Parser::new(r#"QUERY t COMPUTE deg = M_TO_DEG_LON(55660.0, 60.0) SELECT deg;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("deg") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.01, "55660m at 60deg lat ~1 deg lon, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── IS_VALID_LAT / VALID_LATITUDE ─────────────────────────────────────────────

#[test]
fn test_is_valid_lat_valid() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_LAT(45.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_lat_out_of_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_LAT(91.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_latitude_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_LATITUDE(-90.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_VALID_LON / VALID_LONGITUDE ────────────────────────────────────────────

#[test]
fn test_is_valid_lon_valid() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_LON(179.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_lon_out_of_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_LON(-181.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_longitude_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_LONGITUDE(0.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

// ── IS_VALID_COORDS / VALID_COORDINATES ──────────────────────────────────────

#[test]
fn test_is_valid_coords_valid() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_COORDS(51.5074, -0.1278) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_valid_coords_invalid_lat() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = IS_VALID_COORDS(100.0, 0.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn test_valid_coordinates_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE ok = VALID_COORDINATES(0.0, 200.0) SELECT ok;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("ok"), Some(&Value::Bool(false)));
}

// ── GEO_CLUSTER_CENTER / CENTROID ─────────────────────────────────────────────

#[test]
fn test_geo_cluster_center_simple() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Centroid of (0,0), (2,0), (1,2) -> lat=1, lon=0.666...
    let mut p = Parser::new(r#"QUERY t COMPUTE c = GEO_CLUSTER_CENTER([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "centroid should have lat");
            assert!(obj.contains_key("lon"), "centroid should have lon");
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 1.0).abs() < 1e-10, "centroid lat should be 1.0, got {}", lat);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_centroid_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = CENTROID([[10.0, 20.0], [20.0, 30.0]]) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "CENTROID alias should have lat");
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 15.0).abs() < 1e-10, "centroid of (10,20) and (20,30) lat should be 15, got {}", lat);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geo_cluster_center_single_point() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
    // Centroid of a single point should be that point itself
    let mut p = Parser::new(r#"QUERY t COMPUTE c = GEO_CLUSTER_CENTER([[42.0, 13.0]]) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 42.0).abs() < 1e-10, "single-point centroid lat should be 42, got {}", lat);
            }
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 13.0).abs() < 1e-10, "single-point centroid lon should be 13, got {}", lon);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}
