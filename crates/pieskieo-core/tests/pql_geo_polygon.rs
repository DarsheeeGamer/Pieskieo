/// Integration tests for advanced PQL geospatial polygon functions.
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

// ── GEO_BBOX / BOUNDING_BOX ───────────────────────────────────────────────────

#[test]
fn test_bounding_box_square_polygon() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // A unit square from (0,0) to (1,1)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE bbox = BOUNDING_BOX([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT bbox;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(min_lat)) = obj.get("min_lat") {
                assert!((*min_lat - 0.0).abs() < 1e-10, "min_lat should be 0.0, got {}", min_lat);
            } else { panic!("min_lat missing or wrong type"); }
            if let Some(Value::Float(max_lat)) = obj.get("max_lat") {
                assert!((*max_lat - 1.0).abs() < 1e-10, "max_lat should be 1.0, got {}", max_lat);
            } else { panic!("max_lat missing or wrong type"); }
            if let Some(Value::Float(min_lon)) = obj.get("min_lon") {
                assert!((*min_lon - 0.0).abs() < 1e-10, "min_lon should be 0.0, got {}", min_lon);
            } else { panic!("min_lon missing or wrong type"); }
            if let Some(Value::Float(max_lon)) = obj.get("max_lon") {
                assert!((*max_lon - 1.0).abs() < 1e-10, "max_lon should be 1.0, got {}", max_lon);
            } else { panic!("max_lon missing or wrong type"); }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_bounding_box_alias_returns_object() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Triangle with known extremes
    let mut p = Parser::new(
        r#"QUERY t COMPUTE bbox = BOUNDING_BOX([[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]) SELECT bbox;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("min_lat"), "should have min_lat");
            assert!(obj.contains_key("max_lat"), "should have max_lat");
            assert!(obj.contains_key("min_lon"), "should have min_lon");
            assert!(obj.contains_key("max_lon"), "should have max_lon");
            if let Some(Value::Float(max_lat)) = obj.get("max_lat") {
                assert!((*max_lat - 30.0).abs() < 1e-10, "max_lat should be 30.0, got {}", max_lat);
            }
            if let Some(Value::Float(max_lon)) = obj.get("max_lon") {
                assert!((*max_lon - 40.0).abs() < 1e-10, "max_lon should be 40.0, got {}", max_lon);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEO_CENTROID / POLYGON_CENTROID ──────────────────────────────────────────

#[test]
fn test_geo_centroid_unit_square() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Centroid of a unit square (0,0)-(1,0)-(1,1)-(0,1) should be (0.5, 0.5)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE c = GEO_CENTROID([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 0.5).abs() < 1e-10, "centroid lat should be 0.5, got {}", lat);
            } else { panic!("lat missing or wrong type"); }
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 0.5).abs() < 1e-10, "centroid lon should be 0.5, got {}", lon);
            } else { panic!("lon missing or wrong type"); }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_polygon_centroid_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Equilateral triangle with centroid at (1.0, 1.0)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE c = POLYGON_CENTROID([[0.0, 1.0], [2.0, 1.0], [1.0, 1.0]]) SELECT c;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "should have lat");
            assert!(obj.contains_key("lon"), "should have lon");
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 1.0).abs() < 1e-10, "centroid lat should be 1.0, got {}", lat);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEO_AREA_KM2 / POLYGON_AREA ──────────────────────────────────────────────

#[test]
fn test_geo_area_km2_equator_degree_square() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // 1-degree square at equator: approx 111.32 * 111.32 = ~12392 km²
    let mut p = Parser::new(
        r#"QUERY t COMPUTE area = GEO_AREA_KM2([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT area;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("area") {
        Some(Value::Float(f)) => {
            assert!(*f > 10000.0 && *f < 15000.0,
                "1-deg square at equator should be ~12392 km², got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_polygon_area_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE area = POLYGON_AREA([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT area;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("area") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.0, "area should be positive, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_polygon_area_too_few_points_returns_zero() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Less than 3 points => area = 0
    let mut p = Parser::new(
        r#"QUERY t COMPUTE area = GEO_AREA_KM2([[0.0, 0.0], [1.0, 0.0]]) SELECT area;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("area") {
        Some(Value::Float(f)) => assert!((*f).abs() < 1e-10, "expected 0 for 2-point polygon, got {}", f),
        other => panic!("expected Float(0.0), got {:?}", other),
    }
}

// ── POINT_IN_POLYGON / GEO_CONTAINS ──────────────────────────────────────────

#[test]
fn test_point_in_polygon_inside() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Point (0.5, 0.5) is inside the unit square
    let mut p = Parser::new(
        r#"QUERY t COMPUTE inside = POINT_IN_POLYGON([0.5, 0.5], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT inside;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("inside"), Some(&Value::Bool(true)), "point should be inside");
}

#[test]
fn test_point_in_polygon_outside() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Point (2.0, 2.0) is outside the unit square
    let mut p = Parser::new(
        r#"QUERY t COMPUTE inside = POINT_IN_POLYGON([2.0, 2.0], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT inside;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("inside"), Some(&Value::Bool(false)), "point should be outside");
}

#[test]
fn test_geo_contains_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // GEO_CONTAINS alias: point (0.5, 0.5) inside the unit square
    let mut p = Parser::new(
        r#"QUERY t COMPUTE inside = GEO_CONTAINS([0.5, 0.5], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT inside;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("inside"), Some(&Value::Bool(true)), "GEO_CONTAINS alias should return true");
}

#[test]
fn test_point_in_polygon_outside_via_geo_contains() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Point (-1.0, -1.0) is outside the unit square
    let mut p = Parser::new(
        r#"QUERY t COMPUTE outside = GEO_CONTAINS([-1.0, -1.0], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT outside;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("outside"), Some(&Value::Bool(false)), "point should be outside");
}

// ── GEO_PERIMETER / POLYGON_PERIMETER ────────────────────────────────────────

#[test]
fn test_geo_perimeter_1_degree_square() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // 1-degree square at equator: perimeter ~ 4 * 111.32 km = ~445.28 km
    let mut p = Parser::new(
        r#"QUERY t COMPUTE perim = GEO_PERIMETER([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT perim;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("perim") {
        Some(Value::Float(f)) => {
            assert!(*f > 400.0 && *f < 500.0,
                "1-deg square perimeter should be ~445 km, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_polygon_perimeter_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE perim = POLYGON_PERIMETER([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) SELECT perim;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("perim") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "perimeter should be positive, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── GEO_BEARING / COMPASS_BEARING ────────────────────────────────────────────

#[test]
fn test_compass_bearing_north() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // From (0,0) going due north to (1,0): bearing should be ~0 degrees
    let mut p = Parser::new(
        r#"QUERY t COMPUTE b = COMPASS_BEARING(0.0, 0.0, 1.0, 0.0) SELECT b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => {
            assert!(f.abs() < 1.0 || (*f - 360.0).abs() < 1.0,
                "due north bearing should be ~0 or ~360, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_compass_bearing_east() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // From (0,0) going due east to (0,1): bearing should be ~90 degrees
    let mut p = Parser::new(
        r#"QUERY t COMPUTE b = COMPASS_BEARING(0.0, 0.0, 0.0, 1.0) SELECT b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => {
            assert!((*f - 90.0).abs() < 1.0, "due east bearing should be ~90, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_compass_bearing_south() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // From (1,0) going due south to (0,0): bearing should be ~180 degrees
    let mut p = Parser::new(
        r#"QUERY t COMPUTE b = COMPASS_BEARING(1.0, 0.0, 0.0, 0.0) SELECT b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => {
            assert!((*f - 180.0).abs() < 1.0, "due south bearing should be ~180, got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_geo_bearing_alias_result_in_range() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(),
        serde_json::json!({"lat1": 40.7128, "lon1": -74.0060, "lat2": 51.5074, "lon2": -0.1278})).unwrap();
    // GEO_BEARING is the existing function; COMPASS_BEARING is the new alias
    let mut p = Parser::new(
        r#"QUERY t COMPUTE b = COMPASS_BEARING(lat1, lon1, lat2, lon2) SELECT b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => {
            assert!(*f >= 0.0 && *f < 360.0, "bearing should be in [0,360), got {}", f);
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── GEO_DESTINATION / GEO_PROJECT ────────────────────────────────────────────

#[test]
fn test_geo_destination_north_increases_lat() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Travel north (bearing=0) from (0,0) for 100 km: lat should increase
    let mut p = Parser::new(
        r#"QUERY t COMPUTE dest = GEO_PROJECT(0.0, 0.0, 0.0, 100.0) SELECT dest;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dest") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!(*lat > 0.5, "traveling north 100km should increase lat, got {}", lat);
            } else { panic!("lat missing or wrong type in destination"); }
            assert!(obj.contains_key("lon"), "destination should have lon");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geo_project_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // GEO_PROJECT: travel east (bearing=90) from (0,0) for 111.32 km: lon should increase ~1 degree
    let mut p = Parser::new(
        r#"QUERY t COMPUTE dest = GEO_PROJECT(0.0, 0.0, 90.0, 111.32) SELECT dest;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("dest") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "destination should have lat");
            assert!(obj.contains_key("lon"), "destination should have lon");
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!(*lon > 0.5, "traveling east 111km should increase lon, got {}", lon);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEO_MIDPOINT / GEO_MIDPOINT_COORD ────────────────────────────────────────

#[test]
fn test_geo_midpoint_same_point_returns_same() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Midpoint of (10.0, 20.0) with itself should be (10.0, 20.0)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE m = GEO_MIDPOINT_COORD(10.0, 20.0, 10.0, 20.0) SELECT m;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Object(obj)) => {
            if let Some(Value::Float(lat)) = obj.get("lat") {
                assert!((*lat - 10.0).abs() < 1e-6, "midpoint lat should be 10.0, got {}", lat);
            } else { panic!("lat missing or wrong type"); }
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 20.0).abs() < 1e-6, "midpoint lon should be 20.0, got {}", lon);
            } else { panic!("lon missing or wrong type"); }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_geo_midpoint_coord_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Midpoint between (0,0) and (0,90) should be approximately (0,45)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE m = GEO_MIDPOINT_COORD(0.0, 0.0, 0.0, 90.0) SELECT m;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "midpoint should have lat");
            assert!(obj.contains_key("lon"), "midpoint should have lon");
            if let Some(Value::Float(lon)) = obj.get("lon") {
                assert!((*lon - 45.0).abs() < 1.0, "midpoint lon should be ~45, got {}", lon);
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

// ── GEO_INTERSECTION / LINES_INTERSECT ───────────────────────────────────────

#[test]
fn test_lines_intersect_crossing() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Segment (0,1)-(1,0) crosses segment (0,0)-(1,1)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE x = GEO_INTERSECTION(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0) SELECT x;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("x"), Some(&Value::Bool(true)), "crossing segments should intersect");
}

#[test]
fn test_lines_intersect_parallel() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Segment (0,0)-(0,1) and segment (1,0)-(1,1) are parallel (don't cross)
    let mut p = Parser::new(
        r#"QUERY t COMPUTE x = GEO_INTERSECTION(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0) SELECT x;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("x"), Some(&Value::Bool(false)), "parallel segments should not intersect");
}

#[test]
fn test_lines_intersect_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // LINES_INTERSECT alias: crossing segments
    let mut p = Parser::new(
        r#"QUERY t COMPUTE x = LINES_INTERSECT(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0) SELECT x;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("x"), Some(&Value::Bool(true)), "LINES_INTERSECT alias should return true for crossing segments");
}

#[test]
fn test_lines_intersect_non_crossing() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // Collinear segments that don't overlap
    let mut p = Parser::new(
        r#"QUERY t COMPUTE x = LINES_INTERSECT(0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0) SELECT x;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("x"), Some(&Value::Bool(false)), "non-crossing segments should not intersect");
}

// ── GEO_BUFFER / GEO_CIRCLE_POLYGON ──────────────────────────────────────────

#[test]
fn test_geo_buffer_default_16_sides() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // GEO_BUFFER with 3 args should default to 16 sides
    let mut p = Parser::new(
        r#"QUERY t COMPUTE poly = GEO_BUFFER(0.0, 0.0, 1.0) SELECT poly;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("poly") {
        Some(Value::Array(pts)) => {
            assert_eq!(pts.len(), 16, "default GEO_BUFFER should produce 16 vertices, got {}", pts.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_geo_buffer_custom_n_sides() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // GEO_BUFFER with 4 args specifying n_sides=8
    let mut p = Parser::new(
        r#"QUERY t COMPUTE poly = GEO_BUFFER(0.0, 0.0, 1.0, 8) SELECT poly;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("poly") {
        Some(Value::Array(pts)) => {
            assert_eq!(pts.len(), 8, "GEO_BUFFER with n=8 should produce 8 vertices, got {}", pts.len());
            // Each vertex should be an array of [lat, lon]
            match &pts[0] {
                Value::Array(coord) => assert_eq!(coord.len(), 2, "each vertex should have 2 coords"),
                other => panic!("vertex should be Array, got {:?}", other),
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_geo_circle_polygon_alias() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // GEO_CIRCLE_POLYGON alias with 32 sides
    let mut p = Parser::new(
        r#"QUERY t COMPUTE poly = GEO_CIRCLE_POLYGON(10.0, 20.0, 5.0, 32) SELECT poly;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("poly") {
        Some(Value::Array(pts)) => {
            assert_eq!(pts.len(), 32, "GEO_CIRCLE_POLYGON with n=32 should produce 32 vertices, got {}", pts.len());
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_geo_buffer_vertices_near_center() {
    let (_dir, db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({})).unwrap();
    // A circle of radius 100km around (0,0): each vertex should be within ~1 degree of center
    let mut p = Parser::new(
        r#"QUERY t COMPUTE poly = GEO_BUFFER(0.0, 0.0, 100.0, 4) SELECT poly;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("poly") {
        Some(Value::Array(pts)) => {
            assert_eq!(pts.len(), 4, "should produce 4 vertices");
            for pt in pts {
                match pt {
                    Value::Array(coord) if coord.len() == 2 => {
                        let lat = match &coord[0] { Value::Float(f) => *f, _ => panic!("lat not float") };
                        let lon = match &coord[1] { Value::Float(f) => *f, _ => panic!("lon not float") };
                        // 100km ~= 0.9 degrees; should be within ~2 degrees of center
                        assert!(lat.abs() < 2.0, "vertex lat should be near 0, got {}", lat);
                        assert!(lon.abs() < 2.0, "vertex lon should be near 0, got {}", lon);
                    }
                    other => panic!("vertex should be [lat, lon], got {:?}", other),
                }
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
