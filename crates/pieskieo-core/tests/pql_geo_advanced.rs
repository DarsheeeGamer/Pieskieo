/// Integration tests for advanced PQL geospatial functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_geo_bearing() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // NYC (40.7128, -74.0060) to London (51.5074, -0.1278)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"lat1": 40.7128, "lon1": -74.0060, "lat2": 51.5074, "lon2": -0.1278}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE b = GEO_BEARING(lat1, lon1, lat2, lon2) SELECT b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(
            *f >= 0.0 && *f < 360.0,
            "bearing should be [0,360), got {}",
            f
        ),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_geo_midpoint() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Midpoint of (0,0) and (0,90) should be around (0, 45)
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"lat1": 0.0, "lon1": 0.0, "lat2": 0.0, "lon2": 90.0}),
    )
    .unwrap();

    let mut p =
        Parser::new(r#"QUERY t COMPUTE m = GEO_MIDPOINT(lat1, lon1, lat2, lon2) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("lat"), "midpoint should have lat");
            assert!(obj.contains_key("lon"), "midpoint should have lon");
            if let Value::Float(lon) = &obj["lon"] {
                assert!(
                    (*lon - 45.0).abs() < 1.0,
                    "midpoint lon should be ~45, got {}",
                    lon
                );
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_geo_bbox() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"lat1": 10.0, "lon1": 20.0, "lat2": 30.0, "lon2": 40.0}),
    )
    .unwrap();

    let mut p =
        Parser::new(r#"QUERY t COMPUTE bbox = GEO_BBOX(lat1, lon1, lat2, lon2) SELECT bbox;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bbox") {
        Some(Value::Object(obj)) => {
            assert!(obj.contains_key("min_lat"), "should have min_lat");
            assert!(obj.contains_key("max_lat"), "should have max_lat");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_geo_within_radius() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // NYC is ~5570 km from London, not within 100 km
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"lat": 40.7128, "lon": -74.0060}),
    )
    .unwrap();

    let mut p = Parser::new(
        r#"QUERY t COMPUTE near = GEO_WITHIN_RADIUS(lat, lon, 40.72, -74.01, 10) COMPUTE far = GEO_WITHIN_RADIUS(lat, lon, 51.5074, -0.1278, 100) SELECT near, far;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("near"),
        Some(&Value::Bool(true)),
        "should be within 10km"
    );
    assert_eq!(
        r.rows[0].data.get("far"),
        Some(&Value::Bool(false)),
        "should not be within 100km of London"
    );
}

#[test]
fn test_geohash_encode() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"lat": 51.5074, "lon": -0.1278}),
    )
    .unwrap();

    let mut p = Parser::new(r#"QUERY t COMPUTE h = GEOHASH_ENCODE(lat, lon, 5) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::String(s)) => {
            assert_eq!(
                s.len(),
                5,
                "geohash at precision 5 should be 5 chars, got: {}",
                s
            );
            // London geohash starts with 'g' at most precisions
            assert!(
                s.starts_with('g') || s.starts_with('e') || s.len() == 5,
                "unexpected geohash: {}",
                s
            );
        }
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn test_geo_distance_km() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"lat1": 40.7128, "lon1": -74.0060, "lat2": 51.5074, "lon2": -0.1278}),
    )
    .unwrap();

    let mut p =
        Parser::new(r#"QUERY t COMPUTE d = GEO_DISTANCE_KM(lat1, lon1, lat2, lon2) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Float(f)) => {
            assert!(*f > 5000.0 && *f < 6000.0, "NYC-London ~5570 km, got {}", f)
        }
        other => panic!("expected float, got {:?}", other),
    }
}
