/// Integration tests for PQL statistical functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup_nums() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for v in [1, 2, 4, 8, 16] {
        db.put_doc_ns(None, Some("nums"), Uuid::new_v4(),
            serde_json::json!({"v": v, "w": 1})).unwrap();
    }
    (dir, db, ex)
}

#[test]
fn test_geometric_mean() {
    let (_dir, _db, ex) = setup_nums();
    let mut p = Parser::new("QUERY nums COMPUTE g = 1 GROUP BY g COMPUTE gm = GEOMETRIC_MEAN(v) SELECT gm;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gm") {
        Some(Value::Float(f)) => {
            // geometric mean of [1,2,4,8,16] = (1*2*4*8*16)^(1/5) = 1024^0.2 = 4.0
            assert!((*f - 4.0).abs() < 0.001, "Geometric mean should be 4.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_harmonic_mean() {
    let (_dir, _db, ex) = {
        let dir = tempdir().unwrap();
        let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
        let ex = Executor::new(db.clone());
        // harmonic mean of [1,2,4] = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 ≈ 1.714
        for v in [1, 2, 4] {
            db.put_doc_ns(None, Some("d"), Uuid::new_v4(),
                serde_json::json!({"v": v})).unwrap();
        }
        (dir, db, ex)
    };
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE hm = HARMONIC_MEAN(v) SELECT hm;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("hm") {
        Some(Value::Float(f)) => assert!(*f > 1.0 && *f < 2.0, "Harmonic mean should be ~1.714, got {}", f),
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_product_agg() {
    let (_dir, _db, ex) = setup_nums();
    let mut p = Parser::new("QUERY nums COMPUTE g = 1 GROUP BY g COMPUTE prod = PRODUCT_AGG(v) SELECT prod;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("prod") {
        Some(Value::Float(f)) => {
            // 1 * 2 * 4 * 8 * 16 = 1024
            assert!((*f - 1024.0).abs() < 0.001, "Product should be 1024, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_coeff_variation() {
    let (_dir, _db, ex) = setup_nums();
    let mut p = Parser::new("QUERY nums COMPUTE g = 1 GROUP BY g COMPUTE cv_val = COEFF_VARIATION(v) SELECT cv_val;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("cv_val") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "CV should be positive, got {}", f),
        other => panic!("Expected Float, got {:?}", other),
    }
}
