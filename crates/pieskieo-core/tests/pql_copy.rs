use pieskieo_core::{
    pql::{Executor, Parser},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

#[test]
fn test_copy_to_csv_and_from_csv() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    // Insert some rows
    for (name, age) in [("Alice", 30), ("Bob", 25)] {
        db.put_doc_ns(
            None,
            Some("people"),
            Uuid::new_v4(),
            serde_json::json!({"name": name, "age": age}),
        )
        .unwrap();
    }
    // Copy to CSV
    let csv_path = dir.path().join("people.csv");
    let csv_path_str = csv_path.to_string_lossy().to_string();
    let mut p = Parser::new(&format!(
        "COPY people TO '{}' FORMAT CSV HEADER true;",
        csv_path_str
    ));
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows.len(), 1); // status row
    assert!(csv_path.exists());

    // Copy from CSV into new collection
    let mut p2 = Parser::new(&format!(
        "COPY people2 FROM '{}' FORMAT CSV HEADER true;",
        csv_path_str
    ));
    let r2 = ex.execute(p2.parse().unwrap()).unwrap();
    assert_eq!(r2.rows.len(), 1);

    // Verify data
    let mut p3 = Parser::new("QUERY people2 SELECT *;");
    let r3 = ex.execute(p3.parse().unwrap()).unwrap();
    assert_eq!(r3.rows.len(), 2);
}

#[test]
fn test_copy_to_json_and_from_json() {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("items"),
        Uuid::new_v4(),
        serde_json::json!({"id": 1, "label": "widget"}),
    )
    .unwrap();

    let json_path = dir.path().join("items.json");
    let json_path_str = json_path.to_string_lossy().to_string();

    // Copy to JSON
    let mut p = Parser::new(&format!("COPY items TO '{}' FORMAT JSON;", json_path_str));
    ex.execute(p.parse().unwrap()).unwrap();
    assert!(json_path.exists());

    // Copy from JSON into new collection
    let mut p2 = Parser::new(&format!(
        "COPY items2 FROM '{}' FORMAT JSON;",
        json_path_str
    ));
    ex.execute(p2.parse().unwrap()).unwrap();

    let mut p3 = Parser::new("QUERY items2 SELECT *;");
    let r3 = ex.execute(p3.parse().unwrap()).unwrap();
    assert_eq!(r3.rows.len(), 1);
}
