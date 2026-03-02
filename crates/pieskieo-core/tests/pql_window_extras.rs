/// Integration tests for additional PQL window/aggregate functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup_db_with_data(ns: &str, docs: Vec<serde_json::Value>) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    for doc in docs {
        db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    }
    (dir, db, ex)
}

#[test]
fn test_harm_mean_alias() {
    // HARM_MEAN is an alias for HARMONIC_MEAN
    // harmonic mean of [1, 2, 4] = 3 / (1 + 0.5 + 0.25) = 3 / 1.75 ≈ 1.7143
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 1}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 4}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE hm = HARM_MEAN(v) SELECT hm;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("hm") {
        Some(Value::Float(f)) => {
            let expected = 3.0 / (1.0_f64 + 0.5 + 0.25);
            assert!((*f - expected).abs() < 0.001, "HARM_MEAN should be ~{}, got {}", expected, f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_trimmed_mean() {
    // TRIMMED_MEAN([1,2,3,4,100], 0.2) excludes bottom 20% (1 value from each end)
    // Sorted: [1, 2, 3, 4, 100] -> trim 1 from each end -> [2, 3, 4] -> mean = 3.0
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 1}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 3}),
        serde_json::json!({"v": 4}),
        serde_json::json!({"v": 100}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE tm = TRIMMED_MEAN(v, 0.2) SELECT tm;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("tm") {
        Some(Value::Float(f)) => {
            assert!((*f - 3.0).abs() < 0.001, "TRIMMED_MEAN should be 3.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_coefficient_of_variation() {
    // COEFFICIENT_OF_VARIATION alias for CV/COEFF_VARIATION
    // values [10, 20, 30]: mean=20, variance=((100+0+100)/3)=66.67, stddev~8.165, CV~40.82%
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 10}),
        serde_json::json!({"v": 20}),
        serde_json::json!({"v": 30}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE cv = COEFFICIENT_OF_VARIATION(v) SELECT cv;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("cv") {
        Some(Value::Float(f)) => {
            assert!(*f > 0.0, "COEFFICIENT_OF_VARIATION should be positive, got {}", f);
            // CV should be around 40.82%
            assert!((*f - 40.82).abs() < 1.0, "Expected ~40.82, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_mean_abs_dev_alias() {
    // MEAN_ABS_DEV alias for MAD / MEAN_ABSOLUTE_DEVIATION
    // values [2, 4, 6]: mean=4, MAD = (|2-4| + |4-4| + |6-4|) / 3 = (2+0+2)/3 ≈ 1.333
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 4}),
        serde_json::json!({"v": 6}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE mad_val = MEAN_ABS_DEV(v) SELECT mad_val;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("mad_val") {
        Some(Value::Float(f)) => {
            let expected = 4.0 / 3.0;
            assert!((*f - expected).abs() < 0.001, "MEAN_ABS_DEV should be ~{:.4}, got {}", expected, f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_median_absolute_deviation() {
    // MEDIAN_ABSOLUTE_DEVIATION alias: median(|xi - median(x)|)
    // values [1, 2, 3, 4, 5]: median=3, deviations=[2,1,0,1,2], median(devs)=1
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 1}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 3}),
        serde_json::json!({"v": 4}),
        serde_json::json!({"v": 5}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE mad_med = MEDIAN_ABSOLUTE_DEVIATION(v) SELECT mad_med;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("mad_med") {
        Some(Value::Float(f)) => {
            assert!((*f - 1.0).abs() < 0.001, "MEDIAN_ABSOLUTE_DEVIATION should be 1.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_spearman_correlation() {
    // SPEARMAN_CORRELATION of perfectly correlated data should be 1.0
    // x=[1,2,3,4,5], y=[2,4,6,8,10] — monotonically increasing together
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"x": 1, "y": 2}),
        serde_json::json!({"x": 2, "y": 4}),
        serde_json::json!({"x": 3, "y": 6}),
        serde_json::json!({"x": 4, "y": 8}),
        serde_json::json!({"x": 5, "y": 10}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE sc = SPEARMAN_CORRELATION(x, y) SELECT sc;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => {
            assert!((*f - 1.0).abs() < 0.001, "SPEARMAN_CORRELATION of perfectly correlated data should be 1.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_spearman_r_alias() {
    // SPEARMAN_R alias: perfectly anti-correlated should be -1.0
    // x=[1,2,3], y=[3,2,1]
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"x": 1, "y": 3}),
        serde_json::json!({"x": 2, "y": 2}),
        serde_json::json!({"x": 3, "y": 1}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE sc = SPEARMAN_R(x, y) SELECT sc;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("sc") {
        Some(Value::Float(f)) => {
            assert!((*f - (-1.0)).abs() < 0.001, "SPEARMAN_R of anti-correlated data should be -1.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_nth_value_agg() {
    // NTH_VALUE_AGG(field, 2) — second value in group (1-based index)
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 10, "g": 1}),
        serde_json::json!({"v": 20, "g": 1}),
        serde_json::json!({"v": 30, "g": 1}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE gk = g GROUP BY gk COMPUTE nth = NTH_VALUE_AGG(v, 2) SELECT nth;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    // The second value in the group (1-based index). Order is not guaranteed by insertion,
    // but the value must be one of the three integers in the group.
    match r.rows[0].data.get("nth") {
        Some(Value::Integer(i)) => {
            assert!(
                [10i64, 20, 30].contains(i),
                "NTH_VALUE_AGG(v, 2) should return one of [10,20,30], got {}",
                i
            );
        }
        other => panic!("Expected Integer, got {:?}", other),
    }
}

#[test]
fn test_group_concat() {
    // GROUP_CONCAT alias for STRING_AGG
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"name": "Alice", "g": 1}),
        serde_json::json!({"name": "Bob", "g": 1}),
        serde_json::json!({"name": "Carol", "g": 1}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE gk = g GROUP BY gk COMPUTE names = GROUP_CONCAT(name, \",\") SELECT names;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("names") {
        Some(Value::String(s)) => {
            assert!(s.contains("Alice"), "Result should contain Alice, got: {}", s);
            assert!(s.contains("Bob"), "Result should contain Bob, got: {}", s);
            assert!(s.contains("Carol"), "Result should contain Carol, got: {}", s);
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_bit_and_agg() {
    // BIT_AND_AGG: 0b1110 & 0b1101 & 0b1011 = 0b1000 = 8
    // 14 = 0b1110, 13 = 0b1101, 11 = 0b1011
    // 14 & 13 = 12 (0b1100), 12 & 11 = 8 (0b1000)
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 14}),
        serde_json::json!({"v": 13}),
        serde_json::json!({"v": 11}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE band = BIT_AND_AGG(v) SELECT band;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("band") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 8, "BIT_AND_AGG(14, 13, 11) should be 8, got {}", i);
        }
        other => panic!("Expected Integer, got {:?}", other),
    }
}

#[test]
fn test_bit_or_agg() {
    // BIT_OR_AGG: 0b0001 | 0b0010 | 0b0100 = 0b0111 = 7
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 1}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 4}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE bor = BIT_OR_AGG(v) SELECT bor;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("bor") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 7, "BIT_OR_AGG(1, 2, 4) should be 7, got {}", i);
        }
        other => panic!("Expected Integer, got {:?}", other),
    }
}

#[test]
fn test_bit_xor_agg() {
    // BIT_XOR_AGG: 5 ^ 3 ^ 6 = (0b101 ^ 0b011) ^ 0b110 = 0b110 ^ 0b110 = 0
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 5}),
        serde_json::json!({"v": 3}),
        serde_json::json!({"v": 6}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE bxor = BIT_XOR_AGG(v) SELECT bxor;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("bxor") {
        Some(Value::Integer(i)) => {
            assert_eq!(*i, 0, "BIT_XOR_AGG(5, 3, 6) should be 0, got {}", i);
        }
        other => panic!("Expected Integer, got {:?}", other),
    }
}

#[test]
fn test_collect_set() {
    // COLLECT_SET: distinct values only, no duplicates
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 1}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 3}),
        serde_json::json!({"v": 1}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE s = COLLECT_SET(v) SELECT s;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("s") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "COLLECT_SET should have 3 distinct values, got {}", arr.len());
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_collect_list() {
    // COLLECT_LIST: all values including duplicates
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 1}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 3}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE lst = COLLECT_LIST(v) SELECT lst;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("lst") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 4, "COLLECT_LIST should have 4 values (with duplicates), got {}", arr.len());
        }
        other => panic!("Expected Array, got {:?}", other),
    }
}

#[test]
fn test_median_ad_alias() {
    // MEDIAN_AD is an alias for MEDIAN_ABSOLUTE_DEVIATION
    // values [2, 2, 3, 4, 14]: median=3, deviations=[1,1,0,1,11], sorted=[0,1,1,1,11], median=1
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 2}),
        serde_json::json!({"v": 3}),
        serde_json::json!({"v": 4}),
        serde_json::json!({"v": 14}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE mad_v = MEDIAN_AD(v) SELECT mad_v;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("mad_v") {
        Some(Value::Float(f)) => {
            assert!((*f - 1.0).abs() < 0.001, "MEDIAN_AD should be 1.0, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_pearson_r_alias() {
    // PEARSON_R alias for PEARSON_CORRELATION / CORR
    // x=[1,2,3], y=[2,4,6] — perfectly correlated, r=1.0
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"x": 1, "y": 2}),
        serde_json::json!({"x": 2, "y": 4}),
        serde_json::json!({"x": 3, "y": 6}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE pr = PEARSON_R(x, y) SELECT pr;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("pr") {
        Some(Value::Float(f)) => {
            assert!((*f - 1.0).abs() < 0.001, "PEARSON_R should be 1.0 for perfect correlation, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_rank_corr_alias() {
    // RANK_CORR alias for SPEARMAN_CORRELATION
    // x=[1,2,3], y=[3,2,1] — perfectly anti-correlated, rho=-1.0
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"x": 1, "y": 3}),
        serde_json::json!({"x": 2, "y": 2}),
        serde_json::json!({"x": 3, "y": 1}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE rc = RANK_CORR(x, y) SELECT rc;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("rc") {
        Some(Value::Float(f)) => {
            assert!((*f - (-1.0)).abs() < 0.001, "RANK_CORR should be -1.0 for anti-correlated data, got {}", f);
        }
        other => panic!("Expected Float, got {:?}", other),
    }
}

#[test]
fn test_listagg_alias() {
    // LISTAGG alias for STRING_AGG / GROUP_CONCAT
    let (_dir, _db, ex) = setup_db_with_data("d", vec![
        serde_json::json!({"name": "X"}),
        serde_json::json!({"name": "Y"}),
        serde_json::json!({"name": "Z"}),
    ]);
    let mut p = Parser::new("QUERY d COMPUTE g = 1 GROUP BY g COMPUTE agg = LISTAGG(name, \"|\") SELECT agg;");
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!r.rows.is_empty(), "Expected at least one result row");
    match r.rows[0].data.get("agg") {
        Some(Value::String(s)) => {
            assert!(s.contains('|'), "LISTAGG result should contain separator '|', got: {}", s);
            assert!(s.contains('X'), "LISTAGG result should contain 'X', got: {}", s);
        }
        other => panic!("Expected String, got {:?}", other),
    }
}
