/// Integration tests for PQL matrix and linear algebra functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => f64::NAN,
    }
}

fn make_db(ns: &str, doc: serde_json::Value) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    (dir, db, ex)
}

// ---------------------------------------------------------------------------
// MATRIX_ADD
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_add() {
    let (_dir, _db, ex) = make_db(
        "t_mat_add",
        serde_json::json!({"a": [[1,2],[3,4]], "b": [[5,6],[7,8]]}),
    );
    let mut p = Parser::new(r#"QUERY t_mat_add COMPUTE out = MATRIX_ADD(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => {
            // [[6,8],[10,12]]
            match &rows[0] {
                Value::Array(cols) => {
                    assert!(
                        (as_f64(&cols[0]) - 6.0).abs() < 0.01,
                        "expected 6, got {}",
                        as_f64(&cols[0])
                    );
                    assert!(
                        (as_f64(&cols[1]) - 8.0).abs() < 0.01,
                        "expected 8, got {}",
                        as_f64(&cols[1])
                    );
                }
                _ => panic!("expected inner array"),
            }
            match &rows[1] {
                Value::Array(cols) => {
                    assert!(
                        (as_f64(&cols[0]) - 10.0).abs() < 0.01,
                        "expected 10, got {}",
                        as_f64(&cols[0])
                    );
                }
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected matrix array, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_SUBTRACT
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_subtract() {
    let (_dir, _db, ex) = make_db(
        "t_mat_sub",
        serde_json::json!({"a": [[10,20],[30,40]], "b": [[1,2],[3,4]]}),
    );
    let mut p = Parser::new(r#"QUERY t_mat_sub COMPUTE out = MATRIX_SUBTRACT(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => match &rows[0] {
            Value::Array(cols) => {
                assert!(
                    (as_f64(&cols[0]) - 9.0).abs() < 0.01,
                    "expected 9 for [0][0], got {}",
                    as_f64(&cols[0])
                );
            }
            _ => panic!("expected inner array"),
        },
        other => panic!("expected matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_SCALAR_MULTIPLY
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_scalar_multiply() {
    let (_dir, _db, ex) = make_db(
        "t_mat_scale",
        serde_json::json!({"m": [[1,2],[3,4]], "s": 3}),
    );
    let mut p =
        Parser::new(r#"QUERY t_mat_scale COMPUTE out = MATRIX_SCALAR_MULTIPLY(m, s) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => match &rows[0] {
            Value::Array(cols) => {
                assert!(
                    (as_f64(&cols[0]) - 3.0).abs() < 0.01,
                    "expected 3, got {}",
                    as_f64(&cols[0])
                );
                assert!(
                    (as_f64(&cols[1]) - 6.0).abs() < 0.01,
                    "expected 6, got {}",
                    as_f64(&cols[1])
                );
            }
            _ => panic!("expected inner array"),
        },
        other => panic!("expected matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_DETERMINANT – 2×2
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_determinant_2x2() {
    // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    let (_dir, _db, ex) = make_db("t_det2", serde_json::json!({"m": [[1,2],[3,4]]}));
    let mut p = Parser::new(r#"QUERY t_det2 COMPUTE d = MATRIX_DETERMINANT(m) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(v) => {
            let d = as_f64(v);
            assert!((d - (-2.0)).abs() < 1e-9, "det 2x2 expected -2, got {}", d);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_DETERMINANT – 3×3
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_determinant_3x3() {
    // det(I_3) = 1
    let (_dir, _db, ex) = make_db(
        "t_det3",
        serde_json::json!({"m": [[1,0,0],[0,1,0],[0,0,1]]}),
    );
    let mut p = Parser::new(r#"QUERY t_det3 COMPUTE d = MAT_DET(m) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(v) => {
            let d = as_f64(v);
            assert!((d - 1.0).abs() < 1e-9, "det(I_3) expected 1, got {}", d);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_INVERSE – 2×2
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_inverse_2x2() {
    // inv([[2,0],[0,4]]) = [[0.5,0],[0,0.25]]
    let (_dir, _db, ex) = make_db("t_inv2", serde_json::json!({"m": [[2,0],[0,4]]}));
    let mut p = Parser::new(r#"QUERY t_inv2 COMPUTE out = MATRIX_INVERSE(m) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => match &rows[0] {
            Value::Array(cols) => {
                assert!(
                    (as_f64(&cols[0]) - 0.5).abs() < 1e-9,
                    "expected 0.5, got {}",
                    as_f64(&cols[0])
                );
            }
            _ => panic!("expected inner array"),
        },
        other => panic!("expected matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_INVERSE – singular returns Null
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_inverse_singular_is_null() {
    // [[1,2],[2,4]] has det=0 -> Null
    let (_dir, _db, ex) = make_db("t_inv_sing", serde_json::json!({"m": [[1,2],[2,4]]}));
    let mut p = Parser::new(r#"QUERY t_inv_sing COMPUTE out = MATRIX_INVERSE(m) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Null) | None => {}
        other => panic!("expected Null for singular matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// FROBENIUS_NORM
// ---------------------------------------------------------------------------

#[test]
fn test_frobenius_norm() {
    // norm([[3,4]]) = sqrt(9+16) = 5
    let (_dir, _db, ex) = make_db("t_frob", serde_json::json!({"m": [[3,4]]}));
    let mut p = Parser::new(r#"QUERY t_frob COMPUTE n = FROBENIUS_NORM(m) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("n") {
        Some(v) => {
            let n = as_f64(v);
            assert!((n - 5.0).abs() < 1e-9, "expected 5, got {}", n);
        }
        other => panic!("expected float, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_RANK
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_rank() {
    // [[1,2,3],[4,5,6],[7,8,9]] has rank 2
    let (_dir, _db, ex) = make_db(
        "t_rank",
        serde_json::json!({"m": [[1,2,3],[4,5,6],[7,8,9]]}),
    );
    let mut p = Parser::new(r#"QUERY t_rank COMPUTE r = MATRIX_RANK(m) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Integer(rk)) => {
            assert_eq!(*rk, 2, "expected rank 2, got {}", rk);
        }
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// HADAMARD_PRODUCT
// ---------------------------------------------------------------------------

#[test]
fn test_hadamard_product() {
    // [[1,2],[3,4]] .* [[2,3],[4,5]] = [[2,6],[12,20]]
    let (_dir, _db, ex) = make_db(
        "t_hadamard",
        serde_json::json!({"a": [[1,2],[3,4]], "b": [[2,3],[4,5]]}),
    );
    let mut p = Parser::new(r#"QUERY t_hadamard COMPUTE out = HADAMARD_PRODUCT(a, b) SELECT out;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("out") {
        Some(Value::Array(rows)) => match &rows[0] {
            Value::Array(cols) => {
                assert!(
                    (as_f64(&cols[0]) - 2.0).abs() < 0.01,
                    "expected 2, got {}",
                    as_f64(&cols[0])
                );
                assert!(
                    (as_f64(&cols[1]) - 6.0).abs() < 0.01,
                    "expected 6, got {}",
                    as_f64(&cols[1])
                );
            }
            _ => panic!("expected inner array"),
        },
        other => panic!("expected matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// PAIRWISE_DISTANCE (euclidean)
// ---------------------------------------------------------------------------

#[test]
fn test_pairwise_distance_euclidean() {
    // points [0,0], [3,4] -> dist = 5
    let (_dir, _db, ex) = make_db("t_pdist", serde_json::json!({"pts": [[0,0],[3,4]]}));
    let mut p =
        Parser::new(r#"QUERY t_pdist COMPUTE d = PAIRWISE_DISTANCE(pts, "euclidean") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Array(rows)) => {
            // rows[0][1] = dist([0,0],[3,4]) = 5
            match &rows[0] {
                Value::Array(cols) => {
                    assert!(
                        (as_f64(&cols[1]) - 5.0).abs() < 1e-9,
                        "expected 5, got {}",
                        as_f64(&cols[1])
                    );
                }
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected distance matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// PAIRWISE_DISTANCE (manhattan)
// ---------------------------------------------------------------------------

#[test]
fn test_pairwise_distance_manhattan() {
    // [0,0] to [3,4] manhattan = 7
    let (_dir, _db, ex) = make_db("t_pdist_m", serde_json::json!({"pts": [[0,0],[3,4]]}));
    let mut p =
        Parser::new(r#"QUERY t_pdist_m COMPUTE d = DISTANCE_MATRIX(pts, "manhattan") SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Array(rows)) => match &rows[0] {
            Value::Array(cols) => {
                assert!(
                    (as_f64(&cols[1]) - 7.0).abs() < 1e-9,
                    "expected 7, got {}",
                    as_f64(&cols[1])
                );
            }
            _ => panic!("expected inner array"),
        },
        other => panic!("expected distance matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// GRAM_MATRIX
// ---------------------------------------------------------------------------

#[test]
fn test_gram_matrix() {
    // [[1,2],[3,4]] -> [[1*1+2*2, 1*3+2*4],[3*1+4*2, 3*3+4*4]]
    //               = [[5,11],[11,25]]
    let (_dir, _db, ex) = make_db("t_gram", serde_json::json!({"m": [[1,2],[3,4]]}));
    let mut p = Parser::new(r#"QUERY t_gram COMPUTE g = GRAM_MATRIX(m) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Array(rows)) => {
            match &rows[0] {
                Value::Array(cols) => {
                    assert!(
                        (as_f64(&cols[0]) - 5.0).abs() < 1e-9,
                        "expected 5, got {}",
                        as_f64(&cols[0])
                    );
                    assert!(
                        (as_f64(&cols[1]) - 11.0).abs() < 1e-9,
                        "expected 11, got {}",
                        as_f64(&cols[1])
                    );
                }
                _ => panic!("expected inner array"),
            }
            match &rows[1] {
                Value::Array(cols) => {
                    assert!(
                        (as_f64(&cols[1]) - 25.0).abs() < 1e-9,
                        "expected 25, got {}",
                        as_f64(&cols[1])
                    );
                }
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected gram matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// COVARIANCE_MATRIX
// ---------------------------------------------------------------------------

#[test]
fn test_covariance_matrix() {
    // Two identical rows -> off-diagonal cov = variance of the row
    // row = [1,2,3,4,5], mean=3, var=2
    let (_dir, _db, ex) = make_db("t_cov", serde_json::json!({"m": [[1,2,3,4,5],[1,2,3,4,5]]}));
    let mut p = Parser::new(r#"QUERY t_cov COMPUTE c = COVARIANCE_MATRIX(m) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Array(rows)) => {
            // cov[0][0] = variance = 2.0
            match &rows[0] {
                Value::Array(cols) => {
                    let v = as_f64(&cols[0]);
                    assert!((v - 2.0).abs() < 1e-9, "expected variance 2, got {}", v);
                }
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected covariance matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// CORRELATION_MATRIX
// ---------------------------------------------------------------------------

#[test]
fn test_correlation_matrix_identical_rows() {
    // Two identical rows -> correlation = 1.0 everywhere (including off-diag)
    let (_dir, _db, ex) = make_db(
        "t_corr",
        serde_json::json!({"m": [[1,2,3,4,5],[1,2,3,4,5]]}),
    );
    let mut p = Parser::new(r#"QUERY t_corr COMPUTE c = CORRELATION_MATRIX(m) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Array(rows)) => {
            // off-diagonal should be 1.0
            match &rows[0] {
                Value::Array(cols) => {
                    let v = as_f64(&cols[1]);
                    assert!(
                        (v - 1.0).abs() < 1e-9,
                        "expected correlation 1.0, got {}",
                        v
                    );
                }
                _ => panic!("expected inner array"),
            }
        }
        other => panic!("expected correlation matrix, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// SOLVE_LINEAR
// ---------------------------------------------------------------------------

#[test]
fn test_solve_linear_2x2() {
    // A = [[2,1],[5,7]], b = [11,13]
    // Solution: x = [7.444..., -3.888...]  ->  x0 ≈ 58/9, x1 ≈ -9/9... let's verify
    // Actually: 2x+y=11, 5x+7y=13
    // y = 11-2x  =>  5x + 7(11-2x) = 13  =>  5x + 77 - 14x = 13  =>  -9x = -64  => x=64/9
    // y = 11 - 128/9 = (99-128)/9 = -29/9
    let (_dir, _db, ex) = make_db(
        "t_solve",
        serde_json::json!({"a": [[2,1],[5,7]], "b": [11,13]}),
    );
    let mut p = Parser::new(r#"QUERY t_solve COMPUTE x = SOLVE_LINEAR(a, b) SELECT x;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("x") {
        Some(Value::Array(sol)) => {
            let x0 = as_f64(&sol[0]);
            let x1 = as_f64(&sol[1]);
            // Verify: 2*x0 + x1 ≈ 11
            let check1 = 2.0 * x0 + x1;
            let check2 = 5.0 * x0 + 7.0 * x1;
            assert!(
                (check1 - 11.0).abs() < 1e-9,
                "2*x0 + x1 should be 11, got {}",
                check1
            );
            assert!(
                (check2 - 13.0).abs() < 1e-9,
                "5*x0 + 7*x1 should be 13, got {}",
                check2
            );
        }
        other => panic!("expected solution array, got {:?}", other),
    }
}
