/// Integration tests for new PQL matrix and linear algebra functions.
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

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Float(f) => *f as i64,
        _ => i64::MIN,
    }
}

fn make_db() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1}))
        .unwrap();
    (dir, db, ex)
}

fn get_row(mat: &Value, row: usize) -> &Vec<Value> {
    match mat {
        Value::Array(rows) => match &rows[row] {
            Value::Array(cols) => cols,
            _ => panic!("expected inner array at row {}", row),
        },
        _ => panic!("expected array, got {:?}", mat),
    }
}

// ---------------------------------------------------------------------------
// MATRIX_ADD / MAT_ADD (existing, test alias inline literal)
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_add_inline() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_ADD([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 6.0).abs() < 1e-9, "expected 6.0 got {}", as_f64(&r0[0]));
    assert!((as_f64(&r0[1]) - 8.0).abs() < 1e-9, "expected 8.0 got {}", as_f64(&r0[1]));
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 10.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 12.0).abs() < 1e-9);
}

#[test]
fn test_mat_add_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_ADD([[1.0, 0.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 3.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 3.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_SUB (new alias)
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_sub() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_SUB([[10.0, 8.0], [6.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 9.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 6.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 3.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 0.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_MUL (new alias)
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_mul_identity() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_MUL([[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 2.0).abs() < 1e-9);
}

#[test]
fn test_matrix_mul_2x2() {
    let (_dir, _db, ex) = make_db();
    // [[1,2],[3,4]] * [[2,0],[1,2]] = [[4,4],[10,8]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_MUL([[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 4.0).abs() < 1e-9, "expected 4 got {}", as_f64(&r0[0]));
    assert!((as_f64(&r0[1]) - 4.0).abs() < 1e-9, "expected 4 got {}", as_f64(&r0[1]));
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 10.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 8.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_SCALAR_MUL (new alias)
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_scalar_mul() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_SCALAR_MUL([[1.0, 2.0], [3.0, 4.0]], 3.0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 3.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 6.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 9.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 12.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MAT_T (new alias for MATRIX_TRANSPOSE)
// ---------------------------------------------------------------------------

#[test]
fn test_mat_t_alias() {
    let (_dir, _db, ex) = make_db();
    // [[1,2,3],[4,5,6]]^T = [[1,4],[2,5],[3,6]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_T([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 4.0).abs() < 1e-9);
    let r2 = get_row(mat, 2);
    assert!((as_f64(&r2[0]) - 3.0).abs() < 1e-9);
    assert!((as_f64(&r2[1]) - 6.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_IDENTITY / MAT_EYE
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_identity_3x3() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_IDENTITY(3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 0.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[1]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r1[0]) - 0.0).abs() < 1e-9);
}

#[test]
fn test_mat_eye_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_EYE(2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 0.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_ZEROS / MAT_ZEROS
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_zeros() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_ZEROS(2, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    if let Value::Array(rows) = mat {
        assert_eq!(rows.len(), 2);
        if let Value::Array(cols) = &rows[0] {
            assert_eq!(cols.len(), 3);
            for c in cols { assert!((as_f64(c) - 0.0).abs() < 1e-9); }
        }
    } else {
        panic!("expected array");
    }
}

#[test]
fn test_mat_zeros_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_ZEROS(3, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(rows) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(rows.len(), 3);
    }
}

// ---------------------------------------------------------------------------
// MATRIX_ONES / MAT_ONES
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_ones() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_ONES(2, 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 1.0).abs() < 1e-9);
}

#[test]
fn test_mat_ones_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_ONES(1, 4) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(rows) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(rows.len(), 1);
        if let Value::Array(cols) = &rows[0] {
            assert_eq!(cols.len(), 4);
            for c in cols { assert!((as_f64(c) - 1.0).abs() < 1e-9); }
        }
    }
}

// ---------------------------------------------------------------------------
// MAT_DIAG
// ---------------------------------------------------------------------------

#[test]
fn test_mat_diag_from_vec() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_DIAG([2.0, 3.0, 5.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 2.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 0.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[1]) - 3.0).abs() < 1e-9);
    let r2 = get_row(mat, 2);
    assert!((as_f64(&r2[2]) - 5.0).abs() < 1e-9);
}

#[test]
fn test_mat_diag_extract() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_DIAG([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(vals) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(vals.len(), 2);
        assert!((as_f64(&vals[0]) - 1.0).abs() < 1e-9);
        assert!((as_f64(&vals[1]) - 4.0).abs() < 1e-9);
    } else {
        panic!("expected array");
    }
}

// ---------------------------------------------------------------------------
// MAT_TRACE
// ---------------------------------------------------------------------------

#[test]
fn test_mat_trace() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_TRACE([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 5.0).abs() < 1e-9);
}

#[test]
fn test_mat_trace_3x3() {
    let (_dir, _db, ex) = make_db();
    // [[1,2,3],[4,5,6],[7,8,9]] -> trace = 15
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_TRACE([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 15.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_ROWS / MAT_NROWS
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_rows() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_ROWS([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(as_i64(r.rows[0].data.get("res").unwrap()), 3);
}

#[test]
fn test_mat_nrows() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_NROWS([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(as_i64(r.rows[0].data.get("res").unwrap()), 2);
}

// ---------------------------------------------------------------------------
// MATRIX_COLS / MAT_NCOLS
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_cols() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_COLS([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(as_i64(r.rows[0].data.get("res").unwrap()), 3);
}

#[test]
fn test_mat_ncols() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_NCOLS([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(as_i64(r.rows[0].data.get("res").unwrap()), 2);
}

// ---------------------------------------------------------------------------
// MATRIX_SHAPE / MAT_SHAPE
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_shape() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_SHAPE([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Object(obj) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(as_i64(obj.get("rows").unwrap()), 2);
        assert_eq!(as_i64(obj.get("cols").unwrap()), 3);
    } else {
        panic!("expected object");
    }
}

#[test]
fn test_mat_shape_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_SHAPE([[1.0], [2.0], [3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Object(obj) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(as_i64(obj.get("rows").unwrap()), 3);
        assert_eq!(as_i64(obj.get("cols").unwrap()), 1);
    } else {
        panic!("expected object");
    }
}

// ---------------------------------------------------------------------------
// MATRIX_IS_SQUARE / IS_SQUARE_MAT
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_is_square_true() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_IS_SQUARE([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res").unwrap(), &Value::Bool(true));
}

#[test]
fn test_matrix_is_square_false() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_SQUARE_MAT([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res").unwrap(), &Value::Bool(false));
}

// ---------------------------------------------------------------------------
// MATRIX_IS_SYMMETRIC / IS_SYMMETRIC_MAT
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_is_symmetric_true() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_IS_SYMMETRIC([[1.0, 2.0], [2.0, 3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res").unwrap(), &Value::Bool(true));
}

#[test]
fn test_matrix_is_symmetric_false() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_SYMMETRIC_MAT([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res").unwrap(), &Value::Bool(false));
}

#[test]
fn test_identity_is_symmetric() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_SYMMETRIC_MAT([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res").unwrap(), &Value::Bool(true));
}

// ---------------------------------------------------------------------------
// MATRIX_FROBENIUS_NORM / MAT_FROB_NORM
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_frobenius_norm() {
    let (_dir, _db, ex) = make_db();
    // [[3,4]] -> sqrt(9+16) = 5
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_FROBENIUS_NORM([[3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 5.0).abs() < 1e-9);
}

#[test]
fn test_mat_frob_norm_identity() {
    let (_dir, _db, ex) = make_db();
    // Identity 2x2 -> sqrt(2)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_FROB_NORM([[1.0, 0.0], [0.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 2f64.sqrt()).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_MAX_ELEMENT / MAT_MAX
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_max_element() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_MAX_ELEMENT([[1.0, 5.0], [3.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 5.0).abs() < 1e-9);
}

#[test]
fn test_mat_max_negative() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_MAX([[-1.0, -2.0], [-3.0, -0.5]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - (-0.5)).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_MIN_ELEMENT / MAT_MIN
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_min_element() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_MIN_ELEMENT([[4.0, 5.0], [1.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 1.0).abs() < 1e-9);
}

#[test]
fn test_mat_min_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_MIN([[10.0, 20.0], [30.0, 5.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 5.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_SUM_ALL / MAT_SUM
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_sum_all() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_SUM_ALL([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 10.0).abs() < 1e-9);
}

#[test]
fn test_mat_sum_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_SUM([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 6.0).abs() < 1e-9);
}

#[test]
fn test_mat_sum_zeros() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_SUM(MATRIX_ZEROS(3, 3)) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 0.0).abs() < 1e-9);
}

#[test]
fn test_mat_sum_ones_3x3() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_SUM(MATRIX_ONES(3, 3)) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 9.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_INVERSE_2X2 / MAT_INV_2X2
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_inverse_2x2() {
    let (_dir, _db, ex) = make_db();
    // [[1,2],[3,4]]^-1 = [[-2, 1], [1.5, -0.5]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_INVERSE_2X2([[1.0, 2.0], [3.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - (-2.0)).abs() < 1e-9, "expected -2, got {}", as_f64(&r0[0]));
    assert!((as_f64(&r0[1]) - 1.0).abs() < 1e-9, "expected 1, got {}", as_f64(&r0[1]));
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 1.5).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - (-0.5)).abs() < 1e-9);
}

#[test]
fn test_mat_inv_2x2_verify() {
    let (_dir, _db, ex) = make_db();
    // A = [[2,1],[5,3]], A^-1 = [[3,-1],[-5,2]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_INVERSE_2X2([[2.0, 1.0], [5.0, 3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 3.0).abs() < 1e-9, "expected 3 got {}", as_f64(&r0[0]));
    assert!((as_f64(&r0[1]) - (-1.0)).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - (-5.0)).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 2.0).abs() < 1e-9);
}

#[test]
fn test_mat_inv_2x2_singular() {
    let (_dir, _db, ex) = make_db();
    // [[1,2],[2,4]] is singular -> Null
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_INV_2X2([[1.0, 2.0], [2.0, 4.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res").unwrap(), &Value::Null);
}

// ---------------------------------------------------------------------------
// MATRIX_SOLVE / MAT_SOLVE
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_solve() {
    let (_dir, _db, ex) = make_db();
    // [[2,1],[1,3]] * [x,y] = [5,10] -> x=1, y=3
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_SOLVE([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(sol) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&sol[0]) - 1.0).abs() < 1e-6, "expected x=1 got {}", as_f64(&sol[0]));
        assert!((as_f64(&sol[1]) - 3.0).abs() < 1e-6, "expected y=3 got {}", as_f64(&sol[1]));
    } else {
        panic!("expected array");
    }
}

#[test]
fn test_mat_solve_identity() {
    let (_dir, _db, ex) = make_db();
    // [[1,0],[0,1]] * [x,y] = [7,8] -> x=7, y=8
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_SOLVE([[1.0, 0.0], [0.0, 1.0]], [7.0, 8.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(sol) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&sol[0]) - 7.0).abs() < 1e-9);
        assert!((as_f64(&sol[1]) - 8.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// MATRIX_EIGENVALUES_2X2 / EIGENVALUES_2X2
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_eigenvalues_diagonal() {
    let (_dir, _db, ex) = make_db();
    // [[2,0],[0,3]] -> eigenvalues 3, 2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_EIGENVALUES_2X2([[2.0, 0.0], [0.0, 3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(eigs) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(eigs.len(), 2);
        let mut vals: Vec<f64> = eigs.iter().map(as_f64).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 2.0).abs() < 1e-9);
        assert!((vals[1] - 3.0).abs() < 1e-9);
    } else {
        panic!("expected array");
    }
}

#[test]
fn test_eigenvalues_2x2_alias() {
    let (_dir, _db, ex) = make_db();
    // [[4,1],[2,3]] -> trace=7, det=10, disc=9, eigs=5,2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = EIGENVALUES_2X2([[4.0, 1.0], [2.0, 3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(eigs) = r.rows[0].data.get("res").unwrap() {
        let mut vals: Vec<f64> = eigs.iter().map(as_f64).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 2.0).abs() < 1e-9, "expected 2 got {}", vals[0]);
        assert!((vals[1] - 5.0).abs() < 1e-9, "expected 5 got {}", vals[1]);
    }
}

#[test]
fn test_eigenvalues_identity_2x2() {
    let (_dir, _db, ex) = make_db();
    // [[1,0],[0,1]] -> both eigenvalues = 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = EIGENVALUES_2X2([[1.0, 0.0], [0.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(eigs) = r.rows[0].data.get("res").unwrap() {
        for e in eigs {
            assert!((as_f64(e) - 1.0).abs() < 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// MATRIX_HADAMARD / MAT_HADAMARD
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_hadamard() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_HADAMARD([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 5.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 12.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 21.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 32.0).abs() < 1e-9);
}

#[test]
fn test_mat_hadamard_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_HADAMARD([[2.0, 3.0], [4.0, 5.0]], [[1.0, 1.0], [1.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 2.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 3.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// MATRIX_POWER / MAT_POW
// ---------------------------------------------------------------------------

#[test]
fn test_matrix_power_zero_is_identity() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_POWER([[2.0, 1.0], [1.0, 2.0]], 0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 0.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 0.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 1.0).abs() < 1e-9);
}

#[test]
fn test_matrix_power_one_is_self() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_POW([[3.0, 1.0], [2.0, 4.0]], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 3.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 1.0).abs() < 1e-9);
}

#[test]
fn test_matrix_power_squared() {
    let (_dir, _db, ex) = make_db();
    // [[1,1],[0,1]]^2 = [[1,2],[0,1]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_POW([[1.0, 1.0], [0.0, 1.0]], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 1.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 2.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 0.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 1.0).abs() < 1e-9);
}

#[test]
fn test_matrix_power_cubed() {
    let (_dir, _db, ex) = make_db();
    // [[2,0],[0,2]]^3 = [[8,0],[0,8]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_POW([[2.0, 0.0], [0.0, 2.0]], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 8.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[1]) - 8.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// VEC_DOT
// ---------------------------------------------------------------------------

#[test]
fn test_vec_dot() {
    let (_dir, _db, ex) = make_db();
    // [1,2,3] . [4,5,6] = 32
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_DOT([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 32.0).abs() < 1e-9);
}

#[test]
fn test_vec_dot_orthogonal() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_DOT([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 0.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// VEC_CROSS_3D / CROSS_PRODUCT
// ---------------------------------------------------------------------------

#[test]
fn test_vec_cross_3d_unit() {
    let (_dir, _db, ex) = make_db();
    // [1,0,0] x [0,1,0] = [0,0,1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_CROSS_3D([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[1]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[2]) - 1.0).abs() < 1e-9);
    } else {
        panic!("expected array");
    }
}

#[test]
fn test_cross_product_alias() {
    let (_dir, _db, ex) = make_db();
    // [2,3,4] x [5,6,7] = [-3, 6, -3]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CROSS_PRODUCT([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - (-3.0)).abs() < 1e-9, "expected -3 got {}", as_f64(&v[0]));
        assert!((as_f64(&v[1]) - 6.0).abs() < 1e-9, "expected 6 got {}", as_f64(&v[1]));
        assert!((as_f64(&v[2]) - (-3.0)).abs() < 1e-9, "expected -3 got {}", as_f64(&v[2]));
    }
}

#[test]
fn test_cross_product_anticommutative() {
    let (_dir, _db, ex) = make_db();
    // [0,1,0] x [1,0,0] = [0,0,-1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CROSS_PRODUCT([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[2]) - (-1.0)).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// VEC_MAGNITUDE / VEC_NORM
// ---------------------------------------------------------------------------

#[test]
fn test_vec_magnitude_3_4() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_MAGNITUDE([3.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 5.0).abs() < 1e-9);
}

#[test]
fn test_vec_norm_unit() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_NORM([0.0, 0.0, 1.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 1.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// VEC_NORMALIZE / UNIT_VECTOR
// ---------------------------------------------------------------------------

#[test]
fn test_vec_normalize() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_NORMALIZE([3.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - 0.6).abs() < 1e-9, "expected 0.6 got {}", as_f64(&v[0]));
        assert!((as_f64(&v[1]) - 0.8).abs() < 1e-9, "expected 0.8 got {}", as_f64(&v[1]));
    } else {
        panic!("expected array");
    }
}

#[test]
fn test_unit_vector_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = UNIT_VECTOR([1.0, 0.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - 1.0).abs() < 1e-9);
        assert!((as_f64(&v[1]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[2]) - 0.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// VEC_ANGLE / VECTOR_ANGLE_DEG
// ---------------------------------------------------------------------------

#[test]
fn test_vec_angle_90() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_ANGLE([1.0, 0.0], [0.0, 1.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 90.0).abs() < 1e-9);
}

#[test]
fn test_vec_angle_zero() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VECTOR_ANGLE_DEG([1.0, 0.0], [1.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 0.0).abs() < 1e-9);
}

#[test]
fn test_vec_angle_180() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_ANGLE([1.0, 0.0], [-1.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!((as_f64(r.rows[0].data.get("res").unwrap()) - 180.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// VEC_PROJECT
// ---------------------------------------------------------------------------

#[test]
fn test_vec_project_onto_x_axis() {
    let (_dir, _db, ex) = make_db();
    // Project [3,4] onto [1,0] -> [3,0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_PROJECT([3.0, 4.0], [1.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - 3.0).abs() < 1e-9);
        assert!((as_f64(&v[1]) - 0.0).abs() < 1e-9);
    } else {
        panic!("expected array");
    }
}

// ---------------------------------------------------------------------------
// VEC_REJECT
// ---------------------------------------------------------------------------

#[test]
fn test_vec_reject_from_x_axis() {
    let (_dir, _db, ex) = make_db();
    // Reject [3,4] from [1,0] -> [0,4]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_REJECT([3.0, 4.0], [1.0, 0.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[1]) - 4.0).abs() < 1e-9);
    } else {
        panic!("expected array");
    }
}

// ---------------------------------------------------------------------------
// VEC_OUTER
// ---------------------------------------------------------------------------

#[test]
fn test_vec_outer() {
    let (_dir, _db, ex) = make_db();
    // [1,2] outer [3,4] = [[3,4],[6,8]]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = VEC_OUTER([1.0, 2.0], [3.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let mat = r.rows[0].data.get("res").unwrap();
    let r0 = get_row(mat, 0);
    assert!((as_f64(&r0[0]) - 3.0).abs() < 1e-9);
    assert!((as_f64(&r0[1]) - 4.0).abs() < 1e-9);
    let r1 = get_row(mat, 1);
    assert!((as_f64(&r1[0]) - 6.0).abs() < 1e-9);
    assert!((as_f64(&r1[1]) - 8.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// LINSPACE / LIN_SPACE
// ---------------------------------------------------------------------------

#[test]
fn test_linspace_basic() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LINSPACE(0.0, 1.0, 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(v.len(), 5);
        assert!((as_f64(&v[0]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[2]) - 0.5).abs() < 1e-9);
        assert!((as_f64(&v[4]) - 1.0).abs() < 1e-9);
    } else {
        panic!("expected array");
    }
}

#[test]
fn test_lin_space_alias() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LIN_SPACE(0.0, 10.0, 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(v.len(), 3);
        assert!((as_f64(&v[0]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[1]) - 5.0).abs() < 1e-9);
        assert!((as_f64(&v[2]) - 10.0).abs() < 1e-9);
    }
}

#[test]
fn test_linspace_single_point() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LINSPACE(3.0, 7.0, 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(v.len(), 1);
        assert!((as_f64(&v[0]) - 3.0).abs() < 1e-9);
    }
}

#[test]
fn test_linspace_negative_to_positive() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LINSPACE(-2.0, 2.0, 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Array(v) = r.rows[0].data.get("res").unwrap() {
        assert!((as_f64(&v[0]) - (-2.0)).abs() < 1e-9);
        assert!((as_f64(&v[2]) - 0.0).abs() < 1e-9);
        assert!((as_f64(&v[4]) - 2.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Combined / composition tests
// ---------------------------------------------------------------------------

#[test]
fn test_zeros_shape() {
    let (_dir, _db, ex) = make_db();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MATRIX_SHAPE(MATRIX_ZEROS(4, 5)) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    if let Value::Object(obj) = r.rows[0].data.get("res").unwrap() {
        assert_eq!(as_i64(obj.get("rows").unwrap()), 4);
        assert_eq!(as_i64(obj.get("cols").unwrap()), 5);
    }
}

#[test]
fn test_mat_mul_mat_mul() {
    let (_dir, _db, ex) = make_db();
    // MATRIX_MUL with MAT_MUL existing alias both work
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAT_MUL([[1.0, 0.0], [0.0, 1.0]], [[5.0, 6.0], [7.0, 8.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let r0 = get_row(r.rows[0].data.get("res").unwrap(), 0);
    assert!((as_f64(&r0[0]) - 5.0).abs() < 1e-9);
}
