/// Integration tests for PQL machine learning utility functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn make_db(ns: &str, doc: serde_json::Value) -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(None, Some(ns), Uuid::new_v4(), doc).unwrap();
    (dir, db, ex)
}

fn get_float(ex: &Executor, ns: &str, query: &str, field: &str) -> f64 {
    let mut p = Parser::new(query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get(field) {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for {}, got {:?}", field, other),
    }
}

fn get_array(ex: &Executor, ns: &str, query: &str, field: &str) -> Vec<Value> {
    let _ = ns;
    let mut p = Parser::new(query);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get(field) {
        Some(Value::Array(arr)) => arr.clone(),
        other => panic!("expected Array for {}, got {:?}", field, other),
    }
}

fn val_to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Integer(i) => *i as f64,
        _ => panic!("expected numeric value, got {:?}", v),
    }
}

// ── ONE_HOT_ENCODE ──────────────────────────────────────────────────────────

#[test]
fn test_one_hot_encode_basic() {
    // ONE_HOT_ENCODE(1, 4) -> [0, 1, 0, 0]
    let (_dir, _db, ex) = make_db("t_ohe1", serde_json::json!({"idx": 1, "nc": 4}));
    let arr = get_array(&ex, "t_ohe1", "QUERY t_ohe1 COMPUTE res = ONE_HOT_ENCODE(idx, nc) SELECT res;", "res");
    assert_eq!(arr.len(), 4, "ONE_HOT_ENCODE(1,4) should produce 4 elements");
    let vals: Vec<i64> = arr.iter().map(|v| match v {
        Value::Integer(i) => *i,
        Value::Float(f) => *f as i64,
        _ => panic!("expected int in one-hot array"),
    }).collect();
    assert_eq!(vals, vec![0, 1, 0, 0], "ONE_HOT_ENCODE(1,4) should be [0,1,0,0]");
}

#[test]
fn test_one_hot_encode_first_class() {
    // ONE_HOT_ENCODE(0, 3) -> [1, 0, 0]
    let (_dir, _db, ex) = make_db("t_ohe2", serde_json::json!({"idx": 0, "nc": 3}));
    let arr = get_array(&ex, "t_ohe2", "QUERY t_ohe2 COMPUTE res = ONE_HOT_ENCODE(idx, nc) SELECT res;", "res");
    assert_eq!(arr.len(), 3);
    assert_eq!(val_to_f64(&arr[0]) as i64, 1);
    assert_eq!(val_to_f64(&arr[1]) as i64, 0);
    assert_eq!(val_to_f64(&arr[2]) as i64, 0);
}

#[test]
fn test_one_hot_alias() {
    // ONE_HOT is an alias for ONE_HOT_ENCODE
    let (_dir, _db, ex) = make_db("t_ohe3", serde_json::json!({"idx": 2, "nc": 4}));
    let arr = get_array(&ex, "t_ohe3", "QUERY t_ohe3 COMPUTE res = ONE_HOT(idx, nc) SELECT res;", "res");
    assert_eq!(arr.len(), 4);
    assert_eq!(val_to_f64(&arr[2]) as i64, 1);
    assert_eq!(val_to_f64(&arr[0]) as i64, 0);
}

// ── SOFTMAX ──────────────────────────────────────────────────────────────────

#[test]
fn test_softmax_sums_to_one() {
    // SOFTMAX([1,2,3]) should sum to 1.0
    let (_dir, _db, ex) = make_db("t_sm1", serde_json::json!({"arr": [1.0, 2.0, 3.0]}));
    let arr = get_array(&ex, "t_sm1", "QUERY t_sm1 COMPUTE res = SOFTMAX(arr) SELECT res;", "res");
    assert_eq!(arr.len(), 3, "SOFTMAX should preserve length");
    let sum: f64 = arr.iter().map(|v| val_to_f64(v)).sum();
    assert!((sum - 1.0).abs() < 1e-9, "SOFTMAX should sum to 1.0, got {}", sum);
}

#[test]
fn test_softmax_array_alias() {
    // SOFTMAX_ARRAY is an alias for SOFTMAX
    let (_dir, _db, ex) = make_db("t_sm2", serde_json::json!({"arr": [0.0, 1.0, 2.0]}));
    let arr1 = get_array(&ex, "t_sm2", "QUERY t_sm2 COMPUTE res = SOFTMAX(arr) SELECT res;", "res");
    let arr2 = get_array(&ex, "t_sm2", "QUERY t_sm2 COMPUTE res = SOFTMAX_ARRAY(arr) SELECT res;", "res");
    assert_eq!(arr1.len(), arr2.len());
    for (a, b) in arr1.iter().zip(arr2.iter()) {
        assert!((val_to_f64(a) - val_to_f64(b)).abs() < 1e-10, "SOFTMAX and SOFTMAX_ARRAY should agree");
    }
}

#[test]
fn test_softmax_all_elements_positive() {
    // All softmax outputs should be positive
    let (_dir, _db, ex) = make_db("t_sm3", serde_json::json!({"arr": [-1.0, 0.0, 1.0, 5.0]}));
    let arr = get_array(&ex, "t_sm3", "QUERY t_sm3 COMPUTE res = SOFTMAX(arr) SELECT res;", "res");
    for v in &arr {
        assert!(val_to_f64(v) > 0.0, "all softmax outputs should be positive");
    }
}

// ── SIGMOID ──────────────────────────────────────────────────────────────────

#[test]
fn test_sigmoid_at_zero() {
    // sigmoid(0) = 0.5
    let (_dir, _db, ex) = make_db("t_sig1", serde_json::json!({"xval": 0.0}));
    let f = get_float(&ex, "t_sig1", "QUERY t_sig1 COMPUTE res = SIGMOID(xval) SELECT res;", "res");
    assert!((f - 0.5).abs() < 1e-9, "sigmoid(0) should be 0.5, got {}", f);
}

#[test]
fn test_logistic_alias() {
    // LOGISTIC is an alias for SIGMOID
    let (_dir, _db, ex) = make_db("t_sig2", serde_json::json!({"xval": 1.0}));
    let a = get_float(&ex, "t_sig2", "QUERY t_sig2 COMPUTE res = SIGMOID(xval) SELECT res;", "res");
    let b = get_float(&ex, "t_sig2", "QUERY t_sig2 COMPUTE res = LOGISTIC(xval) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "SIGMOID and LOGISTIC should agree");
}

#[test]
fn test_sigmoid_large_positive() {
    // sigmoid(100) ≈ 1.0
    let (_dir, _db, ex) = make_db("t_sig3", serde_json::json!({"xval": 100.0}));
    let f = get_float(&ex, "t_sig3", "QUERY t_sig3 COMPUTE res = SIGMOID(xval) SELECT res;", "res");
    assert!((f - 1.0).abs() < 1e-9, "sigmoid(100) should be ≈1.0, got {}", f);
}

// ── RELU ──────────────────────────────────────────────────────────────────────

#[test]
fn test_relu_negative_input() {
    // RELU(-3) -> 0
    let (_dir, _db, ex) = make_db("t_relu1", serde_json::json!({"xval": -3.0}));
    let f = get_float(&ex, "t_relu1", "QUERY t_relu1 COMPUTE res = RELU(xval) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "RELU(-3) should be 0, got {}", f);
}

#[test]
fn test_relu_positive_input() {
    // RELU(5) -> 5
    let (_dir, _db, ex) = make_db("t_relu2", serde_json::json!({"xval": 5.0}));
    let f = get_float(&ex, "t_relu2", "QUERY t_relu2 COMPUTE res = RELU(xval) SELECT res;", "res");
    assert!((f - 5.0).abs() < 1e-9, "RELU(5) should be 5, got {}", f);
}

#[test]
fn test_relu_activation_alias() {
    // RELU_ACTIVATION is an alias for RELU
    let (_dir, _db, ex) = make_db("t_relu3", serde_json::json!({"xval": 3.0}));
    let a = get_float(&ex, "t_relu3", "QUERY t_relu3 COMPUTE res = RELU(xval) SELECT res;", "res");
    let b = get_float(&ex, "t_relu3", "QUERY t_relu3 COMPUTE res = RELU_ACTIVATION(xval) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "RELU and RELU_ACTIVATION should agree");
}

// ── LEAKY_RELU ───────────────────────────────────────────────────────────────

#[test]
fn test_leaky_relu_negative_default_alpha() {
    // LEAKY_RELU(-2.0) with default alpha=0.01 -> -0.02
    let (_dir, _db, ex) = make_db("t_lrelu1", serde_json::json!({"xval": -2.0}));
    let f = get_float(&ex, "t_lrelu1", "QUERY t_lrelu1 COMPUTE res = LEAKY_RELU(xval) SELECT res;", "res");
    assert!((f - (-0.02)).abs() < 1e-9, "LEAKY_RELU(-2.0) should be -0.02, got {}", f);
}

#[test]
fn test_lrelu_alias() {
    // LRELU is an alias for LEAKY_RELU
    let (_dir, _db, ex) = make_db("t_lrelu2", serde_json::json!({"xval": -1.0}));
    let a = get_float(&ex, "t_lrelu2", "QUERY t_lrelu2 COMPUTE res = LEAKY_RELU(xval) SELECT res;", "res");
    let b = get_float(&ex, "t_lrelu2", "QUERY t_lrelu2 COMPUTE res = LRELU(xval) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "LEAKY_RELU and LRELU should agree");
}

// ── ELU ──────────────────────────────────────────────────────────────────────

#[test]
fn test_elu_negative_input() {
    // ELU(-1.0) with default alpha=1.0 -> exp(-1) - 1 ≈ -0.6321
    let (_dir, _db, ex) = make_db("t_elu1", serde_json::json!({"xval": -1.0}));
    let f = get_float(&ex, "t_elu1", "QUERY t_elu1 COMPUTE res = ELU(xval) SELECT res;", "res");
    let expected = (-1.0_f64).exp() - 1.0;
    assert!((f - expected).abs() < 1e-9, "ELU(-1.0) should be {}, got {}", expected, f);
}

#[test]
fn test_elu_activation_alias() {
    // ELU_ACTIVATION is an alias for ELU
    let (_dir, _db, ex) = make_db("t_elu2", serde_json::json!({"xval": 2.0}));
    let a = get_float(&ex, "t_elu2", "QUERY t_elu2 COMPUTE res = ELU(xval) SELECT res;", "res");
    let b = get_float(&ex, "t_elu2", "QUERY t_elu2 COMPUTE res = ELU_ACTIVATION(xval) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "ELU and ELU_ACTIVATION should agree");
}

// ── GELU ──────────────────────────────────────────────────────────────────────

#[test]
fn test_gelu_at_zero() {
    // GELU(0) should be 0
    let (_dir, _db, ex) = make_db("t_gelu1", serde_json::json!({"xval": 0.0}));
    let f = get_float(&ex, "t_gelu1", "QUERY t_gelu1 COMPUTE res = GELU(xval) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "GELU(0) should be 0, got {}", f);
}

#[test]
fn test_gelu_activation_alias() {
    // GELU_ACTIVATION is an alias for GELU
    let (_dir, _db, ex) = make_db("t_gelu2", serde_json::json!({"xval": 1.0}));
    let a = get_float(&ex, "t_gelu2", "QUERY t_gelu2 COMPUTE res = GELU(xval) SELECT res;", "res");
    let b = get_float(&ex, "t_gelu2", "QUERY t_gelu2 COMPUTE res = GELU_ACTIVATION(xval) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "GELU and GELU_ACTIVATION should agree");
}

// ── SWISH ──────────────────────────────────────────────────────────────────────

#[test]
fn test_swish_at_zero() {
    // SWISH(0) = 0 * sigmoid(0) = 0
    let (_dir, _db, ex) = make_db("t_swish1", serde_json::json!({"xval": 0.0}));
    let f = get_float(&ex, "t_swish1", "QUERY t_swish1 COMPUTE res = SWISH(xval) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "SWISH(0) should be 0, got {}", f);
}

#[test]
fn test_swish_activation_alias() {
    // SWISH_ACTIVATION is an alias for SWISH
    let (_dir, _db, ex) = make_db("t_swish2", serde_json::json!({"xval": 2.0}));
    let a = get_float(&ex, "t_swish2", "QUERY t_swish2 COMPUTE res = SWISH(xval) SELECT res;", "res");
    let b = get_float(&ex, "t_swish2", "QUERY t_swish2 COMPUTE res = SWISH_ACTIVATION(xval) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "SWISH and SWISH_ACTIVATION should agree");
}

// ── LOG_SOFTMAX ───────────────────────────────────────────────────────────────

#[test]
fn test_log_softmax_exp_sums_to_one() {
    // exp(log_softmax(x)) should sum to 1.0
    let (_dir, _db, ex) = make_db("t_lsm1", serde_json::json!({"arr": [1.0, 2.0]}));
    let arr = get_array(&ex, "t_lsm1", "QUERY t_lsm1 COMPUTE res = LOG_SOFTMAX(arr) SELECT res;", "res");
    assert_eq!(arr.len(), 2);
    let sum_exp: f64 = arr.iter().map(|v| val_to_f64(v).exp()).sum();
    assert!((sum_exp - 1.0).abs() < 1e-9, "exp(log_softmax) should sum to 1.0, got {}", sum_exp);
}

#[test]
fn test_lse_alias() {
    // LSE is an alias for LOG_SOFTMAX
    let (_dir, _db, ex) = make_db("t_lsm2", serde_json::json!({"arr": [0.0, 1.0, 2.0]}));
    let a = get_array(&ex, "t_lsm2", "QUERY t_lsm2 COMPUTE res = LOG_SOFTMAX(arr) SELECT res;", "res");
    let b = get_array(&ex, "t_lsm2", "QUERY t_lsm2 COMPUTE res = LSE(arr) SELECT res;", "res");
    assert_eq!(a.len(), b.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        assert!((val_to_f64(av) - val_to_f64(bv)).abs() < 1e-10, "LOG_SOFTMAX and LSE should agree");
    }
}

// ── MSE_LOSS ──────────────────────────────────────────────────────────────────

#[test]
fn test_mse_loss_known_value() {
    // MSE_LOSS([2,3], [1,3]) = ((2-1)^2 + (3-3)^2) / 2 = 0.5
    let (_dir, _db, ex) = make_db("t_mse1", serde_json::json!({"pred": [2.0, 3.0], "tru": [1.0, 3.0]}));
    let f = get_float(&ex, "t_mse1", "QUERY t_mse1 COMPUTE res = MSE_LOSS(pred, tru) SELECT res;", "res");
    assert!((f - 0.5).abs() < 1e-9, "MSE_LOSS([2,3],[1,3]) should be 0.5, got {}", f);
}

#[test]
fn test_mean_squared_error_alias() {
    // MEAN_SQUARED_ERROR is an alias for MSE / MSE_LOSS
    let (_dir, _db, ex) = make_db("t_mse2", serde_json::json!({"pred": [0.0, 0.0], "tru": [1.0, 1.0]}));
    let a = get_float(&ex, "t_mse2", "QUERY t_mse2 COMPUTE res = MSE_LOSS(pred, tru) SELECT res;", "res");
    let b = get_float(&ex, "t_mse2", "QUERY t_mse2 COMPUTE res = MEAN_SQUARED_ERROR(pred, tru) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "MSE_LOSS and MEAN_SQUARED_ERROR should agree");
}

// ── MAE_LOSS ──────────────────────────────────────────────────────────────────

#[test]
fn test_mae_loss_known_value() {
    // MAE_LOSS([2,3], [1,3]) = (|2-1| + |3-3|) / 2 = 0.5
    let (_dir, _db, ex) = make_db("t_mae1", serde_json::json!({"pred": [2.0, 3.0], "tru": [1.0, 3.0]}));
    let f = get_float(&ex, "t_mae1", "QUERY t_mae1 COMPUTE res = MAE_LOSS(pred, tru) SELECT res;", "res");
    assert!((f - 0.5).abs() < 1e-9, "MAE_LOSS([2,3],[1,3]) should be 0.5, got {}", f);
}

#[test]
fn test_mean_absolute_error_alias() {
    // MEAN_ABSOLUTE_ERROR is an alias for MAE_SCORE / MAE_LOSS
    let (_dir, _db, ex) = make_db("t_mae2", serde_json::json!({"pred": [1.0, 3.0], "tru": [0.0, 0.0]}));
    let a = get_float(&ex, "t_mae2", "QUERY t_mae2 COMPUTE res = MAE_LOSS(pred, tru) SELECT res;", "res");
    let b = get_float(&ex, "t_mae2", "QUERY t_mae2 COMPUTE res = MEAN_ABSOLUTE_ERROR(pred, tru) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "MAE_LOSS and MEAN_ABSOLUTE_ERROR should agree");
}

// ── HUBER_LOSS ────────────────────────────────────────────────────────────────

#[test]
fn test_huber_loss_known_value() {
    // HUBER_LOSS(1.5, 0.0, 1.0): diff=1.5 > delta=1.0 -> 1.0*(1.5-0.5*1.0)=1.0
    let (_dir, _db, ex) = make_db("t_hl1", serde_json::json!({"yt": 1.5, "yp": 0.0, "delta": 1.0}));
    let f = get_float(&ex, "t_hl1", "QUERY t_hl1 COMPUTE res = HUBER_LOSS(yt, yp, delta) SELECT res;", "res");
    assert!((f - 1.0).abs() < 1e-9, "HUBER_LOSS(1.5, 0.0, 1.0) should be 1.0, got {}", f);
}

#[test]
fn test_smooth_l1_alias() {
    // SMOOTH_L1 is an alias for HUBER_LOSS
    let (_dir, _db, ex) = make_db("t_hl2", serde_json::json!({"yt": 0.5, "yp": 0.0, "delta": 1.0}));
    let a = get_float(&ex, "t_hl2", "QUERY t_hl2 COMPUTE res = HUBER_LOSS(yt, yp, delta) SELECT res;", "res");
    let b = get_float(&ex, "t_hl2", "QUERY t_hl2 COMPUTE res = SMOOTH_L1(yt, yp, delta) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "HUBER_LOSS and SMOOTH_L1 should agree");
}

// ── BINARY_CROSS_ENTROPY ───────────────────────────────────────────────────────

#[test]
fn test_binary_cross_entropy_small_positive() {
    // BCE([0.9], [1.0]) -> small positive number
    // Note: existing BCE takes (y_true, y_pred) as scalar, but we test the existing behavior
    // BCE(1.0, 0.9) = -(1.0 * log(0.9) + 0.0 * log(0.1)) ≈ 0.1054
    let (_dir, _db, ex) = make_db("t_bce1", serde_json::json!({"yt": 1.0, "yp": 0.9}));
    let f = get_float(&ex, "t_bce1", "QUERY t_bce1 COMPUTE res = BINARY_CROSS_ENTROPY(yt, yp) SELECT res;", "res");
    assert!(f > 0.0 && f < 1.0, "BCE(1.0, 0.9) should be a small positive number, got {}", f);
}

#[test]
fn test_bce_loss_alias() {
    // BCE_LOSS is an alias for BINARY_CROSS_ENTROPY
    let (_dir, _db, ex) = make_db("t_bce2", serde_json::json!({"yt": 0.0, "yp": 0.1}));
    let a = get_float(&ex, "t_bce2", "QUERY t_bce2 COMPUTE res = BINARY_CROSS_ENTROPY(yt, yp) SELECT res;", "res");
    let b = get_float(&ex, "t_bce2", "QUERY t_bce2 COMPUTE res = BCE_LOSS(yt, yp) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "BINARY_CROSS_ENTROPY and BCE_LOSS should agree");
}

// ── L1_REGULARIZE ─────────────────────────────────────────────────────────────

#[test]
fn test_l1_regularize_known_value() {
    // L1_REGULARIZE([1.0, -2.0, 3.0], 0.1) -> 0.1 * (1+2+3) = 0.6
    let (_dir, _db, ex) = make_db("t_l1r1", serde_json::json!({"wts": [1.0, -2.0, 3.0], "lam": 0.1}));
    let f = get_float(&ex, "t_l1r1", "QUERY t_l1r1 COMPUTE res = L1_REGULARIZE(wts, lam) SELECT res;", "res");
    assert!((f - 0.6).abs() < 1e-9, "L1_REGULARIZE([1,-2,3], 0.1) should be 0.6, got {}", f);
}

#[test]
fn test_l1_penalty_alias() {
    // L1_PENALTY is an alias for L1_REGULARIZE
    let (_dir, _db, ex) = make_db("t_l1r2", serde_json::json!({"wts": [2.0, 3.0], "lam": 1.0}));
    let a = get_float(&ex, "t_l1r2", "QUERY t_l1r2 COMPUTE res = L1_REGULARIZE(wts, lam) SELECT res;", "res");
    let b = get_float(&ex, "t_l1r2", "QUERY t_l1r2 COMPUTE res = L1_PENALTY(wts, lam) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "L1_REGULARIZE and L1_PENALTY should agree");
}

// ── L2_REGULARIZE ─────────────────────────────────────────────────────────────

#[test]
fn test_l2_regularize_known_value() {
    // L2_REGULARIZE([1.0, 2.0], 1.0) -> 1.0 * (1^2 + 2^2) / 2 = 2.5
    let (_dir, _db, ex) = make_db("t_l2r1", serde_json::json!({"wts": [1.0, 2.0], "lam": 1.0}));
    let f = get_float(&ex, "t_l2r1", "QUERY t_l2r1 COMPUTE res = L2_REGULARIZE(wts, lam) SELECT res;", "res");
    assert!((f - 2.5).abs() < 1e-9, "L2_REGULARIZE([1,2], 1.0) should be 2.5, got {}", f);
}

#[test]
fn test_l2_penalty_alias() {
    // L2_PENALTY is an alias for L2_REGULARIZE
    let (_dir, _db, ex) = make_db("t_l2r2", serde_json::json!({"wts": [3.0, 4.0], "lam": 0.5}));
    let a = get_float(&ex, "t_l2r2", "QUERY t_l2r2 COMPUTE res = L2_REGULARIZE(wts, lam) SELECT res;", "res");
    let b = get_float(&ex, "t_l2r2", "QUERY t_l2r2 COMPUTE res = L2_PENALTY(wts, lam) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "L2_REGULARIZE and L2_PENALTY should agree");
}

// ── CLIP_GRADIENTS ───────────────────────────────────────────────────────────

#[test]
fn test_clip_gradients_clips_correctly() {
    // CLIP_GRADIENTS([3.0, 4.0], 1.0) -> norm=5, scale=0.2 -> [0.6, 0.8]
    let (_dir, _db, ex) = make_db("t_cg1", serde_json::json!({"grads": [3.0, 4.0], "mn": 1.0}));
    let arr = get_array(&ex, "t_cg1", "QUERY t_cg1 COMPUTE res = CLIP_GRADIENTS(grads, mn) SELECT res;", "res");
    assert_eq!(arr.len(), 2);
    let g0 = val_to_f64(&arr[0]);
    let g1 = val_to_f64(&arr[1]);
    assert!((g0 - 0.6).abs() < 1e-9, "clipped gradient[0] should be 0.6, got {}", g0);
    assert!((g1 - 0.8).abs() < 1e-9, "clipped gradient[1] should be 0.8, got {}", g1);
}

#[test]
fn test_gradient_clip_alias_noop() {
    // GRADIENT_CLIP is an alias for CLIP_GRADIENTS; no-op when norm < max_norm
    let (_dir, _db, ex) = make_db("t_cg2", serde_json::json!({"grads": [0.1, 0.2], "mn": 10.0}));
    let arr = get_array(&ex, "t_cg2", "QUERY t_cg2 COMPUTE res = GRADIENT_CLIP(grads, mn) SELECT res;", "res");
    assert_eq!(arr.len(), 2);
    // norm ≈ 0.224 < 10.0, so no clipping: values should be unchanged
    let g0 = val_to_f64(&arr[0]);
    let g1 = val_to_f64(&arr[1]);
    assert!((g0 - 0.1).abs() < 1e-9, "gradient[0] should be unchanged at 0.1, got {}", g0);
    assert!((g1 - 0.2).abs() < 1e-9, "gradient[1] should be unchanged at 0.2, got {}", g1);
}

// ── ACCURACY_SCORE ────────────────────────────────────────────────────────────

#[test]
fn test_accuracy_score_known_value() {
    // ACCURACY_SCORE([1,0,1], [1,0,0]) = 2/3
    let (_dir, _db, ex) = make_db("t_acc1", serde_json::json!({"pred": [1.0, 0.0, 1.0], "tru": [1.0, 0.0, 0.0]}));
    let f = get_float(&ex, "t_acc1", "QUERY t_acc1 COMPUTE res = ACCURACY_SCORE(pred, tru) SELECT res;", "res");
    let expected = 2.0 / 3.0;
    assert!((f - expected).abs() < 1e-9, "ACCURACY_SCORE should be 2/3={}, got {}", expected, f);
}

#[test]
fn test_classification_accuracy_alias() {
    // CLASSIFICATION_ACCURACY is an alias for ACCURACY_SCORE / BATCH_ACCURACY
    let (_dir, _db, ex) = make_db("t_acc2", serde_json::json!({"pred": [1.0, 1.0, 0.0], "tru": [1.0, 0.0, 0.0]}));
    let a = get_float(&ex, "t_acc2", "QUERY t_acc2 COMPUTE res = ACCURACY_SCORE(pred, tru) SELECT res;", "res");
    let b = get_float(&ex, "t_acc2", "QUERY t_acc2 COMPUTE res = CLASSIFICATION_ACCURACY(pred, tru) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "ACCURACY_SCORE and CLASSIFICATION_ACCURACY should agree");
}

// ── F1_SCORE ──────────────────────────────────────────────────────────────────

#[test]
fn test_f1_score_known_value() {
    // F1_SCORE([1,1,0], [1,0,1]): TP=1, FP=1, FN=1 -> F1 = 2/(2+1+1) = 0.5
    let (_dir, _db, ex) = make_db("t_f1_1", serde_json::json!({"pred": [1.0, 1.0, 0.0], "tru": [1.0, 0.0, 1.0]}));
    let f = get_float(&ex, "t_f1_1", "QUERY t_f1_1 COMPUTE res = F1_SCORE(pred, tru) SELECT res;", "res");
    assert!((f - 0.5).abs() < 1e-9, "F1_SCORE should be 0.5, got {}", f);
}

#[test]
fn test_f1_binary_alias() {
    // F1_BINARY is an alias for F1_SCORE
    let (_dir, _db, ex) = make_db("t_f1_2", serde_json::json!({"pred": [1.0, 0.0], "tru": [1.0, 0.0]}));
    let a = get_float(&ex, "t_f1_2", "QUERY t_f1_2 COMPUTE res = F1_SCORE(pred, tru) SELECT res;", "res");
    let b = get_float(&ex, "t_f1_2", "QUERY t_f1_2 COMPUTE res = F1_BINARY(pred, tru) SELECT res;", "res");
    assert!((a - b).abs() < 1e-12, "F1_SCORE and F1_BINARY should agree");
}

#[test]
fn test_f1_score_perfect_prediction() {
    // F1 = 1.0 when all predictions are correct
    let (_dir, _db, ex) = make_db("t_f1_3", serde_json::json!({"pred": [1.0, 1.0, 0.0, 0.0], "tru": [1.0, 1.0, 0.0, 0.0]}));
    let f = get_float(&ex, "t_f1_3", "QUERY t_f1_3 COMPUTE res = F1_SCORE(pred, tru) SELECT res;", "res");
    assert!((f - 1.0).abs() < 1e-9, "F1_SCORE with perfect predictions should be 1.0, got {}", f);
}

#[test]
fn test_f1_score_all_wrong() {
    // F1 = 0.0 when all positive predictions are wrong and none are caught
    // pred=[1,1], tru=[0,0] -> TP=0, FP=2, FN=0 -> F1 = 0/(0+2+0) = 0
    let (_dir, _db, ex) = make_db("t_f1_4", serde_json::json!({"pred": [1.0, 1.0], "tru": [0.0, 0.0]}));
    let f = get_float(&ex, "t_f1_4", "QUERY t_f1_4 COMPUTE res = F1_SCORE(pred, tru) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "F1_SCORE with all FP should be 0.0, got {}", f);
}

// ── Additional edge case / coverage tests ─────────────────────────────────────

#[test]
fn test_one_hot_last_class() {
    // ONE_HOT_ENCODE(3, 4) -> [0, 0, 0, 1]
    let (_dir, _db, ex) = make_db("t_ohe4", serde_json::json!({"idx": 3, "nc": 4}));
    let arr = get_array(&ex, "t_ohe4", "QUERY t_ohe4 COMPUTE res = ONE_HOT_ENCODE(idx, nc) SELECT res;", "res");
    assert_eq!(arr.len(), 4);
    assert_eq!(val_to_f64(&arr[3]) as i64, 1);
    assert_eq!(val_to_f64(&arr[0]) as i64, 0);
}

#[test]
fn test_l1_regularize_zero_weights() {
    // L1_REGULARIZE([0,0,0], 5.0) -> 0.0
    let (_dir, _db, ex) = make_db("t_l1r3", serde_json::json!({"wts": [0.0, 0.0, 0.0], "lam": 5.0}));
    let f = get_float(&ex, "t_l1r3", "QUERY t_l1r3 COMPUTE res = L1_REGULARIZE(wts, lam) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "L1 of zero weights should be 0, got {}", f);
}

#[test]
fn test_l2_regularize_zero_weights() {
    // L2_REGULARIZE([0,0], 3.0) -> 0.0
    let (_dir, _db, ex) = make_db("t_l2r3", serde_json::json!({"wts": [0.0, 0.0], "lam": 3.0}));
    let f = get_float(&ex, "t_l2r3", "QUERY t_l2r3 COMPUTE res = L2_REGULARIZE(wts, lam) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "L2 of zero weights should be 0, got {}", f);
}

#[test]
fn test_clip_gradients_no_clip_needed() {
    // CLIP_GRADIENTS([0.3, 0.4], 10.0) -> norm=0.5 < 10.0, no change
    let (_dir, _db, ex) = make_db("t_cg3", serde_json::json!({"grads": [0.3, 0.4], "mn": 10.0}));
    let arr = get_array(&ex, "t_cg3", "QUERY t_cg3 COMPUTE res = CLIP_GRADIENTS(grads, mn) SELECT res;", "res");
    assert_eq!(arr.len(), 2);
    assert!((val_to_f64(&arr[0]) - 0.3).abs() < 1e-9);
    assert!((val_to_f64(&arr[1]) - 0.4).abs() < 1e-9);
}

#[test]
fn test_log_softmax_length_preserved() {
    // LOG_SOFTMAX output length should equal input length
    let (_dir, _db, ex) = make_db("t_lsm3", serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0]}));
    let arr = get_array(&ex, "t_lsm3", "QUERY t_lsm3 COMPUTE res = LOG_SOFTMAX(arr) SELECT res;", "res");
    assert_eq!(arr.len(), 4, "LOG_SOFTMAX should preserve array length");
}

#[test]
fn test_accuracy_score_all_correct() {
    // ACCURACY_SCORE when all correct -> 1.0
    let (_dir, _db, ex) = make_db("t_acc3", serde_json::json!({"pred": [1.0, 0.0, 1.0], "tru": [1.0, 0.0, 1.0]}));
    let f = get_float(&ex, "t_acc3", "QUERY t_acc3 COMPUTE res = ACCURACY_SCORE(pred, tru) SELECT res;", "res");
    assert!((f - 1.0).abs() < 1e-9, "accuracy of perfect predictions should be 1.0, got {}", f);
}

#[test]
fn test_mse_loss_zero_error() {
    // MSE_LOSS([5,5], [5,5]) = 0.0
    let (_dir, _db, ex) = make_db("t_mse3", serde_json::json!({"pred": [5.0, 5.0], "tru": [5.0, 5.0]}));
    let f = get_float(&ex, "t_mse3", "QUERY t_mse3 COMPUTE res = MSE_LOSS(pred, tru) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "MSE_LOSS of identical arrays should be 0, got {}", f);
}

#[test]
fn test_mae_loss_zero_error() {
    // MAE_LOSS([3,4], [3,4]) = 0.0
    let (_dir, _db, ex) = make_db("t_mae3", serde_json::json!({"pred": [3.0, 4.0], "tru": [3.0, 4.0]}));
    let f = get_float(&ex, "t_mae3", "QUERY t_mae3 COMPUTE res = MAE_LOSS(pred, tru) SELECT res;", "res");
    assert!(f.abs() < 1e-9, "MAE_LOSS of identical arrays should be 0, got {}", f);
}
