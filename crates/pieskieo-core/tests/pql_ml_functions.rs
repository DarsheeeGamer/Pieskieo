/// Integration tests for PQL ML activation and loss functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
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

#[test]
fn test_sigmoid_at_zero() {
    // sigmoid(0) = 0.5
    let (_dir, _db, ex) = make_db("t", serde_json::json!({"x": 0.0}));
    let mut p = Parser::new(r#"QUERY t COMPUTE s = SIGMOID(x) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.5).abs() < 0.001,
            "sigmoid(0) should be 0.5, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_sigmoid_logistic_alias() {
    // LOGISTIC is an alias for SIGMOID
    let (_dir, _db, ex) = make_db("t2", serde_json::json!({"x": 2.0}));
    let mut p =
        Parser::new(r#"QUERY t2 COMPUTE a = SIGMOID(x) COMPUTE b = LOGISTIC(x) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for a, got {:?}", other),
    };
    let b = match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for b, got {:?}", other),
    };
    assert!(
        (a - b).abs() < 1e-10,
        "SIGMOID and LOGISTIC should produce same result, got {} vs {}",
        a,
        b
    );
}

#[test]
fn test_relu_positive_and_negative() {
    // RELU(5) = 5, RELU(-3) = 0
    let (_dir, _db, ex) = make_db("t3", serde_json::json!({"pos": 5.0, "neg": -3.0}));
    let mut p =
        Parser::new(r#"QUERY t3 COMPUTE rp = RELU(pos) COMPUTE rn = RELU(neg) SELECT rp, rn;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("rp") {
        Some(Value::Float(f)) => {
            assert!((*f - 5.0).abs() < 1e-9, "RELU(5) should be 5.0, got {}", f)
        }
        other => panic!("expected Float for rp, got {:?}", other),
    }
    match r.rows[0].data.get("rn") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-9, "RELU(-3) should be 0.0, got {}", f),
        other => panic!("expected Float for rn, got {:?}", other),
    }
}

#[test]
fn test_leaky_relu_negative_input() {
    // LEAKY_RELU(-2, 0.1) = 0.1 * (-2) = -0.2
    let (_dir, _db, ex) = make_db("t4", serde_json::json!({"x": -2.0}));
    let mut p = Parser::new(r#"QUERY t4 COMPUTE r = LEAKY_RELU(x, 0.1) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - (-0.2)).abs() < 1e-9,
            "LEAKY_RELU(-2, 0.1) should be -0.2, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_softmax_array_sums_to_one() {
    // softmax of any array should sum to 1.0
    let (_dir, _db, ex) = make_db("t5", serde_json::json!({"arr": [1.0, 2.0, 3.0]}));
    let mut p = Parser::new(r#"QUERY t5 COMPUTE sm = SOFTMAX_ARRAY(arr) SELECT sm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("sm") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3, "softmax output should have 3 elements");
            let sum: f64 = arr
                .iter()
                .map(|v| match v {
                    Value::Float(f) => *f,
                    _ => 0.0,
                })
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "softmax should sum to 1.0, got {}",
                sum
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_mse_perfect_prediction() {
    // MSE when predictions equal truth should be 0
    let (_dir, _db, ex) = make_db(
        "t6",
        serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}),
    );
    let mut p = Parser::new(r#"QUERY t6 COMPUTE m = MSE(a, b) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "MSE of identical arrays should be 0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_mse_known_value() {
    // MSE([0,0,0], [1,1,1]) = 1.0
    let (_dir, _db, ex) = make_db(
        "t7",
        serde_json::json!({"a": [0.0, 0.0, 0.0], "b": [1.0, 1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY t7 COMPUTE m = MEAN_SQUARED_ERROR(a, b) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "MSE([0,0,0],[1,1,1]) should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_rmse_known_value() {
    // RMSE([0,0,0,0], [2,2,2,2]) = sqrt(4) = 2.0
    let (_dir, _db, ex) = make_db(
        "t8",
        serde_json::json!({"a": [0.0, 0.0, 0.0, 0.0], "b": [2.0, 2.0, 2.0, 2.0]}),
    );
    let mut p = Parser::new(r#"QUERY t8 COMPUTE r = RMSE(a, b) SELECT r;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 1e-9, "RMSE should be 2.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_r_squared_perfect_fit() {
    // R^2 = 1 when predictions are perfect
    let (_dir, _db, ex) = make_db(
        "t9",
        serde_json::json!({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]}),
    );
    let mut p = Parser::new(r#"QUERY t9 COMPUTE r2 = R_SQUARED(a, b) SELECT r2;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("r2") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "R^2 for perfect prediction should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_logit_inverse_sigmoid() {
    // logit(sigmoid(x)) should equal x (approximately, for values away from 0/1)
    // logit(0.5) = log(0.5/0.5) = log(1) = 0
    let (_dir, _db, ex) = make_db("t10", serde_json::json!({"p": 0.5}));
    let mut p = Parser::new(r#"QUERY t10 COMPUTE l = LOGIT(p) SELECT l;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("l") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-9, "LOGIT(0.5) should be 0.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_feature_scale_midpoint() {
    // FEATURE_SCALE(5, 0, 10) = 0.5
    let (_dir, _db, ex) = make_db("t11", serde_json::json!({"x": 5.0}));
    let mut p = Parser::new(r#"QUERY t11 COMPUTE s = FEATURE_SCALE(x, 0, 10) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.5).abs() < 1e-9,
            "FEATURE_SCALE(5,0,10) should be 0.5, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_feature_scale_boundaries() {
    // FEATURE_SCALE(min) = 0, FEATURE_SCALE(max) = 1
    let (_dir, _db, ex) = make_db("t12", serde_json::json!({"lo": 0.0, "hi": 10.0}));
    let mut p = Parser::new(
        r#"QUERY t12 COMPUTE a = FEATURE_SCALE(lo, 0, 10) COMPUTE b = FEATURE_SCALE(hi, 0, 10) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("a") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-9, "scale at min should be 0, got {}", f),
        other => panic!("expected Float for a, got {:?}", other),
    }
    match r.rows[0].data.get("b") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "scale at max should be 1, got {}",
            f
        ),
        other => panic!("expected Float for b, got {:?}", other),
    }
}

#[test]
fn test_hinge_loss_correct_classification() {
    // HINGE_LOSS(1.0, 2.0) = max(0, 1 - 1*2) = max(0, -1) = 0
    let (_dir, _db, ex) = make_db("t13", serde_json::json!({"y": 1.0, "yhat": 2.0}));
    let mut p = Parser::new(r#"QUERY t13 COMPUTE h = HINGE_LOSS(y, yhat) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => {
            assert!(f.abs() < 1e-9, "HINGE_LOSS(1,2) should be 0.0, got {}", f)
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_hinge_loss_misclassification() {
    // HINGE_LOSS(1.0, -0.5) = max(0, 1 - 1*(-0.5)) = max(0, 1.5) = 1.5
    let (_dir, _db, ex) = make_db("t14", serde_json::json!({"y": 1.0, "yhat": -0.5}));
    let mut p = Parser::new(r#"QUERY t14 COMPUTE h = HINGE_LOSS(y, yhat) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.5).abs() < 1e-9,
            "HINGE_LOSS(1,-0.5) should be 1.5, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_batch_accuracy_perfect() {
    // 100% accuracy when arrays are identical
    let (_dir, _db, ex) = make_db(
        "t15",
        serde_json::json!({"a": [1.0, 0.0, 1.0], "b": [1.0, 0.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY t15 COMPUTE acc = BATCH_ACCURACY(a, b) SELECT acc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("acc") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "perfect accuracy should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_batch_accuracy_partial() {
    // 2 out of 4 correct = 0.5
    let (_dir, _db, ex) = make_db(
        "t16",
        serde_json::json!({"a": [1.0, 1.0, 0.0, 0.0], "b": [1.0, 0.0, 1.0, 0.0]}),
    );
    let mut p = Parser::new(r#"QUERY t16 COMPUTE acc = ACCURACY_SCORE(a, b) SELECT acc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("acc") {
        Some(Value::Float(f)) => {
            assert!((*f - 0.5).abs() < 1e-9, "accuracy should be 0.5, got {}", f)
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_gelu_at_zero() {
    // GELU(0) = 0 * 0.5 * (1 + erf(0)) = 0
    let (_dir, _db, ex) = make_db("t17", serde_json::json!({"x": 0.0}));
    let mut p = Parser::new(r#"QUERY t17 COMPUTE g = GELU(x) SELECT g;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("g") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-9, "GELU(0) should be 0.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_swish_at_zero() {
    // SWISH(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let (_dir, _db, ex) = make_db("t18", serde_json::json!({"x": 0.0}));
    let mut p = Parser::new(r#"QUERY t18 COMPUTE s = SWISH(x) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(f.abs() < 1e-9, "SWISH(0) should be 0.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_elu_positive_input() {
    // ELU(3.0) = 3.0 (positive inputs pass through)
    let (_dir, _db, ex) = make_db("t19", serde_json::json!({"x": 3.0}));
    let mut p = Parser::new(r#"QUERY t19 COMPUTE e = ELU(x) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("e") {
        Some(Value::Float(f)) => {
            assert!((*f - 3.0).abs() < 1e-9, "ELU(3.0) should be 3.0, got {}", f)
        }
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_mae_score_known_value() {
    // MAE([0,0,0], [1,2,3]) = (1+2+3)/3 = 2.0
    let (_dir, _db, ex) = make_db(
        "t20",
        serde_json::json!({"a": [0.0, 0.0, 0.0], "b": [1.0, 2.0, 3.0]}),
    );
    let mut p = Parser::new(r#"QUERY t20 COMPUTE m = MAE_SCORE(a, b) SELECT m;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("m") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 1e-9, "MAE should be 2.0, got {}", f),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_huber_loss_small_error() {
    // |y - yhat| = 0.5 <= delta=1: Huber = 0.5 * 0.5^2 = 0.125
    let (_dir, _db, ex) = make_db("t21", serde_json::json!({"y": 1.0, "yhat": 1.5}));
    let mut p = Parser::new(r#"QUERY t21 COMPUTE h = HUBER_LOSS(y, yhat, 1.0) SELECT h;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("h") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.125).abs() < 1e-9,
            "HUBER_LOSS small error should be 0.125, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_cross_entropy_one_hot() {
    // CROSS_ENTROPY([0,1,0], [0.1,0.8,0.1]) = -(0*ln(0.1+eps) + 1*ln(0.8+eps) + 0*ln(0.1+eps))
    //                                        = -ln(0.8+eps) ≈ 0.2231
    let (_dir, _db, ex) = make_db(
        "t22",
        serde_json::json!({"a": [0.0, 1.0, 0.0], "b": [0.1, 0.8, 0.1]}),
    );
    let mut p = Parser::new(r#"QUERY t22 COMPUTE ce = CROSS_ENTROPY(a, b) SELECT ce;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ce") {
        Some(Value::Float(f)) => assert!(
            (*f - 0.2231_f64).abs() < 0.01,
            "CROSS_ENTROPY one-hot should be ~0.2231, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_softmax_score_index() {
    // For uniform input [0,0,0], each softmax element = 1/3 ≈ 0.3333
    let (_dir, _db, ex) = make_db("t23", serde_json::json!({"arr": [0.0, 0.0, 0.0]}));
    let mut p = Parser::new(r#"QUERY t23 COMPUTE s = SOFTMAX_SCORE(arr, 0) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0 / 3.0).abs() < 1e-6,
            "SOFTMAX_SCORE on uniform input should be 1/3, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_binary_cross_entropy_perfect_prediction() {
    // BCE(1, 1) = -(1*ln(1+eps) + 0*ln(eps)) ≈ 0 (very close to 0)
    let (_dir, _db, ex) = make_db("t24", serde_json::json!({"y": 1.0, "yhat": 1.0}));
    let mut p = Parser::new(r#"QUERY t24 COMPUTE bce = BINARY_CROSS_ENTROPY(y, yhat) SELECT bce;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("bce") {
        Some(Value::Float(f)) => assert!(
            *f < 1e-4,
            "BCE for perfect prediction should be near 0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}
