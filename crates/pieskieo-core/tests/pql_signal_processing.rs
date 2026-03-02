/// Integration tests for PQL signal processing built-in functions.
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

// ── SIGNAL_POWER ─────────────────────────────────────────────────────────────

#[test]
fn test_signal_power_uniform() {
    // SIGNAL_POWER([2,2,2,2]) = sqrt(mean([4,4,4,4])) = sqrt(4) = 2.0
    let (_dir, _db, ex) = make_db(
        "sp_uniform",
        serde_json::json!({"arr": [2.0, 2.0, 2.0, 2.0]}),
    );
    let mut p = Parser::new(r#"QUERY sp_uniform COMPUTE r = SIGNAL_POWER(arr) SELECT r;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 2.0).abs() < 1e-9,
            "SIGNAL_POWER([2,2,2,2]) should be 2.0 (RMS), got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_array_power_alias() {
    // ARRAY_POWER is an alias for SIGNAL_POWER
    let (_dir, _db, ex) = make_db(
        "ap_alias",
        serde_json::json!({"arr": [1.0, 1.0, 1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY ap_alias COMPUTE r = ARRAY_POWER(arr) SELECT r;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "ARRAY_POWER([1,1,1,1]) should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── RMS_AMPLITUDE ─────────────────────────────────────────────────────────────

#[test]
fn test_rms_amplitude_uniform() {
    // RMS_AMPLITUDE([1,1,1,1]) = sqrt(mean([1,1,1,1])) = sqrt(1) = 1.0
    let (_dir, _db, ex) = make_db(
        "rms_uniform",
        serde_json::json!({"arr": [1.0, 1.0, 1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY rms_uniform COMPUTE r = RMS_AMPLITUDE(arr) SELECT r;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "RMS_AMPLITUDE([1,1,1,1]) should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_rms_value_alias() {
    // RMS_VALUE is an alias for RMS_AMPLITUDE: [3,4] -> sqrt((9+16)/2) = sqrt(12.5)
    let (_dir, _db, ex) = make_db(
        "rms_val_alias",
        serde_json::json!({"arr": [3.0, 4.0]}),
    );
    let mut p = Parser::new(r#"QUERY rms_val_alias COMPUTE r = RMS_VALUE(arr) SELECT r;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let expected = 12.5_f64.sqrt();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - expected).abs() < 1e-9,
            "RMS_VALUE([3,4]) should be {}, got {}",
            expected,
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── ZERO_CROSSING_RATE ───────────────────────────────────────────────────────

#[test]
fn test_zero_crossing_rate_alternating() {
    // [1,-1,1,-1]: every adjacent pair crosses sign => ZCR = 3/3 = 1.0
    let (_dir, _db, ex) = make_db(
        "zcr_alt",
        serde_json::json!({"arr": [1.0, -1.0, 1.0, -1.0]}),
    );
    let mut p = Parser::new(r#"QUERY zcr_alt COMPUTE z = ZERO_CROSSING_RATE(arr) SELECT z;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("z") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "ZERO_CROSSING_RATE([1,-1,1,-1]) should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_zcr_alias() {
    // ZCR is an alias for ZERO_CROSSING_RATE
    let (_dir, _db, ex) = make_db(
        "zcr_alias",
        serde_json::json!({"arr": [2.0, 2.0, 2.0, 2.0]}),
    );
    let mut p = Parser::new(r#"QUERY zcr_alias COMPUTE z = ZCR(arr) SELECT z;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("z") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "ZCR of constant positive signal should be 0.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── APPLY_HAMMING ─────────────────────────────────────────────────────────────

#[test]
fn test_apply_hamming_length() {
    // APPLY_HAMMING should return same length as input
    let (_dir, _db, ex) = make_db(
        "hamm_len",
        serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0]}),
    );
    let mut p = Parser::new(r#"QUERY hamm_len COMPUTE w = APPLY_HAMMING(arr) SELECT w;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("w") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            4,
            "APPLY_HAMMING output length should equal input length"
        ),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_apply_hamming_endpoints() {
    // Hamming window endpoints are 0.08 (=0.54-0.46), so first and last elements
    // of a 4-element signal [1,1,1,1] should be ~0.08
    let (_dir, _db, ex) = make_db(
        "hamm_ep",
        serde_json::json!({"arr": [1.0, 1.0, 1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY hamm_ep COMPUTE w = APPLY_HAMMING(arr) SELECT w;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("w") {
        Some(Value::Array(a)) => {
            let first = as_f64(&a[0]);
            let last = as_f64(a.last().unwrap());
            assert!(
                (first - 0.08).abs() < 1e-9,
                "APPLY_HAMMING first element of [1,1,1,1] should be ~0.08, got {}",
                first
            );
            assert!(
                (last - 0.08).abs() < 1e-9,
                "APPLY_HAMMING last element of [1,1,1,1] should be ~0.08, got {}",
                last
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── APPLY_HANN ────────────────────────────────────────────────────────────────

#[test]
fn test_apply_hann_length() {
    // APPLY_HANN should return same length as input
    let (_dir, _db, ex) = make_db(
        "hann_len",
        serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0]}),
    );
    let mut p = Parser::new(r#"QUERY hann_len COMPUTE w = APPLY_HANN(arr) SELECT w;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("w") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            4,
            "APPLY_HANN output length should equal input length"
        ),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_apply_hann_zeroes_endpoints() {
    // Hann window: first and last elements should be 0 (w(0)=0, w(N-1)=0 for N>1)
    let (_dir, _db, ex) = make_db(
        "hann_ep",
        serde_json::json!({"arr": [5.0, 10.0, 10.0, 5.0]}),
    );
    let mut p = Parser::new(r#"QUERY hann_ep COMPUTE w = APPLY_HANN(arr) SELECT w;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("w") {
        Some(Value::Array(a)) => {
            let first = as_f64(&a[0]);
            let last = as_f64(a.last().unwrap());
            assert!(
                first.abs() < 1e-9,
                "APPLY_HANN first element should be 0, got {}",
                first
            );
            assert!(
                last.abs() < 1e-9,
                "APPLY_HANN last element should be 0, got {}",
                last
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SIGNAL_CONVOLVE ───────────────────────────────────────────────────────────

#[test]
fn test_signal_convolve_length() {
    // SIGNAL_CONVOLVE([1,0,0], [2,3]) => output length = 3+2-1 = 4
    let (_dir, _db, ex) = make_db(
        "sconv_len",
        serde_json::json!({"sig": [1.0, 0.0, 0.0], "ker": [2.0, 3.0]}),
    );
    let mut p = Parser::new(r#"QUERY sconv_len COMPUTE c = SIGNAL_CONVOLVE(sig, ker) SELECT c;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("c") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            4,
            "SIGNAL_CONVOLVE([1,0,0],[2,3]) output length should be 4, got {}",
            a.len()
        ),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_signal_convolve_values() {
    // SIGNAL_CONVOLVE([1,0,0], [2,3]) => [2, 3, 0, 0]
    let (_dir, _db, ex) = make_db(
        "sconv_vals",
        serde_json::json!({"sig": [1.0, 0.0, 0.0], "ker": [2.0, 3.0]}),
    );
    let mut p = Parser::new(r#"QUERY sconv_vals COMPUTE c = SIGNAL_CONVOLVE(sig, ker) SELECT c;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("c") {
        Some(Value::Array(a)) => {
            let expected = [2.0, 3.0, 0.0, 0.0];
            for (i, (&e, v)) in expected.iter().zip(a.iter()).enumerate() {
                assert!(
                    (as_f64(v) - e).abs() < 1e-9,
                    "SIGNAL_CONVOLVE[{}] expected {}, got {}",
                    i,
                    e,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── CROSS_CORRELATE ───────────────────────────────────────────────────────────

#[test]
fn test_cross_correlate_length() {
    // CROSS_CORRELATE of n=3 and m=3 arrays => output length = 3+3-1 = 5
    let (_dir, _db, ex) = make_db(
        "xcorr_len",
        serde_json::json!({"s1": [1.0, 2.0, 3.0], "s2": [1.0, 0.0, -1.0]}),
    );
    let mut p = Parser::new(r#"QUERY xcorr_len COMPUTE c = CROSS_CORRELATE(s1, s2) SELECT c;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("c") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            5,
            "CROSS_CORRELATE of n=3,m=3 should have length 5, got {}",
            a.len()
        ),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_xcorr_is_cross_correlate() {
    // Verify CROSS_CORRELATE and XCORR (via CROSS_CORRELATION alias) exist and return arrays
    let (_dir, _db, ex) = make_db(
        "xcorr_alias_check",
        serde_json::json!({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY xcorr_alias_check COMPUTE c = XCORR(s1, s2) SELECT c;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("c") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            3,
            "XCORR of n=2,m=2 should have length 3, got {}",
            a.len()
        ),
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SIGNAL_ENTROPY ────────────────────────────────────────────────────────────

#[test]
fn test_signal_entropy_uniform_vs_spike() {
    // Uniform distribution has higher entropy than a spike
    let (_dir, _db, ex) = make_db(
        "entropy_cmp",
        serde_json::json!({"uniform": [1.0, 1.0, 1.0, 1.0], "spike": [0.0, 0.0, 0.0, 4.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY entropy_cmp COMPUTE eu = SIGNAL_ENTROPY(uniform) COMPUTE es = SIGNAL_ENTROPY(spike) SELECT eu, es;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let eu = match res.rows[0].data.get("eu") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for eu, got {:?}", other),
    };
    let es = match res.rows[0].data.get("es") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for es, got {:?}", other),
    };
    assert!(
        eu > es,
        "Uniform distribution entropy ({}) should be > spike entropy ({})",
        eu,
        es
    );
}

#[test]
fn test_spectral_entropy_alias() {
    // SPECTRAL_ENTROPY is an alias for SIGNAL_ENTROPY
    let (_dir, _db, ex) = make_db(
        "sp_entropy_alias",
        serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY sp_entropy_alias COMPUTE a = SIGNAL_ENTROPY(arr) COMPUTE b = SPECTRAL_ENTROPY(arr) SELECT a, b;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let a = match res.rows[0].data.get("a") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for a, got {:?}", other),
    };
    let b = match res.rows[0].data.get("b") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for b, got {:?}", other),
    };
    assert!(
        (a - b).abs() < 1e-12,
        "SIGNAL_ENTROPY and SPECTRAL_ENTROPY should agree, got {} vs {}",
        a,
        b
    );
}

// ── DOMINANT_FREQ_IDX ─────────────────────────────────────────────────────────

#[test]
fn test_dominant_freq_idx_peak_at_2() {
    // [0,1,5,2,0]: max is 5 at index 2 => DOMINANT_FREQ_IDX = 2
    let (_dir, _db, ex) = make_db(
        "dfi_peak",
        serde_json::json!({"mags": [0.0, 1.0, 5.0, 2.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY dfi_peak COMPUTE idx = DOMINANT_FREQ_IDX(mags) SELECT idx;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("idx") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 2,
            "DOMINANT_FREQ_IDX([0,1,5,2,0]) should return 2, got {}",
            i
        ),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_argmax_freq_alias() {
    // ARGMAX_FREQ is an alias for DOMINANT_FREQ_IDX
    let (_dir, _db, ex) = make_db(
        "argmax_alias",
        serde_json::json!({"mags": [3.0, 7.0, 1.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY argmax_alias COMPUTE idx = ARGMAX_FREQ(mags) SELECT idx;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("idx") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 1,
            "ARGMAX_FREQ([3,7,1]) should return 1, got {}",
            i
        ),
        other => panic!("expected Integer, got {:?}", other),
    }
}

// ── SAVITZKY_GOLAY ────────────────────────────────────────────────────────────

#[test]
fn test_savitzky_golay_length() {
    // SAVITZKY_GOLAY output should have same length as input
    let (_dir, _db, ex) = make_db(
        "sg_len",
        serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
    );
    let mut p = Parser::new(r#"QUERY sg_len COMPUTE s = SAVITZKY_GOLAY(arr) SELECT s;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("s") {
        Some(Value::Array(a)) => assert_eq!(
            a.len(),
            6,
            "SAVITZKY_GOLAY output length should match input length"
        ),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_sg_smooth_linear_signal() {
    // For a linear signal [1,2,3,4,5,6,7], SG smoother should preserve middle values closely
    let (_dir, _db, ex) = make_db(
        "sg_linear",
        serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}),
    );
    let mut p = Parser::new(r#"QUERY sg_linear COMPUTE s = SG_SMOOTH(arr) SELECT s;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("s") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 7, "SG_SMOOTH output length should match input length");
            // For a linear signal, the Savitzky-Golay filter should preserve it exactly
            // at interior points (polynomial of degree 2 fits line exactly)
            let mid = as_f64(&a[3]); // index 3 -> value should be ~4.0
            assert!(
                (mid - 4.0).abs() < 1e-6,
                "SG_SMOOTH of linear signal at index 3 should be ~4.0, got {}",
                mid
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── FFT_MAGNITUDES ────────────────────────────────────────────────────────────

#[test]
fn test_fft_magnitudes_dc() {
    // For constant signal [1,1,1,1], DC magnitude should be ~4.0
    let (_dir, _db, ex) = make_db(
        "fft_dc",
        serde_json::json!({"arr": [1.0, 1.0, 1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY fft_dc COMPUTE mags = FFT_MAGNITUDES(arr) SELECT mags;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("mags") {
        Some(Value::Array(a)) => {
            assert!(!a.is_empty(), "FFT_MAGNITUDES should return non-empty array");
            let dc = as_f64(&a[0]);
            assert!(
                (dc - 4.0).abs() < 1e-6,
                "FFT_MAGNITUDES DC of [1,1,1,1] should be ~4.0, got {}",
                dc
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_fft_magnitudes_agrees_with_dft_dc() {
    // FFT_MAGNITUDES and DFT_MAGNITUDES should agree on the DC component (bin 0)
    // Note: FFT_MAGNITUDES returns one-sided (N/2+1 bins), DFT_MAGNITUDES returns full N bins
    let (_dir, _db, ex) = make_db(
        "fft_dft_agree",
        serde_json::json!({"arr": [1.0, 0.0, -1.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY fft_dft_agree COMPUTE a = FFT_MAGNITUDES(arr) COMPUTE b = DFT_MAGNITUDES(arr) SELECT a, b;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let a = match res.rows[0].data.get("a") {
        Some(Value::Array(v)) => v.clone(),
        other => panic!("expected Array for a, got {:?}", other),
    };
    let b = match res.rows[0].data.get("b") {
        Some(Value::Array(v)) => v.clone(),
        other => panic!("expected Array for b, got {:?}", other),
    };
    // FFT_MAGNITUDES is one-sided (N/2+1 bins), DFT_MAGNITUDES is full-spectrum (N bins)
    // Both should return arrays with at least 1 element
    assert!(!a.is_empty(), "FFT_MAGNITUDES should return non-empty array");
    assert!(!b.is_empty(), "DFT_MAGNITUDES should return non-empty array");
    // DC component (bin 0) should match
    assert!(
        (as_f64(&a[0]) - as_f64(&b[0])).abs() < 1e-9,
        "DC component of FFT_MAGNITUDES ({}) and DFT_MAGNITUDES ({}) should agree",
        as_f64(&a[0]),
        as_f64(&b[0])
    );
}

// ── DOMINANT_FREQ ─────────────────────────────────────────────────────────────

#[test]
fn test_dominant_freq_returns_float() {
    // DOMINANT_FREQ should return a Float (Hz value)
    let (_dir, _db, ex) = make_db(
        "dom_freq_float",
        serde_json::json!({"mags": [0.0, 1.0, 5.0, 2.0, 0.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY dom_freq_float COMPUTE f = DOMINANT_FREQ(mags, 44100.0, 10) SELECT f;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("f") {
        Some(Value::Float(_)) => {} // just verify it returns a float
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_dominant_freq_dc_component() {
    // When DC (bin 0) is dominant, frequency should be 0 Hz
    let (_dir, _db, ex) = make_db(
        "dom_freq_dc",
        serde_json::json!({"mags": [10.0, 1.0, 0.5]}),
    );
    let mut p = Parser::new(
        r#"QUERY dom_freq_dc COMPUTE f = DOMINANT_FREQ(mags, 1000.0, 6) SELECT f;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("f") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "DOMINANT_FREQ with DC-dominant should return 0.0 Hz, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── SIGNAL_NORMALIZE ──────────────────────────────────────────────────────────

#[test]
fn test_signal_normalize_basic() {
    // SIGNAL_NORMALIZE([2,4,8]) -> [0.25, 0.5, 1.0]
    let (_dir, _db, ex) = make_db(
        "sig_norm_basic",
        serde_json::json!({"arr": [2.0, 4.0, 8.0]}),
    );
    let mut p = Parser::new(r#"QUERY sig_norm_basic COMPUTE n = SIGNAL_NORMALIZE(arr) SELECT n;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("n") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "SIGNAL_NORMALIZE output length should be 3");
            let expected = [0.25, 0.5, 1.0];
            for (i, (&e, v)) in expected.iter().zip(a.iter()).enumerate() {
                assert!(
                    (as_f64(v) - e).abs() < 1e-9,
                    "SIGNAL_NORMALIZE[{}] expected {}, got {}",
                    i,
                    e,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_normalize_signal_alias() {
    // NORMALIZE_SIGNAL is an alias for SIGNAL_NORMALIZE
    let (_dir, _db, ex) = make_db(
        "norm_sig_alias",
        serde_json::json!({"arr": [-4.0, 2.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY norm_sig_alias COMPUTE n = NORMALIZE_SIGNAL(arr) SELECT n;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("n") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "NORMALIZE_SIGNAL output length should be 3");
            // max_abs = 4.0, so [-4/4, 2/4, 0/4] = [-1.0, 0.5, 0.0]
            let expected = [-1.0, 0.5, 0.0];
            for (i, (&e, v)) in expected.iter().zip(a.iter()).enumerate() {
                assert!(
                    (as_f64(v) - e).abs() < 1e-9,
                    "NORMALIZE_SIGNAL[{}] expected {}, got {}",
                    i,
                    e,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ── SPECTRAL_CENTROID ─────────────────────────────────────────────────────────

#[test]
fn test_spectral_centroid_returns_float() {
    // SPECTRAL_CENTROID should return a Float
    let (_dir, _db, ex) = make_db(
        "sc_float",
        serde_json::json!({"mags": [0.0, 2.0, 4.0, 2.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY sc_float COMPUTE c = SPECTRAL_CENTROID(mags, 44100.0) SELECT c;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("c") {
        Some(Value::Float(_)) => {} // just verify type
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_freq_centroid_alias() {
    // FREQ_CENTROID is an alias for SPECTRAL_CENTROID and should return same result
    let (_dir, _db, ex) = make_db(
        "fc_alias",
        serde_json::json!({"mags": [1.0, 3.0, 1.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY fc_alias COMPUTE a = SPECTRAL_CENTROID(mags, 8000.0) COMPUTE b = FREQ_CENTROID(mags, 8000.0) SELECT a, b;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    let a = match res.rows[0].data.get("a") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for a, got {:?}", other),
    };
    let b = match res.rows[0].data.get("b") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float for b, got {:?}", other),
    };
    assert!(
        (a - b).abs() < 1e-12,
        "SPECTRAL_CENTROID and FREQ_CENTROID should agree: {} vs {}",
        a,
        b
    );
}

// ── ENERGY_RATIO ──────────────────────────────────────────────────────────────

#[test]
fn test_energy_ratio_full_band() {
    // All energy in the full range [0..len) => ratio = 1.0
    let (_dir, _db, ex) = make_db(
        "er_full",
        serde_json::json!({"mags": [1.0, 2.0, 3.0, 4.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY er_full COMPUTE r = ENERGY_RATIO(mags, 0, 4) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "ENERGY_RATIO over full range should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_band_energy_ratio_alias() {
    // BAND_ENERGY_RATIO is an alias for ENERGY_RATIO
    let (_dir, _db, ex) = make_db(
        "ber_alias",
        serde_json::json!({"mags": [0.0, 0.0, 5.0, 0.0]}),
    );
    // All energy at index 2, range [2..3) => ratio = 1.0
    let mut p = Parser::new(
        r#"QUERY ber_alias COMPUTE r = BAND_ENERGY_RATIO(mags, 2, 3) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "BAND_ENERGY_RATIO with all energy in band should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

// ── Additional edge-case tests ────────────────────────────────────────────────

#[test]
fn test_rms_amplitude_single_element() {
    // RMS_AMPLITUDE([3.0]) = sqrt(9/1) = 3.0
    let (_dir, _db, ex) = make_db(
        "rms_single",
        serde_json::json!({"arr": [3.0]}),
    );
    let mut p = Parser::new(r#"QUERY rms_single COMPUTE r = RMS_AMPLITUDE(arr) SELECT r;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            (*f - 3.0).abs() < 1e-9,
            "RMS_AMPLITUDE([3.0]) should be 3.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_apply_hamming_empty() {
    // APPLY_HAMMING on empty array should return empty array
    let (_dir, _db, ex) = make_db(
        "hamm_empty",
        serde_json::json!({"arr": []}),
    );
    let mut p = Parser::new(r#"QUERY hamm_empty COMPUTE w = APPLY_HAMMING(arr) SELECT w;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("w") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 0, "APPLY_HAMMING([]) should return []"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_apply_hann_empty() {
    // APPLY_HANN on empty array should return empty array
    let (_dir, _db, ex) = make_db(
        "hann_empty",
        serde_json::json!({"arr": []}),
    );
    let mut p = Parser::new(r#"QUERY hann_empty COMPUTE w = APPLY_HANN(arr) SELECT w;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("w") {
        Some(Value::Array(a)) => assert_eq!(a.len(), 0, "APPLY_HANN([]) should return []"),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_signal_entropy_zero_array() {
    // SIGNAL_ENTROPY of all-zero array should be 0.0
    let (_dir, _db, ex) = make_db(
        "ent_zeros",
        serde_json::json!({"arr": [0.0, 0.0, 0.0]}),
    );
    let mut p = Parser::new(r#"QUERY ent_zeros COMPUTE e = SIGNAL_ENTROPY(arr) SELECT e;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("e") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "SIGNAL_ENTROPY([0,0,0]) should be 0.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_signal_normalize_negative_values() {
    // SIGNAL_NORMALIZE([-6, 3]) -> max_abs=6, so [-1.0, 0.5]
    let (_dir, _db, ex) = make_db(
        "sig_norm_neg",
        serde_json::json!({"arr": [-6.0, 3.0]}),
    );
    let mut p = Parser::new(r#"QUERY sig_norm_neg COMPUTE n = SIGNAL_NORMALIZE(arr) SELECT n;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("n") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 2);
            assert!(
                (as_f64(&a[0]) - (-1.0)).abs() < 1e-9,
                "normalize[-6,3][0] should be -1.0, got {}",
                as_f64(&a[0])
            );
            assert!(
                (as_f64(&a[1]) - 0.5).abs() < 1e-9,
                "normalize[-6,3][1] should be 0.5, got {}",
                as_f64(&a[1])
            );
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn test_dominant_freq_idx_first_element() {
    // [5,1,2,0]: max at index 0 => DOMINANT_FREQ_IDX = 0
    let (_dir, _db, ex) = make_db(
        "dfi_first",
        serde_json::json!({"mags": [5.0, 1.0, 2.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY dfi_first COMPUTE idx = DOMINANT_FREQ_IDX(mags) SELECT idx;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("idx") {
        Some(Value::Integer(i)) => assert_eq!(
            *i, 0,
            "DOMINANT_FREQ_IDX([5,1,2,0]) should return 0, got {}",
            i
        ),
        other => panic!("expected Integer, got {:?}", other),
    }
}

#[test]
fn test_energy_ratio_empty_band() {
    // ENERGY_RATIO with lo==hi (empty range) should be 0.0
    let (_dir, _db, ex) = make_db(
        "er_empty_band",
        serde_json::json!({"mags": [1.0, 2.0, 3.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY er_empty_band COMPUTE r = ENERGY_RATIO(mags, 1, 1) SELECT r;"#,
    );
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("r") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "ENERGY_RATIO with empty band should be 0.0, got {}",
            f
        ),
        other => panic!("expected Float, got {:?}", other),
    }
}

#[test]
fn test_savitzky_golay_short_passthrough() {
    // Arrays shorter than 5 elements should be returned unchanged
    let (_dir, _db, ex) = make_db(
        "sg_short",
        serde_json::json!({"arr": [1.0, 2.0, 3.0]}),
    );
    let mut p = Parser::new(r#"QUERY sg_short COMPUTE s = SAVITZKY_GOLAY(arr) SELECT s;"#);
    let res = ex.execute(p.parse().unwrap()).unwrap();
    match res.rows[0].data.get("s") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 3, "Short array should pass through unchanged");
            let expected = [1.0, 2.0, 3.0];
            for (i, (&e, v)) in expected.iter().zip(a.iter()).enumerate() {
                assert!(
                    (as_f64(v) - e).abs() < 1e-9,
                    "SG short passthrough[{}] expected {}, got {}",
                    i,
                    e,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array, got {:?}", other),
    }
}
