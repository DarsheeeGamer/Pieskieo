/// Integration tests for PQL signal processing and frequency domain functions.
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

// ── DFT_MAGNITUDES ───────────────────────────────────────────────────────────

#[test]
fn test_dft_magnitudes_dc() {
    // DFT of a constant signal [1,1,1,1] should have DC magnitude = 4.0, all others ~0
    let (_dir, _db, ex) = make_db("t_dft_dc", serde_json::json!({"arr": [1.0, 1.0, 1.0, 1.0]}));
    let mut p = Parser::new(r#"QUERY t_dft_dc COMPUTE mags = DFT_MAGNITUDES(arr) SELECT mags;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("mags") {
        Some(Value::Array(a)) => {
            assert!(!a.is_empty(), "expected non-empty magnitudes array");
            let dc = as_f64(&a[0]);
            assert!(
                (dc - 4.0).abs() < 0.01,
                "DC component of [1,1,1,1] should be 4.0, got {}",
                dc
            );
            // All non-DC components should be near zero
            for (k, v) in a.iter().enumerate().skip(1) {
                let m = as_f64(v);
                assert!(
                    m < 0.01,
                    "Non-DC magnitude[{}] should be ~0 for constant signal, got {}",
                    k,
                    m
                );
            }
        }
        other => panic!("expected Array for mags, got {:?}", other),
    }
}

#[test]
fn test_dft_magnitudes_alias() {
    // SPECTRAL_MAGNITUDES is an alias for DFT_MAGNITUDES
    let (_dir, _db, ex) = make_db(
        "t_spec_mag",
        serde_json::json!({"arr": [1.0, 0.0, -1.0, 0.0]}),
    );
    let mut p = Parser::new(
        r#"QUERY t_spec_mag COMPUTE a = DFT_MAGNITUDES(arr) COMPUTE b = SPECTRAL_MAGNITUDES(arr) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let a = match r.rows[0].data.get("a") {
        Some(Value::Array(v)) => v.clone(),
        other => panic!("expected Array for a, got {:?}", other),
    };
    let b = match r.rows[0].data.get("b") {
        Some(Value::Array(v)) => v.clone(),
        other => panic!("expected Array for b, got {:?}", other),
    };
    assert_eq!(a.len(), b.len(), "aliases must return same length");
    for (av, bv) in a.iter().zip(b.iter()) {
        assert!(
            (as_f64(av) - as_f64(bv)).abs() < 1e-10,
            "DFT_MAGNITUDES and SPECTRAL_MAGNITUDES must agree"
        );
    }
}

// ── ZERO_CROSSING_RATE ───────────────────────────────────────────────────────

#[test]
fn test_zero_crossing_rate_alternating() {
    // Signal that alternates sign: every consecutive pair has a crossing => ZCR = 1.0
    let (_dir, _db, ex) = make_db(
        "t_zcr",
        serde_json::json!({"arr": [1.0, -1.0, 1.0, -1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY t_zcr COMPUTE z = ZERO_CROSSING_RATE(arr) SELECT z;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("z") {
        Some(Value::Float(f)) => assert!(
            (*f - 1.0).abs() < 1e-9,
            "alternating signal ZCR should be 1.0, got {}",
            f
        ),
        other => panic!("expected Float for z, got {:?}", other),
    }
}

#[test]
fn test_zero_crossing_rate_constant() {
    // Constant positive signal: no zero crossings => ZCR = 0.0
    let (_dir, _db, ex) = make_db("t_zcr2", serde_json::json!({"arr": [3.0, 3.0, 3.0, 3.0]}));
    let mut p = Parser::new(r#"QUERY t_zcr2 COMPUTE z = ZCR(arr) SELECT z;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("z") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "constant signal ZCR should be 0.0, got {}",
            f
        ),
        other => panic!("expected Float for z, got {:?}", other),
    }
}

// ── SIGNAL_ENERGY & SIGNAL_POWER ─────────────────────────────────────────────

#[test]
fn test_signal_energy() {
    // SIGNAL_ENERGY([3, 4]) = (9 + 16) / 2 = 12.5
    let (_dir, _db, ex) = make_db("t_se", serde_json::json!({"arr": [3.0, 4.0]}));
    let mut p = Parser::new(r#"QUERY t_se COMPUTE e = SIGNAL_ENERGY(arr) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("e") {
        Some(Value::Float(f)) => assert!(
            (*f - 12.5).abs() < 1e-9,
            "SIGNAL_ENERGY([3,4]) should be 12.5, got {}",
            f
        ),
        other => panic!("expected Float for e, got {:?}", other),
    }
}

#[test]
fn test_signal_power_rms() {
    // SIGNAL_POWER([3, 4]) = sqrt((9+16)/2) = sqrt(12.5)
    let (_dir, _db, ex) = make_db("t_sp", serde_json::json!({"arr": [3.0, 4.0]}));
    let mut p = Parser::new(r#"QUERY t_sp COMPUTE p = SIGNAL_POWER(arr) SELECT p;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let expected = 12.5_f64.sqrt();
    match r.rows[0].data.get("p") {
        Some(Value::Float(f)) => assert!(
            (*f - expected).abs() < 1e-9,
            "SIGNAL_POWER([3,4]) should be {}, got {}",
            expected,
            f
        ),
        other => panic!("expected Float for p, got {:?}", other),
    }
}

// ── HANN_WINDOW ──────────────────────────────────────────────────────────────

#[test]
fn test_hann_window_endpoints() {
    // Hann window: first and last values should be 0 for n > 1
    let (_dir, _db, ex) = make_db("t_hann", serde_json::json!({"n": 8}));
    let mut p = Parser::new(r#"QUERY t_hann COMPUTE w = HANN_WINDOW(n) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("w") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 8, "Hann window length should be 8");
            let first = as_f64(&a[0]);
            let last = as_f64(&a[7]);
            assert!(
                first.abs() < 1e-9,
                "Hann window first value should be 0, got {}",
                first
            );
            assert!(
                last.abs() < 1e-9,
                "Hann window last value should be 0, got {}",
                last
            );
            // Middle values should be positive
            let mid = as_f64(&a[4]);
            assert!(mid > 0.0, "Hann window middle value should be positive");
        }
        other => panic!("expected Array for w, got {:?}", other),
    }
}

#[test]
fn test_hamming_window_length() {
    // Hamming window should have correct length and non-zero endpoints
    let (_dir, _db, ex) = make_db("t_hamming", serde_json::json!({"n": 5}));
    let mut p = Parser::new(r#"QUERY t_hamming COMPUTE w = HAMMING_WINDOW(n) SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("w") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "Hamming window length should be 5");
            // Hamming endpoints = 0.54 - 0.46 = 0.08
            let first = as_f64(&a[0]);
            assert!(
                (first - 0.08).abs() < 1e-9,
                "Hamming window first value should be 0.08, got {}",
                first
            );
        }
        other => panic!("expected Array for w, got {:?}", other),
    }
}

// ── CONVOLUTION ──────────────────────────────────────────────────────────────

#[test]
fn test_convolution_basic() {
    // CONVOLUTION([1,2,3], [1,1]) should be [1, 3, 5, 3]
    let (_dir, _db, ex) = make_db(
        "t_conv",
        serde_json::json!({"sig": [1.0, 2.0, 3.0], "kern": [1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY t_conv COMPUTE c = CONVOLUTION(sig, kern) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("c") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 4, "convolution length should be 4");
            let expected = [1.0, 3.0, 5.0, 3.0];
            for (i, (v, &e)) in a.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (as_f64(v) - e).abs() < 1e-9,
                    "convolution[{}] should be {}, got {}",
                    i,
                    e,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array for c, got {:?}", other),
    }
}

// ── LOW_PASS_FILTER ──────────────────────────────────────────────────────────

#[test]
fn test_low_pass_filter_removes_high_freq() {
    // A low-pass filter with small cutoff should smooth the signal
    // For a constant signal, LPF should return the same signal
    let (_dir, _db, ex) = make_db(
        "t_lpf",
        serde_json::json!({"arr": [5.0, 5.0, 5.0, 5.0, 5.0]}),
    );
    let mut p =
        Parser::new(r#"QUERY t_lpf COMPUTE filtered = LOW_PASS_FILTER(arr, 0.5) SELECT filtered;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("filtered") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "LPF should return same length array");
            for (i, v) in a.iter().enumerate() {
                assert!(
                    (as_f64(v) - 5.0).abs() < 1e-9,
                    "LPF of constant signal[{}] should be 5.0, got {}",
                    i,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array for filtered, got {:?}", other),
    }
}

// ── BAND_ENERGY ──────────────────────────────────────────────────────────────

#[test]
fn test_band_energy_dc_only() {
    // For [1,1,1,1], all energy is at DC (index 0).
    // BAND_ENERGY with range [0,0] should equal magnitude^2 = 16.0
    let (_dir, _db, ex) = make_db("t_be", serde_json::json!({"arr": [1.0, 1.0, 1.0, 1.0]}));
    let mut p = Parser::new(r#"QUERY t_be COMPUTE e = BAND_ENERGY(arr, 0, 0) SELECT e;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("e") {
        Some(Value::Float(f)) => assert!(
            (*f - 16.0).abs() < 0.01,
            "BAND_ENERGY at DC for [1,1,1,1] should be 16.0, got {}",
            f
        ),
        other => panic!("expected Float for e, got {:?}", other),
    }
}

// ── CROSS_CORRELATION ────────────────────────────────────────────────────────

#[test]
fn test_cross_correlation_identical() {
    // Cross-correlation of a signal with itself at lag 0 is maximized
    // For [1,2,3], xcorr at center (lag=0) should be 1+4+9 = 14
    let (_dir, _db, ex) = make_db("t_xcorr", serde_json::json!({"arr": [1.0, 2.0, 3.0]}));
    let mut p = Parser::new(r#"QUERY t_xcorr COMPUTE xc = CROSS_CORRELATION(arr, arr) SELECT xc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("xc") {
        Some(Value::Array(a)) => {
            // 2n-1 = 5 elements for n=3; center index is 2
            assert_eq!(a.len(), 5, "xcorr of n=3 should have 2*3-1=5 elements");
            let center = as_f64(&a[2]);
            assert!(
                (center - 14.0).abs() < 1e-9,
                "xcorr at lag=0 for [1,2,3] should be 14.0, got {}",
                center
            );
        }
        other => panic!("expected Array for xc, got {:?}", other),
    }
}

// ── AUTOCORRELATION_FULL ─────────────────────────────────────────────────────

#[test]
fn test_autocorrelation_full_lag0_is_one() {
    // Normalized autocorrelation at lag 0 should always be 1.0
    let (_dir, _db, ex) = make_db(
        "t_acf",
        serde_json::json!({"arr": [1.0, 3.0, 2.0, 5.0, 4.0]}),
    );
    let mut p = Parser::new(r#"QUERY t_acf COMPUTE ac = AUTOCORRELATION_FULL(arr) SELECT ac;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("ac") {
        Some(Value::Array(a)) => {
            assert!(!a.is_empty(), "autocorrelation array should not be empty");
            let lag0 = as_f64(&a[0]);
            assert!(
                (lag0 - 1.0).abs() < 1e-9,
                "AUTOCORRELATION_FULL at lag=0 should be 1.0, got {}",
                lag0
            );
        }
        other => panic!("expected Array for ac, got {:?}", other),
    }
}

// ── SNR ──────────────────────────────────────────────────────────────────────

#[test]
fn test_snr_equal_energy() {
    // When signal and noise have equal energy, SNR = 0 dB
    let (_dir, _db, ex) = make_db(
        "t_snr",
        serde_json::json!({"sig": [1.0, 1.0], "noise": [1.0, 1.0]}),
    );
    let mut p = Parser::new(r#"QUERY t_snr COMPUTE s = SNR(sig, noise) SELECT s;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("s") {
        Some(Value::Float(f)) => assert!(
            f.abs() < 1e-9,
            "SNR with equal signal and noise should be 0 dB, got {}",
            f
        ),
        other => panic!("expected Float for s, got {:?}", other),
    }
}

// ── APPLY_WINDOW ─────────────────────────────────────────────────────────────

#[test]
fn test_apply_window_rectangular() {
    // Rectangular window is identity: output = input
    let (_dir, _db, ex) = make_db("t_aw", serde_json::json!({"arr": [1.0, 2.0, 3.0, 4.0]}));
    let mut p = Parser::new(r#"QUERY t_aw COMPUTE w = APPLY_WINDOW(arr, "rectangular") SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("w") {
        Some(Value::Array(a)) => {
            assert_eq!(
                a.len(),
                4,
                "rectangular windowed array should have same length"
            );
            for (i, v) in a.iter().enumerate() {
                let expected = (i + 1) as f64;
                assert!(
                    (as_f64(v) - expected).abs() < 1e-9,
                    "rectangular window[{}] should be {}, got {}",
                    i,
                    expected,
                    as_f64(v)
                );
            }
        }
        other => panic!("expected Array for w, got {:?}", other),
    }
}

#[test]
fn test_apply_window_hann_zeroes_endpoints() {
    // Hann window applied to any signal zeroes out first and last elements
    let (_dir, _db, ex) = make_db(
        "t_aw_hann",
        serde_json::json!({"arr": [10.0, 20.0, 30.0, 20.0, 10.0]}),
    );
    let mut p = Parser::new(r#"QUERY t_aw_hann COMPUTE w = APPLY_WINDOW(arr, "hann") SELECT w;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("w") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5);
            let first = as_f64(&a[0]);
            let last = as_f64(&a[4]);
            assert!(
                first.abs() < 1e-9,
                "Hann-windowed first element should be 0, got {}",
                first
            );
            assert!(
                last.abs() < 1e-9,
                "Hann-windowed last element should be 0, got {}",
                last
            );
        }
        other => panic!("expected Array for w, got {:?}", other),
    }
}

// ── DOMINANT_FREQUENCY ───────────────────────────────────────────────────────

#[test]
fn test_dominant_frequency_single_tone() {
    // A pure sinusoid at frequency bin k should have its peak at that bin.
    // Generate 8 samples of cos(2*pi*k*n/N) for k=2, N=8
    // The dominant frequency should be index 2.
    let n = 8usize;
    let k = 2usize;
    let samples: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64).cos())
        .collect();
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    db.put_doc_ns(
        None,
        Some("t_dom"),
        Uuid::new_v4(),
        serde_json::json!({"arr": samples}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t_dom COMPUTE d = DOMINANT_FREQUENCY(arr) SELECT d;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("d") {
        Some(Value::Integer(i)) => {
            // A real cosine at bin k produces equal energy at bins k and N-k (mirror frequency).
            // DOMINANT_FREQUENCY may return either 2 or 8-2=6 — both are correct.
            let mirror = (n - k) as i64;
            assert!(
                *i == k as i64 || *i == mirror,
                "dominant frequency of cos(2*pi*2*n/8) should be index 2 or its mirror {}, got {}",
                mirror,
                i
            );
        }
        other => panic!("expected Integer for d, got {:?}", other),
    }
}

// ── HIGH_PASS_FILTER ─────────────────────────────────────────────────────────

#[test]
fn test_high_pass_filter_removes_dc() {
    // HPF with high cutoff on a constant signal should yield ~0 (removes DC)
    let (_dir, _db, ex) = make_db(
        "t_hpf",
        serde_json::json!({"arr": [5.0, 5.0, 5.0, 5.0, 5.0]}),
    );
    // cutoff=0.9 means 1-cutoff=0.1, window=10 => moving average is long enough to approximate DC
    let mut p = Parser::new(
        r#"QUERY t_hpf COMPUTE filtered = HIGH_PASS_FILTER(arr, 0.9) SELECT filtered;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("filtered") {
        Some(Value::Array(a)) => {
            assert_eq!(a.len(), 5, "HPF should return same length array");
            // Last element: moving average over all 5 elements equals the constant, so HPF output = 0
            let last = as_f64(&a[4]);
            assert!(
                last.abs() < 1e-9,
                "HPF of constant signal (last element) should be 0.0, got {}",
                last
            );
        }
        other => panic!("expected Array for filtered, got {:?}", other),
    }
}
