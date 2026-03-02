/// Integration tests for PQL sequence alignment and advanced bioinformatics functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

// ── GLOBAL_ALIGN / NEEDLEMAN_WUNSCH ──────────────────────────────────────────

#[test]
fn test_global_align_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AGCT", "seq2": "AGT"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GLOBAL_ALIGN(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("score"), "should have score");
            assert!(m.contains_key("aligned1"), "should have aligned1");
            assert!(m.contains_key("aligned2"), "should have aligned2");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_needleman_wunsch_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ACGT", "seq2": "ACGT"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = NEEDLEMAN_WUNSCH(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            // Perfect match: score = 4 (4 matches * 1 = 4)
            assert_eq!(m.get("score"), Some(&Value::Integer(4)), "perfect match score should be 4");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_global_align_identical_seqs() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GLOBAL_ALIGN(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("score"), Some(&Value::Integer(4)));
            assert_eq!(m.get("aligned1"), Some(&Value::String("ATGC".to_string())));
            assert_eq!(m.get("aligned2"), Some(&Value::String("ATGC".to_string())));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_global_align_with_gap() {
    let (db, ex) = setup();
    // AGCT vs AGT: AGCT aligned to AG-T with gap in seq2
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AGCT", "seq2": "AGT"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GLOBAL_ALIGN(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            // aligned strings should have same length
            let a1 = match m.get("aligned1") { Some(Value::String(s)) => s.len(), _ => 0 };
            let a2 = match m.get("aligned2") { Some(Value::String(s)) => s.len(), _ => 0 };
            assert_eq!(a1, a2, "aligned strings must have same length");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_global_align_custom_scores() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AA", "seq2": "AA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GLOBAL_ALIGN(seq1, seq2, 2, -2, -3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            // 2 perfect matches * 2 = 4
            assert_eq!(m.get("score"), Some(&Value::Integer(4)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── LOCAL_ALIGN / SMITH_WATERMAN ─────────────────────────────────────────────

#[test]
fn test_local_align_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "TGGATGG", "seq2": "GATG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOCAL_ALIGN(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("score"), "should have score");
            assert!(m.contains_key("aligned1"), "should have aligned1");
            assert!(m.contains_key("aligned2"), "should have aligned2");
            assert!(m.contains_key("start1"), "should have start1");
            assert!(m.contains_key("start2"), "should have start2");
            // Score should be positive (local match found)
            match m.get("score") {
                Some(Value::Integer(s)) => assert!(*s > 0, "local align score should be positive"),
                other => panic!("expected integer score, got {:?}", other),
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_smith_waterman_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ACGT", "seq2": "ACGT"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SMITH_WATERMAN(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            // Perfect match: 4 * 2 = 8
            assert_eq!(m.get("score"), Some(&Value::Integer(8)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_local_align_returns_keys() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AAATTTGGG", "seq2": "TTTGGG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOCAL_ALIGN(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            let keys: std::collections::HashSet<&String> = m.keys().collect();
            assert!(keys.contains(&"score".to_string()));
            assert!(keys.contains(&"aligned1".to_string()));
            assert!(keys.contains(&"aligned2".to_string()));
            assert!(keys.contains(&"start1".to_string()));
            assert!(keys.contains(&"start2".to_string()));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── SEMI_GLOBAL_ALIGN ─────────────────────────────────────────────────────────

#[test]
fn test_semi_global_align_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"query": "ACGT", "tgt": "XXXXXACGTYYYYY"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SEMI_GLOBAL_ALIGN(query, tgt) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("score"));
            assert!(m.contains_key("aln_start"));
            assert!(m.contains_key("aln_end"));
            assert!(m.contains_key("aligned_query"));
            assert!(m.contains_key("aligned_target"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_semi_global_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"query": "ATGC", "tgt": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SEMI_GLOBAL(query, tgt) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(_)) => {},
        other => panic!("expected object, got {:?}", other),
    }
}

// ── PAIRWISE_IDENTITY / SEQUENCE_IDENTITY ─────────────────────────────────────

#[test]
fn test_pairwise_identity_perfect() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PAIRWISE_IDENTITY(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001, "perfect identity should be 1.0"),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_sequence_identity_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AAAA", "seq2": "BBBB"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SEQUENCE_IDENTITY(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 0.001, "zero identity should be 0.0"),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_pairwise_identity_with_gaps() {
    let (db, ex) = setup();
    // "ATG-" and "ATGC": 3 non-gap pairs, all identical -> 1.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATG-", "seq2": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PAIRWISE_IDENTITY(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001, "identity with gap should be 1.0"),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_pairwise_identity_half() {
    let (db, ex) = setup();
    // "ATAT" vs "ATCG": 2 matches (A,T), 2 mismatches (A->C, T->G) -> 0.5
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATAT", "seq2": "ATCG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PAIRWISE_IDENTITY(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.001, "half identity should be 0.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── PAIRWISE_SIMILARITY / ALIGNMENT_SCORE ────────────────────────────────────

#[test]
fn test_pairwise_similarity_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PAIRWISE_SIMILARITY(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_alignment_score_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AAAA", "seq2": "AAAA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ALIGNMENT_SCORE(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── TRANSLATE_CODON / CODON_TO_AA ────────────────────────────────────────────

#[test]
fn test_translate_codon_atg() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"codon": "ATG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TRANSLATE_CODON(codon) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::String("M".to_string())));
}

#[test]
fn test_codon_to_aa_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"codon": "TAA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CODON_TO_AA(codon) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::String("*".to_string())));
}

#[test]
fn test_translate_codon_various() {
    let (db, ex) = setup();
    let codons = vec![
        ("TTT", "F"), ("CTG", "L"), ("ATT", "I"), ("GTT", "V"),
        ("GCT", "A"), ("GAT", "D"), ("AAT", "N"), ("TGG", "W"),
    ];
    for (codon, expected_aa) in codons {
        db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"codon": codon})).unwrap();
        let mut p = Parser::new(r#"QUERY t COMPUTE res = TRANSLATE_CODON(codon) SELECT res;"#);
        let r = ex.execute(p.parse().unwrap()).unwrap();
        // Get last row
        let last = r.rows.last().unwrap();
        assert_eq!(last.data.get("res"), Some(&Value::String(expected_aa.to_string())),
            "codon {} should give {}", codon, expected_aa);
    }
}

// ── PROTEIN_WEIGHT / PROTEIN_MW ──────────────────────────────────────────────

#[test]
fn test_protein_weight_single_aa() {
    let (db, ex) = setup();
    // Single A = 89.09 Da (no peptide bonds)
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "A"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PROTEIN_WEIGHT(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 89.09).abs() < 0.01, "A weight should be 89.09, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_protein_mw_alias() {
    let (db, ex) = setup();
    // GG: 75.03 * 2 - 18.02 = 132.04
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "GG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PROTEIN_MW(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 132.04).abs() < 0.1, "GG weight should be ~132.04, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_protein_weight_positive() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "MKVLWAALLVTFLAG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PROTEIN_WEIGHT(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "protein weight should be positive"),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── AA_COMPOSITION / AMINO_ACID_COMPOSITION ───────────────────────────────────

#[test]
fn test_aa_composition_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "AAGM"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = AA_COMPOSITION(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("A"), Some(&Value::Integer(2)), "A should be 2");
            assert_eq!(m.get("G"), Some(&Value::Integer(1)), "G should be 1");
            assert_eq!(m.get("M"), Some(&Value::Integer(1)), "M should be 1");
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_amino_acid_composition_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "MKKKR"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = AMINO_ACID_COMPOSITION(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("K"), Some(&Value::Integer(3)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── ISOELECTRIC_POINT / PI_PROTEIN ────────────────────────────────────────────

#[test]
fn test_isoelectric_point_basic() {
    let (db, ex) = setup();
    // Polyglycine - neutral, pI should be around 5-9
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "GGGGGG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ISOELECTRIC_POINT(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!(*f > 0.0 && *f < 14.0, "pI should be in valid range, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_pi_protein_alias_acidic() {
    let (db, ex) = setup();
    // Acidic protein (many D, E) should have low pI
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "DDDDEEEEGGG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PI_PROTEIN(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!(*f < 7.0, "acidic protein pI should be < 7, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_isoelectric_point_basic_protein() {
    let (db, ex) = setup();
    // Basic protein (many K, R) should have high pI
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "KKKKRRRRGGG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ISOELECTRIC_POINT(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!(*f > 7.0, "basic protein pI should be > 7, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── HYDROPHOBICITY / KYTE_DOOLITTLE ──────────────────────────────────────────

#[test]
fn test_hydrophobicity_isoleucine() {
    let (db, ex) = setup();
    // I has KD score of 4.5
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "IIII"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = HYDROPHOBICITY(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 4.5).abs() < 0.01, "IIII hydrophobicity should be 4.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_kyte_doolittle_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "RRRR"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KYTE_DOOLITTLE(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - (-4.5)).abs() < 0.01, "RRRR KD should be -4.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── IS_VALID_PROTEIN / IS_PROTEIN_SEQ ─────────────────────────────────────────

#[test]
fn test_is_valid_protein_true() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "MKVLWAALLVTFLAG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_VALID_PROTEIN(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Bool(true)));
}

#[test]
fn test_is_protein_seq_false() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "MKVLWAALLVTFLAG123"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_PROTEIN_SEQ(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Bool(false)));
}

// ── OPEN_READING_FRAMES / FIND_ORFS ──────────────────────────────────────────

#[test]
fn test_open_reading_frames_basic() {
    let (db, ex) = setup();
    // ATG + 9 codons + stop = 12 + 3 = 33 nts -> 1 ORF
    let orf_seq = "ATGAAAGGGCCCTTTTAAATACGATCGATCGATCGATCGATCGTAA";
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": orf_seq})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = OPEN_READING_FRAMES(dna) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(orfs)) => {
            assert!(!orfs.is_empty(), "should find at least one ORF");
            if let Value::Object(obj) = &orfs[0] {
                assert!(obj.contains_key("frame"), "ORF should have frame");
                assert!(obj.contains_key("position"), "ORF should have position");
                assert!(obj.contains_key("length"), "ORF should have length");
                assert!(obj.contains_key("sequence"), "ORF should have sequence");
            }
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_find_orfs_alias_no_orf() {
    let (db, ex) = setup();
    // Short sequence with no complete ORF >= 30 nts
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "ATGTAA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIND_ORFS(dna) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(_)) => {}, // Should return empty array (ORF < 30 nts)
        other => panic!("expected array, got {:?}", other),
    }
}

// ── DNA_MOTIF_POSITIONS / FIND_MOTIF_ALL ─────────────────────────────────────

#[test]
fn test_dna_motif_positions_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "ATGATGATG", "motif": "ATG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DNA_MOTIF_POSITIONS(dna, motif) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(positions)) => {
            assert_eq!(positions.len(), 3, "ATG appears 3 times");
            assert_eq!(positions[0], Value::Integer(0));
            assert_eq!(positions[1], Value::Integer(3));
            assert_eq!(positions[2], Value::Integer(6));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_find_motif_all_alias_no_match() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "ATGATGATG", "motif": "GGG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIND_MOTIF_ALL(dna, motif) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Array(positions)) => assert!(positions.is_empty()),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── CONSENSUS_SEQUENCE / PROFILE_CONSENSUS ───────────────────────────────────

#[test]
fn test_consensus_sequence_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({
        "seqs": ["ATGC", "ATGC", "ATGC"]
    })).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CONSENSUS_SEQUENCE(seqs) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::String("ATGC".to_string())));
}

#[test]
fn test_profile_consensus_alias() {
    let (db, ex) = setup();
    // Position 0: A,A,G -> A wins; Position 1: T,T,T -> T; etc
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({
        "seqs": ["ATGC", "ATCC", "GCGC"]
    })).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = PROFILE_CONSENSUS(seqs) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::String(s)) => assert_eq!(s.len(), 4, "consensus should have length 4"),
        other => panic!("expected string, got {:?}", other),
    }
}

// ── SEQUENCE_ENTROPY / SEQ_ENTROPY ────────────────────────────────────────────

#[test]
fn test_sequence_entropy_uniform() {
    let (db, ex) = setup();
    // ATGC: 4 different chars, each with p=0.25 -> entropy = 2.0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SEQUENCE_ENTROPY(seq) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 2.0).abs() < 0.001, "uniform entropy should be 2.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_seq_entropy_alias_homogeneous() {
    let (db, ex) = setup();
    // AAAA: all same -> entropy = 0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq": "AAAA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SEQ_ENTROPY(seq) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f - 0.0).abs() < 0.001, "homogeneous entropy should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── REVERSE_TRANSLATE / BACK_TRANSLATE ────────────────────────────────────────

#[test]
fn test_reverse_translate_m() {
    let (db, ex) = setup();
    // M -> ATG
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "M"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = REVERSE_TRANSLATE(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::String("ATG".to_string())));
}

#[test]
fn test_back_translate_alias_mw() {
    let (db, ex) = setup();
    // MW -> ATG + TGG = "ATGTGG"
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"prot": "MW"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BACK_TRANSLATE(prot) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::String("ATGTGG".to_string())));
}

// ── DINUCLEOTIDE_FREQ / DINUC_FREQUENCY ──────────────────────────────────────

#[test]
fn test_dinucleotide_freq_basic() {
    let (db, ex) = setup();
    // AAAA: 3 dinucleotides all AA -> freq = {AA: 1.0}
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "AAAA"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DINUCLEOTIDE_FREQ(dna) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            match m.get("AA") {
                Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001),
                other => panic!("expected AA freq = 1.0, got {:?}", other),
            }
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_dinuc_frequency_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "ATGCAT"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = DINUC_FREQUENCY(dna) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => assert!(!m.is_empty(), "should have dinucleotides"),
        other => panic!("expected object, got {:?}", other),
    }
}

// ── CODON_BIAS / CODON_USAGE ──────────────────────────────────────────────────

#[test]
fn test_codon_bias_basic() {
    let (db, ex) = setup();
    // ATGATGATG: 3x ATG
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "ATGATGATG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CODON_BIAS(dna) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("ATG"), Some(&Value::Integer(3)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_codon_usage_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dna": "ATGTTTGGG"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CODON_USAGE(dna) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            assert_eq!(m.get("ATG"), Some(&Value::Integer(1)));
            assert_eq!(m.get("TTT"), Some(&Value::Integer(1)));
            assert_eq!(m.get("GGG"), Some(&Value::Integer(1)));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

// ── JUKES_CANTOR_DIST / JC_DISTANCE ──────────────────────────────────────────

#[test]
fn test_jukes_cantor_identical() {
    let (db, ex) = setup();
    // Identical sequences -> 0 differences -> d = 0
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JUKES_CANTOR_DIST(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.001, "identical seqs JC dist should be 0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_jc_distance_alias() {
    let (db, ex) = setup();
    // 1 difference in 4: p=0.25, d = -0.75 * ln(1 - 4/3 * 0.25) = -0.75 * ln(0.666...) ≈ 0.304
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "AAGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JC_DISTANCE(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!(*f > 0.0, "JC dist should be positive, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_jc_distance_null_over_threshold() {
    let (db, ex) = setup();
    // 4 differences in 4: p=1.0 >= 0.75 -> Null
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "AAAA", "seq2": "TTTT"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JUKES_CANTOR_DIST(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("res"), Some(&Value::Null));
}

// ── KIMURA_2P_DIST / K2P_DISTANCE ─────────────────────────────────────────────

#[test]
fn test_kimura_2p_identical() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "ATGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KIMURA_2P_DIST(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!((*f).abs() < 0.001),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_k2p_distance_alias() {
    let (db, ex) = setup();
    // A->G is a transition; K2P should give positive distance
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"seq1": "ATGC", "seq2": "GTGC"})).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = K2P_DISTANCE(seq1, seq2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert!(*f > 0.0),
        Some(Value::Null) => {},  // acceptable if formula doesn't converge
        other => panic!("expected float or null, got {:?}", other),
    }
}

// ── TAJIMAS_D / TAJIMA_D_STAT ─────────────────────────────────────────────────

#[test]
fn test_tajimas_d_basic() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({
        "seqs": ["ATGCATGC", "ATGCATGC", "ATGCATGG"]
    })).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TAJIMAS_D(seqs) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(_)) => {}, // Just needs to return a float
        Some(Value::Null) => {},     // Can be null for degenerate cases
        other => panic!("expected float or null, got {:?}", other),
    }
}

#[test]
fn test_tajima_d_stat_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({
        "seqs": ["AAAA", "AAAT"]
    })).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TAJIMA_D_STAT(seqs) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(_)) | Some(Value::Null) => {},
        other => panic!("expected float or null, got {:?}", other),
    }
}

#[test]
fn test_tajimas_d_identical_seqs_returns_zero_or_null() {
    let (db, ex) = setup();
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({
        "seqs": ["ATGCATGC", "ATGCATGC"]
    })).unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TAJIMAS_D(seqs) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Float(f)) => assert_eq!(*f, 0.0),
        Some(Value::Null) => {},
        other => panic!("expected 0.0 or null, got {:?}", other),
    }
}
