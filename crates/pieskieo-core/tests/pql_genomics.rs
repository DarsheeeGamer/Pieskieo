/// Integration tests for PQL DNA/genomics built-in functions.
use pieskieo_core::{
    pql::{Executor, Parser, Value},
    PieskieoDb,
};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (db, ex)
}

#[test]
fn test_dna_complement() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = DNA_COMPLEMENT(seq) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("c"), Some(&Value::String("TACG".into())));
}

#[test]
fn test_reverse_complement() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rc = REVERSE_COMPLEMENT(seq) SELECT rc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("rc"),
        Some(&Value::String("GCAT".into()))
    );
}

#[test]
fn test_rev_comp_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "AATTGG"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rc = REV_COMP(seq) SELECT rc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("rc"),
        Some(&Value::String("CCAATT".into()))
    );
}

#[test]
fn test_gc_content() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGCATGC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gc = GC_CONTENT(seq) SELECT gc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gc") {
        Some(Value::Float(f)) => assert!((*f - 0.5).abs() < 0.001, "GC should be 0.5, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_gc_percent_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "GGGGCCCC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE gc = GC_PERCENT(seq) SELECT gc;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("gc") {
        Some(Value::Float(f)) => assert!((*f - 1.0).abs() < 0.001, "GC should be 1.0, got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn test_transcribe_dna() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGCAT"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE rna = TRANSCRIBE_DNA(seq) SELECT rna;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("rna"),
        Some(&Value::String("AUGCAU".into()))
    );
}

#[test]
fn test_translate_dna() {
    let (db, ex) = setup();
    // ATG=M, TTT=F, GGG=G, TAA=stop
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGTTTGGGTAA"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE prot = TRANSLATE_DNA(seq) SELECT prot;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(
        r.rows[0].data.get("prot"),
        Some(&Value::String("MFG".into()))
    );
}

#[test]
fn test_dna_kmers() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kmers = DNA_KMERS(seq, 2) SELECT kmers;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kmers") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert!(arr.contains(&Value::String("AT".into())));
            assert!(arr.contains(&Value::String("TG".into())));
            assert!(arr.contains(&Value::String("GC".into())));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_kmer_list_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "AAATTT"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE kmers = KMER_LIST(seq, 3) SELECT kmers;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("kmers") {
        Some(Value::Array(arr)) => assert_eq!(arr.len(), 4),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_kmer_frequency() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATAT"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE freq = KMER_FREQUENCY(seq, 2) SELECT freq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("freq") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("AT"));
            assert!(m.contains_key("TA"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_is_dna() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGC", "bad": "ATUX"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = IS_DNA(seq) COMPUTE b = IS_DNA(bad) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_is_rna() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "AUGC", "bad": "ATGC"}),
    )
    .unwrap();
    let mut p =
        Parser::new(r#"QUERY t COMPUTE a = IS_RNA(seq) COMPUTE b = IS_RNA(bad) SELECT a, b;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_hamming_distance() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "ATGC", "s2": "AAGC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hd = HAMMING_DISTANCE(s1, s2) SELECT hd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("hd"), Some(&Value::Integer(1)));
}

#[test]
fn test_hamming_dist_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"s1": "AAAA", "s2": "TTTT"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE hd = HAMMING_DIST(s1, s2) SELECT hd;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("hd"), Some(&Value::Integer(4)));
}

#[test]
fn test_codon_count() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGATGATG"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE n = CODON_COUNT(seq) SELECT n;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("n"), Some(&Value::Integer(3)));
}

#[test]
fn test_dna_palindrome() {
    let (db, ex) = setup();
    // GAATTC is its own reverse complement (EcoRI site)
    // ATGCTT is NOT palindromic: rev_comp = AAGCAT != ATGCTT
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "GAATTC", "nonpal": "ATGCTT"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE a = DNA_PALINDROME(seq) COMPUTE b = DNA_PALINDROME(nonpal) SELECT a, b;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("a"), Some(&Value::Bool(true)));
    assert_eq!(r.rows[0].data.get("b"), Some(&Value::Bool(false)));
}

#[test]
fn test_nucleotide_freq() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "AAGG"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE freq = NUCLEOTIDE_FREQ(seq) SELECT freq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("freq") {
        Some(Value::Object(m)) => {
            assert!(m.contains_key("A"));
            assert!(m.contains_key("G"));
        }
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_base_freq_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATTT"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE freq = BASE_FREQ(seq) SELECT freq;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("freq") {
        Some(Value::Object(m)) => match m.get("T") {
            Some(Value::Float(f)) => assert!((*f - 0.75).abs() < 0.001),
            other => panic!("expected T=0.75, got {:?}", other),
        },
        other => panic!("expected object, got {:?}", other),
    }
}

#[test]
fn test_find_motif() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGATGATG", "motif": "ATG"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pos = FIND_MOTIF(seq, motif) SELECT pos;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pos") {
        Some(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], Value::Integer(0));
            assert_eq!(arr[1], Value::Integer(3));
            assert_eq!(arr[2], Value::Integer(6));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_motif_positions_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "GCGCGC", "motif": "GCG"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE pos = MOTIF_POSITIONS(seq, motif) SELECT pos;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("pos") {
        Some(Value::Array(arr)) => assert!(!arr.is_empty()),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn test_melting_temp() {
    let (db, ex) = setup();
    // AAAA: 4*2=8, GGGG: 4*4=16
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"at_seq": "AAAA", "gc_seq": "GGGG"}),
    )
    .unwrap();
    let mut p = Parser::new(
        r#"QUERY t COMPUTE tm_at = MELTING_TEMP(at_seq) COMPUTE tm_gc = MELTING_TEMP(gc_seq) SELECT tm_at, tm_gc;"#,
    );
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(r.rows[0].data.get("tm_at"), Some(&Value::Float(8.0)));
    assert_eq!(r.rows[0].data.get("tm_gc"), Some(&Value::Float(16.0)));
}

#[test]
fn test_tm_basic_alias() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "ATGC"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE tm = TM_BASIC(seq) SELECT tm;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // AT=2, GC=2 → 2*2 + 4*2 = 4+8 = 12
    assert_eq!(r.rows[0].data.get("tm"), Some(&Value::Float(12.0)));
}

#[test]
fn test_dna_complement_lowercase() {
    let (db, ex) = setup();
    db.put_doc_ns(
        None,
        Some("t"),
        Uuid::new_v4(),
        serde_json::json!({"seq": "atgc"}),
    )
    .unwrap();
    let mut p = Parser::new(r#"QUERY t COMPUTE c = DNA_COMPLEMENT(seq) SELECT c;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // lowercase gets uppercased internally
    match r.rows[0].data.get("c") {
        Some(Value::String(s)) => assert_eq!(s, "TACG"),
        other => panic!("expected TACG, got {:?}", other),
    }
}
