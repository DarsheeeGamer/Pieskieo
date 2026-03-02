/// Integration tests for PQL built-in research and optimization functions.
use pieskieo_core::{PieskieoDb, pql::{Executor, Parser, Value}};
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

fn setup() -> (tempfile::TempDir, Arc<PieskieoDb>, Executor) {
    let dir = tempdir().unwrap();
    let db = Arc::new(PieskieoDb::open(dir.path()).unwrap());
    let ex = Executor::new(db.clone());
    (dir, db, ex)
}

fn seed(db: &PieskieoDb) {
    db.put_doc_ns(None, Some("t"), Uuid::new_v4(), serde_json::json!({"dummy": 1})).unwrap();
}

fn get_int(r: &pieskieo_core::pql::QueryResult, key: &str) -> i64 {
    match r.rows[0].data.get(key) {
        Some(Value::Integer(i)) => *i,
        Some(Value::Float(f)) => *f as i64,
        other => panic!("expected Integer for '{}', got {:?}", key, other),
    }
}

fn get_float(r: &pieskieo_core::pql::QueryResult, key: &str) -> f64 {
    match r.rows[0].data.get(key) {
        Some(Value::Float(f)) => *f,
        Some(Value::Integer(i)) => *i as f64,
        other => panic!("expected Float for '{}', got {:?}", key, other),
    }
}

fn get_bool(r: &pieskieo_core::pql::QueryResult, key: &str) -> bool {
    match r.rows[0].data.get(key) {
        Some(Value::Bool(b)) => *b,
        other => panic!("expected Boolean for '{}', got {:?}", key, other),
    }
}

fn get_array(r: &pieskieo_core::pql::QueryResult, key: &str) -> Vec<Value> {
    match r.rows[0].data.get(key) {
        Some(Value::Array(a)) => a.clone(),
        other => panic!("expected Array for '{}', got {:?}", key, other),
    }
}

fn get_obj_int(r: &pieskieo_core::pql::QueryResult, key: &str, field: &str) -> i64 {
    match r.rows[0].data.get(key) {
        Some(Value::Object(m)) => match m.get(field) {
            Some(Value::Integer(i)) => *i,
            Some(Value::Float(f)) => *f as i64,
            other => panic!("expected Integer for field '{}', got {:?}", field, other),
        },
        other => panic!("expected Object for '{}', got {:?}", key, other),
    }
}

fn get_obj_float(r: &pieskieo_core::pql::QueryResult, key: &str, field: &str) -> f64 {
    match r.rows[0].data.get(key) {
        Some(Value::Object(m)) => match m.get(field) {
            Some(Value::Float(f)) => *f,
            Some(Value::Integer(i)) => *i as f64,
            other => panic!("expected Float for field '{}', got {:?}", field, other),
        },
        other => panic!("expected Object for '{}', got {:?}", key, other),
    }
}

// ── Knapsack ─────────────────────────────────────────────────────────────────

#[test]
fn test_knapsack_01_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // weights=[2,3,4,5], values=[3,4,5,6], capacity=5
    // Optimal: items 0 and 1 (weight 5, value 7)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK_01([2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_obj_int(&r, "res", "max_value");
    assert_eq!(v, 7, "max value should be 7, got {}", v);
}

#[test]
fn test_knapsack_01_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK([2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_obj_int(&r, "res", "max_value");
    assert_eq!(v, 7);
}

#[test]
fn test_knapsack_01_items_taken() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // capacity=5 items 0(w=2,v=3) and 1(w=3,v=4) -> total weight 5, value 7
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK_01([2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            if let Some(Value::Array(items)) = m.get("items_taken") {
                assert!(!items.is_empty(), "items_taken should not be empty");
            }
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_knapsack_01_zero_capacity() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK_01([1.0, 2.0], [10.0, 20.0], 0) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_obj_int(&r, "res", "max_value");
    assert_eq!(v, 0, "zero capacity should give 0 value");
}

#[test]
fn test_knapsack_unbounded_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // weights=[1,3,4,5], values=[1,4,5,7], capacity=7
    // Unbounded: 2 items of weight 3 (4+4=8, total weight 6) or 1 item weight 4 + 1 item weight 3 (9)
    // or 7 items of weight 1 (7). Best is item w=3,v=4 twice + item w=1,v=1 = 9?
    // Actually: dp approach: at cap=7, w=4+w=3 = 4+5 = 9
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK_UNBOUNDED([1.0, 3.0, 4.0, 5.0], [1.0, 4.0, 5.0, 7.0], 7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert!(v >= 9, "unbounded knapsack at cap=7 should be >= 9, got {}", v);
}

#[test]
fn test_unbounded_knapsack_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = UNBOUNDED_KNAPSACK([2.0], [3.0], 6) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    // 3 items of weight 2, value 3 each -> total 9
    assert_eq!(v, 9);
}

#[test]
fn test_fractional_knapsack_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // weights=[10,20,30], values=[60,100,120], capacity=50
    // Fractional: take all of item 0 (60), all of item 1 (100), 2/3 of item 2 (80) -> 240
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FRACTIONAL_KNAPSACK([10.0, 20.0, 30.0], [60.0, 100.0, 120.0], 50) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 240.0).abs() < 0.01, "fractional knapsack should be 240.0, got {}", v);
}

#[test]
fn test_greedy_knapsack_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GREEDY_KNAPSACK([10.0, 20.0, 30.0], [60.0, 100.0, 120.0], 50) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 240.0).abs() < 0.01);
}

// ── Subset sum / Coin change ──────────────────────────────────────────────────

#[test]
fn test_subset_sum_true() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SUBSET_SUM([3.0, 1.0, 1.0, 2.0, 2.0, 1.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(get_bool(&r, "res"), "should find subset summing to 5");
}

#[test]
fn test_subset_sum_false() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SUBSET_SUM([2.0, 4.0, 6.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!get_bool(&r, "res"), "no subset sums to 5 with all evens");
}

#[test]
fn test_can_sum_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CAN_SUM([1.0, 2.0, 3.0], 6) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(get_bool(&r, "res"));
}

#[test]
fn test_coin_change_min_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // 25+10+1 = 36, 3 coins
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COIN_CHANGE_MIN([1.0, 5.0, 10.0, 25.0], 36) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 3, "25+10+1 = 3 coins");
}

#[test]
fn test_min_coins_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MIN_COINS([1.0, 5.0, 10.0, 25.0], 11) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 2, "10+1 = 2 coins");
}

#[test]
fn test_coin_change_min_impossible() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COIN_CHANGE_MIN([3.0, 7.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), -1, "cannot make change for 1 with [3,7]");
}

#[test]
fn test_coin_change_count_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // coins=[1,2,5], amount=5: ways = 4 (5; 2+2+1; 2+1+1+1; 1+1+1+1+1)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COIN_CHANGE_COUNT([1.0, 2.0, 5.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 4);
}

#[test]
fn test_count_coin_ways_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COUNT_COIN_WAYS([1.0, 2.0], 4) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // [1,1,1,1], [1,1,2], [2,2] = 3 ways
    assert_eq!(get_int(&r, "res"), 3);
}

// ── Scheduling ────────────────────────────────────────────────────────────────

#[test]
fn test_job_scheduling_edf_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // jobs: [deadline, duration] -> sorted by deadline
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JOB_SCHEDULING_EDF([[5.0, 2.0], [1.0, 1.0], [3.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    // job indices sorted by deadline: job1(d=1), job2(d=3), job0(d=5)
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0], Value::Integer(1));
    assert_eq!(arr[1], Value::Integer(2));
    assert_eq!(arr[2], Value::Integer(0));
}

#[test]
fn test_earliest_deadline_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = EARLIEST_DEADLINE([[3.0, 1.0], [1.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert_eq!(arr[0], Value::Integer(1));
    assert_eq!(arr[1], Value::Integer(0));
}

#[test]
fn test_sjf_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JOB_SCHEDULING_SJF([4.0, 1.0, 3.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    // sorted by duration: 1(d=1), 3(d=2), 2(d=3), 0(d=4)
    assert_eq!(arr[0], Value::Integer(1));
    assert_eq!(arr[3], Value::Integer(0));
}

#[test]
fn test_shortest_job_first_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SHORTEST_JOB_FIRST([5.0, 1.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert_eq!(arr[0], Value::Integer(1));
}

#[test]
fn test_job_scheduling_profit() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // jobs: [deadline, profit]
    // j0=[2,20], j1=[1,15], j2=[2,10], j3=[1,5]
    // Greedy by profit: j0(p=20,d=2)->slot2, j1(p=15,d=1)->slot1, j2(p=10,d=2)->slot2 taken, skip, j3(p=5,d=1)->slot1 taken, skip
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JOB_SCHEDULING_PROFIT([[2.0, 20.0], [1.0, 15.0], [2.0, 10.0], [1.0, 5.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let profit = get_obj_int(&r, "res", "total_profit");
    assert_eq!(profit, 35, "max profit should be 35 (20+15), got {}", profit);
}

#[test]
fn test_makespan_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TOTAL_COMPLETION_TIME([3.0, 5.0, 2.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 14.0).abs() < 0.01, "sum of durations should be 14, got {}", v);
}

#[test]
fn test_makespan_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAKESPAN([1.0, 2.0, 3.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 6.0).abs() < 0.01);
}

#[test]
fn test_weighted_job_schedule() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // jobs [start, end, weight]: [1,3,50], [2,5,20], [4,6,100]
    // Non-overlapping sets: {[1,3,50],[4,6,100]}=150 or {[2,5,20]}=20
    let mut p = Parser::new(r#"QUERY t COMPUTE res = WEIGHTED_JOB_SCHEDULE([[1.0, 3.0, 50.0], [2.0, 5.0, 20.0], [4.0, 6.0, 100.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 150.0).abs() < 0.01, "max weight should be 150, got {}", v);
}

#[test]
fn test_wjs_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = WJS([[0.0, 1.0, 5.0], [0.0, 2.0, 3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!(v >= 5.0, "WJS should select job with weight 5, got {}", v);
}

// ── Bin packing / assignment ──────────────────────────────────────────────────

#[test]
fn test_bin_packing_first_fit() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // items=[6,5,5,4,4,3,3], cap=10 -> FFD: sorted=[6,5,5,4,4,3,3]
    // bin1: 6+4=10; bin2: 5+5=10 -> wait, 5+5=10 ok; bin3: 4+3+3=10 -> 3 bins
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BIN_PACKING_FIRST_FIT([6.0, 5.0, 5.0, 4.0, 4.0, 3.0, 3.0], 10) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert!(v >= 1 && v <= 7, "bin count should be reasonable, got {}", v);
}

#[test]
fn test_first_fit_bin_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FIRST_FIT_BIN([3.0, 3.0, 3.0], 9) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 1, "all 3 items of size 3 fit in one bin of capacity 9");
}

#[test]
fn test_bin_packing_best_fit() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BIN_PACKING_BEST_FIT([4.0, 4.0, 4.0, 4.0], 8) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 2, "four items of 4 with cap 8 should use 2 bins");
}

#[test]
fn test_best_fit_bin_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BEST_FIT_BIN([5.0, 5.0], 10) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 1, "two items of 5 with cap 10 should fit in 1 bin");
}

#[test]
fn test_hungarian_assignment_2x2() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // cost matrix: [[9,2],[3,7]] -> optimal: worker0->task1(2), worker1->task0(3) = 5
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ASSIGNMENT_COST([[9.0, 2.0], [3.0, 7.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cost = get_obj_float(&r, "res", "min_cost");
    assert!((cost - 5.0).abs() < 0.01, "min cost should be 5.0, got {}", cost);
}

#[test]
fn test_hungarian_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = HUNGARIAN_ASSIGNMENT([[1.0, 2.0], [2.0, 1.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cost = get_obj_float(&r, "res", "min_cost");
    assert!((cost - 2.0).abs() < 0.01, "min cost for identity should be 2, got {}", cost);
}

#[test]
fn test_load_balance_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // 4 jobs [3,1,4,1] on 2 processors -> greedy: p0=3, p1=1, p0=3+4=7? No, job2->min loaded
    // After job0: p0=3, p1=0. job1->p1: p0=3,p1=1. job2->p1: p0=3,p1=5. job3->p0: p0=4,p1=5. makespan=5
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOAD_BALANCE([3.0, 1.0, 4.0, 1.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ms = get_obj_float(&r, "res", "makespan");
    assert!(ms > 0.0, "makespan should be positive");
    assert!(ms <= 9.0, "makespan should be at most total sum");
}

#[test]
fn test_multiprocessor_schedule_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MULTIPROCESSOR_SCHEDULE([2.0, 2.0, 2.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ms = get_obj_float(&r, "res", "makespan");
    // 3 jobs of equal size on 3 processors -> makespan = 2
    assert!((ms - 2.0).abs() < 0.01, "equal jobs on equal processors -> makespan=2, got {}", ms);
}

#[test]
fn test_interval_scheduling_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // jobs: [start, end] = [[1,4],[3,5],[0,6],[5,7],[3,9],[5,9],[6,10],[8,11]]
    // Greedy by end: [1,4],[5,7],[8,11] -> 3 non-overlapping
    let mut p = Parser::new(r#"QUERY t COMPUTE res = INTERVAL_SCHEDULING([[1.0, 4.0], [3.0, 5.0], [0.0, 6.0], [5.0, 7.0], [3.0, 9.0], [5.0, 9.0], [6.0, 10.0], [8.0, 11.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 3, "max non-overlapping jobs should be 3, got {}", v);
}

#[test]
fn test_max_jobs_nooverlap_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAX_JOBS_NOOVERLAP([[0.0, 1.0], [2.0, 3.0], [1.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 3, "non-overlapping touching jobs: 3");
}

#[test]
fn test_activity_selection_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ACTIVITY_SELECTION([[1.0, 4.0], [3.0, 5.0], [0.0, 6.0], [5.0, 7.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert!(!arr.is_empty(), "should select at least one activity");
    assert!(arr.len() >= 2, "should select at least 2 non-overlapping activities");
}

#[test]
fn test_activity_select_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ACTIVITY_SELECT([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert_eq!(arr.len(), 3, "three non-overlapping touching intervals -> 3 selected");
}

// ── LP ────────────────────────────────────────────────────────────────────────

#[test]
fn test_lp_simplex_2var_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Maximize x1 + x2 s.t. x1+x2<=4, x1<=3, x2<=3, x1>=0, x2>=0
    // Optimal: x1=1, x2=3 or x1=3,x2=1? Actually x1+x2<=4 means max is 4 at any (x1,x2) with x1+x2=4
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LP_SIMPLEX_2VAR(1, 1, [[1.0, 1.0, 4.0], [1.0, 0.0, 3.0], [0.0, 1.0, 3.0]], true) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let obj = get_obj_float(&r, "res", "obj_value");
    assert!((obj - 4.0).abs() < 0.01, "LP optimal should be 4.0, got {}", obj);
}

#[test]
fn test_simplex_2d_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Maximize 3x1 + 5x2 s.t. x1<=4, 2x2<=12, 3x1+5x2<=25
    // Optimal: x2=5,x1=8/3? Wait: 3(4)+5x2=25 -> x2=13/5=2.6 at x1=4, obj=12+13=25? No...
    // At x1=0,x2=5 (from 2x2<=12->x2<=6, 5x2<=25->x2<=5): obj=25. At x1=4,x2=13/5=2.6: obj=12+13=25
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SIMPLEX_2D(3, 5, [[1.0, 0.0, 4.0], [0.0, 2.0, 12.0], [3.0, 5.0, 25.0]], true) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let obj = get_obj_float(&r, "res", "obj_value");
    assert!((obj - 25.0).abs() < 0.5, "LP objective should be ~25, got {}", obj);
}

#[test]
fn test_lp_objective_value() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LP_OBJECTIVE_VALUE([3.0, 5.0], [2.0, 4.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    // 3*2 + 5*4 = 26
    assert!((v - 26.0).abs() < 0.01, "dot product should be 26, got {}", v);
}

#[test]
fn test_lp_obj_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LP_OBJ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 6.0).abs() < 0.01, "dot product should be 6, got {}", v);
}

#[test]
fn test_lp_feasible_true() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // constraints [[1,0,4],[0,1,3]] x=[2,2]: 1*2+0*2=2<=4, 0*2+1*2=2<=3 -> feasible
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_FEASIBLE_LP([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]], [2.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(get_bool(&r, "res"), "x=[2,2] should be feasible");
}

#[test]
fn test_lp_feasible_false() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // constraint [[1,1,3]] x=[2,2]: 2+2=4 > 3 -> infeasible
    let mut p = Parser::new(r#"QUERY t COMPUTE res = IS_FEASIBLE_LP([[1.0, 1.0, 3.0]], [2.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert!(!get_bool(&r, "res"), "x=[2,2] should violate 1x1+1x2<=3");
}

#[test]
fn test_lp_feasible_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LP_FEASIBLE([[2.0, 1.0, 10.0]], [3.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    // 2*3+1*2=8 <= 10 -> feasible
    assert!(get_bool(&r, "res"));
}

// ── TSP / Combinatorial ───────────────────────────────────────────────────────

#[test]
fn test_tsp_nearest_basic() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // 3-node symmetric: d[0][1]=1, d[1][2]=2, d[0][2]=3
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TRAVELING_SALESMAN_NN([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let dist = get_obj_float(&r, "res", "total_distance");
    assert!(dist > 0.0, "tour distance should be positive");
    // nearest neighbor from 0: go to 1(dist=1), from 1 go to 2(dist=2), back to 0(dist=3) -> total=6
    assert!((dist - 6.0).abs() < 0.01, "NN tour should be 6, got {}", dist);
}

#[test]
fn test_tsp_nearest_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TSP_NEAREST([[0.0, 5.0], [5.0, 0.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let dist = get_obj_float(&r, "res", "total_distance");
    assert!((dist - 10.0).abs() < 0.01, "2-node tour dist=10 (5+5), got {}", dist);
}

#[test]
fn test_tsp_tour_length() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // matrix, tour [0,1,2,0]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TSP_TOUR_LENGTH([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]], [0, 1, 2, 0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    // 0->1(1) + 1->2(2) + 2->0(3) = 6
    assert!((v - 6.0).abs() < 0.01, "tour length should be 6, got {}", v);
}

#[test]
fn test_tour_dist_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TOUR_DIST([[0.0, 2.0], [2.0, 0.0]], [0, 1]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 2.0).abs() < 0.01);
}

#[test]
fn test_max_independent_set_triangle() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Triangle: adj = [[0,1,1],[1,0,1],[1,1,0]] -> max independent set size = 1
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAX_INDEPENDENT_SET([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert_eq!(arr.len(), 1, "triangle has max ind set of size 1, got {:?}", arr);
}

#[test]
fn test_max_ind_set_bipartite() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Path graph 0-1-2: adj=[[0,1,0],[1,0,1],[0,1,0]] -> {0,2} is max independent set size 2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = MAX_IND_SET([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert_eq!(arr.len(), 2, "path graph has max ind set of size 2, got {:?}", arr);
}

#[test]
fn test_graph_coloring_greedy_triangle() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Triangle needs 3 colors
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRAPH_COLORING_GREEDY([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            let colors: Vec<i64> = (0..3).map(|i| match m.get(&i.to_string()) {
                Some(Value::Integer(c)) => *c,
                _ => -1,
            }).collect();
            // All 3 nodes should have different colors
            let c0 = colors[0]; let c1 = colors[1]; let c2 = colors[2];
            assert!(c0 != c1, "adjacent nodes 0,1 must have different colors");
            assert!(c0 != c2, "adjacent nodes 0,2 must have different colors");
            assert!(c1 != c2, "adjacent nodes 1,2 must have different colors");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_greedy_color_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Simple edge 0-1: 2 colors needed
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GREEDY_COLOR([[0, 1], [1, 0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    match r.rows[0].data.get("res") {
        Some(Value::Object(m)) => {
            let c0 = match m.get("0") { Some(Value::Integer(i)) => *i, _ => -1 };
            let c1 = match m.get("1") { Some(Value::Integer(i)) => *i, _ => -1 };
            assert_ne!(c0, c1, "adjacent nodes 0,1 must have different colors");
        }
        other => panic!("expected Object, got {:?}", other),
    }
}

#[test]
fn test_chromatic_number_triangle() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = CHROMATIC_NUMBER_APPROX([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 3, "triangle chromatic number is 3, got {}", v);
}

#[test]
fn test_approx_chromatic_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Complete graph K4 needs 4 colors
    let mut p = Parser::new(r#"QUERY t COMPUTE res = APPROX_CHROMATIC([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 4, "K4 chromatic number is 4, got {}", v);
}

// ── Search and optimization ───────────────────────────────────────────────────

#[test]
fn test_golden_section_search_min() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Minimize f(x) = (x-2)^2 = x^2 - 4x + 4 on [0,5]
    // Coefficients: [4, -4, 1] (constant, linear, quadratic)
    // Minimum at x=2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GOLDEN_SECTION_SEARCH([4.0, -4.0, 1.0], 0, 5, false) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 2.0).abs() < 0.01, "minimum of (x-2)^2 should be at x=2, got {}", v);
}

#[test]
fn test_golden_search_alias_max() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Maximize f(x) = -x^2 + 4x = 4x - x^2 on [0,4]
    // Coefficients: [0, 4, -1], max at x=2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GOLDEN_SEARCH([0.0, 4.0, -1.0], 0, 4, true) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 2.0).abs() < 0.05, "maximum of -x^2+4x should be at x=2, got {}", v);
}

#[test]
fn test_gradient_descent_1d() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Minimize f(x) = (x-3)^2 = x^2 - 6x + 9
    // Coefficients: [9, -6, 1]
    // Minimum at x=3
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GRADIENT_DESCENT_1D([9.0, -6.0, 1.0], 0, 0.01, 5000) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 3.0).abs() < 0.1, "GD should converge to x=3, got {}", v);
}

#[test]
fn test_gd_1d_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Minimize f(x) = x^2 (minimum at 0)
    let mut p = Parser::new(r#"QUERY t COMPUTE res = GD_1D([0.0, 0.0, 1.0], 5, 0.1, 100) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!(v.abs() < 0.5, "GD on x^2 from x=5 should converge near 0, got {}", v);
}

#[test]
fn test_binary_search_found() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BINARY_SEARCH_TARGET([1.0, 3.0, 5.0, 7.0, 9.0], 7) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 3, "7 is at index 3, got {}", v);
}

#[test]
fn test_binary_search_not_found() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BINARY_SEARCH_TARGET([1.0, 3.0, 5.0, 7.0, 9.0], 6) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, -1, "6 not in array, should return -1, got {}", v);
}

#[test]
fn test_binary_search_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BINARY_SEARCH([2.0, 4.0, 6.0, 8.0], 2) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 0, "2 is at index 0, got {}", v);
}

#[test]
fn test_ternary_search_max() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Maximize f(x) = -x^2 + 4x on [0,4] -> max at x=2
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TERNARY_SEARCH_MAX([0.0, 4.0, -1.0], 0, 4) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 2.0).abs() < 0.01, "max of -x^2+4x at x=2, got {}", v);
}

#[test]
fn test_ternary_max_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Maximize f(x) = -(x-5)^2 + 25 on [0,10] -> max at x=5
    // f(x) = -x^2 + 10x = 10x - x^2, coefficients [0, 10, -1]
    let mut p = Parser::new(r#"QUERY t COMPUTE res = TERNARY_MAX([0.0, 10.0, -1.0], 0, 10) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 5.0).abs() < 0.01, "max at x=5, got {}", v);
}

#[test]
fn test_simulated_annealing_approx_finds_min() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Array with clear minimum at index 3
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SIMULATED_ANNEALING_APPROX([10.0, 8.0, 5.0, 1.0, 7.0, 9.0], 2000, 42) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 3, "SA should find minimum at index 3, got {}", v);
}

#[test]
fn test_sa_min_alias() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Simple: minimum at index 0
    let mut p = Parser::new(r#"QUERY t COMPUTE res = SA_MIN([0.0, 5.0, 10.0, 15.0], 500, 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_int(&r, "res");
    assert_eq!(v, 0, "minimum at index 0, got {}", v);
}

// ── Extra edge-case / combined tests ─────────────────────────────────────────

#[test]
fn test_knapsack_single_item_fits() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK_01([3.0], [10.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_obj_int(&r, "res", "max_value");
    assert_eq!(v, 10);
}

#[test]
fn test_knapsack_no_items_fit() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = KNAPSACK_01([10.0, 20.0], [5.0, 8.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_obj_int(&r, "res", "max_value");
    assert_eq!(v, 0, "no items fit in capacity 3");
}

#[test]
fn test_coin_change_single_coin_exact() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = COIN_CHANGE_MIN([5.0], 25) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 5);
}

#[test]
fn test_interval_scheduling_single_job() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = INTERVAL_SCHEDULING([[0.0, 10.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 1);
}

#[test]
fn test_load_balance_single_job() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LOAD_BALANCE([7.0], 3) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let ms = get_obj_float(&r, "res", "makespan");
    assert!((ms - 7.0).abs() < 0.01);
}

#[test]
fn test_fractional_knapsack_full_capacity() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // All items fit completely
    let mut p = Parser::new(r#"QUERY t COMPUTE res = FRACTIONAL_KNAPSACK([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 100) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    // All items taken: 10+20+30=60
    assert!((v - 60.0).abs() < 0.01, "all items should fit, value=60, got {}", v);
}

#[test]
fn test_binary_search_first_element() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BINARY_SEARCH_TARGET([1.0, 2.0, 3.0, 4.0, 5.0], 1) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 0);
}

#[test]
fn test_binary_search_last_element() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = BINARY_SEARCH_TARGET([1.0, 2.0, 3.0, 4.0, 5.0], 5) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    assert_eq!(get_int(&r, "res"), 4);
}

#[test]
fn test_lp_simplex_maximize_single_var() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // Maximize 2*x1 s.t. x1<=5
    let mut p = Parser::new(r#"QUERY t COMPUTE res = LP_SIMPLEX_2VAR(2, 0, [[1.0, 0.0, 5.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let obj = get_obj_float(&r, "res", "obj_value");
    assert!((obj - 10.0).abs() < 0.01, "2*5=10, got {}", obj);
}

#[test]
fn test_sjf_empty_ties_resolved() {
    let (_dir, db, ex) = setup();
    seed(&db);
    let mut p = Parser::new(r#"QUERY t COMPUTE res = JOB_SCHEDULING_SJF([2.0, 2.0, 2.0]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let arr = get_array(&r, "res");
    assert_eq!(arr.len(), 3, "should return all 3 jobs");
}

#[test]
fn test_weighted_job_nonoverlap_all_separate() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // All jobs separate, pick all
    let mut p = Parser::new(r#"QUERY t COMPUTE res = WEIGHTED_JOB_SCHEDULE([[0.0, 1.0, 10.0], [2.0, 3.0, 20.0], [4.0, 5.0, 30.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let v = get_float(&r, "res");
    assert!((v - 60.0).abs() < 0.01, "all separate jobs, total=60, got {}", v);
}

#[test]
fn test_hungarian_3x3() {
    let (_dir, db, ex) = setup();
    seed(&db);
    // cost [[4,1,3],[2,0,5],[3,2,2]] -> optimal assignment
    let mut p = Parser::new(r#"QUERY t COMPUTE res = ASSIGNMENT_COST([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]) SELECT res;"#);
    let r = ex.execute(p.parse().unwrap()).unwrap();
    let cost = get_obj_float(&r, "res", "min_cost");
    // Optimal: row0->col1(1), row1->col0 or col2, row2->...
    // row0->col1(1), row1->col2(5)? No. row0->col1(1), row1->col0(2), row2->col2(2) = 5
    // or row0->col2(3), row1->col1(0), row2->col0(3) = 6. Min is 5.
    assert!((cost - 5.0).abs() < 0.01, "min cost should be 5, got {}", cost);
}
