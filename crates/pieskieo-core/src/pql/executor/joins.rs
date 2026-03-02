use crate::error::Result;
use crate::pql::ast::{Condition, JoinType, SourceExpr};

use super::{expressions, source, ExecutionStats, Executor, Row};

pub(crate) fn execute_join(
    executor: &Executor,
    left: Vec<Row>,
    join_type: JoinType,
    right_source: SourceExpr,
    condition: Condition,
) -> Result<Vec<Row>> {
    let mut stats = ExecutionStats::default();
    let right = source::load_source(executor, &right_source, &mut stats, None)?;
    let mut result = Vec::new();

    match join_type {
        JoinType::Inner => {
            for left_row in &left {
                for right_row in &right {
                    let joined = source::merge_rows(left_row, right_row);
                    if expressions::evaluate_condition(executor, &condition, &joined) {
                        result.push(joined);
                    }
                }
            }
        }
        JoinType::Left => {
            for left_row in &left {
                let mut matched = false;
                for right_row in &right {
                    let joined = source::merge_rows(left_row, right_row);
                    if expressions::evaluate_condition(executor, &condition, &joined) {
                        result.push(joined);
                        matched = true;
                    }
                }
                if !matched {
                    result.push(left_row.clone());
                }
            }
        }
        JoinType::Right => {
            for right_row in &right {
                let mut matched = false;
                for left_row in &left {
                    let joined = source::merge_rows(left_row, right_row);
                    if expressions::evaluate_condition(executor, &condition, &joined) {
                        result.push(joined);
                        matched = true;
                    }
                }
                if !matched {
                    result.push(source::row_with_right_only(right_row));
                }
            }
        }
        JoinType::Full => {
            let mut right_matched = vec![false; right.len()];
            for left_row in &left {
                let mut matched = false;
                for (idx, right_row) in right.iter().enumerate() {
                    let joined = source::merge_rows(left_row, right_row);
                    if expressions::evaluate_condition(executor, &condition, &joined) {
                        result.push(joined);
                        matched = true;
                        right_matched[idx] = true;
                    }
                }
                if !matched {
                    result.push(left_row.clone());
                }
            }
            for (idx, right_row) in right.iter().enumerate() {
                if !right_matched[idx] {
                    result.push(source::row_with_right_only(right_row));
                }
            }
        }
        JoinType::Cross => {
            for left_row in &left {
                for right_row in &right {
                    let joined = source::merge_rows(left_row, right_row);
                    if expressions::evaluate_condition(executor, &condition, &joined) {
                        result.push(joined);
                    }
                }
            }
        }
    }

    Ok(result)
}
