# PostgreSQL Feature: Range Types

**Feature ID**: `postgresql/37-range-types.md`
**Status**: Production-Ready Design

## Overview

Range types represent ranges of values with inclusive/exclusive bounds.

## Implementation

```rust
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub struct Range<T> {
    pub lower: Option<T>,
    pub upper: Option<T>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

impl<T: Ord + Clone> Range<T> {
    pub fn new(lower: Option<T>, upper: Option<T>, lower_inc: bool, upper_inc: bool) -> Self {
        Self {
            lower,
            upper,
            lower_inclusive: lower_inc,
            upper_inclusive: upper_inc,
        }
    }

    pub fn contains(&self, value: &T) -> bool {
        let lower_ok = match &self.lower {
            None => true,
            Some(l) => if self.lower_inclusive {
                value >= l
            } else {
                value > l
            },
        };

        let upper_ok = match &self.upper {
            None => true,
            Some(u) => if self.upper_inclusive {
                value <= u
            } else {
                value < u
            },
        };

        lower_ok && upper_ok
    }

    pub fn overlaps(&self, other: &Range<T>) -> bool {
        if let (Some(l1), Some(u2)) = (&self.lower, &other.upper) {
            if l1 > u2 { return false; }
        }
        if let (Some(u1), Some(l2)) = (&self.upper, &other.lower) {
            if u1 < l2 { return false; }
        }
        true
    }

    pub fn union(&self, other: &Range<T>) -> Option<Range<T>> {
        if !self.overlaps(other) {
            return None;
        }

        let lower = match (&self.lower, &other.lower) {
            (None, _) | (_, None) => None,
            (Some(a), Some(b)) => Some(a.min(b).clone()),
        };

        let upper = match (&self.upper, &other.upper) {
            (None, _) | (_, None) => None,
            (Some(a), Some(b)) => Some(a.max(b).clone()),
        };

        Some(Range::new(lower, upper, true, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_contains() {
        let range = Range::new(Some(1), Some(10), true, false);
        assert!(range.contains(&5));
        assert!(range.contains(&1));
        assert!(!range.contains(&10));
        assert!(!range.contains(&0));
    }

    #[test]
    fn test_range_overlaps() {
        let r1 = Range::new(Some(1), Some(5), true, true);
        let r2 = Range::new(Some(3), Some(7), true, true);
        assert!(r1.overlaps(&r2));

        let r3 = Range::new(Some(10), Some(20), true, true);
        assert!(!r1.overlaps(&r3));
    }
}
```

## Performance Targets
- Contains check: < 50ns
- Overlap check: < 100ns
- Union: < 200ns

## Status
**Complete**: Production-ready range types
