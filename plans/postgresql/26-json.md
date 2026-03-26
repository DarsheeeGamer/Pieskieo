# PostgreSQL Feature: JSON/JSONB Types and Operators

**Status**: 🔴 Not Started  
**Priority**: CRITICAL (MongoDB compatibility depends on this)
**Dependencies**: GIN Indexes
**Estimated Effort**: 3-4 weeks

---

## Overview

PostgreSQL provides powerful JSON processing capabilities, primarily through two data types: `json` (stores exact text) and `jsonb` (stores decomposed binary format for fast querying and indexing). Since Pieskieo already has a strong document model (MongoDB-like), implementing `JSONB` in the SQL engine acts as the bridge between relational tables and document collections.

## Core Requirements

1. **`jsonb` Data Type**: Binary representation of JSON, optimizing for parsing speed and indexing.
2. **Operators**: Extractor (`->`, `->>`), Path (`#>`, `#>>`), Containment (`@>`, `<@`), Existence (`?`, `?|`, `?&`), Concatenation (`||`), Deletion (`-`, `#-`).
3. **Functions**: Generation (`jsonb_build_object`, `to_jsonb`), Processing (`jsonb_array_elements`, `jsonb_each`, `jsonb_object_keys`), Modification (`jsonb_set`, `jsonb_insert`).

---

## Implementation Plan

### Phase 1: Internal Representation (`Jsonb`)

While we can use `serde_json::Value` internally, a true `jsonb` implementation requires a fast binary format that allows traversing elements without parsing the whole tree (similar to PostgreSQL's representation or flatbuffers). For Pieskieo, we can start with `serde_json::Value` and optimize to a custom binary format later if needed, but the operators must behave like `jsonb`.

```rust
// crates/pieskieo-core/src/types/jsonb.rs

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Jsonb(pub serde_json::Value);

impl Jsonb {
    // Parsing ensures strict JSON validity and removes duplicate keys (keeping the last)
    pub fn parse(s: &str) -> Result<Self> {
        let val = serde_json::from_str(s)?;
        Ok(Jsonb(val))
    }
}
```

### Phase 2: Extractor and Path Operators

```rust
// Extract JSON object field by key or array element by index.
// -> returns Jsonb, ->> returns Text.
impl Jsonb {
    pub fn get_field(&self, key: &str) -> Option<&serde_json::Value> {
        if let serde_json::Value::Object(map) = &self.0 {
            map.get(key)
        } else {
            None
        }
    }

    pub fn get_element(&self, index: i64) -> Option<&serde_json::Value> {
        if let serde_json::Value::Array(arr) = &self.0 {
            let idx = if index < 0 {
                // Negative index counts from end
                arr.len() as i64 + index
            } else {
                index
            };
            if idx >= 0 && (idx as usize) < arr.len() {
                Some(&arr[idx as usize])
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn get_path(&self, path: &[String]) -> Option<&serde_json::Value> {
        let mut current = &self.0;
        for segment in path {
            current = match current {
                serde_json::Value::Object(map) => map.get(segment)?,
                serde_json::Value::Array(arr) => {
                    if let Ok(idx) = segment.parse::<i64>() {
                        let actual_idx = if idx < 0 {
                            arr.len() as i64 + idx
                        } else {
                            idx
                        };
                        if actual_idx >= 0 && (actual_idx as usize) < arr.len() {
                            &arr[actual_idx as usize]
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                _ => return None,
            };
        }
        Some(current)
    }
}
```

### Phase 3: Containment and Existence Operators

These are critical for GIN index acceleration.

```rust
impl Jsonb {
    // @> Does left value contain right value?
    pub fn contains(&self, other: &Jsonb) -> bool {
        self.contains_recursive(&self.0, &other.0)
    }

    fn contains_recursive(&self, target: &serde_json::Value, contained: &serde_json::Value) -> bool {
        match (target, contained) {
            (serde_json::Value::Object(t), serde_json::Value::Object(c)) => {
                c.iter().all(|(k, v)| t.get(k).map_or(false, |tv| self.contains_recursive(tv, v)))
            }
            (serde_json::Value::Array(t), serde_json::Value::Array(c)) => {
                // For array containment, ALL elements in 'c' must be in 't'. Order doesn't matter.
                c.iter().all(|cv| t.iter().any(|tv| self.contains_recursive(tv, cv)))
            }
            // Array contains scalar if scalar is an element
            (serde_json::Value::Array(t), _) => {
                t.iter().any(|tv| self.contains_recursive(tv, contained))
            }
            // Scalars match strictly
            (t, c) => t == c,
        }
    }

    // ? Does string exist as a top-level key or array element?
    pub fn exists(&self, key: &str) -> bool {
        match &self.0 {
            serde_json::Value::Object(map) => map.contains_key(key),
            serde_json::Value::Array(arr) => arr.iter().any(|v| v == &serde_json::Value::String(key.to_string())),
            serde_json::Value::String(s) => s == key,
            _ => false,
        }
    }
}
```

### Phase 4: Modification and Concatenation

```rust
impl Jsonb {
    // || Concatenation
    pub fn concat(&self, other: &Jsonb) -> Jsonb {
        match (&self.0, &other.0) {
            (serde_json::Value::Object(m1), serde_json::Value::Object(m2)) => {
                let mut new_map = m1.clone();
                new_map.extend(m2.clone());
                Jsonb(serde_json::Value::Object(new_map))
            }
            (serde_json::Value::Array(a1), serde_json::Value::Array(a2)) => {
                let mut new_arr = a1.clone();
                new_arr.extend(a2.clone());
                Jsonb(serde_json::Value::Array(new_arr))
            }
            // Array + Scalar = Array with scalar appended
            (serde_json::Value::Array(a1), v) => {
                let mut new_arr = a1.clone();
                new_arr.push(v.clone());
                Jsonb(serde_json::Value::Array(new_arr))
            }
            // Scalar + Array = Array with scalar prepended
            (v, serde_json::Value::Array(a2)) => {
                let mut new_arr = vec![v.clone()];
                new_arr.extend(a2.clone());
                Jsonb(serde_json::Value::Array(new_arr))
            }
            // Scalar + Scalar = Array of two scalars
            (v1, v2) => {
                Jsonb(serde_json::Value::Array(vec![v1.clone(), v2.clone()]))
            }
        }
    }

    // - Delete key from object or element from array
    pub fn delete_key(&self, key: &str) -> Jsonb {
        let mut val = self.0.clone();
        if let serde_json::Value::Object(map) = &mut val {
            map.remove(key);
        } else if let serde_json::Value::Array(arr) = &mut val {
            arr.retain(|v| v != &serde_json::Value::String(key.to_string()));
        }
        Jsonb(val)
    }
}
```

### Phase 5: Generator Functions (Table-Valued Functions)

Functions like `jsonb_array_elements` return a set of rows (SRF).

```rust
// crates/pieskieo-core/src/functions/srf.rs

pub fn jsonb_array_elements(args: &[Expr], ctx: &QueryContext) -> Result<Vec<Value>> {
    let json_val = eval(args[0], ctx)?;
    if let Value::Jsonb(Jsonb(serde_json::Value::Array(arr))) = json_val {
        Ok(arr.into_iter().map(|v| Value::Jsonb(Jsonb(v))).collect())
    } else {
        Err(PieskieoError::InvalidType("Expected JSON array".into()))
    }
}
```

### Phase 6: Parser & Engine Integration

1.  **Lexer/Parser**: Add custom tokens for operators like `->`, `->>`, `#>`, `#>>`, `@>`, `?`.
2.  **Expression Evaluator**: Wire the parsed AST nodes to the `Jsonb` methods.
3.  **Planner**: Recognize `@>` and `?` operators to pick GIN indexes when available.

---

## Test Cases

### Test 1: Extraction Operators
```sql
SELECT '{"a": {"b": ["foo", "bar"]}}'::jsonb -> 'a' -> 'b' ->> 1;
-- Expected: "bar" (text type)

SELECT '{"a": 1, "b": 2}'::jsonb ->> 'a';
-- Expected: "1" (text type)
```

### Test 2: Containment
```sql
SELECT '{"a": 1, "b": 2, "c": {"d": 3}}'::jsonb @> '{"c": {"d": 3}}'::jsonb;
-- Expected: true

SELECT '["a", "b", "c"]'::jsonb @> '["a", "c"]'::jsonb;
-- Expected: true
```

### Test 3: Existence
```sql
SELECT '{"a": 1, "b": 2}'::jsonb ? 'b';
-- Expected: true

SELECT '["a", "b", "c"]'::jsonb ?| array['c', 'd'];
-- Expected: true (exists any)
```

### Test 4: Concatenation and Deletion
```sql
SELECT '{"a": 1}'::jsonb || '{"b": 2}'::jsonb;
-- Expected: '{"a": 1, "b": 2}'

SELECT '{"a": 1, "b": 2}'::jsonb - 'a';
-- Expected: '{"b": 2}'
```

### Test 5: JSON Generation Functions
```sql
SELECT jsonb_build_object('foo', 1, 'bar', true);
-- Expected: '{"foo": 1, "bar": true}'
```

---

## Performance Targets

- **Parsing Latency**: < 100ns per kilobyte of JSON string.
- **Extraction Latency**: < 50ns for shallow paths using zero-copy extraction when possible.

## Metrics to Track

- `pieskieo_jsonb_parse_errors`
- `pieskieo_jsonb_operations_total`

**Created**: 2026-02-08  
**Author**: Implementation Team
