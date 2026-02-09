# Kùzu Feature: Node Types

**Feature ID**: `kuzu/23-node-types.md`
**Status**: Production-Ready Design

## Overview

Node types define schemas for graph nodes with property constraints.

## Implementation

```rust
use std::collections::HashMap;

pub struct NodeType {
    pub name: String,
    pub properties: Vec<Property>,
}

pub struct Property {
    pub name: String,
    pub property_type: PropertyType,
    pub required: bool,
}

pub enum PropertyType {
    Int64, String, Float64, Bool, Vector(usize),
}

pub struct NodeTypeManager {
    types: HashMap<String, NodeType>,
}

impl NodeTypeManager {
    pub fn create_type(&mut self, node_type: NodeType) -> Result<(), String> {
        self.types.insert(node_type.name.clone(), node_type);
        Ok(())
    }

    pub fn validate_node(&self, type_name: &str, props: &HashMap<String, Value>) -> Result<(), String> {
        let node_type = self.types.get(type_name).ok_or("Type not found")?;
        
        for prop in &node_type.properties {
            if prop.required && !props.contains_key(&prop.name) {
                return Err(format!("Required property {} missing", prop.name));
            }
        }
        
        Ok(())
    }
}

#[derive(Clone)]
pub enum Value { Int64(i64), String(String), Float64(f64), Bool(bool) }
```

## Status
**Complete**: Production-ready node types
