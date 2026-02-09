# Feature Plan: Node Label Management

**Feature ID**: kuzu-030  
**Status**: ✅ Complete - Production-ready multi-label node system with dynamic labeling

---

## Overview

Implements **multi-label node system** allowing nodes to have multiple type labels simultaneously (e.g., a Person can also be an Employee and Manager). Provides **dynamic label addition/removal** and **label-based indexing** for efficient queries.

### PQL Examples

```pql
-- Create node with multiple labels
QUERY Person:Employee:Manager
CREATE NODE {
  id: "emp_001",
  name: "Alice",
  department: "Engineering",
  team_size: 10
};

-- Query by label
QUERY Person:Employee WHERE department = "Engineering"
SELECT id, name;

-- Add label dynamically
QUERY Person WHERE id = "emp_001"
ADD LABEL Executive;

-- Remove label
QUERY Person:Manager WHERE id = "emp_001"
REMOVE LABEL Manager;

-- Query nodes with any of multiple labels
QUERY Person:Employee | Person:Contractor
WHERE active = true
SELECT id, name, LABELS() AS node_labels;
```

---

## Implementation

```rust
use std::collections::HashSet;

pub struct NodeLabelManager {
    /// Maps node_id -> set of labels
    node_labels: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    
    /// Reverse index: label -> set of node_ids
    label_index: Arc<RwLock<HashMap<String, HashSet<String>>>>,
}

impl NodeLabelManager {
    pub fn new() -> Self {
        Self {
            node_labels: Arc::new(RwLock::new(HashMap::new())),
            label_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn add_label(&self, node_id: &str, label: &str) -> Result<()> {
        let mut node_labels = self.node_labels.write();
        let mut label_index = self.label_index.write();
        
        node_labels.entry(node_id.to_string())
            .or_insert_with(HashSet::new)
            .insert(label.to_string());
        
        label_index.entry(label.to_string())
            .or_insert_with(HashSet::new)
            .insert(node_id.to_string());
        
        Ok(())
    }
    
    pub fn remove_label(&self, node_id: &str, label: &str) -> Result<()> {
        let mut node_labels = self.node_labels.write();
        let mut label_index = self.label_index.write();
        
        if let Some(labels) = node_labels.get_mut(node_id) {
            labels.remove(label);
        }
        
        if let Some(nodes) = label_index.get_mut(label) {
            nodes.remove(node_id);
        }
        
        Ok(())
    }
    
    pub fn has_label(&self, node_id: &str, label: &str) -> bool {
        let node_labels = self.node_labels.read();
        node_labels.get(node_id)
            .map(|labels| labels.contains(label))
            .unwrap_or(false)
    }
    
    pub fn get_labels(&self, node_id: &str) -> Vec<String> {
        let node_labels = self.node_labels.read();
        node_labels.get(node_id)
            .map(|labels| labels.iter().cloned().collect())
            .unwrap_or_default()
    }
    
    pub fn get_nodes_with_label(&self, label: &str) -> Vec<String> {
        let label_index = self.label_index.read();
        label_index.get(label)
            .map(|nodes| nodes.iter().cloned().collect())
            .unwrap_or_default()
    }
    
    pub fn get_nodes_with_all_labels(&self, labels: &[String]) -> Vec<String> {
        if labels.is_empty() {
            return vec![];
        }
        
        let label_index = self.label_index.read();
        
        // Start with nodes having first label
        let mut result: HashSet<String> = label_index.get(&labels[0])
            .cloned()
            .unwrap_or_default();
        
        // Intersect with nodes having each subsequent label
        for label in &labels[1..] {
            if let Some(nodes) = label_index.get(label) {
                result = result.intersection(nodes).cloned().collect();
            } else {
                return vec![];
            }
        }
        
        result.into_iter().collect()
    }
    
    pub fn get_nodes_with_any_label(&self, labels: &[String]) -> Vec<String> {
        let label_index = self.label_index.read();
        let mut result = HashSet::new();
        
        for label in labels {
            if let Some(nodes) = label_index.get(label) {
                result.extend(nodes.iter().cloned());
            }
        }
        
        result.into_iter().collect()
    }
}
```

---

## Performance Targets

| Operation | Target (p99) | Notes |
|-----------|--------------|-------|
| Add label | < 10μs | HashSet insert with RwLock |
| Remove label | < 10μs | HashSet remove |
| Check label | < 5μs | HashSet lookup |
| Query by label | < 1ms per 1K nodes | Index scan |

---

**Status**: ✅ Complete  
Production-ready multi-label system with dynamic labeling and efficient label-based queries.
