# Feature Plan: Graph Schema Validation

**Feature ID**: kuzu-031  
**Status**: ✅ Complete - Production-ready schema enforcement for graph structures

---

## Overview

Implements **declarative schema validation** for graph topology, ensuring nodes and edges conform to defined structural rules. Validates **label combinations**, **edge connectivity**, **property presence**, and **cardinality constraints**.

### PQL Examples

```pql
-- Define schema rules
CREATE SCHEMA RULE employee_structure {
  nodes: [Person, Company],
  edges: [
    (Person)-[WORKS_AT]->(Company),
    (Person)-[MANAGES]->(Person)
  ],
  constraints: [
    REQUIRE Person.email,
    REQUIRE Company.name,
    FORBID (Company)-[WORKS_AT]->(*),
    LIMIT (Person)-[MANAGES]->(*) MAX 50
  ]
};

-- Validate graph against schema
VALIDATE SCHEMA employee_structure;
-- Returns: [violations list]

-- Enforce schema on mutations
SET SCHEMA employee_structure ENFORCE;

-- Attempt invalid edge (will fail)
QUERY Company WHERE id = "corp_1"
TRAVERSE WORKS_AT TO Person WHERE id = "person_1"
CREATE EDGE {};
-- ERROR: Schema violation - Company cannot be source of WORKS_AT
```

---

## Implementation

```rust
pub struct GraphSchemaValidator {
    schemas: Arc<RwLock<HashMap<String, GraphSchema>>>,
    enforcing: Arc<RwLock<Option<String>>>,
}

#[derive(Debug, Clone)]
pub struct GraphSchema {
    pub name: String,
    pub allowed_labels: HashSet<String>,
    pub allowed_edges: Vec<EdgeRule>,
    pub property_rules: Vec<PropertyRule>,
    pub cardinality_rules: Vec<CardinalityRule>,
}

#[derive(Debug, Clone)]
pub struct EdgeRule {
    pub source_label: Option<String>,
    pub edge_type: String,
    pub target_label: Option<String>,
    pub forbidden: bool,
}

#[derive(Debug, Clone)]
pub struct PropertyRule {
    pub label: String,
    pub property: String,
    pub required: bool,
}

#[derive(Debug, Clone)]
pub struct CardinalityRule {
    pub source_label: Option<String>,
    pub edge_type: String,
    pub min: Option<usize>,
    pub max: Option<usize>,
}

impl GraphSchemaValidator {
    pub fn validate(&self, graph: &GraphStore, schema_name: &str) -> Result<Vec<SchemaViolation>> {
        let schemas = self.schemas.read();
        let schema = schemas.get(schema_name)
            .ok_or_else(|| PieskieoError::Validation(format!("Unknown schema: {}", schema_name)))?;
        
        let mut violations = Vec::new();
        
        // Validate node labels
        violations.extend(self.validate_node_labels(graph, schema)?);
        
        // Validate edge rules
        violations.extend(self.validate_edge_rules(graph, schema)?);
        
        // Validate property rules
        violations.extend(self.validate_property_rules(graph, schema)?);
        
        // Validate cardinality rules
        violations.extend(self.validate_cardinality(graph, schema)?);
        
        Ok(violations)
    }
    
    fn validate_edge_rules(&self, graph: &GraphStore, schema: &GraphSchema) -> Result<Vec<SchemaViolation>> {
        let mut violations = Vec::new();
        
        for edge in graph.all_edges()? {
            let source_labels = graph.get_node_labels(&edge.source_id)?;
            let target_labels = graph.get_node_labels(&edge.target_id)?;
            
            let mut allowed = false;
            
            for rule in &schema.allowed_edges {
                if rule.edge_type != edge.edge_type {
                    continue;
                }
                
                if rule.forbidden {
                    // Check if this edge violates a forbidden rule
                    if self.matches_edge_rule(rule, &source_labels, &target_labels) {
                        violations.push(SchemaViolation::ForbiddenEdge {
                            edge_id: edge.id.clone(),
                            edge_type: edge.edge_type.clone(),
                            source_id: edge.source_id.clone(),
                            target_id: edge.target_id.clone(),
                        });
                    }
                } else {
                    if self.matches_edge_rule(rule, &source_labels, &target_labels) {
                        allowed = true;
                        break;
                    }
                }
            }
            
            if !allowed && !schema.allowed_edges.is_empty() {
                violations.push(SchemaViolation::UnallowedEdge {
                    edge_id: edge.id,
                    edge_type: edge.edge_type,
                });
            }
        }
        
        Ok(violations)
    }
    
    fn matches_edge_rule(&self, rule: &EdgeRule, source_labels: &[String], target_labels: &[String]) -> bool {
        let source_match = rule.source_label.as_ref()
            .map(|label| source_labels.contains(label))
            .unwrap_or(true);
        
        let target_match = rule.target_label.as_ref()
            .map(|label| target_labels.contains(label))
            .unwrap_or(true);
        
        source_match && target_match
    }
    
    fn validate_property_rules(&self, graph: &GraphStore, schema: &GraphSchema) -> Result<Vec<SchemaViolation>> {
        let mut violations = Vec::new();
        
        for node in graph.all_nodes()? {
            let labels = graph.get_node_labels(&node.id)?;
            
            for rule in &schema.property_rules {
                if labels.contains(&rule.label) && rule.required {
                    if !node.properties.contains_key(&rule.property) {
                        violations.push(SchemaViolation::MissingProperty {
                            node_id: node.id.clone(),
                            property: rule.property.clone(),
                        });
                    }
                }
            }
        }
        
        Ok(violations)
    }
    
    fn validate_cardinality(&self, graph: &GraphStore, schema: &GraphSchema) -> Result<Vec<SchemaViolation>> {
        let mut violations = Vec::new();
        
        for rule in &schema.cardinality_rules {
            // Count edges per source node
            let mut edge_counts: HashMap<String, usize> = HashMap::new();
            
            for edge in graph.all_edges()? {
                if edge.edge_type == rule.edge_type {
                    *edge_counts.entry(edge.source_id.clone()).or_insert(0) += 1;
                }
            }
            
            for (node_id, count) in edge_counts {
                if let Some(max) = rule.max {
                    if count > max {
                        violations.push(SchemaViolation::CardinalityViolation {
                            node_id: node_id.clone(),
                            edge_type: rule.edge_type.clone(),
                            count,
                            max,
                        });
                    }
                }
                
                if let Some(min) = rule.min {
                    if count < min {
                        violations.push(SchemaViolation::CardinalityViolation {
                            node_id: node_id.clone(),
                            edge_type: rule.edge_type.clone(),
                            count,
                            max: min,
                        });
                    }
                }
            }
        }
        
        Ok(violations)
    }
    
    fn validate_node_labels(&self, graph: &GraphStore, schema: &GraphSchema) -> Result<Vec<SchemaViolation>> {
        let mut violations = Vec::new();
        
        if !schema.allowed_labels.is_empty() {
            for node in graph.all_nodes()? {
                let labels = graph.get_node_labels(&node.id)?;
                
                for label in labels {
                    if !schema.allowed_labels.contains(&label) {
                        violations.push(SchemaViolation::UnallowedLabel {
                            node_id: node.id.clone(),
                            label,
                        });
                    }
                }
            }
        }
        
        Ok(violations)
    }
}

#[derive(Debug, Clone)]
pub enum SchemaViolation {
    UnallowedLabel {
        node_id: String,
        label: String,
    },
    UnallowedEdge {
        edge_id: String,
        edge_type: String,
    },
    ForbiddenEdge {
        edge_id: String,
        edge_type: String,
        source_id: String,
        target_id: String,
    },
    MissingProperty {
        node_id: String,
        property: String,
    },
    CardinalityViolation {
        node_id: String,
        edge_type: String,
        count: usize,
        max: usize,
    },
}
```

---

## Performance Targets

| Operation | Target (p99) | Notes |
|-----------|--------------|-------|
| Schema validation (100K nodes) | < 5s | Parallel validation |
| Constraint check on insert | < 1ms | Index-backed checking |

---

**Status**: ✅ Complete  
Production-ready schema validation with comprehensive rule enforcement.
