# Kùzu Feature: CSR Graph Storage

**Feature ID**: `kuzu/20-csr-storage.md`
**Status**: Production-Ready Design

## Overview

Compressed Sparse Row (CSR) format provides memory-efficient graph storage with O(1) neighbor access and excellent cache locality.

## Implementation

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CSRGraph {
    /// Offset array: offsets[i] = start index of node i's neighbors
    offsets: Vec<usize>,
    /// Neighbor array: flat list of all neighbors
    neighbors: Vec<u64>,
    /// Edge properties (parallel to neighbors array)
    edge_props: Vec<EdgeProperties>,
    /// Number of nodes
    num_nodes: usize,
    /// Number of edges
    num_edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeProperties {
    pub weight: f32,
    pub edge_type: String,
}

impl CSRGraph {
    pub fn new(edges: &[(u64, u64, EdgeProperties)], num_nodes: usize) -> Self {
        // Build adjacency list
        let mut adj_list: Vec<Vec<(u64, EdgeProperties)>> = vec![Vec::new(); num_nodes];
        
        for &(src, dst, ref props) in edges {
            adj_list[src as usize].push((dst, props.clone()));
        }

        // Convert to CSR format
        let mut offsets = vec![0];
        let mut neighbors = Vec::new();
        let mut edge_props = Vec::new();

        for node_neighbors in adj_list {
            for (neighbor, props) in node_neighbors {
                neighbors.push(neighbor);
                edge_props.push(props);
            }
            offsets.push(neighbors.len());
        }

        let num_edges = neighbors.len();

        Self {
            offsets,
            neighbors,
            edge_props,
            num_nodes,
            num_edges,
        }
    }

    pub fn get_neighbors(&self, node: u64) -> &[u64] {
        let start = self.offsets[node as usize];
        let end = self.offsets[node as usize + 1];
        &self.neighbors[start..end]
    }

    pub fn get_edges(&self, node: u64) -> Vec<(u64, &EdgeProperties)> {
        let start = self.offsets[node as usize];
        let end = self.offsets[node as usize + 1];
        
        self.neighbors[start..end].iter()
            .zip(self.edge_props[start..end].iter())
            .map(|(&neighbor, props)| (neighbor, props))
            .collect()
    }

    pub fn degree(&self, node: u64) -> usize {
        self.offsets[node as usize + 1] - self.offsets[node as usize]
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<usize>() * self.offsets.len() +
        std::mem::size_of::<u64>() * self.neighbors.len() +
        std::mem::size_of::<EdgeProperties>() * self.edge_props.len()
    }
}
```

## Performance Targets
- Memory: ~12-16 bytes per edge
- Neighbor lookup: O(1), < 50ns
- Cache efficiency: 90%+
- BFS traversal: 2x faster than adjacency list

## Status
**Complete**: Production-ready CSR with O(1) neighbor access and edge properties
