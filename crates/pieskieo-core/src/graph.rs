use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub src: Uuid,
    pub dst: Uuid,
    pub weight: f32,
    pub edge_type: Option<String>,
    pub properties: Option<serde_json::Value>,
}

#[derive(Default, Clone)]
pub struct GraphStore {
    adj: Arc<RwLock<HashMap<Uuid, Vec<Edge>>>>,
    adj_in: Arc<RwLock<HashMap<Uuid, Vec<Edge>>>>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_edge(&self, src: Uuid, dst: Uuid, weight: f32) {
        self.add_typed_edge(src, dst, weight, None, None);
    }

    pub fn add_typed_edge(
        &self,
        src: Uuid,
        dst: Uuid,
        weight: f32,
        edge_type: Option<String>,
        properties: Option<serde_json::Value>,
    ) {
        let edge = Edge {
            src,
            dst,
            weight,
            edge_type: edge_type.clone(),
            properties: properties.clone(),
        };
        {
            let mut adj = self.adj.write();
            let entry = adj.entry(src).or_insert_with(Vec::new);
            if let Some(existing) = entry.iter_mut().find(|e| e.dst == dst) {
                existing.weight = weight;
                existing.edge_type = edge_type.clone();
                existing.properties = properties.clone();
            } else {
                entry.push(edge.clone());
            }
        }
        {
            let mut adj_in = self.adj_in.write();
            let entry = adj_in.entry(dst).or_insert_with(Vec::new);
            if let Some(existing) = entry.iter_mut().find(|e| e.src == src) {
                existing.weight = weight;
            } else {
                entry.push(edge);
            }
        }
    }

    pub fn remove_edge(&self, src: Uuid, dst: Uuid) {
        {
            let mut adj = self.adj.write();
            if let Some(edges) = adj.get_mut(&src) {
                edges.retain(|e| e.dst != dst);
            }
        }
        {
            let mut adj_in = self.adj_in.write();
            if let Some(edges) = adj_in.get_mut(&dst) {
                edges.retain(|e| e.src != src);
            }
        }
    }

    /// Outgoing neighbors (default direction).
    pub fn neighbors(&self, id: Uuid, limit: usize) -> Vec<Edge> {
        let adj = self.adj.read();
        adj.get(&id)
            .map(|edges| edges.iter().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Incoming neighbors.
    pub fn neighbors_in(&self, id: Uuid, limit: usize) -> Vec<Edge> {
        let adj_in = self.adj_in.read();
        adj_in
            .get(&id)
            .map(|edges| edges.iter().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Both incoming and outgoing neighbors.
    pub fn neighbors_both(&self, id: Uuid, limit: usize) -> Vec<Edge> {
        let mut edges = self.neighbors(id, limit);
        let incoming = self.neighbors_in(id, limit);
        edges.extend(incoming);
        edges.truncate(limit);
        edges
    }

    pub fn bfs(&self, start: Uuid, limit: usize) -> Vec<Edge> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut out = Vec::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(node) = queue.pop_front() {
            for e in self.neighbors(node, limit) {
                if visited.insert(e.dst) {
                    out.push(e.clone());
                    if out.len() >= limit {
                        return out;
                    }
                    queue.push_back(e.dst);
                }
            }
        }
        out
    }

    pub fn dfs(&self, start: Uuid, limit: usize) -> Vec<Edge> {
        let mut visited = std::collections::HashSet::new();
        let mut stack = Vec::new();
        let mut out = Vec::new();
        stack.push(start);
        while let Some(node) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }
            for e in self.neighbors(node, limit).into_iter().rev() {
                out.push(e.clone());
                if out.len() >= limit {
                    return out;
                }
                stack.push(e.dst);
            }
        }
        out
    }

    /// Simple PageRank (power iteration).
    /// edge_type: if Some, only consider edges with matching edge_type.
    pub fn pagerank(
        &self,
        _edge_type: Option<&str>,
        iterations: usize,
        damping: f64,
    ) -> HashMap<Uuid, f64> {
        let adj = self.adj.read();
        let nodes: Vec<Uuid> = adj.keys().cloned().collect();
        if nodes.is_empty() {
            return HashMap::new();
        }
        let n = nodes.len();
        let mut rank: HashMap<Uuid, f64> = nodes.iter().map(|&id| (id, 1.0 / n as f64)).collect();
        for _ in 0..iterations {
            let mut new_rank: HashMap<Uuid, f64> = nodes
                .iter()
                .map(|&id| (id, (1.0 - damping) / n as f64))
                .collect();
            for (&src, edges) in adj.iter() {
                let filtered: Vec<_> = edges
                    .iter()
                    .filter(|e| _edge_type.map_or(true, |et| e.edge_type.as_deref() == Some(et)))
                    .collect();
                let out_degree = filtered.len() as f64;
                if out_degree > 0.0 {
                    let contribution =
                        damping * rank.get(&src).cloned().unwrap_or(0.0) / out_degree;
                    for e in filtered {
                        *new_rank.entry(e.dst).or_insert(0.0) += contribution;
                    }
                }
            }
            rank = new_rank;
        }
        rank
    }

    /// Connected components (undirected). Returns map of node -> component_id.
    pub fn connected_components(&self, _edge_type: Option<&str>) -> HashMap<Uuid, usize> {
        // Collect all edges first to avoid holding multiple read locks
        let all_edges: Vec<Edge> = {
            let adj = self.adj.read();
            adj.values()
                .flat_map(|edges| edges.iter().cloned())
                .collect()
        };
        // Build undirected adjacency from edges
        let mut undirected: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for e in &all_edges {
            if _edge_type.map_or(true, |et| e.edge_type.as_deref() == Some(et)) {
                undirected.entry(e.src).or_default().push(e.dst);
                undirected.entry(e.dst).or_default().push(e.src);
            }
        }
        let mut visited: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        let mut result: HashMap<Uuid, usize> = HashMap::new();
        let mut comp_id = 0usize;
        let all_nodes: Vec<Uuid> = undirected.keys().cloned().collect();
        for node in all_nodes {
            if !visited.contains(&node) {
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(node);
                visited.insert(node);
                while let Some(n) = queue.pop_front() {
                    result.insert(n, comp_id);
                    if let Some(neighbors) = undirected.get(&n) {
                        for &nb in neighbors {
                            if visited.insert(nb) {
                                queue.push_back(nb);
                            }
                        }
                    }
                }
                comp_id += 1;
            }
        }
        result
    }

    /// Betweenness centrality.
    pub fn betweenness_centrality(
        &self,
        _edge_type: Option<&str>,
        _normalized: bool,
    ) -> HashMap<Uuid, f64> {
        let adj = self.adj.read();
        let nodes: Vec<Uuid> = adj.keys().cloned().collect();
        let mut centrality: HashMap<Uuid, f64> = nodes.iter().map(|&id| (id, 0.0)).collect();
        for &src in &nodes {
            let mut dist: HashMap<Uuid, i64> = HashMap::new();
            let mut sigma: HashMap<Uuid, f64> = HashMap::new();
            let mut pred: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
            let mut queue = std::collections::VecDeque::new();
            dist.insert(src, 0);
            sigma.insert(src, 1.0);
            queue.push_back(src);
            let mut stack = Vec::new();
            while let Some(v) = queue.pop_front() {
                stack.push(v);
                if let Some(edges) = adj.get(&v) {
                    for e in edges {
                        if !_edge_type.map_or(true, |et| e.edge_type.as_deref() == Some(et)) {
                            continue;
                        }
                        let w = e.dst;
                        if !dist.contains_key(&w) {
                            queue.push_back(w);
                            dist.insert(w, dist[&v] + 1);
                        }
                        if dist[&w] == dist[&v] + 1 {
                            *sigma.entry(w).or_insert(0.0) += sigma.get(&v).cloned().unwrap_or(0.0);
                            pred.entry(w).or_insert_with(Vec::new).push(v);
                        }
                    }
                }
            }
            let empty = vec![];
            let mut delta: HashMap<Uuid, f64> = nodes.iter().map(|&id| (id, 0.0)).collect();
            while let Some(w) = stack.pop() {
                for &v in pred.get(&w).unwrap_or(&empty) {
                    let coeff = (sigma.get(&v).cloned().unwrap_or(0.0)
                        / sigma.get(&w).cloned().unwrap_or(1.0))
                        * (1.0 + delta.get(&w).cloned().unwrap_or(0.0));
                    *delta.entry(v).or_insert(0.0) += coeff;
                }
                if w != src {
                    *centrality.entry(w).or_insert(0.0) += delta.get(&w).cloned().unwrap_or(0.0);
                }
            }
        }
        if _normalized {
            let n = nodes.len() as f64;
            if n > 2.0 {
                let scale = 1.0 / ((n - 1.0) * (n - 2.0));
                for v in centrality.values_mut() {
                    *v *= scale;
                }
            }
        }
        centrality
    }

    /// Closeness centrality.
    pub fn closeness_centrality(
        &self,
        _edge_type: Option<&str>,
        _normalized: bool,
    ) -> HashMap<Uuid, f64> {
        let adj = self.adj.read();
        let nodes: Vec<Uuid> = adj.keys().cloned().collect();
        let n = nodes.len();
        let mut centrality = HashMap::new();
        for &src in &nodes {
            let mut dist: HashMap<Uuid, usize> = HashMap::new();
            let mut queue = std::collections::VecDeque::new();
            dist.insert(src, 0);
            queue.push_back(src);
            while let Some(v) = queue.pop_front() {
                if let Some(edges) = adj.get(&v) {
                    for e in edges {
                        if !_edge_type.map_or(true, |et| e.edge_type.as_deref() == Some(et)) {
                            continue;
                        }
                        if !dist.contains_key(&e.dst) {
                            dist.insert(e.dst, dist[&v] + 1);
                            queue.push_back(e.dst);
                        }
                    }
                }
            }
            let total_dist: usize = dist.values().sum();
            let reachable = dist.len();
            let cc = if total_dist > 0 && reachable > 1 {
                (reachable - 1) as f64 / total_dist as f64
                    * if _normalized {
                        (reachable - 1) as f64 / (n - 1).max(1) as f64
                    } else {
                        1.0
                    }
            } else {
                0.0
            };
            centrality.insert(src, cc);
        }
        centrality
    }

    /// Louvain-style community detection. Returns map of node -> community_id.
    pub fn louvain_communities(&self, _edge_type: Option<&str>) -> HashMap<Uuid, usize> {
        let adj = self.adj.read();
        let nodes: Vec<Uuid> = adj.keys().cloned().collect();
        if nodes.is_empty() {
            return HashMap::new();
        }
        let mut community: HashMap<Uuid, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        for &node in &nodes {
            let mut community_counts: HashMap<usize, usize> = HashMap::new();
            if let Some(edges) = adj.get(&node) {
                for e in edges {
                    if !_edge_type.map_or(true, |et| e.edge_type.as_deref() == Some(et)) {
                        continue;
                    }
                    if let Some(&c) = community.get(&e.dst) {
                        *community_counts.entry(c).or_insert(0) += 1;
                    }
                }
            }
            if let Some((&best_c, _)) = community_counts.iter().max_by_key(|(_, &cnt)| cnt) {
                community.insert(node, best_c);
            }
        }
        community
    }

    /// All edges in the graph.
    pub fn all_edges(&self) -> Vec<Edge> {
        let adj = self.adj.read();
        adj.values()
            .flat_map(|edges| edges.iter().cloned())
            .collect()
    }

    /// All node IDs in the graph.
    pub fn all_nodes(&self) -> Vec<Uuid> {
        let adj = self.adj.read();
        adj.keys().cloned().collect()
    }
}
