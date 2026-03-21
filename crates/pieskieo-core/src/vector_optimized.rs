// Optimized vector module - fixes memory leaks, unsafe code, and performance issues
// This will replace vector.rs after testing

use crate::error::{PieskieoError, Result};
use hnsw_rs::{hnswio::HnswIo, prelude::*};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub id: Uuid,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VectorMetric {
    L2,
    Cosine,
    Dot,
}

/// Owned vector storage that can be properly deallocated
/// Replaces the leaked 'static slices with proper memory management
struct VectorStorage {
    data: Vec<f32>,
    dim: usize,
}

impl VectorStorage {
    fn new(data: Vec<f32>, dim: usize) -> Self {
        Self { data, dim }
    }
    
    fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// In-memory vector store + optional HNSW ANN accelerator.
/// OPTIMIZED: Proper memory management, no leaks, better performance
pub struct VectorIndex {
