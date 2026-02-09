# Weaviate Feature: Scalar Quantization

**Feature ID**: `weaviate/19-scalar-quantization.md`
**Status**: Production-Ready Design

## Overview

Scalar Quantization (SQ) compresses vectors by quantizing float32 to int8, achieving 4x compression with minimal accuracy loss.

## Implementation

```rust
pub struct ScalarQuantizer {
    min: Vec<f32>,
    max: Vec<f32>,
    dim: usize,
}

impl ScalarQuantizer {
    pub fn train(vectors: &[Vec<f32>]) -> Self {
        let dim = vectors[0].len();
        let mut min = vec![f32::MAX; dim];
        let mut max = vec![f32::MIN; dim];
        
        for vec in vectors {
            for (i, &val) in vec.iter().enumerate() {
                min[i] = min[i].min(val);
                max[i] = max[i].max(val);
            }
        }
        
        Self { min, max, dim }
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<i8> {
        vector.iter().enumerate().map(|(i, &val)| {
            let normalized = (val - self.min[i]) / (self.max[i] - self.min[i]);
            ((normalized * 255.0) as i32 - 128).clamp(-128, 127) as i8
        }).collect()
    }

    pub fn decode(&self, codes: &[i8]) -> Vec<f32> {
        codes.iter().enumerate().map(|(i, &code)| {
            let normalized = (code as f32 + 128.0) / 255.0;
            self.min[i] + normalized * (self.max[i] - self.min[i])
        }).collect()
    }

    pub fn distance_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| {
                let diff = (x as i32 - y as i32) as f32;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }
}
```

## Performance Targets
- Encoding: < 10µs per vector
- Compression: 4x (32-bit → 8-bit)
- Accuracy: > 98% recall@10

## Status
**Complete**: Production-ready SQ with min/max normalization
