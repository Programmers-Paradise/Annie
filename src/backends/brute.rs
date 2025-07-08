use crate::distance::{Distance, euclidean, cosine, manhattan, chebyshev};
use super::ann_backend::AnnBackend;
use rust_annie_macros::py_annindex;

#[py_annindex(backend = "BruteForce", distance = "Euclidean")]
pub struct BruteForceIndex {
    vectors: Vec<Vec<f32>>,
    distance: Distance,
}

impl BruteForceIndex {
    pub fn new(distance: Distance) -> Self {
        Self { vectors: Vec::new(), distance }
    }
}

impl AnnBackend for BruteForceIndex {
    fn add(&mut self, vector: Vec<f32>) {
        self.vectors.push(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<_> = self.vectors.iter().enumerate().map(|(i, v)| {
            let d = match self.distance {
                Distance::Euclidean => euclidean(query, v),
                Distance::Cosine    => cosine(query, v),
                Distance::Manhattan => manhattan(query, v),
                Distance::Chebyshev => chebyshev(query, v),
            };
            (i, d)
        }).collect();

        // SAFE: Use total_cmp for NaN-resistant sorting
        scored.retain(|(_, d)| !d.is_nan());
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}
