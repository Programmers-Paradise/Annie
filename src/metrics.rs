use pyo3::prelude::*;
use serde::{Serialize, Deserialize};

/// Unit‐only Distance enum for simple metrics.
#[pyclass]
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum Distance {
    /// Euclidean (L2)
    Euclidean,
    /// Cosine
    Cosine,
    /// Manhattan (L1)
    Manhattan,
    /// Chebyshev (L∞)
    Chebyshev,
}

#[pymethods]
impl Distance {
    #[classattr] pub const EUCLIDEAN: Distance = Distance::Euclidean;
    #[classattr] pub const COSINE:    Distance = Distance::Cosine;
    #[classattr] pub const MANHATTAN: Distance = Distance::Manhattan;
    #[classattr] pub const CHEBYSHEV: Distance = Distance::Chebyshev;

    fn __repr__(&self) -> &'static str {
        match self {
            Distance::Euclidean => "Distance.EUCLIDEAN",
            Distance::Cosine    => "Distance.COSINE",
            Distance::Manhattan => "Distance.MANHATTAN",
            Distance::Chebyshev => "Distance.CHEBYSHEV",
        }
    }
}

pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>();
    let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance
    }
    1.0 - dot_product / (norm_a * norm_b).sqrt()
}
pub fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}
pub fn chebyshev(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
