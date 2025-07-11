use pyo3::prelude::*;
use serde::{Serialize, Deserialize};

/// Distance enum that supports both built-in and custom metrics.
#[pyclass]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Distance {
    /// Euclidean (L2)
    Euclidean(),
    /// Cosine
    Cosine(),
    /// Manhattan (L1)
    Manhattan(),
    /// Chebyshev (L∞)
    Chebyshev(),
    /// Custom metric identified by name
    Custom(String),
}

#[pymethods]
impl Distance {
    #[classattr] pub const EUCLIDEAN: Distance = Distance::Euclidean();
    #[classattr] pub const COSINE:    Distance = Distance::Cosine();
    #[classattr] pub const MANHATTAN: Distance = Distance::Manhattan();
    #[classattr] pub const CHEBYSHEV: Distance = Distance::Chebyshev();

    #[new]
    pub fn new(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "euclidean" => Distance::Euclidean(),
            "cosine" => Distance::Cosine(),
            "manhattan" => Distance::Manhattan(),
            "chebyshev" => Distance::Chebyshev(),
            _ => Distance::Custom(name.to_string()),
        }
    }

    /// Create a custom distance metric.
    #[staticmethod]
    pub fn custom(name: &str) -> Self {
        Distance::Custom(name.to_string())
    }

    /// Get the name of the distance metric.
    pub fn name(&self) -> String {
        match self {
            Distance::Euclidean() => "euclidean".to_string(),
            Distance::Cosine() => "cosine".to_string(),
            Distance::Manhattan() => "manhattan".to_string(),
            Distance::Chebyshev() => "chebyshev".to_string(),
            Distance::Custom(name) => name.clone(),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Distance::Euclidean() => "Distance.EUCLIDEAN".to_string(),
            Distance::Cosine()    => "Distance.COSINE".to_string(),
            Distance::Manhattan() => "Distance.MANHATTAN".to_string(),
            Distance::Chebyshev() => "Distance.CHEBYSHEV".to_string(),
            Distance::Custom(name) => format!("Distance.custom('{}')", name),
        }
    }
}

impl Distance {
    /// Check if this distance metric is a custom metric
    pub fn is_custom(&self) -> bool {
        matches!(self, Distance::Custom(_))
    }
    
    /// Get the metric name for use with the registry
    pub fn registry_name(&self) -> String {
        match self {
            Distance::Euclidean() => "euclidean".to_string(),
            Distance::Cosine() => "cosine".to_string(),
            Distance::Manhattan() => "manhattan".to_string(),
            Distance::Chebyshev() => "chebyshev".to_string(),
            Distance::Custom(name) => name.clone(),
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
