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
    /// Minkowski (Lp norm)
    Minkowski(f32),
    /// Hamming (bit difference count)
    Hamming(),
    /// Jaccard (set dissimilarity)
    Jaccard(),
    /// Angular (angle in radians)
    Angular(),
    /// Canberra (weighted Manhattan)
    Canberra(),
    /// Custom metric identified by name
    Custom(String),
}

#[pymethods]
impl Distance {
    #[classattr] pub const EUCLIDEAN: Distance = Distance::Euclidean();
    #[classattr] pub const COSINE:    Distance = Distance::Cosine();
    #[classattr] pub const MANHATTAN: Distance = Distance::Manhattan();
    #[classattr] pub const CHEBYSHEV: Distance = Distance::Chebyshev();
    #[classattr] pub const HAMMING:   Distance = Distance::Hamming();
    #[classattr] pub const JACCARD:   Distance = Distance::Jaccard();
    #[classattr] pub const ANGULAR:   Distance = Distance::Angular();
    #[classattr] pub const CANBERRA:  Distance = Distance::Canberra();

    #[new]
    pub fn new(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "euclidean" => Distance::Euclidean(),
            "cosine"    => Distance::Cosine(),
            "manhattan" => Distance::Manhattan(),
            "chebyshev" => Distance::Chebyshev(),
            "hamming"   => Distance::Hamming(),
            "jaccard"   => Distance::Jaccard(),
            "angular"   => Distance::Angular(),
            "canberra"  => Distance::Canberra(),
            _            => Distance::Custom(name.to_string()),
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
            Distance::Euclidean()    => "euclidean".to_string(),
            Distance::Cosine()       => "cosine".to_string(),
            Distance::Manhattan()    => "manhattan".to_string(),
            Distance::Chebyshev()    => "chebyshev".to_string(),
            Distance::Minkowski(p)   => format!("minkowski({})", p),
            Distance::Hamming()      => "hamming".to_string(),
            Distance::Jaccard()      => "jaccard".to_string(),
            Distance::Angular()      => "angular".to_string(),
            Distance::Canberra()     => "canberra".to_string(),
            Distance::Custom(name)   => name.clone(),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            Distance::Euclidean()    => "Distance.EUCLIDEAN".to_string(),
            Distance::Cosine()       => "Distance.COSINE".to_string(),
            Distance::Manhattan()    => "Distance.MANHATTAN".to_string(),
            Distance::Chebyshev()    => "Distance.CHEBYSHEV".to_string(),
            Distance::Minkowski(p)   => format!("Distance.Minkowski({})", p),
            Distance::Hamming()      => "Distance.HAMMING".to_string(),
            Distance::Jaccard()      => "Distance.JACCARD".to_string(),
            Distance::Angular()      => "Distance.ANGULAR".to_string(),
            Distance::Canberra()     => "Distance.CANBERRA".to_string(),
            Distance::Custom(name)   => format!("Distance.custom('{}')", name),
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
            Distance::Euclidean()    => "euclidean".to_string(),
            Distance::Cosine()       => "cosine".to_string(),
            Distance::Manhattan()    => "manhattan".to_string(),
            Distance::Chebyshev()    => "chebyshev".to_string(),
            Distance::Minkowski(p)   => format!("minkowski({})", p),
            Distance::Hamming()      => "hamming".to_string(),
            Distance::Jaccard()      => "jaccard".to_string(),
            Distance::Angular()      => "angular".to_string(),
            Distance::Canberra()     => "canberra".to_string(),
            Distance::Custom(name)   => name.clone(),
        }
    }

    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Distance::Euclidean()  => euclidean(a, b),
            Distance::Cosine()     => cosine(a, b),
            Distance::Manhattan()  => manhattan(a, b),
            Distance::Chebyshev()  => chebyshev(a, b),
            Distance::Minkowski(p) => minkowski(a, b, *p),
            Distance::Hamming()    => hamming(a, b),
            Distance::Jaccard()    => jaccard(a, b),
            Distance::Angular()    => angular(a, b),
            Distance::Canberra()   => canberra(a, b),
            Distance::Custom(_)    => panic!("Custom metrics should be handled separately"),
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

pub fn minkowski(a: &[f32], b: &[f32], p: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    if p == 1.0 || p == 2.0 || p == f32::INFINITY {
        let mut acc = 0.0;
        if p == 1.0 {
            for (x, y) in a.iter().zip(b) { acc += (x - y).abs(); }
            return acc;
        } else if p == 2.0 {
            for (x, y) in a.iter().zip(b) { acc += (x - y).powi(2); }
            return acc.sqrt();
        } else {
            for (x, y) in a.iter().zip(b) {
                let diff = (x - y).abs();
                if diff > acc { acc = diff; }
            }
            return acc;
        }
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p)
}

pub fn hamming(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    a.iter()
        .zip(b)
        .map(|(x, y)| if (x - y).abs() > 1e-5 { 1.0 } else { 0.0 })
        .sum()
}

pub fn jaccard(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    let (intersection, union) = a.iter().zip(b).fold((0.0, 0.0), |(i, u), (x, y)| {
        let x_bin = *x > 0.5;
        let y_bin = *y > 0.5;
        (
            i + if x_bin && y_bin { 1.0 } else { 0.0 },
            u + if x_bin || y_bin { 1.0 } else { 0.0 },
        )
    });
    if union == 0.0 { 0.0 } else { 1.0 - (intersection / union) }
}

pub fn angular(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return std::f32::consts::FRAC_PI_2; }
    let cosine_sim = dot_product / (norm_a * norm_b);
    cosine_sim.clamp(-1.0, 1.0).acos()
}

pub fn canberra(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let diff = (x - y).abs();
            let denom = x.abs() + y.abs();
            if denom > 0.0 { diff / denom } else { 0.0 }
        })
        .sum()
}

pub fn euclidean_sq(a: &[f32], b: &[f32], a_sq: f32, b_sq: f32) -> f32 {
    let dot = dot_product(a, b);
    a_sq + b_sq - 2.0 * dot
}

pub fn angular_distance(a: &[f32], b: &[f32], a_sq: f32, b_sq: f32) -> f32 {
    let dot = dot_product(a, b);
    let a_norm = a_sq.sqrt();
    let b_norm = b_sq.sqrt();
    1.0 - (dot / (a_norm * b_norm))
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}