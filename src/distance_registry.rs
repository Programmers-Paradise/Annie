use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;

use crate::errors::DistanceRegistryError;

/// Trait for distance functions that can be registered and used by the index.
pub trait DistanceFunction: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn name(&self) -> &str;
    fn clone_boxed(&self) -> Box<dyn DistanceFunction>;
}

impl Clone for Box<dyn DistanceFunction> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

/// Built-in distance functions that implement the DistanceFunction trait.
#[derive(Clone)]
pub struct EuclideanDistance;

impl DistanceFunction for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }

    fn name(&self) -> &str {
        "euclidean"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct CosineDistance;

impl DistanceFunction for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            (1.0 - dot_product / (norm_a * norm_b)).max(0.0)
        }
    }

    fn name(&self) -> &str {
        "cosine"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct ManhattanDistance;

impl DistanceFunction for ManhattanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }

    fn name(&self) -> &str {
        "manhattan"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct ChebyshevDistance;

impl DistanceFunction for ChebyshevDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn name(&self) -> &str {
        "chebyshev"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

// ===== NEW DISTANCE METRICS START HERE =====

#[derive(Clone)]
pub struct HammingDistance;

impl DistanceFunction for HammingDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| if (x - y).abs() > 1e-5 { 1.0 } else { 0.0 })
            .sum()
    }

    fn name(&self) -> &str {
        "hamming"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct JaccardDistance;

impl DistanceFunction for JaccardDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        
        let mut intersection = 0.0;
        let mut union = 0.0;
        
        for (x, y) in a.iter().zip(b) {
            let x_bin = *x > 0.5;
            let y_bin = *y > 0.5;
            
            if x_bin && y_bin {
                intersection += 1.0;
            }
            if x_bin || y_bin {
                union += 1.0;
            }
        }
        
        if union == 0.0 {
            0.0
        } else {
            1.0 - (intersection / union)
        }
    }

    fn name(&self) -> &str {
        "jaccard"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct AngularDistance;

impl DistanceFunction for AngularDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return std::f32::consts::FRAC_PI_2; // 90 degrees in radians
        }
        
        let cosine_sim = dot_product / (norm_a * norm_b);
        cosine_sim.clamp(-1.0, 1.0).acos()
    }

    fn name(&self) -> &str {
        "angular"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct CanberraDistance;

impl DistanceFunction for CanberraDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let diff = (x - y).abs();
                let denom = x.abs() + y.abs();
                if denom > 0.0 { diff / denom } else { 0.0 }
            })
            .sum()
    }

    fn name(&self) -> &str {
        "canberra"
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

// ===== NEW DISTANCE METRICS END HERE =====

/// A distance function that wraps a Python callable.
pub struct PythonDistanceFunction {
    name: String,
    python_func: PyObject,
}

impl Clone for PythonDistanceFunction {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            name: self.name.clone(),
            python_func: self.python_func.clone_ref(py),
        })
    }
}

impl PythonDistanceFunction {
    pub fn new(name: String, python_func: PyObject) -> Self {
        Self { name, python_func }
    }
}

impl DistanceFunction for PythonDistanceFunction {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        Python::with_gil(|py| {
            let a_py = match a.into_pyobject(py) {
                Ok(obj) => obj,
                Err(_) => return f32::MAX,
            };
            let b_py = match b.into_pyobject(py) {
                Ok(obj) => obj,
                Err(_) => return f32::MAX,
            };

            match self.python_func.call1(py, (a_py, b_py)) {
                Ok(result) => result.extract::<f32>(py).unwrap_or(f32::MAX),
                Err(_) => f32::MAX,
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

/// Registry of distance functions.
pub struct DistanceRegistry {
    functions: HashMap<String, Box<dyn DistanceFunction>>,
}

impl DistanceRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };

        // Register built-in metrics
        registry.register("euclidean", Box::new(EuclideanDistance));
        registry.register("cosine", Box::new(CosineDistance));
        registry.register("manhattan", Box::new(ManhattanDistance));
        registry.register("chebyshev", Box::new(ChebyshevDistance));
        
        // Register new metrics
        registry.register("hamming", Box::new(HammingDistance));
        registry.register("jaccard", Box::new(JaccardDistance));
        registry.register("angular", Box::new(AngularDistance));
        registry.register("canberra", Box::new(CanberraDistance));

        registry
    }

    pub fn register(&mut self, name: &str, func: Box<dyn DistanceFunction>) {
        self.functions.insert(name.to_string(), func);
    }

    pub fn get(&self, name: &str) -> Option<Box<dyn DistanceFunction>> {
        self.functions.get(name).map(|f| f.clone())
    }

    pub fn list_metrics(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

static DISTANCE_REGISTRY: Mutex<Option<DistanceRegistry>> = Mutex::new(None);

pub fn init_distance_registry() {
    if let Ok(mut registry) = DISTANCE_REGISTRY.lock() {
        if registry.is_none() {
            *registry = Some(DistanceRegistry::new());
        }
    }
}

/// Register a custom distance function.
pub fn register_distance_function(name: &str, func: Box<dyn DistanceFunction>) -> Result<(), String> {
    let mut registry_guard = DISTANCE_REGISTRY.lock().map_err(|_| "Lock poisoned".to_string())?;
    match registry_guard.as_mut() {
        Some(registry) => {
            registry.register(name, func);
            Ok(())
        }
        None => Err("Distance registry not initialized".to_string()),
    }
}

/// Old function (returns Option)
pub fn get_distance_function(name: &str) -> Option<Box<dyn DistanceFunction>> {
    let registry_guard = DISTANCE_REGISTRY.lock().ok()?;
    registry_guard.as_ref()?.get(name)
}

/// New safe version (returns Result)
pub fn get_distance_function_safe(name: &str) -> Result<Box<dyn DistanceFunction>, DistanceRegistryError> {
    let registry_guard = DISTANCE_REGISTRY.lock()?;
    let registry = registry_guard.as_ref().ok_or(DistanceRegistryError::RegistryNotInitialized)?;
    registry.get(name).ok_or(DistanceRegistryError::MetricNotFound(name.to_string()))
}

pub fn list_distance_metrics() -> Vec<String> {
    if let Ok(guard) = DISTANCE_REGISTRY.lock() {
        if let Some(reg) = &*guard {
            return reg.list_metrics();
        }
    }
    vec![]
}

#[pyfunction]
pub fn register_metric(name: &str, func: PyObject) -> PyResult<()> {
    let distance_func = PythonDistanceFunction::new(name.to_string(), func);
    register_distance_function(name, Box::new(distance_func))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

#[pyfunction]
pub fn list_metrics() -> Vec<String> {
    list_distance_metrics()
}