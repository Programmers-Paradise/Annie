#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_poisoned_lock_distance_registry() {
        // Poison the lock
        let _ = thread::spawn(|| {
            let _guard = DISTANCE_REGISTRY.lock().unwrap();
            panic!("Poisoning lock intentionally");
        })
        .join();

        // Try to access the registry after poisoning
        let result = get_distance_function_safe("euclidean");
        match result {
            Err(err) => {
                let msg = err.to_string();
                println!("DistanceRegistry returned error: {}", msg);
                assert!(msg.to_lowercase().contains("poison"));
            }
            Ok(_) => panic!("Expected error due to poisoned lock"),
        }
    }
}
/// Helper to acquire the DISTANCE_REGISTRY lock and handle poisoning consistently
fn get_registry_lock() -> Result<std::sync::MutexGuard<'static, Option<DistanceRegistry>>, String> {
    DISTANCE_REGISTRY
        .lock()
        .map_err(|_| "Lock poisoned".to_string())
}
use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;

use crate::errors::DistanceRegistryError;

/// Trait for distance functions that can be registered and used by the index.
pub trait DistanceFunction: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    /// Fallible variant of [`DistanceFunction::distance`]. Implementations
    /// that can fail (e.g. user-supplied Python callables) override this so
    /// errors propagate to the caller instead of becoming sentinel values.
    fn try_distance(&self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        Ok(self.distance(a, b))
    }
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
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
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
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
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
                if denom > 0.0 {
                    diff / denom
                } else {
                    0.0
                }
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
        self.try_distance(a, b).unwrap_or(f32::MAX)
    }

    fn try_distance(&self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        Python::with_gil(|py| {
            let result = self
                .python_func
                .call1(
                    py,
                    (
                        numpy::PyArray1::from_slice(py, a),
                        numpy::PyArray1::from_slice(py, b),
                    ),
                )
                .map_err(|e| format!("custom metric '{}' raised: {}", self.name, e))?;
            result
                .extract::<f32>(py)
                .map_err(|e| format!("custom metric '{}' must return a float: {}", self.name, e))
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

impl Default for DistanceRegistry {
    fn default() -> Self {
        Self::new()
    }
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
        self.functions.get(name).cloned()
    }

    pub fn list_metrics(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

static DISTANCE_REGISTRY: Mutex<Option<DistanceRegistry>> = Mutex::new(None);

pub fn init_distance_registry() {
    if let Ok(mut registry) = get_registry_lock() {
        if registry.is_none() {
            *registry = Some(DistanceRegistry::new());
        }
    }
}

/// Register a custom distance function.
pub fn register_distance_function(
    name: &str,
    func: Box<dyn DistanceFunction>,
) -> Result<(), String> {
    let mut registry_guard = get_registry_lock()?;
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
    let registry_guard = get_registry_lock().ok()?;
    registry_guard.as_ref()?.get(name)
}

/// New safe version (returns Result)
pub fn get_distance_function_safe(
    name: &str,
) -> Result<Box<dyn DistanceFunction>, DistanceRegistryError> {
    let registry_guard = DISTANCE_REGISTRY
        .lock()
        .map_err(|_| DistanceRegistryError::LockPoisoned)?;
    let registry = registry_guard
        .as_ref()
        .ok_or(DistanceRegistryError::RegistryNotInitialized)?;
    registry
        .get(name)
        .ok_or(DistanceRegistryError::MetricNotFound(name.to_string()))
}

pub fn list_distance_metrics() -> Vec<String> {
    if let Ok(guard) = get_registry_lock() {
        if let Some(reg) = &*guard {
            return reg.list_metrics();
        }
    }
    vec![]
}

/// Register a Python callable `func(a, b) -> float` as a named distance metric.
///
/// `a` and `b` are passed as 1-D `float32` NumPy arrays. The metric can then
/// be used with `AnnIndex.new_with_metric(dim, name)` or
/// `AnnIndex(dim, Distance.custom(name))`. Built-in metric names cannot be
/// overridden.
#[pyfunction]
pub fn register_metric(py: Python<'_>, name: &str, func: PyObject) -> PyResult<()> {
    if !func.bind(py).is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "func must be callable",
        ));
    }
    if name.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "metric name must not be empty",
        ));
    }
    if !crate::metrics::Distance::new(name).is_custom() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "'{}' is a built-in metric and cannot be overridden",
            name
        )));
    }
    let distance_func = PythonDistanceFunction::new(name.to_string(), func);
    register_distance_function(name, Box::new(distance_func))
        .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

#[pyfunction]
pub fn list_metrics() -> Vec<String> {
    list_distance_metrics()
}
