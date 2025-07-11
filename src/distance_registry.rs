use std::collections::HashMap;
use std::sync::Mutex;
use pyo3::prelude::*;

/// Trait for distance functions that can be registered and used by the index.
pub trait DistanceFunction: Send + Sync {
    /// Calculate the distance between two vectors.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// Get the name of this distance function.
    fn name(&self) -> &str;
    
    /// Clone the distance function (for trait objects).
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
        assert_eq!(a.len(), b.len(), "Input slices must have the same length");
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
        let norm_a = a.iter().map(|x| x.powi(2)).sum::<f32>();
        let norm_b = b.iter().map(|x| x.powi(2)).sum::<f32>();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0; // Maximum distance
        }
        1.0 - dot_product / (norm_a * norm_b).sqrt()
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
        assert_eq!(a.len(), b.len(), "Input slices must have the same length");
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
        assert_eq!(a.len(), b.len(), "Input slices must have the same length");
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }
    
    fn name(&self) -> &str {
        "chebyshev"
    }
    
    fn clone_boxed(&self) -> Box<dyn DistanceFunction> {
        Box::new(self.clone())
    }
}

/// A distance function that wraps a Python callable.
pub struct PythonDistanceFunction {
    name: String,
    python_func: PyObject,
}

impl Clone for PythonDistanceFunction {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            Self {
                name: self.name.clone(),
                python_func: self.python_func.clone_ref(py),
            }
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
            let a_py = a.into_pyobject(py).expect("Failed to convert vector a to Python");
            let b_py = b.into_pyobject(py).expect("Failed to convert vector b to Python");
            
            match self.python_func.call1(py, (a_py, b_py)) {
                Ok(result) => result.extract::<f32>(py).unwrap_or(f32::NAN),
                Err(_) => f32::NAN,
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

/// Global registry for distance functions.
pub struct DistanceRegistry {
    functions: HashMap<String, Box<dyn DistanceFunction>>,
}

impl DistanceRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };
        
        // Register built-in distance functions
        registry.register("euclidean", Box::new(EuclideanDistance));
        registry.register("cosine", Box::new(CosineDistance));
        registry.register("manhattan", Box::new(ManhattanDistance));
        registry.register("chebyshev", Box::new(ChebyshevDistance));
        
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

/// Global distance registry instance.
static DISTANCE_REGISTRY: Mutex<Option<DistanceRegistry>> = Mutex::new(None);

/// Initialize the global distance registry.
pub fn init_distance_registry() {
    let mut registry = DISTANCE_REGISTRY.lock().unwrap();
    if registry.is_none() {
        *registry = Some(DistanceRegistry::new());
    }
}

/// Register a custom distance function.
pub fn register_distance_function(name: &str, func: Box<dyn DistanceFunction>) -> Result<(), String> {
    let mut registry_guard = DISTANCE_REGISTRY.lock().unwrap();
    match registry_guard.as_mut() {
        Some(registry) => {
            registry.register(name, func);
            Ok(())
        }
        None => Err("Distance registry not initialized".to_string()),
    }
}

/// Get a distance function by name.
pub fn get_distance_function(name: &str) -> Option<Box<dyn DistanceFunction>> {
    let registry_guard = DISTANCE_REGISTRY.lock().unwrap();
    registry_guard.as_ref()?.get(name)
}

/// List all available distance metrics.
pub fn list_distance_metrics() -> Vec<String> {
    let registry_guard = DISTANCE_REGISTRY.lock().unwrap();
    registry_guard.as_ref().map(|r| r.list_metrics()).unwrap_or_default()
}

/// Python function to register a custom distance metric.
#[pyfunction]
pub fn register_metric(name: &str, func: PyObject) -> PyResult<()> {
    let distance_func = PythonDistanceFunction::new(name.to_string(), func);
    register_distance_function(name, Box::new(distance_func))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Python function to list all available distance metrics.
#[pyfunction]
pub fn list_metrics() -> Vec<String> {
    list_distance_metrics()
}
