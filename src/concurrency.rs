//! Concurrency utilities: Python-visible thread-safe wrapper around `AnnIndex`.

use pyo3::PyErr;

/// Helper to acquire a write lock and handle poisoning consistently
fn get_write_lock<'a>(lock: &'a Arc<RwLock<AnnIndex>>) -> Result<std::sync::RwLockWriteGuard<'a, AnnIndex>, PyErr> {
    lock.write().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to acquire write lock: {}", e)))
}

/// Helper to acquire a read lock and handle poisoning consistently
fn get_read_lock<'a>(lock: &'a Arc<RwLock<AnnIndex>>) -> Result<std::sync::RwLockReadGuard<'a, AnnIndex>, PyErr> {
    lock.read().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to acquire read lock: {}", e)))
}

use std::sync::{Arc, RwLock};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use crate::index::AnnIndex;
use crate::metrics::Distance;

/// A thread-safe, Python-visible wrapper around [`AnnIndex`].
#[pyclass]
pub struct ThreadSafeAnnIndex {
    inner: Arc<RwLock<AnnIndex>>,
}

#[pymethods]
impl ThreadSafeAnnIndex {
    /// Create a new thread-safe ANN index.
    #[new]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        let idx = AnnIndex::new(dim, metric)?;
        Ok(ThreadSafeAnnIndex {
            inner: Arc::new(RwLock::new(idx)),
        })
    }

    /// Add vectors with IDs.
    pub fn add(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
    let mut guard = get_write_lock(&self.inner)?;
        guard.add(py, data, ids)
    }

    /// Remove by ID.
    pub fn remove(&self, _py: Python, ids: Vec<i64>) -> PyResult<()> {
    let mut guard = get_write_lock(&self.inner)?;
        guard.remove(ids)
    }

    pub fn update(&self, _py: Python, id: i64, vector: Vec<f32>) -> PyResult<()> {
    let mut guard = get_write_lock(&self.inner)?;
        guard.update(id, vector)
    }

    pub fn compact(&self, _py: Python) -> PyResult<()> {
    let mut guard = get_write_lock(&self.inner)?;
        guard.compact()
    }
    
    pub fn version(&self, _py: Python) -> u64 {
        match get_read_lock(&self.inner) {
            Ok(guard) => guard.version(),
            Err(_) => 0 // Optionally propagate error instead of returning 0
        }
    }

    /// Single-vector k-NN search.
    pub fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
    let guard = get_read_lock(&self.inner)?;
        guard.search(py, query, k, None)
    }

    /// Batch k-NN search.
    pub fn search_batch(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
    let guard = get_read_lock(&self.inner)?;
        guard.search_batch(py, data, k, None)
    }

    /// Save to disk.
    pub fn save(&self, _py: Python, path: &str) -> PyResult<()> {
    let guard = get_read_lock(&self.inner)?;
        guard.save(path)
    }

    /// Load and wrap.
    #[staticmethod]
    pub fn load(_py: Python, path: &str) -> PyResult<Self> {
        let idx = AnnIndex::load(path)?;
        Ok(ThreadSafeAnnIndex {
            inner: Arc::new(RwLock::new(idx)),
        })
    }
}

impl ThreadSafeAnnIndex {
    /// Internal constructor for testing: wraps an existing Arc<RwLock<AnnIndex>>.
    pub fn from_arc(inner: Arc<RwLock<AnnIndex>>) -> Self {
        Self { inner }
    }
}
