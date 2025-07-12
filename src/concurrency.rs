//! Concurrency utilities: Python-visible thread-safe wrapper around `AnnIndex`.

use std::sync::{Arc, RwLock};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use crate::index::AnnIndex;
use crate::metrics::Distance;
use crate::errors::RustAnnError;

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
        let mut guard = self.inner.write().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire write lock: {}", e))
        })?;
        guard.add(py, data, ids)
    }

    /// Remove by ID.
    pub fn remove(&self, _py: Python, ids: Vec<i64>) -> PyResult<()> {
        let mut guard = self.inner.write().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire write lock: {}", e))
        })?;
        guard.remove(ids)
    }

    /// Single-vector k-NN search.
    pub fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let guard = self.inner.read().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire read lock: {}", e))
        })?;
        guard.search(py, query, k)
    }

    /// Batch k-NN search.
    pub fn search_batch(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let guard = self.inner.read().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire read lock: {}", e))
        })?;
        guard.search_batch(py, data, k)
    }

    /// Save to disk.
    pub fn save(&self, _py: Python, path: &str) -> PyResult<()> {
        let guard = self.inner.read().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire read lock: {}", e))
        })?;
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
