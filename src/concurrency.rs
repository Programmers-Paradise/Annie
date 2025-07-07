//! Concurrency utilities: Python-visible thread-safe wrapper around `AnnIndex`.
//!
//! This module exposes a `ThreadSafeAnnIndex` that allows concurrent, thread-safe
//! usage of the approximate nearest neighbor index from Python via PyO3.
//!
//! Internally, the index is protected by a `RwLock` so reads (searches) can happen
//! concurrently, while writes (add/remove) are exclusive.

use std::sync::{Arc, RwLock};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use crate::index::AnnIndex;
use crate::metrics::Distance;
use crate::errors::RustAnnError;

/// A thread-safe, Python-visible wrapper around [`AnnIndex`].
///
/// Internally, the index is guarded with an `RwLock`, allowing:
/// - Concurrent reads (e.g., multiple searches in parallel)
/// - Exclusive writes (e.g., add/remove operations)
///
/// This allows safe concurrent access from multiple Python threads or async contexts.
#[pyclass]
pub struct ThreadSafeAnnIndex {
    inner: Arc<RwLock<AnnIndex>>,
}

#[pymethods]
impl ThreadSafeAnnIndex {
    /// Create a new thread-safe ANN index with the given dimension and distance metric.
    #[new]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        let idx = AnnIndex::new(dim, metric)?;
        Ok(ThreadSafeAnnIndex {
            inner: Arc::new(RwLock::new(idx)),
        })
    }

    /// Add new vectors and IDs to the index.
    ///
    /// This acquires a write lock and must be called with the Python GIL held.
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

    /// Remove points from the index by ID.
    ///
    /// This acquires a write lock.
    pub fn remove(&self, _py: Python, ids: Vec<i64>) -> PyResult<()> {
        let mut guard = self.inner.write().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire write lock: {}", e))
        })?;
        guard.remove(ids)
    }

    /// Perform a k-NN search for a single query vector.
    ///
    /// This acquires a shared read lock and can run concurrently with other readers.
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

    /// Perform a batched k-NN search for multiple query vectors.
    ///
    /// This acquires a shared read lock and propagates any parallel search errors.
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

    /// Save the index to disk.
    ///
    /// This acquires a shared read lock.
    pub fn save(&self, _py: Python, path: &str) -> PyResult<()> {
        let guard = self.inner.read().map_err(|e| {
            RustAnnError::py_err("Lock Error", format!("Failed to acquire read lock: {}", e))
        })?;
        guard.save(path)
    }

    /// Load an index from disk and wrap it as a thread-safe object.
    #[staticmethod]
    pub fn load(_py: Python, path: &str) -> PyResult<Self> {
        let idx = AnnIndex::load(path)?;
        Ok(ThreadSafeAnnIndex {
            inner: Arc::new(RwLock::new(idx)),
        })
    }
}
